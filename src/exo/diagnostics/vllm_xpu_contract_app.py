"""Standalone web console for exercising the Intel vLLM Engine contract."""

from __future__ import annotations

import base64
import binascii
import contextlib
import hashlib
import hmac
import html
import importlib
import json
import os
import queue
import secrets
import threading
import time
from collections import deque
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Iterable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, Protocol, cast, final
from urllib.parse import parse_qs, quote, urlsplit

import httpx
from anyio import create_task_group, to_thread
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from pydantic import Field, field_validator
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response

from exo.shared.models.model_cards import ModelId
from exo.shared.types.chunks import (
    ErrorChunk,
    ImageChunk,
    PrefillProgressChunk,
    TokenChunk,
    ToolCallChunk,
)
from exo.shared.types.common import CommandId
from exo.shared.types.tasks import CANCEL_ALL_TASKS, TaskId, TextGeneration
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.instances import InstanceId
from exo.shared.types.worker.runner_response import (
    CancelledResponse,
    FinishedResponse,
)
from exo.utils.channels import MpReceiver, MpSender, mp_channel
from exo.utils.pydantic_ext import FrozenModel
from exo.worker.engines.base import Engine
from exo.worker.engines.vllm_xpu.sidecar import (
    SidecarResult,
    VllmSidecarEngine,
    VllmSidecarProcess,
    VllmSidecarSettings,
)

MAX_PROMPT_CHARACTERS = 32_768
MAX_OUTPUT_TOKENS = 1_024
SESSION_QUEUE_SIZE = 4_096
MAX_TRACKED_SESSIONS = 256
DEFAULT_HEALTH_TIMEOUT_SECONDS = 1.0
DEFAULT_SIMULATION_STATUS_PATH = Path("/run/exo-thunderbolt-simulation/status.json")
REAL_NODE_ID = "slazenger-panther-lake"
SIMULATED_NODE_ID = "thunderbolt-peer-simulated"
ACCESS_COOKIE = "exo_vllm_demo_session"
ACCESS_SESSION_SECONDS = 12 * 60 * 60
PASSWORD_HASH_ITERATIONS = 600_000
MAX_LOGIN_BODY_BYTES = 4_096

STATIC_DIRECTORY = Path(__file__).with_name("static")
SECURITY_HEADERS: Mapping[str, str] = {
    "Cache-Control": "no-store",
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self'; "
        "connect-src 'self'; img-src 'none'; object-src 'none'; "
        "base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
    ),
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
}
DASHBOARD_SECURITY_HEADERS: Mapping[str, str] = {
    "Cache-Control": "no-store",
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; connect-src 'self'; "
        "img-src 'self' data: blob:; font-src 'self' data:; "
        "object-src 'none'; base-uri 'none'; frame-ancestors 'none'; "
        "form-action 'self'"
    ),
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
}
LOGIN_SECURITY_HEADERS: Mapping[str, str] = {
    "Cache-Control": "no-store",
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'none'; style-src 'self'; "
        "connect-src 'self'; img-src 'self' data:; object-src 'none'; "
        "base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
    ),
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
}

SessionState = Literal["active", "finished", "cancelled", "failed"]
CancelDisposition = Literal["accepted", "unknown", "terminal"]
TorchState = Literal["available", "unavailable", "not-installed", "error"]
type BrokerResult = SidecarResult | BrokerFailure


@dataclass(frozen=True)
class DemoAuthSettings:
    password_hash: str | None = None
    session_secret: bytes = field(default_factory=lambda: secrets.token_bytes(32))
    session_seconds: int = ACCESS_SESSION_SECONDS

    def __post_init__(self) -> None:
        if self.password_hash is not None:
            _parse_password_hash(self.password_hash)
        if len(self.session_secret) < 32:
            raise ValueError("demo session secret must contain at least 32 bytes")
        if not 1 <= self.session_seconds <= ACCESS_SESSION_SECONDS:
            raise ValueError("demo session duration is outside the supported range")

    @property
    def enabled(self) -> bool:
        return self.password_hash is not None


def _base64_without_padding(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _decode_base64_without_padding(value: str) -> bytes:
    padding = "=" * ((4 - len(value) % 4) % 4)
    try:
        return base64.b64decode(
            value + padding,
            altchars=b"-_",
            validate=True,
        )
    except (binascii.Error, ValueError) as error:
        raise ValueError("demo password hash contains invalid base64") from error


def create_demo_password_hash(
    password: str,
    *,
    salt: bytes | None = None,
    iterations: int = PASSWORD_HASH_ITERATIONS,
) -> str:
    if not password:
        raise ValueError("demo password cannot be empty")
    if not 100_000 <= iterations <= 2_000_000:
        raise ValueError(
            "demo password hash iterations are outside the supported range"
        )
    resolved_salt = secrets.token_bytes(16) if salt is None else salt
    if len(resolved_salt) < 16:
        raise ValueError("demo password hash salt must contain at least 16 bytes")
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        resolved_salt,
        iterations,
    )
    return "$".join(
        (
            "pbkdf2_sha256",
            str(iterations),
            _base64_without_padding(resolved_salt),
            _base64_without_padding(digest),
        )
    )


def _parse_password_hash(encoded_hash: str) -> tuple[int, bytes, bytes]:
    try:
        algorithm, iterations_text, salt_text, digest_text = encoded_hash.split("$")
        iterations = int(iterations_text)
    except ValueError as error:
        raise ValueError("demo password hash has an invalid format") from error
    if algorithm != "pbkdf2_sha256":
        raise ValueError("demo password hash uses an unsupported algorithm")
    if not 100_000 <= iterations <= 2_000_000:
        raise ValueError(
            "demo password hash iterations are outside the supported range"
        )
    salt = _decode_base64_without_padding(salt_text)
    digest = _decode_base64_without_padding(digest_text)
    if len(salt) < 16 or len(digest) != hashlib.sha256().digest_size:
        raise ValueError("demo password hash has invalid salt or digest length")
    return iterations, salt, digest


def verify_demo_password(password: str, encoded_hash: str) -> bool:
    iterations, salt, expected = _parse_password_hash(encoded_hash)
    actual = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        iterations,
    )
    return hmac.compare_digest(actual, expected)


def demo_auth_settings_from_environment(
    environment: Mapping[str, str] | None = None,
) -> DemoAuthSettings:
    values = os.environ if environment is None else environment
    password_hash = values.get("EXO_VLLM_DEMO_PASSWORD_HASH", "").strip() or None
    return DemoAuthSettings(password_hash=password_hash)


def _create_access_token(settings: DemoAuthSettings, now: int | None = None) -> str:
    current = int(time.time()) if now is None else now
    expires_at = current + settings.session_seconds
    nonce = secrets.token_urlsafe(12)
    payload = f"v1:{expires_at}:{nonce}"
    signature = hmac.new(
        settings.session_secret,
        payload.encode("ascii"),
        hashlib.sha256,
    ).hexdigest()
    return f"{expires_at}.{nonce}.{signature}"


def _verify_access_token(
    token: str | None,
    settings: DemoAuthSettings,
    now: int | None = None,
) -> bool:
    if not token:
        return False
    try:
        expires_text, nonce, signature = token.split(".", 2)
        expires_at = int(expires_text)
    except (TypeError, ValueError):
        return False
    current = int(time.time()) if now is None else now
    if expires_at < current or expires_at > current + settings.session_seconds:
        return False
    payload = f"v1:{expires_at}:{nonce}"
    expected = hmac.new(
        settings.session_secret,
        payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(signature, expected)


def _safe_next(value: str | None) -> str:
    if not value or not value.startswith("/") or value.startswith("//"):
        return "/"
    if any(character in value for character in ("\r", "\n", "\x00")):
        return "/"
    return value


def _valid_email(value: str) -> bool:
    if not value or len(value) > 120 or any(character.isspace() for character in value):
        return False
    local, separator, domain = value.rpartition("@")
    return bool(separator and local and domain and "." in domain)


def _login_page(
    next_path: str = "/",
    *,
    failed: bool = False,
    identity: str = "",
) -> str:
    error = (
        '<p class="error" role="alert">The email or password was not accepted.</p>'
        if failed
        else ""
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="robots" content="noindex">
  <title>Intel AI PC Demo Login</title>
  <link rel="stylesheet" href="/vllm-login.css?v=1">
</head>
<body>
  <div class="page-shell">
    <header class="brand-bar">
      <a class="brand-lockup" href="/login" aria-label="HP Intel AI PC demo login">
        <img src="/hp-logo.svg?v=1" alt="HP">
        <span>Intel AI PC Demo</span>
      </a>
      <span class="security-label">Protected application</span>
    </header>
    <main class="login-layout">
      <section class="login-card">
        <div class="section-marker" aria-hidden="true"></div>
        <p class="eyebrow">Panther Lake local AI</p>
        <h1>Exo vLLM Demo</h1>
        <p class="intro">Sign in to view the two-node Thunderbolt simulation and run the model on the Intel XPU.</p>
        {error}
        <form method="post" action="/login">
          <input type="hidden" name="next" value="{html.escape(next_path, quote=True)}">
          <label for="identity">Email</label>
          <input id="identity" name="identity" type="email" maxlength="120" autocomplete="username" placeholder="name@example.com" value="{html.escape(identity, quote=True)}" required autofocus>
          <label for="password">Access password</label>
          <input id="password" name="password" type="password" autocomplete="current-password" required>
          <button type="submit">
            <span>Continue</span>
            <span class="arrow" aria-hidden="true">→</span>
          </button>
        </form>
        <p class="access-note">Authorized HP demonstration users only.</p>
      </section>
      <aside class="brand-stage" aria-label="Intel AI PC demonstration">
        <div class="stripe stripe-one"></div>
        <div class="stripe stripe-two"></div>
        <div class="stage-copy">
          <p>HP Consumer AI</p>
          <h2>Local AI.<br>Real Intel XPU.</h2>
          <span>Exo cluster view. vLLM Engine contract. Panther Lake hardware.</span>
        </div>
      </aside>
    </main>
  </div>
</body>
</html>"""


@final
class ChatRequest(FrozenModel):
    prompt: str = Field(min_length=1, max_length=MAX_PROMPT_CHARACTERS)
    max_tokens: int = Field(default=MAX_OUTPUT_TOKENS, ge=1, le=MAX_OUTPUT_TOKENS)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)

    @field_validator("prompt")
    @classmethod
    def reject_blank_prompt(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must contain a non-whitespace character")
        return value


@final
class IntelXpuSysfsDevice(FrozenModel):
    pci_address: str
    device_id: str | None = None
    driver: str | None = None


@final
class XpuRuntimeStatus(FrozenModel):
    torch_state: TorchState
    torch_version: str | None = None
    torch_xpu_available: bool | None = None
    torch_xpu_device_count: int | None = None
    torch_xpu_device_names: tuple[str, ...] = ()
    intel_sysfs_devices: tuple[IntelXpuSysfsDevice, ...] = ()
    detail: str | None = None


@final
class SidecarHealthStatus(FrozenModel):
    ready: bool
    detail: str | None = None


@final
class HardwareGateStatus(FrozenModel):
    ready: bool
    detail: str | None = None


@final
class ContractStatus(FrozenModel):
    status: Literal["ready", "not-ready"]
    contract: Literal["Engine"] = "Engine"
    implementation: Literal["VllmSidecarEngine"] = "VllmSidecarEngine"
    model: ModelId
    sidecar_mode: Literal["external"] = "external"
    sidecar_endpoint: str
    sidecar: SidecarHealthStatus
    xpu: XpuRuntimeStatus
    hardware_gate: HardwareGateStatus
    simulation: ThunderboltSimulationStatus | None = None
    active_requests: int
    performance_claim: Literal["none"] = "none"


@final
class SimulationNodeStatus(FrozenModel):
    namespace: str
    address: str
    interface_name: str
    present: bool
    link_up: bool


@final
class ThunderboltSimulationStatus(FrozenModel):
    version: Literal[1] = 1
    mode: Literal["network-namespace"] = "network-namespace"
    observed_at: datetime | None = None
    active: bool = False
    link_up: bool = False
    nodes: tuple[SimulationNodeStatus, ...] = ()
    detail: str | None = None

    @field_validator("nodes")
    @classmethod
    def require_zero_or_two_nodes(
        cls, value: tuple[SimulationNodeStatus, ...]
    ) -> tuple[SimulationNodeStatus, ...]:
        if len(value) not in {0, 2}:
            raise ValueError("simulation status must contain zero or two nodes")
        return value


@final
class BrokerFailure(FrozenModel):
    message: str


class _XpuApi(Protocol):
    def is_available(self) -> bool: ...

    def device_count(self) -> int: ...

    def get_device_name(self, device: int) -> str: ...


class _TorchModule(Protocol):
    __version__: str
    xpu: _XpuApi


type SimulationStatusProvider = Callable[[], ThunderboltSimulationStatus]


def read_thunderbolt_simulation_status(
    status_path: Path = DEFAULT_SIMULATION_STATUS_PATH,
) -> ThunderboltSimulationStatus:
    """Read the root-published, sanitized network namespace status."""

    try:
        return ThunderboltSimulationStatus.model_validate_json(
            status_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as error:
        return ThunderboltSimulationStatus(detail=type(error).__name__)


def dashboard_directory_from_environment(
    environment: Mapping[str, str] | None = None,
) -> Path | None:
    values = os.environ if environment is None else environment
    configured = values.get("EXO_VLLM_DEMO_DASHBOARD_DIR", "").strip()
    if not configured:
        return None
    dashboard_directory = Path(configured)
    if not (dashboard_directory / "index.html").is_file():
        raise ValueError(
            "EXO_VLLM_DEMO_DASHBOARD_DIR must contain a built dashboard index.html"
        )
    return dashboard_directory


def build_demo_cluster_state(
    simulation: ThunderboltSimulationStatus,
) -> dict[str, object]:
    """Project the live namespace simulation into Exo's dashboard state schema."""

    real_address = "192.168.253.1"
    simulated_address = "192.168.253.2"
    nodes = [REAL_NODE_ID]
    identities: dict[str, object] = {
        REAL_NODE_ID: {
            "modelId": "Intel Panther Lake 12 Xe iGPU",
            "chipId": "Intel Arc B390",
            "friendlyName": "Slazenger (real XPU)",
            "osVersion": "Linux",
            "osBuildVersion": "simulated-cluster-demo",
        }
    }
    node_memory: dict[str, object] = {
        REAL_NODE_ID: {
            "ramTotal": {"inBytes": 62_432_477_184},
            "ramAvailable": {"inBytes": 20_000_000_000},
            "swapTotal": {"inBytes": 0},
            "swapAvailable": {"inBytes": 0},
        }
    }
    node_system: dict[str, object] = {
        REAL_NODE_ID: {"gpuUsage": 0.0, "temp": 0.0, "sysPower": 0.0}
    }
    node_network: dict[str, object] = {
        REAL_NODE_ID: {
            "interfaces": [
                {
                    "name": "thunderbolt0",
                    "ipAddress": real_address,
                    "interfaceType": "thunderbolt",
                }
            ]
        }
    }
    connections: dict[str, object] = {}
    if simulation.active:
        nodes.append(SIMULATED_NODE_ID)
        identities[SIMULATED_NODE_ID] = {
            "modelId": "Virtual Linux node",
            "chipId": "No accelerator (simulated)",
            "friendlyName": "Thunderbolt Peer (simulated)",
            "osVersion": "Linux",
            "osBuildVersion": "network-namespace",
        }
        node_memory[SIMULATED_NODE_ID] = {
            "ramTotal": {"inBytes": 0},
            "ramAvailable": {"inBytes": 0},
            "swapTotal": {"inBytes": 0},
            "swapAvailable": {"inBytes": 0},
        }
        node_system[SIMULATED_NODE_ID] = {
            "gpuUsage": 0.0,
            "temp": 0.0,
            "sysPower": 0.0,
        }
        node_network[SIMULATED_NODE_ID] = {
            "interfaces": [
                {
                    "name": "thunderbolt0",
                    "ipAddress": simulated_address,
                    "interfaceType": "thunderbolt",
                }
            ]
        }
        if simulation.link_up:
            connections = {
                REAL_NODE_ID: {
                    SIMULATED_NODE_ID: [
                        {
                            "sinkMultiaddr": {
                                "address": f"/ip4/{simulated_address}/tcp/52415",
                                "address_type": "ip4",
                                "ip_address": simulated_address,
                                "port": 52415,
                            }
                        }
                    ]
                },
                SIMULATED_NODE_ID: {
                    REAL_NODE_ID: [
                        {
                            "sinkMultiaddr": {
                                "address": f"/ip4/{real_address}/tcp/52415",
                                "address_type": "ip4",
                                "ip_address": real_address,
                                "port": 52415,
                            }
                        }
                    ]
                },
            }

    return {
        "instances": {},
        "runners": {},
        "downloads": {},
        "tasks": {},
        "lastSeen": {},
        "topology": {"nodes": nodes, "connections": connections},
        "lastEventAppliedIdx": -1,
        "nodeIdentities": identities,
        "nodeMemory": node_memory,
        "nodeDisk": {},
        "nodeSystem": node_system,
        "nodeNetwork": node_network,
        "nodeThunderbolt": {},
        "nodeThunderboltBridge": {},
        "nodeRdmaCtl": {},
        "nodeBackends": {REAL_NODE_ID: ["Vllm"], SIMULATED_NODE_ID: []},
        "thunderboltBridgeCycles": [],
        "instanceLinks": {},
        "prefillServerPorts": {},
        "customModelCards": {},
    }


def _read_optional_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def _driver_name(device_path: Path) -> str | None:
    try:
        return (device_path / "driver").resolve(strict=True).name
    except OSError:
        return None


def gather_intel_xpu_sysfs_devices(
    sysfs_root: Path = Path("/sys"),
) -> tuple[IntelXpuSysfsDevice, ...]:
    """Collect Intel DRM devices without assuming PyTorch is installed."""
    devices: list[IntelXpuSysfsDevice] = []
    drm_directory = sysfs_root / "class" / "drm"
    try:
        card_paths = sorted(drm_directory.glob("card[0-9]*"))
    except OSError:
        return ()

    for card_path in card_paths:
        device_path = card_path / "device"
        vendor = _read_optional_text(device_path / "vendor")
        if vendor is None or vendor.lower() not in {"0x8086", "8086"}:
            continue
        try:
            pci_address = device_path.resolve(strict=True).name
        except OSError:
            pci_address = card_path.name
        devices.append(
            IntelXpuSysfsDevice(
                pci_address=pci_address,
                device_id=_read_optional_text(device_path / "device"),
                driver=_driver_name(device_path),
            )
        )
    return tuple(devices)


def gather_xpu_runtime_status(
    *,
    sysfs_root: Path = Path("/sys"),
    module_loader: Callable[[str], object] = importlib.import_module,
) -> XpuRuntimeStatus:
    """Inspect PyTorch XPU and kernel DRM state through injectable boundaries."""
    sysfs_devices = gather_intel_xpu_sysfs_devices(sysfs_root)
    try:
        torch = cast(_TorchModule, module_loader("torch"))
    except ImportError:
        return XpuRuntimeStatus(
            torch_state="not-installed",
            intel_sysfs_devices=sysfs_devices,
            detail="PyTorch is not installed in the console environment",
        )

    try:
        available = torch.xpu.is_available()
        count = torch.xpu.device_count()
        names = tuple(torch.xpu.get_device_name(index) for index in range(count))
    except (AttributeError, RuntimeError) as error:
        return XpuRuntimeStatus(
            torch_state="error",
            torch_version=torch.__version__,
            intel_sysfs_devices=sysfs_devices,
            detail=type(error).__name__,
        )

    return XpuRuntimeStatus(
        torch_state="available" if available else "unavailable",
        torch_version=torch.__version__,
        torch_xpu_available=available,
        torch_xpu_device_count=count,
        torch_xpu_device_names=names,
        intel_sysfs_devices=sysfs_devices,
    )


def evaluate_real_xpu_hardware_gate(status: XpuRuntimeStatus) -> HardwareGateStatus:
    if status.torch_state != "available" or status.torch_xpu_available is not True:
        return HardwareGateStatus(
            ready=False,
            detail="local PyTorch XPU runtime is not available",
        )
    if status.torch_xpu_device_count is None or status.torch_xpu_device_count < 1:
        return HardwareGateStatus(
            ready=False,
            detail="local PyTorch reports no XPU devices",
        )
    if not any(
        device.driver is not None and device.driver.lower() == "xe"
        for device in status.intel_sysfs_devices
    ):
        return HardwareGateStatus(
            ready=False,
            detail="no Intel DRM device bound to the xe driver was found in sysfs",
        )
    return HardwareGateStatus(ready=True)


def redacted_sidecar_endpoint(settings: VllmSidecarSettings) -> str:
    parsed = urlsplit(settings.base_url)
    host = parsed.hostname or "unknown"
    if ":" in host:
        host = f"[{host}]"
    port = parsed.port
    default_port = 443 if parsed.scheme == "https" else 80
    port_suffix = "" if port in {None, default_port} else f":{port}"
    return f"{parsed.scheme}://{host}{port_suffix}"


def probe_sidecar_health(
    settings: VllmSidecarSettings,
    timeout_seconds: float = DEFAULT_HEALTH_TIMEOUT_SECONDS,
) -> SidecarHealthStatus:
    headers: dict[str, str] = {}
    if settings.api_key is not None:
        headers["Authorization"] = f"Bearer {settings.api_key}"
    try:
        response = httpx.get(
            settings.health_url,
            headers=headers,
            timeout=timeout_seconds,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as error:
        return SidecarHealthStatus(
            ready=False,
            detail=f"HTTP {error.response.status_code}",
        )
    except httpx.HTTPError as error:
        return SidecarHealthStatus(ready=False, detail=type(error).__name__)
    return SidecarHealthStatus(ready=True)


def external_settings_from_environment(
    environment: Mapping[str, str] | None = None,
) -> VllmSidecarSettings:
    values = dict(os.environ if environment is None else environment)
    model_value = values.get("EXO_VLLM_MODEL_ID", "").strip()
    sidecar_url = values.get("EXO_VLLM_SIDECAR_URL", "").strip()
    if not model_value:
        raise ValueError("EXO_VLLM_MODEL_ID is required")
    if not sidecar_url:
        raise ValueError("EXO_VLLM_SIDECAR_URL is required")
    settings = VllmSidecarSettings.from_environment(
        ModelId(model_value),
        Path("."),
        values,
    )
    if settings.mode != "external":
        raise ValueError("the contract console supports external sidecars only")
    return settings


@dataclass(eq=False)
class ContractRuntime:
    settings: VllmSidecarSettings
    engine: Engine
    cancel_sender: MpSender[TaskId]
    cancel_receiver: MpReceiver[TaskId]

    def close_channels(self) -> None:
        self.cancel_sender.close()
        self.cancel_receiver.close()


def build_external_runtime(
    environment: Mapping[str, str] | None = None,
) -> ContractRuntime:
    settings = external_settings_from_environment(environment)
    cancel_sender, cancel_receiver = mp_channel[TaskId]()
    sidecar = VllmSidecarProcess(settings)
    engine: Engine = VllmSidecarEngine(
        model_id=settings.model_id,
        settings=settings,
        sidecar=sidecar,
        cancel_receiver=cancel_receiver,
    )
    return ContractRuntime(
        settings=settings,
        engine=engine,
        cancel_sender=cancel_sender,
        cancel_receiver=cancel_receiver,
    )


@dataclass(eq=False)
class BrokerSession:
    task_id: TaskId
    results: queue.Queue[BrokerResult] = field(
        default_factory=lambda: queue.Queue(maxsize=SESSION_QUEUE_SIZE)
    )
    state: SessionState = "active"
    saw_error: bool = False

    def receive(self, timeout_seconds: float = 0.25) -> BrokerResult | None:
        try:
            return self.results.get(timeout=timeout_seconds)
        except queue.Empty:
            return None


@dataclass(eq=False)
class EngineBroker:
    engine: Engine
    cancel_sender: MpSender[TaskId]
    _sessions: dict[TaskId, BrokerSession] = field(default_factory=dict, init=False)
    _terminal_order: deque[TaskId] = field(default_factory=deque, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _engine_lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="vllm-contract-engine-broker",
            daemon=True,
        )
        self._thread.start()

    def submit(self, request: ChatRequest, model_id: ModelId) -> BrokerSession:
        task_id = TaskId()
        session = BrokerSession(task_id=task_id)
        task = TextGeneration(
            task_id=task_id,
            instance_id=InstanceId("vllm-xpu-contract-console"),
            command_id=CommandId(str(task_id)),
            task_params=TextGenerationTaskParams(
                model=model_id,
                input=[
                    InputMessage(
                        role="user",
                        content=InputMessageContent(request.prompt),
                    )
                ],
                max_output_tokens=request.max_tokens,
                temperature=request.temperature,
                stream=True,
            ),
        )
        with self._lock:
            self._prune_terminal_sessions()
            self._sessions[task_id] = session
        try:
            with self._engine_lock:
                self.engine.submit(task)
        except Exception:
            with self._lock:
                self._sessions.pop(task_id, None)
            raise
        return session

    def cancel(self, task_id: TaskId) -> CancelDisposition:
        with self._lock:
            session = self._sessions.get(task_id)
            if session is None:
                return "unknown"
            if session.state != "active":
                return "terminal"
        self.cancel_sender.send(task_id)
        return "accepted"

    def is_active(self, task_id: TaskId) -> bool:
        with self._lock:
            session = self._sessions.get(task_id)
            return session is not None and session.state == "active"

    @property
    def active_request_count(self) -> int:
        with self._lock:
            return sum(session.state == "active" for session in self._sessions.values())

    def _prune_terminal_sessions(self) -> None:
        while len(self._terminal_order) >= MAX_TRACKED_SESSIONS:
            task_id = self._terminal_order.popleft()
            session = self._sessions.get(task_id)
            if session is not None and session.state != "active":
                self._sessions.pop(task_id, None)

    def _mark_terminal(self, session: BrokerSession, state: SessionState) -> None:
        if session.state == "active":
            session.state = state
            self._terminal_order.append(session.task_id)

    def _dispatch(self, task_id: TaskId, result: BrokerResult) -> None:
        with self._lock:
            session = self._sessions.get(task_id)
            if session is None or session.state != "active":
                return
            if isinstance(result, ErrorChunk):
                session.saw_error = True
            if isinstance(result, CancelledResponse):
                self._mark_terminal(session, "cancelled")
            elif isinstance(result, FinishedResponse):
                self._mark_terminal(
                    session, "failed" if session.saw_error else "finished"
                )
            elif isinstance(result, BrokerFailure):
                self._mark_terminal(session, "failed")
            try:
                session.results.put_nowait(result)
            except queue.Full:
                self._mark_terminal(session, "failed")
                with contextlib.suppress(Exception):
                    self.cancel_sender.send(task_id)

    def _fail_active_sessions(self, error: Exception) -> None:
        message = f"Engine broker failed: {type(error).__name__}"
        with self._lock:
            task_ids = [
                task_id
                for task_id, session in self._sessions.items()
                if session.state == "active"
            ]
        for task_id in task_ids:
            self._dispatch(task_id, BrokerFailure(message=message))

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                with self._engine_lock:
                    results = tuple(self.engine.step())
            except Exception as error:
                self._fail_active_sessions(error)
                self._stop.set()
                return
            for task_id, result in results:
                self._dispatch(task_id, result)

    def close(self) -> None:
        if self._stop.is_set() and self._thread is None:
            return
        with contextlib.suppress(Exception):
            self.cancel_sender.send(CANCEL_ALL_TASKS)
        self._stop.set()
        thread = self._thread
        self._thread = None
        if thread is not None:
            thread.join(timeout=2.0)
        with self._engine_lock:
            self.engine.close()


def _sse(event: str, payload: Mapping[str, object]) -> str:
    return f"event: {event}\ndata: {json.dumps(dict(payload), ensure_ascii=False)}\n\n"


def _result_events(result: BrokerResult) -> Iterable[str]:
    match result:
        case TokenChunk():
            if result.text:
                yield _sse(
                    "reasoning_delta" if result.is_thinking else "delta",
                    {"text": result.text},
                )
            if result.usage is not None:
                yield _sse(
                    "usage",
                    {
                        "prompt_tokens": result.usage.prompt_tokens,
                        "completion_tokens": result.usage.completion_tokens,
                        "total_tokens": result.usage.total_tokens,
                    },
                )
            if result.finish_reason is not None:
                yield _sse("finish_reason", {"reason": result.finish_reason})
        case ErrorChunk():
            yield _sse("error", {"message": result.error_message[:1000]})
        case ToolCallChunk():
            yield _sse(
                "tool_call",
                {
                    "count": len(result.tool_calls),
                    "message": "Tool calls are not executed by this console",
                },
            )
        case PrefillProgressChunk():
            yield _sse(
                "prefill_progress",
                {
                    "processed_tokens": result.processed_tokens,
                    "total_tokens": result.total_tokens,
                },
            )
        case ImageChunk():
            yield _sse(
                "error",
                {"message": "Image chunks are not supported by this console"},
            )
        case CancelledResponse():
            yield _sse("cancelled", {"status": "cancelled"})
        case FinishedResponse():
            yield _sse("finished", {"status": "finished"})
        case BrokerFailure():
            yield _sse("error", {"message": result.message})


HealthProbe = Callable[[VllmSidecarSettings, float], SidecarHealthStatus]
XpuStatusProvider = Callable[[], XpuRuntimeStatus]
RuntimeFactory = Callable[[], ContractRuntime]


def create_app(
    *,
    runtime_factory: RuntimeFactory = build_external_runtime,
    health_probe: HealthProbe = probe_sidecar_health,
    xpu_status_provider: XpuStatusProvider = gather_xpu_runtime_status,
    simulation_status_provider: SimulationStatusProvider | None = None,
    dashboard_directory: Path | None = None,
    auth_settings: DemoAuthSettings | None = None,
) -> FastAPI:
    context: tuple[ContractRuntime, EngineBroker] | None = None
    resolved_auth_settings = auth_settings or demo_auth_settings_from_environment()
    resolved_dashboard_directory = (
        dashboard_directory
        if dashboard_directory is not None
        else dashboard_directory_from_environment()
    )
    if simulation_status_provider is None:
        configured_status_path = Path(
            os.environ.get(
                "EXO_THUNDERBOLT_SIMULATION_STATUS_PATH",
                str(DEFAULT_SIMULATION_STATUS_PATH),
            )
        )

        def configured_simulation_status_provider() -> ThunderboltSimulationStatus:
            return read_thunderbolt_simulation_status(configured_status_path)

        simulation_status_provider = configured_simulation_status_provider

    def current_context() -> tuple[ContractRuntime, EngineBroker]:
        if context is None:
            raise HTTPException(status_code=503, detail="console is starting")
        return context

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        nonlocal context
        runtime = runtime_factory()
        broker = EngineBroker(runtime.engine, runtime.cancel_sender)
        broker.start()
        context = (runtime, broker)
        try:
            yield
        finally:
            context = None
            broker.close()
            runtime.close_channels()

    app = FastAPI(
        title="Intel vLLM Engine Contract Console",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )

    async def protect_application_and_add_security_headers(
        request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        public_path = request.url.path in {
            "/login",
            "/vllm-login.css",
            "/hp-logo.svg",
            "/healthz",
            "/readyz",
        }
        authenticated = _verify_access_token(
            request.cookies.get(ACCESS_COOKIE),
            resolved_auth_settings,
        )
        if resolved_auth_settings.enabled and not public_path and not authenticated:
            api_path = request.url.path.startswith(("/api/", "/v1/")) or (
                request.url.path in {"/state", "/node_id", "/onboarding", "/models"}
            )
            if api_path:
                response = JSONResponse(
                    {
                        "status": 401,
                        "code": "access_password_required",
                        "detail": "Sign in to access the Exo Intel AI PC demo.",
                    },
                    status_code=401,
                )
            else:
                destination = request.url.path
                if request.url.query:
                    destination = f"{destination}?{request.url.query}"
                response = RedirectResponse(
                    f"/login?next={quote(_safe_next(destination), safe='/')}",
                    status_code=303,
                )
        else:
            response = await call_next(request)
        contract_path = (
            request.url.path == "/contract"
            or (request.url.path == "/" and resolved_dashboard_directory is None)
            or request.url.path.startswith(("/api/", "/vllm-xpu"))
        )
        if request.url.path in {"/login", "/vllm-login.css", "/hp-logo.svg"}:
            headers = LOGIN_SECURITY_HEADERS
        else:
            headers = SECURITY_HEADERS if contract_path else DASHBOARD_SECURITY_HEADERS
        for name, value in headers.items():
            response.headers[name] = value
        return response

    async def contract_index() -> FileResponse:
        return FileResponse(STATIC_DIRECTORY / "vllm-xpu.html")

    async def javascript() -> FileResponse:
        return FileResponse(
            STATIC_DIRECTORY / "vllm-xpu.js", media_type="text/javascript"
        )

    async def stylesheet() -> FileResponse:
        return FileResponse(STATIC_DIRECTORY / "vllm-xpu.css", media_type="text/css")

    async def login_stylesheet() -> FileResponse:
        return FileResponse(STATIC_DIRECTORY / "vllm-login.css", media_type="text/css")

    async def hp_logo() -> FileResponse:
        return FileResponse(
            STATIC_DIRECTORY / "hp-logo.svg", media_type="image/svg+xml"
        )

    async def login(request: Request) -> Response:
        next_path = _safe_next(request.query_params.get("next"))
        if not resolved_auth_settings.enabled or _verify_access_token(
            request.cookies.get(ACCESS_COOKIE),
            resolved_auth_settings,
        ):
            return RedirectResponse(next_path, status_code=303)
        return HTMLResponse(_login_page(next_path))

    async def login_submit(request: Request) -> Response:
        body = await request.body()
        if len(body) > MAX_LOGIN_BODY_BYTES:
            return HTMLResponse(_login_page(failed=True), status_code=400)
        form = parse_qs(body.decode("utf-8", errors="replace"), keep_blank_values=True)
        next_path = _safe_next(form.get("next", ["/"])[0])
        identity = form.get("identity", [""])[0].strip()
        submitted = form.get("password", [""])[0]
        password_hash = resolved_auth_settings.password_hash
        accepted = (
            _valid_email(identity)
            and password_hash is not None
            and verify_demo_password(submitted, password_hash)
        )
        if not accepted:
            return HTMLResponse(
                _login_page(next_path, failed=True, identity=identity[:120]),
                status_code=401,
            )
        response = RedirectResponse(next_path, status_code=303)
        forwarded_scheme = request.headers.get("X-Forwarded-Proto", "")
        response.set_cookie(
            ACCESS_COOKIE,
            _create_access_token(resolved_auth_settings),
            max_age=resolved_auth_settings.session_seconds,
            httponly=True,
            secure=request.url.scheme == "https" or forwarded_scheme == "https",
            samesite="strict",
            path="/",
        )
        return response

    async def logout() -> Response:
        response = RedirectResponse("/login", status_code=303)
        response.delete_cookie(ACCESS_COOKIE, path="/")
        return response

    async def healthz() -> dict[str, str]:
        return {"status": "alive"}

    async def readyz() -> JSONResponse:
        runtime, _ = current_context()
        health, xpu = await _gather_status_parts(
            runtime.settings,
            health_probe,
            xpu_status_provider,
        )
        hardware_gate = evaluate_real_xpu_hardware_gate(xpu)
        ready = health.ready and hardware_gate.ready
        return JSONResponse(
            {
                "status": "ready" if ready else "not-ready",
                "sidecar_ready": health.ready,
                "hardware_gate": hardware_gate.model_dump(mode="json"),
            },
            status_code=200 if ready else 503,
        )

    async def status() -> ContractStatus:
        runtime, broker = current_context()
        health, xpu = await _gather_status_parts(
            runtime.settings,
            health_probe,
            xpu_status_provider,
        )
        hardware_gate = evaluate_real_xpu_hardware_gate(xpu)
        return ContractStatus(
            status="ready" if health.ready and hardware_gate.ready else "not-ready",
            model=runtime.settings.model_id,
            sidecar_endpoint=redacted_sidecar_endpoint(runtime.settings),
            sidecar=health,
            xpu=xpu,
            hardware_gate=hardware_gate,
            simulation=simulation_status_provider(),
            active_requests=broker.active_request_count,
        )

    async def simulation_status() -> ThunderboltSimulationStatus:
        return simulation_status_provider()

    async def dashboard_state() -> JSONResponse:
        return JSONResponse(build_demo_cluster_state(simulation_status_provider()))

    async def node_id() -> str:
        return REAL_NODE_ID

    async def onboarding() -> dict[str, bool]:
        return {"completed": True}

    async def models() -> dict[str, object]:
        runtime, _ = current_context()
        model_id = str(runtime.settings.model_id)
        return {
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "name": "DeepSeek R1 Distill Qwen 1.5B on Intel XPU",
                    "hugging_face_id": model_id,
                    "tasks": ["TextGeneration"],
                    "capabilities": ["text"],
                    "family": "deepseek",
                }
            ],
        }

    async def feature_flags() -> dict[str, bool]:
        return {
            "vllmContractDemo": True,
            "thunderboltSimulation": True,
        }

    async def chat(payload: ChatRequest) -> StreamingResponse:
        runtime, broker = current_context()
        health, xpu = await _gather_status_parts(
            runtime.settings,
            health_probe,
            xpu_status_provider,
        )
        hardware_gate = evaluate_real_xpu_hardware_gate(xpu)
        if not health.ready or not hardware_gate.ready:
            raise HTTPException(
                status_code=503,
                detail="real Intel XPU and vLLM sidecar readiness checks must pass",
            )
        try:
            session = broker.submit(payload, runtime.settings.model_id)
        except Exception as error:
            raise HTTPException(
                status_code=503,
                detail=f"Engine rejected request: {type(error).__name__}",
            ) from error

        async def event_stream() -> AsyncGenerator[str]:
            reached_terminal = False
            try:
                yield _sse(
                    "accepted",
                    {
                        "request_id": str(session.task_id),
                        "contract": "Engine",
                        "implementation": "VllmSidecarEngine",
                        "model": str(runtime.settings.model_id),
                    },
                )
                while not reached_terminal:
                    result = await to_thread.run_sync(session.receive)
                    if result is None:
                        continue
                    for event in _result_events(result):
                        yield event
                    reached_terminal = isinstance(
                        result,
                        (CancelledResponse, FinishedResponse, BrokerFailure),
                    )
            finally:
                if not reached_terminal and broker.is_active(session.task_id):
                    with contextlib.suppress(Exception):
                        broker.cancel(session.task_id)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "X-Exo-Request-Id": str(session.task_id),
                "X-Accel-Buffering": "no",
            },
        )

    async def cancel(task_id: str) -> JSONResponse:
        _, broker = current_context()
        disposition = broker.cancel(TaskId(task_id))
        match disposition:
            case "accepted":
                return JSONResponse(
                    {"status": "cancellation-requested", "request_id": task_id},
                    status_code=202,
                )
            case "terminal":
                return JSONResponse(
                    {"status": "already-terminal", "request_id": task_id},
                    status_code=409,
                )
            case "unknown":
                return JSONResponse(
                    {"status": "unknown-request", "request_id": task_id},
                    status_code=404,
                )

    app.middleware("http")(protect_application_and_add_security_headers)
    if resolved_dashboard_directory is None:
        app.add_api_route("/", contract_index, methods=["GET"], include_in_schema=False)
    app.add_api_route(
        "/contract", contract_index, methods=["GET"], include_in_schema=False
    )
    app.add_api_route(
        "/vllm-xpu.js", javascript, methods=["GET"], include_in_schema=False
    )
    app.add_api_route(
        "/vllm-xpu.css", stylesheet, methods=["GET"], include_in_schema=False
    )
    app.add_api_route("/login", login, methods=["GET"], include_in_schema=False)
    app.add_api_route("/login", login_submit, methods=["POST"], include_in_schema=False)
    app.add_api_route("/logout", logout, methods=["POST"], include_in_schema=False)
    app.add_api_route(
        "/vllm-login.css",
        login_stylesheet,
        methods=["GET"],
        include_in_schema=False,
    )
    app.add_api_route("/hp-logo.svg", hp_logo, methods=["GET"], include_in_schema=False)
    app.add_api_route("/healthz", healthz, methods=["GET"])
    app.add_api_route("/readyz", readyz, methods=["GET"], response_model=None)
    app.add_api_route(
        "/api/status",
        status,
        methods=["GET"],
        response_model_by_alias=False,
    )
    app.add_api_route(
        "/api/simulation/status",
        simulation_status,
        methods=["GET"],
        response_model_by_alias=False,
    )
    app.add_api_route("/state", dashboard_state, methods=["GET"])
    app.add_api_route("/node_id", node_id, methods=["GET"])
    app.add_api_route("/onboarding", onboarding, methods=["GET", "POST"])
    app.add_api_route("/models", models, methods=["GET"])
    app.add_api_route("/v1/feature-flags", feature_flags, methods=["GET"])
    app.add_api_route("/api/chat", chat, methods=["POST"], response_model=None)
    app.add_api_route(
        "/api/requests/{task_id}/cancel",
        cancel,
        methods=["POST"],
        response_model=None,
    )
    if resolved_dashboard_directory is not None:
        app.mount(
            "/",
            StaticFiles(directory=resolved_dashboard_directory, html=True),
            name="dashboard",
        )
    return app


async def _gather_status_parts(
    settings: VllmSidecarSettings,
    health_probe: HealthProbe,
    xpu_status_provider: XpuStatusProvider,
) -> tuple[SidecarHealthStatus, XpuRuntimeStatus]:
    health: SidecarHealthStatus | None = None
    xpu: XpuRuntimeStatus | None = None

    async def gather_health() -> None:
        nonlocal health
        health = await to_thread.run_sync(
            health_probe,
            settings,
            DEFAULT_HEALTH_TIMEOUT_SECONDS,
        )

    async def gather_xpu() -> None:
        nonlocal xpu
        xpu = await to_thread.run_sync(xpu_status_provider)

    async with create_task_group() as task_group:
        task_group.start_soon(gather_health)
        task_group.start_soon(gather_xpu)

    assert health is not None
    assert xpu is not None
    return health, xpu
