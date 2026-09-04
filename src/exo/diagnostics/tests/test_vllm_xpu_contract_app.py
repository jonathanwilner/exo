from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Protocol, cast, override

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from exo.diagnostics.vllm_xpu_contract_app import (
    MAX_PROMPT_CHARACTERS,
    BrokerSession,
    ChatRequest,
    ContractRuntime,
    EngineBroker,
    IntelXpuSysfsDevice,
    SidecarHealthStatus,
    SimulationNodeStatus,
    ThunderboltSimulationStatus,
    XpuRuntimeStatus,
    create_app,
    dashboard_directory_from_environment,
    evaluate_real_xpu_hardware_gate,
    external_settings_from_environment,
    gather_intel_xpu_sysfs_devices,
    gather_xpu_runtime_status,
    probe_sidecar_health,
    redacted_sidecar_endpoint,
)
from exo.shared.models.model_cards import ModelId
from exo.shared.types.chunks import TokenChunk
from exo.shared.types.tasks import GenerationTask, TaskId, TextGeneration
from exo.shared.types.worker.runner_response import (
    CancelledResponse,
    FinishedResponse,
)
from exo.utils.channels import MpReceiver, mp_channel
from exo.worker.disaggregated.server import PrefillRequest
from exo.worker.engines.base import Engine
from exo.worker.engines.vllm_xpu.sidecar import (
    SidecarRequestControl,
    VllmSidecarEngine,
    VllmSidecarProcess,
    VllmSidecarSettings,
)

MODEL_ID = ModelId("Qwen/Qwen3-1.7B")


class _RunnerArguments(Protocol):
    host: str
    port: int
    allow_remote: bool


class _RunnerModule(Protocol):
    run: Callable[[_RunnerArguments], None]

    def build_argument_parser(self) -> argparse.ArgumentParser: ...

    def parse_arguments(
        self,
        parser: argparse.ArgumentParser,
        argument_values: Sequence[str] | None = None,
    ) -> _RunnerArguments: ...

    def validate_arguments(
        self,
        arguments: _RunnerArguments,
        parser: argparse.ArgumentParser,
    ) -> None: ...

    def main(self, argument_values: Sequence[str] | None = None) -> int: ...


def load_runner_module() -> _RunnerModule:
    script_path = Path(__file__).parents[4] / "scripts" / "run_vllm_xpu_contract_gui.py"
    spec = importlib.util.spec_from_file_location(
        "_vllm_xpu_contract_runner", script_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load contract console runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return cast(_RunnerModule, cast(object, module))


RUNNER_MODULE = load_runner_module()


def external_settings(
    base_url: str = "http://127.0.0.1:8123",
) -> VllmSidecarSettings:
    return VllmSidecarSettings(
        mode="external",
        model_id=MODEL_ID,
        model_path="/secret/model/path",
        base_url=base_url,
        serve_host="127.0.0.1",
        serve_port=8123,
        startup_timeout_seconds=5.0,
        request_timeout_seconds=5.0,
        api_key="secret-api-key",
    )


def ready_xpu_status() -> XpuRuntimeStatus:
    return XpuRuntimeStatus(
        torch_state="available",
        torch_version="2.11.0+xpu",
        torch_xpu_available=True,
        torch_xpu_device_count=1,
        torch_xpu_device_names=("Intel(R) Arc(TM) Graphics",),
        intel_sysfs_devices=(
            IntelXpuSysfsDevice(
                pci_address="0000:00:02.0",
                device_id="0xb080",
                driver="xe",
            ),
        ),
    )


def healthy_sidecar(
    settings: VllmSidecarSettings, timeout_seconds: float
) -> SidecarHealthStatus:
    del settings, timeout_seconds
    return SidecarHealthStatus(ready=True)


def active_simulation_status() -> ThunderboltSimulationStatus:
    return ThunderboltSimulationStatus(
        observed_at=datetime(2026, 9, 4, 22, tzinfo=timezone.utc),
        active=True,
        link_up=True,
        nodes=(
            SimulationNodeStatus(
                namespace="exo-tb-node-a",
                address="192.168.253.1/30",
                interface_name="thunderbolt0",
                present=True,
                link_up=True,
            ),
            SimulationNodeStatus(
                namespace="exo-tb-node-b",
                address="192.168.253.2/30",
                interface_name="thunderbolt0",
                present=True,
                link_up=True,
            ),
        ),
    )


@dataclass
class StaticTransport:
    lines: tuple[str, ...]
    payloads: list[Mapping[str, object]] = field(default_factory=list)

    def __call__(
        self,
        url: str,
        payload: Mapping[str, object],
        headers: Mapping[str, str],
        timeout_seconds: float,
        request_control: SidecarRequestControl,
    ) -> Iterator[str]:
        del url, headers, timeout_seconds, request_control
        self.payloads.append(payload)
        yield from self.lines


@dataclass(frozen=True)
class EchoTransport:
    def __call__(
        self,
        url: str,
        payload: Mapping[str, object],
        headers: Mapping[str, str],
        timeout_seconds: float,
        request_control: SidecarRequestControl,
    ) -> Iterator[str]:
        del url, headers, timeout_seconds, request_control
        messages = cast(list[dict[str, object]], payload["messages"])
        content = cast(str, messages[-1]["content"])
        yield "data: " + json.dumps({"choices": [{"delta": {"content": content}}]})
        yield 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}'
        yield "data: [DONE]"


@dataclass(frozen=True)
class ClosingTransport:
    entered: threading.Event

    def __call__(
        self,
        url: str,
        payload: Mapping[str, object],
        headers: Mapping[str, str],
        timeout_seconds: float,
        request_control: SidecarRequestControl,
    ) -> Iterator[str]:
        del url, payload, headers, timeout_seconds
        closed = threading.Event()
        request_control.attach_response(closed.set)
        self.entered.set()
        while not closed.wait(timeout=0.005):
            yield ": keepalive"
        raise RuntimeError("closed for cancellation")


def make_real_runtime(
    transport: StaticTransport | EchoTransport | ClosingTransport,
) -> ContractRuntime:
    settings = external_settings()
    cancel_sender, cancel_receiver = mp_channel[TaskId]()
    engine: Engine = VllmSidecarEngine(
        model_id=MODEL_ID,
        settings=settings,
        sidecar=VllmSidecarProcess(settings),
        cancel_receiver=cancel_receiver,
        stream_transport=transport,
    )
    return ContractRuntime(settings, engine, cancel_sender, cancel_receiver)


def close_runtime(runtime: ContractRuntime, broker: EngineBroker) -> None:
    broker.close()
    runtime.close_channels()


def receive_until_terminal(
    session: BrokerSession, timeout_seconds: float = 2.0
) -> list[object]:
    deadline = time.monotonic() + timeout_seconds
    results: list[object] = []
    while time.monotonic() < deadline:
        result = session.receive(0.05)
        if result is None:
            continue
        results.append(result)
        if isinstance(result, (CancelledResponse, FinishedResponse)):
            return results
    raise AssertionError("session did not reach a terminal result")


def sse_payloads(response_text: str, event_name: str) -> list[dict[str, object]]:
    payloads: list[dict[str, object]] = []
    for block in response_text.split("\n\n"):
        lines = block.splitlines()
        if f"event: {event_name}" not in lines:
            continue
        data_lines = [
            line.removeprefix("data: ") for line in lines if line.startswith("data: ")
        ]
        payloads.append(cast(dict[str, object], json.loads("\n".join(data_lines))))
    return payloads


def test_chat_stream_uses_real_engine_contract_and_preserves_prompt() -> None:
    dangerous_prompt = 'quotes " new\nline </script> ${not_code} 中文'
    dangerous_output = '<img src=x onerror="window.pwned=1">'
    transport = StaticTransport(
        (
            "data: "
            + json.dumps({"choices": [{"delta": {"content": dangerous_output}}]}),
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
            'data: {"choices":[],"usage":{"prompt_tokens":8,'
            '"completion_tokens":3,"total_tokens":11}}',
            "data: [DONE]",
        )
    )
    runtime = make_real_runtime(transport)
    app = create_app(
        runtime_factory=lambda: runtime,
        health_probe=healthy_sidecar,
        xpu_status_provider=ready_xpu_status,
    )

    with TestClient(app) as client:
        response = client.post("/api/chat", json={"prompt": dangerous_prompt})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["x-exo-request-id"]
    assert response.text.index("event: accepted") < response.text.index("event: delta")
    assert "event: usage" in response.text
    assert "event: finished" in response.text
    assert sse_payloads(response.text, "delta") == [{"text": dangerous_output}]
    messages = cast(list[dict[str, object]], transport.payloads[0]["messages"])
    assert messages == [{"role": "user", "content": dangerous_prompt}]


def test_broker_routes_concurrent_real_engine_results_without_cross_delivery() -> None:
    runtime = make_real_runtime(EchoTransport())
    broker = EngineBroker(runtime.engine, runtime.cancel_sender)
    broker.start()
    try:
        first = broker.submit(ChatRequest(prompt="first"), MODEL_ID)
        second = broker.submit(ChatRequest(prompt="second"), MODEL_ID)
        first_results = receive_until_terminal(first)
        second_results = receive_until_terminal(second)
    finally:
        close_runtime(runtime, broker)

    first_text = "".join(
        result.text for result in first_results if isinstance(result, TokenChunk)
    )
    second_text = "".join(
        result.text for result in second_results if isinstance(result, TokenChunk)
    )
    assert first_text == "first"
    assert second_text == "second"


def test_broker_cancellation_closes_real_sidecar_stream() -> None:
    entered = threading.Event()
    runtime = make_real_runtime(ClosingTransport(entered))
    broker = EngineBroker(runtime.engine, runtime.cancel_sender)
    broker.start()
    try:
        session = broker.submit(
            ChatRequest(prompt="continue for a long time"), MODEL_ID
        )
        assert entered.wait(timeout=1.0)
        assert broker.cancel(session.task_id) == "accepted"
        results = receive_until_terminal(session)
    finally:
        close_runtime(runtime, broker)

    assert any(isinstance(result, CancelledResponse) for result in results)
    assert not any(isinstance(result, FinishedResponse) for result in results)


def test_hardware_gate_requires_torch_xpu_and_intel_xe_sysfs_device() -> None:
    assert evaluate_real_xpu_hardware_gate(ready_xpu_status()).ready is True

    no_torch = ready_xpu_status().model_copy(
        update={"torch_state": "unavailable", "torch_xpu_available": False}
    )
    no_xe = ready_xpu_status().model_copy(
        update={
            "intel_sysfs_devices": (
                IntelXpuSysfsDevice(pci_address="0000:00:02.0", driver="i915"),
            )
        }
    )

    assert evaluate_real_xpu_hardware_gate(no_torch).ready is False
    assert evaluate_real_xpu_hardware_gate(no_xe).ready is False


@pytest.mark.parametrize(
    ("sidecar_ready", "xpu_status", "expected_ready"),
    [
        (True, ready_xpu_status(), True),
        (False, ready_xpu_status(), False),
        (
            True,
            ready_xpu_status().model_copy(update={"torch_xpu_device_count": 0}),
            False,
        ),
    ],
)
def test_readiness_and_chat_are_gated_on_real_hardware(
    sidecar_ready: bool,
    xpu_status: XpuRuntimeStatus,
    expected_ready: bool,
) -> None:
    transport = EchoTransport()
    runtime = make_real_runtime(transport)

    def health(
        settings: VllmSidecarSettings, timeout_seconds: float
    ) -> SidecarHealthStatus:
        del settings, timeout_seconds
        return SidecarHealthStatus(ready=sidecar_ready)

    app = create_app(
        runtime_factory=lambda: runtime,
        health_probe=health,
        xpu_status_provider=lambda: xpu_status,
    )
    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200
        readiness = client.get("/readyz")
        status = client.get("/api/status")
        chat = client.post("/api/chat", json={"prompt": "hello"})

    assert readiness.status_code == (200 if expected_ready else 503)
    assert status.json()["status"] == ("ready" if expected_ready else "not-ready")
    assert status.json()["hardware_gate"]["ready"] is (
        evaluate_real_xpu_hardware_gate(xpu_status).ready
    )
    assert chat.status_code == (200 if expected_ready else 503)


def test_status_redacts_credentials_paths_and_environment() -> None:
    settings = external_settings("http://user:password@sidecar.internal:8123")
    runtime = make_real_runtime(EchoTransport())
    runtime.settings = settings
    engine = cast(VllmSidecarEngine, runtime.engine)
    engine.settings = settings
    app = create_app(
        runtime_factory=lambda: runtime,
        health_probe=healthy_sidecar,
        xpu_status_provider=ready_xpu_status,
    )

    with TestClient(app) as client:
        response = client.get("/api/status")

    serialized = response.text
    assert response.status_code == 200
    assert response.json()["sidecar_endpoint"] == "http://sidecar.internal:8123"
    assert "password" not in serialized
    assert "secret-api-key" not in serialized
    assert "/secret/model/path" not in serialized


def test_security_headers_and_static_client_avoid_active_html_and_storage() -> None:
    runtime = make_real_runtime(EchoTransport())
    app = create_app(
        runtime_factory=lambda: runtime,
        health_probe=healthy_sidecar,
        xpu_status_provider=ready_xpu_status,
    )
    with TestClient(app) as client:
        response = client.get("/")
        javascript = client.get("/vllm-xpu.js")

    assert response.status_code == 200
    assert "script-src 'self'" in response.headers["content-security-policy"]
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "textContent" in javascript.text
    assert "innerHTML" not in javascript.text
    assert "localStorage" not in javascript.text
    assert "sessionStorage" not in javascript.text
    assert "Compute 17 times 23. Give only the integer." in response.text
    assert 'value="0"' in response.text
    assert ">RUN DEMO</button>" in response.text
    assert (
        'id="max-tokens" type="number" min="1" max="1024" value="1024"' in response.text
    )
    assert 'id="temperature"' in response.text
    assert "temperature: Number(elements.temperature.value)" in javascript.text
    assert "Maximum token limit reached" in javascript.text


def test_dashboard_mode_serves_real_dashboard_and_two_node_simulation(
    tmp_path: Path,
) -> None:
    dashboard_directory = tmp_path / "dashboard"
    dashboard_directory.mkdir()
    (dashboard_directory / "index.html").write_text(
        "<!doctype html><title>dashboard-marker</title>",
        encoding="utf-8",
    )
    runtime = make_real_runtime(EchoTransport())
    app = create_app(
        runtime_factory=lambda: runtime,
        health_probe=healthy_sidecar,
        xpu_status_provider=ready_xpu_status,
        simulation_status_provider=active_simulation_status,
        dashboard_directory=dashboard_directory,
    )

    with TestClient(app) as client:
        dashboard = client.get("/")
        contract = client.get("/contract")
        state_response = client.get("/state")
        status_response = client.get("/api/simulation/status")
        feature_flags_response = client.get("/v1/feature-flags")

    assert "dashboard-marker" in dashboard.text
    assert "Intel vLLM Engine Contract Console" in contract.text
    assert '"nodes":["slazenger-panther-lake","thunderbolt-peer-simulated"]' in (
        state_response.text
    )
    assert '"ip_address":"192.168.253.2"' in state_response.text
    assert '"interfaceType":"thunderbolt"' in state_response.text
    assert '"active":true' in status_response.text
    assert '"link_up":true' in status_response.text
    assert '"vllmContractDemo":true' in feature_flags_response.text
    assert "'unsafe-inline'" in dashboard.headers["content-security-policy"]
    assert "img-src 'none'" in contract.headers["content-security-policy"]


def test_unavailable_simulation_keeps_virtual_peer_out_of_cluster() -> None:
    runtime = make_real_runtime(EchoTransport())
    app = create_app(
        runtime_factory=lambda: runtime,
        health_probe=healthy_sidecar,
        xpu_status_provider=ready_xpu_status,
        simulation_status_provider=lambda: ThunderboltSimulationStatus(
            detail="FileNotFoundError"
        ),
    )

    with TestClient(app) as client:
        state_response = client.get("/state")

    assert '"nodes":["slazenger-panther-lake"]' in state_response.text
    assert '"connections":{}' in state_response.text


def test_dashboard_directory_requires_built_index(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="built dashboard index.html"):
        dashboard_directory_from_environment(
            {"EXO_VLLM_DEMO_DASHBOARD_DIR": str(tmp_path)}
        )


def test_cancel_endpoint_reports_unknown_and_terminal_requests() -> None:
    runtime = make_real_runtime(EchoTransport())
    app = create_app(
        runtime_factory=lambda: runtime,
        health_probe=healthy_sidecar,
        xpu_status_provider=ready_xpu_status,
    )
    with TestClient(app) as client:
        unknown = client.post("/api/requests/not-known/cancel")
        completed = client.post("/api/chat", json={"prompt": "complete"})
        task_id = completed.headers["x-exo-request-id"]
        terminal = client.post(f"/api/requests/{task_id}/cancel")

    assert unknown.status_code == 404
    assert terminal.status_code == 409


@pytest.mark.parametrize(
    "payload",
    [
        {"prompt": ""},
        {"prompt": "  \n"},
        {"prompt": "x" * (MAX_PROMPT_CHARACTERS + 1)},
        {"prompt": "valid", "max_tokens": 0},
        {"prompt": "valid", "max_tokens": 1025},
    ],
)
def test_chat_request_rejects_unsafe_bounds(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ChatRequest.model_validate(payload)


def test_sysfs_and_torch_status_are_injectable(tmp_path: Path) -> None:
    sysfs_root = tmp_path / "sys"
    drm = sysfs_root / "class" / "drm"
    intel_device = sysfs_root / "devices" / "pci0000:00" / "0000:00:02.0"
    amd_device = sysfs_root / "devices" / "pci0000:00" / "0000:03:00.0"
    xe_driver = sysfs_root / "bus" / "pci" / "drivers" / "xe"
    for path in (drm, intel_device, amd_device, xe_driver):
        path.mkdir(parents=True, exist_ok=True)
    (intel_device / "vendor").write_text("0x8086\n", encoding="utf-8")
    (intel_device / "device").write_text("0xb080\n", encoding="utf-8")
    (amd_device / "vendor").write_text("0x1002\n", encoding="utf-8")
    (drm / "card0").mkdir()
    (drm / "card1").mkdir()
    (drm / "card0" / "device").symlink_to(intel_device, target_is_directory=True)
    (drm / "card1" / "device").symlink_to(amd_device, target_is_directory=True)
    (intel_device / "driver").symlink_to(xe_driver, target_is_directory=True)

    @dataclass(frozen=True)
    class FakeXpu:
        def is_available(self) -> bool:
            return True

        def device_count(self) -> int:
            return 1

        def get_device_name(self, device: int) -> str:
            assert device == 0
            return "Injected Intel XPU"

    @dataclass(frozen=True)
    class FakeTorch:
        __version__: str = "test-torch"
        xpu: FakeXpu = FakeXpu()

    devices = gather_intel_xpu_sysfs_devices(sysfs_root)
    status = gather_xpu_runtime_status(
        sysfs_root=sysfs_root,
        module_loader=lambda name: FakeTorch() if name == "torch" else object(),
    )

    assert devices == (
        IntelXpuSysfsDevice(
            pci_address="0000:00:02.0",
            device_id="0xb080",
            driver="xe",
        ),
    )
    assert status.torch_state == "available"
    assert status.torch_xpu_device_names == ("Injected Intel XPU",)
    assert evaluate_real_xpu_hardware_gate(status).ready is True


def test_torch_not_installed_remains_distinct_from_missing_sysfs(
    tmp_path: Path,
) -> None:
    def missing_torch(name: str) -> object:
        raise ImportError(name)

    status = gather_xpu_runtime_status(
        sysfs_root=tmp_path,
        module_loader=missing_torch,
    )

    assert status.torch_state == "not-installed"
    assert status.intel_sysfs_devices == ()
    assert evaluate_real_xpu_hardware_gate(status).ready is False


def test_external_settings_require_explicit_url_and_model() -> None:
    with pytest.raises(ValueError, match="MODEL_ID"):
        external_settings_from_environment({"EXO_VLLM_SIDECAR_URL": "http://x"})
    with pytest.raises(ValueError, match="SIDECAR_URL"):
        external_settings_from_environment({"EXO_VLLM_MODEL_ID": "model"})

    settings = external_settings_from_environment(
        {
            "EXO_VLLM_MODEL_ID": str(MODEL_ID),
            "EXO_VLLM_SIDECAR_URL": "http://127.0.0.1:8000",
        }
    )
    assert settings.mode == "external"


def test_redacted_endpoint_removes_url_userinfo() -> None:
    assert (
        redacted_sidecar_endpoint(
            external_settings("https://user:password@example.test:8443")
        )
        == "https://example.test:8443"
    )


def test_health_probe_sends_secret_but_does_not_return_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_headers: dict[str, str] = {}

    def fake_get(
        url: str, *, headers: dict[str, str], timeout: float
    ) -> httpx.Response:
        del timeout
        captured_headers.update(headers)
        return httpx.Response(503, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)
    status = probe_sidecar_health(external_settings())

    assert captured_headers["Authorization"] == "Bearer secret-api-key"
    assert status == SidecarHealthStatus(ready=False, detail="HTTP 503")
    assert "secret-api-key" not in status.model_dump_json()


@dataclass(eq=False)
class CancelAwareFakeEngine(Engine):
    cancel_receiver: MpReceiver[TaskId]
    submitted: threading.Event = field(default_factory=threading.Event)
    task_id: TaskId | None = None
    closed: bool = False
    _cancelled_tasks: set[TaskId] = field(default_factory=set)

    @override
    def warmup(self) -> None:
        pass

    @override
    def submit(self, task: GenerationTask) -> None:
        assert isinstance(task, TextGeneration)
        self.task_id = task.task_id
        self.submitted.set()

    @override
    def step(
        self,
    ) -> list[tuple[TaskId, CancelledResponse]]:
        if self.task_id is None:
            time.sleep(0.005)
            return []
        cancellations = self.cancel_receiver.collect()
        if self.task_id in cancellations:
            task_id = self.task_id
            self.task_id = None
            return [(task_id, CancelledResponse())]
        time.sleep(0.005)
        return []

    @override
    def close(self) -> None:
        self.closed = True

    @override
    def serve_prefill(self, request: PrefillRequest, wfile: BinaryIO) -> None:
        del request, wfile
        raise RuntimeError("not supported")


def test_cancel_endpoint_accepts_active_request() -> None:
    cancel_sender, cancel_receiver = mp_channel[TaskId]()
    fake_engine = CancelAwareFakeEngine(cancel_receiver)
    runtime = ContractRuntime(
        external_settings(), fake_engine, cancel_sender, cancel_receiver
    )
    app = create_app(
        runtime_factory=lambda: runtime,
        health_probe=healthy_sidecar,
        xpu_status_provider=ready_xpu_status,
    )

    with TestClient(app) as client:
        request_thread = threading.Thread(
            target=lambda: client.post("/api/chat", json={"prompt": "wait"}),
            daemon=True,
        )
        request_thread.start()
        assert fake_engine.submitted.wait(timeout=1.0)
        assert fake_engine.task_id is not None
        response = client.post(f"/api/requests/{fake_engine.task_id}/cancel")
        request_thread.join(timeout=2.0)

    assert response.status_code == 202
    assert not request_thread.is_alive()
    assert fake_engine.closed is True


@pytest.mark.parametrize(
    "argument_values",
    [
        ["--port", "0"],
        ["--port", "65536"],
        ["--host", "0.0.0.0"],
        ["--host", "192.0.2.1"],
    ],
)
def test_runner_rejects_invalid_or_implicit_remote_listeners(
    argument_values: list[str],
) -> None:
    parser = RUNNER_MODULE.build_argument_parser()
    arguments = RUNNER_MODULE.parse_arguments(parser, argument_values)

    with pytest.raises(SystemExit) as error:
        RUNNER_MODULE.validate_arguments(arguments, parser)

    assert error.value.code == 2


def test_runner_accepts_loopback_and_explicit_remote_listener() -> None:
    parser = RUNNER_MODULE.build_argument_parser()
    loopback = RUNNER_MODULE.parse_arguments(
        parser, ["--host", "::1", "--port", "52416"]
    )
    explicit_remote = RUNNER_MODULE.parse_arguments(
        parser, ["--host", "0.0.0.0", "--port", "52416", "--allow-remote"]
    )

    RUNNER_MODULE.validate_arguments(loopback, parser)
    RUNNER_MODULE.validate_arguments(explicit_remote, parser)


def test_runner_main_invokes_synchronous_uvicorn_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: list[_RunnerArguments] = []

    def fake_run(arguments: _RunnerArguments) -> None:
        received.append(arguments)

    monkeypatch.setattr(RUNNER_MODULE, "run", fake_run)

    assert RUNNER_MODULE.main(["--host", "localhost", "--port", "52417"]) == 0
    assert len(received) == 1
    assert received[0].host == "localhost"
    assert received[0].port == 52417
