from __future__ import annotations

import contextlib
import os
import queue
import shlex
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Generator, Iterable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Literal, Protocol, final
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, model_validator

from exo.api.types import (
    CompletionTokensDetails,
    PromptTokensDetails,
    ToolCallItem,
    TopLogprobItem,
    Usage,
)
from exo.download.download_utils import build_model_path
from exo.shared.constants import EXO_MAX_CONCURRENT_REQUESTS
from exo.shared.models.model_cards import ModelId
from exo.shared.types.chunks import Chunk, ErrorChunk, TokenChunk, ToolCallChunk
from exo.shared.types.tasks import (
    CANCEL_ALL_TASKS,
    GenerationTask,
    TaskId,
    TextGeneration,
)
from exo.shared.types.text_generation import ChatTemplateValue, TextGenerationTaskParams
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runner_response import (
    CancelledResponse,
    FinishedResponse,
    ModelLoadingResponse,
)
from exo.utils.channels import MpReceiver, MpSender
from exo.utils.pydantic_ext import FrozenModel
from exo.worker.disaggregated.server import PrefillRequest
from exo.worker.engines.base import Builder, Engine
from exo.worker.engines.vllm_xpu.distributed import (
    VllmDataType,
    VllmXpuDistributedConfig,
    build_vllm_environment,
    build_vllm_serve_command,
)

if TYPE_CHECKING:
    from exo.shared.types.events import Event

SidecarMode = Literal["managed", "external"]
TokenFinishReason = Literal["stop", "length", "content_filter"]
type JsonValue = (
    str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]
)
type SidecarResult = Chunk | CancelledResponse | FinishedResponse


def _optional_int(environment: Mapping[str, str], name: str) -> int | None:
    value = environment.get(name)
    return None if value in (None, "") else int(value)


def _optional_float(environment: Mapping[str, str], name: str) -> float | None:
    value = environment.get(name)
    return None if value in (None, "") else float(value)


def _positive_float(environment: Mapping[str, str], name: str, default: float) -> float:
    value = float(environment.get(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _environment_bool(
    environment: Mapping[str, str], name: str, default: bool = False
) -> bool:
    value = environment.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def _vllm_dtype(environment: Mapping[str, str]) -> VllmDataType:
    match environment.get("EXO_VLLM_DTYPE", "auto"):
        case "auto":
            return "auto"
        case "half":
            return "half"
        case "float16":
            return "float16"
        case "bfloat16":
            return "bfloat16"
        case "float":
            return "float"
        case "float32":
            return "float32"
        case unsupported:
            raise ValueError(f"unsupported EXO_VLLM_DTYPE: {unsupported}")


@final
class VllmSidecarSettings(FrozenModel):
    mode: SidecarMode
    model_id: ModelId
    model_path: str
    base_url: str
    executable: str = "vllm"
    serve_host: str = "127.0.0.1"
    serve_port: int = 8000
    startup_timeout_seconds: float = 300.0
    request_timeout_seconds: float = 600.0
    api_key: str | None = None
    distributed: VllmXpuDistributedConfig | None = None
    extra_arguments: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_settings(self) -> VllmSidecarSettings:
        parsed = urlsplit(self.base_url)
        if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
            raise ValueError("base_url must be an HTTP or HTTPS URL")
        if parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
            raise ValueError("base_url must not include a path, query, or fragment")
        if not 1 <= self.serve_port <= 65535:
            raise ValueError("serve_port must be between 1 and 65535")
        if self.startup_timeout_seconds <= 0 or self.request_timeout_seconds <= 0:
            raise ValueError("sidecar timeouts must be greater than zero")
        if self.mode == "managed" and self.distributed is None:
            raise ValueError("managed mode requires distributed settings")
        if self.mode == "external" and self.distributed is not None:
            raise ValueError("external mode must not include distributed settings")
        return self

    @property
    def chat_completions_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/v1/chat/completions"

    @property
    def health_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/health"

    @classmethod
    def from_environment(
        cls,
        model_id: ModelId,
        model_path: Path,
        environment: Mapping[str, str] | None = None,
    ) -> VllmSidecarSettings:
        values = dict(os.environ if environment is None else environment)
        external_url = values.get("EXO_VLLM_SIDECAR_URL")
        api_key = values.get("EXO_VLLM_API_KEY") or None
        startup_timeout = _positive_float(
            values, "EXO_VLLM_STARTUP_TIMEOUT_SECONDS", 300.0
        )
        request_timeout = _positive_float(
            values, "EXO_VLLM_REQUEST_TIMEOUT_SECONDS", 600.0
        )

        if external_url:
            parsed = urlsplit(external_url)
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            return cls(
                mode="external",
                model_id=model_id,
                model_path=str(model_path),
                base_url=external_url.rstrip("/"),
                serve_host=parsed.hostname or "127.0.0.1",
                serve_port=port,
                startup_timeout_seconds=startup_timeout,
                request_timeout_seconds=request_timeout,
                api_key=api_key,
            )

        serve_host = values.get("EXO_VLLM_SIDECAR_HOST", "127.0.0.1")
        serve_port = int(values.get("EXO_VLLM_SIDECAR_PORT", "8000"))
        host_ip = values.get("EXO_VLLM_HOST_IP", serve_host)
        distributed = VllmXpuDistributedConfig(
            model_id=str(model_path),
            host_ip=host_ip,
            head_ip=values.get("EXO_VLLM_HEAD_IP", host_ip),
            interface_name=values.get("EXO_VLLM_INTERFACE", "lo"),
            head_port=int(values.get("EXO_VLLM_RAY_PORT", "6379")),
            tensor_parallel_size=int(values.get("EXO_VLLM_TENSOR_PARALLEL_SIZE", "1")),
            pipeline_parallel_size=int(
                values.get("EXO_VLLM_PIPELINE_PARALLEL_SIZE", "1")
            ),
            max_model_length=_optional_int(values, "EXO_VLLM_MAX_MODEL_LENGTH"),
            gpu_memory_utilization=_optional_float(
                values, "EXO_VLLM_GPU_MEMORY_UTILIZATION"
            ),
            kv_cache_memory_bytes=_optional_int(
                values, "EXO_VLLM_KV_CACHE_MEMORY_BYTES"
            ),
            enforce_eager=_environment_bool(values, "EXO_VLLM_ENFORCE_EAGER"),
            dtype=_vllm_dtype(values),
            trust_remote_code=_environment_bool(values, "EXO_VLLM_TRUST_REMOTE_CODE"),
        )
        return cls(
            mode="managed",
            model_id=model_id,
            model_path=str(model_path),
            base_url=f"http://{serve_host}:{serve_port}",
            executable=values.get("EXO_VLLM_EXECUTABLE", "vllm"),
            serve_host=serve_host,
            serve_port=serve_port,
            startup_timeout_seconds=startup_timeout,
            request_timeout_seconds=request_timeout,
            api_key=api_key,
            distributed=distributed,
            extra_arguments=tuple(
                shlex.split(values.get("EXO_VLLM_EXTRA_ARGUMENTS", ""))
            ),
        )


def build_managed_sidecar_command(settings: VllmSidecarSettings) -> list[str]:
    if settings.mode != "managed" or settings.distributed is None:
        raise ValueError("cannot build a managed command for an external sidecar")
    return [
        *build_vllm_serve_command(
            settings.distributed,
            executable=settings.executable,
        ),
        f"--host={settings.serve_host}",
        f"--port={settings.serve_port}",
        f"--served-model-name={settings.model_id}",
        *settings.extra_arguments,
    ]


@dataclass(eq=False)
class VllmSidecarProcess:
    settings: VllmSidecarSettings
    _process: subprocess.Popen[str] | None = field(default=None, init=False)

    def _headers(self) -> dict[str, str]:
        if self.settings.api_key is None:
            return {}
        return {"Authorization": f"Bearer {self.settings.api_key}"}

    def start(self) -> None:
        if self.settings.mode == "managed":
            assert self.settings.distributed is not None
            environment = build_vllm_environment(self.settings.distributed, os.environ)
            self._process = subprocess.Popen(
                build_managed_sidecar_command(self.settings),
                env=environment,
                text=True,
                start_new_session=True,
            )
        self.ensure_ready(self.settings.startup_timeout_seconds)

    def ensure_ready(self, timeout_seconds: float | None = None) -> None:
        timeout = timeout_seconds or self.settings.startup_timeout_seconds
        deadline = time.monotonic() + timeout
        last_error = "health endpoint did not respond"
        while time.monotonic() < deadline:
            if self._process is not None and self._process.poll() is not None:
                raise RuntimeError(
                    f"managed vLLM sidecar exited with code {self._process.returncode}"
                )
            try:
                response = httpx.get(
                    self.settings.health_url,
                    headers=self._headers(),
                    timeout=min(5.0, max(0.1, deadline - time.monotonic())),
                )
                response.raise_for_status()
                return
            except httpx.HTTPError as error:
                last_error = str(error)
                time.sleep(0.25)
        self.close()
        raise TimeoutError(
            f"vLLM sidecar was not ready after {timeout:.1f} seconds: {last_error}"
        )

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None or process.poll() is not None:
            return
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5.0)


class _VllmFunctionDelta(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    name: str | None = None
    arguments: str | None = None


class _VllmToolCallDelta(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    index: int = 0
    id: str | None = None
    function: _VllmFunctionDelta | None = None


class _VllmDelta(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[_VllmToolCallDelta] | None = None


class _VllmLogprobToken(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    token: str
    logprob: float
    bytes: list[int] | None = None


class _VllmChoiceLogprobs(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    content: list[_VllmLogprobToken] | None = None


class _VllmStreamChoice(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    delta: _VllmDelta
    finish_reason: str | None = None
    logprobs: _VllmChoiceLogprobs | None = None


class _VllmUsage(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class _VllmStreamEvent(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    choices: list[_VllmStreamChoice] = []
    usage: _VllmUsage | None = None


@dataclass
class _ToolCallParts:
    identifier: str | None = None
    name: str = ""
    arguments: str = ""


@dataclass
class _StreamAccumulator:
    model_id: ModelId
    token_index: int = 0
    finish_reason: str | None = None
    usage: Usage | None = None
    tool_calls: dict[int, _ToolCallParts] = field(default_factory=dict)

    def _next_token_id(self) -> int:
        token_id = self.token_index
        self.token_index += 1
        return token_id

    def consume(self, event: _VllmStreamEvent) -> list[Chunk]:
        if event.usage is not None:
            self.usage = Usage(
                prompt_tokens=event.usage.prompt_tokens,
                completion_tokens=event.usage.completion_tokens,
                total_tokens=event.usage.total_tokens,
                prompt_tokens_details=PromptTokensDetails(),
                completion_tokens_details=CompletionTokensDetails(),
            )

        chunks: list[Chunk] = []
        for choice in event.choices:
            if choice.finish_reason is not None:
                self.finish_reason = choice.finish_reason
            for tool_delta in choice.delta.tool_calls or []:
                parts = self.tool_calls.setdefault(tool_delta.index, _ToolCallParts())
                if tool_delta.id is not None:
                    parts.identifier = tool_delta.id
                if tool_delta.function is not None:
                    parts.name += tool_delta.function.name or ""
                    parts.arguments += tool_delta.function.arguments or ""

            logprob: float | None = None
            top_logprobs: list[TopLogprobItem] | None = None
            if choice.logprobs is not None and choice.logprobs.content:
                item = choice.logprobs.content[-1]
                logprob = item.logprob
                top_logprobs = [
                    TopLogprobItem(
                        token=item.token,
                        logprob=item.logprob,
                        bytes=item.bytes,
                    )
                ]

            if choice.delta.reasoning_content:
                chunks.append(
                    TokenChunk(
                        model=self.model_id,
                        text=choice.delta.reasoning_content,
                        token_id=self._next_token_id(),
                        usage=None,
                        is_thinking=True,
                    )
                )
            if choice.delta.content:
                chunks.append(
                    TokenChunk(
                        model=self.model_id,
                        text=choice.delta.content,
                        token_id=self._next_token_id(),
                        usage=None,
                        logprob=logprob,
                        top_logprobs=top_logprobs,
                    )
                )
        return chunks

    def finish(self) -> Chunk:
        if self.finish_reason in {"tool_calls", "function_call"}:
            completed = [
                ToolCallItem(
                    id=parts.identifier or f"sidecar-tool-{index}",
                    name=parts.name,
                    arguments=parts.arguments,
                )
                for index, parts in sorted(self.tool_calls.items())
            ]
            if not completed:
                raise RuntimeError(
                    "vLLM finished with tool_calls but returned no calls"
                )
            return ToolCallChunk(
                model=self.model_id,
                tool_calls=completed,
                usage=self.usage,
            )

        match self.finish_reason:
            case "stop":
                finish_reason: TokenFinishReason = "stop"
            case "length":
                finish_reason = "length"
            case "content_filter":
                finish_reason = "content_filter"
            case unsupported:
                raise RuntimeError(
                    "vLLM stream ended without a supported finish reason: "
                    f"{unsupported}"
                )
        return TokenChunk(
            model=self.model_id,
            text="",
            token_id=self._next_token_id(),
            usage=self.usage,
            finish_reason=finish_reason,
        )


class SidecarStreamTransport(Protocol):
    def __call__(
        self,
        url: str,
        payload: Mapping[str, object],
        headers: Mapping[str, str],
        timeout_seconds: float,
        request_control: SidecarRequestControl,
    ) -> Iterator[str]: ...


@dataclass(eq=False)
class SidecarRequestControl:
    cancelled: threading.Event = field(default_factory=threading.Event)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _close_response: Callable[[], None] | None = None

    def attach_response(self, close_response: Callable[[], None]) -> None:
        close_immediately = False
        with self._lock:
            if self.cancelled.is_set():
                close_immediately = True
            else:
                self._close_response = close_response
        if close_immediately:
            with contextlib.suppress(Exception):
                close_response()

    def detach_response(self) -> None:
        with self._lock:
            self._close_response = None

    def cancel(self) -> None:
        self.cancelled.set()
        with self._lock:
            close_response = self._close_response
        if close_response is not None:
            with contextlib.suppress(Exception):
                close_response()


def _httpx_stream_transport(
    url: str,
    payload: Mapping[str, object],
    headers: Mapping[str, str],
    timeout_seconds: float,
    request_control: SidecarRequestControl,
) -> Iterator[str]:
    timeout = httpx.Timeout(timeout_seconds, connect=min(10.0, timeout_seconds))
    with (
        httpx.Client(timeout=timeout) as client,
        client.stream(
            "POST", url, json=dict(payload), headers=dict(headers)
        ) as response,
    ):
        request_control.attach_response(response.close)
        if response.status_code >= 400:
            response.read()
            body = response.text[:1000]
            raise RuntimeError(
                f"vLLM sidecar returned HTTP {response.status_code}: {body}"
            )
        try:
            for line in response.iter_lines():
                yield line
        finally:
            request_control.detach_response()


def _json_chat_value(value: ChatTemplateValue) -> JsonValue:
    if isinstance(value, str):
        return str(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, list):
        return [_json_chat_value(item) for item in value]
    return {key: _json_chat_value(item) for key, item in value.items()}


def build_chat_completions_payload(
    task_params: TextGenerationTaskParams,
    served_model_name: ModelId,
) -> dict[str, object]:
    if task_params.chat_template_messages is not None:
        messages: list[dict[str, JsonValue]] = [
            {key: _json_chat_value(value) for key, value in message.items()}
            for message in task_params.chat_template_messages
        ]
    else:
        messages = []
        if task_params.instructions is not None:
            messages.append(
                {"role": "system", "content": str(task_params.instructions)}
            )
        messages.extend(
            {"role": message.role, "content": str(message.content)}
            for message in task_params.input
        )

    payload: dict[str, object] = {
        "model": str(served_model_name),
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    optional_values: tuple[tuple[str, object | None], ...] = (
        ("max_tokens", task_params.max_output_tokens),
        ("temperature", task_params.temperature),
        ("top_p", task_params.top_p),
        ("top_k", task_params.top_k),
        ("min_p", task_params.min_p),
        ("stop", task_params.stop),
        ("seed", task_params.seed),
        ("repetition_penalty", task_params.repetition_penalty),
        ("presence_penalty", task_params.presence_penalty),
        ("frequency_penalty", task_params.frequency_penalty),
        ("tools", task_params.tools),
    )
    for name, value in optional_values:
        if value is not None:
            payload[name] = value
    if task_params.logprobs:
        payload["logprobs"] = True
        if task_params.top_logprobs is not None:
            payload["top_logprobs"] = task_params.top_logprobs
    if task_params.enable_thinking is not None:
        payload["chat_template_kwargs"] = {
            "enable_thinking": task_params.enable_thinking
        }
    return payload


@dataclass(eq=False)
class VllmSidecarEngine(Engine):
    model_id: ModelId
    settings: VllmSidecarSettings
    sidecar: VllmSidecarProcess
    cancel_receiver: MpReceiver[TaskId]
    stream_transport: SidecarStreamTransport = _httpx_stream_transport
    _cancelled_tasks: set[TaskId] = field(default_factory=set, init=False)
    _active: dict[TaskId, SidecarRequestControl] = field(
        default_factory=dict, init=False
    )
    _results: queue.Queue[tuple[TaskId, SidecarResult]] = field(
        default_factory=queue.Queue, init=False
    )
    _executor: ThreadPoolExecutor = field(init=False)
    _closed: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=EXO_MAX_CONCURRENT_REQUESTS,
            thread_name_prefix="vllm-sidecar",
        )

    def warmup(self) -> None:
        self.sidecar.ensure_ready()

    def submit(self, task: GenerationTask) -> None:
        if not isinstance(task, TextGeneration):
            raise TypeError("the vLLM sidecar supports text generation only")
        if self._closed:
            raise RuntimeError("the vLLM sidecar engine is closed")
        self._cancelled_tasks.discard(CANCEL_ALL_TASKS)
        request_control = SidecarRequestControl()
        self._active[task.task_id] = request_control
        self._executor.submit(self._run_request, task, request_control)

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "text/event-stream"}
        if self.settings.api_key is not None:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"
        return headers

    def _run_request(
        self, task: TextGeneration, request_control: SidecarRequestControl
    ) -> None:
        accumulator = _StreamAccumulator(self.model_id)
        payload = build_chat_completions_payload(task.task_params, self.model_id)
        try:
            for line in self.stream_transport(
                self.settings.chat_completions_url,
                payload,
                self._headers(),
                self.settings.request_timeout_seconds,
                request_control,
            ):
                if request_control.cancelled.is_set():
                    return
                if not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if data == "[DONE]":
                    break
                event = _VllmStreamEvent.model_validate_json(data)
                for chunk in accumulator.consume(event):
                    self._results.put((task.task_id, chunk))
            if request_control.cancelled.is_set():
                return
            self._results.put((task.task_id, accumulator.finish()))
        except Exception as error:
            if not request_control.cancelled.is_set():
                self._results.put(
                    (
                        task.task_id,
                        ErrorChunk(model=self.model_id, error_message=str(error)),
                    )
                )
        finally:
            terminal: CancelledResponse | FinishedResponse = (
                CancelledResponse()
                if request_control.cancelled.is_set()
                else FinishedResponse()
            )
            self._results.put((task.task_id, terminal))

    def _collect_cancellations(self) -> None:
        for task_id in self.cancel_receiver.collect():
            self._cancelled_tasks.add(task_id)
            if task_id == CANCEL_ALL_TASKS:
                for request_control in self._active.values():
                    request_control.cancel()
            elif (request_control := self._active.get(task_id)) is not None:
                request_control.cancel()

    def step(self) -> Iterable[tuple[TaskId, SidecarResult]]:
        self._collect_cancellations()
        results: list[tuple[TaskId, SidecarResult]] = []
        try:
            results.append(self._results.get(timeout=0.05))
        except queue.Empty:
            return results
        while True:
            try:
                results.append(self._results.get_nowait())
            except queue.Empty:
                break
        for task_id, result in results:
            if isinstance(result, (CancelledResponse, FinishedResponse)):
                self._active.pop(task_id, None)
        return results

    def serve_prefill(self, request: PrefillRequest, wfile: BinaryIO) -> None:
        del request, wfile
        raise RuntimeError("vLLM sidecars do not support Exo disaggregated prefill")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for request_control in self._active.values():
            request_control.cancel()
        self._executor.shutdown(wait=True, cancel_futures=True)
        self.sidecar.close()


@dataclass(eq=False)
class VllmSidecarBuilder(Builder):
    model_id: ModelId
    event_sender: MpSender[Event]
    cancel_receiver: MpReceiver[TaskId]
    settings: VllmSidecarSettings | None = None
    sidecar: VllmSidecarProcess | None = None

    def connect(self, bound_instance: BoundInstance) -> None:
        del bound_instance

    def load(self, bound_instance: BoundInstance) -> Generator[ModelLoadingResponse]:
        model_path = build_model_path(self.model_id)
        settings = self.settings or VllmSidecarSettings.from_environment(
            self.model_id, model_path
        )
        sidecar = self.sidecar or VllmSidecarProcess(settings)
        self.settings = settings
        self.sidecar = sidecar
        sidecar.start()
        total_layers = bound_instance.bound_shard.n_layers
        yield ModelLoadingResponse(
            layers_loaded=total_layers,
            total=total_layers,
        )

    def build(self) -> Engine:
        if self.settings is None or self.sidecar is None:
            raise RuntimeError("the vLLM sidecar must be loaded before build")
        return VllmSidecarEngine(
            model_id=self.model_id,
            settings=self.settings,
            sidecar=self.sidecar,
            cancel_receiver=self.cancel_receiver,
        )

    def close(self) -> None:
        if self.sidecar is not None:
            self.sidecar.close()
