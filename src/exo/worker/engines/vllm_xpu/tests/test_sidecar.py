from __future__ import annotations

import threading
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from pydantic import ValidationError

from exo.shared.models.model_cards import ModelId
from exo.shared.types.chunks import ErrorChunk, TokenChunk, ToolCallChunk
from exo.shared.types.common import CommandId
from exo.shared.types.tasks import TaskId, TextGeneration
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
from exo.utils.channels import mp_channel
from exo.worker.engines.vllm_xpu.sidecar import (
    SidecarRequestControl,
    SidecarResult,
    VllmSidecarEngine,
    VllmSidecarProcess,
    VllmSidecarSettings,
    build_chat_completions_payload,
    build_managed_sidecar_command,
)

MODEL_ID = ModelId("Qwen/Qwen3-1.7B")


def external_settings() -> VllmSidecarSettings:
    return VllmSidecarSettings(
        mode="external",
        model_id=MODEL_ID,
        model_path="/models/qwen",
        base_url="http://127.0.0.1:8123",
        serve_host="127.0.0.1",
        serve_port=8123,
        startup_timeout_seconds=5.0,
        request_timeout_seconds=5.0,
    )


def make_task(**updates: object) -> TextGeneration:
    params: dict[str, object] = {
        "model": MODEL_ID,
        "input": [InputMessage(role="user", content=InputMessageContent("Hello"))],
        "max_output_tokens": 12,
    }
    params.update(updates)
    return TextGeneration(
        task_id=TaskId("task-1"),
        instance_id=InstanceId("instance-1"),
        command_id=CommandId("command-1"),
        task_params=TextGenerationTaskParams.model_validate(params),
    )


@dataclass(frozen=True)
class StaticTransport:
    lines: tuple[str, ...]

    def __call__(
        self,
        url: str,
        payload: Mapping[str, object],
        headers: Mapping[str, str],
        timeout_seconds: float,
        request_control: SidecarRequestControl,
    ) -> Iterator[str]:
        del url, payload, headers, timeout_seconds, request_control
        yield from self.lines


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
        raise RuntimeError("HTTP response closed")


class _SidecarHandler(BaseHTTPRequestHandler):
    request_body = b""

    def do_GET(self) -> None:
        if self.path != "/health":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        content_length = int(self.headers.get("Content-Length", "0"))
        type(self).request_body = self.rfile.read(content_length)
        body = (
            b'data: {"choices":[{"delta":{"content":"HTTP works"}}]}\n\n'
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
            b"data: [DONE]\n\n"
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        del format, args


def collect_until_terminal(
    engine: VllmSidecarEngine, timeout_seconds: float = 2.0
) -> list[tuple[TaskId, SidecarResult]]:
    deadline = time.monotonic() + timeout_seconds
    results: list[tuple[TaskId, SidecarResult]] = []
    while time.monotonic() < deadline:
        results.extend(engine.step())
        if any(
            isinstance(result, (FinishedResponse, CancelledResponse))
            for _, result in results
        ):
            return results
    raise AssertionError("sidecar engine did not return a terminal response")


def test_managed_settings_and_command_from_environment() -> None:
    settings = VllmSidecarSettings.from_environment(
        MODEL_ID,
        model_path=Path("/models/qwen"),
        environment={
            "EXO_VLLM_SIDECAR_HOST": "127.0.0.1",
            "EXO_VLLM_SIDECAR_PORT": "8123",
            "EXO_VLLM_HOST_IP": "192.168.253.1",
            "EXO_VLLM_HEAD_IP": "192.168.253.1",
            "EXO_VLLM_INTERFACE": "thunderbolt0",
            "EXO_VLLM_TENSOR_PARALLEL_SIZE": "2",
            "EXO_VLLM_MAX_MODEL_LENGTH": "512",
            "EXO_VLLM_GPU_MEMORY_UTILIZATION": "0.15",
            "EXO_VLLM_KV_CACHE_MEMORY_BYTES": "67108864",
            "EXO_VLLM_ENFORCE_EAGER": "true",
            "EXO_VLLM_DTYPE": "bfloat16",
            "EXO_VLLM_TRUST_REMOTE_CODE": "true",
            "EXO_VLLM_EXTRA_ARGUMENTS": "--disable-log-requests",
        },
    )

    assert settings.mode == "managed"
    assert build_managed_sidecar_command(settings) == [
        "vllm",
        "serve",
        "/models/qwen",
        "--distributed-executor-backend=ray",
        "--tensor-parallel-size=2",
        "--pipeline-parallel-size=1",
        "--max-model-len=512",
        "--gpu-memory-utilization=0.15",
        "--kv-cache-memory-bytes=67108864",
        "--enforce-eager",
        "--dtype=bfloat16",
        "--trust-remote-code",
        "--host=127.0.0.1",
        "--port=8123",
        "--served-model-name=Qwen/Qwen3-1.7B",
        "--disable-log-requests",
    ]


def test_external_settings_do_not_create_distributed_configuration() -> None:
    settings = VllmSidecarSettings.from_environment(
        MODEL_ID,
        model_path=Path("/models/qwen"),
        environment={"EXO_VLLM_SIDECAR_URL": "http://192.168.253.1:8000"},
    )

    assert settings.mode == "external"
    assert settings.distributed is None
    assert settings.chat_completions_url.endswith("/v1/chat/completions")
    with pytest.raises(ValueError, match="external sidecar"):
        build_managed_sidecar_command(settings)


@pytest.mark.parametrize(
    "environment",
    [
        {"EXO_VLLM_SIDECAR_URL": "unix:///tmp/vllm.sock"},
        {"EXO_VLLM_SIDECAR_PORT": "70000"},
        {"EXO_VLLM_ENFORCE_EAGER": "sometimes"},
        {"EXO_VLLM_DTYPE": "int8"},
        {"EXO_VLLM_REQUEST_TIMEOUT_SECONDS": "0"},
    ],
)
def test_invalid_environment_is_rejected(environment: dict[str, str]) -> None:
    with pytest.raises((ValidationError, ValueError)):
        VllmSidecarSettings.from_environment(
            MODEL_ID,
            model_path=Path("/models/qwen"),
            environment=environment,
        )


def test_request_control_closes_an_attached_http_response() -> None:
    request_control = SidecarRequestControl()
    closed = threading.Event()
    request_control.attach_response(closed.set)

    request_control.cancel()

    assert request_control.cancelled.is_set()
    assert closed.is_set()


def test_payload_preserves_messages_and_sampling_parameters() -> None:
    payload = build_chat_completions_payload(
        make_task(
            chat_template_messages=[
                {"role": "system", "content": "Be concise"},
                {"role": "user", "content": "Hello"},
            ],
            temperature=0.2,
            top_p=0.9,
            top_k=20,
            stop=["END"],
            seed=7,
            enable_thinking=False,
            logprobs=True,
            top_logprobs=2,
        ).task_params,
        MODEL_ID,
    )

    assert payload["messages"] == [
        {"role": "system", "content": "Be concise"},
        {"role": "user", "content": "Hello"},
    ]
    assert payload["temperature"] == 0.2
    assert payload["top_k"] == 20
    assert payload["stop"] == ["END"]
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["stream_options"] == {"include_usage": True}


def test_engine_maps_streaming_content_reasoning_usage_and_finish() -> None:
    cancel_sender, cancel_receiver = mp_channel[TaskId]()
    settings = external_settings()
    engine = VllmSidecarEngine(
        model_id=MODEL_ID,
        settings=settings,
        sidecar=VllmSidecarProcess(settings),
        cancel_receiver=cancel_receiver,
        stream_transport=StaticTransport(
            (
                'data: {"choices":[{"delta":{"reasoning_content":"Think "}}]}',
                'data: {"choices":[{"delta":{"content":"Hello"}}]}',
                'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
                'data: {"choices":[],"usage":{"prompt_tokens":3,'
                '"completion_tokens":2,"total_tokens":5}}',
                "data: [DONE]",
            )
        ),
    )
    try:
        engine.submit(make_task())
        results = collect_until_terminal(engine)
    finally:
        engine.close()
        cancel_sender.close()
        cancel_receiver.close()

    chunks = [result for _, result in results]
    assert isinstance(chunks[0], TokenChunk)
    assert chunks[0].text == "Think "
    assert chunks[0].is_thinking is True
    assert isinstance(chunks[1], TokenChunk)
    assert chunks[1].text == "Hello"
    finish = chunks[2]
    assert isinstance(finish, TokenChunk)
    assert finish.finish_reason == "stop"
    assert finish.usage is not None
    assert finish.usage.total_tokens == 5
    assert isinstance(chunks[3], FinishedResponse)


def test_engine_assembles_streamed_tool_call_fragments() -> None:
    cancel_sender, cancel_receiver = mp_channel[TaskId]()
    settings = external_settings()
    engine = VllmSidecarEngine(
        model_id=MODEL_ID,
        settings=settings,
        sidecar=VllmSidecarProcess(settings),
        cancel_receiver=cancel_receiver,
        stream_transport=StaticTransport(
            (
                'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
                '"id":"call-1","function":{"name":"get_","arguments":"{\\""}}]}}]}',
                'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
                '"function":{"name":"weather","arguments":"city\\":\\"SEA\\"}"}}]},'
                '"finish_reason":"tool_calls"}]}',
                "data: [DONE]",
            )
        ),
    )
    try:
        engine.submit(make_task(tools=[{"type": "function"}]))
        results = collect_until_terminal(engine)
    finally:
        engine.close()
        cancel_sender.close()
        cancel_receiver.close()

    tool_chunk = next(
        result for _, result in results if isinstance(result, ToolCallChunk)
    )
    assert len(tool_chunk.tool_calls) == 1
    assert tool_chunk.tool_calls[0].id == "call-1"
    assert tool_chunk.tool_calls[0].name == "get_weather"
    assert tool_chunk.tool_calls[0].arguments == '{"city":"SEA"}'


def test_engine_reports_invalid_stream_as_error_then_finishes() -> None:
    cancel_sender, cancel_receiver = mp_channel[TaskId]()
    settings = external_settings()
    engine = VllmSidecarEngine(
        model_id=MODEL_ID,
        settings=settings,
        sidecar=VllmSidecarProcess(settings),
        cancel_receiver=cancel_receiver,
        stream_transport=StaticTransport(("data: [DONE]",)),
    )
    try:
        engine.submit(make_task())
        results = collect_until_terminal(engine)
    finally:
        engine.close()
        cancel_sender.close()
        cancel_receiver.close()

    assert isinstance(results[0][1], ErrorChunk)
    assert "finish reason" in results[0][1].error_message
    assert isinstance(results[1][1], FinishedResponse)


def test_engine_cancels_an_active_http_stream_without_finished_response() -> None:
    cancel_sender, cancel_receiver = mp_channel[TaskId]()
    settings = external_settings()
    entered = threading.Event()
    engine = VllmSidecarEngine(
        model_id=MODEL_ID,
        settings=settings,
        sidecar=VllmSidecarProcess(settings),
        cancel_receiver=cancel_receiver,
        stream_transport=ClosingTransport(entered),
    )
    try:
        task = make_task()
        engine.submit(task)
        assert entered.wait(timeout=1.0)
        cancel_sender.send(task.task_id)
        results = collect_until_terminal(engine)
    finally:
        engine.close()
        cancel_sender.close()
        cancel_receiver.close()

    assert any(isinstance(result, CancelledResponse) for _, result in results)
    assert not any(isinstance(result, FinishedResponse) for _, result in results)
    assert not any(isinstance(result, ErrorChunk) for _, result in results)


def test_external_sidecar_health_and_real_http_stream() -> None:
    _SidecarHandler.request_body = b""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _SidecarHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    settings = external_settings().model_copy(
        update={"base_url": f"http://127.0.0.1:{server.server_port}"}
    )
    sidecar = VllmSidecarProcess(settings)
    cancel_sender, cancel_receiver = mp_channel[TaskId]()
    engine = VllmSidecarEngine(
        model_id=MODEL_ID,
        settings=settings,
        sidecar=sidecar,
        cancel_receiver=cancel_receiver,
    )
    try:
        sidecar.start()
        engine.submit(make_task())
        results = collect_until_terminal(engine)
    finally:
        engine.close()
        cancel_sender.close()
        cancel_receiver.close()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=1.0)

    assert b'"model":"Qwen/Qwen3-1.7B"' in _SidecarHandler.request_body
    chunks = [result for _, result in results]
    assert isinstance(chunks[0], TokenChunk)
    assert chunks[0].text == "HTTP works"
    assert isinstance(chunks[-1], FinishedResponse)
