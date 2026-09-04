from __future__ import annotations

from collections.abc import Mapping
from ipaddress import ip_address
from typing import Literal, Self, final

from pydantic import model_validator

from exo.utils.pydantic_ext import FrozenModel

RayNodeRole = Literal["head", "worker"]


@final
class VllmXpuDistributedConfig(FrozenModel):
    """Validated inputs for one vLLM XPU process in a distributed cluster."""

    model_id: str
    host_ip: str
    head_ip: str
    interface_name: str
    head_port: int = 6379
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    max_model_length: int | None = None
    gpu_memory_utilization: float | None = None
    kv_cache_memory_bytes: int | None = None
    enforce_eager: bool = False

    @property
    def world_size(self) -> int:
        return self.tensor_parallel_size * self.pipeline_parallel_size

    @model_validator(mode="after")
    def validate_values(self) -> Self:
        ip_address(self.host_ip)
        ip_address(self.head_ip)
        if self.interface_name.strip() == "":
            raise ValueError("interface_name must not be empty")
        if not 1 <= self.head_port <= 65535:
            raise ValueError("head_port must be between 1 and 65535")
        if self.tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size must be at least 1")
        if self.pipeline_parallel_size < 1:
            raise ValueError("pipeline_parallel_size must be at least 1")
        if self.max_model_length is not None and self.max_model_length < 1:
            raise ValueError("max_model_length must be at least 1")
        if self.gpu_memory_utilization is not None and not (
            0 < self.gpu_memory_utilization <= 1
        ):
            raise ValueError(
                "gpu_memory_utilization must be greater than 0 and at most 1"
            )
        if self.kv_cache_memory_bytes is not None and self.kv_cache_memory_bytes < 1:
            raise ValueError("kv_cache_memory_bytes must be at least 1")
        return self


def build_vllm_environment(
    config: VllmXpuDistributedConfig,
    base_environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build the explicit Intel collective environment for a selected interface."""
    environment = dict(base_environment or {})
    environment.update(
        {
            "VLLM_HOST_IP": config.host_ip,
            "VLLM_TARGET_DEVICE": "xpu",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "SYCL_CACHE_PERSISTENT": "1",
            "GLOO_SOCKET_IFNAME": config.interface_name,
            "CCL_ATL_TRANSPORT": "ofi",
            "FI_PROVIDER": "tcp",
            "FI_TCP_IFACE": config.interface_name,
        }
    )
    return environment


def build_ray_start_command(
    config: VllmXpuDistributedConfig,
    role: RayNodeRole,
    *,
    executable: str = "ray",
) -> list[str]:
    """Build a Ray command whose advertised address uses the selected data path."""
    command = [executable, "start", f"--node-ip-address={config.host_ip}"]
    if role == "head":
        command.extend(["--head", f"--port={config.head_port}"])
    else:
        command.append(f"--address={config.head_ip}:{config.head_port}")
    return command


def build_vllm_serve_command(
    config: VllmXpuDistributedConfig,
    *,
    executable: str = "vllm",
) -> list[str]:
    """Build a multi-node-capable Intel XPU vLLM serving command."""
    command = [
        executable,
        "serve",
        config.model_id,
        "--device=xpu",
        "--distributed-executor-backend=ray",
        f"--tensor-parallel-size={config.tensor_parallel_size}",
        f"--pipeline-parallel-size={config.pipeline_parallel_size}",
    ]
    if config.max_model_length is not None:
        command.append(f"--max-model-len={config.max_model_length}")
    if config.gpu_memory_utilization is not None:
        command.append(f"--gpu-memory-utilization={config.gpu_memory_utilization}")
    if config.kv_cache_memory_bytes is not None:
        command.append(f"--kv-cache-memory-bytes={config.kv_cache_memory_bytes}")
    if config.enforce_eager:
        command.append("--enforce-eager")
    return command
