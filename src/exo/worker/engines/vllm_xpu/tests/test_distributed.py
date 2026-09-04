import pytest
from pydantic import ValidationError

from exo.worker.engines.vllm_xpu.distributed import (
    VllmXpuDistributedConfig,
    build_ray_start_command,
    build_vllm_environment,
    build_vllm_serve_command,
)


def create_config(**updates: object) -> VllmXpuDistributedConfig:
    values: dict[str, object] = {
        "model_id": "Qwen/Qwen3-1.7B",
        "host_ip": "192.168.253.2",
        "head_ip": "192.168.253.1",
        "interface_name": "thunderbolt0",
        "tensor_parallel_size": 2,
        "pipeline_parallel_size": 1,
    }
    values.update(updates)
    return VllmXpuDistributedConfig.model_validate(values)


def test_build_vllm_environment_preserves_unrelated_values() -> None:
    environment = build_vllm_environment(create_config(), {"HF_HOME": "/models"})

    assert environment == {
        "HF_HOME": "/models",
        "VLLM_HOST_IP": "192.168.253.2",
        "VLLM_TARGET_DEVICE": "xpu",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "SYCL_CACHE_PERSISTENT": "1",
        "GLOO_SOCKET_IFNAME": "thunderbolt0",
        "CCL_ATL_TRANSPORT": "ofi",
        "FI_PROVIDER": "tcp",
        "FI_TCP_IFACE": "thunderbolt0",
    }


def test_build_ray_commands_use_selected_addresses() -> None:
    config = create_config()

    assert build_ray_start_command(config, "head") == [
        "ray",
        "start",
        "--node-ip-address=192.168.253.2",
        "--head",
        "--port=6379",
    ]
    assert build_ray_start_command(config, "worker") == [
        "ray",
        "start",
        "--node-ip-address=192.168.253.2",
        "--address=192.168.253.1:6379",
    ]


def test_build_vllm_command_contains_xpu_parallelism() -> None:
    command = build_vllm_serve_command(
        create_config(
            max_model_length=8192,
            gpu_memory_utilization=0.15,
            kv_cache_memory_bytes=67108864,
            enforce_eager=True,
        )
    )

    assert command == [
        "vllm",
        "serve",
        "Qwen/Qwen3-1.7B",
        "--device=xpu",
        "--distributed-executor-backend=ray",
        "--tensor-parallel-size=2",
        "--pipeline-parallel-size=1",
        "--max-model-len=8192",
        "--gpu-memory-utilization=0.15",
        "--kv-cache-memory-bytes=67108864",
        "--enforce-eager",
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("host_ip", "not-an-ip"),
        ("interface_name", ""),
        ("head_port", 70000),
        ("tensor_parallel_size", 0),
        ("pipeline_parallel_size", 0),
        ("max_model_length", 0),
        ("gpu_memory_utilization", 0),
        ("gpu_memory_utilization", 1.1),
        ("kv_cache_memory_bytes", 0),
    ],
)
def test_invalid_configuration_is_rejected(field: str, value: object) -> None:
    with pytest.raises((ValidationError, ValueError)):
        create_config(**{field: value})
