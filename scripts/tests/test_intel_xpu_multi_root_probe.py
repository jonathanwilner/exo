from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from scripts.intel_xpu_multi_root_probe import (
    TORCH_XPU_COUNT_PROGRAM,
    CommandResult,
    build_argument_parser,
    build_probe_environment,
    build_sycl_inventory_command,
    build_torch_inventory_command,
    build_xccl_smoke_command,
    reported_xpu_device_count,
    validate_arguments,
)


def test_default_environment_removes_ambient_multi_root_opt_in() -> None:
    source = {
        "PATH": "/bin",
        "NEOReadDebugKeys": "1",
        "CreateMultipleRootDevices": "8",
    }

    result = build_probe_environment(
        source,
        enable_experimental_multi_root=False,
        force_preemption_mode_3=False,
    )

    assert "NEOReadDebugKeys" not in result
    assert "CreateMultipleRootDevices" not in result
    assert source["CreateMultipleRootDevices"] == "8"


def test_experimental_environment_sets_exact_two_root_configuration() -> None:
    result = build_probe_environment(
        {"PATH": "/bin"},
        enable_experimental_multi_root=True,
        force_preemption_mode_3=False,
    )

    assert result["NEOReadDebugKeys"] == "1"
    assert result["CreateMultipleRootDevices"] == "2"
    assert "ForcePreemptionMode" not in result


def test_force_preemption_mode_is_only_explicit_or_inherited() -> None:
    inherited = build_probe_environment(
        {"ForcePreemptionMode": "7"},
        enable_experimental_multi_root=False,
        force_preemption_mode_3=False,
    )
    explicit = build_probe_environment(
        {},
        enable_experimental_multi_root=False,
        force_preemption_mode_3=True,
    )

    assert inherited["ForcePreemptionMode"] == "7"
    assert explicit["ForcePreemptionMode"] == "3"


def test_inventory_command_builders_are_small_and_read_only() -> None:
    assert build_sycl_inventory_command("/usr/bin/sycl-ls") == ("/usr/bin/sycl-ls",)
    assert build_torch_inventory_command("/usr/bin/python") == (
        "/usr/bin/python",
        "-c",
        TORCH_XPU_COUNT_PROGRAM,
    )


def test_xccl_command_runs_two_workers_through_torchrun() -> None:
    command = build_xccl_smoke_command(
        "/usr/bin/python",
        Path("/repo/scripts/intel_xpu_multi_root_probe.py"),
        timeout_seconds=15.0,
    )

    assert command == (
        "/usr/bin/python",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=2",
        "/repo/scripts/intel_xpu_multi_root_probe.py",
        "--xccl-worker",
        "--timeout-seconds",
        "15.0",
    )


@pytest.mark.parametrize(
    ("standard_output", "expected"),
    [
        ('{"available": true, "device_count": 2}\n', 2),
        ('diagnostic\n{"device_count": 1}\n', 1),
        ('{"device_count": true}\n', None),
        ("not json\n", None),
    ],
)
def test_reported_xpu_device_count(standard_output: str, expected: int | None) -> None:
    result = CommandResult(
        label="inventory",
        command=("python",),
        return_code=0,
        standard_output=standard_output,
        standard_error="",
    )

    assert reported_xpu_device_count(result) == expected


def test_xccl_requires_explicit_multi_root_opt_in() -> None:
    parser = build_argument_parser()
    arguments = argparse.Namespace(
        timeout_seconds=30.0,
        run_xccl_smoke=True,
        enable_experimental_multi_root=False,
    )

    with pytest.raises(SystemExit, match="2"):
        validate_arguments(arguments, parser)


@pytest.mark.parametrize("timeout_seconds", [4.9, 120.1])
def test_timeout_has_strict_bounds(timeout_seconds: float) -> None:
    parser = build_argument_parser()
    arguments = argparse.Namespace(
        timeout_seconds=timeout_seconds,
        run_xccl_smoke=False,
        enable_experimental_multi_root=False,
    )

    with pytest.raises(SystemExit, match="2"):
        validate_arguments(arguments, parser)
