#!/usr/bin/env python3
"""Inspect Intel XPU support and optionally run a simulated two-rank XCCL probe.

The default action is a read-only baseline inventory. The experimental Level Zero
multi-root variables are only enabled with ``--enable-experimental-multi-root``.
This tool never launches vLLM or loads a model.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

EXPERIMENTAL_DEVICE_COUNT = "2"
DEFAULT_TIMEOUT_SECONDS = 30.0
MINIMUM_TIMEOUT_SECONDS = 5.0
MAXIMUM_TIMEOUT_SECONDS = 120.0

TORCH_XPU_COUNT_PROGRAM = """\
import json
try:
    import torch
except Exception as error:
    print(json.dumps({"available": False, "error": f"torch import failed: {error}"}))
    raise SystemExit(2)

available = bool(torch.xpu.is_available())
count = int(torch.xpu.device_count()) if available else 0
print(json.dumps({"available": available, "device_count": count}))
raise SystemExit(0 if available else 2)
"""


@dataclass(frozen=True)
class CommandResult:
    label: str
    command: tuple[str, ...]
    return_code: int | None
    standard_output: str
    standard_error: str
    timed_out: bool = False
    skipped_reason: str | None = None

    @property
    def successful(self) -> bool:
        return self.return_code == 0 and not self.timed_out


def build_probe_environment(
    base_environment: Mapping[str, str],
    *,
    enable_experimental_multi_root: bool,
    force_preemption_mode_3: bool,
) -> dict[str, str]:
    """Build a child environment without mutating the caller's environment."""
    environment = dict(base_environment)

    # An ambient shell must not accidentally opt the probe into an experimental
    # topology. ForcePreemptionMode is different: an inherited workaround remains
    # in force unless the caller explicitly launches from a clean environment.
    environment.pop("NEOReadDebugKeys", None)
    environment.pop("CreateMultipleRootDevices", None)
    if enable_experimental_multi_root:
        environment["NEOReadDebugKeys"] = "1"
        environment["CreateMultipleRootDevices"] = EXPERIMENTAL_DEVICE_COUNT

    if force_preemption_mode_3:
        environment["ForcePreemptionMode"] = "3"

    return environment


def build_sycl_inventory_command(sycl_ls_path: str) -> tuple[str, ...]:
    return (sycl_ls_path,)


def build_torch_inventory_command(python_path: str) -> tuple[str, ...]:
    return (python_path, "-c", TORCH_XPU_COUNT_PROGRAM)


def build_xccl_smoke_command(
    python_path: str,
    script_path: Path,
    *,
    timeout_seconds: float,
) -> tuple[str, ...]:
    return (
        python_path,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=2",
        str(script_path),
        "--xccl-worker",
        "--timeout-seconds",
        str(timeout_seconds),
    )


def run_command(
    label: str,
    command: Sequence[str],
    *,
    environment: Mapping[str, str],
    timeout_seconds: float,
) -> CommandResult:
    process = subprocess.Popen(
        command,
        env=dict(environment),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        standard_output, standard_error = process.communicate(timeout=timeout_seconds)
        return CommandResult(
            label=label,
            command=tuple(command),
            return_code=process.returncode,
            standard_output=standard_output,
            standard_error=standard_error,
        )
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            standard_output, standard_error = process.communicate(timeout=2.0)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            standard_output, standard_error = process.communicate()
        return CommandResult(
            label=label,
            command=tuple(command),
            return_code=process.returncode,
            standard_output=standard_output,
            standard_error=standard_error,
            timed_out=True,
        )


def skipped_result(label: str, reason: str) -> CommandResult:
    return CommandResult(
        label=label,
        command=(),
        return_code=None,
        standard_output="",
        standard_error="",
        skipped_reason=reason,
    )


def reported_xpu_device_count(result: CommandResult) -> int | None:
    for line in reversed(result.standard_output.splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        count = payload.get("device_count")
        if isinstance(count, int) and not isinstance(count, bool):
            return count
    return None


def run_inventory(
    label_prefix: str,
    *,
    environment: Mapping[str, str],
    timeout_seconds: float,
) -> list[CommandResult]:
    results: list[CommandResult] = []
    sycl_ls_path = shutil.which("sycl-ls", path=environment.get("PATH"))
    if sycl_ls_path is None:
        results.append(skipped_result(f"{label_prefix} sycl-ls", "sycl-ls not found"))
    else:
        results.append(
            run_command(
                f"{label_prefix} sycl-ls",
                build_sycl_inventory_command(sycl_ls_path),
                environment=environment,
                timeout_seconds=timeout_seconds,
            )
        )

    results.append(
        run_command(
            f"{label_prefix} torch.xpu.device_count",
            build_torch_inventory_command(sys.executable),
            environment=environment,
            timeout_seconds=timeout_seconds,
        )
    )
    return results


def print_result(result: CommandResult) -> None:
    if result.skipped_reason is not None:
        print(f"[{result.label}] SKIPPED: {result.skipped_reason}")
        return

    status = (
        "TIMEOUT" if result.timed_out else ("PASS" if result.successful else "FAIL")
    )
    print(f"[{result.label}] {status}")
    if result.standard_output.strip():
        print(result.standard_output.rstrip())
    if result.standard_error.strip():
        print(result.standard_error.rstrip(), file=sys.stderr)


def run_xccl_worker(timeout_seconds: float) -> int:
    """Run inside torchrun; allocate only four XPU float32 values per rank."""
    import torch
    from torch import distributed

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    torch.xpu.set_device(local_rank)
    distributed.init_process_group(
        backend="xccl",
        timeout=timedelta(seconds=timeout_seconds),
    )
    try:
        values = torch.full(
            (4,),
            float(rank + 1),
            dtype=torch.float32,
            device=f"xpu:{local_rank}",
        )
        distributed.all_reduce(values)
        expected = torch.full_like(values, 3.0)
        passed = bool(torch.equal(values, expected))
        print(
            json.dumps(
                {
                    "rank": rank,
                    "local_rank": local_rank,
                    "result": values.cpu().tolist(),
                    "passed": passed,
                    "simulation": "experimental single-machine multi-root",
                },
                sort_keys=True,
            ),
            flush=True,
        )
        distributed.barrier()
        return 0 if passed else 1
    finally:
        distributed.destroy_process_group()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--enable-experimental-multi-root",
        action="store_true",
        help=(
            "Explicitly set NEOReadDebugKeys=1 and CreateMultipleRootDevices=2 "
            "for an experimental second inventory"
        ),
    )
    parser.add_argument(
        "--run-xccl-smoke",
        action="store_true",
        help="Run a tiny two-rank XCCL all-reduce after both inventories pass",
    )
    parser.add_argument(
        "--force-preemption-mode-3",
        action="store_true",
        help="Explicitly set the Panther Lake ForcePreemptionMode=3 workaround",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Per-command timeout from {MINIMUM_TIMEOUT_SECONDS:g} to {MAXIMUM_TIMEOUT_SECONDS:g} seconds",
    )
    parser.add_argument("--xccl-worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def validate_arguments(
    arguments: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    if (
        not MINIMUM_TIMEOUT_SECONDS
        <= arguments.timeout_seconds
        <= MAXIMUM_TIMEOUT_SECONDS
    ):
        parser.error(
            f"--timeout-seconds must be between {MINIMUM_TIMEOUT_SECONDS:g} "
            f"and {MAXIMUM_TIMEOUT_SECONDS:g}"
        )
    if arguments.run_xccl_smoke and not arguments.enable_experimental_multi_root:
        parser.error("--run-xccl-smoke requires --enable-experimental-multi-root")


def main(argument_values: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    arguments = parser.parse_args(argument_values)
    validate_arguments(arguments, parser)

    if arguments.xccl_worker:
        return run_xccl_worker(arguments.timeout_seconds)

    print(
        "SIMULATED/EXPERIMENTAL PROBE: results do not represent two-machine "
        "correctness or performance. No model will be launched."
    )
    baseline_environment = build_probe_environment(
        os.environ,
        enable_experimental_multi_root=False,
        force_preemption_mode_3=arguments.force_preemption_mode_3,
    )
    baseline_results = run_inventory(
        "baseline",
        environment=baseline_environment,
        timeout_seconds=arguments.timeout_seconds,
    )
    for result in baseline_results:
        print_result(result)

    if not arguments.enable_experimental_multi_root:
        print(
            "Dry-run inspection complete. Use --enable-experimental-multi-root "
            "to opt in to the Level Zero simulation."
        )
        return (
            0
            if all(
                result.successful or result.skipped_reason is not None
                for result in baseline_results
            )
            else 1
        )

    experimental_environment = build_probe_environment(
        os.environ,
        enable_experimental_multi_root=True,
        force_preemption_mode_3=arguments.force_preemption_mode_3,
    )
    experimental_results = run_inventory(
        "experimental multi-root",
        environment=experimental_environment,
        timeout_seconds=arguments.timeout_seconds,
    )
    for result in experimental_results:
        print_result(result)

    if not all(result.successful for result in experimental_results):
        print("Experimental inventory failed; XCCL smoke test will not run.")
        return 1

    torch_inventory_result = experimental_results[-1]
    if reported_xpu_device_count(torch_inventory_result) != 2:
        print(
            "Experimental inventory did not expose exactly two XPU devices; "
            "XCCL smoke test will not run."
        )
        return 1

    if not arguments.run_xccl_smoke:
        print("Experimental inventory complete. XCCL smoke test was not requested.")
        return 0

    smoke_result = run_command(
        "experimental XCCL all-reduce",
        build_xccl_smoke_command(
            sys.executable,
            Path(__file__).resolve(),
            timeout_seconds=arguments.timeout_seconds,
        ),
        environment=experimental_environment,
        timeout_seconds=arguments.timeout_seconds,
    )
    print_result(smoke_result)
    return 0 if smoke_result.successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
