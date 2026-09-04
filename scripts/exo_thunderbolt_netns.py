#!/usr/bin/env python3
"""Create an isolated two-node network for Linux Thunderbolt simulation.

The harness deliberately uses fixed namespace and interface names.  A random
ownership token is recorded in a private state file and in each interface's
alias.  Destructive actions refuse to continue unless both records agree.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, cast

NODE_A_NAMESPACE = "exo-tb-node-a"
NODE_B_NAMESPACE = "exo-tb-node-b"
ROOT_A_INTERFACE = "exo-tb-a"
ROOT_B_INTERFACE = "exo-tb-b"
THUNDERBOLT_INTERFACE = "thunderbolt0"
NODE_A_ADDRESS = "192.168.253.1/30"
NODE_B_ADDRESS = "192.168.253.2/30"
STATE_VERSION = 1
STATUS_VERSION = 1
ALIAS_PREFIX = "exo-thunderbolt-netns"

LATENCY_PATTERN = re.compile(r"^(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:us|ms|s)$")
LOSS_PATTERN = re.compile(r"^(?:100(?:\.0+)?|[0-9]{1,2}(?:\.[0-9]+)?)%$")
RATE_PATTERN = re.compile(
    r"^(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:kbit|mbit|gbit|tbit)$",
    re.IGNORECASE,
)

Command = tuple[str, ...]


@dataclass(frozen=True)
class CommandResult:
    """Relevant output from one command invocation."""

    stdout: str = ""


class CommandRunner(Protocol):
    """Injectable command execution boundary used by rootless unit tests."""

    def __call__(self, command: Command) -> CommandResult: ...


@dataclass(frozen=True)
class HarnessState:
    """Resources owned by one harness invocation."""

    version: int
    owner_uid: int
    ownership_token: str
    node_a_namespace: str
    node_b_namespace: str
    interface_name: str
    node_a_address: str
    node_b_address: str


@dataclass(frozen=True)
class NetemSettings:
    """Optional impairment settings applied in both directions."""

    latency: str | None = None
    loss: str | None = None
    rate: str | None = None

    def arguments(self) -> Command:
        arguments: list[str] = []
        if self.latency is not None:
            arguments.extend(("delay", self.latency))
        if self.loss is not None:
            arguments.extend(("loss", self.loss))
        if self.rate is not None:
            arguments.extend(("rate", self.rate))
        return tuple(arguments)


class HarnessError(RuntimeError):
    """A safe, user-actionable harness failure."""


def _json_object_list(value: str) -> list[dict[str, object]]:
    decoded = cast(object, json.loads(value or "[]"))
    if not isinstance(decoded, list):
        raise HarnessError("expected a JSON array from ip")
    result: list[dict[str, object]] = []
    for item in cast(list[object], decoded):
        if not isinstance(item, dict):
            raise HarnessError("expected JSON objects from ip")
        result.append(cast(dict[str, object], item))
    return result


def _required_state_value(
    values: Mapping[str, object], name: str, value_type: type[int] | type[str]
) -> int | str:
    value = values.get(name)
    if not isinstance(value, value_type):
        raise TypeError(f"invalid state field: {name}")
    return value


def default_state_path() -> Path:
    """Return a per-user state path without relying on a broad shared name."""

    runtime_directory = os.environ.get("XDG_RUNTIME_DIR")
    if runtime_directory:
        return Path(runtime_directory) / "exo-thunderbolt-netns.json"
    return Path("/tmp") / f"exo-thunderbolt-netns-{os.getuid()}.json"


def subprocess_runner(command: Command) -> CommandResult:
    """Run one argv-only command without a shell."""

    completed = subprocess.run(  # noqa: S603 - argv is constructed internally.
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return CommandResult(stdout=completed.stdout)


class ThunderboltNetworkHarness:
    """Manage the two namespaces and their simulated Thunderbolt link."""

    def __init__(
        self,
        *,
        runner: CommandRunner = subprocess_runner,
        state_path: Path | None = None,
        uid_provider: Callable[[], int] = os.getuid,
        token_provider: Callable[[], str] = lambda: uuid.uuid4().hex,
        output: Callable[[str], None] = print,
    ) -> None:
        self._runner = runner
        self._state_path = state_path or default_state_path()
        self._uid_provider = uid_provider
        self._token_provider = token_provider
        self._output = output

    def start(self, settings: NetemSettings, *, dry_run: bool) -> None:
        """Create both nodes and configure their point-to-point link."""

        self._require_root_or_dry_run(dry_run)
        if not dry_run:
            if self._state_path.exists():
                raise HarnessError(
                    f"state file already exists: {self._state_path}; run status or stop"
                )
            existing_namespaces = self._namespace_names()
            collisions = existing_namespaces.intersection(self._expected_namespaces())
            if collisions:
                raise HarnessError(
                    "refusing to replace existing namespace(s): "
                    + ", ".join(sorted(collisions))
                )

        state = HarnessState(
            version=STATE_VERSION,
            owner_uid=self._uid_provider(),
            ownership_token=self._token_provider(),
            node_a_namespace=NODE_A_NAMESPACE,
            node_b_namespace=NODE_B_NAMESPACE,
            interface_name=THUNDERBOLT_INTERFACE,
            node_a_address=NODE_A_ADDRESS,
            node_b_address=NODE_B_ADDRESS,
        )
        commands = self._start_commands(state, settings)
        if dry_run:
            self._print_commands(commands)
            return

        created_namespaces: list[str] = []
        try:
            self._write_state(state)
            for command in commands:
                self._runner(command)
                if command[:3] == ("ip", "netns", "add"):
                    created_namespaces.append(command[3])
        except Exception:
            rollback_complete = True
            for namespace in reversed(created_namespaces):
                try:
                    self._runner(("ip", "netns", "delete", namespace))
                except Exception:
                    rollback_complete = False
            if rollback_complete:
                self._state_path.unlink(missing_ok=True)
            else:
                self._output(
                    "automatic rollback was incomplete; state file preserved for safe cleanup"
                )
            raise

        self._output(
            "started exo Thunderbolt simulation: "
            f"{NODE_A_NAMESPACE} {NODE_A_ADDRESS} <-> "
            f"{NODE_B_NAMESPACE} {NODE_B_ADDRESS}"
        )

    def status(self, *, dry_run: bool) -> None:
        """Report ownership, interface state, addresses, and impairments."""

        state = self._read_and_validate_state()
        commands = self._status_commands(state)
        if dry_run:
            self._print_commands(commands)
            return

        live_namespaces = self._namespace_names()
        report: dict[str, object] = {
            "state_file": str(self._state_path),
            "nodes": {},
        }
        nodes = cast(dict[str, object], report["nodes"])
        for namespace in self._expected_namespaces():
            if namespace not in live_namespaces:
                nodes[namespace] = {"present": False}
                continue
            self._verify_interface_ownership(state, namespace)
            address_result = self._runner(
                (
                    "ip",
                    "-n",
                    namespace,
                    "-json",
                    "address",
                    "show",
                    "dev",
                    state.interface_name,
                )
            )
            qdisc_result = self._runner(
                (
                    "ip",
                    "netns",
                    "exec",
                    namespace,
                    "tc",
                    "-json",
                    "qdisc",
                    "show",
                    "dev",
                    state.interface_name,
                )
            )
            nodes[namespace] = {
                "present": True,
                "addresses": json.loads(address_result.stdout or "[]"),
                "qdisc": json.loads(qdisc_result.stdout or "[]"),
            }
        self._output(json.dumps(report, indent=2, sort_keys=True))

    def export_status(self, output_path: Path, *, dry_run: bool) -> None:
        """Publish a sanitized status snapshot for an unprivileged demo app."""

        self._require_root_or_dry_run(dry_run)
        if dry_run:
            self._output(f"write sanitized simulation status to {output_path}")
            return

        state = self._read_and_validate_state()
        live_namespaces = self._namespace_names()
        nodes: list[dict[str, object]] = []
        link_up = True
        for namespace, address in (
            (state.node_a_namespace, state.node_a_address),
            (state.node_b_namespace, state.node_b_address),
        ):
            if namespace not in live_namespaces:
                link_up = False
                nodes.append(
                    {
                        "namespace": namespace,
                        "address": address,
                        "interface_name": state.interface_name,
                        "present": False,
                        "link_up": False,
                    }
                )
                continue

            self._verify_interface_ownership(state, namespace)
            result = self._runner(
                (
                    "ip",
                    "-n",
                    namespace,
                    "-details",
                    "-json",
                    "link",
                    "show",
                    "dev",
                    state.interface_name,
                )
            )
            links = _json_object_list(result.stdout)
            interface_up = len(links) == 1 and links[0].get("operstate") == "UP"
            link_up = link_up and interface_up
            nodes.append(
                {
                    "namespace": namespace,
                    "address": address,
                    "interface_name": state.interface_name,
                    "present": True,
                    "link_up": interface_up,
                }
            )

        snapshot = {
            "version": STATUS_VERSION,
            "mode": "network-namespace",
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "active": all(bool(node["present"]) for node in nodes),
            "link_up": link_up,
            "nodes": nodes,
        }
        self._write_public_snapshot(output_path, snapshot)
        self._output(f"exported sanitized simulation status to {output_path}")

    def fail(self, *, dry_run: bool) -> None:
        """Simulate a cable failure by taking down both owned interfaces."""

        self._set_link_state("down", dry_run=dry_run)

    def restore(self, *, dry_run: bool) -> None:
        """Restore both interfaces after a simulated cable failure."""

        self._set_link_state("up", dry_run=dry_run)

    def stop(self, *, dry_run: bool) -> None:
        """Delete only namespaces whose live ownership markers match state."""

        self._require_root_or_dry_run(dry_run)
        state = self._read_and_validate_state()
        live_namespaces = self._namespace_names()
        owned_namespaces: list[str] = []
        for namespace in self._expected_namespaces():
            if namespace in live_namespaces:
                if not dry_run:
                    self._verify_interface_ownership(state, namespace)
                owned_namespaces.append(namespace)

        commands = tuple(
            ("ip", "netns", "delete", namespace) for namespace in owned_namespaces
        )
        if dry_run:
            self._print_commands(commands)
            return
        for command in commands:
            self._runner(command)
        self._state_path.unlink()
        self._output("stopped exo Thunderbolt simulation")

    def _set_link_state(self, link_state: str, *, dry_run: bool) -> None:
        self._require_root_or_dry_run(dry_run)
        if link_state not in {"up", "down"}:
            raise HarnessError(f"invalid link state: {link_state}")
        state = self._read_and_validate_state()
        commands: list[Command] = []
        live_namespaces = self._namespace_names()
        for namespace in self._expected_namespaces():
            if namespace not in live_namespaces:
                raise HarnessError(f"expected namespace is missing: {namespace}")
            if not dry_run:
                self._verify_interface_ownership(state, namespace)
            commands.append(
                (
                    "ip",
                    "-n",
                    namespace,
                    "link",
                    "set",
                    "dev",
                    state.interface_name,
                    link_state,
                )
            )
        if dry_run:
            self._print_commands(tuple(commands))
            return
        for command in commands:
            self._runner(command)
        self._output(f"set simulated Thunderbolt link {link_state}")

    def _start_commands(
        self, state: HarnessState, settings: NetemSettings
    ) -> tuple[Command, ...]:
        alias_a = self._expected_alias(state, NODE_A_NAMESPACE)
        alias_b = self._expected_alias(state, NODE_B_NAMESPACE)
        commands: list[Command] = [
            ("ip", "netns", "add", NODE_A_NAMESPACE),
            ("ip", "netns", "add", NODE_B_NAMESPACE),
            (
                "ip",
                "link",
                "add",
                ROOT_A_INTERFACE,
                "type",
                "veth",
                "peer",
                "name",
                ROOT_B_INTERFACE,
            ),
            ("ip", "link", "set", ROOT_A_INTERFACE, "netns", NODE_A_NAMESPACE),
            ("ip", "link", "set", ROOT_B_INTERFACE, "netns", NODE_B_NAMESPACE),
        ]
        for namespace, root_interface, address, alias in (
            (NODE_A_NAMESPACE, ROOT_A_INTERFACE, NODE_A_ADDRESS, alias_a),
            (NODE_B_NAMESPACE, ROOT_B_INTERFACE, NODE_B_ADDRESS, alias_b),
        ):
            commands.extend(
                (
                    (
                        "ip",
                        "-n",
                        namespace,
                        "link",
                        "set",
                        root_interface,
                        "name",
                        THUNDERBOLT_INTERFACE,
                    ),
                    (
                        "ip",
                        "-n",
                        namespace,
                        "link",
                        "set",
                        "dev",
                        THUNDERBOLT_INTERFACE,
                        "alias",
                        alias,
                    ),
                    ("ip", "-n", namespace, "link", "set", "lo", "up"),
                    (
                        "ip",
                        "-n",
                        namespace,
                        "address",
                        "add",
                        address,
                        "dev",
                        THUNDERBOLT_INTERFACE,
                    ),
                    (
                        "ip",
                        "-n",
                        namespace,
                        "link",
                        "set",
                        "dev",
                        THUNDERBOLT_INTERFACE,
                        "up",
                    ),
                )
            )
            if settings.arguments():
                commands.append(
                    (
                        "ip",
                        "netns",
                        "exec",
                        namespace,
                        "tc",
                        "qdisc",
                        "replace",
                        "dev",
                        THUNDERBOLT_INTERFACE,
                        "root",
                        "netem",
                        *settings.arguments(),
                    )
                )
        return tuple(commands)

    def _status_commands(self, state: HarnessState) -> tuple[Command, ...]:
        commands: list[Command] = [("ip", "netns", "list")]
        for namespace in self._expected_namespaces():
            commands.extend(
                (
                    (
                        "ip",
                        "-n",
                        namespace,
                        "-details",
                        "-json",
                        "link",
                        "show",
                        "dev",
                        state.interface_name,
                    ),
                    (
                        "ip",
                        "-n",
                        namespace,
                        "-json",
                        "address",
                        "show",
                        "dev",
                        state.interface_name,
                    ),
                    (
                        "ip",
                        "netns",
                        "exec",
                        namespace,
                        "tc",
                        "-json",
                        "qdisc",
                        "show",
                        "dev",
                        state.interface_name,
                    ),
                )
            )
        return tuple(commands)

    def _namespace_names(self) -> set[str]:
        result = self._runner(("ip", "netns", "list"))
        return {
            line.split(maxsplit=1)[0]
            for line in result.stdout.splitlines()
            if line.strip()
        }

    def _verify_interface_ownership(self, state: HarnessState, namespace: str) -> None:
        if namespace not in self._expected_namespaces():
            raise HarnessError(f"refusing unexpected namespace: {namespace}")
        result = self._runner(
            (
                "ip",
                "-n",
                namespace,
                "-details",
                "-json",
                "link",
                "show",
                "dev",
                state.interface_name,
            )
        )
        links = _json_object_list(result.stdout)
        if len(links) != 1:
            raise HarnessError(
                f"cannot verify owned interface in namespace {namespace}"
            )
        link = links[0]
        expected_alias = self._expected_alias(state, namespace)
        if link.get("ifalias") != expected_alias:
            raise HarnessError(
                f"ownership marker mismatch in {namespace}; refusing destructive action"
            )

    def _read_and_validate_state(self) -> HarnessState:
        try:
            decoded_state = cast(
                object,
                json.loads(self._state_path.read_text(encoding="utf-8")),
            )
            if not isinstance(decoded_state, dict):
                raise TypeError("state must be a JSON object")
            raw_state = cast(dict[str, object], decoded_state)
            state = HarnessState(
                version=cast(int, _required_state_value(raw_state, "version", int)),
                owner_uid=cast(int, _required_state_value(raw_state, "owner_uid", int)),
                ownership_token=cast(
                    str, _required_state_value(raw_state, "ownership_token", str)
                ),
                node_a_namespace=cast(
                    str, _required_state_value(raw_state, "node_a_namespace", str)
                ),
                node_b_namespace=cast(
                    str, _required_state_value(raw_state, "node_b_namespace", str)
                ),
                interface_name=cast(
                    str, _required_state_value(raw_state, "interface_name", str)
                ),
                node_a_address=cast(
                    str, _required_state_value(raw_state, "node_a_address", str)
                ),
                node_b_address=cast(
                    str, _required_state_value(raw_state, "node_b_address", str)
                ),
            )
        except (FileNotFoundError, json.JSONDecodeError, TypeError) as error:
            raise HarnessError(
                f"cannot read valid state file: {self._state_path}"
            ) from error
        if state.version != STATE_VERSION:
            raise HarnessError(f"unsupported state version: {state.version}")
        if state.owner_uid != self._uid_provider():
            raise HarnessError("state belongs to a different user")
        if (
            state.node_a_namespace != NODE_A_NAMESPACE
            or state.node_b_namespace != NODE_B_NAMESPACE
            or state.interface_name != THUNDERBOLT_INTERFACE
            or state.node_a_address != NODE_A_ADDRESS
            or state.node_b_address != NODE_B_ADDRESS
        ):
            raise HarnessError(
                "state contains unexpected resource names; refusing action"
            )
        if not re.fullmatch(r"[0-9a-f]{32}", state.ownership_token):
            raise HarnessError("state contains an invalid ownership token")
        return state

    def _write_state(self, state: HarnessState) -> None:
        self._state_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor = os.open(
            self._state_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as state_file:
            json.dump(asdict(state), state_file, indent=2, sort_keys=True)
            state_file.write("\n")

    def _write_public_snapshot(
        self, output_path: Path, snapshot: Mapping[str, object]
    ) -> None:
        output_path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as output_file:
                os.fchmod(output_file.fileno(), 0o644)
                json.dump(snapshot, output_file, indent=2, sort_keys=True)
                output_file.write("\n")
            os.replace(temporary_path, output_path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

    def _expected_alias(self, state: HarnessState, namespace: str) -> str:
        side = "a" if namespace == NODE_A_NAMESPACE else "b"
        return f"{ALIAS_PREFIX}:{state.ownership_token}:{side}"

    def _expected_namespaces(self) -> tuple[str, str]:
        return (NODE_A_NAMESPACE, NODE_B_NAMESPACE)

    def _require_root_or_dry_run(self, dry_run: bool) -> None:
        if not dry_run and self._uid_provider() != 0:
            raise HarnessError("this action requires root; use sudo or --dry-run")

    def _print_commands(self, commands: Sequence[Command]) -> None:
        for command in commands:
            self._output(" ".join(command))


def validated_latency(value: str) -> str:
    if not LATENCY_PATTERN.fullmatch(value):
        raise argparse.ArgumentTypeError(
            "latency must use us, ms, or s, for example 5ms"
        )
    return value


def validated_loss(value: str) -> str:
    if not LOSS_PATTERN.fullmatch(value):
        raise argparse.ArgumentTypeError("loss must be a percentage from 0% to 100%")
    return value


def validated_rate(value: str) -> str:
    if not RATE_PATTERN.fullmatch(value):
        raise argparse.ArgumentTypeError(
            "rate must use kbit, mbit, gbit, or tbit, for example 4gbit"
        )
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage a safe two-node Thunderbolt network simulation for Exo."
    )
    parser.add_argument(
        "action",
        choices=("start", "status", "export-status", "fail", "restore", "stop"),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print commands without changing namespaces or state",
    )
    parser.add_argument("--latency", type=validated_latency)
    parser.add_argument("--loss", type=validated_loss)
    parser.add_argument("--rate", type=validated_rate)
    parser.add_argument(
        "--state-path",
        type=Path,
        default=default_state_path(),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        help="status snapshot destination for export-status",
    )
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = build_parser().parse_args(arguments)
    action = cast(str, parsed.action)
    dry_run = cast(bool, parsed.dry_run)
    state_path = cast(Path, parsed.state_path)
    output_path = cast(Path | None, parsed.output_path)
    latency = cast(str | None, parsed.latency)
    loss = cast(str | None, parsed.loss)
    rate = cast(str | None, parsed.rate)
    harness = ThunderboltNetworkHarness(state_path=state_path)
    settings = NetemSettings(
        latency=latency,
        loss=loss,
        rate=rate,
    )
    try:
        if action != "start" and settings.arguments():
            raise HarnessError("traffic impairment options are valid only with start")
        if action == "start":
            harness.start(settings, dry_run=dry_run)
        elif action == "status":
            harness.status(dry_run=dry_run)
        elif action == "export-status":
            if output_path is None:
                raise HarnessError("export-status requires --output-path")
            harness.export_status(output_path, dry_run=dry_run)
        elif action == "fail":
            harness.fail(dry_run=dry_run)
        elif action == "restore":
            harness.restore(dry_run=dry_run)
        else:
            harness.stop(dry_run=dry_run)
    except (HarnessError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
