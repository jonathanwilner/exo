from __future__ import annotations

import json
from argparse import ArgumentTypeError
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import cast

import pytest

from scripts.exo_thunderbolt_netns import (
    NODE_A_NAMESPACE,
    NODE_B_NAMESPACE,
    THUNDERBOLT_INTERFACE,
    Command,
    CommandResult,
    HarnessError,
    NetemSettings,
    ThunderboltNetworkHarness,
    validated_latency,
    validated_loss,
    validated_rate,
)

TEST_TOKEN = "1" * 32


class RecordingRunner:
    def __init__(self, responses: Iterator[str] | None = None) -> None:
        self.commands: list[Command] = []
        self._responses = responses or iter(())

    def __call__(self, command: Command) -> CommandResult:
        self.commands.append(command)
        return CommandResult(next(self._responses, ""))


def harness(
    tmp_path: Path,
    runner: RecordingRunner,
    *,
    uid: int = 0,
    output: list[str] | None = None,
) -> ThunderboltNetworkHarness:
    messages = output if output is not None else []
    return ThunderboltNetworkHarness(
        runner=runner,
        state_path=tmp_path / "state.json",
        uid_provider=lambda: uid,
        token_provider=lambda: TEST_TOKEN,
        output=messages.append,
    )


def write_state(tmp_path: Path, *, token: str = TEST_TOKEN) -> None:
    (tmp_path / "state.json").write_text(
        json.dumps(
            {
                "version": 1,
                "owner_uid": 0,
                "ownership_token": token,
                "node_a_namespace": NODE_A_NAMESPACE,
                "node_b_namespace": NODE_B_NAMESPACE,
                "interface_name": THUNDERBOLT_INTERFACE,
                "node_a_address": "192.168.253.1/30",
                "node_b_address": "192.168.253.2/30",
            }
        ),
        encoding="utf-8",
    )


def ownership_json(side: str) -> str:
    return json.dumps(
        [
            {
                "ifname": THUNDERBOLT_INTERFACE,
                "ifalias": f"exo-thunderbolt-netns:{TEST_TOKEN}:{side}",
            }
        ]
    )


def test_start_builds_two_owned_nodes_and_symmetric_netem(tmp_path: Path) -> None:
    runner = RecordingRunner(iter([""]))
    target = harness(tmp_path, runner)

    target.start(
        NetemSettings(latency="5ms", loss="0.25%", rate="4gbit"),
        dry_run=False,
    )

    assert runner.commands[0] == ("ip", "netns", "list")
    assert ("ip", "netns", "add", NODE_A_NAMESPACE) in runner.commands
    assert ("ip", "netns", "add", NODE_B_NAMESPACE) in runner.commands
    assert (
        "ip",
        "link",
        "add",
        "exo-tb-a",
        "type",
        "veth",
        "peer",
        "name",
        "exo-tb-b",
    ) in runner.commands
    netem_commands = [command for command in runner.commands if "netem" in command]
    assert len(netem_commands) == 2
    assert all(
        command[-6:] == ("delay", "5ms", "loss", "0.25%", "rate", "4gbit")
        for command in netem_commands
    )
    assert (tmp_path / "state.json").stat().st_mode & 0o777 == 0o600


def test_start_dry_run_needs_no_root_and_writes_no_state(tmp_path: Path) -> None:
    runner = RecordingRunner()
    output: list[str] = []
    target = harness(tmp_path, runner, uid=1000, output=output)

    target.start(NetemSettings(), dry_run=True)

    assert runner.commands == []
    assert not (tmp_path / "state.json").exists()
    assert output[0] == f"ip netns add {NODE_A_NAMESPACE}"
    assert any("name thunderbolt0" in line for line in output)


def test_start_refuses_existing_namespace(tmp_path: Path) -> None:
    runner = RecordingRunner(iter([f"{NODE_A_NAMESPACE}\n"]))
    target = harness(tmp_path, runner)

    with pytest.raises(HarnessError, match="refusing to replace"):
        target.start(NetemSettings(), dry_run=False)

    assert runner.commands == [("ip", "netns", "list")]


def test_start_failure_rolls_back_only_namespaces_created_by_this_run(
    tmp_path: Path,
) -> None:
    class FailingRunner(RecordingRunner):
        def __call__(self, command: Command) -> CommandResult:
            self.commands.append(command)
            expected_veth_command = (
                "ip",
                "link",
                "add",
                "exo-tb-a",
                "type",
                "veth",
                "peer",
                "name",
                "exo-tb-b",
            )
            if command == expected_veth_command:
                raise RuntimeError("simulated veth failure")
            return CommandResult()

    runner = FailingRunner()
    target = harness(tmp_path, runner)

    with pytest.raises(RuntimeError, match="simulated veth failure"):
        target.start(NetemSettings(), dry_run=False)

    assert runner.commands[-2:] == [
        ("ip", "netns", "delete", NODE_B_NAMESPACE),
        ("ip", "netns", "delete", NODE_A_NAMESPACE),
    ]
    assert not (tmp_path / "state.json").exists()


def test_fail_verifies_both_aliases_before_link_changes(tmp_path: Path) -> None:
    write_state(tmp_path)
    runner = RecordingRunner(
        iter(
            [
                f"{NODE_A_NAMESPACE}\n{NODE_B_NAMESPACE}\n",
                ownership_json("a"),
                ownership_json("b"),
            ]
        )
    )
    target = harness(tmp_path, runner)

    target.fail(dry_run=False)

    assert runner.commands[-2:] == [
        (
            "ip",
            "-n",
            NODE_A_NAMESPACE,
            "link",
            "set",
            "dev",
            THUNDERBOLT_INTERFACE,
            "down",
        ),
        (
            "ip",
            "-n",
            NODE_B_NAMESPACE,
            "link",
            "set",
            "dev",
            THUNDERBOLT_INTERFACE,
            "down",
        ),
    ]


def test_stop_refuses_alias_mismatch_without_deleting_anything(tmp_path: Path) -> None:
    write_state(tmp_path)
    runner = RecordingRunner(
        iter(
            [
                f"{NODE_A_NAMESPACE}\n{NODE_B_NAMESPACE}\n",
                json.dumps([{"ifalias": "foreign-resource"}]),
            ]
        )
    )
    target = harness(tmp_path, runner)

    with pytest.raises(HarnessError, match="ownership marker mismatch"):
        target.stop(dry_run=False)

    assert not any(
        command[:3] == ("ip", "netns", "delete") for command in runner.commands
    )
    assert (tmp_path / "state.json").exists()


def test_stop_deletes_only_exact_owned_namespaces(tmp_path: Path) -> None:
    write_state(tmp_path)
    runner = RecordingRunner(
        iter(
            [
                f"{NODE_A_NAMESPACE}\n{NODE_B_NAMESPACE}\nexo-tb-node-other\n",
                ownership_json("a"),
                ownership_json("b"),
            ]
        )
    )
    target = harness(tmp_path, runner)

    target.stop(dry_run=False)

    delete_commands = [
        command
        for command in runner.commands
        if command[:3] == ("ip", "netns", "delete")
    ]
    assert delete_commands == [
        ("ip", "netns", "delete", NODE_A_NAMESPACE),
        ("ip", "netns", "delete", NODE_B_NAMESPACE),
    ]
    assert not (tmp_path / "state.json").exists()


def test_export_status_writes_sanitized_two_node_snapshot(tmp_path: Path) -> None:
    write_state(tmp_path)
    runner = RecordingRunner(
        iter(
            [
                f"{NODE_A_NAMESPACE}\n{NODE_B_NAMESPACE}\n",
                ownership_json("a"),
                json.dumps([{"operstate": "UP"}]),
                ownership_json("b"),
                json.dumps([{"operstate": "UP"}]),
            ]
        )
    )
    target = harness(tmp_path, runner)
    output_path = tmp_path / "public" / "status.json"

    target.export_status(output_path, dry_run=False)

    snapshot = cast(
        dict[str, object],
        cast(object, json.loads(output_path.read_text(encoding="utf-8"))),
    )
    nodes = cast(list[dict[str, object]], snapshot["nodes"])
    assert snapshot["active"] is True
    assert snapshot["link_up"] is True
    assert [node["namespace"] for node in nodes] == [
        NODE_A_NAMESPACE,
        NODE_B_NAMESPACE,
    ]
    assert all(node["interface_name"] == "thunderbolt0" for node in nodes)
    assert TEST_TOKEN not in output_path.read_text(encoding="utf-8")
    assert output_path.stat().st_mode & 0o777 == 0o644


def test_tampered_state_names_are_rejected_before_commands(tmp_path: Path) -> None:
    write_state(tmp_path)
    state_path = tmp_path / "state.json"
    state = cast(
        dict[str, object],
        cast(object, json.loads(state_path.read_text(encoding="utf-8"))),
    )
    state["node_a_namespace"] = "foreign"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    runner = RecordingRunner()
    target = harness(tmp_path, runner)

    with pytest.raises(HarnessError, match="unexpected resource names"):
        target.stop(dry_run=False)

    assert runner.commands == []


@pytest.mark.parametrize(
    ("validator", "valid", "invalid"),
    [
        (validated_latency, "2.5ms", "2 milliseconds"),
        (validated_loss, "99.5%", "101%"),
        (validated_rate, "40gbit", "40gbps"),
    ],
)
def test_impairment_validation(
    validator: Callable[[str], str], valid: str, invalid: str
) -> None:
    assert validator(valid) == valid
    with pytest.raises(ArgumentTypeError):
        validator(invalid)
