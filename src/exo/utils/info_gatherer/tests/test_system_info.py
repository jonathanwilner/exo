import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from exo.utils.info_gatherer.system_info import (
    _classify_linux_interface,  # pyright: ignore[reportPrivateUsage]
    _get_interface_types,  # pyright: ignore[reportPrivateUsage]
    _get_interface_types_from_linux_sysfs,  # pyright: ignore[reportPrivateUsage]
    _parse_networksetup_interface_types,  # pyright: ignore[reportPrivateUsage]
)

type InterfaceFixture = Callable[[str], Path]


@pytest.fixture
def sysfs_root(tmp_path: Path) -> Path:
    root = tmp_path / "sys"
    (root / "class" / "net").mkdir(parents=True)
    (root / "devices").mkdir()
    (root / "bus" / "thunderbolt").mkdir(parents=True)
    return root


@pytest.fixture
def interface_fixture(sysfs_root: Path) -> InterfaceFixture:
    def create_interface(interface_name: str) -> Path:
        device = sysfs_root / "devices" / interface_name
        device.mkdir()
        interface = sysfs_root / "class" / "net" / interface_name
        interface.mkdir()
        (interface / "device").symlink_to(device)
        return device

    return create_interface


def test_linux_interface_requires_sysfs_evidence(sysfs_root: Path) -> None:
    assert _classify_linux_interface("thunderbolt0", sysfs_root=sysfs_root) == "unknown"


def test_linux_interface_rejects_unsafe_name_before_resolving(
    sysfs_root: Path,
) -> None:
    def fail_if_called(_path: Path) -> Path:
        pytest.fail("unsafe interface name must not reach the filesystem")

    assert (
        _classify_linux_interface(
            "../thunderbolt0",
            sysfs_root=sysfs_root,
            resolve_path=fail_if_called,
        )
        == "unknown"
    )


@pytest.mark.parametrize("driver_name", ["thunderbolt-net", "thunderbolt_net"])
def test_linux_interface_detects_thunderbolt_net_driver(
    sysfs_root: Path,
    interface_fixture: InterfaceFixture,
    driver_name: str,
) -> None:
    device = interface_fixture("host-link")
    driver = sysfs_root / "bus" / "thunderbolt" / "drivers" / driver_name
    driver.mkdir(parents=True)
    (device / "driver").symlink_to(driver)

    assert (
        _classify_linux_interface("host-link", sysfs_root=sysfs_root) == "thunderbolt"
    )


def test_linux_interface_detects_thunderbolt_net_module(
    sysfs_root: Path,
    interface_fixture: InterfaceFixture,
) -> None:
    device = interface_fixture("host-link")
    driver = sysfs_root / "bus" / "platform" / "drivers" / "network-service"
    module = sysfs_root / "module" / "thunderbolt_net"
    driver.mkdir(parents=True)
    module.mkdir(parents=True)
    (device / "driver").symlink_to(driver)
    (driver / "module").symlink_to(module)

    assert (
        _classify_linux_interface("host-link", sysfs_root=sysfs_root) == "thunderbolt"
    )


def test_linux_interface_detects_thunderbolt_bus_ancestor(
    sysfs_root: Path,
) -> None:
    thunderbolt_device = sysfs_root / "devices" / "pci0" / "domain0" / "0-1"
    thunderbolt_device.mkdir(parents=True)
    (thunderbolt_device.parent / "subsystem").symlink_to(
        sysfs_root / "bus" / "thunderbolt"
    )
    interface = sysfs_root / "class" / "net" / "fabric0"
    interface.mkdir()
    (interface / "device").symlink_to(thunderbolt_device)

    assert _classify_linux_interface("fabric0", sysfs_root=sysfs_root) == "thunderbolt"


def test_linux_interface_ignores_unrelated_bus(
    sysfs_root: Path,
    interface_fixture: InterfaceFixture,
) -> None:
    device = interface_fixture("ethernet0")
    pci_bus = sysfs_root / "bus" / "pci"
    pci_bus.mkdir()
    (device / "subsystem").symlink_to(pci_bus)

    assert _classify_linux_interface("ethernet0", sysfs_root=sysfs_root) == "unknown"


def test_linux_interface_ignores_broken_sysfs_links(
    sysfs_root: Path,
    interface_fixture: InterfaceFixture,
) -> None:
    device = interface_fixture("broken0")
    (device / "driver").symlink_to(sysfs_root / "missing" / "thunderbolt-net")

    assert _classify_linux_interface("broken0", sysfs_root=sysfs_root) == "unknown"


def test_linux_interface_uses_injected_path_resolver(
    sysfs_root: Path,
    interface_fixture: InterfaceFixture,
) -> None:
    interface_fixture("ethernet0")
    resolved_paths: list[Path] = []

    def recording_resolver(path: Path) -> Path:
        resolved_paths.append(path)
        return path.resolve(strict=True)

    assert (
        _classify_linux_interface(
            "ethernet0",
            sysfs_root=sysfs_root,
            resolve_path=recording_resolver,
        )
        == "unknown"
    )
    assert sysfs_root in resolved_paths
    assert sysfs_root / "class" / "net" / "ethernet0" / "device" in resolved_paths


def test_linux_sysfs_classification_is_exhaustive(
    sysfs_root: Path,
    interface_fixture: InterfaceFixture,
) -> None:
    thunderbolt_device = interface_fixture("link0")
    driver = sysfs_root / "bus" / "thunderbolt" / "drivers" / "thunderbolt-net"
    driver.mkdir(parents=True)
    (thunderbolt_device / "driver").symlink_to(driver)
    interface_fixture("ethernet0")

    assert _get_interface_types_from_linux_sysfs(
        (name for name in ["link0", "ethernet0", "missing0"]),
        sysfs_root=sysfs_root,
    ) == {
        "link0": "thunderbolt",
        "ethernet0": "unknown",
        "missing0": "unknown",
    }


@pytest.mark.anyio
async def test_platform_dispatcher_uses_linux_sysfs(
    monkeypatch: pytest.MonkeyPatch,
    sysfs_root: Path,
    interface_fixture: InterfaceFixture,
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    device = interface_fixture("link0")
    driver = sysfs_root / "bus" / "thunderbolt" / "drivers" / "thunderbolt-net"
    driver.mkdir(parents=True)
    (device / "driver").symlink_to(driver)

    assert await _get_interface_types(["link0"], sysfs_root=sysfs_root) == {
        "link0": "thunderbolt"
    }


def test_networksetup_parser_preserves_macos_classification() -> None:
    output = """Hardware Port: Wi-Fi
Device: en0

Hardware Port: USB 10/100/1000 LAN
Device: en7

Hardware Port: Thunderbolt Bridge
Device: bridge0

Hardware Port: Other
Device: other0
"""

    assert _parse_networksetup_interface_types(output) == {
        "en0": "wifi",
        "en7": "maybe_ethernet",
        "bridge0": "thunderbolt",
        "other0": "unknown",
    }
