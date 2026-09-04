import platform
import socket
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from subprocess import CalledProcessError

import psutil
from anyio import run_process

from exo.shared.types.profiling import InterfaceType, NetworkInterfaceInfo

type PathResolver = Callable[[Path], Path]


def _resolve_existing_path(path: Path) -> Path:
    return path.resolve(strict=True)


def get_os_version() -> str:
    """Return the OS version string for this node.

    On macOS this is the macOS version (e.g. ``"15.3"``).
    On other platforms it falls back to the platform name (e.g. ``"Linux"``).
    """
    if sys.platform == "darwin":
        version = platform.mac_ver()[0]
        return version if version else "Unknown"
    return platform.system() or "Unknown"


async def get_os_build_version() -> str:
    """Return the macOS build version string (e.g. ``"24D5055b"``).

    On non-macOS platforms, returns ``"Unknown"``.
    """
    if sys.platform != "darwin":
        return "Unknown"

    try:
        process = await run_process(["sw_vers", "-buildVersion"])
    except CalledProcessError:
        return "Unknown"

    return process.stdout.decode("utf-8", errors="replace").strip() or "Unknown"


async def get_friendly_name() -> str:
    """
    Asynchronously gets the 'Computer Name' (friendly name) of a Mac.
    e.g., "John's MacBook Pro"
    Returns the name as a string, or None if an error occurs or not on macOS.
    """
    hostname = socket.gethostname()

    if sys.platform != "darwin":
        return hostname

    try:
        process = await run_process(["scutil", "--get", "ComputerName"])
    except CalledProcessError:
        return hostname

    return process.stdout.decode("utf-8", errors="replace").strip() or hostname


def _parse_networksetup_interface_types(output: str) -> dict[str, InterfaceType]:
    """Parse ``networksetup -listallhardwareports`` output."""
    types: dict[str, InterfaceType] = {}
    current_type: InterfaceType = "unknown"

    for line in output.splitlines():
        if line.startswith("Hardware Port:"):
            port_name = line.split(":", 1)[1].strip()
            if "Wi-Fi" in port_name:
                current_type = "wifi"
            elif "Ethernet" in port_name or "LAN" in port_name:
                current_type = "ethernet"
            elif port_name.startswith("Thunderbolt"):
                current_type = "thunderbolt"
            else:
                current_type = "unknown"
        elif line.startswith("Device:"):
            device = line.split(":", 1)[1].strip()
            # enX is ethernet adapters or thunderbolt - these must be deprioritised
            if device.startswith("en") and device not in ["en0", "en1"]:
                current_type = "maybe_ethernet"
            types[device] = current_type

    return types


async def _get_interface_types_from_networksetup() -> dict[str, InterfaceType]:
    """Get macOS interface types from ``networksetup``."""
    if sys.platform != "darwin":
        return {}

    try:
        result = await run_process(["networksetup", "-listallhardwareports"])
    except CalledProcessError:
        return {}

    return _parse_networksetup_interface_types(result.stdout.decode())


def _try_resolve(path: Path, resolve_path: PathResolver) -> Path | None:
    try:
        return resolve_path(path)
    except (OSError, RuntimeError):
        # Missing and cyclic sysfs links are not evidence of a device type.
        return None


def _is_thunderbolt_driver_path(path: Path) -> bool:
    return path.name.replace("-", "_") == "thunderbolt_net"


def _is_thunderbolt_bus_path(path: Path) -> bool:
    return len(path.parts) >= 2 and path.parts[-2:] == ("bus", "thunderbolt")


def _has_thunderbolt_bus_ancestor(
    device_path: Path,
    *,
    sysfs_root: Path,
    resolve_path: PathResolver,
) -> bool:
    resolved_device_path = _try_resolve(device_path, resolve_path)
    resolved_sysfs_root = _try_resolve(sysfs_root, resolve_path)
    if resolved_device_path is None or resolved_sysfs_root is None:
        return False

    try:
        relative_device_path = resolved_device_path.relative_to(resolved_sysfs_root)
    except ValueError:
        return False

    current_path = resolved_sysfs_root / relative_device_path
    while current_path != resolved_sysfs_root:
        subsystem_path = _try_resolve(current_path / "subsystem", resolve_path)
        if subsystem_path is not None and _is_thunderbolt_bus_path(subsystem_path):
            return True
        current_path = current_path.parent

    return False


def _classify_linux_interface(
    interface_name: str,
    *,
    sysfs_root: Path = Path("/sys"),
    resolve_path: PathResolver = _resolve_existing_path,
) -> InterfaceType:
    """Classify a Linux interface using kernel-exported sysfs evidence.

    Interface names are intentionally not evidence. This prevents an unrelated
    interface named ``thunderbolt0`` from receiving Thunderbolt placement
    priority.
    """
    if Path(interface_name).name != interface_name or interface_name in {".", ".."}:
        return "unknown"

    device_path = sysfs_root / "class" / "net" / interface_name / "device"
    for driver_path in (
        device_path / "driver",
        device_path / "driver" / "module",
    ):
        resolved_driver_path = _try_resolve(driver_path, resolve_path)
        if resolved_driver_path is not None and _is_thunderbolt_driver_path(
            resolved_driver_path
        ):
            return "thunderbolt"

    if _has_thunderbolt_bus_ancestor(
        device_path,
        sysfs_root=sysfs_root,
        resolve_path=resolve_path,
    ):
        return "thunderbolt"

    return "unknown"


def _get_interface_types_from_linux_sysfs(
    interface_names: Iterable[str],
    *,
    sysfs_root: Path = Path("/sys"),
    resolve_path: PathResolver = _resolve_existing_path,
) -> dict[str, InterfaceType]:
    """Classify Linux network interfaces from an injectable sysfs tree."""
    return {
        interface_name: _classify_linux_interface(
            interface_name,
            sysfs_root=sysfs_root,
            resolve_path=resolve_path,
        )
        for interface_name in interface_names
    }


async def _get_interface_types(
    interface_names: Iterable[str],
    *,
    sysfs_root: Path = Path("/sys"),
    resolve_path: PathResolver = _resolve_existing_path,
) -> dict[str, InterfaceType]:
    match sys.platform:
        case "darwin":
            return await _get_interface_types_from_networksetup()
        case "linux":
            return _get_interface_types_from_linux_sysfs(
                interface_names,
                sysfs_root=sysfs_root,
                resolve_path=resolve_path,
            )
        case _:
            return {}


async def get_network_interfaces() -> list[NetworkInterfaceInfo]:
    """
    Retrieves detailed network interface information.

    macOS types come from ``networksetup``. Linux Thunderbolt interfaces are
    identified from sysfs driver and bus evidence. Other Linux interface types
    remain unknown.
    Returns a list of NetworkInterfaceInfo objects.
    """
    interfaces_info: list[NetworkInterfaceInfo] = []
    network_addresses = psutil.net_if_addrs()
    interface_types = await _get_interface_types(network_addresses)

    for iface, services in network_addresses.items():
        for service in services:
            match service.family:
                case socket.AF_INET | socket.AF_INET6:
                    interfaces_info.append(
                        NetworkInterfaceInfo(
                            name=iface,
                            ip_address=service.address,
                            interface_type=interface_types.get(iface, "unknown"),
                        )
                    )
                case _:
                    pass

    return interfaces_info


async def get_model_and_chip() -> tuple[str, str]:
    """Get Mac system information using system_profiler."""
    model = "Unknown Model"
    chip = "Unknown Chip"

    # TODO: better non mac support
    if sys.platform != "darwin":
        return (model, chip)

    try:
        process = await run_process(
            [
                "system_profiler",
                "SPHardwareDataType",
            ]
        )
    except CalledProcessError:
        return (model, chip)

    # less interested in errors here because this value should be hard coded
    output = process.stdout.decode().strip()

    model_line = next(
        (line for line in output.split("\n") if "Model Name" in line), None
    )
    model = model_line.split(": ")[1] if model_line else "Unknown Model"

    chip_line = next((line for line in output.split("\n") if "Chip" in line), None)
    chip = chip_line.split(": ")[1] if chip_line else "Unknown Chip"

    return (model, chip)
