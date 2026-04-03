"""torch_rbln Python bindings related with device.

This package offers a lightweight, Pythonic interface for working with the
RBLN device backend.  Rather than focusing on individual kernels, it bundles
utility functions along with a thin wrapper over the compiled `_C`
extension, so you can query and manage RBLN hardware just as naturally as
you would in native PyTorch.
"""

from typing import Any, List, Union  # noqa: UP035

import torch

import torch_rbln._C


__all__ = [
    "current_device",
    "device_count",
    "physical_device_count",
    "is_available",
    "get_amp_supported_dtype",
    "set_device",
    "device",
    "device_of",
    "device_summary",
]


def current_device() -> int:
    """
    Get the index of the currently selected RBLN device.

    Returns:
        int: The index of the currently selected RBLN device.
    """
    return torch_rbln._C.current_device()


def device_count() -> int:
    """
    Get the number of available RBLN devices.

    Returns:
        int: The number of available RBLN devices.
    """
    return torch_rbln._C.device_count()


def physical_device_count() -> int:
    """
    Get the number of physical RBLN devices in the system.

    This function returns the actual number of physical devices, regardless of
    whether RSD mode is enabled. Unlike device_count(),
    this function always returns the physical device count, even when RSD mode
    is active (which makes device_count() return 1).

    Returns:
        int: The number of physical RBLN devices.
    """
    return torch_rbln._C.physical_device_count()


def is_available() -> bool:
    """
    Check if any RBLN devices are available.

    Returns:
        bool: True if at least one RBLN device is available, False otherwise.
    """
    return device_count() > 0


def get_amp_supported_dtype() -> List[torch.dtype]:
    """
    Get a list of data types supported by automatic mixed precision (AMP) on RBLN devices.

    Returns:
        List[torch.dtype]: A list of data types supported by AMP.

    Note:
        This function currently returns only `torch.float16`. It may need review to include other processable data types.
    """
    return [torch.float16]  # TODO: Needs review regarding processable dtypes


def set_device(device: Union[int, torch.device, str]) -> None:
    r"""Set the current device.

    Usage of this function is discouraged in favor of :any:`device`. In most
    cases it's better to use device context managers.

    Args:
        device (torch.device or int or str): selected device. This function is a no-op
            if this argument is negative.

    Example::
        >>> import torch
        >>> torch.rbln.set_device(0)  # Set device 0 as current
        >>> torch.rbln.set_device(torch.device("rbln:1"))  # Set device 1 as current
    """
    device_idx = _get_device_index(device, optional=True)
    if device_idx >= 0:
        torch_rbln._C.set_device(device_idx)


def _get_device_index(device: Any, optional: bool = False) -> int:
    """
    Helper function to extract device index from various device representations.

    Args:
        device: Can be an int, torch.device, or None (if optional=True)
        optional: If True, allows None and returns -1

    Returns:
        int: Device index, or -1 if optional and device is None
    """
    if device is None:
        if optional:
            return -1
        raise ValueError("device argument must be specified")
    if isinstance(device, int):
        return device
    if isinstance(device, torch.device):
        if device.type != "rbln":
            raise ValueError(f"Expected rbln device, but got {device.type}")
        if device.index is None:
            return current_device()
        return device.index
    if isinstance(device, str):
        dev = torch.device(device)
        if dev.type != "rbln":
            raise ValueError(f"Expected rbln device, but got {dev.type}")
        if dev.index is None:
            return current_device()
        return dev.index
    raise TypeError(f"Invalid device type: {type(device)}")


def _exchange_device(device: Union[int, torch.device]) -> int:
    """
    Exchange the current device and return the previous device index.

    Args:
        device: Device index or torch.device to set as current

    Returns:
        int: The previous device index
    """
    device_idx = _get_device_index(device)
    if device_idx < 0:
        return -1
    prev_device_idx = torch_rbln._C._exchange_device(device_idx)
    return prev_device_idx


def _maybe_exchange_device(device: int) -> int:
    """
    Exchange the current device if device >= 0, otherwise return -1.

    Args:
        device: Device index to set as current, or -1 for no-op

    Returns:
        int: The previous device index, or -1 if device < 0
    """
    if device < 0:
        return -1
    return _exchange_device(device)


class device:
    r"""Context-manager that changes the selected device.

    Args:
        device (torch.device or int): device index to select. It's a no-op if
            this argument is a negative integer or ``None``.

    Example::
        >>> import torch
        >>> with torch.rbln.device(0):
        ...     x = torch.randn(2, 2, device='rbln:0')
    """

    def __init__(self, device: Any):
        self.idx = _get_device_index(device, optional=True)
        self.prev_idx = -1

    def __enter__(self):
        self.prev_idx = _exchange_device(self.idx)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):
        _maybe_exchange_device(self.prev_idx)
        return False


class device_of(device):
    r"""Context-manager that changes the current device to that of given object.

    You can use both tensors and storages as arguments. If a given object is
    not allocated on an RBLN device, this is a no-op.

    Args:
        obj (Tensor or Storage): object allocated on the selected device.

    Example::
        >>> import torch
        >>> x = torch.randn(2, 2, device='rbln:1')
        >>> with torch.rbln.device_of(x):
        ...     # Current device is automatically set to device 1
        ...     y = torch.randn(2, 2, device='rbln:1')
    """

    def __init__(self, obj: Any):
        if isinstance(obj, torch.Tensor):
            if obj.device.type == "rbln":
                idx = obj.device.index if obj.device.index is not None else current_device()
            else:
                idx = -1  # Not an RBLN device, no-op
        elif hasattr(obj, "device") and hasattr(obj.device, "type"):
            # Handle Storage-like objects
            if obj.device.type == "rbln":
                idx = obj.device.index if obj.device.index is not None else current_device()
            else:
                idx = -1
        else:
            raise TypeError(f"Expected Tensor or Storage, but got {type(obj)}")
        super().__init__(idx)


_device_summary_debug_done = False


def _on_device_mapping_ready_from_cpp() -> None:
    """Hook from C++ on first ``DeviceMappingManager::getInstance()`` (e.g. first rbln alloc / device_count)."""
    global _device_summary_debug_done
    from torch_rbln._internal.log_utils import rbln_is_debug_enabled, rbln_log_debug

    # _device_summary_debug_done: belt-and-suspenders if this were ever invoked twice from Python.
    # not rbln_is_debug_enabled(): avoid building the table when DEBUG is off (see rbln_is_debug_enabled).
    if _device_summary_debug_done or not rbln_is_debug_enabled():
        return
    _device_summary_debug_done = True
    try:
        rbln_log_debug("Device configuration complete:\n" + _device_summary_string())
    except Exception:
        pass


def _device_summary_string() -> str:
    topology = torch_rbln._C._get_device_topology()
    rows = []
    max_logical_width = len("Logical Device")
    max_physical_width = len("Physical NPU IDs")
    max_status_width = len("Active (Aggregated)")
    for entry in topology.entries:
        physical_str = "[ " + ", ".join(str(pid) for pid in entry.physical_device_ids) + " ]"
        status = "Active (Aggregated)" if entry.is_aggregated else "Active"
        logical_device = f"rbln:{entry.logical_device_index}"
        rows.append((logical_device, physical_str, status))
        max_logical_width = max(max_logical_width, len(logical_device))
        max_physical_width = max(max_physical_width, len(physical_str))
        max_status_width = max(max_status_width, len(status))
    if topology.unused_physical_device_ids:
        unused_str = "[ " + ", ".join(str(pid) for pid in topology.unused_physical_device_ids) + " ]"
        rows.append(("-", unused_str, "Unused"))
        max_physical_width = max(max_physical_width, len(unused_str))
    max_logical_width = max(max_logical_width, len("Logical Device"))
    max_physical_width = max(max_physical_width, len("Physical NPU IDs"))
    max_status_width = max(max_status_width, len("Status"))
    header_sep = (
        "+"
        + "-" * (max_logical_width + 2)
        + "+"
        + "-" * (max_physical_width + 2)
        + "+"
        + "-" * (max_status_width + 2)
        + "+"
    )
    header_row = (
        f"| {'Logical Device':<{max_logical_width}} | "
        f"{'Physical NPU IDs':<{max_physical_width}} | "
        f"{'Status':<{max_status_width}} |"
    )
    lines = [
        "[RBLN] Device Topology Initialized:",
        header_sep,
        header_row,
        header_sep,
    ]
    for logical_device, physical_ids, status in rows:
        row = (
            f"| {logical_device:<{max_logical_width}} | "
            f"{physical_ids:<{max_physical_width}} | "
            f"{status:<{max_status_width}} |"
        )
        lines.append(row)
    lines.append(header_sep)
    if topology.unused_physical_device_ids:
        nu = len(topology.unused_physical_device_ids)
        lines.append(f"[Warning] {nu} physical NPU(s) are unused due to grouping constraints.")
    return "\n".join(lines)


def device_summary() -> None:
    """
    Print a summary of the RBLN device topology showing the mapping between
    logical devices and physical NPU IDs.

    This function displays a table showing:
    - Logical device indices (e.g., rbln:0, rbln:1)
    - Physical NPU IDs mapped to each logical device
    - Status of each device (Active, Active (Aggregated), or Unused)
    - Warnings about unused physical devices if any

    The device mapping is determined by the following environment variables
    (in order of priority):
    1. RBLN_DEVICE_MAP: Explicit mapping (e.g., "[0,1],[2,3,4,5]")
    2. RBLN_NPUS_PER_DEVICE: Group devices by count (e.g., "2")

    Example::
        >>> import torch
        >>> torch.rbln.device_summary()
        [RBLN] Device Topology Initialized:
        +-------------------+-------------------+----------------------+
        | Logical Device    | Physical NPU IDs  | Status               |
        +-------------------+-------------------+----------------------+
        | rbln:0            | [ 0, 1 ]          | Active (Aggregated)  |
        | rbln:1            | [ 2, 3 ]          | Active (Aggregated)  |
        +-------------------+-------------------+----------------------+
    """
    print(_device_summary_string())
