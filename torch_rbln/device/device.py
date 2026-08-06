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
from torch_rbln._internal.ops_utils import SupportedDtypes


__all__ = [
    "current_device",
    "device_count",
    "physical_device_count",
    "is_available",
    "is_dummy_device",
    "is_initialized",
    "get_amp_supported_dtype",
    "set_device",
    "synchronize",
    "device",
    "device_of",
    "device_summary",
]


# Whether this process has selected an RBLN device (torch.cuda-style lazy-init
# flag). DeviceMesh reads it to decide whether to auto-select a per-rank device.
_initialized: bool = False


def current_device() -> int:
    """
    Get the index of the currently selected RBLN device.

    Raises:
        RuntimeError: if no RBLN device is available (mirrors
            ``torch.cuda.current_device()`` on a host with no accelerator), or with the
            detailed ``RBLN_*`` configuration error when the mapping is malformed.

    Returns:
        int: The index of the currently selected RBLN device.
    """
    # Point of use: raise in full detail here, since :func:`device_count` is a quiet
    # probe (``c10::cuda::device_count_ensure_non_zero()`` plays the same role).
    torch_rbln._C.device_count_ensure_non_zero()
    return torch_rbln._C.current_device()


def device_count() -> int:
    """Number of RBLN logical devices. Never raises (``torch.cuda`` parity).

    Returns ``0`` when the runtime is absent, no NPU is visible, or the ``RBLN_*``
    configuration is malformed; the malformed case also warns once. torch treats
    enumeration as infallible -- ``ATen/DeviceAccelerator.h`` says ``deviceCount()``
    "is *REQUIRED* to not raise any exception", and ``c10/cuda/CUDAFunctions.h`` keeps
    ``device_count() noexcept`` with a separate throwing
    ``device_count_ensure_non_zero()``.

    The detailed configuration error is not lost: it is raised by
    :func:`current_device`, :func:`set_device` and by any actual device allocation.

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

    Note: this queries the runtime directly, which resolves and therefore *seals*
    ``RBLN_DEVICES`` -- a later change to that variable is rejected, in this process and
    in any it forks. Prefer :func:`device_count` / :func:`is_available` in code that runs
    before a launcher has assigned devices; see ``TORCH_RBLN_AVAILABILITY_PROBE`` in
    docs/CONFIGURATION.md.

    Returns:
        int: The number of physical RBLN devices.
    """
    return torch_rbln._C.physical_device_count()


def is_available() -> bool:
    """Whether RBLN is usable as an accelerator (``torch.cuda.is_available()`` parity).

    **Never raises.** ``False`` when the runtime is absent or torn down, no device is
    present, or the ``RBLN_*`` configuration is malformed. ``torch.xpu.is_available()``
    documents "This function never throws" and ``torch.cuda.is_available()`` "never
    throws and returns 0 if the driver is missing or can't be initialized".

    This matters beyond RBLN code: torch evaluates it while importing
    ``torch.testing._internal.common_utils`` (``TEST_PRIVATEUSE1``), inside
    ``torch._utils._get_available_device_type()``, and in ``DataLoader(pin_memory=True)``
    via ``torch.accelerator.is_available()``. A raise there breaks callers that never
    asked for an NPU.

    Bound to the same C++ predicate as ``RBLNHooksInterface::hasRBLN()``, so the Python
    and C++ answers cannot diverge.
    """
    return torch_rbln._C.is_available()


def is_dummy_device() -> bool:
    """Whether ``RBLN_DUMMY_DEVICE`` (host-backed, no NPU) mode is active.

    ``is_available()`` is ``True`` in both dummy and real modes, so use this to
    tell a compile-only dummy device apart from real NPU availability.
    """
    return torch_rbln._C.is_dummy_device()


def is_initialized() -> bool:
    """True once :func:`set_device` has run, or when there are no devices at all.

    The no-device case is deliberate (not a bug): it makes ``torch.distributed``'s
    ``init_device_mesh`` skip its ``get_rank() % device_count()`` auto-select,
    which would otherwise fail on a host with no NPU.

    Note: this calls :func:`device_count`, which *plans* the RBLN device mapping from the
    current ``RBLN_*`` environment. It does not claim an NPU or freeze the mapping -- both
    happen on first device use -- so a launcher may still assign ``RBLN_DEVICES``
    afterwards.
    """
    return _initialized or device_count() == 0


def get_amp_supported_dtype() -> List[torch.dtype]:
    """
    Get a list of data types supported by automatic mixed precision (AMP) on RBLN devices.

    Returns:
        List[torch.dtype]: A list of data types supported by AMP.
    """
    return list(SupportedDtypes.amp)


def synchronize(device: Union[int, torch.device, str, None] = None) -> None:
    """Wait for all pending async transfers on the given RBLN device.

    If no device is specified, the current device is used.

    Args:
        device (torch.device or int or str, optional): The device to synchronize.
            Defaults to the current device.

    Example::
        >>> import torch
        >>> cpu_tensor = rbln_tensor.to("cpu", non_blocking=True)
        >>> torch.rbln.synchronize()  # wait for the transfer to complete
    """
    if device is None:
        device_idx = current_device()
    else:
        device_idx = _get_device_index(device)
    torch_rbln._C.synchronize(device_idx)


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
    global _initialized
    device_idx = _get_device_index(device, optional=True)
    if device_idx >= 0:
        # torch.cuda parity: selecting a device needs hardware. Point of use, so the
        # detailed RBLN_* configuration error surfaces here rather than a bare "0".
        torch_rbln._C.device_count_ensure_non_zero()
        torch_rbln._C.set_device(device_idx)
        _initialized = True


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
    # torch.cuda parity: selecting a device needs hardware (see :func:`set_device`).
    torch_rbln._C.device_count_ensure_non_zero()
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
