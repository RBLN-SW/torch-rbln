"""torch_rbln Python bindings for memory management.

This module provides functions for managing RBLN device memory, including
cache management, memory statistics, and memory monitoring capabilities.
"""

import contextlib
import threading
from typing import Dict, Iterator, Optional, Union  # noqa: UP035

import torch

import torch_rbln._C


__all__ = [
    "empty_cache",
    "set_device_layout_like",
    "max_memory_allocated",
    "max_memory_reserved",
    "memory_allocated",
    "memory_reserved",
    "memory_stats",
    "offload",
    "reset_accumulated_memory_stats",
    "reset_peak_memory_stats",
]


def _normalize_device(device: Optional[Union[int, str, torch.device]]) -> torch.device:
    """
    Normalize a device argument to a concrete ``rbln`` ``torch.device``.

    ``None``, ``"rbln"``, and ``torch.device("rbln")`` resolve to the current
    device. A bare int is treated as an ``rbln`` index. Non-``rbln`` devices
    (``"cpu"``, ``"cuda:0"``, …) and out-of-range indices are rejected.
    """
    if device is None:
        return torch.device("rbln", torch_rbln._C.current_device())
    if isinstance(device, bool):  # bool is an int subclass; reject to avoid rbln:0/1 surprises
        raise TypeError(f"device must be None, int, str, or torch.device, not bool ({device!r})")

    # Resolve the index as a Python int *before* building a torch.device. Its index field
    # is an int8_t, so torch.device("rbln", 256) silently wraps to rbln:0 and would slip
    # past the range check below. A torch.device passed in has already wrapped and cannot
    # be recovered, so the raw int/str forms are validated from the original value here.
    if isinstance(device, int):
        dtype, index = "rbln", device
    elif isinstance(device, str):
        dtype, _, suffix = device.partition(":")
        try:
            index = int(suffix) if suffix else None
        except ValueError:
            raise ValueError(f"Invalid rbln device string {device!r}") from None
    elif isinstance(device, torch.device):
        dtype, index = device.type, device.index
    else:
        raise TypeError(f"device must be None, int, str, or torch.device, got {type(device).__name__}")

    if dtype != "rbln":
        raise ValueError(f"Expected an 'rbln' device, got '{dtype}' (from {device!r})")
    if index is None:
        index = torch_rbln._C.current_device()

    # Only range-check when devices exist; on a no-device host callers no-op.
    count = torch_rbln._C.device_count()
    if count > 0 and not (0 <= index < count):
        raise ValueError(f"rbln device index {index} is out of range [0, {count})")
    return torch.device("rbln", index)


def set_device_layout_like(target: torch.Tensor, ref: torch.Tensor) -> None:
    """Configure ``target``'s device-allocation layout to match ``ref`` (no copy).

    Both must be RBLN tensors with the same dtype, on the same device, and each a
    *whole base allocation* — not a view/slice.  ``ref`` must be device-resident.
    ``target`` adopts ``ref``'s layout and dtype while keeping its own size; no
    data is transferred.  A subsequent device-to-device copy between ``target``
    and ``ref`` then stays on the fast path.

    Typical use: make a host→device staging buffer match a KV cache's layout so
    the bulk upload and the per-slot device-to-device scatter are both fast.
    """
    torch_rbln._C._set_device_layout_like(target, ref)


def _no_rbln_device() -> bool:
    """No RBLN device available; memory-management ops no-op when True (torch.cuda parity)."""
    return torch_rbln._C.device_count() == 0


def empty_cache(device: Optional[Union[int, str, torch.device]] = None) -> None:
    """
    Release all unoccupied cached memory currently held by the caching allocator.

    This function releases cached memory blocks that are not currently in use,
    allowing them to be used by other applications or returned to the system.

    Unlike the generic ``torch.accelerator.empty_cache()`` (which drains only the
    caching allocator), this also drops the WarmCache and view-recipe caches, so it
    releases everything the caller is not still holding.

    Args:
        device (Optional[Union[int, str, torch.device]]): The device to empty cache for.
            If None, uses the current device. Defaults to None.
    """
    device = _normalize_device(device)
    # WarmCache holds strong refs to DynamoRuntime instances and the rbln
    # runtime buffers behind them. empty_cache() means "let go of everything
    # the user isn't holding"; if we kept warm entries the freed bytes would
    # show up unchanged in memory_stats. Clearing first puts us in the same
    # state as the cold dispatch path — entries get re-installed naturally.
    torch_rbln._C._warmcache_clear()
    # The view-recipe cache holds only metadata-derived recipes (no device
    # buffers), so it never shows up in memory_stats — but it has no eviction
    # and grows once per distinct view geometry, so clear it here too to keep
    # "let go of everything the user isn't holding" complete and to bound it
    # under variable-shape workloads. Lazy import avoids a load-time cycle
    # (ops_utils imports broadly).
    from torch_rbln._internal.ops_utils import view_recipe_cache_reset

    view_recipe_cache_reset()
    # No NPU: host caches above still dropped, but skip the device-side flush.
    if _no_rbln_device():
        return
    torch_rbln._C.empty_cache(device)


def memory_stats(device: Optional[Union[int, str, torch.device]] = None) -> Dict[str, int]:
    """
    Return a dictionary of accelerator device memory allocator statistics.

    The returned dictionary contains various memory statistics. Keys are the
    RBLN allocator's dotted names, e.g.:
    - allocated.current, allocated.peak, allocated.total_allocated, allocated.total_freed
    - reserved.current, reserved.peak, reserved.total_allocated, reserved.total_freed
    - active.current, active.peak

    Args:
        device (Optional[Union[int, str, torch.device]]): The device to query.
            If None, uses the current device. Defaults to None.

    Returns:
        Dict[str, int]: A dictionary containing memory statistics.
    """
    # No NPU: no allocator -> empty stats (memory_allocated() etc. then read 0).
    if _no_rbln_device():
        return {}
    device = _normalize_device(device)
    return torch_rbln._C.memory_stats(device)


def memory_allocated(device: Optional[Union[int, str, torch.device]] = None) -> int:
    """
    Return the current accelerator device memory occupied by tensors in bytes.

    Args:
        device (Optional[Union[int, str, torch.device]]): The device to query.
            If None, uses the current device. Defaults to None.

    Returns:
        int: The current memory allocated in bytes.
    """
    return memory_stats(device).get("allocated.current", 0)


def max_memory_allocated(device: Optional[Union[int, str, torch.device]] = None) -> int:
    """
    Return the current accelerator maximum device memory occupied by tensors in bytes.

    Args:
        device (Optional[Union[int, str, torch.device]]): The device to query.
            If None, uses the current device. Defaults to None.

    Returns:
        int: The maximum memory allocated in bytes.
    """
    return memory_stats(device).get("allocated.peak", 0)


def memory_reserved(device: Optional[Union[int, str, torch.device]] = None) -> int:
    """
    Return the current accelerator device memory managed by the caching allocator in bytes.

    Args:
        device (Optional[Union[int, str, torch.device]]): The device to query.
            If None, uses the current device. Defaults to None.

    Returns:
        int: The current memory reserved in bytes.
    """
    return memory_stats(device).get("reserved.current", 0)


def max_memory_reserved(device: Optional[Union[int, str, torch.device]] = None) -> int:
    """
    Return the current accelerator maximum device memory managed by the caching allocator in bytes.

    Args:
        device (Optional[Union[int, str, torch.device]]): The device to query.
            If None, uses the current device. Defaults to None.

    Returns:
        int: The maximum memory reserved in bytes.
    """
    return memory_stats(device).get("reserved.peak", 0)


def reset_accumulated_memory_stats(device: Optional[Union[int, str, torch.device]] = None) -> None:
    """
    Reset the "accumulated" (historical) stats tracked by the current accelerator memory allocator.

    This resets the accumulated counters for allocated, freed, and other historical statistics.

    Args:
        device (Optional[Union[int, str, torch.device]]): The device to reset stats for.
            If None, uses the current device. Defaults to None.
    """
    if _no_rbln_device():
        return
    device = _normalize_device(device)
    torch_rbln._C.reset_accumulated_memory_stats(device)


def reset_peak_memory_stats(device: Optional[Union[int, str, torch.device]] = None) -> None:
    """
    Reset the "peak" stats tracked by the current accelerator memory allocator.

    This resets the peak memory usage counters to their current values.

    Args:
        device (Optional[Union[int, str, torch.device]]): The device to reset stats for.
            If None, uses the current device. Defaults to None.
    """
    if _no_rbln_device():
        return
    device = _normalize_device(device)
    torch_rbln._C.reset_peak_memory_stats(device)


_offload_lock = threading.Lock()
_offload_depth = 0


@contextlib.contextmanager
def offload() -> Iterator[None]:
    """
    Context manager that enables RBLN file offloading for its scope.

    Inside the ``with`` block, the process-wide file offloading switch is on, so
    host-side regions backing RBLN tensors allocated within the block may be paged
    out to disk. Use this around code paths that allocate large host-resident
    tensors (for example, KV-cache initialization) where host RAM pressure
    matters.

    Nested ``offload`` blocks are tracked via a thread-safe depth counter; the
    switch is flipped back off only when the outermost context exits.

    Example::

        with torch.rbln.offload():
            tensor = torch.zeros(1 << 30, device="rbln:0")  # offloaded
    """
    global _offload_depth
    with _offload_lock:
        _offload_depth += 1
        if _offload_depth == 1:
            torch_rbln._C._set_file_offloading_enabled(True)
    try:
        yield
    finally:
        with _offload_lock:
            _offload_depth -= 1
            if _offload_depth == 0:
                torch_rbln._C._set_file_offloading_enabled(False)
