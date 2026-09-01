"""torch_rbln Python bindings for memory management.

This module provides functions for managing RBLN device memory, including
cache management, memory statistics, and memory monitoring capabilities.
"""

import contextlib
import operator
import os
import sys
import threading
from typing import Dict, Iterator, Optional, Union  # noqa: UP035

import torch

import torch_rbln._C


__all__ = [
    "bind_device_memory",
    "empty_cache",
    "huge_host_empty",
    "set_device_layout_like",
    "max_memory_allocated",
    "max_memory_reserved",
    "memory_allocated",
    "memory_reserved",
    "memory_stats",
    "memory_stats_per_chiplet",
    "memory_summary",
    "offload",
    "release_offload_temp_storage",
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

    # torch.device's index field is an int8_t, so torch.device("rbln", 256) silently wraps
    # to rbln:0 and would slip past the range check below. Read the index from the raw int
    # or the original string suffix rather than a constructed device's .index. (A torch.device
    # passed in has already wrapped and cannot be recovered, so it is validated as-is.)
    if isinstance(device, int):
        dtype, index = "rbln", device
    elif isinstance(device, str):
        # torch.device validates the canonical grammar (rejects "rbln:", "rbln:00",
        # "rbln: 0", "rbln:0_0", … that int() would silently accept); the index still
        # comes from the raw suffix so "rbln:256" is range-checked, not wrap-normalized.
        try:
            dtype = torch.device(device).type
        except RuntimeError:
            raise ValueError(f"Invalid device string {device!r}") from None
        _, sep, suffix = device.partition(":")
        index = int(suffix) if sep else None
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

    Both must be RBLN tensors with the same dtype, on the same device, and each
    covering its whole storage.  ``ref`` must be device-resident.
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


def _physical_npu_ids(device: torch.device) -> str:
    """Physical NPU ids behind a logical device, for the memory_summary() scope line."""
    try:
        topology = torch_rbln._C._get_device_topology()
        for entry in topology.entries:
            if entry.logical_device_index == device.index:
                return "[" + ", ".join(str(pid) for pid in entry.physical_device_ids) + "]"
    except Exception:
        pass
    return "[unknown]"


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
    # NB: the SDPA attn-weights cache is deliberately NOT flushed here. A pending backward's
    # entry is still live and needed, so a global flush would silently downgrade a
    # forward -> empty_cache() -> backward sequence to a CPU recompute. Inference never caches
    # (the forward caches only when a backward may run), so there is no inference leak to
    # reclaim; a grad-forward-with-no-backward entry lingers only until the next forward that
    # reuses its output address overwrites or discards it -- see kernels/sdpa.py.
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

    Scope, same as ``torch.cuda.memory_stats``: the caching allocator of the context
    **this process** holds on ``device``. Weights and other direct device allocations
    are not counted, and a second process on the same NPU is invisible here -- these
    numbers are not the NPU's occupancy. For that, use ``rbln-smi``.

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


def memory_stats_per_chiplet(
    device: Optional[Union[int, str, torch.device]] = None,
) -> Dict[str, int]:
    """
    Return memory allocator statistics broken down per chiplet.

    Same keys as :func:`memory_stats`, each prefixed with ``npu.<n>.chiplet.<c>.`` --
    e.g. ``npu.0.chiplet.0.allocated.current``. A device runs out on its heaviest
    chiplet, which the aggregate :func:`memory_stats` hides.

    ``npu.<n>`` is the n-th physical NPU of this logical device (``RBLN_NPUS_PER_DEVICE``
    / ``RBLN_DEVICE_MAP``), not a physical NPU id; a 1:1 mapping yields ``npu.0`` only.
    Same scope as :func:`memory_stats`.

    Args:
        device (Optional[Union[int, str, torch.device]]): The device to query.
            If None, uses the current device. Defaults to None.

    Returns:
        Dict[str, int]: A dictionary containing per-chiplet memory statistics.
    """
    if _no_rbln_device():
        return {}
    device = _normalize_device(device)
    return torch_rbln._C.memory_stats_per_chiplet(device)


def memory_summary(device: Optional[Union[int, str, torch.device]] = None) -> str:
    """
    Return a human-readable printout of the current memory allocator statistics.

    Rows are per (NPU, chiplet), so an imbalance is visible at a glance. The scope line
    under the title names what the numbers cover: they are this process's allocator, not
    the NPU's occupancy (see :func:`memory_stats`).

    Peaks are tracked per chiplet, so the peak columns of the ``total`` row bound the
    joint peak from above rather than reporting it. :func:`memory_stats` carries the
    exact joint peak.

    Args:
        device (Optional[Union[int, str, torch.device]]): The device to query.
            If None, uses the current device. Defaults to None.

    Returns:
        str: The formatted table, or a short notice when no stats are available.
    """
    per_chiplet = memory_stats_per_chiplet(device)
    if not per_chiplet:
        return "torch_rbln memory summary: no statistics (no RBLN device or allocator uninitialized)\n"

    rows = sorted({(int(k.split(".")[1]), int(k.split(".")[3])) for k in per_chiplet})
    columns = [
        ("allocated.current", "allocated"),
        ("allocated.peak", "alloc peak"),
        ("reserved.current", "reserved"),
        ("reserved.peak", "resv peak"),
        ("active.current", "active"),
        ("cached.current", "cached"),
    ]

    def mib(value: int) -> str:
        return f"{value / 1024**2:.1f}"

    def stat(npu: int, chiplet: int, key: str) -> int:
        return per_chiplet.get(f"npu.{npu}.chiplet.{chiplet}.{key}", 0)

    device = _normalize_device(device)
    header = f"{'npu':>5}{'chiplet':>9}" + "".join(f"{label:>13}" for _, label in columns)
    lines = [
        f"torch_rbln memory summary (device={device}, MiB)",
        f"scope: pid {os.getpid()}, physical NPU {_physical_npu_ids(device)} "
        "-- caching allocator only, this process only",
        header,
        "-" * len(header),
    ]
    for npu, chiplet in rows:
        cells = "".join(f"{mib(stat(npu, chiplet, key)):>13}" for key, _ in columns)
        lines.append(f"{npu:>5}{chiplet:>9}" + cells)
    totals = "".join(f"{mib(sum(stat(npu, chiplet, key) for npu, chiplet in rows)):>13}" for key, _ in columns)
    lines.append("-" * len(header))
    lines.append(f"{'total':>14}" + totals)

    retries = sum(stat(npu, chiplet, "num_alloc_retries") for npu, chiplet in rows)
    ooms = sum(stat(npu, chiplet, "num_ooms") for npu, chiplet in rows)
    lines.append(f"alloc retries: {retries}   ooms: {ooms}")
    return "\n".join(lines) + "\n"


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


def bind_device_memory(tensor: torch.Tensor) -> None:
    """
    Materialize ``tensor``'s device allocation instead of leaving it lazy.

    A device allocation reserves a virtual address and materializes the physical
    memory behind it on first use through a torch op. A consumer that reads those
    physical buffers *out of band* -- a collective library, direct storage (NVMe)
    DMA -- never runs such an op, so it would find nothing there. Call this after
    allocating a buffer you are handing to one.

    The region is laid out flat and 1:1 with no dtype transform, on the device's
    main node -- what a consumer treating it as bytes expects. Use
    :func:`set_device_layout_like` instead when the layout has to match another
    tensor's. Binding an already-bound region is allowed.

    Args:
        tensor: An RBLN tensor covering its whole storage: contiguous, with zero
            storage offset. A view that still spans the whole storage is fine; a
            slice or an interior view is not, since its address is not the
            allocation's.

    Raises:
        RuntimeError: if ``tensor`` is not an RBLN tensor, does not cover its
            whole storage, or the runtime rejects the allocation.

    Example::

        staging = torch.empty(nbytes, dtype=torch.uint8, device="rbln:0")
        torch.rbln.bind_device_memory(staging)
    """
    torch_rbln._C._bind_device_memory(tensor)


def huge_host_empty(nbytes: int) -> torch.Tensor:
    """
    Allocate a host buffer the device can DMA into without a staging copy.

    An ordinary CPU tensor is 64 B aligned; the runtime's copy path stages any
    host address that is not page aligned through a bounce buffer, and even an
    aligned one pays a page fault per 4 KiB the first time the runtime resolves
    its host addresses -- which lands in the transfer, not in setup. This returns
    2 MiB-aligned memory instead, prefaulted, so neither happens. It also asks
    for transparent huge pages, but that part is best effort: with THP disabled
    the alignment still holds and nothing is huge-page backed.

    The buffer is released when the last reference to the returned tensor (or to
    a view of it) goes away.

    Args:
        nbytes: Size of the buffer in bytes. Must be in ``1..sys.maxsize``.

    Returns:
        torch.Tensor: A zero-filled 1-D ``uint8`` CPU tensor of ``nbytes`` bytes,
        sharing the buffer's memory rather than copying it.

    Raises:
        TypeError: if ``nbytes`` is not an integer.
        ValueError: if ``nbytes`` is outside ``1..sys.maxsize``.
        MemoryError: if the allocation fails.

    Example::

        slab = torch.rbln.huge_host_empty(1 << 30)
        slab.view(...)  # hand to whatever consumes the host side
    """
    # Third Party
    from rebel.host_memory import HugeHostBuffer

    # Bound the request to what a size_t can carry. Past that the provider's
    # round-up to the alignment wraps to zero, which allocates nothing and then
    # prefaults the original size over it -- a segfault instead of an error.
    # Reported upstream; this is the range check this API owes its callers either
    # way. `operator.index` rather than `int()`, so a float is a TypeError rather
    # than a silent truncation.
    nbytes = operator.index(nbytes)
    if not 0 < nbytes <= sys.maxsize:
        raise ValueError(f"nbytes must be in 1..{sys.maxsize}, but got {nbytes}")

    # frombuffer takes a buffer-protocol reference, which is what keeps the
    # allocation alive for as long as the tensor is: HugeHostBuffer frees itself
    # on collection and nothing else holds it.
    return torch.frombuffer(HugeHostBuffer(nbytes), dtype=torch.uint8)


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


def release_offload_temp_storage() -> int:
    """
    Remove this process's file offloading temp files and directories.

    :func:`offload` writes into a per-process directory under ``RBLN_OFFLOAD_DIR`` (default
    ``$HOME/.cache/rbln_cache/offload``) that the runtime removes on teardown. Call this on a
    shutdown path that may be killed first. Offloaded tensors must not be used afterwards.

    Returns:
        int: The number of temp files removed.
    """
    return torch_rbln._C._release_offload_temp_storage()
