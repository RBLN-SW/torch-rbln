"""torch_rbln Python bindings for memory management.

This module provides functions for managing RBLN device memory, including
cache management, memory statistics, and memory monitoring capabilities.
"""

from typing import Dict, Optional, Union  # noqa: UP035

import torch

import torch_rbln._C


__all__ = [
    "empty_cache",
    "max_memory_allocated",
    "max_memory_reserved",
    "memory_allocated",
    "memory_reserved",
    "memory_stats",
    "reset_accumulated_memory_stats",
    "reset_peak_memory_stats",
]


def _normalize_device(device: Optional[Union[int, str, torch.device]]) -> torch.device:
    """
    Normalize device input to torch.device object.

    Args:
        device: Device specification (int, str, torch.device, or None)

    Returns:
        torch.device: Normalized device object
    """
    if device is None:
        return torch.device("rbln", torch_rbln._C.current_device())
    elif isinstance(device, int):
        return torch.device(f"rbln:{device}")
    elif isinstance(device, str):
        return torch.device(device)
    else:
        return device


def empty_cache(device: Optional[Union[int, str, torch.device]] = None) -> None:
    """
    Release all unoccupied cached memory currently held by the caching allocator.

    This function releases cached memory blocks that are not currently in use,
    allowing them to be used by other applications or returned to the system.

    Args:
        device (Optional[Union[int, str, torch.device]]): The device to empty cache for.
            If None, uses the current device. Defaults to None.
    """
    device = _normalize_device(device)
    torch_rbln._C.empty_cache(device)


def memory_stats(device: Optional[Union[int, str, torch.device]] = None) -> Dict[str, int]:
    """
    Return a dictionary of accelerator device memory allocator statistics.

    The returned dictionary contains various memory statistics including:
    - allocated_bytes: current, peak, allocated, freed
    - reserved_bytes: current, peak, allocated, freed
    - active_bytes: current, peak
    - cached_bytes: current, peak
    - num_alloc_retries: number of allocation retries
    - num_ooms: number of out-of-memory errors
    - num_device_alloc: number of device allocations
    - num_device_free: number of device frees

    Args:
        device (Optional[Union[int, str, torch.device]]): The device to query.
            If None, uses the current device. Defaults to None.

    Returns:
        Dict[str, int]: A dictionary containing memory statistics.
    """
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
    return memory_stats(device).get("allocated_bytes.all.current", 0)


def max_memory_allocated(device: Optional[Union[int, str, torch.device]] = None) -> int:
    """
    Return the current accelerator maximum device memory occupied by tensors in bytes.

    Args:
        device (Optional[Union[int, str, torch.device]]): The device to query.
            If None, uses the current device. Defaults to None.

    Returns:
        int: The maximum memory allocated in bytes.
    """
    return memory_stats(device).get("allocated_bytes.all.peak", 0)


def memory_reserved(device: Optional[Union[int, str, torch.device]] = None) -> int:
    """
    Return the current accelerator device memory managed by the caching allocator in bytes.

    Args:
        device (Optional[Union[int, str, torch.device]]): The device to query.
            If None, uses the current device. Defaults to None.

    Returns:
        int: The current memory reserved in bytes.
    """
    return memory_stats(device).get("reserved_bytes.all.current", 0)


def max_memory_reserved(device: Optional[Union[int, str, torch.device]] = None) -> int:
    """
    Return the current accelerator maximum device memory managed by the caching allocator in bytes.

    Args:
        device (Optional[Union[int, str, torch.device]]): The device to query.
            If None, uses the current device. Defaults to None.

    Returns:
        int: The maximum memory reserved in bytes.
    """
    return memory_stats(device).get("reserved_bytes.all.peak", 0)


def reset_accumulated_memory_stats(device: Optional[Union[int, str, torch.device]] = None) -> None:
    """
    Reset the "accumulated" (historical) stats tracked by the current accelerator memory allocator.

    This resets the accumulated counters for allocated, freed, and other historical statistics.

    Args:
        device (Optional[Union[int, str, torch.device]]): The device to reset stats for.
            If None, uses the current device. Defaults to None.
    """
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
    device = _normalize_device(device)
    torch_rbln._C.reset_peak_memory_stats(device)
