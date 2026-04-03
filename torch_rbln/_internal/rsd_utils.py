"""torch_rbln utilities for RSD functionality.

This module provides utility functions related to RSD environment variables
and device mapping configuration for tensor parallelism.
"""

from typing import Optional

import torch_rbln._C
from torch_rbln.device import device_count


def get_physical_device_ids(device_id: int) -> Optional[list[int]]:
    """
    Get physical device IDs mapped to a logical device.

    This function queries the device topology to find the physical NPU IDs
    that are mapped to the given logical device index.

    Args:
        device_id: Logical device index.

    Returns:
        Optional[List[int]]: List of physical device IDs mapped to the logical device,
            or None if the device_id is invalid or not found.

    Examples:
        >>> get_physical_device_ids(0)
        [0]  # Default 1:1 mapping
        >>> get_physical_device_ids(0)
        [0, 1]  # With RBLN_NPUS_PER_DEVICE=2
        >>> get_physical_device_ids(0)
        [0, 4]  # With RBLN_DEVICE_MAP="[0,4],[1,5]"
    """
    topology = torch_rbln._C._get_device_topology()
    entry = next(
        (e for e in topology.entries if e.logical_device_index == device_id),
        None,
    )
    if entry is None:
        return None
    return entry.physical_device_ids


def auto_determine_tensor_parallel_size(device_id: Optional[int]) -> Optional[int]:
    """
    Automatically determine tensor_parallel_size based on RSD environment variables.

    This function reads RBLN_NPUS_PER_DEVICE or RBLN_DEVICE_MAP environment variables
    to determine the number of physical NPUs mapped to the given logical device.

    Args:
        device_id: Logical device index. If None, defaults to 0.

    Returns:
        Optional[int]: The number of physical NPUs mapped to the logical device,
            or None if unable to determine (e.g., no RBLN devices available,
            device_id is invalid, or environment variables are not set).

    Examples:
        - With RBLN_NPUS_PER_DEVICE=2: returns 2
        - With RBLN_DEVICE_MAP="[0,1],[2,3]" and device_id=0: returns 2
        - With RBLN_DEVICE_MAP="[0,1,2,3],[4,5]" and device_id=0: returns 4
        - With default mapping (1:1): returns 1

    Note:
        This function is used by torch.compile backend to automatically set
        tensor_parallel_size when not explicitly provided.
    """
    # Use device_id=0 as default if not provided
    target_device_id = device_id if device_id is not None else 0

    # Check if RBLN devices are available
    try:
        logical_device_count = device_count()
    except Exception:
        # device_count() may fail if the RBLN runtime is unavailable or misconfigured.
        return None

    if logical_device_count == 0:
        return None

    # Validate device_id is within range
    if target_device_id < 0 or target_device_id >= logical_device_count:
        return None

    # Get physical device IDs for the logical device
    physical_device_ids = get_physical_device_ids(target_device_id)
    if not physical_device_ids:
        return None
    return len(physical_device_ids)
