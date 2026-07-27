"""torch_rbln Python bindings for random number generation.

This module provides functions for managing random number generators on RBLN
devices and integrates with PyTorch's accelerator random-seeding interface.
"""

import torch_rbln._C


__all__ = [
    "manual_seed",
    "manual_seed_all",
]


def manual_seed(seed: int) -> None:
    """Set the seed for the current RBLN device.

    Args:
        seed (int): The seed to set.
    """
    torch_rbln._C.get_default_generator(torch_rbln._C.current_device()).manual_seed(int(seed))


def manual_seed_all(seed: int) -> None:
    """Set the seed for all RBLN devices.

    Args:
        seed (int): The seed to set.
    """
    for device_index in range(torch_rbln._C.device_count()):
        torch_rbln._C.get_default_generator(device_index).manual_seed(int(seed))
