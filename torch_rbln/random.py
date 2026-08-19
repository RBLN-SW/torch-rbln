"""torch_rbln Python bindings for random number generation.

This module provides functions for managing random number generators on RBLN
devices and integrates with PyTorch's accelerator random-seeding interface.
"""

from typing import Union

import torch

import torch_rbln._C


__all__ = [
    "manual_seed",
    "manual_seed_all",
    "get_rng_state",
    "set_rng_state",
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


def _get_device_index(device: Union[int, str, torch.device]) -> int:
    """Normalize a device argument to an RBLN device index."""

    if isinstance(device, str):
        device = torch.device(device)
    if isinstance(device, torch.device):
        if device.type != "rbln":
            raise ValueError(f"Expected an RBLN device, but got: {device}")
        device = device.index if device.index is not None else -1
    if not isinstance(device, int):
        raise TypeError(f"Expected an int, str, or torch.device, but got: {type(device)}")
    if device == -1:
        device = torch_rbln._C.current_device()
    return device


def get_rng_state(device: Union[int, str, torch.device] = "rbln") -> torch.Tensor:
    """Return the RNG state of the specified RBLN device as a ByteTensor.

    Args:
        device (int, str, or torch.device, optional): The device whose RNG
            state to return. Default: ``"rbln"`` (the current RBLN device).
    """
    return torch_rbln._C.get_default_generator(_get_device_index(device)).get_state()


def set_rng_state(new_state: torch.Tensor, device: Union[int, str, torch.device] = "rbln") -> None:
    """Set the RNG state of the specified RBLN device.

    Args:
        new_state (torch.Tensor): The desired state.
        device (int, str, or torch.device, optional): The device whose RNG
            state to set. Default: ``"rbln"`` (the current RBLN device).
    """
    new_state_copy = new_state.clone(memory_format=torch.contiguous_format)
    torch_rbln._C.get_default_generator(_get_device_index(device)).set_state(new_state_copy)
