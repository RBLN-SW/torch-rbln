"""torch_rbln Python bindings with device tensor for internal purpose.

This module provides functions for creating, manipulating, and printing tensors on the RBLN device.
It includes utilities to convert data types, manage tensor metadata, and handle memory contiguity.
"""

from collections.abc import Sequence

import torch

import torch_rbln._C


__all__ = [
    "_create_tensor_from_ptr",
]


def _create_tensor_from_ptr(data_ptr: int, shape: Sequence[int], dtype: torch.dtype) -> torch.Tensor:
    """
    Create a tensor from a device raw data pointer.

    Args:
        data_ptr: The raw data pointer.
        shape: A sequence of integers defining the shape of the tensor.
        dtype: The data type of the tensor elements.

    Returns:
        A PyTorch tensor created from the given data pointer, shape, and dtype.
    """
    return torch_rbln._C._create_tensor_from_ptr(data_ptr, shape, dtype)
