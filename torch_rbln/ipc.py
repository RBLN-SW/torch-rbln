"""Cross-process RBLN device memory, exposed as ``torch.rbln.ipc``.

A tensor from :func:`empty_exportable` can be shared with another process on the same
device: :func:`export_tensor` turns it into an :class:`IpcHandle`, and :func:`open_tensor`
in the other process maps the same memory back as a tensor. Callers only deal with the
logical tensor. How the runtime lays the memory out on device (sharding across chiplets,
dtype transforms) travels inside ``IpcHandle.payload`` and is rebuilt on open.

Both processes must be able to open the device's DRM accel node (``/dev/accel/accelN``).
"""

from typing import NamedTuple, Optional, Union

import torch

import torch_rbln._C
from torch_rbln.memory import _normalize_device


__all__ = [
    "IpcHandle",
    "empty_exportable",
    "export_tensor",
    "open_tensor",
]


class IpcHandle(NamedTuple):
    """Handle of an exported tensor (see :func:`export_tensor`).

    Everything but ``fds`` pickles as-is. ``fds`` are dma-buf file descriptors owned by
    the holder. Pass them to the other process with fd-passing
    (``multiprocessing.reduction.send_handle`` / ``SCM_RIGHTS``) and close them once sent.
    Their number is an implementation detail; pass them all, in order.
    """

    fds: list[int]
    payload: bytes
    nbytes: int
    shape: torch.Size
    dtype: torch.dtype
    device_index: int


def empty_exportable(
    shape,
    dtype: torch.dtype = torch.float32,
    device: Optional[Union[int, str, torch.device]] = None,
) -> torch.Tensor:
    """
    Allocate an uninitialized RBLN tensor that :func:`export_tensor` can export.

    Only tensors from this function can be exported. The runtime places the memory
    on device like any other tensor, for example in the layout a compiled model asks
    for on its first use. :func:`export_tensor` fixes that layout from then on.

    Args:
        shape: Shape of the tensor.
        dtype: dtype of the tensor.
        device: The rbln device to allocate on. Defaults to the current device.
    """
    dev = _normalize_device(device)
    return torch_rbln._C._empty_exportable(dev.index, list(shape), dtype)


def export_tensor(tensor: torch.Tensor) -> IpcHandle:
    """
    Export ``tensor`` so another process on the same device can open it.

    The tensor must come from :func:`empty_exportable` and cover its whole storage
    (contiguous, zero storage offset). If nothing has placed it on device yet, it is
    placed flat first. From the first export on, its device layout stays fixed:
    operations that would move or resize it raise instead. The exporter must keep
    ``tensor`` alive while any other process uses the handle.

    Args:
        tensor: An RBLN tensor from :func:`empty_exportable`.
    """
    if tensor.device.type != "rbln":
        raise RuntimeError(f"export_tensor: expected an rbln tensor, got {tensor.device}")
    nbytes, payload, fds = torch_rbln._C._export_device_memory(tensor)
    return IpcHandle(
        fds=list(fds),
        payload=payload,
        nbytes=nbytes,
        shape=tensor.shape,
        dtype=tensor.dtype,
        device_index=tensor.device.index,
    )


def open_tensor(
    handle: IpcHandle,
    fds: Optional[list[int]] = None,
    device: Optional[Union[int, str, torch.device]] = None,
) -> torch.Tensor:
    """
    Open a tensor exported by :func:`export_tensor` in another process.

    The returned tensor has the exporter's shape and dtype and owns the mapping:
    dropping its last reference releases it. ``fds`` are not consumed; close them
    after this call.

    Args:
        handle: The exporter's handle, typically unpickled.
        fds: The dma-buf fds as received in this process, in the order of
            ``handle.fds``. Defaults to ``handle.fds``, for a handle that never
            crossed a process boundary.
        device: The rbln device the memory lives on. Must be the same physical
            device as the exporter's. Defaults to the current device.
    """
    dev = _normalize_device(device)
    fds = handle.fds if fds is None else list(fds)
    if len(fds) != len(handle.fds):
        raise RuntimeError(f"open_tensor: the handle has {len(handle.fds)} fd(s) but {len(fds)} were given")
    return torch_rbln._C._import_device_memory(
        dev.index, handle.nbytes, handle.payload, fds, list(handle.shape), handle.dtype
    )
