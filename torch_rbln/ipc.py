"""Cross-process RBLN device memory, exposed as ``torch.rbln.ipc``.

A tensor from :func:`empty_exportable` can be shared with another process on the same
device: :func:`export_tensor` turns it into an :class:`IpcHandle`, and
:func:`open_tensor` in the other process maps the same memory back as a tensor. Callers
only deal with the logical tensor. How the runtime lays the memory out on device
(sharding across chiplets, dtype transforms) travels inside ``IpcHandle.payload`` and is
rebuilt on open.

Experimental. Supported on RBLN-CR NPUs only, with a logical device that maps to a
single NPU (no RSD). Both processes must be able to open the device's DRM accel node
(``/dev/accel/accelN``), which needs RBLN driver 4.0 or later.
"""

import os
from pathlib import Path
from typing import NamedTuple

import torch

import torch_rbln._C
from torch_rbln._internal.device_arch_utils import _arch_from_npu_name
from torch_rbln.device.device import get_device_properties
from torch_rbln.memory import _normalize_device


__all__ = [
    "IpcHandle",
    "empty_exportable",
    "export_tensor",
    "open_tensor",
]


def _accel_node_accessible() -> bool:
    return any(os.access(p, os.R_OK | os.W_OK) for p in Path("/dev/accel").glob("accel*"))


def _check_supported(dev: torch.device) -> None:
    props = get_device_properties(dev)
    if _arch_from_npu_name(props.name) != "rebel":
        msg = f"torch.rbln.ipc supports RBLN-CR NPUs only, but {dev} is {props.name}"
        raise RuntimeError(msg)
    if props.npu_count != 1:
        msg = (
            f"torch.rbln.ipc does not support RSD: {dev} maps to {props.npu_count} NPUs. "
            "Use a logical device that maps to a single NPU"
        )
        raise RuntimeError(msg)
    if not _accel_node_accessible():
        msg = "torch.rbln.ipc needs RBLN driver 4.0 or later and read/write access to /dev/accel/accelN"
        raise RuntimeError(msg)


class IpcHandle(NamedTuple):
    """Picklable description of an exported tensor (see :func:`export_tensor`).

    The dma-buf fds that back it are returned next to it, not inside it: they only
    mean something in the process that holds them.
    """

    payload: bytes
    nbytes: int
    shape: torch.Size
    dtype: torch.dtype
    num_fds: int


def empty_exportable(
    shape,
    dtype: torch.dtype = torch.float32,
    device: int | str | torch.device | None = None,
) -> torch.Tensor:
    """Allocate an uninitialized RBLN tensor that :func:`export_tensor` can export.

    Only tensors from this function can be exported. The runtime places the memory
    on device like any other tensor, for example in the layout a compiled model asks
    for on its first use. :func:`export_tensor` fixes that layout from then on.

    Args:
        shape: Shape of the tensor.
        dtype: dtype of the tensor.
        device: The rbln device to allocate on. Defaults to the current device.
    """
    dev = _normalize_device(device)
    _check_supported(dev)
    return torch_rbln._C._empty_exportable(dev.index, list(shape), dtype)


def export_tensor(tensor: torch.Tensor) -> tuple[IpcHandle, list[int]]:
    """Export ``tensor`` so another process on the same device can open it.

    The tensor must come from :func:`empty_exportable` and cover its whole storage
    (contiguous, zero storage offset). If nothing has placed it on device yet, it is
    placed flat first. That default placement keeps the whole tensor in one contiguous
    area, so a tensor too large for it must be placed before its first export, for
    example by the compiled model that uses it. From the first export on, its device
    layout stays fixed: operations that would move or resize it raise instead. The
    exporter must keep ``tensor`` alive while any other process uses the handle.

    Returns ``(handle, fds)``. ``handle`` pickles. ``fds`` are dma-buf file
    descriptors owned by the caller: pass them to the other process with fd-passing
    (``multiprocessing.reduction.send_handle`` / ``SCM_RIGHTS``), in order, and close
    them once sent. Their number is an implementation detail.

    Args:
        tensor: An RBLN tensor from :func:`empty_exportable`.
    """
    nbytes, payload, fds = torch_rbln._C._export_device_memory(tensor)
    handle = IpcHandle(
        payload=payload,
        nbytes=nbytes,
        shape=tensor.shape,
        dtype=tensor.dtype,
        num_fds=len(fds),
    )
    return handle, list(fds)


def open_tensor(
    handle: IpcHandle,
    fds: list[int],
    device: int | str | torch.device | None = None,
) -> torch.Tensor:
    """Open a tensor exported by :func:`export_tensor` in another process.

    The returned tensor has the exporter's shape and dtype and owns the mapping:
    dropping its last reference releases it. ``fds`` are not consumed; close them
    after this call.

    Args:
        handle: The exporter's handle, typically unpickled.
        fds: The dma-buf fds as received in this process, in the exporter's order.
        device: The rbln device the memory lives on, as a logical device of this
            process. It must map to the exporter's physical NPUs. If omitted, the
            device that holds those NPUs is used, and opening raises when no device
            or several devices do.
    """
    fds = list(fds)
    if len(fds) != handle.num_fds:
        msg = f"open_tensor: the handle has {handle.num_fds} fd(s) but {len(fds)} were given"
        raise RuntimeError(msg)
    if device is None:
        dev = torch.device("rbln", torch_rbln._C._resolve_import_device(handle.payload))
    else:
        dev = _normalize_device(device)
    _check_supported(dev)
    return torch_rbln._C._import_device_memory(
        dev.index, handle.nbytes, handle.payload, fds, list(handle.shape), handle.dtype
    )
