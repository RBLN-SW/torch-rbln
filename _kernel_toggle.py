"""Shared ctypes helper for C-kernel bench scripts.

The C-kernel runtime toggle and timing counters live in the torch_rbln
Python extension (``torch_rbln/_C.cpython-<tag>.so``). This helper locates
that .so, opens it, and exposes ``set_enabled`` / ``get_enabled`` /
``read_counters`` for benches.

Timing counters are only populated when the library was built with
``-DTORCH_RBLN_C_KERNEL_TIMING=ON``; otherwise ``read_counters`` returns
zeros for every stage.
"""

from __future__ import annotations

import ctypes
import glob
from pathlib import Path

import torch_rbln

_so_glob = str(Path(torch_rbln.__file__).parent / "_C*.so")
_so_candidates = sorted(glob.glob(_so_glob))
if not _so_candidates:
    raise RuntimeError(f"could not locate torch_rbln _C extension at {_so_glob}")

_LIB = ctypes.CDLL(_so_candidates[0], mode=ctypes.RTLD_GLOBAL)
_LIB.c10_rbln_c_kernel_set_enabled.argtypes = [ctypes.c_int]
_LIB.c10_rbln_c_kernel_set_enabled.restype = None
_LIB.c10_rbln_c_kernel_get_enabled.argtypes = []
_LIB.c10_rbln_c_kernel_get_enabled.restype = ctypes.c_int
_LIB.c10_rbln_c_kernel_read_timing.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
_LIB.c10_rbln_c_kernel_read_timing.restype = None

# Order must match torch_rbln/csrc/rbln/kernels/RBLNKernelCache.cpp
#   :c10_rbln_c_kernel_read_timing.
STAGES = ("n_calls", "alloc", "build_maps", "prepare_in", "prepare_out", "run", "total")


def set_enabled(enabled: bool) -> None:
    _LIB.c10_rbln_c_kernel_set_enabled(1 if enabled else 0)


def get_enabled() -> bool:
    return bool(_LIB.c10_rbln_c_kernel_get_enabled())


def read_counters() -> dict[str, int]:
    """Return cumulative per-stage counters in nanoseconds and reset to 0."""
    buf = (ctypes.c_uint64 * len(STAGES))()
    _LIB.c10_rbln_c_kernel_read_timing(buf)
    return {name: int(buf[i]) for i, name in enumerate(STAGES)}
