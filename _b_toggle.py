"""Shared ctypes helper for Option B bench scripts.

After the Phase 2 restructure the C ABI toggle + timing counters live in
``torch_rbln/_C.cpython-<tag>.so`` (not ``libc10_rbln.so``). This helper
locates and opens the right .so and exposes set_b / read_counters to the
bench.
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
_LIB.c10_rbln_set_b_enabled.argtypes = [ctypes.c_int]
_LIB.c10_rbln_set_b_enabled.restype = None
_LIB.c10_rbln_get_b_enabled.argtypes = []
_LIB.c10_rbln_get_b_enabled.restype = ctypes.c_int
_LIB.c10_rbln_hp_read_and_reset.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
_LIB.c10_rbln_hp_read_and_reset.restype = None

# Order must match c10/rbln/impl/RBLNKernelCache.cpp:c10_rbln_hp_read_and_reset.
STAGES = ("n_calls", "alloc", "build_maps", "prepare_in", "prepare_out", "run", "total")


def set_b(enabled: bool) -> None:
    _LIB.c10_rbln_set_b_enabled(1 if enabled else 0)


def get_b() -> bool:
    return bool(_LIB.c10_rbln_get_b_enabled())


def read_counters() -> dict[str, int]:
    """Return cumulative per-stage counters in nanoseconds and reset to 0.

    When the library was built without ``-DTORCH_RBLN_B_TIMING=ON`` the
    counters don't exist and this returns zeros for every stage.
    """
    buf = (ctypes.c_uint64 * len(STAGES))()
    _LIB.c10_rbln_hp_read_and_reset(buf)
    return {name: int(buf[i]) for i, name in enumerate(STAGES)}
