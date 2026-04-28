"""Bootstrap helpers for the C++ warm-runtime cache.

Lifecycle
---------
On a shim op's miss path (C++ shim didn't find a matching entry in the
warm cache), the C++ shim:
  1. Saves a thread-local "pending" `CacheKey` built from the live args.
  2. Calls the generated Python wrapper (e.g. ``add_out_rbln``) via pybind.

The Python wrapper's ``else`` branch (device path) now passes an empty
``_runtime_holder`` list via the compile ``options``. After the first
backend compile, rebel's ``rbln_backend`` appends the
``DynamoRuntime`` that owns the compiled ``PyRblnSyncRuntime`` to this
list. The generated wrapper then calls :func:`install_pending` with
that runtime and the post-compile output-tensor profiles.

:func:`install_pending` extracts the raw C++ pointer out of the
``PyRblnSyncRuntime`` pybind instance (see :func:`_raw_cpp_ptr`), packs
the output profiles into the shape the C++ side expects, and hands
everything to ``torch_rbln._C._warmcache_install_pending``. The C++
side matches against the pending key it saved on the way in and
inserts a :class:`CacheEntry` keyed by (op name, input profile,
scalars, device index).

Subsequent dispatches of the same op with a matching input profile
hit the warm cache on the C++ side and drive rebel's
``PyRblnSyncRuntime.{PrepareInputs, PrepareOutputs, Run}`` directly —
no pybind hop, no Python wrapper, no Dynamo recompile check.

Raw pointer extraction
----------------------
rebel's ``_C`` module is built against pybind11 2.x while torch
(and therefore torch-rbln) uses pybind11 3.x. Their internal type
registries are disjoint, so ``py::cast<PyRblnSyncRuntime*>(handle)``
fails across DSOs. We bypass the type caster by reading the C++
instance pointer directly from the pybind instance layout: for a
simple-layout pybind instance with a standard holder (``unique_ptr``
or ``shared_ptr``), the first ``void*`` after ``PyObject_HEAD`` is the
C++ instance pointer. On CPython 3.12 x86_64 that's at offset 16.

This is verified at runtime by calling a method on the recovered
pointer via ctypes (``PyRblnSyncRuntime::Run`` is a virtual that must
succeed on any valid instance). Mismatched layouts raise at install
time, not at first dispatch, so the cache stays empty instead of
issuing a bad pointer to rebel.
"""

from __future__ import annotations

import ctypes
from typing import Any

import torch

import torch_rbln._C as _C


# CPython 3.12 x86_64: PyObject_HEAD is (ob_refcnt: uint64, ob_type: PyTypeObject*).
_PYOBJECT_HEAD_SIZE = 16


def _raw_cpp_ptr(pybound_instance: Any) -> int:
    """Return the raw C++ pointer held by a pybind11 instance as ``uintptr_t``.

    Relies on pybind11's "simple layout" for single-inheritance types with a
    standard holder. Verified against rebel's ``PyRblnSyncRuntime``.
    """
    addr = id(pybound_instance)
    raw = ctypes.c_void_p.from_address(addr + _PYOBJECT_HEAD_SIZE).value
    if raw is None:
        raise RuntimeError(
            "warm_cache: unexpected null C++ pointer extracted from pybind instance; layout assumption may be wrong"
        )
    return int(raw)


_DTYPE_KEY = {
    torch.float16: "float16",
    torch.float32: "float32",
    torch.bfloat16: "bfloat16",
    torch.int64: "int64",
    torch.int32: "int32",
    torch.int16: "int16",
    torch.int8: "int8",
    torch.uint8: "uint8",
    torch.bool: "bool",
}


def install_pending(runtime_holder: list, outputs: Any) -> bool:
    # Hot codegen-injected path: skip work whenever WarmCache is disabled.
    # `_warmcache_is_enabled` is one C call; cheaper than the rest of this
    # function and makes WC OFF nearly free on the cold path.
    if not runtime_holder or not _C._warmcache_is_enabled():
        if runtime_holder:
            runtime_holder.clear()
        return False

    dyn_runtime = runtime_holder[-1]
    runtime_handle = dyn_runtime._runtime_handle
    raw_ptr = ctypes.c_void_p.from_address(id(runtime_handle) + _PYOBJECT_HEAD_SIZE).value
    if raw_ptr is None:
        runtime_holder.clear()
        return False

    if isinstance(outputs, torch.Tensor):
        outputs = (outputs,)
    profiles = []
    for t in outputs:
        if not isinstance(t, torch.Tensor):
            continue
        dt = _DTYPE_KEY.get(t.dtype)
        if dt is None:
            runtime_holder.clear()
            return False
        profiles.append((list(t.shape), dt, t.device.type == "rbln"))
    if not profiles:
        runtime_holder.clear()
        return False

    num_inputs = getattr(dyn_runtime, "_num_inputs", 0)
    num_outputs = getattr(dyn_runtime, "_num_outputs", len(profiles))

    ok = _C._warmcache_install_pending(
        dyn_runtime=dyn_runtime,
        runtime_raw_ptr=raw_ptr,
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        out_profiles=profiles,
    )
    # Drop the harvested DynamoRuntime so subsequent compile invocations on the
    # same compiled callable don't grow the list unboundedly.
    runtime_holder.clear()
    return bool(ok)


# ---------------------------------------------------------------------------
# Toggles / introspection (thin wrappers for tests and benchmarks).
# ---------------------------------------------------------------------------


def set_enabled(enabled: bool) -> None:
    """Globally enable/disable the warm-cache hot path."""
    _C._warmcache_set_enabled(bool(enabled))


def is_enabled() -> bool:
    return bool(_C._warmcache_is_enabled())


def size() -> int:
    return int(_C._warmcache_size())


def clear() -> None:
    _C._warmcache_clear()
