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
C++ instance pointer.

The offset of that slot is ``sizeof(PyObject)``, which varies by
Python build (standard vs debug, PEP 703 free-threaded, 32- vs
64-bit). We let the C++ side compute it at compile time against the
Python ABI we link against; the Python side just calls the C++
helper. See ``torch_rbln._C._pybind_instance_raw_ptr`` in
``torch_rbln/csrc/rbln/Module.cpp``.

Type gate
---------
The raw-pointer trick is only valid for a pybind11 simple-layout
``PyRblnSyncRuntime``. Any other object (rebel refactor / version skew)
would still yield a non-null integer that the C++ side caches and later
dereferences, segfaulting. :func:`install_pending` therefore checks the
handle's type (:func:`_is_expected_runtime_handle`) before extracting the
pointer; on mismatch it skips the cache and ops use the safe pybind path.
A fully robust ABI handshake is a rebel-compiler follow-up.
"""

from __future__ import annotations

from typing import Any

import torch

import torch_rbln._C as _C


def _raw_cpp_ptr(pybound_instance: Any) -> int:
    """Return the raw C++ pointer held by a pybind11 instance as ``uintptr_t``.

    Relies on pybind11's "simple layout" for single-inheritance types with a
    standard holder. Verified against rebel's ``PyRblnSyncRuntime``.
    """
    raw = _C._pybind_instance_raw_ptr(pybound_instance)
    if not raw:
        raise RuntimeError(
            "warm_cache: unexpected null C++ pointer extracted from pybind instance; layout assumption may be wrong"
        )
    return int(raw)


# pybind type name of rebel's runtime handle; gating on it catches a
# renamed/replaced handle before we trust the raw-pointer read (see docstring).
_EXPECTED_RUNTIME_TYPE_NAME = "PyRblnSyncRuntime"


def _is_expected_runtime_handle(handle: Any) -> bool:
    """True iff ``handle`` is the rebel runtime type the raw-ptr bridge expects.

    Conservative type-name check: rejects a None/wrong/refactored handle. It
    can't catch a same-named type with a different ABI (offset skew) — that
    needs a rebel-side accessor (follow-up). When unsure, returns False.
    """
    if handle is None:
        return False
    return type(handle).__name__ == _EXPECTED_RUNTIME_TYPE_NAME


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
    runtime_handle = getattr(dyn_runtime, "_runtime_handle", None)
    # Type gate (see module docstring): skip the fast path on a wrong/changed
    # handle rather than cache a pointer we'd dereference blindly -> segfault.
    if not _is_expected_runtime_handle(runtime_handle):
        runtime_holder.clear()
        return False
    try:
        raw_ptr = _raw_cpp_ptr(runtime_handle)
    except RuntimeError:
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


def consume_force_recompile() -> bool:
    """Consume the thread-local force-recompile flag.

    Set by the C++ shim when ``try_warmcache_hit`` had to ``erase`` a
    broken entry. ``compile_and_run_view_aware`` consumes the flag right
    before calling ``compile_rbln_cached`` so the next compile bypasses
    the Python compile cache for this key, re-runs ``torch.compile``, and
    re-populates ``_runtime_holder`` so install can fire again.
    """
    return bool(_C._warmcache_consume_force_recompile())
