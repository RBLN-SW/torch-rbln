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
``DynamoRuntime`` that owns the compiled sync runtime to this list. The generated wrapper then calls :func:`install_pending` with
that runtime and the post-compile output-tensor profiles.

:func:`install_pending` packs the output profiles into the shape the
C++ side expects and hands them, with the runtime handle, to
``torch_rbln._C._warmcache_install_pending``. The C++ side matches
against the pending key it saved on the way in and inserts a
:class:`CacheEntry` keyed by (op name, input profile, scalars, device
index).

Subsequent dispatches of the same op with a matching input profile hit
the warm cache on the C++ side, which calls the handle's
``prepare_inputs`` / ``prepare_outputs`` / ``run`` — no Python wrapper,
no Dynamo recompile check.
"""

from __future__ import annotations

from typing import Any

import torch

import torch_rbln._C as _C


# What the C++ hit path calls on the runtime handle; see WarmCache.h.
_RUNTIME_METHODS = ("prepare_inputs", "prepare_outputs", "run")


def _is_drivable_runtime_handle(handle: Any) -> bool:
    """True iff the C++ hit path can drive ``handle``.

    The three methods are the whole contract; a handle missing any of them is
    skipped so the op stays on the (correct, slower) Python wrapper path.
    """
    return all(callable(getattr(handle, name, None)) for name in _RUNTIME_METHODS)


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
    if not _is_drivable_runtime_handle(runtime_handle):
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
        runtime_handle=runtime_handle,
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
