"""Bootstrap helpers for the C++ warm-runtime cache.

Lifecycle
---------
On a shim op's miss path (C++ shim didn't find a matching entry in the
warm cache), the C++ shim:
  1. Saves a thread-local "pending" `CacheKey` built from the live args.
  2. Calls the generated Python wrapper (e.g. ``add_out_rbln``) via pybind.

The Python wrapper's device path runs the op through a ``torch.compile``'d
callable whose backend is :func:`eager_backend`: rebel's ``rbln`` backend,
with each compiled graph wrapped so that executing it records its
``DynamoRuntime`` in a thread-local slot. After the call the wrapper takes
the record (:func:`take_recorded_runtime`) and hands it, with the output
tensors' profiles, to :func:`install_pending`.

:func:`install_pending` packs the output profiles into the shape the C++
side expects and hands them, with the runtime handle, to
``torch_rbln._C._warmcache_install_pending``. The C++ side matches
against the pending key it saved on the way in and inserts a
:class:`CacheEntry` keyed by (op name, input profile, scalars, device
index).

The record is written on every execution, not only when a graph is
compiled. A key that lost its entry (``torch.rbln.empty_cache``) therefore
gets it back on its next miss from the callable the Python compile cache
still holds: no Dynamo recompile, no second runtime. And a key whose graph
was compiled inside a callable another call created (a Dynamo recompile
for a new scalar value) is installed from the runtime that actually ran.

Subsequent dispatches of the same op with a matching input profile hit
the warm cache on the C++ side, which drives one execution through
rebel's C ABI (``rbln_exec_api.h``) — no pybind hop, no Python wrapper,
no Dynamo guard check. Driving it needs the runtime's
``native_handle()``; :func:`install_pending` skips the cache for a
runtime without one, leaving the op on the Python wrapper path. A key
whose hit failed at run time is retired by the C++ side (see
``WarmCache::disable``): it stays on the Python wrapper path and is not
offered for install again until the cache is cleared.
"""

from __future__ import annotations

import threading
from typing import Any

import torch
from torch._dynamo.backends.registry import lookup_backend

import torch_rbln._C as _C


_recorded = threading.local()


def _compile_graph(graph_module: Any, inputs: Any, options: Any) -> Any:
    """Compile one Dynamo graph with rebel's backend.

    Module-level so a test can count compiles under ``eager_backend``.
    """
    kwargs = {} if options is None else {"options": options}
    return lookup_backend("rbln")(graph_module, inputs, **kwargs)


def eager_backend(graph_module: Any, inputs: Any, *, options: Any = None) -> Any:
    """``torch.compile`` backend for the shim's Python path.

    Rebel's ``rbln`` backend, with the compiled graph wrapped so that every
    execution records its runtime for :func:`take_recorded_runtime`. The
    record is written after the graph has run: an op nested inside the run
    records and takes its own runtime first, so the outer record is never a
    nested op's.
    """
    runtime = _compile_graph(graph_module, inputs, options)

    def run(*args: Any) -> Any:
        out = runtime(*args)
        _recorded.runtime = runtime
        return out

    return run


def take_recorded_runtime() -> Any:
    """Return and clear the runtime the last :func:`eager_backend` graph run on this thread recorded."""
    runtime = getattr(_recorded, "runtime", None)
    _recorded.runtime = None
    return runtime


def _is_drivable_runtime_handle(handle: Any) -> bool:
    """True iff the C++ hit path can take a native handle off ``handle``."""
    return callable(getattr(handle, "native_handle", None))


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


def install_pending(dyn_runtime: Any, outputs: Any) -> bool:
    # Hot path of the Python wrapper: skip work whenever nothing ran on the
    # device or WarmCache is disabled. `_warmcache_is_enabled` is one C call;
    # cheaper than the rest of this function and makes WC OFF nearly free on
    # the cold path.
    if dyn_runtime is None or not _C._warmcache_is_enabled():
        return False

    runtime_handle = getattr(dyn_runtime, "_runtime_handle", None)
    if not _is_drivable_runtime_handle(runtime_handle):
        return False

    if isinstance(outputs, torch.Tensor):
        outputs = (outputs,)
    profiles = []
    for t in outputs:
        if not isinstance(t, torch.Tensor):
            continue
        dt = _DTYPE_KEY.get(t.dtype)
        if dt is None:
            return False
        profiles.append((list(t.shape), dt, t.device.type == "rbln"))
    if not profiles:
        return False

    ok = _C._warmcache_install_pending(
        dyn_runtime=dyn_runtime,
        runtime_handle=runtime_handle,
        out_profiles=profiles,
    )
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


def clear(device: int | None = None) -> None:
    """Drop the entries whose inputs live on ``device``, or all of them when ``None``."""
    _C._warmcache_clear(device)
