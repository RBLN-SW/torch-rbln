"""Per-module ``torch.compile`` cache used by the eager-dispatch wrappers.

Each shim op (``add_rbln``, ``sub_rbln``, …) calls
:func:`compile_rbln_cached` with its own ``OpModule`` instance to get a
torch.compile'd callable. We cache by ``(module-id, dynamic, device-id,
options)`` so that repeated eager-path dispatches of the same op reuse
the same compiled graph.

``options`` handling
--------------------
The rebel backend accepts a side-channel option ``_runtime_holder`` (a
mutable list) into which it appends the freshly-created ``DynamoRuntime``
on its first compile pass — used by ``torch_rbln._internal.warm_cache``
to harvest the runtime for the C++ warm cache.

If we naively fed the holder's identity into the cache key, every call
would produce a distinct key (fresh list per call) and torch.compile
would re-trigger on every invocation. Instead we strip the holder from
the cache key while still passing the *full* options through on miss so
the backend can still populate it.

Only the **first** miss for a given (module, options-minus-holder) key
actually runs the rebel backend (and therefore populates the holder);
subsequent hits return the already-compiled callable without touching
the holder. That is exactly the semantic the warm-cache bootstrap needs
— it installs the cache entry only when the holder is populated.

The entry keeps that first holder (:class:`CompiledOp`), so a later hit
can reach the ``DynamoRuntime`` and drive it directly with the caller's
``out`` buffers, the way the C++ warm cache does, instead of going back
through the compiled callable.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from typing import Any

import torch


_compiled_op_cache_lock = threading.Lock()
_compiled_op_cache: dict[tuple[Any, ...], CompiledOp] = {}

# Key used to strip the runtime-holder side-channel from cache keys.
_RUNTIME_HOLDER_KEY = "_runtime_holder"


class _IdentityKey:
    """Hashable identity wrapper that keeps the referenced object alive."""

    __slots__ = ("value", "_hash")

    def __init__(self, value: Any):
        self.value = value
        self._hash = id(value)

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _IdentityKey) and self.value is other.value


def _cache_sort_key(value: Any) -> tuple[str, str]:
    if isinstance(value, _IdentityKey):
        return ("identity", str(hash(value)))
    return (type(value).__name__, repr(value))


def _freeze_cache_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(sorted((key, _freeze_cache_value(item)) for key, item in value.items()))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_cache_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_freeze_cache_value(item) for item in value), key=_cache_sort_key))
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return _IdentityKey(value)


def _options_cache_view(options: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Drop per-call side-channel keys from a copy of ``options``.

    Currently strips ``_runtime_holder`` only, since that's the single
    mutable identity-bearing option we use. Add more entries here if new
    side-channels appear.
    """
    if not options:
        return {}
    return {k: v for k, v in options.items() if k != _RUNTIME_HOLDER_KEY}


class CompiledOp:
    """A compiled callable, the runtimes the rebel backend built for it, and
    how a call's tensors map onto the graph's inputs once that is known."""

    __slots__ = ("compiled", "runtime_holder", "input_order", "copy_fallback_warned")

    def __init__(self, compiled: Any, runtime_holder: list):
        self.compiled = compiled
        self.runtime_holder = runtime_holder
        self.input_order: tuple[int, ...] | None = None
        self.copy_fallback_warned = False

    def runtime(self) -> Any:
        """The one ``DynamoRuntime`` behind ``compiled``, or ``None``.

        ``None`` before the callable has run once, since the backend builds the
        runtime during that run, and when the holder has grown past one: Dynamo
        recompiled inside the callable, so a guard the cache key does not cover
        differed between calls, and only the callable knows which runtime a
        given call takes.
        """
        if len(self.runtime_holder) != 1:
            return None
        return self.runtime_holder[0]


def compile_rbln_cached(
    model: Any,
    *,
    dynamic: bool = False,
    options: dict[str, Any] | None = None,
    device_cache_key: Any = None,
    force_recompile: bool = False,
) -> Any:
    return compile_rbln_cached_entry(
        model,
        dynamic=dynamic,
        options=options,
        device_cache_key=device_cache_key,
        force_recompile=force_recompile,
    ).compiled


def compile_rbln_cached_entry(
    model: Any,
    *,
    dynamic: bool = False,
    options: dict[str, Any] | None = None,
    device_cache_key: Any = None,
    force_recompile: bool = False,
) -> CompiledOp:
    # The cache key excludes ``_runtime_holder`` (see module docstring).
    # ``device_cache_key`` accepts any hashable; callers that want per-shape
    # warm-cache harvesting pass a (device_index, shape_sig, dtype_sig) tuple
    # so distinct input profiles end up in distinct compile_rbln_cached
    # entries and therefore receive fresh ``_runtime_holder`` slots.
    #
    # ``force_recompile`` is set by ``compile_and_run_view_aware`` when the
    # C++ warm-cache erase'd an entry for this key on the prior dispatch
    # (e.g. rebel runtime soft-failed). Without this, a cache-hit would
    # return the same compiled callable that the rebel backend has already
    # instantiated once and does not re-instantiate again, so
    # ``_runtime_holder`` would stay empty and ``_install_warm_cache_pending``
    # would never fire for this op+shape again. Drop the existing entry
    # so the miss-path below re-runs ``torch.compile``, which re-populates
    # the holder. See ``WarmCache::request_force_recompile`` in
    # ``torch_rbln/csrc/rbln/DispatchShim.cpp`` for the producer side.
    cache_options = _options_cache_view(options)
    cache_key = (
        _IdentityKey(model),
        dynamic,
        _freeze_cache_value(device_cache_key),
        _freeze_cache_value(cache_options),
    )

    if force_recompile:
        with _compiled_op_cache_lock:
            _compiled_op_cache.pop(cache_key, None)

    entry = _compiled_op_cache.get(cache_key)
    if entry is not None:
        return entry

    with _compiled_op_cache_lock:
        entry = _compiled_op_cache.get(cache_key)
        if entry is None:
            # Use the *full* options on miss so the backend sees the holder.
            compiled = torch.compile(model, backend="rbln", dynamic=dynamic, options=options)
            holder = options.get(_RUNTIME_HOLDER_KEY) if options is not None else None
            entry = CompiledOp(compiled, holder if holder is not None else [])
            _compiled_op_cache[cache_key] = entry
        return entry


def clear_rbln_compile_cache() -> None:
    with _compiled_op_cache_lock:
        _compiled_op_cache.clear()
