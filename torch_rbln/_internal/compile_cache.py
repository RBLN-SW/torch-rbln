"""Per-module ``torch.compile`` cache used by the eager-dispatch wrappers.

Each shim op (``add_rbln``, ``sub_rbln``, …) calls
:func:`compile_rbln_cached` with its own ``OpModule`` instance to get a
torch.compile'd callable. We cache by ``(module-id, dynamic, device-cache-key,
options, backend)`` so that repeated eager-path dispatches of the same op
reuse the same compiled graph.

This cache owns the compiled callables and, through Dynamo's cache for
their graphs, the ``DynamoRuntime`` instances behind them. The C++ warm
cache (``torch_rbln._internal.warm_cache``) only borrows those runtimes:
an entry there is installed from the runtime a cached callable ran, so
dropping the C++ side (``torch.rbln.empty_cache``) does not touch this
cache and the next miss installs the same runtime again. Clearing *this*
cache (:func:`clear_rbln_compile_cache`, from ``torch._dynamo.reset``) is
what makes the next call run Dynamo and the rebel backend again.

A cached callable is a ``torch.compile`` wrapper, and Dynamo reuses a graph
across wrappers whose backend and ``options`` compare equal. Keep
``options`` to plain values: an option that compares unequal on every call
turns each miss here into a Dynamo recompile that leaves the previous
graph, and its runtime, in Dynamo's own cache until the recompile limit
resets everything.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from typing import Any

import torch


_compiled_op_cache_lock = threading.Lock()
_compiled_op_cache: dict[tuple[Any, ...], Any] = {}


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


def compile_rbln_cached(
    model: Any,
    *,
    dynamic: bool = False,
    options: dict[str, Any] | None = None,
    device_cache_key: Any = None,
    backend: Any = "rbln",
) -> Any:
    # ``device_cache_key`` accepts any hashable; callers that want per-shape
    # warm-cache entries pass a (device_index, shape_sig, dtype_sig) tuple so
    # distinct input profiles end up in distinct compile_rbln_cached entries.
    cache_key = (
        _IdentityKey(model),
        dynamic,
        _freeze_cache_value(device_cache_key),
        _freeze_cache_value(options),
        _freeze_cache_value(backend),
    )

    compiled = _compiled_op_cache.get(cache_key)
    if compiled is not None:
        return compiled

    with _compiled_op_cache_lock:
        compiled = _compiled_op_cache.get(cache_key)
        if compiled is None:
            compiled = torch.compile(model, backend=backend, dynamic=dynamic, options=options)
            _compiled_op_cache[cache_key] = compiled
        return compiled


def clear_rbln_compile_cache() -> None:
    with _compiled_op_cache_lock:
        _compiled_op_cache.clear()
