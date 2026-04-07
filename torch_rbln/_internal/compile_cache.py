import threading
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from torch_rbln._internal.profiling import profile_phase, record_counter


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
    device_cache_key: int | None = None,
) -> Any:
    with profile_phase("compile_cache.total"):
        cache_key = (
            _IdentityKey(model),
            dynamic,
            device_cache_key,
            _freeze_cache_value(options or {}),
        )

        compiled = _compiled_op_cache.get(cache_key)
        if compiled is not None:
            record_counter("compile_cache.hit")
            return compiled

        with profile_phase("compile_cache.lock_wait"):
            with _compiled_op_cache_lock:
                compiled = _compiled_op_cache.get(cache_key)
                if compiled is None:
                    record_counter("compile_cache.miss")
                    with profile_phase("compile_cache.torch_compile_api"):
                        compiled = torch.compile(model, backend="rbln", dynamic=dynamic, options=options)
                    _compiled_op_cache[cache_key] = compiled
                else:
                    record_counter("compile_cache.hit_after_lock")
                return compiled


def clear_rbln_compile_cache() -> None:
    with _compiled_op_cache_lock:
        _compiled_op_cache.clear()
