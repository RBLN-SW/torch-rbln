import threading
from collections.abc import Mapping, Sequence
from typing import Any

import torch


_compiled_op_cache_lock = threading.Lock()
_compiled_op_cache: dict[tuple[Any, ...], Any] = {}


def _freeze_cache_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(sorted((key, _freeze_cache_value(item)) for key, item in value.items()))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_cache_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted(_freeze_cache_value(item) for item in value))
    if callable(value):
        return ("callable", id(value))
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return ("object", id(value))


def compile_rbln_cached(
    model: Any,
    *,
    dynamic: bool = False,
    options: dict[str, Any] | None = None,
    device_cache_key: int | None = None,
) -> Any:
    cache_key = (
        id(model),
        dynamic,
        device_cache_key,
        _freeze_cache_value(options or {}),
    )

    compiled = _compiled_op_cache.get(cache_key)
    if compiled is not None:
        return compiled

    with _compiled_op_cache_lock:
        compiled = _compiled_op_cache.get(cache_key)
        if compiled is None:
            compiled = torch.compile(model, backend="rbln", dynamic=dynamic, options=options)
            _compiled_op_cache[cache_key] = compiled
        return compiled


def clear_rbln_compile_cache() -> None:
    with _compiled_op_cache_lock:
        _compiled_op_cache.clear()
