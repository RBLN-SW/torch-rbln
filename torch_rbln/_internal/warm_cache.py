"""Bootstrap helpers for the C++ warm-op cache.

Lifecycle
---------
On a shim op's miss path (C++ shim didn't find a matching entry in the
warm cache), the C++ shim:
  1. Saves a thread-local "pending" `CacheKey` built from the live args, with
     the call's tensors.
  2. Calls the generated Python wrapper (e.g. ``add_out_rbln``) via pybind.

The wrapper compiles the op into a ``torch_rbln._C._OpFunction``, runs it,
and calls :func:`install_pending` with the function and the tensors it ran
over. The C++ side matches those tensors to the call's by identity and inserts
a ``CacheEntry`` keyed by (op name, input profile, scalars). A wrapper that ran
the function over tensors of its own making (a copy, a view's base) installs
nothing: the hit path binds the call's tensors as they are.

Subsequent dispatches of the same op with a matching input profile hit the
warm cache on the C++ side, which runs the function over the stack's tensors —
no pybind hop, no Python wrapper. A hit whose tensors the function cannot take
(an ``out`` of another shape) falls through to the wrapper; the entry stays.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch_rbln._C as _C


if TYPE_CHECKING:
    import torch


def install_pending(function: Any, tensors: list[torch.Tensor]) -> bool:
    if not _C._warmcache_is_enabled():
        return False
    return bool(_C._warmcache_install_pending(function, tensors))


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
