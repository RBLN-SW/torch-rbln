"""Bootstrap helpers for the C++ warm-runtime cache.

Lifecycle
---------
On a shim op's miss path (C++ shim didn't find a matching entry in the
warm cache), the C++ shim:
  1. Saves a thread-local "pending" `CacheKey` built from the live args.
  2. Calls the generated Python wrapper (e.g. ``add_out_rbln``) via pybind.

The Python wrapper's ``else`` branch (device path) passes an empty
``_runtime_holder`` list via the compile ``options``. After the first
backend compile, rebel's ``rbln_backend`` appends the ``DynamoRuntime``
that owns the compiled sync runtime to this list. The generated wrapper
then calls :func:`install_pending` with that runtime and the
post-compile output-tensor profiles.

:func:`install_pending` resolves the runtime's ``prepare_inputs`` /
``prepare_outputs`` / ``run`` through
:mod:`torch_rbln._internal.rebel_contract`, which declares those names
and is the only place they are spelled, and hands the bound methods and
the output profiles to ``torch_rbln._C._warmcache_install_pending``. The
C++ side matches against the pending key it saved on the way in and
inserts a :class:`CacheEntry` keyed by (op name, input profile, scalars,
device index).

Subsequent dispatches of the same op with a matching input profile hit
the warm cache on the C++ side, which calls those three bound methods —
no Python wrapper, no Dynamo recompile check.

A hit that raises ``TypeError`` means rebel's runtime no longer accepts
this build's call. The C++ side then disables the cache for the process
and flags it; :func:`install_pending` consumes the flag and reports which
declared names diverged.
"""

from __future__ import annotations

import warnings
from typing import Any

import torch

import torch_rbln._C as _C
from torch_rbln._internal import rebel_contract


def _warn_contract_break() -> None:
    """Report the rebel divergence that turned the warm cache off.

    The C++ hit path sets the flag once and disables the cache in the same step, so this runs
    at most once per process. The divergence list comes from the contract declaration, which is
    why the message can name what changed instead of only that something did.
    """
    divergences = rebel_contract.verify()
    detail = "\n  ".join(str(d) for d in divergences) or "rebel_contract.verify() found no divergence"
    warnings.warn(
        f"The RBLN warm cache is off for this process: rebel's runtime rejected the call this "
        f"torch-rbln makes, so eager ops take the slower Python path and results are unaffected. "
        f"Against the installed rebel-compiler:\n  {detail}",
        RuntimeWarning,
        stacklevel=2,
    )


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
    if not _C._warmcache_is_enabled():
        # The break disables the cache, and the miss it falls through to reaches here with an
        # empty holder (nothing recompiled), so the flag has to be read before that early exit.
        if _C._warmcache_take_contract_break():
            _warn_contract_break()
        if runtime_holder:
            runtime_holder.clear()
        return False
    if not runtime_holder:
        return False

    dyn_runtime = runtime_holder[-1]
    runtime_handle = getattr(dyn_runtime, rebel_contract.RUNTIME_HANDLE_ATTR, None)
    methods = rebel_contract.runtime_methods(runtime_handle)
    if methods is None:
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

    prepare_inputs, prepare_outputs, run = methods
    ok = _C._warmcache_install_pending(
        dyn_runtime=dyn_runtime,
        prepare_inputs=prepare_inputs,
        prepare_outputs=prepare_outputs,
        run=run,
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
