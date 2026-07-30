"""Monkey patches applied to PyTorch to enable RBLN functionality."""

import threading
import warnings

from torch_rbln._internal.compile_cache import clear_rbln_compile_cache
from torch_rbln._internal.torch_compile_patch_helpers import CompiledFunctionWrapper, is_rbln_backend


# Module-level state to track if patches have been applied
_torch_compile_patched: bool = False
_torch_dynamo_reset_patched: bool = False
_rbln_backend_registered: bool = False
_original_torch_compile = None
_original_dynamo_reset = None
_dynamo_guard_repr_patched: bool = False
_original_id_match_unchecked = None
_original_tensor_repr = None
_guard_repr_tls = threading.local()


def _is_backend_registered(backend_name: str) -> bool:
    """Whether backend_name is registered with torch._dynamo."""
    try:
        import torch

        if hasattr(torch._dynamo, "list_backends"):
            return backend_name in torch._dynamo.list_backends()
        if hasattr(torch._dynamo, "backends"):
            return backend_name in torch._dynamo.backends
        return False
    except Exception:
        return False


def _register_rbln_backend() -> bool:
    """Register the RBLN backend with torch._dynamo; True on success."""
    global _rbln_backend_registered

    if _rbln_backend_registered or _is_backend_registered("rbln"):
        _rbln_backend_registered = True
        return True

    try:
        # Importing registers 'rbln' via module-level register_backend() side effects.
        import rebel.core.torch_compile  # noqa: F401

        if _is_backend_registered("rbln"):
            _rbln_backend_registered = True
            return True
        else:
            warnings.warn(
                "RBLN backend import succeeded but backend was not registered. "
                "torch.compile with backend='rbln' may not work.",
                UserWarning,
            )
            return False

    except ImportError as e:
        warnings.warn(
            f"Failed to register rbln backend for torch.compile: {e}. "
            "torch.compile will work but 'rbln' backend may not be available.",
            UserWarning,
        )
        return False


def patch_torch_compile() -> None:
    """
    Monkey patch torch.compile() to automatically register the RBLN backend on first use
    and add automatic num_devices determination and tensor-parallel failover support.

    This patch wraps torch.compile() to ensure the RBLN backend is registered before
    the first compilation. The registration is lazy (happens on first call) to avoid
    import-time dependencies.
    """
    global _original_dynamo_reset, _original_torch_compile, _torch_compile_patched, _torch_dynamo_reset_patched

    import torch

    if _original_torch_compile is None:
        _original_torch_compile = torch.compile
    if _original_dynamo_reset is None:
        _original_dynamo_reset = torch._dynamo.reset

    if not _torch_compile_patched:
        original_torch_compile = _original_torch_compile

        def wrapper(*args, **kwargs):
            """Wrapper that registers RBLN backend on first use, then calls original torch.compile."""
            # Lazy registration: register backend on first use
            if not _rbln_backend_registered:
                _register_rbln_backend()

            # Early return for non-RBLN backends
            backend = kwargs.get("backend", "inductor")
            if not is_rbln_backend(backend):
                return original_torch_compile(*args, **kwargs)

            # Detect the model from either call form: torch.compile(m, ...) passes it in
            # args[0]; torch.compile(model=m, ...) passes it in kwargs. Normalize it out of
            # kwargs so the compile options stored on CompiledFunctionWrapper (reused for
            # failover recompiles) never carry a stray model= that would clash with the
            # positional model on recompile.
            model = args[0] if args else kwargs.pop("model", None)

            # Model provided (either form): wrap the compiled callable so RBLN failover,
            # CPU fallback, and the RBLN_DUMMY_DEVICE execution guard apply.
            if model is not None:
                compiled_fn = original_torch_compile(model, **kwargs)
                return CompiledFunctionWrapper(compiled_fn, model, original_torch_compile, kwargs.copy())

            # No model: factory/decorator form, e.g.
            #   f = torch.compile(backend="rbln")(fn)   or   @torch.compile(backend="rbln")
            # torch returns a *decorator* here; wrap the eventual compiled fn so the same
            # guard/failover/fallback apply to this form as to the model path above.
            torch_decorator = original_torch_compile(**kwargs)

            def rbln_compile_decorator(m):
                return CompiledFunctionWrapper(torch_decorator(m), m, original_torch_compile, kwargs.copy())

            return rbln_compile_decorator

        torch.compile = wrapper
        _torch_compile_patched = True

    if not _torch_dynamo_reset_patched:
        original_dynamo_reset = _original_dynamo_reset

        def reset_wrapper(*args, **kwargs):
            clear_rbln_compile_cache()
            # torch._dynamo.reset() means "forget all compiled state". Also flush
            # the C++ warm cache — otherwise a matching input profile still hits a
            # stale runtime directly after reset, bypassing the cleared Python
            # cache and Dynamo. Lazy import avoids an import cycle.
            #
            # Caveat: the flush can drop the last reference to a warm runtime and
            # free its device buffers. Drain in-flight work (torch.rbln.synchronize)
            # before reset(); resetting a device with pending DMA is undefined.
            from torch_rbln._internal import warm_cache

            warm_cache.clear()
            return original_dynamo_reset(*args, **kwargs)

        torch._dynamo.reset = reset_wrapper
        _torch_dynamo_reset_patched = True


def patch_dynamo_guard_repr() -> None:
    """Keep Dynamo guard build from materializing tensor values via ``repr``.

    Substitute value->type only during guard build; user ``repr(tensor)`` is
    unchanged. Backport of the pytorch-main fix; a no-op on torch >= 2.13.
    See rebellions-sw/fsw-inference#413.
    """
    global _original_id_match_unchecked, _original_tensor_repr, _dynamo_guard_repr_patched
    if _dynamo_guard_repr_patched:
        return

    import torch

    # Fixed upstream in torch 2.13 (id_match_unchecked reprs the type, not the
    # value), so there is nothing to patch there. 2.11 and 2.12 still repr(val).
    if torch.__version__ >= (2, 13):
        return

    try:
        from torch._dynamo.guards import GuardBuilder
    except Exception as e:  # pragma: no cover
        warnings.warn(f"Could not patch Dynamo guard repr: {e}", stacklevel=2)
        return

    _original_id_match_unchecked = GuardBuilder.id_match_unchecked
    _original_tensor_repr = torch.Tensor.__repr__

    def _id_match_unchecked(self, guard, recompile_hint=None):
        prev = getattr(_guard_repr_tls, "active", False)
        _guard_repr_tls.active = True
        try:
            return _original_id_match_unchecked(self, guard, recompile_hint)
        finally:
            _guard_repr_tls.active = prev

    def _tensor_repr(self, *args, **kwargs):
        # Guard build only needs the type; avoid materializing the tensor.
        if getattr(_guard_repr_tls, "active", False):
            return repr(type(self))
        return _original_tensor_repr(self, *args, **kwargs)

    GuardBuilder.id_match_unchecked = _id_match_unchecked
    torch.Tensor.__repr__ = _tensor_repr
    _dynamo_guard_repr_patched = True


def apply_all_patches() -> None:
    """Apply all RBLN monkey patches. Idempotent."""
    patch_torch_compile()
    patch_dynamo_guard_repr()


def remove_all_patches() -> None:
    """
    Remove all monkey patches (restore original behavior).

    WARNING: This function is primarily for testing purposes.
    """
    global _rbln_backend_registered, _torch_compile_patched, _torch_dynamo_reset_patched
    global _dynamo_guard_repr_patched

    import torch

    if _original_torch_compile is not None:
        torch.compile = _original_torch_compile
    if _original_dynamo_reset is not None:
        torch._dynamo.reset = _original_dynamo_reset

    if _dynamo_guard_repr_patched:
        from torch._dynamo.guards import GuardBuilder

        GuardBuilder.id_match_unchecked = _original_id_match_unchecked
        torch.Tensor.__repr__ = _original_tensor_repr
        _dynamo_guard_repr_patched = False

    clear_rbln_compile_cache()
    # Mirror reset_wrapper's teardown: drop the C++ warm cache's runtime
    # references too, so removal fully undoes apply. Lazy import avoids a cycle.
    from torch_rbln._internal import warm_cache

    warm_cache.clear()
    _torch_compile_patched = False
    _torch_dynamo_reset_patched = False
    _rbln_backend_registered = False
