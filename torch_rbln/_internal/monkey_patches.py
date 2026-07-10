"""Monkey patches applied to PyTorch to enable RBLN functionality."""

import warnings

from torch_rbln._internal.compile_cache import clear_rbln_compile_cache
from torch_rbln._internal.torch_compile_patch_helpers import CompiledFunctionWrapper, is_rbln_backend


# Module-level state to track if patches have been applied
_torch_compile_patched: bool = False
_torch_dynamo_reset_patched: bool = False
_rbln_backend_registered: bool = False
_original_torch_compile = None
_original_dynamo_reset = None


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

        # Tag the wrapper so a leak guard can verify torch.compile IS the RBLN wrapper by
        # identity, not just trust the _torch_compile_patched flag (a test may rebind
        # torch.compile to something else without going through remove_all_patches).
        wrapper._rbln_patch_marker = True
        torch.compile = wrapper
        _torch_compile_patched = True

    if not _torch_dynamo_reset_patched:
        original_dynamo_reset = _original_dynamo_reset

        def reset_wrapper(*args, **kwargs):
            clear_rbln_compile_cache()
            return original_dynamo_reset(*args, **kwargs)

        reset_wrapper._rbln_patch_marker = True
        torch._dynamo.reset = reset_wrapper
        _torch_dynamo_reset_patched = True


def patches_active() -> bool:
    """Whether the RBLN torch.compile / torch._dynamo.reset patches are actually installed.

    Checks callable identity (the ``_rbln_patch_marker`` tag), not just the bookkeeping
    flags, so a test that rebinds ``torch.compile`` to something else without calling
    remove_all_patches() is still detected as having lost the patches."""
    import torch

    return (
        _torch_compile_patched
        and _torch_dynamo_reset_patched
        and getattr(torch.compile, "_rbln_patch_marker", False)
        and getattr(torch._dynamo.reset, "_rbln_patch_marker", False)
    )


def apply_all_patches() -> None:
    """Apply all RBLN monkey patches. Idempotent."""
    patch_torch_compile()


def remove_all_patches() -> None:
    """
    Remove all monkey patches (restore original behavior).

    WARNING: This function is primarily for testing purposes.
    """
    global _rbln_backend_registered, _torch_compile_patched, _torch_dynamo_reset_patched

    import torch

    if _original_torch_compile is not None:
        torch.compile = _original_torch_compile
    if _original_dynamo_reset is not None:
        torch._dynamo.reset = _original_dynamo_reset

    clear_rbln_compile_cache()
    _torch_compile_patched = False
    _torch_dynamo_reset_patched = False
    _rbln_backend_registered = False
