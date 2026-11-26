"""Monkey patches applied to PyTorch to enable RBLN functionality."""

import warnings

import torch

from torch_rbln._internal.compile_cache import clear_rbln_compile_cache
from torch_rbln._internal.torch_compile_patch_helpers import CompiledFunctionWrapper, is_rbln_backend


# Module-level state to track if patches have been applied
_torch_compile_patched: bool = False
_rbln_backend_registered: bool = False
_original_torch_compile = None


def _is_backend_registered(backend_name: str) -> bool:
    """Whether backend_name is registered with torch._dynamo."""
    try:
        backends = torch._dynamo.list_backends() if hasattr(torch._dynamo, "list_backends") else []
    except Exception:
        return False
    return backend_name in backends


def _rbln_backend_reset() -> None:
    """Defined as ``reset`` on the 'rbln' backend; torch._dynamo.reset() calls it."""
    clear_rbln_compile_cache()
    # torch._dynamo.reset() means "forget all compiled state". Also flush
    # the C++ warm cache — otherwise a matching input profile still hits a
    # stale runtime directly after reset, bypassing the cleared Python
    # cache and Dynamo. Lazy import avoids an import cycle.
    #
    # Caveat: the flush can drop the last reference to a warm runtime and
    # free its device buffers. Drain in-flight work (torch.rbln.synchronize)
    # before reset(); resetting a device with pending DMA is undefined.
    from torch_rbln._internal import warm_cache  # noqa: PLC0415

    warm_cache.clear()


def _register_rbln_backend() -> bool:
    """Register the RBLN backend with torch._dynamo; True on success."""
    global _rbln_backend_registered

    if _rbln_backend_registered or _is_backend_registered("rbln"):
        setattr(torch._dynamo.lookup_backend("rbln"), "reset", _rbln_backend_reset)  # noqa: B010
        _rbln_backend_registered = True
        return True

    try:
        # Importing registers 'rbln' via module-level register_backend() side effects.
        import rebel.core.torch_compile  # noqa: F401,PLC0415
    except ImportError as e:
        warnings.warn(
            f"Failed to register rbln backend for torch.compile: {e}. "
            "torch.compile will work but 'rbln' backend may not be available.",
            UserWarning,
            stacklevel=1,
        )
        return False

    if _is_backend_registered("rbln"):
        setattr(torch._dynamo.lookup_backend("rbln"), "reset", _rbln_backend_reset)  # noqa: B010
        _rbln_backend_registered = True
        return True
    warnings.warn(
        "RBLN backend import succeeded but backend was not registered. torch.compile with backend='rbln' may not work.",
        UserWarning,
        stacklevel=1,
    )
    return False


def patch_torch_compile() -> None:
    """Monkey patch torch.compile() to automatically register the RBLN backend on first
    use and add automatic num_devices determination and tensor-parallel failover
    support.

    This patch wraps torch.compile() to ensure the RBLN backend is registered before
    the first compilation. The registration is lazy (happens on first call) to avoid
    import-time dependencies.
    """
    global _original_torch_compile, _torch_compile_patched

    if _original_torch_compile is None:
        _original_torch_compile = torch.compile

    if not _torch_compile_patched:
        original_torch_compile = _original_torch_compile

        def wrapper(*args, **kwargs):
            """Wrapper that registers RBLN backend on first use, then calls original
            torch.compile.
            """
            # Lazy registration: register backend on first use
            if not _rbln_backend_registered:
                _register_rbln_backend()

            # Early return for non-RBLN backends
            backend = kwargs.get("backend", "inductor")
            if not is_rbln_backend(backend):
                return original_torch_compile(*args, **kwargs)

            # Detect the model from either call form: torch.compile(m, ...) passes it in
            # args[0]; torch.compile(model=m, ...) passes it in kwargs. Normalize it out
            # of kwargs so the compile options stored on CompiledFunctionWrapper (reused
            # for failover recompiles) never carry a stray model= that would clash with
            # the positional model on recompile.
            model = args[0] if args else kwargs.pop("model", None)

            # Model provided (either form): wrap the compiled callable so RBLN failover
            # and CPU fallback apply.
            if model is not None:
                compiled_fn = original_torch_compile(model, **kwargs)
                return CompiledFunctionWrapper(compiled_fn, model, original_torch_compile, kwargs.copy())

            # No model: factory/decorator form, e.g. torch.compile(backend="rbln")(fn)
            # or @torch.compile(backend="rbln"). torch returns a *decorator* here; wrap
            # the eventual compiled fn so the same failover/fallback apply to this
            # form as to the model path above.
            torch_decorator = original_torch_compile(**kwargs)

            def rbln_compile_decorator(m):
                return CompiledFunctionWrapper(torch_decorator(m), m, original_torch_compile, kwargs.copy())

            return rbln_compile_decorator

        torch.compile = wrapper
        _torch_compile_patched = True


def apply_all_patches() -> None:
    """Apply all RBLN monkey patches. Idempotent."""
    patch_torch_compile()


def remove_all_patches() -> None:
    """Remove all monkey patches (restore original behavior).

    WARNING: This function is primarily for testing purposes.
    """
    global _rbln_backend_registered, _torch_compile_patched

    if _original_torch_compile is not None:
        torch.compile = _original_torch_compile
    if _rbln_backend_registered:
        delattr(torch._dynamo.lookup_backend("rbln"), "reset")

    _rbln_backend_reset()
    _torch_compile_patched = False
    _rbln_backend_registered = False
