"""
Monkey patches for torch_rbln.

This module contains all monkey patches applied to PyTorch to enable RBLN functionality.
Patches are organized by functionality and include proper error handling and idempotency checks.
"""

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
    """
    Check if a backend is already registered with torch._dynamo.

    Args:
        backend_name: Name of the backend to check.

    Returns:
        True if the backend is registered, False otherwise.
    """
    try:
        import torch

        # Try to list backends - this may not be available in all PyTorch versions
        if hasattr(torch._dynamo, "list_backends"):
            backends = torch._dynamo.list_backends()
            return backend_name in backends

        # Fallback: try to get the backend directly
        if hasattr(torch._dynamo, "backends"):
            return backend_name in torch._dynamo.backends

        # If we can't check, assume it's not registered
        return False
    except Exception:
        # If anything fails, assume not registered
        return False


def _register_rbln_backend() -> bool:
    """
    Register the RBLN backend with torch._dynamo.

    Returns:
        True if registration was successful, False otherwise.
    """
    global _rbln_backend_registered

    # Check if already registered
    if _rbln_backend_registered or _is_backend_registered("rbln"):
        _rbln_backend_registered = True
        return True

    try:
        # Import rebel_compiler's torch_compile module to register backend
        # This will execute the register_backend calls at module level
        import rebel.core.torch_compile  # noqa: F401

        # Verify registration succeeded
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
    and add automatic tensor parallel size determination and failover support.

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
            global _rbln_backend_registered
            if not _rbln_backend_registered:
                _register_rbln_backend()

            # Early return for non-RBLN backends
            backend = kwargs.get("backend", "inductor")
            if not is_rbln_backend(backend):
                return original_torch_compile(*args, **kwargs)

            # RBLN backend: compile and wrap if model provided
            original_fn = args[0] if args else None
            compiled_fn = original_torch_compile(*args, **kwargs)
            if args:
                return CompiledFunctionWrapper(
                    compiled_fn,
                    original_fn,
                    original_torch_compile,
                    kwargs.copy(),
                )
            # fallthrough for model is not provided (e.g. torch.compile(backend="rbln"))
            return compiled_fn

        # Apply patch
        torch.compile = wrapper
        _torch_compile_patched = True

    if not _torch_dynamo_reset_patched:
        original_dynamo_reset = _original_dynamo_reset

        def reset_wrapper(*args, **kwargs):
            clear_rbln_compile_cache()
            return original_dynamo_reset(*args, **kwargs)

        torch._dynamo.reset = reset_wrapper
        _torch_dynamo_reset_patched = True


def apply_all_patches() -> None:
    """
    Apply all monkey patches for RBLN functionality.

    This function applies patches in the correct order:
    1. torch.compile() patch

    Idempotent: Safe to call multiple times.
    """
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
