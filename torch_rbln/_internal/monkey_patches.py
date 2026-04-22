"""
Monkey patches for torch_rbln.

This module contains all monkey patches applied to PyTorch to enable RBLN functionality.
Patches are organized by functionality and include proper error handling and idempotency checks.
"""

import warnings

from torch_rbln._internal.torch_compile_patch_helpers import CompiledFunctionWrapper, is_rbln_backend


# Module-level state to track if patches have been applied
_torch_compile_patched: bool = False
_rbln_backend_registered: bool = False
_get_torch_hash_patched: bool = False


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
    global _torch_compile_patched

    if _torch_compile_patched:
        return

    import torch

    # Store original function
    original_torch_compile = torch.compile

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


def patch_get_torch_hash() -> None:
    """Force ``rebel.compilation.get_torch_hash`` to include graph structure.

    Without ``include_graph=True`` the hash signature for parameter-less
    fx.GraphModules is identical (``"no-param" + "no-buffer" + "" + version +
    "class:GraphModule" + "tp:N"``). Distinct ops (e.g. eager ``mul int32``
    triggered from torch-rbln's register_ops vs the vllm sampler's
    ``softmax + top_k_top_p`` graph) collide on the same ``mod_name`` under
    the shared CompileContext + ``use_weight_sharing=True``. The collision
    surfaces as ``Graph Generation: [DEVICE_GRAPH_CONVERSION]`` at the
    second compile (FINE-542 / WeightReusabilityCheck).

    This patch wraps ``rebel.core.compilation._impl.get_torch_hash`` and
    sets ``include_graph=True`` so the forward code becomes part of the
    signature, giving distinct graphs distinct ``mod_name``s.

    Patches all three module bindings (``_impl``, ``compile_from_any``,
    ``core.compilation``) since some import the symbol by name (the import
    locks an early reference that wouldn't see a single-module patch).
    """
    global _get_torch_hash_patched
    if _get_torch_hash_patched:
        return

    try:
        import rebel.compile_from_any as _cfa
        import rebel.core.compilation as _rc
        import rebel.core.compilation._impl as _impl
    except ImportError:
        return

    original = _impl.get_torch_hash

    def patched(mod, tensor_parallel_size=None):
        from rebel.core.compilation._torch_hash import TorchModelHasher
        meta = f"tp:{tensor_parallel_size}" if tensor_parallel_size is not None else None
        digest = TorchModelHasher().get_model_hash(
            mod, include_param=True, include_graph=True, meta=meta,
        )
        return digest[:6]

    patched.__wrapped__ = original  # type: ignore[attr-defined]
    for _mod in (_impl, _cfa, _rc):
        if getattr(_mod, "get_torch_hash", None) is original:
            _mod.get_torch_hash = patched

    _get_torch_hash_patched = True


def apply_all_patches() -> None:
    """
    Apply all monkey patches for RBLN functionality.

    This function applies patches in the correct order:
    1. torch.compile() patch
    2. rebel get_torch_hash include-graph fix (FINE-542 workaround)

    Idempotent: Safe to call multiple times.
    """
    patch_torch_compile()
    patch_get_torch_hash()


def remove_all_patches() -> None:
    """
    Remove all monkey patches (restore original behavior).

    WARNING: This function is primarily for testing purposes. Removing patches
    in production code may cause unexpected behavior.

    Note: This function only resets the patch flags. The actual patches remain
    applied. To fully restore, you would need to reload the module.
    """
    global _torch_compile_patched, _get_torch_hash_patched

    # Reset state flags (actual patches remain applied)
    _torch_compile_patched = False
    _get_torch_hash_patched = False
