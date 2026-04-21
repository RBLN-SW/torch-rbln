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
_dynamo_runtime_patched: bool = False
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


def patch_dynamo_runtime() -> None:
    """Replace ``rebel.sync_runtime.DynamoRuntime.run`` with a slimmer version.

    The upstream implementation does several per-call jobs that are wasteful in
    steady state on the eager dispatch path:

      - ``os.environ.pop(CPU_NUM_THREADS, None)`` is called unconditionally even
        when ``cpu_threads`` was never set (most common case).
      - ``self._runtime_utils.prepare_inputs`` runs Python validity checks
        (dtype/shape match per input) on every call. In deploy mode the caller
        is trusted, so the walk + ``_listify_inputs`` reorder can be skipped.
      - It walks the input list once to bucket cpu/rbln pointers in Python.

    This patch keeps semantics identical for the documented input shapes used
    by torch-rbln (rbln/cpu/meta tensors, no exotic devices) while skipping the
    unnecessary work. Falls back to the original implementation for anything it
    cannot prove safe (so error paths are unchanged).

    Validity skip is gated on ``TORCH_RBLN_DEPLOY=ON`` or
    ``TORCH_RBLN_SKIP_RUNTIME_VALIDITY=1`` and is re-evaluated per call so the
    env var can be toggled at runtime.
    """
    global _dynamo_runtime_patched
    if _dynamo_runtime_patched:
        return

    try:
        import os
        import torch
        from rebel import sync_runtime as _sr
    except ImportError:
        return

    DynamoRuntime = _sr.DynamoRuntime
    eager_execution_helper = _sr.eager_execution_helper
    CPU_NUM_THREADS = _sr.CPU_NUM_THREADS

    original_run = DynamoRuntime.run

    def patched_run(self, *input_args, out=None, **input_kwargs):  # type: ignore[no-redef]
        # Bail out to the original implementation for anything we don't
        # explicitly handle; avoids semantic drift.
        if input_kwargs or out is not None:
            return original_run(self, *input_args, out=out, **input_kwargs)

        cpu_threads = self.cpu_threads
        if cpu_threads is not None and isinstance(cpu_threads, int):
            os.environ[CPU_NUM_THREADS] = str(cpu_threads)
            _need_pop = True
        else:
            _need_pop = False

        eager_helper = eager_execution_helper()

        # Fast input listification + skip validity (deploy mode).
        # Original: prepare_inputs -> _listify_inputs (count check + reorder)
        # + check_input_validity (dtype/shape per input). For positional-only
        # matching arity, this is just list(input_args).
        if len(input_args) == self._num_inputs and (
            os.environ.get("TORCH_RBLN_DEPLOY") == "ON"
            or os.environ.get("TORCH_RBLN_SKIP_RUNTIME_VALIDITY") == "1"
        ):
            inputs = list(input_args)
        else:
            inputs = self._runtime_utils.prepare_inputs(*input_args)

        device_inputs: dict[int, int] = {}
        cpu_inputs: dict[int, int] = {}
        input_rbln_device = None
        for input_index, inp in enumerate(inputs):
            dt = inp.device.type
            if dt == "rbln":
                device_inputs[input_index] = inp.data_ptr()
                input_rbln_device = inp.device
            elif dt == "cpu":
                cpu_inputs[input_index] = inp.data_ptr()
            elif dt == "meta":
                continue
            else:
                # Unsupported device type — let the original raise.
                if _need_pop:
                    os.environ.pop(CPU_NUM_THREADS, None)
                return original_run(self, *input_args)

        self._runtime_handle.prepare_inputs(device_inputs, cpu_inputs)

        outputs = []
        device_outputs: dict[int, int] = {}
        cpu_outputs: dict[int, int] = {}

        eager_outs = eager_helper.out_tensors
        use_eager_out = len(eager_outs) > 0
        eager_iter = iter(eager_outs) if use_eager_out else None

        out_profile = self._output_profile
        for output_index in range(self._num_outputs):
            prof = out_profile[output_index]
            if prof.device == "rbln":
                if use_eager_out:
                    output_tensor = next(eager_iter)
                else:
                    output_tensor = torch.empty(
                        size=prof.shape, dtype=prof.dtype, device=input_rbln_device
                    )
                device_outputs[output_index] = output_tensor.data_ptr()
            else:
                output_tensor = torch.empty(size=prof.shape, dtype=prof.dtype, device="cpu")
                cpu_outputs[output_index] = output_tensor.data_ptr()
            outputs.append(output_tensor)

        self._runtime_handle.prepare_outputs(device_outputs, cpu_outputs)
        self._runtime_handle.run()

        if _need_pop:
            os.environ.pop(CPU_NUM_THREADS, None)

        self._capture_reports_if_needed()
        return outputs

    DynamoRuntime.run = patched_run
    _dynamo_runtime_patched = True


def apply_all_patches() -> None:
    """
    Apply all monkey patches for RBLN functionality.

    This function applies patches in the correct order:
    1. torch.compile() patch
    2. rebel DynamoRuntime jacket trim

    Idempotent: Safe to call multiple times.
    """
    patch_torch_compile()
    patch_dynamo_runtime()


def remove_all_patches() -> None:
    """
    Remove all monkey patches (restore original behavior).

    WARNING: This function is primarily for testing purposes.
    """
    global _rbln_backend_registered, _torch_compile_patched, _torch_dynamo_reset_patched, _dynamo_runtime_patched

    import torch

    if _original_torch_compile is not None:
        torch.compile = _original_torch_compile
    if _original_dynamo_reset is not None:
        torch._dynamo.reset = _original_dynamo_reset

    clear_rbln_compile_cache()
    _torch_compile_patched = False
    _torch_dynamo_reset_patched = False
    _rbln_backend_registered = False
    _dynamo_runtime_patched = False
