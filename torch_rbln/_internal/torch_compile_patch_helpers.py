"""
Helper functions and classes for torch.compile patching.

This module contains utilities for wrapping compiled functions with num_devices
auto-determination, tensor-parallel failover support, and CPU fallback functionality.
"""

import threading

import torch


try:
    from torch._dynamo.utils import get_chromium_event_logger as _get_chromium_event_logger
except Exception:  # pragma: no cover - torch internals may move
    _get_chromium_event_logger = None

from torch_rbln._internal.env_utils import is_fallback_disabled, use_tp_failover
from torch_rbln._internal.log_utils import rbln_log_error, rbln_log_warn
from torch_rbln._internal.ops_utils import extract_device_id_from_inputs, to_cpu
from torch_rbln._internal.rsd_utils import auto_determine_num_devices, get_physical_device_ids


# Thread-local reentrancy guard: when we're already inside an RBLN op that uses torch.compile,
# any nested dispatch (e.g. from compiled graph running torch.add -> add_rbln again, or from
# print/repr of a tensor triggering dispatch) must take CPU fallback to avoid infinite recursion.
_rbln_compile_op_depth = threading.local()


def get_rbln_compile_op_depth() -> int:
    """Return current reentrancy depth (0 = not inside an RBLN compile op)."""
    return getattr(_rbln_compile_op_depth, "depth", 0)


def _enter_rbln_compile_op() -> None:
    _rbln_compile_op_depth.depth = get_rbln_compile_op_depth() + 1


def _exit_rbln_compile_op() -> None:
    d = get_rbln_compile_op_depth()
    _rbln_compile_op_depth.depth = max(0, d - 1)


def _isolate_chromium_event_state():
    """Isolate dynamo's thread-local chromium event state across a nested op-compile.

    RBLN ATen-op fallbacks are ``torch.compile``-d, so dispatching one during an outer
    compile's ``build_guards`` re-enters dynamo; the nested compile's exit resets the
    *shared* chromium event stack, wiping the outer compile's events and crashing it with
    "No toplevel event active". Swap in throwaway containers for the op-compile, then
    restore the originals unconditionally.

    Returns a zero-arg restorer, or ``None`` if there is nothing to protect.
    """
    if _get_chromium_event_logger is None:
        return None
    try:
        log = _get_chromium_event_logger()
        tls = log.tls
        # No outer compile in flight -> tls.stack is absent or empty; bail fast
        # without paying for the get_stack() call (which would also lazily create it).
        stack = getattr(tls, "stack", None)
        if not stack:
            return None
        saved_stack = stack
        saved_substack = tls.pt2_compile_substack
        saved_event_data = tls.event_data
    except Exception:
        return None

    # The nested op-compile's reset_event_log_on_exit now clears these throwaways instead.
    tls.stack = []
    tls.pt2_compile_substack = []
    tls.event_data = {}

    def _restore():
        try:
            tls.stack = saved_stack
            tls.pt2_compile_substack = saved_substack
            tls.event_data = saved_event_data
        except Exception:
            pass

    return _restore


def is_recompile_limit_exception(exception):
    """Check if exception is FailOnRecompileLimitHit."""
    try:
        import torch._dynamo.exc as dynamo_exc

        return isinstance(exception, dynamo_exc.FailOnRecompileLimitHit)
    except ImportError:
        # If dynamo.exc is not available, check by exception name
        return type(exception).__name__ == "FailOnRecompileLimitHit"


def extract_device_from_inputs(*args, **kwargs):
    """Extract the original output device from RBLN tensor inputs."""
    device_id = extract_device_id_from_inputs(*args, **kwargs)
    if device_id is None:
        raise RuntimeError("RBLN CPU fallback requires at least one RBLN tensor input.")
    return torch.device("rbln", device_id)


def _convert_result_to_device(result, target_device):
    """Recursively convert result containers back to the target device."""
    if isinstance(result, torch.Tensor):
        return result.to(target_device)
    elif isinstance(result, dict):
        return {key: _convert_result_to_device(value, target_device) for key, value in result.items()}
    elif isinstance(result, tuple):
        return tuple(_convert_result_to_device(item, target_device) for item in result)
    elif isinstance(result, list):
        return [_convert_result_to_device(item, target_device) for item in result]
    return result


def attempt_cpu_fallback(original_fn, args, kwargs, original_device):
    """Attempt to execute original function on CPU with fallback."""
    # Execute original function on CPU instead of compiled function
    cpu_args = to_cpu(args)
    cpu_kwargs = to_cpu(kwargs)
    if original_fn is None:
        raise ValueError("original_fn is not provided")
    result = original_fn(*cpu_args, **cpu_kwargs)

    # Move result back to original device if needed
    if original_device and original_device.type != "cpu":
        result = _convert_result_to_device(result, original_device)
    return result


def recompile_with_num_devices(model, compile_kwargs, num_devices, original_compile_fn):
    """Recompile model with the specified ``num_devices``."""
    recompile_kwargs = compile_kwargs.copy()
    recompile_options = recompile_kwargs.get("options", {})
    if isinstance(recompile_options, dict):
        recompile_options = recompile_options.copy()
    else:
        recompile_options = {}
    recompile_options["num_devices"] = num_devices
    recompile_kwargs["options"] = recompile_options
    return original_compile_fn(model, **recompile_kwargs)


def get_num_devices_from_options(compile_kwargs):
    """Read the caller-pinned device count from compile options.

    Used only to detect whether the caller explicitly set a device count, so
    auto-determination and failover do not override an explicit choice. Both
    ``num_devices`` and the legacy ``tensor_parallel_size`` are recognized
    (``num_devices`` wins); canonicalizing the option for the backend is the
    backend's responsibility, not this function's.
    """
    compile_options = compile_kwargs.get("options", {})
    if not isinstance(compile_options, dict):
        return None
    num_devices = compile_options.get("num_devices")
    if num_devices is None:
        num_devices = compile_options.get("tensor_parallel_size")
    return num_devices


def _is_compile_only(compile_kwargs) -> bool:
    """Whether torch.compile requested mode=compile_only (build artifacts, no run)."""
    options = compile_kwargs.get("options", {})
    if not isinstance(options, dict):
        return False
    mode = options.get("mode")
    if isinstance(mode, (list, tuple)):
        return "compile_only" in mode
    return mode == "compile_only"


def _resolve_current_num_devices(device_id, compile_kwargs):
    """Resolve the ``num_devices`` currently in use for this compiled function.

    An explicit ``options.num_devices`` (or the legacy ``tensor_parallel_size``)
    always wins. Otherwise we fall back to the topology-derived auto-determined
    device count.
    """
    explicit_num_devices = get_num_devices_from_options(compile_kwargs)
    if explicit_num_devices is not None:
        return explicit_num_devices
    return auto_determine_num_devices(device_id)


def auto_determine_num_devices_if_needed(model, compile_kwargs, device_id, original_compile_fn):
    """Auto-determine ``num_devices`` if it's None in compile_kwargs.

    This function checks if ``num_devices`` is explicitly set in compile_kwargs.
    If not, it automatically determines the device count based on the RSD device
    topology (RBLN_NPUS_PER_DEVICE or RBLN_DEVICE_MAP environment variables).

    Args:
        model: The model to compile.
        compile_kwargs: Keyword arguments passed to torch.compile.
        device_id: The RBLN logical device ID.
        original_compile_fn: The original torch.compile function.

    Returns:
        Compiled function with auto-determined ``num_devices``, or None if:
        - ``num_devices`` is already explicitly set
        - Auto-determination fails
    """
    num_devices = get_num_devices_from_options(compile_kwargs)
    if num_devices is not None:
        return None  # Already set, no need to auto-determine

    auto_num_devices = auto_determine_num_devices(device_id)
    if auto_num_devices is None:
        return None  # Cannot determine

    try:
        return recompile_with_num_devices(model, compile_kwargs, auto_num_devices, original_compile_fn)
    except Exception:
        # If recompilation fails, return None to use original compiled_fn
        return None


def should_attempt_failover(device_id, compile_kwargs, current_num_devices):
    """Check if failover should be attempted.

    Failover is attempted when:
    - TORCH_RBLN_USE_TP_FAILOVER=ON
    - ``num_devices`` was not explicitly set by the caller
    - current_num_devices > 1

    Args:
        device_id: The RBLN logical device ID.
        compile_kwargs: Keyword arguments passed to torch.compile.
        current_num_devices: Current number of devices the model is distributed across.

    Returns:
        True if failover should be attempted, False otherwise.
    """
    if not use_tp_failover():
        return False

    # Respect an explicitly requested num_devices. Silent failover is only allowed
    # for topology-driven auto device counts, not for caller-selected configurations.
    if get_num_devices_from_options(compile_kwargs) is not None:
        return False

    if current_num_devices is None or current_num_devices <= 1:
        return False  # No need to failover

    return True


def handle_tp_failover(model, compile_kwargs, device_id, original_compile_fn):
    """Handle tensor parallel failover by retrying with num_devices=1.

    When a RuntimeError occurs during execution with num_devices > 1,
    this function attempts to recompile the model with num_devices=1 as a fallback.

    This is useful for models that don't support tensor parallelism, allowing
    them to run on a single NPU within the device group.

    Args:
        model: The model to compile.
        compile_kwargs: Keyword arguments passed to torch.compile.
        device_id: The RBLN logical device ID.
        original_compile_fn: The original torch.compile function.

    Returns:
        Compiled function with num_devices=1 (failover), or None if failover is not applicable
        or recompilation fails (caller will then try CPU fallback or re-raise).
    """
    # Determine the num_devices that the current compiled_fn is actually using.
    current_num_devices = _resolve_current_num_devices(device_id, compile_kwargs)

    if not should_attempt_failover(device_id, compile_kwargs, current_num_devices):
        return None

    # Log the failover attempt
    physical_device_ids = get_physical_device_ids(device_id)
    if physical_device_ids:
        model_name = getattr(model, "__name__", str(model))
        rbln_log_warn(
            f"Model '{model_name}' unsupported with num_devices={current_num_devices}. "
            f"Retrying with num_devices=1 on root device (NPU {physical_device_ids[0]})."
        )

    # Recompile with num_devices=1
    try:
        return recompile_with_num_devices(model, compile_kwargs, 1, original_compile_fn)
    except Exception:
        # If recompilation fails, return None to re-raise original error
        return None


class CompiledFunctionWrapper:
    """Wrapper for compiled functions with TP auto-determination and failover support.

    This wrapper provides the following features:

    1. **TP Auto-Determination**: Automatically determines num_devices based on
       RSD device topology (RBLN_NPUS_PER_DEVICE or RBLN_DEVICE_MAP) if not explicitly set.

    2. **TP Failover**: When TORCH_RBLN_USE_TP_FAILOVER=ON and a RuntimeError occurs
       with num_devices > 1, automatically retries with num_devices=1 on the root NPU.

    3. **CPU Fallback**: Falls back to CPU execution when compilation fails (in non-debug mode).

    4. **Recompile Limit Handling**: Handles FailOnRecompileLimitHit by resetting dynamo
       and retrying.

    Args:
        compiled_fn: The compiled function from torch.compile.
        original_fn: The original uncompiled function (for CPU fallback and TP recompilation).
        original_compile_fn: The original torch.compile function.
        compile_kwargs: Keyword arguments passed to torch.compile.

    """

    def __init__(self, compiled_fn, original_fn, original_compile_fn, compile_kwargs=None):
        self._compiled_fn = compiled_fn
        self._original_fn = original_fn
        self._original_compile_fn = original_compile_fn
        self._compile_kwargs = compile_kwargs or {}
        self._max_retries = 1
        self._auto_num_devices_determined = False
        self._failover_attempted = False

    def _try_tp_failover(self, device_id):
        """Try tensor parallel failover on RuntimeError."""
        if self._failover_attempted:
            return None

        failover_compiled_fn = handle_tp_failover(
            self._original_fn,
            self._compile_kwargs,
            device_id,
            self._original_compile_fn,
        )
        if failover_compiled_fn is not None:
            self._compiled_fn = failover_compiled_fn
            self._failover_attempted = True
        return failover_compiled_fn

    def _attempt_cpu_fallback_or_raise(self, error, args, kwargs):
        """Attempt CPU fallback or re-raise error based on fallback configuration."""
        if is_fallback_disabled("compile_error"):
            rbln_log_error(
                "CPU fallback for compilation failure is disabled: "
                "`TORCH_RBLN_DISABLE_FALLBACK` contains 'compile_error' or 'all'."
            )
            raise error

        rbln_log_warn(
            f"{error}.\n"
            "Fallback to CPU execution due to RBLN compilation failure. "
            "The operation will now proceed on the CPU using PyTorch. "
            "Performance may be impacted."
        )
        original_device = extract_device_from_inputs(*args, **kwargs)
        return attempt_cpu_fallback(self._original_fn, args, kwargs, original_device)

    def _handle_runtime_error(self, error, device_id, args, kwargs):
        """Handle RuntimeError with potential TP failover."""
        # Try TP failover first if not already attempted
        if not self._failover_attempted:
            failover_compiled_fn = self._try_tp_failover(device_id)
            if failover_compiled_fn is not None:
                # Signal to retry with failover-compiled function
                return None

        # Failover failed or already attempted, try CPU fallback
        return self._attempt_cpu_fallback_or_raise(error, args, kwargs)

    def _handle_compile_exception(self, error, args, kwargs, *, is_recompile_limit, attempt):
        """Non-RuntimeError path: recompile-limit retries, else CPU fallback or re-raise."""
        if is_recompile_limit:
            torch._dynamo.reset()
            if attempt < self._max_retries:
                return None  # retry same compiled_fn after dynamo reset
        return self._attempt_cpu_fallback_or_raise(error, args, kwargs)

    def __call__(self, *args, **kwargs):
        """Execute the compiled function with reentrancy guard, TP auto-determination and failover."""
        _enter_rbln_compile_op()
        # Isolate an outer compile's chromium event stack from this op-compile's reset.
        restore_chromium = _isolate_chromium_event_state()
        try:
            return self._call_impl(*args, **kwargs)
        finally:
            if restore_chromium is not None:
                restore_chromium()
            _exit_rbln_compile_op()

    def _call_impl(self, *args, **kwargs):
        # RBLN_DUMMY_DEVICE is compile-only (no NPU): a compiled graph cannot be
        # executed on it — the dummy runtime would silently return zeros. Fail
        # loudly here instead. compile_only builds are allowed through (they only
        # write the .rbln artifact; the zero output is never used).
        import torch_rbln._C

        if torch_rbln._C.is_dummy_device() and not _is_compile_only(self._compile_kwargs):
            raise RuntimeError(
                "Cannot execute a compiled graph on RBLN_DUMMY_DEVICE (no NPU). "
                "Use options={'mode': ['compile_only']} to build artifacts, or run "
                "on a host with a real NPU."
            )

        # Extract device_id for TP operations
        device_id = extract_device_id_from_inputs(*args, **kwargs)

        # Auto-determine num_devices if needed (only once)
        if not self._auto_num_devices_determined:
            compiled_fn_with_auto_num_devices = auto_determine_num_devices_if_needed(
                self._original_fn, self._compile_kwargs, device_id, self._original_compile_fn
            )
            if compiled_fn_with_auto_num_devices is not None:
                self._compiled_fn = compiled_fn_with_auto_num_devices
            self._auto_num_devices_determined = True

        for attempt in range(self._max_retries + 1):
            try:
                return self._compiled_fn(*args, **kwargs)
            except RuntimeError as e:
                result = self._handle_runtime_error(e, device_id, args, kwargs)
                if result is None:
                    continue
                return result
            except Exception as e:
                result = self._handle_compile_exception(
                    e,
                    args,
                    kwargs,
                    is_recompile_limit=is_recompile_limit_exception(e),
                    attempt=attempt,
                )
                if result is None:
                    continue
                return result


def is_rbln_backend(backend):
    """Check if backend is RBLN backend."""
    return backend == "rbln" or (callable(backend) and getattr(backend, "__name__", None) == "rbln_backend")
