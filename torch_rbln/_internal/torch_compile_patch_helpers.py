"""
Helper functions and classes for torch.compile patching.

This module contains utilities for wrapping compiled functions with tensor parallel
auto-determination, failover support, and CPU fallback functionality.
"""

import inspect
import threading
import time

import torch

from torch_rbln._internal.env_utils import is_fallback_disabled, use_tp_failover
from torch_rbln._internal.log_utils import rbln_log_error, rbln_log_warn
from torch_rbln._internal.ops_utils import estimate_tensor_nbytes, extract_device_id_from_inputs, to_cpu
from torch_rbln._internal.profiling import (
    begin_dynamo_runtime_call_observation,
    end_dynamo_runtime_call_observation,
    profile_call_context,
    profile_phase,
    record_compile_cause,
    record_counter,
    record_dynamo_state_delta,
    record_shape_signature,
    record_transfer_bytes,
    snapshot_dynamo_state,
)
from torch_rbln._internal.rsd_utils import auto_determine_tensor_parallel_size, get_physical_device_ids


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


def _iter_named_tensors(name, value):
    if isinstance(value, torch.Tensor):
        yield name, value
        return
    if isinstance(value, dict):
        for key, nested in value.items():
            child_name = f"{name}.{key}" if name else str(key)
            yield from _iter_named_tensors(child_name, nested)
        return
    if isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            child_name = f"{name}[{index}]" if name else f"[{index}]"
            yield from _iter_named_tensors(child_name, nested)


def _bind_named_arguments(original_fn, args, kwargs):
    bound_arguments = {}
    target = getattr(original_fn, "forward", original_fn)
    try:
        signature = inspect.signature(target)
        bound_arguments.update(signature.bind_partial(*args, **kwargs).arguments)
    except Exception:
        pass

    for index, value in enumerate(args):
        bound_arguments.setdefault(f"arg{index}", value)
    for key, value in kwargs.items():
        bound_arguments.setdefault(key, value)
    return bound_arguments


def _pick_preferred_tensor(named_tensors, keywords, *, min_ndim=1):
    lowered_keywords = tuple(keyword.lower() for keyword in keywords)
    for keyword in lowered_keywords:
        for name, tensor in named_tensors:
            if tensor.ndim >= min_ndim and keyword in name.lower():
                return tensor
    for _, tensor in named_tensors:
        if tensor.ndim >= min_ndim:
            return tensor
    return None


def _infer_shape_signature(profile_name, original_fn, args, kwargs):
    bound_arguments = _bind_named_arguments(original_fn, args, kwargs)
    named_tensors = []
    for name, value in bound_arguments.items():
        named_tensors.extend(_iter_named_tensors(name, value))

    batch_tensor = _pick_preferred_tensor(
        named_tensors,
        ("input_ids", "position_ids", "positions", "hidden_states", "logits", "tokens", "seq", "query"),
        min_ndim=1,
    )
    seq_tensor = _pick_preferred_tensor(
        named_tensors,
        ("input_ids", "position_ids", "positions", "hidden_states", "logits", "seq", "query"),
        min_ndim=2,
    )

    batch = batch_tensor.shape[0] if batch_tensor is not None and batch_tensor.ndim >= 1 else None
    seq_len = seq_tensor.shape[1] if seq_tensor is not None and seq_tensor.ndim >= 2 else None

    if seq_len is None and batch_tensor is not None and batch_tensor.ndim == 1 and batch == 1:
        seq_len = batch_tensor.shape[0]

    lowered_name = str(profile_name).lower()
    if "prefill" in lowered_name:
        phase = "prefill"
    elif "decode" in lowered_name:
        phase = "decode"
    elif seq_len is not None:
        phase = "decode" if seq_len <= 1 else "prefill"
    else:
        phase = "unknown"

    batch_text = "?" if batch is None else str(batch)
    seq_text = "?" if seq_len is None else str(seq_len)
    return f"phase={phase} batch={batch_text} seq={seq_text}"


def attempt_cpu_fallback(original_fn, args, kwargs, original_device):
    """Attempt to execute original function on CPU with fallback."""
    with profile_phase("compile_wrapper.attempt_cpu_fallback"):
        # Execute original function on CPU instead of compiled function
        with profile_phase("compile_wrapper.cpu_fallback.to_cpu"):
            to_cpu_bytes = estimate_tensor_nbytes(args, predicate=lambda tensor: tensor.device.type != "cpu")
            to_cpu_bytes += estimate_tensor_nbytes(kwargs, predicate=lambda tensor: tensor.device.type != "cpu")
            cpu_args = to_cpu(args)
            cpu_kwargs = to_cpu(kwargs)
        record_transfer_bytes("compile_wrapper.cpu_fallback.to_cpu", to_cpu_bytes)
        if original_fn is None:
            raise ValueError("original_fn is not provided")
        with profile_phase("compile_wrapper.cpu_fallback.exec_original"):
            result = original_fn(*cpu_args, **cpu_kwargs)

        # Move result back to original device if needed
        if original_device and original_device.type != "cpu":
            with profile_phase("compile_wrapper.cpu_fallback.return_to_device"):
                record_transfer_bytes(
                    "compile_wrapper.cpu_fallback.return_to_device",
                    estimate_tensor_nbytes(result, predicate=lambda tensor: tensor.device.type == "cpu"),
                )
                result = _convert_result_to_device(result, original_device)
        return result


def recompile_with_tp_size(model, compile_kwargs, tp_size, original_compile_fn):
    """Recompile model with specified tensor_parallel_size."""
    with profile_phase("compile_wrapper.recompile_with_tp_size"):
        recompile_kwargs = compile_kwargs.copy()
        recompile_options = recompile_kwargs.get("options", {})
        if isinstance(recompile_options, dict):
            recompile_options = recompile_options.copy()
        else:
            recompile_options = {}
        recompile_options["tensor_parallel_size"] = tp_size
        recompile_kwargs["options"] = recompile_options
        return original_compile_fn(model, **recompile_kwargs)


def get_tensor_parallel_size_from_options(compile_kwargs):
    """Extract tensor_parallel_size from compile options."""
    compile_options = compile_kwargs.get("options", {})
    if not isinstance(compile_options, dict):
        return None
    return compile_options.get("tensor_parallel_size")


def _resolve_current_tensor_parallel_size(device_id, compile_kwargs):
    """Resolve the TP size currently in use for this compiled function.

    An explicit ``options.tensor_parallel_size`` always wins. Otherwise we fall
    back to the topology-derived auto-determined TP size.
    """
    explicit_tp_size = get_tensor_parallel_size_from_options(compile_kwargs)
    if explicit_tp_size is not None:
        return explicit_tp_size
    return auto_determine_tensor_parallel_size(device_id)


def auto_determine_tp_if_needed(model, compile_kwargs, device_id, original_compile_fn):
    """Auto-determine tensor_parallel_size if it's None in compile_kwargs.

    This function checks if tensor_parallel_size is explicitly set in compile_kwargs.
    If not, it automatically determines the TP size based on the RSD device topology
    (RBLN_NPUS_PER_DEVICE or RBLN_DEVICE_MAP environment variables).

    Args:
        model: The model to compile.
        compile_kwargs: Keyword arguments passed to torch.compile.
        device_id: The RBLN logical device ID.
        original_compile_fn: The original torch.compile function.

    Returns:
        Compiled function with auto-determined TP size, or None if:
        - TP size is already explicitly set
        - Auto-determination fails
    """
    with profile_phase("compile_wrapper.auto_determine_tp_if_needed"):
        tp_size = get_tensor_parallel_size_from_options(compile_kwargs)
        if tp_size is not None:
            return None  # Already set, no need to auto-determine

        auto_tp = auto_determine_tensor_parallel_size(device_id)
        if auto_tp is None:
            return None  # Cannot determine

        try:
            record_counter("compile_wrapper.auto_tp.attempts")
            return recompile_with_tp_size(model, compile_kwargs, auto_tp, original_compile_fn)
        except Exception:
            record_counter("compile_wrapper.auto_tp.failures")
            # If recompilation fails, return None to use original compiled_fn
            return None


def should_attempt_failover(device_id, compile_kwargs, current_tp):
    """Check if failover should be attempted.

    Failover is attempted when:
    - TORCH_RBLN_USE_TP_FAILOVER=ON
    - tensor_parallel_size was not explicitly set by the caller
    - current_tp > 1

    Args:
        device_id: The RBLN logical device ID.
        compile_kwargs: Keyword arguments passed to torch.compile.
        current_tp: Current tensor parallel size.

    Returns:
        True if failover should be attempted, False otherwise.
    """
    if not use_tp_failover():
        return False

    # Respect an explicitly requested TP size. Silent failover is only allowed
    # for topology-driven auto TP, not for caller-selected configurations.
    if get_tensor_parallel_size_from_options(compile_kwargs) is not None:
        return False

    if current_tp is None or current_tp <= 1:
        return False  # No need to failover

    return True


def handle_tp_failover(model, compile_kwargs, device_id, original_compile_fn):
    """Handle tensor parallel failover by retrying with tp=1.

    When a RuntimeError occurs during execution with tensor parallel size > 1,
    this function attempts to recompile the model with tp_size=1 as a fallback.

    This is useful for models that don't support tensor parallelism, allowing
    them to run on a single NPU within the device group.

    Args:
        model: The model to compile.
        compile_kwargs: Keyword arguments passed to torch.compile.
        device_id: The RBLN logical device ID.
        original_compile_fn: The original torch.compile function.

    Returns:
        Compiled function with tp=1 (failover), or None if failover is not applicable
        or recompilation fails (caller will then try CPU fallback or re-raise).
    """
    with profile_phase("compile_wrapper.handle_tp_failover"):
        # Determine the TP size that the current compiled_fn is actually using.
        current_tp = _resolve_current_tensor_parallel_size(device_id, compile_kwargs)

        if not should_attempt_failover(device_id, compile_kwargs, current_tp):
            return None

        record_counter("compile_wrapper.tp_failover.attempts")

        # Log the failover attempt
        physical_device_ids = get_physical_device_ids(device_id)
        if physical_device_ids:
            model_name = getattr(model, "__name__", str(model))
            rbln_log_warn(
                f"Model '{model_name}' unsupported with tp={current_tp}. "
                f"Retrying with tp=1 on root device (NPU {physical_device_ids[0]})."
            )

        # Recompile with tp=1
        try:
            return recompile_with_tp_size(model, compile_kwargs, 1, original_compile_fn)
        except Exception:
            record_counter("compile_wrapper.tp_failover.failures")
            # If recompilation fails, return None to re-raise original error
            return None


class CompiledFunctionWrapper:
    """Wrapper for compiled functions with TP auto-determination and failover support.

    This wrapper provides the following features:

    1. **TP Auto-Determination**: Automatically determines tensor_parallel_size based on
       RSD device topology (RBLN_NPUS_PER_DEVICE or RBLN_DEVICE_MAP) if not explicitly set.

    2. **TP Failover**: When TORCH_RBLN_USE_TP_FAILOVER=ON and a RuntimeError occurs
       with tp > 1, automatically retries with tp=1 on the root NPU.

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
        self._auto_tp_determined = False
        self._failover_attempted = False
        self._profile_name = getattr(original_fn, "__name__", type(original_fn).__name__)
        self._pending_compile_cause = None
        self._pending_compile_overhead_ns = 0
        self._observed_compile_events = 0

    def _set_pending_compile_cause(self, cause, orchestration_ns=0):
        self._pending_compile_cause = cause
        self._pending_compile_overhead_ns += max(0, orchestration_ns)

    def _record_compile_cause_if_needed(self, delta_summary, runtime_observation=None):
        compile_total_ns = int(delta_summary.get("compile_total_ns", 0))
        if compile_total_ns <= 0:
            return

        cause = self._pending_compile_cause
        cache_hit = None if runtime_observation is None else runtime_observation.get("cache_hit")
        if cause is None:
            if cache_hit is False or int(delta_summary.get("guard_failure_count", 0)) > 0:
                cause = "guard_miss"
            elif self._observed_compile_events == 0:
                cause = "initial_compile"
            else:
                cause = "other_recompile"

        record_compile_cause(cause, compile_total_ns + self._pending_compile_overhead_ns)
        self._observed_compile_events += 1
        self._pending_compile_cause = None
        self._pending_compile_overhead_ns = 0

    def _try_tp_failover(self, device_id):
        """Try tensor parallel failover on RuntimeError."""
        if self._failover_attempted:
            return None

        start_ns = time.perf_counter_ns()
        failover_compiled_fn = handle_tp_failover(
            self._original_fn,
            self._compile_kwargs,
            device_id,
            self._original_compile_fn,
        )
        elapsed_ns = time.perf_counter_ns() - start_ns
        if failover_compiled_fn is not None:
            self._compiled_fn = failover_compiled_fn
            self._failover_attempted = True
            self._set_pending_compile_cause("failover_recompile", elapsed_ns)
        return failover_compiled_fn

    def _attempt_cpu_fallback_or_raise(self, error, args, kwargs):
        """Attempt CPU fallback or re-raise error based on fallback configuration."""
        if is_fallback_disabled("compile_error"):
            rbln_log_error(
                "CPU fallback for compilation failure is disabled: "
                "`TORCH_RBLN_DISABLE_FALLBACK` contains 'compile_error' or 'all'."
            )
            raise error

        record_counter("compile_wrapper.cpu_fallback.calls")
        with profile_phase("compile_wrapper.cpu_fallback"):
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
        with profile_call_context(self._profile_name, "compiled_fn", allow_nested=True):
            record_shape_signature(_infer_shape_signature(self._profile_name, self._original_fn, args, kwargs))
            _enter_rbln_compile_op()
            try:
                return self._call_impl(*args, **kwargs)
            finally:
                _exit_rbln_compile_op()

    def _call_impl(self, *args, **kwargs):
        # Extract device_id for TP operations
        device_id = extract_device_id_from_inputs(*args, **kwargs)

        # Auto-determine tensor_parallel_size if needed (only once)
        if not self._auto_tp_determined:
            auto_tp_start_ns = time.perf_counter_ns()
            with profile_phase("compile_wrapper.auto_tp_resolution"):
                compiled_fn_with_auto_tp = auto_determine_tp_if_needed(
                    self._original_fn, self._compile_kwargs, device_id, self._original_compile_fn
                )
            auto_tp_elapsed_ns = time.perf_counter_ns() - auto_tp_start_ns
            if compiled_fn_with_auto_tp is not None:
                self._compiled_fn = compiled_fn_with_auto_tp
                self._set_pending_compile_cause("auto_tp_recompile", auto_tp_elapsed_ns)
            self._auto_tp_determined = True

        for attempt in range(self._max_retries + 1):
            try:
                before_dynamo_state = snapshot_dynamo_state()
                begin_dynamo_runtime_call_observation()
                try:
                    with profile_phase("compile_wrapper.compiled_call"):
                        return self._compiled_fn(*args, **kwargs)
                finally:
                    runtime_observation = end_dynamo_runtime_call_observation()
                    after_dynamo_state = snapshot_dynamo_state()
                    delta_summary = record_dynamo_state_delta(before_dynamo_state, after_dynamo_state)
                    self._record_compile_cause_if_needed(delta_summary, runtime_observation)
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
