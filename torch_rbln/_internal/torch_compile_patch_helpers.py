"""
Helper functions and classes for torch.compile patching.

This module contains utilities for wrapping compiled functions with CPU fallback functionality.
"""

import functools
import threading
import types

import torch


try:
    from torch._dynamo.utils import get_chromium_event_logger as _get_chromium_event_logger
except Exception:  # pragma: no cover - torch internals may move
    _get_chromium_event_logger = None

from torch_rbln._internal.dummy_device import raise_if_dummy_execution
from torch_rbln._internal.env_utils import is_fallback_disabled, raised_for_disabled_fallback
from torch_rbln._internal.log_utils import rbln_log_error, rbln_log_warn
from torch_rbln._internal.ops_utils import extract_device_id_from_inputs, to_cpu


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


def _copies(given, copied):
    """The pairs of a tensor in `given` and its CPU copy in `copied`, which ``to_cpu`` made."""
    if isinstance(given, torch.Tensor):
        return [(given, copied)] if copied is not given else []
    if isinstance(given, (list, tuple)):
        return [pair for g, c in zip(given, copied) for pair in _copies(g, c)]
    if isinstance(given, dict):
        return [pair for key in given for pair in _copies(given[key], copied[key])]
    return []


def attempt_cpu_fallback(original_fn, args, kwargs, original_device):
    """Runs the original function on CPU copies of the tensors, and writes what it updates in
    place back into the tensors it was given."""
    cpu_args = to_cpu(args)
    cpu_kwargs = to_cpu(kwargs)
    if original_fn is None:
        raise ValueError("original_fn is not provided")
    copies = _copies(args, cpu_args) + _copies(kwargs, cpu_kwargs)
    versions = [copy._version for _, copy in copies]
    result = original_fn(*cpu_args, **cpu_kwargs)
    for (given, copy), version in zip(copies, versions):
        if copy._version != version:
            given.copy_(copy)

    # Move result back to original device if needed
    if original_device and original_device.type != "cpu":
        result = _convert_result_to_device(result, original_device)
    return result


def _is_compile_only(compile_kwargs) -> bool:
    """Whether torch.compile requested mode=compile_only (build artifacts, no run)."""
    options = compile_kwargs.get("options", {})
    if not isinstance(options, dict):
        return False
    mode = options.get("mode")
    if isinstance(mode, (list, tuple)):
        return "compile_only" in mode
    return mode == "compile_only"


class CompiledFunctionWrapper:
    """Wrapper for compiled functions with CPU fallback support.

    This wrapper provides the following features:

    1. **CPU Fallback**: Falls back to CPU execution when compilation fails (in non-debug mode).

    2. **Recompile Limit Handling**: Handles FailOnRecompileLimitHit by resetting dynamo
       and retrying.

    Args:
        compiled_fn: The compiled function from torch.compile.
        original_fn: The original uncompiled function (for CPU fallback).
        compile_kwargs: Keyword arguments passed to torch.compile.

    """

    def __init__(self, compiled_fn, original_fn, compile_kwargs=None):
        # object.__setattr__ so __setattr__'s delegation (which reads _original_fn) isn't
        # triggered before the fields exist. Each field lands in __dict__ and is thereafter
        # treated as the wrapper's own -- no hand-maintained field list to keep in sync.
        object.__setattr__(self, "_compiled_fn", compiled_fn)
        object.__setattr__(self, "_original_fn", original_fn)
        object.__setattr__(self, "_compile_kwargs", compile_kwargs or {})
        object.__setattr__(self, "_max_retries", 1)

    def _is_own_attr(self, name):
        # Own = an existing instance field or anything the wrapper class defines (methods,
        # `forward`). Keeps e.g. patch.object(wrapper, "_attempt_cpu_fallback_or_raise")
        # shadowing the method here rather than leaking onto the model.
        return name in self.__dict__ or hasattr(type(self), name)

    def __setattr__(self, name, value):
        # Own state stays on the wrapper; everything else is delegated to the wrapped model,
        # but only an nn.Module (which has training/param/buffer state to keep in sync). A
        # plain-function target has none, so keep the attr on the wrapper.
        if self._is_own_attr(name):
            object.__setattr__(self, name, value)
        elif isinstance(self._original_fn, torch.nn.Module):
            setattr(self._original_fn, name, value)
        else:
            object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if self._is_own_attr(name):
            object.__delattr__(self, name)
        elif isinstance(self._original_fn, torch.nn.Module):
            delattr(self._original_fn, name)
        else:
            object.__delattr__(self, name)

    def __getattr__(self, name):
        # Fires only for attributes missing on the wrapper: delegate reads to the wrapped
        # model/callable (state_dict/eval/parameters/...). Dunders are excluded so
        # copy/pickle protocol lookups aren't redirected.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        model = object.__getattribute__(self, "_original_fn")
        attr = getattr(model, name)
        # Re-wrap a bound method that returns the model (eval/train/to/...) so
        # `compiled = compiled.eval()` keeps the wrapper, not the bare model.
        if isinstance(attr, types.MethodType):

            @functools.wraps(attr)
            def keep_wrapper(*args, **kwargs):
                result = attr(*args, **kwargs)
                return self if result is model else result

            return keep_wrapper
        return attr

    def __get__(self, obj, objtype=None):
        # Descriptor: bind the owner only for a decorated *function* target (the usual
        # `@torch.compile` on a method). A compiled nn.Module / callable object -- or an
        # already-bound method -- is returned unchanged, so the owner is not injected as a
        # spurious first argument (and a bound method is not bound a second time).
        if obj is None or not isinstance(self._original_fn, types.FunctionType):
            return self
        return functools.partial(self.__call__, obj)

    def forward(self, *args, **kwargs):
        # Route an explicit .forward(...) through the compiled path, like calling the wrapper
        # directly -- else __getattr__ would delegate it to the module and bypass compilation.
        return self(*args, **kwargs)

    def _attempt_cpu_fallback_or_raise(self, error, args, kwargs):
        """Attempt CPU fallback or re-raise error based on fallback configuration."""
        if raised_for_disabled_fallback(error):
            raise error
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

    def _handle_compile_exception(self, error, args, kwargs, *, is_recompile_limit, attempt):
        """Non-RuntimeError path: recompile-limit retries, else CPU fallback or re-raise."""
        if is_recompile_limit:
            torch._dynamo.reset()
            if attempt < self._max_retries:
                return None  # retry same compiled_fn after dynamo reset
        return self._attempt_cpu_fallback_or_raise(error, args, kwargs)

    def __call__(self, *args, **kwargs):
        """Execute the compiled function with reentrancy guard and CPU fallback."""
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
        # A compiled graph is NPU-only; executing it on RBLN_DUMMY_DEVICE would
        # silently return zeros. Reject it through the shared policy gate (compile_only
        # builds are exempt — they only write the artifact).
        raise_if_dummy_execution("a compiled graph", compile_only=_is_compile_only(self._compile_kwargs))

        for attempt in range(self._max_retries + 1):
            try:
                return self._compiled_fn(*args, **kwargs)
            except RuntimeError as e:
                return self._attempt_cpu_fallback_or_raise(e, args, kwargs)
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
