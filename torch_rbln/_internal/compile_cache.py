"""Ops compiled into functions over torch-rbln tensors.

An op is a module called with tensors and other values. It is captured with its
tensors as the function's inputs, in the order ``tree_flatten`` walks the
arguments, and every other value as a constant, and compiled with every input
and result laid out as torch lays out a contiguous tensor (``rebel.v2.LOGICAL``), so
a call's tensors bind to the function as they are, CPU ones by a copy. One the
compiler lays out otherwise goes through the host as the op runs. A tensor
the op holds is state the function encodes as it lays it out. One function serves a
module for each profile of its arguments: the shape and dtype of each tensor and
the value of everything else. A function runs on any device of its NPU kind,
with an executor per device. The process keeps TORCH_RBLN_COMPILED_OPS of them,
letting go of the least recently run, which compiles again when called.
"""

from __future__ import annotations

import functools
import heapq
import threading
from typing import Any

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils._pytree import tree_flatten, tree_unflatten

import torch_rbln._C as _C


_compiled_op_cache_lock = threading.Lock()
_compiled_op_cache: dict[tuple[Any, ...], CompiledOp | _Refused] = {}


class _IdentityKey:
    """Hashable identity wrapper that keeps the referenced object alive."""

    __slots__ = ("value", "_hash")

    def __init__(self, value: Any):
        self.value = value
        self._hash = id(value)

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _IdentityKey) and self.value is other.value


def _profile(value: Any) -> Any:
    """What of an argument its function is compiled for, hashable.

    ``1``, ``1.0`` and ``True`` are tagged apart because they bake into different
    constants; anything unrecognized falls back to ``repr``.
    """
    if isinstance(value, torch.Tensor):
        return ("tensor", tuple(value.shape), value.dtype)
    if value is None or isinstance(value, (bool, int, float, str)):
        return (type(value).__name__, value)
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(_profile(v) for v in value))
    if isinstance(value, dict):
        return ("dict", tuple((k, _profile(v)) for k, v in value.items()))
    if isinstance(value, (torch.dtype, torch.device, torch.memory_format, torch.layout)):
        return (type(value).__name__, str(value))
    return ("repr", repr(value))


class _TensorsIn(torch.nn.Module):
    """``inner`` called with a call's arguments, taking only their tensors."""

    def __init__(self, inner: Any, spec: Any, template: list, slots: list[int]):
        super().__init__()
        self.inner = inner
        self._spec = spec
        self._template = template
        self._slots = slots

    def forward(self, *tensors: torch.Tensor) -> Any:
        flat = list(self._template)
        for slot, tensor in zip(self._slots, tensors):
            flat[slot] = tensor
        args, kwargs = tree_unflatten(flat, self._spec)
        return self.inner(*args, **kwargs)


@functools.cache
def _npu(device_index: int) -> str:
    from torch_rbln.device.device import get_device_name

    return get_device_name(device_index)


def _state(module: torch.nn.Module, graph: Any) -> dict[str, torch.Tensor]:
    """The values of the graph's inputs a call does not pass, on the CPU."""
    held = {**dict(module.named_parameters()), **dict(module.named_buffers())}
    state = {}
    for inp in graph.inputs:
        if inp.kind == "input":
            continue
        value = inp.value if inp.value is not None else held[inp.fqn]
        state[inp.name] = value.detach().cpu()
    return state


class CompiledOp:
    """A module compiled for one profile of its arguments, as a
    ``torch_rbln._C._OpFunction``, and the structure of what it returns."""

    __slots__ = ("function", "out_spec")

    def __init__(self, function: Any, out_spec: Any):
        self.function = function
        self.out_spec = out_spec

    @staticmethod
    def tensors(args: tuple, kwargs: dict) -> list[torch.Tensor]:
        """The call's tensors the function takes, in its order."""
        flat, _ = tree_flatten((args, kwargs))
        return [v for v in flat if isinstance(v, torch.Tensor)]

    def run(self, tensors: list[torch.Tensor], out: list[torch.Tensor] | None = None) -> Any:
        """Runs over ``tensors`` and returns what the module returns; a result
        is written into the tensor of ``out`` given for it when that holds it
        as the function writes it, and into a new tensor otherwise."""
        results = None
        if out:
            results = self.function.run(tensors, out)
        if results is None:
            results = self.function.run(tensors)
        if results is None:
            raise RuntimeError("the tensors of the call do not hold what the op was compiled for")
        return tree_unflatten(results, self.out_spec)


def compile_logical(graph: Any, npu: str, devices: int = 1, cache_dir: Any = None) -> Any:
    """The function of ``graph`` with its inputs and results laid out as torch
    holds contiguous tensors wherever the compiler can; ``_OpFunction`` moves
    the others through the host as it runs."""
    from rebel import v2

    try:
        return v2.api.compile_graph(graph, npu, devices, cache_dir, inputs=v2.LOGICAL, outputs=v2.LOGICAL)
    except v2.UnmetRequest as e:
        return e.function


def _compile(module: Any, args: tuple, kwargs: dict, device_index: int) -> CompiledOp:
    from rebel import v2

    flat, spec = tree_flatten((args, kwargs))
    slots = [i for i, v in enumerate(flat) if isinstance(v, torch.Tensor)]
    template = [None if isinstance(v, torch.Tensor) else v for v in flat]
    wrapper = _TensorsIn(module, spec, template, slots).eval()
    names = [f"arg{i}" for i in range(len(slots))]
    types = {name: v2.TensorType(flat[slot].shape, flat[slot].dtype) for name, slot in zip(names, slots)}
    graph = v2.frontend.capture(wrapper, types)
    fn = compile_logical(graph, _npu(device_index))
    with FakeTensorMode(allow_non_fake_inputs=True):
        _, out_spec = tree_flatten(wrapper(*(torch.empty(flat[slot].shape, dtype=flat[slot].dtype) for slot in slots)))
    return CompiledOp(_C._OpFunction(fn.to_bytes(), names, _state(wrapper, graph)), out_spec)


def device_of(args: tuple, kwargs: dict) -> torch.device:
    """The RBLN device of the call's first RBLN tensor, or the current one."""
    flat, _ = tree_flatten((args, kwargs))
    for v in flat:
        if isinstance(v, torch.Tensor) and v.device.type == "rbln":
            return v.device
    return torch.device("rbln", torch.rbln.current_device())


class UncompilableOp(RuntimeError):
    """The compiler cannot build a function of an op for a profile of its
    arguments; ``unsupported`` when it has no lowering for an op in it."""

    def __init__(self, message: str, unsupported: bool = False):
        super().__init__(message)
        self.unsupported = unsupported


class _Refused:
    """What the cache keeps for a profile the compiler refused."""

    __slots__ = ("error",)

    def __init__(self, error: Exception):
        self.error = error


def compiled_op(module: Any, args: tuple, kwargs: dict, device: torch.device) -> CompiledOp:
    """The function of ``module`` for the profile of ``args`` and ``kwargs``,
    compiled on first use for the NPU of ``device``.

    Raises:
        UncompilableOp: the compiler refused this profile, now or on an earlier call.
    """
    from rebel import v2

    key = (_IdentityKey(module), _npu(device.index), v2.float32_precision(), _profile(args), _profile(kwargs))
    entry = _compiled_op_cache.get(key)
    if entry is None:
        with _compiled_op_cache_lock:
            entry = _compiled_op_cache.get(key)
            if entry is None:
                try:
                    entry = _compile(module, args, kwargs, device.index)
                except (NotImplementedError, ValueError, RuntimeError) as e:
                    entry = _Refused(e)
                _keep(key, entry)
    if isinstance(entry, _Refused):
        raise UncompilableOp(str(entry.error), isinstance(entry.error, v2.frontend.NoLowering)) from entry.error
    return entry


def _last_run(entry: CompiledOp | _Refused) -> int:
    return entry.function.last_run if isinstance(entry, CompiledOp) else 0


def _keep(key: tuple[Any, ...], entry: CompiledOp | _Refused) -> None:
    """Keeps ``entry`` under ``key``, letting go of the least recently run
    entries past TORCH_RBLN_COMPILED_OPS, with what the warm cache holds of
    them. The caller holds ``_compiled_op_cache_lock``."""
    from rebel import v2

    _compiled_op_cache[key] = entry
    excess = len(_compiled_op_cache) - v2.flags.TORCH_RBLN_COMPILED_OPS
    if excess <= 0:
        return
    others = (k for k in _compiled_op_cache if k != key)
    for old in heapq.nsmallest(excess, others, key=lambda k: _last_run(_compiled_op_cache[k])):
        dropped = _compiled_op_cache.pop(old)
        if isinstance(dropped, CompiledOp):
            _C._warmcache_forget(dropped.function)


def run_op(module: Any, *args: Any, **kwargs: Any) -> Any:
    """Runs ``module`` on the device of its tensors, as it would run on the CPU."""
    op = compiled_op(module, args, kwargs, device_of(args, kwargs))
    return op.run(op.tensors(args, kwargs))


def clear_rbln_compile_cache() -> None:
    with _compiled_op_cache_lock:
        _compiled_op_cache.clear()
