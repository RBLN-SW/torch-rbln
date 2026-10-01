"""The backend of ``torch.compile(backend="rbln")`` for graphs over torch-rbln tensors.

Dynamo hands over a graph and the tensors of the call it traced. The graph is captured
with ``rbln.frontend.capture_dynamo`` and compiled for the NPU of those tensors, with every
input a call passes and every result laid out as torch lays out a contiguous tensor
(``rbln.LOGICAL``), so torch-rbln tensors bind in place and CPU ones by a copy. The
module's parameters and buffers are the function's state: they are written to the device
when first seen and again once they change, as another tensor or written in place. A
result the graph leaves on the CPU comes back on the CPU.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils._pytree import tree_flatten

import torch_rbln._C as _C
from torch_rbln import programs
from torch_rbln._internal.compile_cache import compile_logical


_OPTIONS = {"npu", "num_devices", "tensor_parallel_size", "mode", "cache_dir", "disable_logger"}


def _modes(options: dict[str, Any]) -> set[str]:
    mode = options.get("mode")
    if mode is None:
        return set()
    return {mode} if isinstance(mode, str) else set(mode)


def _device_of(example_inputs: list[Any]) -> torch.device:
    for x in example_inputs:
        if isinstance(x, torch.Tensor) and x.device.type == "rbln":
            return x.device
    return torch.device("rbln", torch.rbln.current_device())


def _npu(options: dict[str, Any], device: torch.device) -> str:
    if options.get("npu"):
        return options["npu"]
    if _C.is_dummy_device():
        import rbln

        return rbln.flags.RBLN_FORCE_NPU_NAME or "RBLN-CA25"
    from torch_rbln.device.device import get_device_name

    return get_device_name(device.index)


def _host_results(gm: torch.fx.GraphModule) -> list[bool]:
    """Whether each tensor the graph returns is on the CPU, in order."""
    output = next(n for n in gm.graph.nodes if n.op == "output")
    flat, _ = tree_flatten(output.args[0])
    host = []
    for node in flat:
        value = node.meta.get("example_value") if isinstance(node, torch.fx.Node) else None
        if isinstance(value, torch.Tensor):
            host.append(value.device.type == "cpu")
    return host


class CompiledGraph:
    """Runs a graph on the device of its tensors, taking what the graph takes."""

    def __init__(self, fn: Any, graph: Any, module: torch.nn.Module, host_results: list[bool]):
        self._fn = fn
        by_name = {inp.name: inp for inp in graph.inputs}
        passed = graph.passed if graph.passed is not None else [inp.name for inp in graph.inputs]
        self._roles = [(by_name[name].kind, name) for name in passed]
        self._inputs = [name for kind, name in self._roles if kind == "input"]
        self._held = {
            inp.name: inp.fqn for inp in graph.inputs if inp.kind in ("parameter", "buffer") and inp.name not in passed
        }
        self._constants = {inp.name: inp.value for inp in graph.inputs if inp.kind == "constant"}
        self._module = module
        self._host_results = host_results
        self._function: Any = None
        self._seen: dict[str, tuple[int, int]] = {}

    def _state(self, args: tuple) -> dict[str, torch.Tensor]:
        state = {name: t.detach() for (kind, name), t in zip(self._roles, args) if kind not in ("input", "constant")}
        if self._held:
            held = {**dict(self._module.named_parameters()), **dict(self._module.named_buffers())}
            state.update({name: held[fqn].detach() for name, fqn in self._held.items()})
        return state

    def _write_state(self, state: dict[str, torch.Tensor]) -> None:
        now = {name: (t.data_ptr(), t._version) for name, t in state.items()}
        changed = {name: state[name].cpu() for name, seen in now.items() if self._seen.get(name) != seen}
        if self._function is None:
            self._fn.check_specializations(changed)
            self._function = _C._OpFunction(
                self._fn.to_bytes(), self._inputs, {**self._constants, **changed}, self._host_results
            )
        elif changed:
            self._fn.check_specializations(changed)
            self._function.set_state(changed)
        self._seen.update(now)

    def __call__(self, *args: Any) -> tuple[torch.Tensor, ...]:
        self._write_state(self._state(args))
        inputs = [t for (kind, _), t in zip(self._roles, args) if kind == "input"]
        results = self._function.run(inputs)
        if results is None:
            raise RuntimeError("the tensors of the call do not hold what the graph was compiled for")
        return tuple(results)


def _compile_only(fn: Any, device: torch.device, host_results: list[bool]):
    """What stands in for a graph compiled only into the cache: empty tensors of its
    results."""
    results = [fn.arg(name) for name in fn.results]
    host = host_results or [False] * len(results)

    def forward(*_: Any) -> tuple[torch.Tensor, ...]:
        return tuple(
            torch.empty(tuple(a.logical.shape), dtype=getattr(torch, a.logical.dtype), device="cpu" if on_host else device)
            for a, on_host in zip(results, host)
        )

    return forward


def _program(fn: Any, graph: Any, device: torch.device) -> programs.CompiledProgram:
    arg_of = {tuple(a.sources): a for a in fn.args}
    passed = graph.passed if graph.passed is not None else [inp.name for inp in graph.inputs]
    by_name = {inp.name: inp for inp in graph.inputs}
    inputs = tuple(
        programs.InputSpec(name, tuple(by_name[name].shape), by_name[name].dtype, arg_of.get((name,)))
        for name in passed
        if by_name[name].kind == "input"
    )
    outputs = tuple(
        programs.OutputSpec(tuple(a.logical.shape), getattr(torch, a.logical.dtype), a) for a in (fn.arg(n) for n in fn.results)
    )
    compile_id = torch._guards.CompileContext.current_compile_id()
    return programs.CompiledProgram("" if compile_id is None else str(compile_id), device, fn, inputs, outputs)


def rbln_graph_backend(gm: torch.fx.GraphModule, example_inputs: list[Any], options: dict | None = None):
    """Compiles a graph Dynamo hands over whose tensors are on RBLN devices."""
    import rbln

    options = dict(options or {})
    unknown = set(options) - _OPTIONS
    if unknown:
        raise TypeError(f"unknown rbln options {sorted(unknown)}")
    devices = int(options.get("num_devices") or options.get("tensor_parallel_size") or 1)
    if devices != 1:
        raise NotImplementedError(f"a graph over {devices} devices; torch-rbln runs graphs on one device")
    device = _device_of(example_inputs)
    graph = rbln.frontend.capture_dynamo(gm, example_inputs)
    fn = compile_logical(graph, _npu(options, device), devices, options.get("cache_dir"))
    host_results = _host_results(gm)
    if programs.is_capturing():
        programs.submit_program(_program(fn, graph, device))
    if "compile_only" in _modes(options):
        return _compile_only(fn, device, host_results)
    return CompiledGraph(fn, graph, gm, host_results)


def register() -> None:
    """Makes torch.compile(backend="rbln") compile graphs over RBLN tensors here."""
    import rbln.api.torch_backend

    rbln.api.torch_backend.register_device_backend("rbln", rbln_graph_backend)
