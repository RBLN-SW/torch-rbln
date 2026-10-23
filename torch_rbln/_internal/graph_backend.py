"""The backend of ``torch.compile(backend="rbln")`` for graphs over torch-rbln tensors.

Dynamo hands over a graph and the tensors of the call it traced. The graph is captured
with ``rebel.v2.frontend.capture_dynamo`` and compiled for the NPU of those tensors, with every
input a call passes and every result laid out as torch lays out a contiguous tensor
(``rebel.v2.LOGICAL``), so torch-rbln tensors bind in place and CPU ones by a copy. An input the
device holds otherwise, such as a float16 KV cache it keeps in dlfloat16, binds in place when it
was made in the arg's type (``torch.rbln.empty_typed``), and goes through the host otherwise. The
module's parameters and buffers are the function's state: they are written to the device
when first seen and again once they change, as another tensor or written in place. A
result the graph leaves on the CPU comes back on the CPU.

Each rank compiles its graph apart, and the device collectives of functions whose collectives
differ, in kind, reduce op, dtype, size or order, wait for each other for good or combine buffers
that do not match. So before a graph with collectives first runs, its rank checks with the other
ranks of the group that their functions make the same ones.
"""

from __future__ import annotations

import json
import threading
import warnings
from typing import Any

import torch
from torch.utils._pytree import tree_flatten

import torch_rbln._C as _C
from torch_rbln import programs
from torch_rbln._internal.compile_cache import compile_logical


_OPTIONS = {"npu", "devices", "mode", "cache_dir", "disable_logger"}


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


def _device_npu(device: torch.device) -> str:
    if _C.is_dummy_device():
        from rebel import v2

        return v2.flags.RBLN_FORCE_NPU_NAME or "RBLN-CA25"
    from torch_rbln.device.device import get_device_name

    return get_device_name(device.index)


def _npu(options: dict[str, Any], device: torch.device) -> str:
    """The NPU to compile for: the ``npu`` option, else the device's."""
    own = _device_npu(device)
    asked = options.get("npu")
    if asked and asked != own:
        warnings.warn(f"compiling for {asked}, which differs from {own} of {device}; the graph runs only on an {asked}")
    return asked or own


def _collective_group(gm: torch.fx.GraphModule) -> str | None:
    """The process group the graph's collectives run over, by name; none for a graph without any.

    A compiled function runs its collectives over one communicator, so a graph over several groups
    is refused.
    """
    names = set()
    for node in gm.graph.nodes:
        # Dynamo names an op by its packet, export by an overload of it.
        packet = getattr(node.target, "_overloadpacket", node.target)
        if node.op != "call_function" or not getattr(packet, "_qualified_op_name", "").startswith("_c10d_functional::"):
            continue
        arguments = [a.name for a in getattr(node.target, "default", node.target)._schema.arguments]
        if "group_name" in arguments:
            index = arguments.index("group_name")
            names.add(node.args[index] if index < len(node.args) else node.kwargs["group_name"])
    if len(names) > 1:
        raise NotImplementedError(f"a graph whose collectives run over the process groups {sorted(names)}")
    return next(iter(names), None)


class CollectivesDiffer(RuntimeError):
    """The ranks of a process group compiled graphs whose collectives differ. Run on the CPU
    instead, the graphs would make the same collectives through the group, so no fallback takes
    their place."""


_checks_lock = threading.Lock()
# The checks the graphs over each process group made in this process, by group name.
_checks: dict[str, int] = {}


def _describe(collective: dict | None) -> str:
    if collective is None:
        return "none"
    reduce = f"{collective['reduce']}, " if collective["reduce"] else ""
    return f"{collective['symbol']}({reduce}{collective['count']} x {collective['dtype']}, world {collective['world']})"


def _check_collectives(fn: Any, group: str) -> None:
    """Raises on every rank of process group ``group`` unless the function each compiled makes the
    same collectives, in the order it runs them, as the others'.

    The ranks swap them through the group's store, under the number of checks the group's graphs
    made in the process before, so the n-th graph a rank runs is checked against the n-th of each
    other rank. A rank that runs no such graph leaves the others waiting for the store's timeout.
    """
    from torch.distributed import distributed_c10d

    process_group = distributed_c10d._resolve_process_group(group)
    store = distributed_c10d._get_process_group_store(process_group)
    with _checks_lock:
        check = _checks.get(group, 0)
        _checks[group] = check + 1
    keys = [f"rbln_collectives/{check}/{rank}" for rank in range(process_group.size())]
    store.set(keys[process_group.rank()], json.dumps(fn.collectives))
    store.wait(keys)
    made = [json.loads(value) for value in store.multi_get(keys)]
    for index in range(max(len(m) for m in made)):
        at = [m[index] if index < len(m) else None for m in made]
        if any(c != at[0] for c in at):
            ranks = distributed_c10d.get_process_group_ranks(process_group)
            by_collective: dict[str, list[int]] = {}
            for rank, c in zip(ranks, at):
                by_collective.setdefault(_describe(c), []).append(rank)
            found = "; ".join(f"{c} on ranks {r}" for c, r in by_collective.items())
            raise CollectivesDiffer(
                f"the ranks of process group {group} compiled graphs whose collectives differ, which the device "
                f"would wait on for good or combine wrongly: collective {index} is {found}"
            )


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

    def __init__(
        self, fn: Any, graph: Any, module: torch.nn.Module, host_results: list[bool], group: str | None = None
    ):
        self._fn = fn
        self._group = group
        self._checked = group is None
        by_name = {inp.name: inp for inp in graph.inputs}
        passed = graph.passed if graph.passed is not None else [inp.name for inp in graph.inputs]
        self._roles = [(by_name[name].kind, name) for name in passed]
        self._inputs = [name for kind, name in self._roles if kind == "input"]
        self._held = {
            inp.name: inp.fqn for inp in graph.inputs if inp.kind in ("parameter", "buffer") and inp.name not in passed
        }
        self._constants = {inp.name: inp.value for inp in graph.inputs if inp.kind == "constant"}
        self._module = module
        # The results the graph returns, then those holding the new values of the inputs it writes
        # other than in place, by input.
        self._returned = len(fn.returned)
        self._written = [(self._inputs.index(name), fn.results.index(result)) for name, result in fn.written.items()]
        self._host_results = host_results + [False] * (len(fn.results) - self._returned)
        self._function: Any = None
        self._seen: dict[str, tuple[int, int]] = {}
        # The state the args are made from; the rest the program was built for, which only the
        # specializations check.
        self._sources = {source for a in fn.args for source in a.sources}

    def _state(self, args: tuple) -> dict[str, torch.Tensor]:
        state = {name: t.detach() for (kind, name), t in zip(self._roles, args) if kind not in ("input", "constant")}
        if self._held:
            held = {**dict(self._module.named_parameters()), **dict(self._module.named_buffers())}
            state.update({name: held[fqn].detach() for name, fqn in self._held.items()})
        return state

    def _write_state(self, state: dict[str, torch.Tensor]) -> None:
        now = {name: (t.data_ptr(), t._version) for name, t in state.items()}
        changed = {name: state[name] for name, seen in now.items() if self._seen.get(name) != seen}
        if self._function is None:
            self._fn.check_specializations(changed)
            self._function = _C._OpFunction(
                self._fn.to_bytes(),
                self._inputs,
                {**self._constants, **changed},
                self._host_results,
            )
            if self._group is not None:
                from torch.distributed import distributed_c10d

                process_group = distributed_c10d._resolve_process_group(self._group)
                self._function.set_communicator(process_group._get_backend(torch.device("rbln")))
        elif changed:
            self._fn.check_specializations(changed)
            made = {name: t for name, t in changed.items() if name in self._sources}
            if made:
                self._function.set_state(made)
        self._seen.update(now)

    def __call__(self, *args: Any) -> tuple[torch.Tensor, ...]:
        # Dynamo passes the extents of dynamic axes as well, which the tensors carry.
        args = tuple(a for a in args if isinstance(a, torch.Tensor))
        if programs.is_compiling_only():
            device = next((a.device for a in args if a.device.type != "cpu"), torch.device("rbln"))
            return _zeros(self._fn, device, self._host_results)
        self._write_state(self._state(args))
        if not self._checked:
            _check_collectives(self._fn, self._group)
            self._checked = True
        inputs = [t for (kind, _), t in zip(self._roles, args) if kind == "input"]
        results = self._function.run(inputs)
        if results is None:
            raise RuntimeError("the tensors of the call do not hold what the graph was compiled for")
        for input_index, result in self._written:
            inputs[input_index].copy_(results[result])
        return tuple(results[: self._returned])


def _zeros(fn: Any, device: torch.device, host_results: list[bool]) -> tuple[torch.Tensor, ...]:
    """Zeros of the results `fn` returns, on `device` or on the CPU as `host_results` says."""
    results = [fn.arg(name) for name in fn.returned]
    host = host_results or [False] * len(results)
    return tuple(
        torch.zeros(tuple(a.logical.shape), dtype=getattr(torch, a.logical.dtype), device="cpu" if on_host else device)
        for a, on_host in zip(results, host)
    )


def _compile_only(fn: Any, device: torch.device, host_results: list[bool]):
    """What stands in for a graph compiled only into the cache: empty tensors of its
    results."""
    results = [fn.arg(name) for name in fn.returned]
    host = host_results or [False] * len(results)

    def forward(*_: Any) -> tuple[torch.Tensor, ...]:
        return tuple(
            torch.empty(
                tuple(a.logical.shape), dtype=getattr(torch, a.logical.dtype), device="cpu" if on_host else device
            )
            for a, on_host in zip(results, host)
        )

    return forward


def _program(fn: Any, graph: Any, device: torch.device, example_inputs: list[Any]) -> programs.CompiledProgram:
    arg_of = {tuple(a.sources): a for a in fn.args}
    passed = graph.passed if graph.passed is not None else [inp.name for inp in graph.inputs]
    by_name = {inp.name: inp for inp in graph.inputs}
    tensors = dict(zip(passed, (x for x in example_inputs if isinstance(x, torch.Tensor))))
    inputs = tuple(
        programs.InputSpec(
            name,
            tuple(tensors[name].shape),
            by_name[name].dtype,
            arg_of.get((name,)),
            tensors[name].untyped_storage().data_ptr(),
        )
        for name in passed
        if by_name[name].kind == "input"
    )
    outputs = tuple(
        programs.OutputSpec(tuple(a.logical.shape), getattr(torch, a.logical.dtype), a)
        for a in (fn.arg(n) for n in fn.results)
    )
    compile_id = torch._guards.CompileContext.current_compile_id()
    return programs.CompiledProgram("" if compile_id is None else str(compile_id), device, fn, inputs, outputs)


def rbln_graph_backend(gm: torch.fx.GraphModule, example_inputs: list[Any], options: dict | None = None):
    """Compiles a graph Dynamo hands over whose tensors are on RBLN devices."""
    from rebel import v2

    options = dict(options or {})
    unknown = set(options) - _OPTIONS
    if unknown:
        raise TypeError(f"unknown rbln options {sorted(unknown)}")
    devices = int(options.get("devices", 1))
    if devices != 1:
        raise ValueError(
            f"devices={devices}: a torch device is one NPU, so its graph compiles for devices=1. "
            "Compile with rebel.v2 directly to spread a model over several NPUs."
        )
    device = _device_of(example_inputs)
    graph = v2.frontend.capture_dynamo(gm, example_inputs)
    fn = compile_logical(graph, _npu(options, device), options.get("cache_dir"))
    host_results = _host_results(gm)
    if programs.is_capturing():
        programs.submit_program(_program(fn, graph, device, example_inputs))
    if "compile_only" in _modes(options):
        return _compile_only(fn, device, host_results)
    return CompiledGraph(fn, graph, gm, host_results, _collective_group(gm))


def register() -> None:
    """Makes torch.compile(backend="rbln") compile graphs over RBLN tensors here."""
    from rebel.v2.api import torch_backend

    torch_backend.register_device_backend("rbln", rbln_graph_backend)
