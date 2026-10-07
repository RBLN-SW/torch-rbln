"""``torch.rbln.capture_programs`` -- reach the programs torch.compile built on the rbln backend.

torch.compile keeps the callable a backend returns inside dynamo's code cache, so the
caller never gets a handle to what was compiled. ``capture_programs()`` opens a scope
that records every program the rbln backend builds while it is open, in build order,
the way ``torch.profiler.profile()`` collects the events inside it::

    with torch.rbln.capture_programs() as programs:
        compiled = torch.compile(model, backend="rbln")
        compiled(*warmup_inputs)  # dynamo builds the graphs here

    for program in programs:  # list[CompiledProgram]
        program.name  # dynamo compile id, e.g. "0/0"
        program.device  # rbln device the graph's tensors are on
        program.function  # the compiled rebel.v2.Function behind the callable
        program.input_specs  # tuple[InputSpec]: name, shape, dtype, arg, data_ptr
        program.output_specs  # tuple[OutputSpec]: shape, dtype, arg

``input_specs`` / ``output_specs`` list the IO in the order the callable takes and returns
it. ``arg`` is the function's ``rebel.v2.Arg`` for it, whose ``physical`` layout and ``shards`` say
how the device holds it; ``torch.rbln.empty_typed(arg)`` makes a tensor in that type, which the
program binds in place. ``data_ptr`` is where the storage of the tensor the traced call passed
starts, which tells the caller which of its tensors the input was.

Every open scope receives each program, so nested scopes see the same programs; an
inner scope simply sees fewer. The scope is thread-local.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Iterator

    import torch


__all__ = [
    "capture_programs",
    "compile_only",
    "CompiledProgram",
    "InputSpec",
    "OutputSpec",
]

_ctx = threading.local()


@dataclass(frozen=True)
class InputSpec:
    """An input of a compiled program, named as the traced code names it, of the shape the traced
    call gave it; `arg` says which extents of a dynamic axis it takes, and `data_ptr` where the
    storage of the tensor the traced call gave it starts."""

    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    arg: Any
    data_ptr: int | None = None


@dataclass(frozen=True)
class OutputSpec:
    """One output of a compiled program; the fields read as in `InputSpec`."""

    shape: tuple[int, ...]
    dtype: torch.dtype
    arg: Any


@dataclass(frozen=True)
class CompiledProgram:
    """One graph the rbln backend built for torch.compile.

    `name` is dynamo's compile id for the graph ("0/0", "0/1", ...), empty if the backend
    ran outside a dynamo compile. `device` is the rbln device the graph's tensors are on.
    `function` is the compiled ``rebel.v2.Function`` the callable runs.
    """

    name: str
    device: torch.device
    function: Any
    input_specs: tuple[InputSpec, ...]
    output_specs: tuple[OutputSpec, ...]


def _scopes() -> list[list[CompiledProgram]]:
    scopes = getattr(_ctx, "scopes", None)
    if scopes is None:
        scopes = _ctx.scopes = []
    return scopes


def is_capturing() -> bool:
    return bool(_scopes())


def submit_program(program: CompiledProgram) -> None:
    """Hands `program` to every scope open on this thread."""
    for scope in _scopes():
        scope.append(program)


@contextmanager
def capture_programs() -> Iterator[list[CompiledProgram]]:
    """Record the programs the rbln backend builds inside this scope.

    Yields a list that fills as torch.compile invokes the backend, in build order. The
    scope records the backend calls made from the thread that opened it, which is where
    dynamo compiles the functions that thread calls.
    """
    programs: list[CompiledProgram] = []
    scopes = _scopes()
    scopes.append(programs)
    try:
        yield programs
    finally:
        # By identity: two empty scopes compare equal, so list.remove would take the wrong one.
        for index in range(len(scopes) - 1, -1, -1):
            if scopes[index] is programs:
                del scopes[index]
                break


@contextmanager
def compile_only() -> Iterator[None]:
    """Compile the graphs torch.compile builds inside this scope, but run none of them.

    A call of a graph inside the scope returns zeros of the shapes and dtypes of its results, and
    writes none of its inputs; the graph runs from its first call outside the scope. A caller
    compiles its graphs this way to learn the args they take (see ``capture_programs``) before it
    makes the tensors they write in place, such as KV caches in the types of those args (see
    ``torch.rbln.empty_typed``). The scope is thread-local.
    """
    _ctx.compile_only = getattr(_ctx, "compile_only", 0) + 1
    try:
        yield
    finally:
        _ctx.compile_only -= 1


def is_compiling_only() -> bool:
    """Whether this thread is inside a ``compile_only`` scope."""
    return getattr(_ctx, "compile_only", 0) > 0
