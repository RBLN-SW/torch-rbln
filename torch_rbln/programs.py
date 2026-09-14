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
        program.device  # rbln device the runtime is bound to
        program.runtime  # the execution handle behind the callable
        program.input_specs  # tuple[InputSpec]: name, shape, dtype, physical_placement
        program.output_specs  # tuple[OutputSpec]: shape, dtype, physical_placement

``input_specs`` / ``output_specs`` list the IO in the order ``run()`` takes and returns it.
``physical_placement`` is set only for dynamic-shape IO and gives the device layout as one
``ShardPlacement`` per (node, chiplet) shard.

Every open scope receives each program, so nested scopes see the same programs; an
inner scope simply sees fewer. The scope is thread-local. The implementation lives in
rebel-compiler next to the backend that builds the programs; this module is the
``torch.rbln`` surface for it. Both names resolve on first access so that ``import
torch`` does not pull the ``rebel`` package in through torch's autoload hook.
"""

from typing import Any


__all__ = [  # noqa: F822  # resolved by __getattr__
    "capture_programs",
    "CompiledProgram",
    "InputSpec",
    "OutputSpec",
    "PhysicalPlacement",
    "ShardPlacement",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from rebel.core import program_capture

        return getattr(program_capture, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
