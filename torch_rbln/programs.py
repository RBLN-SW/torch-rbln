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

Every open scope receives each program, so nested scopes see the same programs; an
inner scope simply sees fewer. The scope is thread-local. The implementation lives in
rebel-compiler next to the backend that builds the programs; this module is the
``torch.rbln`` surface for it.
"""

from rebel.core.program_capture import capture_programs, CompiledProgram


__all__ = ["capture_programs", "CompiledProgram"]
