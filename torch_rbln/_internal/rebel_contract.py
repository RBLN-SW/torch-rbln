"""The rebel-compiler Python surface torch-rbln drives.

torch-rbln's C dependency on rebel is declared in ``rebel/runtime/api/rbln_runtime_api.h``,
recorded at build time by FindRebel.cmake and checked against the loaded librbln.so at import
(:mod:`torch_rbln._internal.abi_check`). Its Python dependency was declared nowhere: the names
were spelled at each call site, and a rebel-side rename arrived as an ImportError, as an arch
reported ``"unknown"``, or as a warm cache that quietly stopped serving.

:data:`CONTRACT` is that declaration -- every rebel attribute torch-rbln reaches for and the
positional parameters it passes. This module is also the only one under ``torch_rbln/`` that may
import rebel; ruff's TID251 banned-api rule holds that.

rebel lands interface changes and torch-rbln follows afterwards, so :func:`verify` reports two
kinds of divergence and only one of them is a defect:

``BROKEN``
    The call torch-rbln makes no longer works: a name is gone, renamed, reordered, or gained a
    required parameter. Fails the contract test.
``DRIFTED``
    The call still works; rebel grew a parameter that has a default. Reported, never asserted --
    whoever moves the rebel pin decides whether to follow it.

Both are reported at configure time by FindRebel.cmake and by ``python -m torch_rbln.diagnose``.
Neither runs at import: verifying imports rebel submodules, and rebel's own import triggers
torch's backend autoload, which imports torch_rbln -- checking there re-enters a half-initialized
rebel and reports a break that is not one.
"""

from __future__ import annotations

import ast
import importlib
import inspect
from typing import Any, NamedTuple


BROKEN = "BROKEN"
DRIFTED = "DRIFTED"


class Name(NamedTuple):
    """One rebel attribute torch-rbln depends on.

    ``attr`` is dotted and resolved from ``module``; None declares the module itself, imported
    for a side effect. ``params`` lists the positional parameters torch-rbln passes, in order and
    excluding ``self``; None declares a non-callable.
    """

    module: str
    attr: str | None = None
    params: tuple[str, ...] | None = None

    @property
    def label(self) -> str:
        return f"{self.module}:{self.attr}" if self.attr else self.module


class Divergence(NamedTuple):
    """One way the installed rebel differs from :data:`CONTRACT`."""

    name: str
    kind: str
    detail: str

    def __str__(self) -> str:
        return f"{self.kind} {self.name}: {self.detail}"


# Positional arguments the C++ warm-cache hit path passes to each sync-runtime method.
RUNTIME_METHOD_PARAMS: dict[str, tuple[str, ...]] = {
    "prepare_inputs": ("device_inputs", "cpu_inputs"),
    "prepare_outputs": ("device_outputs", "cpu_outputs"),
    "run": (),
}

# The methods themselves, in the order install resolves them.
RUNTIME_METHODS: tuple[str, ...] = tuple(RUNTIME_METHOD_PARAMS)

# Instance attribute of rebel's DynamoRuntime holding the sync runtime, and the torch.compile
# option rebel's backend appends that runtime into. Both are per-instance, so :func:`verify`
# cannot reach them; the warm-cache hit-path test is what covers them.
RUNTIME_HANDLE_ATTR = "_runtime_handle"
RUNTIME_HOLDER_OPTION = "_runtime_holder"

CONTRACT: tuple[Name, ...] = (
    *(Name("rebel._C", f"PyRblnSyncRuntime.{method}", params) for method, params in RUNTIME_METHOD_PARAMS.items()),
    Name("rebel.device_info", "get_npu_name", ("device_id",)),
    Name("rebel.core.torch_eager", "eager_execution_helper", ()),
    Name("rebel.core.torch_eager", "EagerExecutionHelper.set_out_tensor", ("out_tensors",)),
    Name("rebel.core.torch_eager", "EagerExecutionHelper.clear_out_tensor", ()),
    # Imported for the module-level register_backend() that makes backend="rbln" resolvable.
    Name("rebel.core.torch_compile"),
)


# --------------------------------------------------------------------------------------------
# Call sites
# --------------------------------------------------------------------------------------------


def get_npu_name(device_id: int) -> str:
    """rebel's name for the NPU at ``device_id``."""
    from rebel.device_info import get_npu_name as _get_npu_name

    return str(_get_npu_name(device_id) or "")


def eager_execution_helper() -> Any:
    """A fresh rebel eager-execution helper (``set_out_tensor`` / ``clear_out_tensor``)."""
    from rebel.core.torch_eager import eager_execution_helper as _eager_execution_helper

    return _eager_execution_helper()


def register_torch_compile_backend() -> None:
    """Register rebel's ``"rbln"`` torch.compile backend.

    Registration is a module-level side effect of the import; ImportError reaches the caller.
    """
    import rebel.core.torch_compile  # noqa: F401


def runtime_methods(handle: Any) -> list[Any] | None:
    """``handle``'s bound :data:`RUNTIME_METHODS`, or None if it is missing any of them.

    None keeps the op on the Python wrapper path, which is slower and correct.
    """
    bound = [getattr(handle, name, None) for name in RUNTIME_METHODS]
    if not all(callable(method) for method in bound):
        return None
    return bound


# --------------------------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------------------------


class _Signature(NamedTuple):
    positional: tuple[str, ...]
    defaulted: int
    kwonly_required: tuple[str, ...]


def _read_pybind_signature(fn: Any) -> _Signature | None:
    """Read a pybind11 entry point's signature off the first line of its docstring.

    That line is valid Python, so it parses exactly. Returns None for an overloaded entry point,
    whose first line is ``(*args, **kwargs)`` and carries no parameter names.
    """
    lines = (getattr(fn, "__doc__", None) or "").splitlines()
    if not lines:
        return None
    try:
        tree = ast.parse(f"def {lines[0].strip()}: ...")
    except SyntaxError:
        return None
    node = tree.body[0]
    if not isinstance(node, ast.FunctionDef):
        return None
    args = node.args
    if args.vararg is not None or args.kwarg is not None:
        return None
    return _Signature(
        tuple(arg.arg for arg in (*args.posonlyargs, *args.args)),
        len(args.defaults),
        tuple(arg.arg for arg, default in zip(args.kwonlyargs, args.kw_defaults) if default is None),
    )


def _read_signature(fn: Any) -> _Signature | None:
    """``fn``'s signature, or None if it cannot be read.

    ``inspect.signature`` raises on every pybind11 entry point, which is most of this surface.
    """
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return _read_pybind_signature(fn)
    positional: list[str] = []
    defaulted = 0
    kwonly_required: list[str] = []
    for parameter in signature.parameters.values():
        if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD):
            positional.append(parameter.name)
            if parameter.default is not parameter.empty:
                defaulted += 1
        elif parameter.kind is parameter.KEYWORD_ONLY and parameter.default is parameter.empty:
            kwonly_required.append(parameter.name)
    return _Signature(tuple(positional), defaulted, tuple(kwonly_required))


def _verify_signature(name: Name, fn: Any) -> Divergence | None:
    """Compare ``fn`` against the parameters ``name`` declares torch-rbln passes."""
    assert name.params is not None
    signature = _read_signature(fn)
    if signature is None:
        return None

    # Bound and unbound forms differ by a leading self; the declaration carries neither.
    positional = signature.positional
    if positional and positional[0] == "self":
        positional = positional[1:]

    declared = name.params
    if positional[: len(declared)] != declared:
        return Divergence(
            name.label,
            BROKEN,
            f"torch-rbln passes {list(declared)}, rebel now takes {list(positional)}",
        )
    if signature.kwonly_required:
        return Divergence(
            name.label,
            BROKEN,
            f"rebel now requires keyword-only {list(signature.kwonly_required)}, which torch-rbln does not pass",
        )
    required = len(positional) - signature.defaulted
    if required > len(declared):
        return Divergence(
            name.label,
            BROKEN,
            f"rebel now requires {list(positional[:required])}, and torch-rbln passes only {list(declared)}",
        )
    if len(positional) > len(declared):
        return Divergence(
            name.label,
            DRIFTED,
            f"rebel grew defaulted {list(positional[len(declared) :])}; torch-rbln keeps passing {list(declared)} "
            f"and takes rebel's defaults for the rest",
        )
    return None


def _resolve(name: Name) -> Any:
    """The object ``name`` declares. Raises ImportError or AttributeError if it is not there."""
    obj: Any = importlib.import_module(name.module)
    for part in name.attr.split(".") if name.attr else ():
        obj = getattr(obj, part)
    return obj


def _resolution_error(name: Name) -> Divergence | None:
    """Why ``name`` cannot be reached at all, or None."""
    try:
        obj = _resolve(name)
    except ImportError as e:
        return Divergence(name.label, BROKEN, f"module not importable: {e}")
    except AttributeError:
        return Divergence(name.label, BROKEN, "attribute not found")
    if name.params is not None and not callable(obj):
        return Divergence(name.label, BROKEN, "not callable")
    return None


def _verify(name: Name) -> Divergence | None:
    divergence = _resolution_error(name)
    if divergence is not None or name.params is None:
        return divergence
    return _verify_signature(name, _resolve(name))


def verify() -> list[Divergence]:
    """How the installed rebel differs from :data:`CONTRACT`, worst first."""
    found = [d for d in (_verify(name) for name in CONTRACT) if d is not None]
    return sorted(found, key=lambda d: d.kind != BROKEN)


def broken() -> list[Divergence]:
    """Only the divergences that stop a call torch-rbln makes."""
    return [d for d in verify() if d.kind == BROKEN]
