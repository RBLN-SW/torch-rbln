"""ABI check between torch-rbln and the rbln runtime it runs on.

The runtime's C++ API is its headers, ``rbln/include/rbln/**/*.h``. Their ABI id is a SHA-256
over every header's relative path and SHA-256, computed by ``rbln/cmake/RblnAbi.cmake``; the
runtime exports the id of the headers it was built with as ``rbln_abi_id()``. The torch-rbln
build runs the same script over the headers it compiles against and records the result in
``_abi_snapshot.BUILT_ABI``, regenerated whenever a header changes. Import time compares the two
for equality, before any of torch-rbln's native libraries is loaded.

The id is read with dlsym through ctypes because the check runs before those libraries are
loaded: a runtime whose API differs from theirs would otherwise surface as an ``undefined
symbol`` abort or as corruption inside the runtime, with no readable message.

Cases that leave no verdict warn and continue: a build that recorded no id, a handle that
cannot be taken on the mapped runtime, and a runtime that exports no ``rbln_abi_id``.
``TORCH_RBLN_SKIP_ABI_CHECK=1`` skips the check entirely; see docs/CONFIGURATION.md.
"""

import ctypes
import os
import re
import sys
import warnings
from dataclasses import dataclass


ABI_SYMBOL = "rbln_abi_id"

_SKIP_ENV = "TORCH_RBLN_SKIP_ABI_CHECK"
_ABI_ID_PATTERN = re.compile(r"[0-9a-f]{64}")

# check_runtime_abi returns one of these, raising on VERDICT_MISMATCH instead.
VERDICT_OK = "ok"
VERDICT_MISMATCH = "mismatch"
VERDICT_SKIPPED_DISABLED = "skipped:disabled"
VERDICT_SKIPPED_NO_SNAPSHOT = "skipped:no-snapshot"
VERDICT_SKIPPED_UNREADABLE_RUNTIME = "skipped:unreadable-runtime"
VERDICT_SKIPPED_NO_RUNTIME_ID = "skipped:no-runtime-id"


@dataclass(frozen=True)
class AbiState:
    """What the check compares, and the verdict it reaches."""

    runtime_path: str | None
    built_abi: str | None
    built_include_dir: str | None
    runtime_abi: str | None
    verdict: str


def is_abi_check_disabled() -> bool:
    """True if the user opted out via ``TORCH_RBLN_SKIP_ABI_CHECK``."""
    return os.environ.get(_SKIP_ENV, "").strip().upper() in ("1", "ON", "TRUE", "YES")


def _valid_abi_id(value: object) -> str | None:
    return value if isinstance(value, str) and _ABI_ID_PATTERN.fullmatch(value) else None


def get_built_abi() -> str | None:
    """The ABI id recorded at build time, or None if this build recorded none.

    None covers a source tree that was never built through CMake (module absent) and a
    snapshot whose value is not a SHA-256 hex digest.
    """
    try:
        from torch_rbln._internal._abi_snapshot import BUILT_ABI
    except ImportError:
        return None
    return _valid_abi_id(BUILT_ABI)


def get_built_include_dir() -> str | None:
    """The runtime include directory this build compiled against, if recorded."""
    try:
        from torch_rbln._internal._abi_snapshot import BUILT_INCLUDE_DIR
    except ImportError:
        return None
    return BUILT_INCLUDE_DIR if isinstance(BUILT_INCLUDE_DIR, str) and BUILT_INCLUDE_DIR else None


def open_mapped_runtime(path: str | None) -> ctypes.CDLL | None:
    """A handle on the runtime already mapped at ``path``, or None if one cannot be taken.

    ``RTLD_NOLOAD`` references the existing mapping instead of making a second one, so this
    never brings another copy of the runtime into the process. Failure is not fatal and must
    not be: the library is mapped and only its ABI id becomes unreadable -- a path out of
    /proc/self/maps whose file has since been replaced is enough to get here.
    """
    if not path:
        return None
    try:
        return ctypes.CDLL(path, mode=os.RTLD_NOLOAD)
    except OSError:
        return None


def read_runtime_abi(lib: ctypes.CDLL | None) -> str | None:
    """The ABI id a loaded runtime reports, or None if it exports none."""
    if lib is None:
        return None
    try:
        fn = getattr(lib, ABI_SYMBOL)
    except (AttributeError, OSError):
        return None
    fn.restype = ctypes.c_char_p
    fn.argtypes = []
    raw = fn()
    return raw.decode("ascii", errors="replace") if raw is not None else None


def inspect_runtime_abi(runtime_path: str | None) -> AbiState:
    """Read both ids and decide, without warning or raising. Ignores the opt-out."""
    built = get_built_abi()
    lib = open_mapped_runtime(runtime_path)
    runtime = read_runtime_abi(lib)
    if lib is None:
        verdict = VERDICT_SKIPPED_UNREADABLE_RUNTIME
    elif built is None:
        verdict = VERDICT_SKIPPED_NO_SNAPSHOT
    elif runtime is None:
        verdict = VERDICT_SKIPPED_NO_RUNTIME_ID
    elif runtime == built:
        verdict = VERDICT_OK
    else:
        verdict = VERDICT_MISMATCH
    return AbiState(runtime_path, built, get_built_include_dir(), runtime, verdict)


def _version_of(distribution: str) -> str:
    """Best-effort installed version, for the mismatch report."""
    try:
        from importlib.metadata import version

        return version(distribution)
    except Exception:
        return "unknown"


def _module_location(name: str) -> str:
    module = sys.modules.get(name)
    return getattr(module, "__file__", None) or "not imported"


def mismatch_report(state: AbiState) -> str:
    """The ImportError message for a runtime whose headers differ from the ones built against."""
    built_from = state.built_include_dir or "an unrecorded include directory"
    return (
        "RBLN ABI mismatch: torch-rbln was built against rbln runtime headers other than the ones "
        "the loaded runtime was built with.\n"
        f"  runtime:      {state.runtime_path or 'unknown'}\n"
        f"  runtime ABI:  {state.runtime_abi}\n"
        f"  rbln package: {_module_location('rbln')}\n"
        f"  torch-rbln:   {_version_of('torch-rbln')} ({_module_location('torch_rbln')})\n"
        f"  built ABI:    {state.built_abi} (headers in {built_from})\n"
        "Rebuild torch-rbln with REBEL_HOME set to the rebel-compiler tree this runtime was built "
        "from, or run it with the rbln package of the tree it was built against.\n"
        "Run `python -m torch_rbln.diagnose` for the full environment report."
    )


def _fail_open_warning(state: AbiState) -> str | None:
    path = state.runtime_path or "unknown"
    if state.verdict == VERDICT_SKIPPED_UNREADABLE_RUNTIME:
        return (
            f"The rbln runtime ({path}) is mapped into this process but no handle could be taken "
            "on it, so its ABI id could not be read. Continuing; run `python -m torch_rbln.diagnose` "
            "if anything downstream misbehaves."
        )
    if state.verdict == VERDICT_SKIPPED_NO_SNAPSHOT:
        return (
            "torch-rbln recorded no rbln ABI id at build time, so a runtime built from other headers "
            "cannot be detected here and will surface as a crash inside the runtime instead. "
            "Rebuild and reinstall torch-rbln; the build generates torch_rbln/_internal/_abi_snapshot.py."
        )
    if state.verdict == VERDICT_SKIPPED_NO_RUNTIME_ID:
        return (
            f"The rbln runtime ({path}) exports no {ABI_SYMBOL}(), so it cannot be checked against "
            f"this torch-rbln (built against ABI {state.built_abi}). Continuing; use a runtime "
            "built from a rebel-compiler tree that declares rbln/abi.h."
        )
    return None


def check_runtime_abi(runtime_path: str | None) -> str:
    """Validate the runtime mapped at ``runtime_path`` against this build's ABI id.

    Must run before torch-rbln's native libraries load. Taking the handle is part of the check
    rather than the caller's job, so that everything this does sits behind the opt-out and fails
    open: a guard that can itself break an import it was meant to explain is worse than none.

    Args:
        runtime_path: path of the ``librbln_rt.so`` this process has already mapped.

    Returns:
        str: one of the ``VERDICT_*`` values other than ``VERDICT_MISMATCH``.

    Raises:
        ImportError: the runtime reports an ABI id other than the one this build recorded.
    """
    if is_abi_check_disabled():
        return VERDICT_SKIPPED_DISABLED

    state = inspect_runtime_abi(runtime_path)
    if state.verdict == VERDICT_MISMATCH:
        raise ImportError(mismatch_report(state))
    warning = _fail_open_warning(state)
    if warning is not None:
        warnings.warn(warning, stacklevel=2)
    return state.verdict
