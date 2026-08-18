"""ABI handshake between torch-rbln and the librbln.so it loads at import time.

rebel-compiler (#12426) declares two numbers in ``rebel/runtime/api/rbln_abi.h`` and
exports both as C entry points: ``RBLN_ABI_CURRENT``, the interface a librbln.so
implements, and ``RBLN_ABI_MIN_SUPPORTED``, the oldest consumer contract it still
accepts.

torch-rbln owns no number of its own. FindRebel.cmake records a snapshot of
``RBLN_ABI_CURRENT`` from the header the build compiled against
(``_abi_snapshot.BUILT_ABI``) and import time checks it against the live runtime::

    rbln_abi_min_supported() <= BUILT_ABI <= rbln_abi_current()

Only snapshot-versus-runtime-symbol is meaningful: header and .so ship from the same
rebel-compiler build, so comparing the two macros always passes.

The symbols are read with dlsym rather than linked: CPython opens extensions with
``RTLD_NOW``, so a direct reference would turn an older runtime into an
``undefined symbol`` abort with no chance to print a readable message.

A runtime whose two entry points contradict each other (``min > current``) is rejected
outright, snapshot or no snapshot. Everything else that leaves no verdict to reach warns
and continues: a runtime predating the handshake, one exporting only half the pair, a
handle that cannot be taken on the mapped library, and a build that recorded no snapshot.
``TORCH_RBLN_SKIP_ABI_CHECK=1`` skips the check entirely; see docs/CONFIGURATION.md.
"""

import ctypes
import os
import warnings


# The pair rebel-compiler exports, in the order they are read: (min_supported, current).
ABI_SYMBOLS = ("rbln_abi_min_supported", "rbln_abi_current")

_SKIP_ENV = "TORCH_RBLN_SKIP_ABI_CHECK"

# Verdicts returned by check_librbln_abi; a mismatch raises instead.
VERDICT_OK = "ok"
VERDICT_SKIPPED_DISABLED = "skipped:disabled"
VERDICT_SKIPPED_NO_SNAPSHOT = "skipped:no-snapshot"
VERDICT_SKIPPED_PRE_ABI_RUNTIME = "skipped:pre-abi-runtime"
VERDICT_SKIPPED_MALFORMED_RUNTIME = "skipped:malformed-runtime"
VERDICT_SKIPPED_UNREADABLE_RUNTIME = "skipped:unreadable-runtime"


def is_abi_check_disabled() -> bool:
    """True if the user opted out via ``TORCH_RBLN_SKIP_ABI_CHECK``."""
    return os.environ.get(_SKIP_ENV, "").strip().upper() in ("1", "ON", "TRUE", "YES")


def get_built_abi() -> int | None:
    """The ABI number recorded at build time, or None if this build recorded none.

    None covers a pre-handshake rebel-compiler (``BUILT_ABI = None``) and a source tree
    that was never built through CMake (module absent).
    """
    try:
        from torch_rbln._internal._abi_snapshot import BUILT_ABI
    except ImportError:
        return None
    # bool is an int subclass; True would otherwise compare as ABI 1.
    if isinstance(BUILT_ABI, bool) or not isinstance(BUILT_ABI, int) or BUILT_ABI < 1:
        return None
    return BUILT_ABI


def open_mapped_runtime(path: str | None) -> ctypes.CDLL | None:
    """A handle on the librbln.so already mapped at ``path``, or None if one cannot be taken.

    ``RTLD_NOLOAD`` references the existing mapping instead of making a second one, so this
    never brings another copy of the runtime into the process. Failure is not fatal and must
    not be: the caller mapped the library successfully and only loses the ability to read its
    ABI symbols -- a path out of /proc/self/maps whose file has since been replaced is enough
    to get here.
    """
    if not path:
        return None
    try:
        return ctypes.CDLL(path, mode=os.RTLD_NOLOAD | ctypes.RTLD_GLOBAL)
    except OSError:
        return None


def read_abi_symbols(lib: ctypes.CDLL | None) -> tuple[list[int], list[str]]:
    """Resolve both ABI entry points off a loaded librbln.so.

    Returns the values that resolved, in ``ABI_SYMBOLS`` order, and the names that did not:
    neither present is a runtime older than the handshake, exactly one present is a malformed
    runtime, since the pair is declared in one header and exported by one build.
    """
    if lib is None:
        return [], list(ABI_SYMBOLS)
    values: list[int] = []
    missing: list[str] = []
    for name in ABI_SYMBOLS:
        try:
            fn = getattr(lib, name)
        except (AttributeError, OSError):
            missing.append(name)
            continue
        fn.restype = ctypes.c_uint32
        fn.argtypes = []
        values.append(int(fn()))
    return values, missing


def read_runtime_abi(lib: ctypes.CDLL | None) -> tuple[int, int] | None:
    """Read ``(min_supported, current)`` off a loaded librbln.so, or None if either is absent."""
    values, missing = read_abi_symbols(lib)
    if missing:
        return None
    return values[0], values[1]


def _inconsistent_window_reason(runtime_min: int, runtime_current: int) -> str:
    return (
        f"librbln.so reports an inconsistent ABI window: it accepts consumers from "
        f"{runtime_min} but only implements {runtime_current}, so its two ABI entry "
        f"points disagree."
    )


def abi_mismatch_reason(built_abi: int, runtime_min: int, runtime_current: int) -> str | None:
    """Why this combination is rejected, or None when it is inside the window."""
    if runtime_min > runtime_current:
        return _inconsistent_window_reason(runtime_min, runtime_current)
    if built_abi < runtime_min:
        return (
            f"this torch-rbln was built against rebel ABI {built_abi}, and librbln.so no longer "
            f"accepts consumers below ABI {runtime_min} (it implements {runtime_current}). "
            f"Install a torch-rbln built against this rebel-compiler."
        )
    if built_abi > runtime_current:
        return (
            f"this torch-rbln was built against rebel ABI {built_abi}, but librbln.so only "
            f"implements ABI {runtime_current} (accepting {runtime_min} and up). "
            f"Upgrade rebel-compiler to a build that matches."
        )
    return None


def _version_of(distribution: str) -> str:
    """Best-effort installed version, for the mismatch report."""
    try:
        from importlib.metadata import version

        return version(distribution)
    except Exception:
        return "unknown"


def _mismatch_report(reason: str, librbln_path: str | None, built_abi: int | None) -> str:
    built = f"built against rebel ABI {built_abi}" if built_abi is not None else "no ABI snapshot recorded"
    return (
        f"RBLN ABI mismatch: {reason}\n"
        f"  librbln.so:     {librbln_path or 'unknown'}\n"
        f"  torch-rbln:     {_version_of('torch-rbln')} ({built})\n"
        f"  rebel-compiler: {_version_of('rebel-compiler')}\n"
        f"Run `python -m torch_rbln.diagnose` for the full environment report."
    )


def check_librbln_abi(librbln_path: str | None) -> str:
    """Validate the librbln.so mapped at ``librbln_path`` against this build's ABI snapshot.

    Must run before any other rebel entry point is reached, including the ones our own
    extensions resolve as they load. Taking the handle is part of the check rather than the
    caller's job, so that everything this does sits behind the opt-out and fails open: a
    guard that can itself break an import it was meant to explain is worse than no guard.

    Args:
        librbln_path: path of the librbln.so this process has already mapped.

    Returns:
        str: one of the ``VERDICT_*`` values.

    Raises:
        ImportError: the runtime and this build are outside each other's window, or the
            runtime contradicts itself.
    """
    if is_abi_check_disabled():
        return VERDICT_SKIPPED_DISABLED

    lib = open_mapped_runtime(librbln_path)
    if lib is None:
        warnings.warn(
            f"librbln.so ({librbln_path or 'unknown'}) is mapped into this process but no handle "
            f"could be taken on it, so the rebel ABI contract could not be read. Continuing; run "
            f"`python -m torch_rbln.diagnose` if anything downstream misbehaves.",
            stacklevel=2,
        )
        return VERDICT_SKIPPED_UNREADABLE_RUNTIME

    values, missing = read_abi_symbols(lib)
    built_abi = get_built_abi()

    # A runtime whose own two entry points disagree describes no acceptable consumer, so it is
    # rejected before the snapshot is consulted: there is nothing it could be valid against.
    if not missing and values[0] > values[1]:
        reason = _inconsistent_window_reason(values[0], values[1])
        raise ImportError(_mismatch_report(reason, librbln_path, built_abi))

    if built_abi is None:
        warnings.warn(
            "torch-rbln recorded no rebel ABI number at build time (it was built against a "
            "rebel-compiler older than the ABI handshake), so an incompatible librbln.so cannot "
            "be detected here and will surface as a crash inside the runtime instead. Rebuild "
            "against a rebel-compiler that ships rebel/runtime/api/rbln_abi.h.",
            stacklevel=2,
        )
        return VERDICT_SKIPPED_NO_SNAPSHOT

    if len(missing) == len(ABI_SYMBOLS):
        warnings.warn(
            f"librbln.so ({librbln_path or 'unknown'}) exports no ABI version symbols, so it "
            f"predates the rebel ABI handshake and cannot be checked against this torch-rbln "
            f"(built against rebel ABI {built_abi}). Continuing; upgrade rebel-compiler to get "
            f"the check.",
            stacklevel=2,
        )
        return VERDICT_SKIPPED_PRE_ABI_RUNTIME

    if missing:
        exported = ", ".join(name for name in ABI_SYMBOLS if name not in missing)
        warnings.warn(
            f"librbln.so ({librbln_path or 'unknown'}) exports {exported} but not "
            f"{', '.join(missing)}. The two ship together, so this runtime is malformed rather "
            f"than merely old; half a window is no window, and this torch-rbln (built against "
            f"rebel ABI {built_abi}) could not be checked against it. Continuing; reinstall "
            f"rebel-compiler.",
            stacklevel=2,
        )
        return VERDICT_SKIPPED_MALFORMED_RUNTIME

    reason = abi_mismatch_reason(built_abi, values[0], values[1])
    if reason is not None:
        raise ImportError(_mismatch_report(reason, librbln_path, built_abi))
    return VERDICT_OK
