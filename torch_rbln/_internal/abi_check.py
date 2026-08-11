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

Two cases warn and continue because neither yields a verdict -- a runtime that
predates the handshake, and a build that recorded no snapshot.
``TORCH_RBLN_SKIP_ABI_CHECK=1`` skips the check entirely; see docs/CONFIGURATION.md.
"""

import ctypes
import os
import warnings


_ABI_SYMBOLS = ("rbln_abi_min_supported", "rbln_abi_current")

_SKIP_ENV = "TORCH_RBLN_SKIP_ABI_CHECK"

# Verdicts returned by check_librbln_abi; a mismatch raises instead.
VERDICT_OK = "ok"
VERDICT_SKIPPED_DISABLED = "skipped:disabled"
VERDICT_SKIPPED_NO_SNAPSHOT = "skipped:no-snapshot"
VERDICT_SKIPPED_PRE_ABI_RUNTIME = "skipped:pre-abi-runtime"


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


def read_runtime_abi(lib: ctypes.CDLL | None) -> tuple[int, int] | None:
    """Read ``(min_supported, current)`` off a loaded librbln.so, or None if it predates the ABI."""
    if lib is None:
        return None
    values = []
    for name in _ABI_SYMBOLS:
        try:
            fn = getattr(lib, name)
        except (AttributeError, OSError):
            # A .so exporting only one of the pair is malformed rather than old, but
            # there is still no window to check against either way.
            return None
        fn.restype = ctypes.c_uint32
        fn.argtypes = []
        values.append(int(fn()))
    return values[0], values[1]


def abi_mismatch_reason(built_abi: int, runtime_min: int, runtime_current: int) -> str | None:
    """Why this combination is rejected, or None when it is inside the window."""
    if runtime_min > runtime_current:
        return (
            f"librbln.so reports an inconsistent ABI window: it accepts consumers from "
            f"{runtime_min} but only implements {runtime_current}, so its two ABI entry "
            f"points disagree."
        )
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


def _mismatch_report(reason: str, lib: ctypes.CDLL | None, built_abi: int) -> str:
    return (
        f"RBLN ABI mismatch: {reason}\n"
        f"  librbln.so:     {getattr(lib, '_name', None) or 'unknown'}\n"
        f"  torch-rbln:     {_version_of('torch-rbln')} (built against rebel ABI {built_abi})\n"
        f"  rebel-compiler: {_version_of('rebel-compiler')}\n"
        f"Run `python -m torch_rbln.diagnose` for the full environment report."
    )


def check_librbln_abi(lib: ctypes.CDLL | None) -> str:
    """Validate the loaded librbln.so against this build's ABI snapshot.

    Must run before any other rebel entry point is reached, including the ones our own
    extensions resolve as they load.

    Args:
        lib: the librbln.so handle returned by ``find_and_load_tvm_library``.

    Returns:
        str: one of the ``VERDICT_*`` values.

    Raises:
        ImportError: the runtime and this build are outside each other's window.
    """
    if is_abi_check_disabled():
        return VERDICT_SKIPPED_DISABLED

    built_abi = get_built_abi()
    if built_abi is None:
        warnings.warn(
            "torch-rbln recorded no rebel ABI number at build time (it was built against a "
            "rebel-compiler older than the ABI handshake), so an incompatible librbln.so cannot "
            "be detected here and will surface as a crash inside the runtime instead. Rebuild "
            "against a rebel-compiler that ships rebel/runtime/api/rbln_abi.h.",
            stacklevel=2,
        )
        return VERDICT_SKIPPED_NO_SNAPSHOT

    window = read_runtime_abi(lib)
    if window is None:
        warnings.warn(
            f"librbln.so ({getattr(lib, '_name', None) or 'unknown'}) exports no ABI version "
            f"symbols, so it predates the rebel ABI handshake and cannot be checked against this "
            f"torch-rbln (built against rebel ABI {built_abi}). Continuing; upgrade "
            f"rebel-compiler to get the check.",
            stacklevel=2,
        )
        return VERDICT_SKIPPED_PRE_ABI_RUNTIME

    reason = abi_mismatch_reason(built_abi, window[0], window[1])
    if reason is not None:
        raise ImportError(_mismatch_report(reason, lib, built_abi))
    return VERDICT_OK
