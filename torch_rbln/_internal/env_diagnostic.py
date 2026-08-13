"""Environment diagnostics for RBLN library loading.

Helps users debug "Cannot find libraries: librbln.so / librbln_runtime.so" and
environment-override issues (REBEL_HOME, PYTHONPATH, LD_LIBRARY_PATH).
"""

import ctypes
import os
import re
import subprocess
import sys
from typing import Any

from torch_rbln._internal.rbln_runtime_lib import (
    load_runtime_library,
    loaded_runtime_libraries,
    RUNTIME_LIB_NAME,
    runtime_library_candidates,
)


# Environment variables that affect where we look for .so files.
ENV_VARS = (
    "REBEL_HOME",
    "LD_LIBRARY_PATH",
    "DYLD_LIBRARY_PATH",
    "PYTHONPATH",
    "PATH",
)


def get_gcc_version_from_elf(filepath: str) -> str:
    """Read GCC version from ELF .comment section (Linux). Returns e.g. 'GCC 12.3.0' or 'unknown'."""
    if not sys.platform.startswith("linux"):
        return "N/A (not Linux)"
    if not os.path.isfile(filepath):
        return "missing"
    try:
        out = subprocess.run(
            ["readelf", "-p", ".comment", filepath],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if out.returncode != 0 or not out.stdout:
            return "no .comment or readelf failed"
        # e.g. "  [     0]  GCC: (Ubuntu 12.3.0-1ubuntu1~22.04) 12.3.0" or "  [     0]  GCC: (GNU) 11.2.0"
        match = re.search(r"GCC:\s*\([^)]*\)\s*(\d+\.\d+(?:\.\d+)?)", out.stdout)
        if match:
            return f"GCC {match.group(1)}"
        if "GCC:" in out.stdout:
            return "GCC (version unparsed)"
        return "no GCC in .comment"
    except FileNotFoundError:
        return "readelf not found"
    except subprocess.TimeoutExpired:
        return "timeout"
    except Exception as e:
        return f"error: {e}"


def _env_snapshot() -> dict[str, Any]:
    """Snapshot of env vars that affect library discovery (empty = not set)."""
    return {k: os.environ.get(k, "") for k in ENV_VARS}


def _package_location(name: str) -> dict[str, Any]:
    """Where a package would be imported from, without importing it."""
    try:
        import importlib.util

        spec = importlib.util.find_spec(name)
        if spec is None:
            return {"found": False, "origin": None, "submodule_search_locations": None}
        locations = getattr(spec, "submodule_search_locations", None)
        origin = getattr(spec, "origin", None)
        return {
            "found": True,
            "origin": origin,
            "submodule_search_locations": list(locations) if locations else None,
        }
    except Exception as e:
        return {"found": False, "error": str(e)}


def _torch_rbln_info() -> dict[str, Any]:
    """Installed torch-rbln package version, location, and install type."""
    out: dict[str, Any] = {
        "version": None,
        "location": None,
        "installed_location": None,
        "shadowed": False,
        "install_type": None,
        "lib_dir": None,
        "lib_dir_exists": None,
    }
    try:
        import importlib.metadata as _meta

        out["version"] = _meta.version("torch-rbln")
    except Exception:
        try:
            import torch_rbln as _tr

            out["version"] = getattr(_tr, "__version__", "?")
        except Exception as e:
            out["version"] = f"(error: {e})"
    try:
        import importlib.util

        spec = importlib.util.find_spec("torch_rbln")
        if spec is not None and getattr(spec, "submodule_search_locations", None):
            out["location"] = os.path.realpath(spec.submodule_search_locations[0])
        elif spec is not None and getattr(spec, "origin", None):
            out["location"] = os.path.realpath(os.path.join(os.path.dirname(spec.origin), ".."))
    except Exception as e:
        out["location"] = f"(error: {e})"
    try:
        import importlib.metadata as _meta

        dist = _meta.distribution("torch-rbln")
        # Where the package is installed (wheel/site-packages), not necessarily what was loaded.
        try:
            out["installed_location"] = os.path.realpath(os.path.dirname(dist.locate_file("torch_rbln/__init__.py")))
        except Exception:
            pass
        direct = getattr(dist, "direct_url", None)
        if direct is not None and getattr(direct, "is_editable", None):
            out["install_type"] = "editable"
        elif dist is not None:
            out["install_type"] = "normal"
    except Exception:
        out["install_type"] = "unknown"
    if (
        out.get("location")
        and out.get("installed_location")
        and isinstance(out["location"], str)
        and isinstance(out["installed_location"], str)
    ):
        if os.path.realpath(out["location"]) != os.path.realpath(out["installed_location"]):
            out["shadowed"] = True
    if out["location"] and isinstance(out["location"], str) and os.path.isdir(out["location"]):
        lib_dir = os.path.join(out["location"], "lib")
        out["lib_dir"] = lib_dir
        out["lib_dir_exists"] = os.path.isdir(lib_dir)
        if out["lib_dir_exists"]:
            try:
                libs = [f for f in os.listdir(lib_dir) if f.endswith(".so")]
                out["native_libs"] = sorted(libs)[:20]
            except OSError:
                out["native_libs"] = []
    return out


def _rebel_compiler_info() -> dict[str, Any]:
    """Installed rebel-compiler package version, location, and install type."""
    out: dict[str, Any] = {
        "found": False,
        "version": None,
        "location": None,
        "installed_location": None,
        "install_type": None,
        "error": None,
    }
    try:
        import importlib.metadata as _meta

        out["version"] = _meta.version("rebel-compiler")
        out["found"] = True
    except Exception as e:
        out["error"] = str(e)
        return out
    try:
        import importlib.util

        spec = importlib.util.find_spec("rebel")
        if spec is not None and getattr(spec, "submodule_search_locations", None):
            out["location"] = os.path.realpath(spec.submodule_search_locations[0])
        elif spec is not None and getattr(spec, "origin", None):
            out["location"] = os.path.realpath(os.path.join(os.path.dirname(spec.origin), ".."))
    except Exception as e:
        out["location"] = f"(error: {e})"
    try:
        import importlib.metadata as _meta

        dist = _meta.distribution("rebel-compiler")
        try:
            for entry in ("rebel/__init__.py", "rebel_compiler/__init__.py"):
                try:
                    out["installed_location"] = os.path.realpath(os.path.dirname(dist.locate_file(entry)))
                    break
                except Exception:
                    continue
        except Exception:
            pass
        direct = getattr(dist, "direct_url", None)
        if direct is not None and getattr(direct, "is_editable", None):
            out["install_type"] = "editable"
        elif dist is not None:
            out["install_type"] = "normal"
    except Exception:
        out["install_type"] = "unknown"
    return out


def _runtime_lib_candidates() -> list[dict[str, Any]]:
    """Every candidate path considered for librbln.so: path, where it came from, and existence."""
    return [
        {"path": path, "source": source, "exists": os.path.isfile(path)}
        for path, source in runtime_library_candidates()
    ]


_VERSION_SYMBOL = "REBEL_COMPILER_VERSION"


def _runtime_library_version(path: str) -> str | None:
    """Read ``REBEL_COMPILER_VERSION`` from an already-mapped library.

    ``RTLD_NOLOAD``: a reference to the existing mapping, never a second copy.
    """
    try:
        lib = ctypes.CDLL(path, mode=os.RTLD_NOLOAD | ctypes.RTLD_GLOBAL)
        raw = ctypes.c_char_p.in_dll(lib, _VERSION_SYMBOL).value
    except (OSError, ValueError):
        return None
    return raw.decode(errors="replace") if raw is not None else None


def _runtime_lib_state() -> dict[str, Any]:
    """Which librbln.so this process mapped, and the version it reports.

    Only the one library. rebel-compiler also ships librbln_runtime.so, a runtime-only build its
    loader can take instead of this one, but the two cannot both be mapped -- loading the second
    aborts the process in its constructors.
    """
    loaded = loaded_runtime_libraries()
    return {
        "loaded": loaded,
        "version": _runtime_library_version(loaded[0]) if loaded else None,
    }


def _rebel_abi_info() -> dict[str, Any]:
    """Build-time ABI snapshot versus what librbln.so reports here.

    Loads librbln.so the same way the real import does, so these are the numbers the
    handshake would actually compare.
    """
    out: dict[str, Any] = {
        "built_abi": None,
        "check_disabled": False,
        "librbln_path": None,
        "runtime_min_supported": None,
        "runtime_current": None,
        "verdict": None,
        "error": None,
    }
    try:
        from torch_rbln._internal import abi_check

        out["built_abi"] = abi_check.get_built_abi()
        out["check_disabled"] = abi_check.is_abi_check_disabled()
    except Exception as e:
        out["error"] = f"abi_check unavailable: {e}"
        return out

    try:
        path = load_runtime_library()
        lib = ctypes.CDLL(path, mode=os.RTLD_NOLOAD | ctypes.RTLD_GLOBAL)
    except Exception as e:
        out["error"] = f"could not load librbln.so: {e}"
        return out

    out["librbln_path"] = path
    try:
        window = abi_check.read_runtime_abi(lib)
    except Exception as e:
        out["error"] = f"could not read ABI symbols: {e}"
        return out

    if window is None:
        out["verdict"] = "runtime predates the ABI handshake (no version symbols)"
        return out

    out["runtime_min_supported"], out["runtime_current"] = window
    if out["built_abi"] is None:
        out["verdict"] = "no build-time snapshot to compare against"
        return out
    reason = abi_check.abi_mismatch_reason(out["built_abi"], window[0], window[1])
    out["verdict"] = "OK" if reason is None else f"MISMATCH -- {reason}"
    return out


def _resolve_so_paths(d: dict[str, Any]) -> list[dict[str, Any]]:
    """Resolve paths for libtorch, libtorch_rbln, librbln (and related) and get GCC version from ELF .comment."""
    result: list[dict[str, Any]] = []
    tr = d.get("torch_rbln") or {}
    candidates = d.get("runtime_lib_candidates") or []

    # libtorch.so — from torch package
    libtorch_path: str | None = None
    try:
        import torch

        torch_root = getattr(torch, "__path__", [None])[0]
        if torch_root:
            lib_dir = os.path.join(torch_root, "lib")
            for name in ("libtorch.so", "libtorch.so.2", "libtorch.so.1"):
                p = os.path.join(lib_dir, name)
                if os.path.isfile(p):
                    libtorch_path = os.path.realpath(p)
                    break
            if libtorch_path is None and os.path.isdir(lib_dir):
                for f in os.listdir(lib_dir):
                    if f.startswith("libtorch.so"):
                        libtorch_path = os.path.realpath(os.path.join(lib_dir, f))
                        break
    except Exception:
        pass
    result.append(
        {
            "name": "libtorch.so",
            "path": libtorch_path,
            "gcc": get_gcc_version_from_elf(libtorch_path) if libtorch_path else "path not found",
        }
    )

    # libtorch_rbln.so, libc10_rbln.so — from torch_rbln lib_dir
    for so_name in ("libtorch_rbln.so", "libc10_rbln.so"):
        path: str | None = None
        lib_dir = tr.get("lib_dir")
        if lib_dir and os.path.isdir(lib_dir):
            p = os.path.join(lib_dir, so_name)
            if os.path.isfile(p):
                path = os.path.realpath(p)
        result.append(
            {
                "name": so_name,
                "path": path,
                "gcc": get_gcc_version_from_elf(path) if path else "path not found",
            }
        )

    # librbln.so — the mapped one if this process loaded it, else the first candidate that exists
    librbln_path = None
    loaded = (d.get("runtime_lib") or {}).get("loaded") or []
    if loaded:
        librbln_path = os.path.realpath(loaded[0])
    else:
        for entry in candidates:
            if entry["exists"]:
                librbln_path = os.path.realpath(entry["path"])
                break
    result.append(
        {
            "name": "librbln.so",
            "path": librbln_path,
            "gcc": get_gcc_version_from_elf(librbln_path) if librbln_path else "path not found",
        }
    )

    return result


def collect_diagnostics() -> dict[str, Any]:
    """Gather env snapshot, compiler package locations, and runtime-library resolution state."""
    candidates = _runtime_lib_candidates()
    d = {
        "torch_rbln": _torch_rbln_info(),
        "rebel_compiler": _rebel_compiler_info(),
        "rebel_abi": _rebel_abi_info(),
        "env": _env_snapshot(),
        "rebel": _package_location("rebel"),
        "runtime_lib": _runtime_lib_state(),
        "runtime_lib_candidates": candidates,
        "python_executable": sys.executable,
        "sys_path": list(sys.path),
    }
    try:
        print("Checking GCC versions in .so files (readelf)...", file=sys.stderr)
        d["gcc_versions"] = _resolve_so_paths(d)
    except Exception:
        d["gcc_versions"] = []
    return d


def format_diagnostics(d: dict[str, Any] | None = None, verbose: bool = True) -> str:
    """Format diagnostics for console output. If d is None, collects first."""
    if d is None:
        d = collect_diagnostics()
    lines = [
        "=== torch-rbln environment diagnostics ===",
        "",
        "torch-rbln package:",
    ]
    tr = d.get("torch_rbln") or {}
    lines.append(f"  version: {tr.get('version', 'N/A')}")
    lines.append(f"  install_type: {tr.get('install_type', 'N/A')}")
    lines.append(f"  loaded from: {tr.get('location', 'N/A')}")
    if tr.get("installed_location") is not None:
        lines.append(f"  installed at: {tr['installed_location']}")
    if tr.get("shadowed"):
        lines.append("  >>> Local package is shadowing the installed wheel (loaded from != installed at).")
        lines.append("      Run from outside the project or fix PYTHONPATH/sys.path to use the wheel.")
    if tr.get("lib_dir") is not None:
        lines.append(f"  lib_dir: {tr['lib_dir']} (exists: {tr.get('lib_dir_exists')})")
        if tr.get("native_libs"):
            lines.append(f"  native .so in lib: {', '.join(tr['native_libs'])}")
        else:
            lines.append("  >>> lib_dir is empty or has no .so files (native libs missing for this install).")
    lines.extend(
        [
            "",
            "rebel-compiler package:",
        ]
    )
    rc = d.get("rebel_compiler") or {}
    if rc.get("found"):
        lines.append(f"  version: {rc.get('version', 'N/A')}")
        lines.append(f"  install_type: {rc.get('install_type', 'N/A')}")
        lines.append(f"  loaded from: {rc.get('location', 'N/A')}")
        if rc.get("installed_location") is not None:
            lines.append(f"  installed at: {rc['installed_location']}")
    else:
        lines.append("  not installed" + (f" ({rc.get('error', '')})" if rc.get("error") else ""))
    lines.extend(
        [
            "",
            "rebel ABI handshake:",
        ]
    )
    abi = d.get("rebel_abi") or {}
    built_abi = abi.get("built_abi")
    lines.append(
        f"  built against ABI: {built_abi}"
        if built_abi is not None
        else "  built against ABI: (none recorded — built against a pre-handshake rebel-compiler)"
    )
    if abi.get("librbln_path"):
        lines.append(f"  librbln.so: {abi['librbln_path']}")
    if abi.get("runtime_current") is not None:
        lines.append(
            f"  runtime accepts: {abi.get('runtime_min_supported')}..{abi.get('runtime_current')} "
            f"(min_supported..current)"
        )
    if abi.get("verdict"):
        lines.append(f"  verdict: {abi['verdict']}")
    if abi.get("check_disabled"):
        lines.append("  >>> check is DISABLED via TORCH_RBLN_SKIP_ABI_CHECK; import will not enforce it.")
    if abi.get("error"):
        lines.append(f"  error: {abi['error']}")
    lines.extend(
        [
            "",
            "Environment variables (affecting library search):",
        ]
    )
    for k in ENV_VARS:
        v = d["env"].get(k, "")
        if not v and not verbose:
            continue
        if v:
            # One value per line for readability when multiple paths
            parts = v.replace(":", "\n    ").replace(";", "\n    ").split("\n")
            first = parts[0].strip()
            rest = [p.strip() for p in parts[1:] if p.strip()]
            if rest:
                lines.append(f"  {k}={first}")
                for p in rest:
                    lines.append(f"    {p}")
            else:
                lines.append(f"  {k}={first}")
        else:
            lines.append(f"  {k}= (not set)")
    lines.extend(
        [
            "",
            "Compiler packages (import locations):",
        ]
    )
    rebel_pkg = d.get("rebel") or {}
    if rebel_pkg.get("found"):
        lines.append(f"  rebel: {rebel_pkg.get('submodule_search_locations') or rebel_pkg.get('origin', 'N/A')}")
    else:
        lines.append("  rebel: not found" + (f" ({rebel_pkg.get('error', '')})" if rebel_pkg.get("error") else ""))
    runtime_lib = d.get("runtime_lib") or {}
    lines.extend(
        [
            "",
            "RBLN runtime library:",
            f"  loaded: {', '.join(runtime_lib.get('loaded') or []) or 'none'}",
            f"  version: {runtime_lib.get('version') or 'N/A'}",
        ]
    )
    lines.extend(
        [
            "",
            f"Candidates considered for {RUNTIME_LIB_NAME} (in order):",
        ]
    )
    for entry in d.get("runtime_lib_candidates") or []:
        exists = "exists" if entry["exists"] else "MISSING"
        lines.append(f"  [{exists}] {entry['path']}")
        lines.append(f"    from: {entry['source']}")
    lines.extend(
        [
            "",
            "GCC versions (from ELF .comment of .so files):",
        ]
    )
    for entry in d.get("gcc_versions") or []:
        name = entry.get("name", "?")
        path = entry.get("path")
        gcc = entry.get("gcc", "?")
        if path:
            lines.append(f"  {name}: {gcc}")
            lines.append(f"    path: {path}")
        else:
            lines.append(f"  {name}: {gcc}")
    lines.extend(
        [
            "",
            "Python: " + d.get("python_executable", sys.executable),
            "",
        ]
    )
    return "\n".join(lines)


def print_diagnostics(verbose: bool = True) -> None:
    """Print formatted diagnostics to stderr so they are visible even when stdout is captured."""
    d = collect_diagnostics()
    text = format_diagnostics(d, verbose=verbose)
    print(text, file=sys.stderr)
