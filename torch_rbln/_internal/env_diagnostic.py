"""Environment diagnostics for loading torch-rbln on the rebel.v2 runtime.

Reports where the ``rebel.v2`` package imports from, the ``librebel_v2_rt.so`` it maps, the ABI id this
build recorded against the one the runtime reports, and torch-rbln's own native libraries -- what
an ``import torch_rbln`` that fails needs to be explained.
"""

import os
import re
import subprocess
import sys
from typing import Any

from torch_rbln._internal import abi_check
from torch_rbln._internal.rbln_runtime_lib import load_runtime_library, loaded_runtime_libraries, RUNTIME_LIB_NAME


# Environment variables that decide which rebel.v2 package and runtime this process picks up.
ENV_VARS = (
    "REBEL_HOME",
    "PYTHONPATH",
    "LD_LIBRARY_PATH",
    "PATH",
)

_VERDICT_TEXT = {
    abi_check.VERDICT_OK: "OK",
    abi_check.VERDICT_MISMATCH: "MISMATCH -- import torch_rbln raises; rebuild it against this runtime's headers",
    abi_check.VERDICT_SKIPPED_NO_SNAPSHOT: "no build-time ABI id to compare against",
    abi_check.VERDICT_SKIPPED_UNREADABLE_RUNTIME: "no handle could be taken on the mapped runtime",
    abi_check.VERDICT_SKIPPED_NO_RUNTIME_ID: f"the runtime exports no {abi_check.ABI_SYMBOL}()",
}


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
    """Snapshot of the env vars in ``ENV_VARS`` (empty = not set)."""
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


def _is_under(path: str, directory: str) -> bool:
    return os.path.realpath(path).startswith(os.path.realpath(directory) + os.sep)


def _rbln_runtime_info() -> dict[str, Any]:
    """Where ``rebel.v2`` imports from and the ``librebel_v2_rt.so`` importing it maps.

    Maps the runtime the same way ``import torch_rbln`` does, so this is the library the ABI
    check reads.
    """
    out: dict[str, Any] = {
        "package": _package_location("rebel.v2"),
        "module": None,
        "path": None,
        "mapped": [],
        "under_rebel_home": None,
        "error": None,
    }
    try:
        out["path"] = load_runtime_library()
    except Exception as e:
        out["error"] = str(e)
    module = sys.modules.get("rebel.v2.runtime")
    if module is not None:
        out["module"] = getattr(module, "__file__", None)
    out["mapped"] = loaded_runtime_libraries()
    rebel_home = os.environ.get("REBEL_HOME")
    if out["path"] and rebel_home:
        out["under_rebel_home"] = _is_under(out["path"], rebel_home)
    return out


def _abi_info(runtime_path: str | None) -> dict[str, Any]:
    """This build's ABI id against the one the mapped runtime reports, and the verdict."""
    out: dict[str, Any] = {
        "built_abi": None,
        "built_include_dir": None,
        "runtime_abi": None,
        "check_disabled": False,
        "verdict": None,
        "error": None,
    }
    out["check_disabled"] = abi_check.is_abi_check_disabled()
    try:
        state = abi_check.inspect_runtime_abi(runtime_path)
    except Exception as e:
        out["error"] = f"could not run the ABI check: {e}"
        return out
    out["built_abi"] = state.built_abi
    out["built_include_dir"] = state.built_include_dir
    out["runtime_abi"] = state.runtime_abi
    if runtime_path is None:
        out["verdict"] = "no runtime is mapped to compare against"
    else:
        out["verdict"] = _VERDICT_TEXT.get(state.verdict, state.verdict)
    return out


def _resolve_so_paths(d: dict[str, Any]) -> list[dict[str, Any]]:
    """Paths of libtorch, torch-rbln's native libraries and the runtime, with the GCC that built each."""
    result: list[dict[str, Any]] = []
    tr = d.get("torch_rbln") or {}

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

    runtime_path = (d.get("rbln_runtime") or {}).get("path")
    result.append(
        {
            "name": RUNTIME_LIB_NAME,
            "path": runtime_path,
            "gcc": get_gcc_version_from_elf(runtime_path) if runtime_path else "not mapped",
        }
    )
    return result


def collect_diagnostics() -> dict[str, Any]:
    """Gather the torch-rbln install, the rebel.v2 runtime it would run on, the ABI verdict and env."""
    runtime = _rbln_runtime_info()
    d = {
        "torch_rbln": _torch_rbln_info(),
        "rbln_runtime": runtime,
        "abi": _abi_info(runtime["path"]),
        "env": _env_snapshot(),
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

    rt = d.get("rbln_runtime") or {}
    package = rt.get("package") or {}
    lines.extend(["", "rebel.v2 runtime:"])
    if package.get("found"):
        lines.append(f"  rebel.v2 package: {package.get('origin') or package.get('submodule_search_locations')}")
    else:
        lines.append("  rebel.v2 package: not importable" + (f" ({package['error']})" if package.get("error") else ""))
        lines.append("  >>> Put the rebel.v2 package of a rebel-compiler tree on the path:")
        lines.append("      PYTHONPATH=$REBEL_HOME/rebel/python")
    if rt.get("module"):
        lines.append(f"  rebel.v2.runtime: {rt['module']}")
    lines.append(f"  {RUNTIME_LIB_NAME}: {rt.get('path') or 'not mapped'}")
    if len(rt.get("mapped") or []) > 1:
        lines.append(f"  >>> {len(rt['mapped'])} copies mapped: {', '.join(rt['mapped'])}")
    if rt.get("under_rebel_home") is False:
        lines.append("  >>> The runtime is not from REBEL_HOME, which a build of torch-rbln compiles against.")
    if rt.get("error"):
        lines.append(f"  error: {rt['error']}")

    abi = d.get("abi") or {}
    lines.extend(["", "rbln ABI check:"])
    lines.append(f"  built ABI:   {abi.get('built_abi') or '(none recorded)'}")
    if abi.get("built_include_dir"):
        lines.append(f"    headers:   {abi['built_include_dir']}")
    lines.append(f"  runtime ABI: {abi.get('runtime_abi') or '(not read)'}")
    if abi.get("verdict"):
        lines.append(f"  verdict: {abi['verdict']}")
    if abi.get("check_disabled"):
        lines.append("  >>> check is DISABLED via TORCH_RBLN_SKIP_ABI_CHECK; import will not enforce it.")
    if abi.get("error"):
        lines.append(f"  error: {abi['error']}")

    lines.extend(["", "Environment variables:"])
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

    lines.extend(["", "GCC versions (from ELF .comment of .so files):"])
    for entry in d.get("gcc_versions") or []:
        name = entry.get("name", "?")
        path = entry.get("path")
        gcc = entry.get("gcc", "?")
        lines.append(f"  {name}: {gcc}")
        if path:
            lines.append(f"    path: {path}")
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
