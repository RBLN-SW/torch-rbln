"""TVM shared library path discovery utilities.

This module is derived from ``tvm/_ffi/libinfo.py`` in the Apache TVM project.
It is intentionally duplicated here so that torch-rbln can locate TVM shared
libraries (needed by the RBLN compiler backend) without requiring the ``tvm``
Python package to be installed as a direct dependency.

If TVM's library-loading logic changes upstream, this file should be updated
accordingly.
"""

import importlib.util
import os
import sys


def split_env_var(env_var, split):
    """Splits environment variable string.

    Parameters
    ----------
    env_var : str
        Name of environment variable.

    split : str
        String to split env_var on.

    Returns
    -------
    splits : list(string)
        If env_var exists, split env_var. Otherwise, empty list.
    """
    if os.environ.get(env_var, None):
        # Drop empty entries: a trailing/leading/double separator in
        # LD_LIBRARY_PATH etc. yields "", which os.path.realpath("") resolves
        # to cwd — injecting the current directory as a search path.
        return [p.strip() for p in os.environ[env_var].split(split) if p.strip()]
    return []


def get_dll_directories():
    """Get the possible dll directories"""
    # NB: This will either be the source directory (if TVM is run
    # inplace) or the install directory (if TVM is installed).
    # An installed TVM's curr_path will look something like:
    #   $PREFIX/lib/python3.6/site-packages/tvm/_ffi

    # Find the actual tvm package location (handles User/Global/Venv automatically)
    tvm_spec = importlib.util.find_spec("tvm")
    if tvm_spec is None or not tvm_spec.submodule_search_locations:
        ffi_dir = None
        source_dir = None
        install_lib_dir = None
    else:
        tvm_path = tvm_spec.submodule_search_locations[0]
        ffi_dir = os.path.join(tvm_path, "_ffi")
        source_dir = os.path.join(ffi_dir, "..", "..", "..")
        install_lib_dir = os.path.join(ffi_dir, "..", "..", "..", "..")

    dll_path = []

    if os.environ.get("TVM_LIBRARY_PATH", None):
        dll_path.append(os.environ["TVM_LIBRARY_PATH"])

    if sys.platform.startswith("linux") or sys.platform.startswith("freebsd"):
        dll_path.extend(split_env_var("LD_LIBRARY_PATH", ":"))
        dll_path.extend(split_env_var("PATH", ":"))
    elif sys.platform.startswith("darwin"):
        dll_path.extend(split_env_var("DYLD_LIBRARY_PATH", ":"))
        dll_path.extend(split_env_var("PATH", ":"))
    elif sys.platform.startswith("win32"):
        dll_path.extend(split_env_var("PATH", ";"))

    if ffi_dir is not None:
        # Pip lib directory
        dll_path.append(os.path.join(ffi_dir, ".."))
        # Default cmake build directory
        dll_path.append(os.path.join(source_dir, "build"))
        dll_path.append(os.path.join(source_dir, "build", "Release"))
        # Default make build directory
        dll_path.append(os.path.join(source_dir, "lib"))
        dll_path.append(install_lib_dir)

    # use extra TVM_HOME environment for finding libraries.
    if os.environ.get("TVM_HOME", None):
        tvm_source_home_dir = os.environ["TVM_HOME"]
    else:
        tvm_source_home_dir = source_dir

    if tvm_source_home_dir and os.path.isdir(tvm_source_home_dir):
        dll_path.append(os.path.join(tvm_source_home_dir, "web", "dist", "wasm"))
        dll_path.append(os.path.join(tvm_source_home_dir, "web", "dist"))

    if os.environ.get("REBEL_HOME", None):
        dll_path.append(os.path.join(os.environ["REBEL_HOME"], "build"))

    dll_path = [os.path.realpath(x) for x in dll_path]
    return [x for x in dll_path if os.path.isdir(x)]


def get_dll_directory_candidates():
    """Return (path, exists) for every candidate directory used by get_dll_directories.

    Useful for diagnostics: shows which paths were considered and which exist.
    """
    # Reuse the same logic as get_dll_directories but do not filter by existence.
    tvm_spec = importlib.util.find_spec("tvm")
    if tvm_spec is None or not tvm_spec.submodule_search_locations:
        ffi_dir = None
        source_dir = None
        install_lib_dir = None
    else:
        tvm_path = tvm_spec.submodule_search_locations[0]
        ffi_dir = os.path.join(tvm_path, "_ffi")
        source_dir = os.path.join(ffi_dir, "..", "..", "..")
        install_lib_dir = os.path.join(ffi_dir, "..", "..", "..", "..")

    dll_path = []

    if os.environ.get("TVM_LIBRARY_PATH", None):
        dll_path.append(os.environ["TVM_LIBRARY_PATH"])

    if sys.platform.startswith("linux") or sys.platform.startswith("freebsd"):
        dll_path.extend(split_env_var("LD_LIBRARY_PATH", ":"))
        dll_path.extend(split_env_var("PATH", ":"))
    elif sys.platform.startswith("darwin"):
        dll_path.extend(split_env_var("DYLD_LIBRARY_PATH", ":"))
        dll_path.extend(split_env_var("PATH", ":"))
    elif sys.platform.startswith("win32"):
        dll_path.extend(split_env_var("PATH", ";"))

    if ffi_dir is not None:
        dll_path.append(os.path.join(ffi_dir, ".."))
        dll_path.append(os.path.join(source_dir, "build"))
        dll_path.append(os.path.join(source_dir, "build", "Release"))
        dll_path.append(os.path.join(source_dir, "lib"))
        dll_path.append(install_lib_dir)

    if os.environ.get("TVM_HOME", None):
        tvm_source_home_dir = os.environ["TVM_HOME"]
    else:
        tvm_source_home_dir = source_dir

    if tvm_source_home_dir:
        dll_path.append(os.path.join(tvm_source_home_dir, "web", "dist", "wasm"))
        dll_path.append(os.path.join(tvm_source_home_dir, "web", "dist"))

    if os.environ.get("REBEL_HOME", None):
        dll_path.append(os.path.join(os.environ["REBEL_HOME"], "build"))

    seen = set()
    out = []
    for x in dll_path:
        try:
            r = os.path.realpath(x)
        except OSError:
            r = os.path.abspath(x)
        if r not in seen:
            seen.add(r)
            out.append((r, os.path.isdir(r)))
    return out
