"""Locating and loading the RBLN runtime shared library (``librbln.so``).

This was a vendored copy of ``tvm/_ffi/libinfo.py``, so torch-rbln followed TVM's directory list
to land on the file rebel-compiler loads. That coupling goes away as rebel-compiler drops TVM,
and the copy had drifted: it walked every ``PATH`` and ``LD_LIBRARY_PATH`` entry recursively.

The library is now the one the ``rebel-compiler`` distribution recorded installing, anchored on
the ``rebel`` package this interpreter imports. Recorded paths are relative to the install root,
which is also what the install RPATH is written against, so one record answers where the library
is, where it sits relative to the package, and where the headers are.

``LD_LIBRARY_PATH`` is the override: the dynamic loader and rebel-compiler's loader both honour
it, so all three sides land on the same file.
"""

import ctypes
import importlib.util
import os
import sys
from collections.abc import Iterator
from importlib.metadata import distribution, PackageNotFoundError


RUNTIME_LIB_NAME = "librbln.so"
_COMPILER_DIST_NAME = "rebel-compiler"

_MAPS_PATH = "/proc/self/maps"
_DELETED_SUFFIX = " (deleted)"

# The distribution's record of what it installed


def _split_env_paths(name: str) -> list[str]:
    """Non-empty entries of a ``:``-separated path variable (an empty entry means cwd)."""
    value = os.environ.get(name, "")
    return [entry for entry in (part.strip() for part in value.split(os.pathsep)) if entry]


def _rebel_package_anchor() -> str | None:
    """Directory holding the ``rebel`` package this interpreter would import, or None.

    ``find_spec`` because importing rebel pulls in torch. Anchoring on the imported package keeps
    an editable install, or a stray copy elsewhere, from answering for the one in use.
    """
    try:
        spec = importlib.util.find_spec("rebel")
    except (ImportError, ValueError):
        return None
    locations = getattr(spec, "submodule_search_locations", None) if spec else None
    if not locations:
        return None
    return os.path.dirname(locations[0])


def _dist_files() -> list:
    """Everything the installed rebel-compiler distribution recorded, empty when unavailable."""
    try:
        dist = distribution(_COMPILER_DIST_NAME)
    except PackageNotFoundError:
        return []
    return list(dist.files or [])


def _record_entries() -> list:
    """Recorded entries naming the runtime library."""
    return [entry for entry in _dist_files() if os.path.basename(str(entry)) == RUNTIME_LIB_NAME]


def _record_candidates(entry: object, anchor: str | None) -> Iterator[str]:
    """Both ways a recorded entry can name a file: under the anchor, and where it was installed."""
    if anchor is not None:
        yield os.path.join(anchor, str(entry))
    located = entry.locate()  # type: ignore[attr-defined]
    if located is not None:
        yield str(located)


def record_relative_dir(path: str) -> str | None:
    """Directory holding ``path`` relative to the install root, or None if it is not recorded.

    A recorded path already expresses the relationship the install RPATH is written against, so
    nothing is reconstructed from whichever site-packages the build ran out of.
    """
    resolved = os.path.realpath(path)
    anchor = _rebel_package_anchor()
    for entry in _record_entries():
        for candidate in _record_candidates(entry, anchor):
            if os.path.realpath(candidate) == resolved:
                return os.path.dirname(str(entry))
    return None


def record_include_dir() -> str | None:
    """Include directory the distribution installed, located through a recorded header."""
    anchor = _rebel_package_anchor()
    if anchor is None:
        return None
    marker = f"{os.sep}include{os.sep}"
    for entry in _dist_files():
        name = str(entry)
        if name.endswith(".h") and marker in name:
            return os.path.join(anchor, name.split(marker)[0], "include")
    return None


# Resolution


def _iter_runtime_library_candidates() -> Iterator[tuple[str, str]]:
    """Yield ``(path, source)`` candidates lazily, most authoritative first.

    Lazily because reading the record costs ~18 ms, which an override settles without paying.
    """
    seen: set[str] = set()

    def emit(path: str, source: str) -> Iterator[tuple[str, str]]:
        resolved = os.path.realpath(path)
        if resolved not in seen:
            seen.add(resolved)
            yield resolved, source

    # Ahead of the record: the one channel on which an override reaches all three loaders.
    for directory in _split_env_paths("LD_LIBRARY_PATH"):
        yield from emit(os.path.join(directory, RUNTIME_LIB_NAME), "LD_LIBRARY_PATH")

    anchor = _rebel_package_anchor()
    for entry in _record_entries():
        for candidate in _record_candidates(entry, anchor):
            yield from emit(candidate, f"{_COMPILER_DIST_NAME} record")

    # Editable install: the package is <source>/python/rebel, the library <source>/build, and the
    # record names neither. Behind the record so a leftover build tree cannot shadow it.
    if anchor is not None:
        yield from emit(os.path.join(os.path.dirname(anchor), "build", RUNTIME_LIB_NAME), "rebel source build tree")


def runtime_library_candidates() -> list[tuple[str, str]]:
    """Every candidate path, in order, unfiltered by existence. For diagnostics."""
    return list(_iter_runtime_library_candidates())


def resolve_runtime_library() -> tuple[str, str]:
    """First existing candidate as ``(path, source)``.

    Raises:
        FileNotFoundError: when no candidate exists, listing what was tried.
    """
    tried: list[tuple[str, str]] = []
    for path, source in _iter_runtime_library_candidates():
        if os.path.isfile(path):
            return path, source
        tried.append((path, source))

    listed = "; ".join(f"{path} (from {source})" for path, source in tried[:10])
    env_hint = "; ".join(
        f"{name}={os.environ[name][:80]}" for name in ("LD_LIBRARY_PATH", "PYTHONPATH") if os.environ.get(name)
    )
    raise FileNotFoundError(
        f"Could not find {RUNTIME_LIB_NAME}. Tried (in order): {listed}"
        + (" ..." if len(tried) > 10 else "")
        + ". Install the rebel-compiler package, or put the directory holding the library on "
        "LD_LIBRARY_PATH."
        + (f" Relevant env: {env_hint}." if env_hint else "")
        + " Run 'python -m torch_rbln.diagnose' (or 'TORCH_RBLN_DIAGNOSE=1 python -m torch_rbln.diagnose' if import"
        " fails) for full environment diagnostics."
    )


def load_runtime_library() -> str:
    """Map ``librbln.so`` before the native extensions load, and return its path.

    They declare it NEEDED by SONAME, so the loader reuses this mapping instead of searching a
    RUNPATH baked in at build time -- which is how a library that has since moved still resolves.
    An existing mapping is adopted, so whoever loaded it first keeps their copy.
    """
    already_loaded = loaded_runtime_libraries()
    if already_loaded:
        return already_loaded[0]

    path, source = resolve_runtime_library()
    try:
        ctypes.CDLL(path, ctypes.RTLD_GLOBAL)
    except OSError as e:
        raise OSError(f"Failed to load {path} (found via {source}): {e}") from e
    return path


# What is already mapped


def loaded_runtime_libraries() -> list[str]:
    """Absolute paths of every mapped ``librbln.so``; empty when the mapping table cannot be read.

    Used to adopt a library something mapped before us, and reported by diagnose.
    """
    if not sys.platform.startswith("linux"):
        return []

    paths: list[str] = []
    try:
        with open(_MAPS_PATH) as maps:
            for line in maps:
                # The mapped file path is the last field; anonymous mappings have none.
                fields = line.rstrip("\n").split(" ", 5)
                if len(fields) < 6:
                    continue
                path = fields[5].strip()
                if not path.startswith("/"):
                    continue
                if os.path.basename(path.removesuffix(_DELETED_SUFFIX)) != RUNTIME_LIB_NAME:
                    continue
                if path not in paths:
                    paths.append(path)
    except OSError:
        return []
    return paths
