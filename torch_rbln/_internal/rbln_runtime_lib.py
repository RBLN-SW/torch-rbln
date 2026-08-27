"""Locating and loading the RBLN runtime shared library (``librbln.so``).

The library is the one the ``rebel-compiler`` distribution recorded installing, anchored on the
``rebel`` package this interpreter imports. Recorded paths are relative to the install root,
which is also what the install RPATH is written against, so one record answers where the library
is, where it sits relative to the package, and where the headers are.

Every candidate comes from where that package sits, because rebel-compiler loads the library
relative to its own files too: a second copy answering here would be mapped alongside the one
rebel loads, leaving two runtimes -- two allocators, two device registries -- in one process, which
nothing downstream can detect when both were built from the same version.

``LD_LIBRARY_PATH`` is the override: the dynamic loader and rebel-compiler's loader both honour
it, so all three sides land on the same file.

Kept small and stdlib-only -- ``tools/find_rebel_runtime.py`` loads it by path during the build,
before torch_rbln exists, so anything it reaches for has to work then too.
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


def _rebel_package_anchors() -> list[str]:
    """Directories holding the ``rebel`` package this interpreter would import, first one first.

    ``find_spec`` because importing rebel pulls in torch. Anchoring on the imported package keeps
    an editable install, or a stray copy elsewhere, from answering for the one in use.

    ``origin`` leads: it names the ``__init__.py`` that will run, so it says which tree provides
    rebel even when several are on the path. The search locations follow only when there is no
    origin -- a namespace package, which a directory left behind by an uninstall also produces.
    They arrive from a set for an editable install, holding both the source tree and the build
    tree, so their order changes from process to process and taking one of them was a coin flip.
    """
    try:
        spec = importlib.util.find_spec("rebel")
    except (ImportError, ValueError):
        return []
    if spec is None:
        return []
    anchors = []
    if spec.origin:
        anchors.append(os.path.dirname(os.path.dirname(spec.origin)))
    for location in sorted(spec.submodule_search_locations or ()):
        directory = os.path.dirname(location)
        if directory not in anchors:
            anchors.append(directory)
    return anchors


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


def _record_candidates(entry: object, anchors: list[str]) -> Iterator[str]:
    """Both ways a recorded entry can name a file: under an anchor, and where it was installed.

    Where it was installed counts only when that is the tree the imported package comes from. A
    distribution recorded elsewhere -- a wheel installed in site-packages while a checkout on
    PYTHONPATH provides the package -- names the library belonging to its own Python, and
    rebel-compiler will load that checkout's library rather than this one.
    """
    for anchor in anchors:
        yield os.path.join(anchor, str(entry))
    located = entry.locate()  # type: ignore[attr-defined]
    if located is not None and _inside_an_anchor(str(located), anchors):
        yield str(located)


def _inside_an_anchor(path: str, anchors: list[str]) -> bool:
    """Whether ``path`` sits in a directory the ``rebel`` package is installed in.

    No anchor at all -- nothing imports rebel -- leaves a candidate standing: it is then the only
    statement anything has made about where the library is.
    """
    if not anchors:
        return True
    resolved = os.path.realpath(path)
    return any(resolved.startswith(os.path.realpath(anchor) + os.sep) for anchor in anchors)


def record_relative_dir(path: str) -> str | None:
    """Directory holding ``path`` relative to the install root, or None if it is not recorded.

    A recorded path already expresses the relationship the install RPATH is written against.
    """
    resolved = os.path.realpath(path)
    anchors = _rebel_package_anchors()
    for entry in _record_entries():
        for candidate in _record_candidates(entry, anchors):
            if os.path.realpath(candidate) == resolved:
                return os.path.dirname(str(entry))
    return None


def record_include_dir() -> str | None:
    """Include directory the distribution installed, located through a recorded header."""
    marker = f"{os.sep}include{os.sep}"
    header = next((e for e in _dist_files() if str(e).endswith(".h") and marker in str(e)), None)
    if header is None:
        return None
    anchors = _rebel_package_anchors()
    under = [os.path.join(anchor, str(header).split(marker)[0], "include") for anchor in anchors]
    located = header.locate()
    if located is not None and _inside_an_anchor(str(located), anchors):
        under.append(str(located).split(marker)[0] + os.sep + "include")
    if not under:
        return None
    # The build links against this; a directory that does not exist fails the CMake find instead.
    return next((path for path in under if os.path.isdir(path)), under[0])


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

    anchors = _rebel_package_anchors()

    for entry in _record_entries():
        for candidate in _record_candidates(entry, anchors):
            yield from emit(candidate, f"{_COMPILER_DIST_NAME} record")

    # An editable install records no library, and neither does a checkout that reached this
    # interpreter without being installed at all. rebel-compiler builds into <source>/build and
    # loads it from there itself, relative to its own files.
    for anchor in anchors:
        yield from emit(os.path.join(os.path.dirname(anchor), "build", RUNTIME_LIB_NAME), "rebel source build tree")


def runtime_library_candidates() -> list[tuple[str, str]]:
    """Every candidate path, in order, unfiltered by existence. For diagnostics."""
    return list(_iter_runtime_library_candidates())


def _absent_distribution_note() -> str:
    """Say it outright when nothing was found because rebel-compiler is not installed here.

    Every candidate is derived from an install, so with no install there is nothing to report but
    where this interpreter looked -- which reads like a path problem unless it says otherwise.
    """
    try:
        distribution(_COMPILER_DIST_NAME)
    except PackageNotFoundError:
        anchors = _rebel_package_anchors()
        resolves = f"`rebel` resolves under {', '.join(anchors)}" if anchors else "`rebel` does not import"
        return f" No {_COMPILER_DIST_NAME} distribution is installed for {sys.executable} ({resolves})."
    return ""


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

    listed = "; ".join(f"{path} (from {source})" for path, source in tried[:10]) or "nothing"
    env_hint = "; ".join(
        f"{name}={os.environ[name][:80]}" for name in ("LD_LIBRARY_PATH", "PYTHONPATH") if os.environ.get(name)
    )
    raise FileNotFoundError(
        f"Could not find {RUNTIME_LIB_NAME}.{_absent_distribution_note()} Tried (in order): {listed}"
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
        # /proc/self/maps marks a mapping whose file is gone with " (deleted)". That marker is
        # not part of the path and callers hand this string to dlopen, so it goes no further.
        return already_loaded[0].removesuffix(_DELETED_SUFFIX)

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
