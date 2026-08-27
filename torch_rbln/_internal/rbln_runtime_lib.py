"""Locating and loading the RBLN runtime shared library (``librbln.so``).

The library is the one the ``rebel-compiler`` distribution recorded installing, anchored on the
``rebel`` package this interpreter imports. Recorded paths are relative to the install root,
which is also what the install RPATH is written against, so one record answers where the library
is, where it sits relative to the package, and where the headers are.

An editable install records no library, so two statements the install itself makes answer instead:
the import hook it registers, which names the files it installed, and the source directory
``direct_url.json`` records it was installed from. Both are exact; neither depends on a layout.

``LD_LIBRARY_PATH`` is the override: the dynamic loader and rebel-compiler's loader both honour
it, so all three sides land on the same file.

Kept small and stdlib-only -- ``tools/find_rebel_runtime.py`` loads it by path during the build,
before torch_rbln exists, so anything it reaches for has to work then too.
"""

import ctypes
import importlib.util
import json
import os
import sys
from collections.abc import Iterator
from importlib.metadata import distribution, PackageNotFoundError


RUNTIME_LIB_NAME = "librbln.so"
_COMPILER_DIST_NAME = "rebel-compiler"

_MAPS_PATH = "/proc/self/maps"
_DELETED_SUFFIX = " (deleted)"

# PEP 610: what an installer records about where it installed a distribution from.
_DIRECT_URL_FILE = "direct_url.json"
_FILE_URL_PREFIX = "file://"

# The distribution's record of what it installed


def _split_env_paths(name: str) -> list[str]:
    """Non-empty entries of a ``:``-separated path variable (an empty entry means cwd)."""
    value = os.environ.get(name, "")
    return [entry for entry in (part.strip() for part in value.split(os.pathsep)) if entry]


def _rebel_package_anchors() -> list[str]:
    """Directories holding the ``rebel`` package this interpreter would import.

    ``find_spec`` because importing rebel pulls in torch. Anchoring on the imported package keeps
    an editable install, or a stray copy elsewhere, from answering for the one in use.

    Every location, sorted, rather than the first one: an editable install reports its search
    locations from a set holding both the source tree and the build tree, so which one comes first
    changes from process to process and taking one of them made resolution a coin flip.
    """
    try:
        spec = importlib.util.find_spec("rebel")
    except (ImportError, ValueError):
        return []
    locations = getattr(spec, "submodule_search_locations", None) if spec else None
    if not locations:
        return []
    return sorted({os.path.dirname(location) for location in locations})


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
    """Both ways a recorded entry can name a file: under an anchor, and where it was installed."""
    for anchor in anchors:
        yield os.path.join(anchor, str(entry))
    located = entry.locate()  # type: ignore[attr-defined]
    if located is not None:
        yield str(located)


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
    recorded = next((str(e) for e in _dist_files() if str(e).endswith(".h") and marker in str(e)), None)
    anchors = _rebel_package_anchors()
    if recorded is None or not anchors:
        return None
    return os.path.join(anchors[0], recorded.split(marker)[0], "include")


# What an editable install says about itself


def _editable_source_root() -> str | None:
    """Directory an editable rebel-compiler install was installed from, None when it is not one.

    An editable install records no library of its own, and PEP 610 leaves the directory it was
    installed from in ``direct_url.json`` -- the one exact statement of where the checkout, and
    with it the C++ build output, lives.
    """
    try:
        recorded = distribution(_COMPILER_DIST_NAME).read_text(_DIRECT_URL_FILE)
    except (PackageNotFoundError, OSError):
        return None
    try:
        described = json.loads(recorded) if recorded else None
    except ValueError:
        return None
    if not isinstance(described, dict):
        return None
    directory = described.get("dir_info")
    url = described.get("url")
    if not isinstance(directory, dict) or not directory.get("editable"):
        return None
    if not isinstance(url, str) or not url.startswith(_FILE_URL_PREFIX):
        return None
    from urllib.parse import unquote  # not at module scope: this runs at `import torch`

    return unquote(url[len(_FILE_URL_PREFIX) :])


def _import_hook_libraries() -> Iterator[str]:
    """Library paths an editable install's import hook already holds.

    A hook that redirects imports into a build tree carries the map of files the build installed,
    and the library is one of them -- named outright, so it holds even when the build output was
    configured somewhere this could not have guessed.
    """
    for finder in list(sys.meta_path):
        base = getattr(finder, "dir", None)
        for attribute in ("known_wheel_files", "known_source_files"):
            mapping = getattr(finder, attribute, None)
            if not isinstance(mapping, dict):
                continue
            for target in mapping.values():
                if not isinstance(target, str) or os.path.basename(target) != RUNTIME_LIB_NAME:
                    continue
                if os.path.isabs(target):
                    yield target
                elif isinstance(base, str):
                    yield os.path.join(base, target)


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

    # Ahead of the record: an editable install's hook takes `import rebel` over from site-packages,
    # so what it holds is what this interpreter would load.
    for path in _import_hook_libraries():
        yield from emit(path, "editable install import hook")

    anchors = _rebel_package_anchors()
    for entry in _record_entries():
        for candidate in _record_candidates(entry, anchors):
            yield from emit(candidate, f"{_COMPILER_DIST_NAME} record")

    # Editable install: the record names no library, so the checkout it points at answers instead.
    # rebel-compiler installs from <source>/python and its REBEL_BUILD_DIR defaults to ../build.
    root = _editable_source_root()
    if root is not None:
        candidate = os.path.join(os.path.dirname(root), "build", RUNTIME_LIB_NAME)
        yield from emit(candidate, "editable install source tree")

    # Last: the same build tree derived from where the package sits, for a checkout that reached
    # this interpreter without being installed at all. Behind everything that states a path.
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
