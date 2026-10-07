"""The rebel.v2 runtime library (``librebel_v2_rt.so``) this process runs on.

Importing ``rebel.v2.runtime`` maps the runtime: its extension links ``librebel_v2_rt.so`` through
its own RPATH. torch-rbln's native libraries declare the same library NEEDED by SONAME, so once it
is mapped the dynamic loader reuses that mapping for them instead of searching their RUNPATH. The
process then holds one runtime -- one allocator, one device registry -- shared by ``rebel.v2``
and torch-rbln.

Stdlib-only apart from the ``rebel.v2`` import itself, so diagnose can use it when torch-rbln's own
native libraries are missing or broken.
"""

import os
import sys
from collections.abc import Iterable


RUNTIME_LIB_NAME = "librebel_v2_rt.so"
RUNTIME_PACKAGE = "rebel.v2.runtime"

_MAPS_PATH = "/proc/self/maps"
_DELETED_SUFFIX = " (deleted)"
_DIAGNOSE_HINT = "Run `python -m torch_rbln.diagnose` for the full environment report."


def parse_mapped_libraries(lines: Iterable[str], name: str = RUNTIME_LIB_NAME) -> list[str]:
    """Distinct paths of the files named ``name`` in ``/proc/<pid>/maps`` lines, first mapped first.

    The kernel marks a mapping whose file has since been unlinked with `` (deleted)``. That marker
    is not part of the path and callers hand the result to dlopen, so it is dropped.
    """
    paths: list[str] = []
    for line in lines:
        # The mapped file is the sixth field, which can contain spaces; anonymous mappings have none.
        fields = line.rstrip("\n").split(maxsplit=5)
        if len(fields) < 6 or not fields[5].startswith("/"):
            continue
        path = fields[5].removesuffix(_DELETED_SUFFIX)
        if os.path.basename(path) == name and path not in paths:
            paths.append(path)
    return paths


def loaded_runtime_libraries() -> list[str]:
    """Paths of every mapped ``librebel_v2_rt.so``; empty when the mapping table cannot be read."""
    if not sys.platform.startswith("linux"):
        return []
    try:
        with open(_MAPS_PATH) as maps:
            return parse_mapped_libraries(maps)
    except OSError:
        return []


def import_runtime_package():
    """Import ``rebel.v2.runtime``, which maps ``librebel_v2_rt.so``, and return the module.

    Raises:
        ImportError: ``rebel.v2`` cannot be imported, with what to install or put on the path.
    """
    try:
        from rebel.v2 import runtime
    except ImportError as e:
        raise ImportError(
            f"torch-rbln runs on the rebel.v2 runtime, but `import {RUNTIME_PACKAGE}` failed: {e}. "
            "Put the rebel package of the rebel-compiler tree torch-rbln was built against on the "
            f"path (PYTHONPATH=$REBEL_HOME/rebel/python). {_DIAGNOSE_HINT}"
        ) from e
    return runtime


def load_runtime_library() -> str:
    """Map the runtime through ``rebel.v2.runtime`` and return the path of its
    ``librebel_v2_rt.so``.

    Raises:
        ImportError: ``rebel.v2.runtime`` cannot be imported, maps no ``librebel_v2_rt.so``, or
            more than one copy is mapped -- torch-rbln would bind to one of them and ``rebel.v2``
            possibly to the other.
    """
    module = import_runtime_package()
    mapped = loaded_runtime_libraries()
    if not mapped:
        raise ImportError(
            f"`{RUNTIME_PACKAGE}` imported from {getattr(module, '__file__', 'unknown')}, but no "
            f"{RUNTIME_LIB_NAME} is mapped into this process (or {_MAPS_PATH} cannot be read). "
            f"{_DIAGNOSE_HINT}"
        )
    if len(mapped) > 1:
        raise ImportError(
            f"{len(mapped)} copies of {RUNTIME_LIB_NAME} are mapped into this process "
            f"({', '.join(mapped)}); rebel.v2 and torch-rbln must share one runtime, so nothing may "
            f"open another copy by path. {_DIAGNOSE_HINT}"
        )
    return mapped[0]
