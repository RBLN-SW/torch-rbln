#!/usr/bin/env python3
"""Locate the rebel runtime library and headers for the build.

Runs the same resolver torch-rbln uses at import time, so the build links against the library
that will actually be loaded and neither side hard-codes where rebel-compiler keeps it. Loaded
by path because torch_rbln cannot be imported before it is built.

Every answer comes from the rebel-compiler distribution's own record of what it installed, whose
paths are already relative to the install root -- the same relationship the install RPATH is
written against. Nothing is reconstructed from whichever site-packages the build runs out of,
which a PEP 517 overlay or a split purelib/platlib would get wrong.

Prints ``KEY=VALUE`` lines for CMake: LIBRARY, LIBRARY_DIR, LIBRARY_RELDIR (empty
when the library is not recorded, e.g. an external tree) and INCLUDE_DIR.
"""

import argparse
import importlib.util
import os
import sys
from typing import Any


_RESOLVER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "torch_rbln", "_internal", "rbln_runtime_lib.py"
)


def _load_resolver() -> Any:
    spec = importlib.util.spec_from_file_location("_rbln_runtime_lib_for_build", _RESOLVER)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load the runtime-library resolver from {_RESOLVER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library-dir", help="use this directory instead of resolving (external rebel tree)")
    args = parser.parse_args()

    resolver = _load_resolver()

    if args.library_dir:
        library_dir = os.path.realpath(args.library_dir)
        library = os.path.join(library_dir, resolver.RUNTIME_LIB_NAME)
    else:
        try:
            library, _source = resolver.resolve_runtime_library()
        except FileNotFoundError as e:
            print(f"ERROR={e}")
            return 1
        library_dir = os.path.dirname(library)

    reldir = resolver.record_relative_dir(library) or ""
    if not reldir:
        # Legitimate for an external tree, a bug anywhere else: the RPATH then names a symlink
        # that only exists next to the install prefix. On stderr, since CMake parses stdout.
        print(
            f"NOTE: {library} is not recorded by the rebel-compiler distribution, so the install "
            "RPATH will use the neutral symlink name instead of following the library.",
            file=sys.stderr,
        )

    print(f"LIBRARY={library}")
    print(f"LIBRARY_DIR={library_dir}")
    print(f"LIBRARY_RELDIR={reldir}")
    print(f"INCLUDE_DIR={resolver.record_include_dir() or ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
