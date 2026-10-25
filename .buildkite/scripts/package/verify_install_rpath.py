"""Check that torch-rbln finds the libraries it links through its RPATH.

Importing torch_rbln cannot test this, because torch and librbln.so are
mapped before the extension loads. Run with LD_LIBRARY_PATH unset, which
the loader searches before RUNPATH.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import re
from importlib.metadata import distribution
from pathlib import Path


LINKED_LIBRARIES = {"torch": ("libc10.so", "libtorch.so", "libtorch_cpu.so"), "rebel-compiler": ("librbln.so",)}


def mapped_paths(name: str) -> list[str]:
    maps = Path("/proc/self/maps").read_text()
    return sorted({os.path.realpath(p) for p in re.findall(rf"/\S*/{re.escape(name)}$", maps, re.MULTILINE)})


def installed_file(package: str, name: str) -> str:
    matches = [f for f in distribution(package).files or [] if f.name == name]
    if len(matches) != 1:
        msg = f"Expected {package} to install one {name}, found {len(matches)}"
        raise SystemExit(msg)
    return os.path.realpath(matches[0].locate())


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    expected = {name: installed_file(package, name) for package, names in LINKED_LIBRARIES.items() for name in names}
    already_mapped = [name for name in expected if mapped_paths(name)]
    if already_mapped:
        msg = f"Could not test the RPATH: {', '.join(already_mapped)} loaded before the extension"
        raise SystemExit(msg)
    extension_path = installed_file("torch-rbln", "libtorch_rbln.so")
    try:
        ctypes.CDLL(extension_path)
    except OSError as error:
        msg = f"Install RPATH does not resolve the libraries {extension_path} links: {error}"
        raise SystemExit(msg) from None

    # The loader takes any copy on its search path, so each mapped file
    # must also be the one its package installed.
    actual = {name: mapped_paths(name) for name in expected}
    mismatches = [
        f"{name} to {', '.join(actual[name]) or 'no file'}, expected {path}"
        for name, path in expected.items()
        if actual[name] != [path]
    ]
    if mismatches:
        msg = f"The install RPATH resolved {'; '.join(mismatches)}"
        raise SystemExit(msg)
    summary = ", ".join(f"{name} -> {path}" for name, path in expected.items())
    print(f"Verified the install RPATH of {extension_path}: {summary}")


if __name__ == "__main__":
    main()
