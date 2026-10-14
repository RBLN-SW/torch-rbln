"""Check that torch-rbln finds librbln.so through its install RPATH.

Importing torch_rbln cannot show this: it ctypes-loads librbln.so
before the extension, so RUNPATH is never consulted. Run this with
TORCH_DEVICE_BACKEND_AUTOLOAD=0, without which importing torch loads
torch_rbln, and with LD_LIBRARY_PATH unset, which the loader consults
ahead of RUNPATH.
"""

from __future__ import annotations

import ctypes
import importlib
import os
import re
from importlib.metadata import distribution


def mapped_runtime_libs() -> list[str]:
    with open("/proc/self/maps") as maps:
        return sorted({os.path.realpath(p) for p in re.findall(r"/\S*librbln\.so", maps.read())})


def installed_file(package: str, name: str) -> str:
    """Return where name is installed, per the RECORD of package."""
    found = [f for f in distribution(package).files or [] if f.name == name]
    if len(found) != 1:
        raise SystemExit(f"Expected {package} to install one {name}, found {len(found)}")
    return str(found[0].locate())


def main() -> None:
    # Load libtorch first, so it resolves by SONAME and leaves
    # librbln.so to RUNPATH.
    importlib.import_module("torch")

    # If anything mapped librbln.so already, the load below would pass
    # no matter what the RPATH says.
    already = mapped_runtime_libs()
    if already:
        raise SystemExit(f"librbln.so was already mapped ({already}); RPATH untested")

    # Read the record here rather than through
    # torch_rbln._internal.rbln_runtime_lib: importing that runs
    # torch_backends_entry_point(), which maps the library itself.
    expected = os.path.realpath(installed_file("rebel-compiler", "librbln.so"))
    library = installed_file("torch-rbln", "libtorch_rbln.so")
    try:
        ctypes.CDLL(library)
    except OSError as e:
        raise SystemExit(f"Install RPATH does not resolve librbln.so: {e}") from e

    # Any librbln.so on the loader's search path satisfies the NEEDED
    # entry; RUNPATH must reach the one rebel-compiler installed.
    mapped = mapped_runtime_libs()
    if mapped != [expected]:
        raise SystemExit(f"RUNPATH resolved {mapped}, expected [{expected}]")
    print(f"Verified the install RPATH of {library} -> {expected}")


if __name__ == "__main__":
    main()
