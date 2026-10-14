"""Write what the RBLN runtime sees in this process to a file.

The environment checks read one line:

npu:     <runtime loaded> <UMD path or none> <physical NPUs or ->
no-npu:  <runtime loaded> <UMD path or none> <device count> <available>

The runtime may log to stdout, so the result goes to a file.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path


def loaded_umd() -> str:
    """Return the path of the mapped RBLN user-mode driver, or none."""
    with open("/proc/self/maps") as maps:
        return next((line.split()[-1] for line in maps if "librbln-thunk" in line), "none")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["npu", "no-npu"])
    parser.add_argument("result", type=Path, help="the file to write the line to")
    args = parser.parse_args()
    torch = importlib.import_module("torch")
    torch_rbln = importlib.import_module("torch_rbln")
    loaded = torch_rbln._C.runtime_loaded()
    umd = loaded_umd()
    if args.mode == "npu":
        # physical_device_count() segfaults unless the runtime has
        # loaded the user-mode driver.
        fields = [loaded, umd, torch.rbln.physical_device_count() if loaded else "-"]
    else:
        fields = [loaded, umd, torch.rbln.device_count(), torch.rbln.is_available()]
    args.result.write_text(" ".join(str(field) for field in fields) + "\n")


if __name__ == "__main__":
    main()
