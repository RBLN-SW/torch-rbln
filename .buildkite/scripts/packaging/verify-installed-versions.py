"""Check the installed torch-rbln against the wheel it came from.

The installed torch-rbln must report the wheel's version, carry the
debug marker exactly when it is a Debug build, and, with --torch, run on
that torch version.
"""

from __future__ import annotations

import argparse
import importlib


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--torch-rbln", required=True, help="the version of the installed wheel")
    parser.add_argument("--build-type", required=True, choices=["Release", "Debug"])
    parser.add_argument("--torch", help="the torch version the install must run on")
    args = parser.parse_args()

    rebel = importlib.import_module("rebel")
    torch = importlib.import_module("torch")
    torch_rbln = importlib.import_module("torch_rbln")
    print(f"torch={torch.__version__} torch_rbln={torch_rbln.__version__} rebel={rebel.__version__}")

    failures: list[str] = []
    if torch_rbln.__version__ != args.torch_rbln:
        failures.append(f"torch-rbln version mismatch: expected {args.torch_rbln}, got {torch_rbln.__version__}")
    debug = "debug" in torch_rbln.__version__
    if args.build_type == "Debug" and not debug:
        failures.append(f"Debug build is missing its debug marker: {torch_rbln.__version__}")
    if args.build_type == "Release" and debug:
        failures.append(f"Release build must not carry a debug marker: {torch_rbln.__version__}")
    if args.torch is not None and torch.__version__ != args.torch:
        failures.append(f"torch version mismatch: expected {args.torch}, got {torch.__version__}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"Verified torch-rbln=={torch_rbln.__version__} ({args.build_type}) on torch=={torch.__version__}")


if __name__ == "__main__":
    main()
