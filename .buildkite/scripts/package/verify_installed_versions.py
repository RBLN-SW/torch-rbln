"""Check the version and build type of the installed torch-rbln."""

from __future__ import annotations

import argparse

import rebel
import torch

import torch_rbln


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--torch-rbln-version", required=True)
    parser.add_argument("--build-type", required=True, choices=["Release", "Debug"])
    args = parser.parse_args()

    print(f"torch={torch.__version__} torch_rbln={torch_rbln.__version__} rebel={rebel.__version__}")

    failures: list[str] = []
    if torch_rbln.__version__ != args.torch_rbln_version:
        failures.append(
            f"torch-rbln version mismatch: expected {args.torch_rbln_version}, got {torch_rbln.__version__}"
        )
    has_debug_marker = "debug" in torch_rbln.__version__
    if args.build_type == "Debug" and not has_debug_marker:
        failures.append(f"Debug build is missing its debug marker: {torch_rbln.__version__}")
    if args.build_type == "Release" and has_debug_marker:
        failures.append(f"Release build carries a debug marker: {torch_rbln.__version__}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"Verified torch-rbln=={torch_rbln.__version__} ({args.build_type})")


if __name__ == "__main__":
    main()
