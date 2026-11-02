#!/usr/bin/env -S uv run --isolated --only-group ci --locked python
"""Print a wheel's version."""

from __future__ import annotations

import argparse
from pathlib import Path

from packaging.utils import InvalidWheelFilename, parse_wheel_filename


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wheel", required=True, type=Path)
    args = parser.parse_args()
    try:
        _, version, _, _ = parse_wheel_filename(args.wheel.name)
    except InvalidWheelFilename as error:
        msg = f"Could not parse the wheel filename: {error}"
        raise SystemExit(msg) from None
    print(version)


if __name__ == "__main__":
    main()
