# /// script
# requires-python = ">=3.10,<3.15"
# dependencies = [
#   "packaging==26.3",
# ]
# ///
"""Print a wheel's version, then dev or release."""

from __future__ import annotations

import argparse
from pathlib import Path

from packaging.utils import InvalidWheelFilename, parse_wheel_filename


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    try:
        _, version, _, _ = parse_wheel_filename(args.wheel.name)
    except InvalidWheelFilename as error:
        raise SystemExit(f"Cannot parse the wheel name: {error}") from None
    print(version, "dev" if version.is_devrelease else "release")


if __name__ == "__main__":
    main()
