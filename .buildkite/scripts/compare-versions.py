#!/usr/bin/env -S uv run --isolated --only-group ci --locked python
"""Print true when the candidate version is later than the baseline.

The comparison ignores local version labels, so a rebuild of the same
version is not later.
"""

import argparse

from packaging.version import InvalidVersion, Version


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", required=True)
    args = parser.parse_args()
    try:
        candidate, baseline = (Version(Version(text).public) for text in (args.candidate, args.baseline))
    except InvalidVersion as error:
        raise SystemExit(str(error)) from None
    print("true" if candidate > baseline else "false")


if __name__ == "__main__":
    main()
