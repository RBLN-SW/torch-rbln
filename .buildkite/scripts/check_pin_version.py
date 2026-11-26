#!/usr/bin/env -S uv run --isolated --only-group ci --locked python
"""Exit with an error unless the version can be written into a pin as it is.

That is a public version in its normalized form, such as 0.12.1a1: the form a
release tag takes after its leading "v". Forms that only normalize to one, such
as v0.12.1a1 or 0.12.1.a1, and local versions are rejected.
"""

import argparse
import sys

from packaging.version import InvalidVersion, Version


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    try:
        version = Version(args.version)
    except InvalidVersion:
        print(f"{args.version} is not a version; give one such as 0.12.1a1", file=sys.stderr)
        sys.exit(2)
    if version.local is not None or str(version) != args.version:
        print(f"{args.version} is not a normalized public version; use {version.public}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
