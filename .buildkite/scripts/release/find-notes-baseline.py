# /// script
# requires-python = ">=3.10,<3.15"
# dependencies = [
#   "packaging==26.3",
# ]
# ///
"""Print the newest final or post release tag below the given one.

The candidates are the tags on stdin, compared in PEP 440 order. Tags
other than v followed by a normalized PEP 440 version are ignored.
"""

from __future__ import annotations

import argparse
import sys

from packaging.version import InvalidVersion, Version


def shipped(tag: str) -> Version | None:
    """Return a final or post release tag's version, or None."""
    try:
        version = Version(tag.removeprefix("v"))
    except InvalidVersion:
        return None
    if not tag.startswith("v") or str(version) != tag[1:] or version.is_prerelease or version.local:
        return None
    return version


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tag")
    current = Version(parser.parse_args().tag.removeprefix("v"))
    below = [
        (version, tag)
        for tag in sys.stdin.read().split()
        if (version := shipped(tag)) is not None and version < current
    ]
    if below:
        print(max(below)[1])


if __name__ == "__main__":
    main()
