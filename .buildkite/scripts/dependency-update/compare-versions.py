# /// script
# requires-python = ">=3.10,<3.15"
# dependencies = [
#   "packaging==26.3",
# ]
# ///
"""Print true when the first version is newer than the second.

Versions compare in PEP 440 order by their public part, so a rebuild
that differs in its local version alone is not newer.
"""

import argparse
import sys

from packaging.version import InvalidVersion, Version


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("first")
    parser.add_argument("second")
    args = parser.parse_args()
    try:
        newer, older = (Version(Version(text).public) for text in (args.first, args.second))
    except InvalidVersion as error:
        print(error, file=sys.stderr)
        return 1
    print("true" if newer > older else "false")
    return 0


if __name__ == "__main__":
    sys.exit(main())
