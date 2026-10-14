# /// script
# requires-python = ">=3.10,<3.15"
# dependencies = [
#   "tomli==2.4.1; python_version < '3.11'",
# ]
# ///
"""Print the value of a key of a named uv index in pyproject.toml.

The index is the [[tool.uv.index]] entry with the given name; an entry
without the key prints an empty line.
"""

from __future__ import annotations

import argparse
import sys


if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name", help="the index name")
    parser.add_argument("key", help="a key of the entry, such as url or publish-url")
    args = parser.parse_args()
    with open("pyproject.toml", "rb") as f:
        indexes = tomllib.load(f).get("tool", {}).get("uv", {}).get("index", [])
    named = [index for index in indexes if index.get("name") == args.name]
    if not named:
        parser.error(f"no [[tool.uv.index]] named {args.name}")
    print(named[0].get(args.key, ""))


if __name__ == "__main__":
    main()
