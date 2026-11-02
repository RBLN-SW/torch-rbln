#!/usr/bin/env -S uv run --isolated --only-group ci --locked python
"""Print a setting of a [[tool.uv.index]] entry in pyproject.toml."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--index", metavar="NAME")
    group.add_argument("--index-of", metavar="PACKAGE")
    parser.add_argument("--setting", required=True)
    args = parser.parse_args()
    uv_config = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8")).get("tool", {}).get("uv", {})
    index_name = args.index
    if args.index_of:
        index_name = uv_config.get("sources", {}).get(args.index_of, {}).get("index")
        if not index_name:
            parser.error(f"[tool.uv.sources] has no index for {args.index_of}")
    index_entry = next((index for index in uv_config.get("index", []) if index.get("name") == index_name), None)
    if index_entry is None:
        parser.error(f"no [[tool.uv.index]] named {index_name}")
    value = str(index_entry.get(args.setting, ""))
    if not value:
        parser.error(f"the {index_name} index has no value for {args.setting}")
    print(value)


if __name__ == "__main__":
    main()
