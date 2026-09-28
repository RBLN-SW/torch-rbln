# /// script
# requires-python = ">=3.10,<3.15"
# dependencies = [
#   "packaging==26.3",
#   "tomlkit==0.15.1",
# ]
# ///
"""Pin rebel-compiler to one build in pyproject.toml."""

import argparse
import sys
from collections.abc import MutableSequence
from pathlib import Path
from typing import Any

import tomlkit  # type: ignore[import-not-found]
from packaging.requirements import Requirement
from packaging.specifiers import InvalidSpecifier, Specifier
from packaging.version import InvalidVersion, Version


PACKAGE = "rebel-compiler"
PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def package_entry_index(array: MutableSequence[Any], location: str) -> int:
    hits = [i for i, item in enumerate(array) if Requirement(str(item)).name == PACKAGE]
    if len(hits) != 1:
        sys.exit(f"Expected one {PACKAGE} entry in {location}, got {len(hits)}")
    return hits[0]


def pinned_build(uv_table: Any) -> str:
    build_pin, pin = (
        str(uv_table[key][package_entry_index(uv_table[key], f"[tool.uv].{key}")])
        for key in ("build-constraint-dependencies", "constraint-dependencies")
    )
    if build_pin != pin:
        sys.exit(f"build-constraint-dependencies '{build_pin}' does not match constraint-dependencies '{pin}'")
    specs = list(Requirement(pin).specifier)
    if len(specs) != 1 or specs[0].operator != "==":
        sys.exit(f"Expected {PACKAGE}==<build>, got '{pin}'")
    return specs[0].version


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("build", nargs="?", help="build to pin, including any local segment")
    target.add_argument("--current", action="store_true", help="print the pinned build")
    options = parser.parse_args()

    with open(PYPROJECT, encoding="utf-8") as f:
        doc: Any = tomlkit.parse(f.read())
    uv_table = doc["tool"]["uv"]
    current = pinned_build(uv_table)
    if options.current:
        print(current)
        return

    try:
        build = Version(options.build)
        Specifier(f"~={build.public}")
    except (InvalidVersion, InvalidSpecifier):
        sys.exit(f"'{options.build}' is not a PEP 440 version that ~= accepts")
    exact = f"{PACKAGE}=={build}"
    compatible = f"{PACKAGE}~={build.public}"
    entries = [
        (uv_table["build-constraint-dependencies"], exact, "[tool.uv].build-constraint-dependencies"),
        (uv_table["constraint-dependencies"], exact, "[tool.uv].constraint-dependencies"),
        (doc["project"]["optional-dependencies"]["runtime"], compatible, "[project.optional-dependencies].runtime"),
    ]
    for array, replacement, location in entries:
        array[package_entry_index(array, location)] = replacement
    with open(PYPROJECT, "w", encoding="utf-8") as f:
        f.write(tomlkit.dumps(doc))
    print(f"Pinned {PACKAGE} to {build} (was {current})")


if __name__ == "__main__":
    main()
