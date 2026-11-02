#!/usr/bin/env -S uv run --isolated --only-group ci --locked python
"""Pin torch in pyproject.toml to a version from a uv index."""

from __future__ import annotations

import argparse
from pathlib import Path

import tomlkit
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name


def pin_torch_requirement(requirements: list[str], version: str, location: str) -> None:
    positions = [i for i, text in enumerate(requirements) if canonicalize_name(Requirement(text).name) == "torch"]
    if len(positions) != 1:
        msg = f"Expected one torch requirement in {location}, found {len(positions)}"
        raise SystemExit(msg)
    (position,) = positions
    requirement = Requirement(requirements[position])
    requirement.specifier = SpecifierSet(f"=={version}")
    requirements[position] = str(requirement)
    print(f"{location}: {requirements[position]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--torch-version", required=True)
    parser.add_argument("--torch-index", required=True)
    args = parser.parse_args()
    path = Path("pyproject.toml")
    document = tomlkit.parse(path.read_text(encoding="utf-8"))
    pin_torch_requirement(document["project"]["dependencies"], args.torch_version, "[project].dependencies")
    pin_torch_requirement(document["build-system"]["requires"], args.torch_version, "[build-system].requires")
    pin_torch_requirement(document["dependency-groups"]["build"], args.torch_version, "[dependency-groups].build")
    document["tool"]["uv"]["sources"]["torch"]["index"] = args.torch_index
    print(f"[tool.uv.sources].torch: index {args.torch_index}")
    path.write_text(tomlkit.dumps(document), encoding="utf-8")


if __name__ == "__main__":
    main()
