#!/usr/bin/env -S uv run --isolated --only-group ci --locked python
"""Print the version for a source build of the pinned torch."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version


if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


def torch_build_version(pyproject: str, build_type: str) -> str:
    requirements = []
    for dependency in tomllib.loads(pyproject).get("project", {}).get("dependencies", []):
        try:
            requirement = Requirement(dependency)
        except InvalidRequirement as error:
            msg = f"Invalid requirement in [project].dependencies: {error}"
            raise SystemExit(msg) from None
        if canonicalize_name(requirement.name) == "torch":
            requirements.append(requirement)
    if len(requirements) != 1:
        msg = f"Expected one torch requirement in [project].dependencies, found {len(requirements)}"
        raise SystemExit(msg)
    (requirement,) = requirements
    specifiers = list(requirement.specifier)
    pin = specifiers[0].version if len(specifiers) == 1 and specifiers[0].operator == "==" else ""
    try:
        version: Version | None = Version(pin)
    except InvalidVersion:
        version = None
    if version is None or version.local != "cpu":
        msg = f"Expected a torch==<version>+cpu pin, got {requirement}"
        raise SystemExit(msg)
    if version.is_prerelease or version.is_postrelease or version.epoch or len(version.release) != 3:
        msg = f"Expected a final X.Y.Z torch release, which has a PyTorch tag, got {version.public}"
        raise SystemExit(msg)
    return f"{version.public}+cpu.{build_type.lower()}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build-type", required=True, choices=["Release", "Debug"])
    args = parser.parse_args()
    print(torch_build_version(Path("pyproject.toml").read_text(encoding="utf-8"), args.build_type))


if __name__ == "__main__":
    main()
