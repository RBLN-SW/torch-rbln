# /// script
# requires-python = ">=3.10,<3.15"
# dependencies = [
#   "packaging==26.3",
#   "tomli==2.4.1; python_version < '3.11'",
# ]
# ///
"""Print the version of the debug build of the torch pin.

A torch==X.Y.Z+cpu pin gives X.Y.Z+cpu.debug.
"""

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


def debug_torch_version(pyproject: str) -> str:
    """Return the debug version of the torch pin in pyproject text."""
    requirements = []
    for dependency in tomllib.loads(pyproject).get("project", {}).get("dependencies", []):
        try:
            requirement = Requirement(dependency)
        except InvalidRequirement as error:
            raise SystemExit(f"Invalid requirement in [project].dependencies: {error}") from None
        if canonicalize_name(requirement.name) == "torch":
            requirements.append(requirement)
    if len(requirements) != 1:
        raise SystemExit(f"Expected one torch requirement in [project].dependencies, found {len(requirements)}")
    (requirement,) = requirements
    specifiers = list(requirement.specifier)
    pin = specifiers[0].version if len(specifiers) == 1 and specifiers[0].operator == "==" else ""
    try:
        version: Version | None = Version(pin)
    except InvalidVersion:
        version = None
    if version is None or version.local != "cpu":
        raise SystemExit(f"Expected a torch==<version>+cpu pin, got {requirement}")
    if version.is_prerelease or version.is_postrelease or version.epoch or len(version.release) != 3:
        raise SystemExit(f"Expected a final X.Y.Z torch release, which has a PyTorch tag, got {version.public}")
    return f"{version.public}+cpu.debug"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    print(debug_torch_version(Path("pyproject.toml").read_text()))


if __name__ == "__main__":
    main()
