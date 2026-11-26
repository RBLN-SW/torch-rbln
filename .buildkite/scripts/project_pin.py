#!/usr/bin/env -S uv run --isolated --only-group ci --locked python
"""Read how a uv project pins one of its dependencies.

Commands:
  locked             print the version uv.lock resolves the package to
  allows VERSION     print true when the requirement in pyproject.toml admits
                     the version, pre-releases included
  patch-line VERSION print the specifier that admits the patch line of the
                     version, from its first development release on
"""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version


def locked_version(project: Path, package: str) -> str:
    lock = tomllib.loads((project / "uv.lock").read_text(encoding="utf-8"))
    versions = [entry["version"] for entry in lock.get("package", []) if canonicalize_name(entry["name"]) == package]
    if len(versions) != 1:
        msg = f"uv.lock resolves {package} to {len(versions)} versions, not one: {versions}"
        raise SystemExit(msg)
    return versions[0]


def requirement(project: Path, package: str) -> Requirement:
    pyproject = tomllib.loads((project / "pyproject.toml").read_text(encoding="utf-8"))
    try:
        requirements = [Requirement(text) for text in pyproject.get("project", {}).get("dependencies", [])]
    except InvalidRequirement as error:
        raise SystemExit(str(error)) from None
    matches = [item for item in requirements if canonicalize_name(item.name) == package]
    if len(matches) != 1:
        msg = f"pyproject.toml requires {package} {len(matches)} times, not once"
        raise SystemExit(msg)
    return matches[0]


def parse_version(text: str) -> Version:
    try:
        return Version(text)
    except InvalidVersion as error:
        raise SystemExit(str(error)) from None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", type=Path, required=True, help="the directory with pyproject.toml and uv.lock")
    parser.add_argument("--package", required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("locked")
    commands.add_parser("allows").add_argument("version")
    commands.add_parser("patch-line").add_argument("version")
    args = parser.parse_args()
    package = canonicalize_name(args.package)
    if args.command == "locked":
        print(locked_version(args.project, package))
    elif args.command == "allows":
        version = parse_version(args.version)
        admitted = requirement(args.project, package).specifier.contains(version, prereleases=True)
        print("true" if admitted else "false")
    else:
        major, minor, patch = (*parse_version(args.version).release, 0, 0)[:3]
        print(f">={major}.{minor}.{patch}.dev0,<{major}.{minor}.{patch + 1}")


if __name__ == "__main__":
    main()
