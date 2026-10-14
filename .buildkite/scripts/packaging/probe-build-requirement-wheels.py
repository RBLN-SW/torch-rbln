# /// script
# requires-python = ">=3.10,<3.15"
# dependencies = [
#   "packaging==26.3",
#   "tomli==2.4.1; python_version < '3.11'",
# ]
# ///
"""Exit 4 when torch or rebel-compiler lacks a wheel for a platform.

The platform is a Python version and a CPU architecture. uv pip compile
resolves the build requirements and build constraints of pyproject.toml
against the indexes pyproject.toml gives each package. Any other failure
exits 1, including a pinned version that no index has and credentials
that an index rejects.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name


if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

PACKAGES = ("torch", "rebel-compiler")
# The build image tag pins UBI 9, whose glibc is 2.34.
MANYLINUX = "manylinux_2_34"
CREDENTIAL_HINTS = ("lack of valid authentication credentials", "403 Forbidden")
# uv reports a version that no index has in these words, and a version
# without a wheel for the platform in others.
ABSENT_VERSION_HINTS = ("there is no version of", "was not found in the package registry")
EXIT_MISSING_WHEEL = 4


def requirement_name(requirement: str) -> str:
    try:
        return canonicalize_name(Requirement(requirement).name)
    except InvalidRequirement as error:
        raise SystemExit(f"Invalid requirement {requirement!r}: {error}") from None


def requirement_lines(pyproject: dict[str, Any], package: str) -> list[str]:
    """Return the package's build requirements, then its constraints."""
    uv = pyproject.get("tool", {}).get("uv", {})
    required = [
        line for line in pyproject.get("build-system", {}).get("requires", []) if requirement_name(line) == package
    ]
    if not required:
        raise SystemExit(f"Expected {package} in [build-system].requires")
    constraints = [line for line in uv.get("build-constraint-dependencies", []) if requirement_name(line) == package]
    return required + constraints


def index_config(pyproject: dict[str, Any], package: str) -> tuple[list[dict[str, Any]], str]:
    """Return the indexes and strategy the package resolves with."""
    uv = pyproject.get("tool", {}).get("uv", {})
    indexes: list[dict[str, Any]] = uv.get("index", [])
    if any(index.get("default") or not index.get("name") for index in indexes):
        raise SystemExit("Expected only named, non-default indexes in [tool.uv].index")
    sources: dict[str, Any] = {canonicalize_name(name): source for name, source in uv.get("sources", {}).items()}
    if package not in sources:
        return [index for index in indexes if not index.get("explicit")], uv.get("index-strategy", "first-index")
    source = sources[package]
    if not isinstance(source, dict) or set(source) != {"index"}:
        raise SystemExit(f"Expected the [tool.uv.sources] entry of {package} to name only an index")
    chosen = [index for index in indexes if index["name"] == source["index"]]
    if not chosen:
        raise SystemExit(f"Expected an index named {source['index']!r} in [tool.uv].index")
    return chosen, "first-index"


def compile_requirements(
    requirements: list[str], indexes: list[dict[str, Any]], strategy: str, python_version: str, cpu_architecture: str
) -> subprocess.CompletedProcess[str]:
    command = ["uv", "pip", "compile", "--no-config", "--quiet", "--no-deps", "--only-binary", ":all:"]
    command += ["--python-version", python_version, "--python-platform", f"{cpu_architecture}-{MANYLINUX}"]
    for index in indexes:
        command += ["--index", f"{index['name']}={index['url']}"]
    command += ["--index-strategy", strategy, "-"]
    return subprocess.run(command, input="\n".join(requirements) + "\n", capture_output=True, text=True)


def unsolvable(result: subprocess.CompletedProcess[str]) -> bool:
    return result.returncode == 1 and "No solution found" in result.stderr


def has_wheel(package: str, pyproject: dict[str, Any], python_version: str, cpu_architecture: str) -> bool:
    requirements = requirement_lines(pyproject, package)
    indexes, strategy = index_config(pyproject, package)
    result = compile_requirements(requirements, indexes, strategy, python_version, cpu_architecture)
    if result.returncode == 0:
        return True
    where = f"{package} for Python {python_version} on {cpu_architecture}"
    if not unsolvable(result):
        print(result.stderr.rstrip("\n"), file=sys.stderr)
        raise SystemExit(f"Could not resolve {where} (uv exited {result.returncode})")
    # Under unsafe-first-match, uv silently skips an index that rejects
    # the credentials; asking each index alone exposes that, so a
    # credential problem is not reported as a missing wheel.
    for index in indexes:
        alone = compile_requirements(requirements, [index], "first-index", python_version, cpu_architecture)
        if any(hint in alone.stderr for hint in CREDENTIAL_HINTS):
            print(alone.stderr.rstrip("\n"), file=sys.stderr)
            raise SystemExit(f"Could not resolve {where}: index {index['name']!r} rejected the credentials")
        if alone.returncode != 0 and not unsolvable(alone):
            print(alone.stderr.rstrip("\n"), file=sys.stderr)
            raise SystemExit(
                f"Could not resolve {where}: index {index['name']!r} failed (uv exited {alone.returncode})"
            )
    print(result.stderr.rstrip("\n"), file=sys.stderr)
    if any(hint in result.stderr for hint in ABSENT_VERSION_HINTS):
        raise SystemExit(f"Could not resolve {where}: no index has the required version")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--python-version", required=True)
    parser.add_argument("--cpu-architecture", required=True, choices=["x86_64", "aarch64"])
    args = parser.parse_args()
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    missing = [
        package for package in PACKAGES if not has_wheel(package, pyproject, args.python_version, args.cpu_architecture)
    ]
    if missing:
        print(f"No {' or '.join(missing)} wheel for Python {args.python_version} on {args.cpu_architecture}")
        return EXIT_MISSING_WHEEL
    print(f"{' and '.join(PACKAGES)} have wheels for Python {args.python_version} on {args.cpu_architecture}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
