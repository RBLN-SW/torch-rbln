#!/usr/bin/env python3
"""Upload a file from .buildkite/pipelines/ into the running build.

The agent interpolates the file once for each combination of the given
Python versions and build types, with the TORCH_RBLN_* values of that
combination, and the script uploads the combinations together as one
group. It uploads once per job because the CI server runs one job's
uploads one after another rather than together.

--skip-unbuildable includes a wheel build only for the Python versions
whose torch and rebel-compiler have wheels for its CPU architecture, and
annotates the build with the others. --for-each-built-wheel interpolates
the file once per Release wheel that earlier wheel-build uploads in this
build recorded.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, NoReturn


SCRIPTS_DIR = Path(__file__).resolve().parent
PIPELINES_DIR = SCRIPTS_DIR.parent / "pipelines"
BUILD_TYPES = ("Release", "Debug")
NPU_PRODUCTS = ("ca25", "cr13")
TEST_SUITES = ("core", "ops", "models", "distributed")
WHEEL_BUILDS = {"x86_64-build": "x86_64", "aarch64-build": "aarch64"}
EXIT_ENVIRONMENT = 3
# probe-build-requirement-wheels.py exits with this when a build
# requirement has no wheel for the platform.
EXIT_MISSING_WHEEL = 4


def fail(message: str) -> NoReturn:
    print(message, file=sys.stderr)
    raise SystemExit(EXIT_ENVIRONMENT)


def comma_separated(text: str) -> list[str]:
    return [item for item in text.split(",") if item]


def run_agent(*args: str) -> str:
    result = subprocess.run(["buildkite-agent", *args], capture_output=True, text=True)
    if result.returncode:
        sys.stderr.write(result.stderr)
        fail(f"buildkite-agent {' '.join(args[:2])} exited {result.returncode}")
    return result.stdout


def built_wheels_key(cpu_architecture: str, build_type: str) -> str:
    return f"built-wheels-{cpu_architecture}-{build_type}"


def read_built_wheels(cpu_architecture: str, build_type: str) -> list[str]:
    """Return the Python versions of the recorded wheels."""
    key = built_wheels_key(cpu_architecture, build_type)
    exists = subprocess.run(["buildkite-agent", "meta-data", "exists", key])
    # buildkite-agent exits 100 when the key is not set.
    if exists.returncode == 100:
        return []
    if exists.returncode:
        fail(f"buildkite-agent meta-data exists {key} exited {exists.returncode}")
    return run_agent("meta-data", "get", key).split()


def is_buildable(python_version: str, cpu_architecture: str) -> bool:
    probe = subprocess.run(
        [
            "uv",
            "run",
            "--no-config",
            str(SCRIPTS_DIR / "packaging/probe-build-requirement-wheels.py"),
            "--python-version",
            python_version,
            "--cpu-architecture",
            cpu_architecture,
        ]
    )
    if probe.returncode not in (0, EXIT_MISSING_WHEEL):
        fail(f"Could not check the build requirements' wheels (exit {probe.returncode})")
    return probe.returncode == 0


def interpolate(pipeline: Path, values: dict[str, str]) -> dict[str, Any]:
    """Return the file's group as the agent interpolates it with the values."""
    print(f"Interpolating {pipeline.stem}: " + ", ".join(f"{k}={v}" for k, v in values.items()), flush=True)
    result = subprocess.run(
        ["buildkite-agent", "pipeline", "upload", "--dry-run", "--format", "json", str(pipeline)],
        env={**os.environ, **values},
        stdout=subprocess.PIPE,
        text=True,
    )
    if result.returncode:
        raise SystemExit(result.returncode)
    document: dict[str, Any] = json.loads(result.stdout)
    steps: list[dict[str, Any]] = document["steps"]
    (group,) = steps
    # An upload has one top-level env, so each step carries the env of
    # its own combination.
    for step in group["steps"]:
        step["env"] = {**document.get("env", {}), **step.get("env", {})}
    return group


def upload(groups: list[dict[str, Any]]) -> None:
    group = {**groups[0], "steps": [step for interpolated in groups for step in interpolated["steps"]]}
    # The agent interpolated every group already.
    result = subprocess.run(
        ["buildkite-agent", "pipeline", "upload", "--no-interpolation"],
        input=json.dumps({"steps": [group]}),
        text=True,
    )
    if result.returncode:
        raise SystemExit(result.returncode)


def upload_per_built_wheel(pipeline: Path, shared_values: dict[str, str]) -> None:
    groups = [
        interpolate(
            pipeline,
            {
                **shared_values,
                "TORCH_RBLN_PYTHON_VERSION": python_version,
                "TORCH_RBLN_CPU_ARCHITECTURE": cpu_architecture,
            },
        )
        for cpu_architecture in WHEEL_BUILDS.values()
        for python_version in read_built_wheels(cpu_architecture, "Release")
    ]
    if not groups:
        fail("No Release wheel was recorded; every wheel build was skipped (see the annotations)")
    upload(groups)


def upload_per_combination(
    pipeline: Path,
    shared_values: dict[str, str],
    python_versions: list[str],
    build_types: list[str],
    skip_unbuildable: bool,
) -> None:
    cpu_architecture = WHEEL_BUILDS.get(pipeline.stem, "")
    groups = []
    built: dict[str, list[str]] = {}
    for python_version, build_type in itertools.product(python_versions or [""], build_types or [""]):
        values = dict(shared_values)
        if python_version:
            values["TORCH_RBLN_PYTHON_VERSION"] = python_version
        if build_type:
            values["TORCH_RBLN_BUILD_TYPE"] = build_type
        if skip_unbuildable and not is_buildable(python_version, cpu_architecture):
            message = (
                f"Skipped the Python {python_version} {build_type} wheel on {cpu_architecture}: "
                "torch or rebel-compiler has no wheel for it"
            )
            print(message, flush=True)
            run_agent(
                "annotate",
                "--style",
                "warning",
                "--context",
                f"skipped-wheels-{cpu_architecture}",
                "--append",
                message + "\n",
            )
            continue
        groups.append(interpolate(pipeline, values))
        if cpu_architecture:
            built.setdefault(build_type, []).append(python_version)
    if groups:
        upload(groups)
    for build_type, python_versions_built in built.items():
        run_agent("meta-data", "set", built_wheels_key(cpu_architecture, build_type), " ".join(python_versions_built))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pipeline", help="a file under .buildkite/pipelines/, without .yml")
    parser.add_argument("--python-versions", type=comma_separated, default=[])
    parser.add_argument("--build-types", type=comma_separated, default=[], help=", ".join(BUILD_TYPES))
    parser.add_argument("--npu-products", type=comma_separated, default=[], help=", ".join(NPU_PRODUCTS))
    parser.add_argument("--test-suites", type=comma_separated, default=[], help=", ".join(TEST_SUITES))
    parser.add_argument("--test-mode", choices=["ci", "release"])
    parser.add_argument("--version-kind", choices=["dev", "release"])
    parser.add_argument("--skip-unbuildable", action="store_true")
    parser.add_argument("--for-each-built-wheel", action="store_true")
    args = parser.parse_args()
    pipeline = PIPELINES_DIR / f"{args.pipeline}.yml"
    if not pipeline.is_file():
        parser.error(f"no pipeline file {pipeline}")
    for name, given, allowed in (
        ("build type", args.build_types, BUILD_TYPES),
        ("NPU product", args.npu_products, NPU_PRODUCTS),
        ("test suite", args.test_suites, TEST_SUITES),
    ):
        unknown = sorted(set(given) - set(allowed))
        if unknown:
            parser.error(f"unknown {name}: {', '.join(unknown)}")
    if args.skip_unbuildable and args.pipeline not in WHEEL_BUILDS:
        parser.error(f"--skip-unbuildable takes {' or '.join(WHEEL_BUILDS)}")

    shared_values = {}
    if args.npu_products:
        shared_values["TORCH_RBLN_NPU_PRODUCTS"] = ",".join(args.npu_products)
    if args.test_suites:
        shared_values["TORCH_RBLN_TEST_SUITES"] = ",".join(args.test_suites)
    if args.test_mode:
        shared_values["TORCH_RBLN_TEST_MODE"] = args.test_mode
    if args.version_kind:
        shared_values["TORCH_RBLN_VERSION_KIND"] = args.version_kind
    if args.for_each_built_wheel:
        upload_per_built_wheel(pipeline, shared_values)
    else:
        upload_per_combination(pipeline, shared_values, args.python_versions, args.build_types, args.skip_unbuildable)


if __name__ == "__main__":
    main()
