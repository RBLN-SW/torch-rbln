#!/usr/bin/env python3
"""
Unified test runner for torch-rbln.

Defaults: runs all suites (core, distributed, models, ops) with CI marker (-m "test_set_ci")
using 16 parallel workers (override via --workers).

Usage:
    python test/run_tests.py                                          # All CI tests
    python test/run_tests.py --test_mode=release                      # All release tests
    python test/run_tests.py --suite=core --suite=ops                 # Only core and ops (CI)
    python test/run_tests.py --suite=distributed --test_mode=release  # Only distributed (release)
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Protocol


DEFAULT_NUM_WORKERS = 16

TEST_MODE_MARKERS: dict[str, str] = {
    "ci": "test_set_ci",
    "release": "not (test_set_experimental or test_set_perf)",
}


def _get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _run_pytest(
    test_dir: str,
    *,
    marker: str,
    workers: int,
) -> int:
    """Run pytest for a single test_dir/marker combo. Returns the exit code."""
    base_cmd = [
        sys.executable,
        "-m",
        "pytest",
        test_dir,
        "-m",
        marker,
    ]

    # Collect-only dry run to show which tests will execute
    collect_cmd = [*base_cmd, "--collect-only", "-q"]
    print(f"\n{'=' * 120}")
    print(f"Collecting: pytest {test_dir}  -m '{marker}'")
    print("-" * 120)
    subprocess.run(collect_cmd, check=False)

    # Run the actual tests
    run_cmd = [*base_cmd, f"--numprocesses={workers}"]
    print(f"\n{'-' * 120}")
    print(f"Running: pytest {test_dir}  -m '{marker}'  --numprocesses={workers}")
    print("-" * 120)
    return_code = subprocess.run(run_cmd, check=False).returncode
    print("-" * 120)
    return return_code


class _TestResults:
    """Accumulates pass/fail/no-collect counts across runs."""

    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0
        self.no_collected = 0

    def record(self, exit_code: int, description: str) -> None:
        if exit_code == 5:  # pytest: no tests collected
            print(f"  -> No tests collected: {description}")
            self.no_collected += 1
        elif exit_code != 0:
            print(f"  -> FAILED (exit {exit_code}): {description}")
            self.failed += 1
        else:
            print(f"  -> Passed: {description}")
            self.passed += 1

    def print_summary(self) -> None:
        print(f"\n{'=' * 80}")
        print("Test Summary")
        print(f"  Passed:       {self.passed}")
        print(f"  Failed:       {self.failed}")
        print(f"  No collected: {self.no_collected}")
        print("=" * 80)

    @property
    def ok(self) -> bool:
        return self.failed == 0


def _run_with_worker_split(
    test_dirs: list[str],
    *,
    test_mode: str,
    workers: int,
    results: _TestResults,
    project_root: Path,
) -> None:
    """Run single_worker (serial) then multi-worker (parallel) for each dir."""
    mode_marker = TEST_MODE_MARKERS[test_mode]
    for test_dir in test_dirs:
        abs_test_dir = str(project_root / test_dir)
        for is_single_worker in (True, False):
            worker_marker = "single_worker" if is_single_worker else "not single_worker"
            marker = f"{mode_marker} and {worker_marker}"
            num_processes = 1 if is_single_worker else workers
            desc = f"{test_dir} [{marker}]"
            rc = _run_pytest(abs_test_dir, marker=marker, workers=num_processes)
            results.record(rc, desc)


def _install_model_deps(project_root: Path) -> None:
    """Install model-test dependencies via install-test-deps.sh."""
    script = project_root / "tools" / "test" / "install-test-deps.sh"
    if not script.exists():
        raise RuntimeError(f"install-test-deps.sh not found: {script}")
    print(f"\nInstalling model-test dependencies via {script.name} ...")
    rc = subprocess.run(["bash", str(script)], check=False).returncode
    if rc != 0:
        raise RuntimeError(f"{script.name} failed with exit code {rc}")


# ---------------------------------------------------------------------------
# Per-suite runners
# ---------------------------------------------------------------------------


def run_core_tests(
    test_mode: str,
    workers: int,
    results: _TestResults,
    project_root: Path,
) -> None:
    _run_with_worker_split(
        ["test/internal/", "test/rbln/"],
        test_mode=test_mode,
        workers=workers,
        results=results,
        project_root=project_root,
    )


def run_distributed_tests(
    test_mode: str,
    workers: int,
    results: _TestResults,
    project_root: Path,
) -> None:
    _run_with_worker_split(
        ["test/distributed/"],
        test_mode=test_mode,
        workers=workers,
        results=results,
        project_root=project_root,
    )


def run_models_tests(
    test_mode: str,
    workers: int,
    results: _TestResults,
    project_root: Path,
) -> None:
    _install_model_deps(project_root)
    _run_with_worker_split(
        ["test/models/"],
        test_mode=test_mode,
        workers=workers,
        results=results,
        project_root=project_root,
    )


def run_ops_tests(
    test_mode: str,
    workers: int,
    results: _TestResults,
    project_root: Path,
) -> None:
    _run_with_worker_split(
        ["test/ops/"],
        test_mode=test_mode,
        workers=workers,
        results=results,
        project_root=project_root,
    )


class SuiteRunner(Protocol):
    def __call__(self, test_mode: str, workers: int, results: _TestResults, project_root: Path) -> None: ...


SUITE_RUNNERS: dict[str, SuiteRunner] = {
    "core": run_core_tests,
    "distributed": run_distributed_tests,
    "models": run_models_tests,
    "ops": run_ops_tests,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified torch-rbln test runner")
    parser.add_argument(
        "--suite",
        action="append",
        choices=list(SUITE_RUNNERS),
        default=None,
        dest="suites",
        help="Test suite to run (repeatable, default: all). Choices: %(choices)s",
    )
    parser.add_argument(
        "--test_mode",
        choices=list(TEST_MODE_MARKERS),
        default="ci",
        help="Test mode to run (default: ci). Choices: %(choices)s",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help=f"Number of parallel pytest workers (default: {DEFAULT_NUM_WORKERS})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = _get_project_root()

    suites = args.suites or list(SUITE_RUNNERS)
    test_mode = args.test_mode
    results = _TestResults()

    print(f"Suites:  {', '.join(suites)}")
    print(f"Mode:    {test_mode} (-m '{TEST_MODE_MARKERS[test_mode]}')")
    print(f"Workers: {args.workers}")

    for suite in suites:
        SUITE_RUNNERS[suite](test_mode, args.workers, results, project_root)

    results.print_summary()
    sys.exit(0 if results.ok else 1)


if __name__ == "__main__":
    main()
