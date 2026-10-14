#!/usr/bin/env bash
# Run `lintrunner` with the image's Python, configuring CMake for one build type. A pull request
# build lints the files changed since the merge base with its base branch, or every tracked file
# when it changes a linter or its configuration; any other build lints every tracked file.
#
# Usage:
#   lint-source.sh --build-type <Release|Debug>
#
# Environment: the `UV_INDEX_*` index credentials, and `BUILDKITE_PULL_REQUEST` and
# `BUILDKITE_PULL_REQUEST_BASE_BRANCH`, which pick the files to lint.
set -euo pipefail

# shellcheck source=.buildkite/scripts/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"

build_type=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-type)
      build_type="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit "${EXIT_USAGE}"
      ;;
  esac
done
if [[ -z "${build_type}" ]]; then
  echo "--build-type is required" >&2
  exit "${EXIT_USAGE}"
fi
# A change to any of these can flag files the change does not touch.
lint_configuration=(
  .clang-format
  .clang-tidy
  .cmakelintrc
  .flake8
  .lintrunner.toml
  .shellcheckrc
  mypy-strict.ini
  mypy.ini
  pyproject.toml
  tools/linter
  uv.lock
)
base_branch=""
if [[ "${BUILDKITE_PULL_REQUEST:-false}" != "false" ]]; then
  base_branch="${BUILDKITE_PULL_REQUEST_BASE_BRANCH:-main}"
fi

echo "--- Installing locked environment"
# Each lint job takes a private `uv` cache, because concurrent jobs that share one serialize on
# `uv`'s per-entry build lock past its timeout.
export UV_CACHE_DIR
UV_CACHE_DIR="$(mktemp --directory)"
# `clang-tidy` needs the RBLN runtime headers, which ship only in the build-pinned `rebel-compiler`.
uv sync \
  --locked \
  --no-install-project

echo "--- Initializing linters"
uv run --no-sync lintrunner init

echo "--- Configuring CMake for clang-tidy (${build_type})"
uv run \
  --no-sync \
  cmake \
    -GNinja \
    -B build \
    -S . \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_BUILD_TYPE="${build_type}" \
    -DCMAKE_INSTALL_PREFIX=torch_rbln \
    -DBUILD_TESTING=OFF

if [[ -n "${base_branch}" ]]; then
  echo "--- Fetching base branch ${base_branch}"
  git fetch \
    --no-tags \
    origin \
    "${base_branch}"
  diff_status=0
  git diff \
    --quiet \
    "origin/${base_branch}...HEAD" \
    -- \
    "${lint_configuration[@]}" \
    || diff_status=$?
  case "${diff_status}" in
    0) ;;
    1)
      echo "This change touches the lint configuration; linting every file"
      base_branch=""
      ;;
    *)
      echo "git diff exited ${diff_status}" >&2
      exit "${EXIT_ENVIRONMENT}"
      ;;
  esac
fi

echo "--- Running lintrunner"
# The `mypy` adapter runs plain `mypy` when `GITHUB_ACTIONS` is set; otherwise the `dmypy` daemons
# of `MYPY` and `MYPYSTRICT` race on one status file in a fresh pod.
export GITHUB_ACTIONS=true
if [[ -z "${base_branch}" ]]; then
  uv run \
    --no-sync \
    lintrunner \
      --paths-cmd='git ls-files'
else
  uv run \
    --no-sync \
    lintrunner \
      --merge-base-with "origin/${base_branch}"
fi
