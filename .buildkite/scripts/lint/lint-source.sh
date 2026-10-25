#!/usr/bin/env bash
# Lint the source files that differ from a base branch, or every tracked file
# if the base is empty.
#
# Usage:
#   lint-source.sh --build-type <Release|Debug>
#                  --base <branch>
set -euo pipefail

build_type=""
base_branch=""
base_given="false"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-type)
      build_type="$2"
      shift 2
      ;;
    --base)
      base_branch="$2"
      base_given="true"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${build_type}" ]]; then
  echo "--build-type is required" >&2
  exit 2
fi

if [[ "${base_given}" != "true" ]]; then
  echo "--base is required; pass --base \"\" to lint every tracked file" >&2
  exit 2
fi

# A change to any of these can flag files the change does not touch.
lint_configuration=(
  cmake/
  tools/linter/
  tools/linter_rbln/
  "*CMakeLists.txt"
  .clang-format
  .clang-tidy
  .cmakelintrc
  .lintrunner.toml
  .rumdl.toml
  .shellcheckrc
  .taplo.toml
  .yamllint
  pyproject.toml
  pyrefly.toml
  uv.lock
)

echo "--- Installing locked environment"
uv sync \
  --locked \
  --no-install-project

echo "--- Initializing linters"
uv run \
  --no-sync \
  lintrunner init

echo "--- Configuring CMake for clang-tidy"
uv run \
  --no-sync \
  cmake \
    -GNinja \
    -B build \
    -S . \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_BUILD_TYPE="${build_type}" \
    -DCMAKE_INSTALL_PREFIX=torch_rbln \
    -DBUILD_TESTING=ON

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
      echo "Could not compare the lint configuration with" \
        "origin/${base_branch}: git diff exited ${diff_status}" >&2
      exit 1
      ;;
  esac
fi

echo "--- Running lintrunner"
if [[ -z "${base_branch}" ]]; then
  lint_scope=(--paths-cmd="git ls-files")
else
  lint_scope=(--merge-base-with "origin/${base_branch}")
fi
uv run \
  --no-sync \
  lintrunner \
    "${lint_scope[@]}"
