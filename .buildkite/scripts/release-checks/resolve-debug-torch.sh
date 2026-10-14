#!/usr/bin/env bash
# Resolve the debug `torch` wheel that the Debug suites install, and record its version, its file
# name, and whether the internal index lacks it in the meta-data keys
# `debug-torch-{version,wheel,needs-build}-<Python tag>`, such as `debug-torch-wheel-cp312`.
#
# Usage:
#   resolve-debug-torch.sh --python-version <X.Y>
#
# Environment: `BUILDKITE_BRANCH` and the `rbln-internal` index credentials.
set -euo pipefail

# shellcheck source=.buildkite/scripts/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"

python_version=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --python-version)
      python_version="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit "${EXIT_USAGE}"
      ;;
  esac
done
if [[ -z "${python_version}" ]]; then
  echo "--python-version is required" >&2
  exit "${EXIT_USAGE}"
fi
python_tag="cp${python_version//./}"

echo "--- Resolving debug torch version"
debug_torch_version="$(
  uv run \
    --no-config \
    "${SCRIPTS_DIR}/resolve-debug-torch-version.py"
)"
wheel_file="torch-${debug_torch_version}-${python_tag}-${python_tag}-linux_x86_64.whl"
echo "Debug torch for Python ${python_version}: ${wheel_file}"

echo "--- Checking internal package index"
already_published="$(index_has_wheel rbln-internal torch "${wheel_file}")"
# Every later run installs the first upload of a version, so only a build of `main`, whose changes
# were reviewed, may upload it.
if [[ "${already_published}" != true && "${BUILDKITE_BRANCH:-}" != main ]]; then
  echo "${wheel_file} is not on the internal package index, and only a build of main" \
    "builds and publishes it; run release checks on main once the torch pin has merged" >&2
  exit "${EXIT_ENVIRONMENT}"
fi

echo "--- Recording version for build and publish steps"
needs_build=true
if [[ "${already_published}" == true ]]; then
  echo "${wheel_file} is already on the internal package index; nothing to build"
  needs_build=false
fi
buildkite-agent meta-data set "debug-torch-version-${python_tag}" "${debug_torch_version}"
buildkite-agent meta-data set "debug-torch-wheel-${python_tag}" "${wheel_file}"
buildkite-agent meta-data set "debug-torch-needs-build-${python_tag}" "${needs_build}"
