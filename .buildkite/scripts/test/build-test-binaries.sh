#!/usr/bin/env bash
# Build the C++ test binaries against the wheel a build step uploaded, and upload them as
# `cpp-test-binaries-py<X.Y>-x86_64-<build type>.tar.gz`.
#
# Usage:
#   build-test-binaries.sh --python-version <X.Y> --build-type <Release|Debug>
#
# Environment: `CMAKE_BUILD_PARALLEL_LEVEL` (the build's parallelism) and the `UV_INDEX_*` index
# credentials.
set -euo pipefail

# shellcheck source=.buildkite/scripts/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"

python_version=""
build_type=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --python-version)
      python_version="$2"
      shift 2
      ;;
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
if [[ -z "${python_version}" ]]; then
  echo "--python-version is required" >&2
  exit "${EXIT_USAGE}"
fi
if [[ -z "${build_type}" ]]; then
  echo "--build-type is required" >&2
  exit "${EXIT_USAGE}"
fi
if [[ -z "${CMAKE_BUILD_PARALLEL_LEVEL:-}" ]]; then
  echo "CMAKE_BUILD_PARALLEL_LEVEL is not set; check the step env" >&2
  exit "${EXIT_ENVIRONMENT}"
fi

# The binaries link the release `torch` that the wheel was built against, even for a Debug wheel.
"${SCRIPTS_DIR}/install-wheel.sh" \
  --python-version "${python_version}" \
  --build-type "${build_type}"

echo "--- Building C++ test binaries (${build_type})"
uv run \
  --no-sync \
  cmake \
    -S test/cpp \
    -B build-test \
    -G Ninja \
    -DCMAKE_BUILD_TYPE="${build_type}" \
    -DPython3_EXECUTABLE="${PWD}/.venv/bin/python"
uv run \
  --no-sync \
  cmake \
    --build build-test
shopt -s nullglob
test_binaries=(build-test/bin/test_cpp_*)
shopt -u nullglob
if [[ "${#test_binaries[@]}" -eq 0 ]]; then
  echo "The build produced no test_cpp_* binaries under build-test/bin/" >&2
  exit "${EXIT_FAILURE}"
fi
echo "Built ${#test_binaries[@]} C++ test binaries"

echo "--- Packing binaries"
mkdir \
  --parents \
  cpp-test-binaries/bin \
  cpp-test-binaries/lib
cp "${test_binaries[@]}" cpp-test-binaries/bin/
cp \
  --no-dereference \
  build-test/lib/*.so* \
  cpp-test-binaries/lib/
archive="cpp-test-binaries-py${python_version}-x86_64-${build_type}.tar.gz"
tar \
  --create \
  --gzip \
  --file "${archive}" \
  --directory cpp-test-binaries \
  .
du --human-readable "${archive}"

echo "--- Uploading ${archive}"
buildkite-agent artifact upload "${archive}"
