#!/usr/bin/env bash
# Build the C++ test binaries against the installed wheel.
#
# Usage:
#   build-cpp-test-binaries.sh --build-type <Release|Debug>
#                              --artifact-build-id <build ID>
#                              --artifact-dir <name>
#                              --archive <name>
set -euo pipefail

build_type=""
artifact_build_id=""
artifact_dir=""
archive=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-type)
      build_type="$2"
      shift 2
      ;;
    --artifact-build-id)
      artifact_build_id="$2"
      shift 2
      ;;
    --artifact-dir)
      artifact_dir="$2"
      shift 2
      ;;
    --archive)
      archive="$2"
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

if [[ -z "${artifact_build_id}" ]]; then
  echo "--artifact-build-id is required" >&2
  exit 2
fi

if [[ -z "${artifact_dir}" ]]; then
  echo "--artifact-dir is required" >&2
  exit 2
fi

if [[ -z "${archive}" ]]; then
  echo "--archive is required" >&2
  exit 2
fi

: "${CMAKE_BUILD_PARALLEL_LEVEL:?is not set}"

echo "--- Downloading ${artifact_dir}"
buildkite-agent artifact download \
  --build "${artifact_build_id}" \
  "${artifact_dir}/*" \
  .
wheels=("${artifact_dir}"/*.whl)
wheel="${wheels[0]}"

echo "--- Installing locked environment"
# The downloaded lock, not the checkout's, pins the `torch` the wheel needs.
cp \
  "${artifact_dir}/pyproject.toml" \
  "${artifact_dir}/uv.lock" \
  .
uv sync \
  --locked \
  --no-install-project

echo "--- Installing $(basename "${wheel}")"
uv pip install \
  --python .venv/bin/python \
  "${wheel}[runtime]"

echo "--- Building C++ test binaries"
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
shopt \
  -s \
  nullglob
test_binaries=(build-test/bin/test_cpp_*)
shopt \
  -u \
  nullglob
if [[ "${#test_binaries[@]}" -eq 0 ]]; then
  echo "cmake --build produced no test_cpp_* binaries under build-test/bin/" >&2
  exit 1
fi

echo "--- Packing binaries"
mkdir \
  --parents \
  cpp-test-binaries/bin \
  cpp-test-binaries/lib
cp \
  "${test_binaries[@]}" \
  cpp-test-binaries/bin/
cp \
  --no-dereference \
  build-test/lib/*.so* \
  cpp-test-binaries/lib/
tar \
  --create \
  --verbose \
  --gzip \
  --file "${archive}" \
  --directory cpp-test-binaries \
  .

echo "--- Uploading ${archive}"
buildkite-agent artifact upload \
  "${archive}"
