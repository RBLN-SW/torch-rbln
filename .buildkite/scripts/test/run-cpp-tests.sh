#!/usr/bin/env bash
# Run the C++ test binaries against the installed wheel's shared libraries.
#
# Usage:
#   run-cpp-tests.sh --build-type <Release|Debug>
#                    --artifact-build-id <build ID>
#                    --artifact-dir <name>
#                    --archive <name>
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
if [[ "${build_type}" == "Debug" ]]; then
  libnuma_status="$(
    dpkg-query \
      --show \
      --showformat='${db:Status-Status}' \
      libnuma1 \
      || true
  )"
  if [[ "${libnuma_status}" != "installed" ]]; then
    echo "--- Installing libnuma1"
    apt-get update
    apt-get install \
      --no-install-recommends \
      --yes \
      libnuma1
  fi
fi

echo "--- Installing $(basename "${wheel}")"
uv pip install \
  --python .venv/bin/python \
  "${wheel}[runtime]"

echo "--- Downloading C++ test binaries ${archive}"
buildkite-agent artifact download \
  --build "${artifact_build_id}" \
  "${archive}" \
  .
mkdir \
  --parents \
  cpp-test-binaries
tar \
  --extract \
  --gzip \
  --file "${archive}" \
  --directory cpp-test-binaries

echo "--- Locating shared libraries"
site_packages="$(
  uv run \
    --no-sync \
    python \
      -c 'import site; print(site.getsitepackages()[0])'
)"
if ! rebel_finder_output="$(
  uv run \
    --no-sync \
    python \
      tools/find_rebel_runtime.py
)"; then
  echo "Could not locate librbln.so: ${rebel_finder_output}" >&2
  exit 1
fi
rebel_library_dir="$(
  sed \
    --quiet \
    's/^LIBRARY_DIR=//p' \
    <<<"${rebel_finder_output}"
)"
library_path="${PWD}/cpp-test-binaries/lib"
library_path+=":${site_packages}/torch_rbln/lib"
library_path+=":${site_packages}/torch/lib"
library_path+=":${rebel_library_dir}"
library_path+="${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
echo "LD_LIBRARY_PATH=${library_path}"

shopt \
  -s \
  nullglob
binaries=(cpp-test-binaries/bin/test_cpp_*)
shopt \
  -u \
  nullglob
if [[ "${#binaries[@]}" -eq 0 ]]; then
  echo "No test_cpp_* binaries in ${archive}" >&2
  exit 1
fi

failed_binaries=()
for binary in "${binaries[@]}"; do
  binary_name="$(
    basename \
      "${binary}"
  )"
  echo "--- Running ${binary_name}"
  LD_LIBRARY_PATH="${library_path}" \
  "${binary}" \
    || failed_binaries+=("${binary_name}")
done

if [[ "${#failed_binaries[@]}" -gt 0 ]]; then
  echo "--- Reporting failed binaries"
  for binary_name in "${failed_binaries[@]}"; do
    echo "Failed: ${binary_name}" >&2
  done
  exit 1
fi
