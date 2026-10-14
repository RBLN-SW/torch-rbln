#!/usr/bin/env bash
# Run the C++ test binaries that `build-test-binaries.sh` packed, against the shared libraries of
# the installed wheel.
#
# Usage:
#   run-cpp-tests.sh --python-version <X.Y> --build-type <Release|Debug> --npu-count <n>
#
# A Debug wheel's binaries run on debug `torch`.
#
# Environment: the `UV_INDEX_*` index credentials.
set -euo pipefail

# shellcheck source=.buildkite/scripts/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"

python_version=""
build_type=""
npu_count=""
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
    --npu-count)
      npu_count="$2"
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
if [[ -z "${npu_count}" ]]; then
  echo "--npu-count is required" >&2
  exit "${EXIT_USAGE}"
fi

install_args=(--python-version "${python_version}" --build-type "${build_type}")
if [[ "${build_type}" == Debug ]]; then
  install_args+=(--debug-torch)
fi
"${SCRIPTS_DIR}/install-wheel.sh" "${install_args[@]}"

archive="cpp-test-binaries-py${python_version}-x86_64-${build_type}.tar.gz"
echo "--- Downloading C++ test binaries ${archive}"
if ! buildkite-agent artifact download "${archive}" .; then
  echo "Could not download ${archive} from this build's artifacts" >&2
  exit "${EXIT_ENVIRONMENT}"
fi
mkdir --parents cpp-test-binaries
tar \
  --extract \
  --gzip \
  --file "${archive}" \
  --directory cpp-test-binaries

echo "--- Recording environment"
git rev-parse HEAD
uv pip list --python .venv/bin/python
check_npu_environment "${npu_count}"

echo "--- Locating shared libraries"
site_packages="$(
  uv run \
    --no-sync \
    python \
      -c 'import site; print(site.getsitepackages()[0])'
)"
if ! rebel_finder_output="$(uv run --no-sync python tools/find_rebel_runtime.py)"; then
  echo "Could not locate librbln.so: ${rebel_finder_output}" >&2
  exit "${EXIT_ENVIRONMENT}"
fi
library_path="${PWD}/cpp-test-binaries/lib"
library_path+=":${site_packages}/torch_rbln/lib"
library_path+=":${site_packages}/torch/lib"
library_path+=":$(sed --quiet 's/^LIBRARY_DIR=//p' <<<"${rebel_finder_output}")"
library_path+="${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export LD_LIBRARY_PATH="${library_path}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"

shopt -s nullglob
binaries=(cpp-test-binaries/bin/test_cpp_*)
shopt -u nullglob
if [[ "${#binaries[@]}" -eq 0 ]]; then
  echo "No test_cpp_* binaries in ${archive}" >&2
  exit "${EXIT_ENVIRONMENT}"
fi

exit_code=0
for binary in "${binaries[@]}"; do
  binary_name="$(basename "${binary}")"
  echo "--- Running ${binary_name}"
  status=0
  timeout \
    --kill-after=30s \
    10m \
    "${binary}" \
    || status=$?
  if [[ "${status}" -ne 0 ]]; then
    exit_code="${EXIT_FAILURE}"
    case "${status}" in
      124)
        echo "${binary_name} timed out after 10m" >&2
        ;;
      137)
        echo "${binary_name} was killed by SIGKILL: timeout" \
          "escalation or out of memory" >&2
        ;;
      127)
        echo "${binary_name} could not start: a shared library" \
          "is missing or version-incompatible" >&2
        ;;
      1)
        echo "${binary_name} reported test failures" >&2
        ;;
      *)
        echo "${binary_name} exited ${status};" \
          "RUN_ALL_TESTS returns 0 or 1, so this is a crash or a" \
          "harness fault" >&2
        ;;
    esac
  fi
done
exit "${exit_code}"
