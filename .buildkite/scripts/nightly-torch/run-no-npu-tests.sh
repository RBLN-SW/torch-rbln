#!/usr/bin/env bash
# Run test files that need no NPU on a host without one, against the wheel a build step uploaded.
# Dummy device mode needs the RBLN user-mode driver (UMD), which comes from a mounted directory.
#
# Usage:
#   run-no-npu-tests.sh --umd <directory> --python-version <X.Y> [--torch-index <uv index name>]
#                       <test file> ...
#
# Environment: the `UV_INDEX_*` index credentials.
set -euo pipefail

# shellcheck source=.buildkite/scripts/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"

umd=""
python_version=""
install_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --umd)
      umd="$2"
      shift 2
      ;;
    --python-version)
      python_version="$2"
      shift 2
      ;;
    --torch-index)
      install_args+=(--torch-index "$2")
      shift 2
      ;;
    --*)
      echo "Unknown argument: $1" >&2
      exit "${EXIT_USAGE}"
      ;;
    *) break ;;
  esac
done
if [[ -z "${umd}" ]]; then
  echo "--umd is required" >&2
  exit "${EXIT_USAGE}"
fi
if [[ -z "${python_version}" ]]; then
  echo "--python-version is required" >&2
  exit "${EXIT_USAGE}"
fi
if [[ $# -eq 0 ]]; then
  echo "No test files given" >&2
  exit "${EXIT_USAGE}"
fi
test_files=("$@")

"${SCRIPTS_DIR}/install-wheel.sh" \
  --python-version "${python_version}" \
  --build-type Release \
  "${install_args[@]}"

echo "--- Recording environment"
git rev-parse HEAD
uv pip list --python .venv/bin/python

echo "--- Checking no-NPU environment"
if [[ ! -e "${umd}/librbln-thunk.so" ]]; then
  echo "No librbln-thunk.so in ${umd}; check --umd and the volume that holds it" >&2
  exit "${EXIT_ENVIRONMENT}"
fi
export LD_LIBRARY_PATH="${umd}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
device_nodes="$(compgen -G '/dev/rbln[0-9]*' || true)"
if [[ -n "${device_nodes}" ]]; then
  echo "The pod has NPU device nodes: ${device_nodes//$'\n'/ }" >&2
  exit "${EXIT_ENVIRONMENT}"
fi
# A probe that fails here, unlike in `check_npu_environment`, is the product's failure and stays a
# failed test.
probe="$(mktemp)"
uv run --no-sync python "${SCRIPTS_DIR}/test/probe-rbln-runtime.py" no-npu "${probe}"
read -r loaded loaded_umd device_count available < "${probe}"
echo "No NPU device node; RBLN runtime loaded: ${loaded} (user-mode driver: ${loaded_umd});" \
  "device_count ${device_count}, is_available ${available}"
if [[ "${loaded}" != True ]]; then
  echo "The RBLN runtime did not load its user-mode driver" >&2
  exit "${EXIT_ENVIRONMENT}"
fi
if [[ "${device_count}" != 0 || "${available}" != False ]]; then
  echo "torch_rbln reports ${device_count} devices (is_available ${available})" \
    "on a pod with no device node" >&2
  exit "${EXIT_FAILURE}"
fi

# The source tree would shadow the installed wheel on `sys.path`.
mv ./torch_rbln ./torch_rbln.bak

echo "--- Running CI tests without NPU (UMD ${umd})"
# `test/run_tests.py` selects the same marker for the CI test mode.
uv run \
  --no-sync \
  python \
    -m pytest \
    "${test_files[@]}" \
    -m test_set_ci \
    --numprocesses=1
