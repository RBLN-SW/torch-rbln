#!/usr/bin/env bash
# Run one Python test suite through `test/run_tests.py` against the wheel a build step uploaded.
#
# Usage:
#   run-python-tests.sh --suite <core|ops|models|distributed> --npu-count <n> --python-version <X.Y>
#                       --build-type <Release|Debug> --test-mode <ci|release>
#                       [--torch-index <uv index name>]
#
# A Debug wheel's tests run on debug `torch`. `--torch-index` names the index that serves the
# wheel's `torch` when it is not `pytorch-cpu`.
#
# Environment: the `UV_INDEX_*` index credentials.
set -euo pipefail

# shellcheck source=.buildkite/scripts/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"

suite=""
build_type=""
test_mode=""
python_version=""
npu_count=""
torch_index=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --suite)
      suite="$2"
      shift 2
      ;;
    --build-type)
      build_type="$2"
      shift 2
      ;;
    --test-mode)
      test_mode="$2"
      shift 2
      ;;
    --python-version)
      python_version="$2"
      shift 2
      ;;
    --npu-count)
      npu_count="$2"
      shift 2
      ;;
    --torch-index)
      torch_index="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit "${EXIT_USAGE}"
      ;;
  esac
done
if [[ -z "${suite}" ]]; then
  echo "--suite is required" >&2
  exit "${EXIT_USAGE}"
fi
if [[ -z "${build_type}" ]]; then
  echo "--build-type is required" >&2
  exit "${EXIT_USAGE}"
fi
if [[ -z "${test_mode}" ]]; then
  echo "--test-mode is required" >&2
  exit "${EXIT_USAGE}"
fi
if [[ -z "${python_version}" ]]; then
  echo "--python-version is required" >&2
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
if [[ -n "${torch_index}" ]]; then
  install_args+=(--torch-index "${torch_index}")
fi
"${SCRIPTS_DIR}/install-wheel.sh" "${install_args[@]}"

# `test/run_tests.py` runs `tools/test/install-test-deps.sh`, which installs with `uv` when `UV=1`.
export UV=1

echo "--- Recording environment"
git rev-parse HEAD
uv pip list --python .venv/bin/python
check_npu_environment "${npu_count}"

# The source tree would shadow the installed wheel on `sys.path`.
mv ./torch_rbln ./torch_rbln.bak

echo "--- Running ${suite} suite (${test_mode})"
uv run \
  --no-sync \
  python \
    test/run_tests.py \
    --suite "${suite}" \
    --test_mode "${test_mode}"
