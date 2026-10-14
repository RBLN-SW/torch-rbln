#!/usr/bin/env bash
# Build the debug `torch` wheel that `resolve-debug-torch.sh` recorded, unless the internal index
# already has it, and upload it as `pytorch/dist/<wheel file>`.
#
# Usage:
#   build-debug-torch.sh --python-version <X.Y>
#
# Environment: `MAX_JOBS`, the build's parallelism.
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
if [[ -z "${MAX_JOBS:-}" ]]; then
  echo "MAX_JOBS is not set; check the step env" >&2
  exit "${EXIT_ENVIRONMENT}"
fi
needs_build="$(read_meta_data "debug-torch-needs-build-${python_tag}")"
if [[ "${needs_build}" != true ]]; then
  echo "The debug torch wheel is already on the internal package index; nothing to build"
  exit 0
fi
debug_torch_version="$(read_meta_data "debug-torch-version-${python_tag}")"
wheel_file="$(read_meta_data "debug-torch-wheel-${python_tag}")"
torch_version="${debug_torch_version%+cpu.debug}"

echo "--- Checking out PyTorch v${torch_version}"
git clone \
  --depth 1 \
  --branch "v${torch_version}" \
  --recurse-submodules \
  --shallow-submodules \
  --jobs 8 \
  https://github.com/pytorch/pytorch \
  pytorch

echo "--- Installing build dependencies"
cd pytorch
uv venv .venv
uv pip install \
  --python .venv/bin/python \
  --group dev \
  mkl-include \
  mkl-static

echo "--- Building debug torch wheel"
# `--no-build-isolation` builds in the venv, whose MKL and CMake an isolated build would not see.
# `BLAS=MKL` makes the build fail rather than fall back to Eigen, so the debug `torch` computes like
# the release one.
DEBUG=1 \
BUILD_TEST=0 \
USE_CUDA=0 \
USE_ROCM=0 \
USE_XPU=0 \
USE_MAGMA=0 \
USE_MPI=0 \
BLAS=MKL \
CMAKE_PREFIX_PATH="${PWD}/.venv" \
PYTORCH_BUILD_VERSION="${debug_torch_version}" \
PYTORCH_BUILD_NUMBER=1 \
uv build \
  --wheel \
  --no-build-isolation
wheel="$(single_wheel dist "${EXIT_FAILURE}")"
if [[ "$(basename "${wheel}")" != "${wheel_file}" ]]; then
  echo "Built $(basename "${wheel}"), but the index check looked for ${wheel_file}" >&2
  exit "${EXIT_FAILURE}"
fi
echo "Built ${wheel_file}"
cd ..

echo "--- Uploading debug torch wheel"
buildkite-agent artifact upload "pytorch/${wheel}"
