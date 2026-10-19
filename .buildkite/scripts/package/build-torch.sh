#!/usr/bin/env bash
# Build a wheel of the pinned `torch` from source.
#
# Usage:
#   build-torch.sh --build-type <Release|Debug>
#                  --artifact-dir <name>
set -euo pipefail

build_type=""
artifact_dir=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-type)
      build_type="$2"
      shift 2
      ;;
    --artifact-dir)
      artifact_dir="$2"
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

if [[ -z "${artifact_dir}" ]]; then
  echo "--artifact-dir is required" >&2
  exit 2
fi

: "${MAX_JOBS:?is not set}"

python="$(
  command \
    -v \
    python3
)"
python_tag="$(
  "${python}" \
    -c 'import sys; print("cp%d%d" % sys.version_info[:2])'
)"
compatibility_tags="${python_tag}-${python_tag}-linux_$(uname -m)"

echo "--- Checking rbln-internal index"
torch_version="$(
  .buildkite/scripts/package/resolve-torch-build-version.py \
    --build-type "${build_type}"
)"
wheel_file="torch-${torch_version}-${compatibility_tags}.whl"
already_published="$(
  .buildkite/scripts/check-index-listing.py \
    --index rbln-internal \
    --wheel-filename "${wheel_file}"
)"
if [[ "${already_published}" == "true" ]]; then
  echo "${wheel_file} is already on the rbln-internal index; nothing to build"
  exit 0
fi
public_version="${torch_version%%+*}"

echo "--- Checking out PyTorch v${public_version}"
git clone \
  --depth 1 \
  --branch "v${public_version}" \
  --recurse-submodules \
  --shallow-submodules \
  --jobs "${MAX_JOBS}" \
  https://github.com/pytorch/pytorch \
  pytorch

echo "--- Installing build dependencies"
pushd \
  pytorch
uv venv \
  --python "${python}" \
  .venv
uv pip install \
  --python .venv/bin/python \
  --group dev \
  mkl-include \
  mkl-static

echo "--- Building torch wheel"
debug=0
if [[ "${build_type}" == "Debug" ]]; then
  debug=1
fi
# Without `BLAS=MKL`, a build that cannot find MKL would fall back to Eigen
# instead of failing.
DEBUG="${debug}" \
BUILD_TEST=0 \
USE_CUDA=0 \
USE_ROCM=0 \
USE_XPU=0 \
USE_MAGMA=0 \
USE_MPI=0 \
BLAS="MKL" \
CMAKE_PREFIX_PATH="${PWD}/.venv" \
PYTORCH_BUILD_VERSION="${torch_version}" \
PYTORCH_BUILD_NUMBER=1 \
uv build \
  --wheel \
  --no-build-isolation \
  --out-dir "../${artifact_dir}"
popd
wheel="${artifact_dir}/${wheel_file}"
if [[ ! -f "${wheel}" ]]; then
  echo "uv build did not write ${wheel}," \
    "the file that the index check looked up" >&2
  exit 1
fi

echo "--- Uploading torch wheel"
buildkite-agent artifact upload \
  "${wheel}"
buildkite-agent meta-data set \
  "${artifact_dir}" \
  "${wheel_file}"
