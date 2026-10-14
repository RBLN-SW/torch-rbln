#!/usr/bin/env bash
# Build one wheel, verify its install, and upload it under `wheels/py<X.Y>-<arch>-<build type>/`.
#
# Usage:
#   build-wheel.sh --python-version <X.Y> --build-type <Release|Debug>
#                  [--torch-index <uv index name>] [--torch-version <version>]
#
# `--torch-index` names the index that the verifying install takes `torch` from when it is not
# `pytorch-cpu`, and `--torch-version` the `torch` version the install must run on.
#
# Environment: the `UV_INDEX_*` index credentials and, in a release, `TORCH_RBLN_VERSION_TAG`,
# which the built version must equal without its `v`.
set -euo pipefail

# shellcheck source=.buildkite/scripts/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"

python_version=""
build_type=""
torch_index=pytorch-cpu
torch_version=""
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
    --torch-index)
      torch_index="$2"
      shift 2
      ;;
    --torch-version)
      torch_version="$2"
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
cpu_architecture="$(uname -m)"
# The steps that install or publish the wheel download it by this path.
name="py${python_version}-${cpu_architecture}-${build_type}"

echo "--- Building wheel ${name}"
export TORCH_RBLN_BUILD_TYPE="${build_type}"
export TORCH_RBLN_BUILD_TESTING=OFF
rm \
  --recursive \
  --force \
  "wheels/${name}"
uv build \
  --wheel \
  --out-dir "wheels/${name}"
wheel="$(single_wheel "wheels/${name}" "${EXIT_FAILURE}")"
wheel_info="$(uv run --no-config "${SCRIPTS_DIR}/read-wheel-version.py" "${wheel}")"
read -r version _ <<< "${wheel_info}"
echo "Built torch-rbln==${version}"
if [[ -n "${TORCH_RBLN_VERSION_TAG:-}" \
  && "${version}" != "${TORCH_RBLN_VERSION_TAG#v}" ]]; then
  echo "Built torch-rbln==${version}, not ${TORCH_RBLN_VERSION_TAG#v};" \
    "the checkout lacks the tag" >&2
  exit "${EXIT_FAILURE}"
fi

echo "--- Installing wheel in clean environment"
uv venv verify-install/.venv
# `pyproject.toml` declares PyTorch's indexes explicit, and an install of a built wheel does not use
# them, so `UV_INDEX` names the one `torch` comes from.
torch_index_url="$(index_setting "${torch_index}" url)"
UV_INDEX="${torch_index}=${torch_index_url}" uv pip install \
  --python verify-install/.venv/bin/python \
  "${wheel}[runtime]"

echo "--- Verifying versions and debug marker"
# The checks run as files, which put their own directory rather than the source tree on
# `sys.path`, so the source tree does not shadow the installed wheel.
version_args=(--torch-rbln "${version}" --build-type "${build_type}")
if [[ -n "${torch_version}" ]]; then
  version_args+=(--torch "${torch_version}")
fi
verify-install/.venv/bin/python \
  "${SCRIPTS_DIR}/packaging/verify-installed-versions.py" \
  "${version_args[@]}"

echo "--- Verifying install RPATH"
env --unset=LD_LIBRARY_PATH TORCH_DEVICE_BACKEND_AUTOLOAD=0 verify-install/.venv/bin/python \
  "${SCRIPTS_DIR}/packaging/verify-install-rpath.py"

echo "--- Uploading wheel"
buildkite-agent artifact upload "${wheel}"
