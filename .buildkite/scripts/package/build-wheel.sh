#!/usr/bin/env bash
# Build and verify a wheel, which must match `TORCH_RBLN_VERSION_TAG` if set.
#
# Usage:
#   build-wheel.sh --build-type <Release|Debug>
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

if [[ "${build_type}" == "Debug" ]]; then
  torch_version="$(
    .buildkite/scripts/package/resolve_torch_build_version.py \
      --build-type Debug
  )"

  echo "--- Pinning torch to ${torch_version}"
  .buildkite/scripts/package/pin_torch.py \
    --torch-version "${torch_version}" \
    --torch-index rbln-internal
  uv lock
fi

echo "--- Building ${artifact_dir}"
TORCH_RBLN_BUILD_TYPE="${build_type}" \
TORCH_RBLN_BUILD_TESTING="OFF" \
uv build \
  --wheel \
  --out-dir "${artifact_dir}"
wheels=("${artifact_dir}"/*.whl)
wheel="${wheels[0]}"
version="$(
  .buildkite/scripts/package/read_wheel_version.py \
    --wheel "${wheel}"
)"
echo "Built torch-rbln==${version}"
if [[ -n "${TORCH_RBLN_VERSION_TAG:-}" \
  && "${version}" != "${TORCH_RBLN_VERSION_TAG#v}" ]]; then
  echo "Built torch-rbln==${version}, not ${TORCH_RBLN_VERSION_TAG#v};" \
    "the checkout lacks the tag" >&2
  exit 1
fi

echo "--- Installing wheel in clean environment"
uv venv \
  verify-install/.venv
# Installing a built wheel skips the indexes `pyproject.toml` marks
# `explicit`, so `--index` names the one `torch` comes from.
torch_index="$(
  .buildkite/scripts/read_index_setting.py \
    --index-of torch \
    --setting name
)"
torch_index_url="$(
  .buildkite/scripts/read_index_setting.py \
    --index-of torch \
    --setting url
)"
uv pip install \
  --python verify-install/.venv/bin/python \
  --index "${torch_index}=${torch_index_url}" \
  "${wheel}[runtime]"

echo "--- Verifying versions and debug marker"
verify-install/.venv/bin/python \
  .buildkite/scripts/package/verify_installed_versions.py \
  --torch-rbln-version "${version}" \
  --build-type "${build_type}"

echo "--- Verifying install RPATH"
env \
  --unset=LD_LIBRARY_PATH \
  verify-install/.venv/bin/python \
    .buildkite/scripts/package/verify_install_rpath.py

echo "--- Uploading wheel, pyproject.toml, and uv.lock"
cp \
  pyproject.toml \
  uv.lock \
  "${artifact_dir}/"
buildkite-agent artifact upload \
  "${wheel};${artifact_dir}/pyproject.toml;${artifact_dir}/uv.lock"
buildkite-agent meta-data set \
  "${artifact_dir}" \
  "$(basename "${wheel}")"
