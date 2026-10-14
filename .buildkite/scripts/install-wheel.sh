#!/usr/bin/env bash
# Install the x86_64 wheel a build step uploaded, with its `runtime` extra, into the locked
# environment in `.venv`, which callers run with `uv run --no-sync`.
#
# Usage:
#   install-wheel.sh --python-version <X.Y> --build-type <Release|Debug> [--debug-torch]
#                    [--torch-index <uv index name>]
#
# `--debug-torch` installs the debug build of the `torch` pin in place of the release one, and
# `--torch-index` names the index that serves the wheel's `torch` when it is not `pytorch-cpu`.
#
# Environment: the `UV_INDEX_*` index credentials.
set -euo pipefail

# shellcheck source=.buildkite/scripts/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

python_version=""
build_type=""
debug_torch=false
torch_index=""
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
    --debug-torch)
      debug_torch=true
      shift
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
if [[ -z "${python_version}" ]]; then
  echo "--python-version is required" >&2
  exit "${EXIT_USAGE}"
fi
if [[ -z "${build_type}" ]]; then
  echo "--build-type is required" >&2
  exit "${EXIT_USAGE}"
fi
name="py${python_version}-x86_64-${build_type}"

echo "--- Installing locked environment"
# `UV_INDEX` stays unset here because `--locked` rejects an index the lock does not record.
sync_args=()
if [[ "${debug_torch}" == true || -n "${torch_index}" ]]; then
  sync_args+=(--no-install-package torch)
fi
uv sync \
  --locked \
  --no-install-project \
  "${sync_args[@]}"

echo "--- Downloading wheel ${name}"
mkdir --parents wheel-artifact
if ! buildkite-agent artifact download "wheels/${name}/*.whl" wheel-artifact; then
  echo "Could not download wheels/${name}/ from this build's artifacts" >&2
  exit "${EXIT_ENVIRONMENT}"
fi
wheel="$(single_wheel "wheel-artifact/wheels/${name}" "${EXIT_ENVIRONMENT}")"

echo "--- Installing wheel ${name}"
echo "Installing $(basename "${wheel}")"
install_args=()
if [[ "${debug_torch}" == true ]]; then
  debug_torch_version="$(uv run --no-config "${SCRIPTS_DIR}/resolve-debug-torch-version.py")"
  echo "torch==${debug_torch_version}" > debug-torch-overrides.txt
  install_args+=(--overrides debug-torch-overrides.txt)
  echo "Installing torch==${debug_torch_version} in place of the wheel's torch requirement"
  # `dpkg-query` exits 1 for a package it does not know, which means not installed.
  libnuma_status="$(
    dpkg-query \
      --show \
      --showformat='${db:Status-Status}' \
      libnuma1 2>/dev/null \
      || true
  )"
  if [[ "${libnuma_status}" != installed ]]; then
    apt-get update
    apt-get install \
      --no-install-recommends \
      --yes \
      libnuma1
  fi
fi
if [[ -n "${torch_index}" ]]; then
  # `UV_INDEX` names the index for the wheel's `torch`, because an install of a built wheel does not
  # use the explicit indexes of `pyproject.toml`.
  torch_index_url="$(index_setting "${torch_index}" url)"
  export UV_INDEX="${torch_index}=${torch_index_url}"
fi
uv pip install \
  --python .venv/bin/python \
  "${install_args[@]}" \
  "${wheel}[runtime]"
