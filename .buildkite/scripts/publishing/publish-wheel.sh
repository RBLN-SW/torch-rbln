#!/usr/bin/env bash
# Publish the wheel a build step uploaded to an index that `pyproject.toml` declares with a
# `publish-url`, unless the index already lists it.
#
# Usage:
#   publish-wheel.sh --python-version <X.Y> --cpu-architecture <x86_64|aarch64>
#                    --build-type <Release|Debug> --index <uv index name>
#                    [--version-kind <dev|release>]
#
# `--version-kind dev` skips a wheel without a `.dev` segment, and `--version-kind release` rejects
# one with it.
#
# Environment: the index's credentials, `UV_INDEX_<INDEX>_USERNAME` and `UV_INDEX_<INDEX>_PASSWORD`.
set -euo pipefail

# shellcheck source=.buildkite/scripts/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"

python_version=""
build_type=""
cpu_architecture=""
index=""
version_kind=""
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
    --cpu-architecture)
      cpu_architecture="$2"
      shift 2
      ;;
    --index)
      index="$2"
      shift 2
      ;;
    --version-kind)
      version_kind="$2"
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
if [[ -z "${cpu_architecture}" ]]; then
  echo "--cpu-architecture is required" >&2
  exit "${EXIT_USAGE}"
fi
if [[ -z "${index}" ]]; then
  echo "--index is required" >&2
  exit "${EXIT_USAGE}"
fi
case "${version_kind}" in
  "" | dev | release) ;;
  *)
    echo "--version-kind is '${version_kind}'; expected dev or release" >&2
    exit "${EXIT_USAGE}"
    ;;
esac
name="py${python_version}-${cpu_architecture}-${build_type}"

echo "--- Downloading wheel ${name}"
mkdir --parents wheel-artifact
if ! buildkite-agent artifact download "wheels/${name}/*.whl" wheel-artifact; then
  echo "Could not download wheels/${name}/ from this build's artifacts" >&2
  exit "${EXIT_ENVIRONMENT}"
fi
wheel="$(single_wheel "wheel-artifact/wheels/${name}" "${EXIT_ENVIRONMENT}")"
wheel_file="$(basename "${wheel}")"
wheel_info="$(uv run --no-config "${SCRIPTS_DIR}/read-wheel-version.py" "${wheel}")"
read -r version kind <<< "${wheel_info}"

echo "--- Publishing ${wheel_file} to ${index} index"
case "${version_kind}:${kind}" in
  dev:release)
    echo "torch-rbln==${version} is not a development version; not publishing to the ${index} index"
    exit 0
    ;;
  release:dev)
    echo "torch-rbln==${version} is a development version;" \
      "a release publishes the version of its tag" >&2
    exit "${EXIT_FAILURE}"
    ;;
  *) ;;
esac
already_published="$(index_has_wheel "${index}" torch-rbln "${wheel_file}")"
if [[ "${already_published}" == true ]]; then
  echo "${wheel_file} is already on the ${index} index; nothing to publish"
  exit 0
fi
username_var="$(index_credential_var "${index}" USERNAME)"
password_var="$(index_credential_var "${index}" PASSWORD)"
# The upload of `uv publish` takes credentials only from `UV_PUBLISH_*`, not from the index's
# `UV_INDEX_*`.
UV_PUBLISH_USERNAME="${!username_var}" \
UV_PUBLISH_PASSWORD="${!password_var}" \
uv publish \
  --index "${index}" \
  "${wheel}"
echo "Published torch-rbln==${version} to the ${index} index"
