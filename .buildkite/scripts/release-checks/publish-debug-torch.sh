#!/usr/bin/env bash
# Publish the debug `torch` wheel that `build-debug-torch.sh` built to the internal package index,
# unless the index already has it.
#
# Usage:
#   publish-debug-torch.sh --python-version <X.Y>
#
# Environment: the `rbln-internal` index credentials.
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

echo "--- Checking internal package index"
needs_build="$(read_meta_data "debug-torch-needs-build-${python_tag}")"
if [[ "${needs_build}" != true ]]; then
  echo "The debug torch wheel is already on the internal package index"
  exit 0
fi
wheel_file="$(read_meta_data "debug-torch-wheel-${python_tag}")"
# Another build, or an earlier attempt of this step, may have published it after
# `resolve-debug-torch.sh` checked.
already_published="$(index_has_wheel rbln-internal torch "${wheel_file}")"
if [[ "${already_published}" == true ]]; then
  echo "${wheel_file} is already on the internal package index; nothing to publish"
  exit 0
fi

echo "--- Downloading debug torch wheel"
if ! buildkite-agent artifact download \
  --step "build-debug-torch-py${python_version}" \
  "pytorch/dist/${wheel_file}" \
  .; then
  echo "Could not download pytorch/dist/${wheel_file} from this build's artifacts" >&2
  exit "${EXIT_ENVIRONMENT}"
fi

echo "--- Publishing ${wheel_file} to internal package index"
username_var="$(index_credential_var rbln-internal USERNAME)"
password_var="$(index_credential_var rbln-internal PASSWORD)"
# The upload of `uv publish` takes credentials only from `UV_PUBLISH_*`, not from the index's
# `UV_INDEX_*`.
UV_PUBLISH_USERNAME="${!username_var}" \
UV_PUBLISH_PASSWORD="${!password_var}" \
uv publish \
  --index rbln-internal \
  "pytorch/dist/${wheel_file}"
echo "Published ${wheel_file} to the internal package index"
