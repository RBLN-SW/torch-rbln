#!/usr/bin/env bash
# Publish a wheel to an index.
#
# Usage:
#   publish-wheel.sh --index <uv index name>
#                    --artifact-build-id <build ID>
#                    --artifact-dir <name>
set -euo pipefail

index=""
artifact_build_id=""
artifact_dir=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --index)
      index="$2"
      shift 2
      ;;
    --artifact-build-id)
      artifact_build_id="$2"
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

if [[ -z "${index}" ]]; then
  echo "--index is required" >&2
  exit 2
fi

if [[ -z "${artifact_build_id}" ]]; then
  echo "--artifact-build-id is required" >&2
  exit 2
fi

if [[ -z "${artifact_dir}" ]]; then
  echo "--artifact-dir is required" >&2
  exit 2
fi

echo "--- Looking up wheel recorded under ${artifact_dir}"
status=0
buildkite-agent meta-data exists \
  --build "${artifact_build_id}" \
  "${artifact_dir}" \
  || status=$?
if [[ "${status}" -eq 100 ]]; then
  echo "No wheel is recorded under ${artifact_dir}; nothing to publish"
  exit 0
fi
if [[ "${status}" -ne 0 ]]; then
  exit "${status}"
fi
wheel_file="$(
  buildkite-agent meta-data get \
    --build "${artifact_build_id}" \
    "${artifact_dir}"
)"
wheel="${artifact_dir}/${wheel_file}"

echo "--- Downloading ${wheel}"
buildkite-agent artifact download \
  --build "${artifact_build_id}" \
  "${wheel}" \
  .

echo "--- Publishing ${wheel_file} to ${index} index"
already_published="$(
  .buildkite/scripts/check-index-listing.py \
    --index "${index}" \
    --wheel-filename "${wheel_file}"
)"
if [[ "${already_published}" == "true" ]]; then
  echo "${wheel_file} is already on the ${index} index; nothing to publish"
  exit 0
fi
uv publish \
  --index "${index}" \
  "${wheel}"
echo "Published ${wheel_file} to the ${index} index"
