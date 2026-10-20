#!/usr/bin/env bash
# Build and verify a wheel against the latest PyTorch nightly, or the one
# `TORCH_RBLN_TORCH_VERSION` names.
#
# Usage:
#   build-wheel-nightly-torch.sh --artifact-dir <name>
set -euo pipefail

artifact_dir=""
while [[ $# -gt 0 ]]; do
  case "$1" in
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

if [[ -z "${artifact_dir}" ]]; then
  echo "--artifact-dir is required" >&2
  exit 2
fi

read \
  -r \
  requested \
  <<<"${TORCH_RBLN_TORCH_VERSION:-}"

echo "--- Resolving torch version"
requirement="torch"
if [[ -n "${requested}" ]]; then
  requirement="torch==${requested}"
fi
index_url="$(
  .buildkite/scripts/read-index-setting.py \
    --index pytorch-nightly-cpu \
    --setting url
)"
# Without `--no-config`, a version this index lacks could resolve from another
# index that `pyproject.toml` declares.
if ! pinned_requirement="$(
  uv pip compile \
    - \
    --no-config \
    --index "pytorch-nightly-cpu=${index_url}" \
    --no-deps \
    --no-header \
    --no-annotate \
    --prerelease allow \
    <<<"${requirement}"
)"; then
  echo "Could not resolve ${requirement} from the pytorch-nightly-cpu index" >&2
  exit 1
fi
torch_version="$(
  sed \
    --quiet \
    's/^[^=]*==//p' \
    <<<"${pinned_requirement}"
)"
if [[ -z "${torch_version}" ]]; then
  echo "The pytorch-nightly-cpu index resolved ${requirement} to no version:" \
    "${pinned_requirement}" >&2
  exit 1
fi
echo "Resolved ${requirement} to ${torch_version}"

echo "--- Pinning torch to ${torch_version}"
.buildkite/scripts/package/pin-torch.py \
  --torch-version "${torch_version}" \
  --torch-index pytorch-nightly-cpu
uv lock

.buildkite/scripts/package/build-wheel.sh \
  --build-type Release \
  --artifact-dir "${artifact_dir}"
