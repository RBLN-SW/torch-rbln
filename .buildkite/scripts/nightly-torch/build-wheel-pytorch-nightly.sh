#!/usr/bin/env bash
# Build the Release wheel against the PyTorch nightly that `resolve-pytorch-nightly-version.sh`
# recorded, with the `torch` dependency of `pyproject.toml` repointed to it.
#
# Usage:
#   build-wheel-pytorch-nightly.sh --python-version <X.Y>
#
# Environment: the `UV_INDEX_*` index credentials.
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
target="$(read_meta_data nightly-torch-version)"

echo "--- Repointing torch to nightly wheel"
python3 \
  tools/replace_depends.py \
  --source torch \
  --target "torch==${target}" \
  --pyproject-path pyproject.toml \
  --set-index pytorch-nightly-cpu
if ! grep \
  --quiet \
  --fixed-strings \
  "\"torch==${target}\"" \
  pyproject.toml; then
  echo "Failed to repoint the torch dependency to ${target}" >&2
  exit "${EXIT_FAILURE}"
fi
grep \
  --line-number \
  --extended-regexp \
  '"torch==|^torch = ' \
  pyproject.toml

"${SCRIPTS_DIR}/packaging/build-wheel.sh" \
  --python-version "${python_version}" \
  --build-type Release \
  --torch-index pytorch-nightly-cpu \
  --torch-version "${target}"
