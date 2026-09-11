#!/usr/bin/env bash
# Move tools/linter to the upstream PyTorch tag that pyproject.toml pins.
#
# tools/linter is a verbatim copy of upstream PyTorch's tools/linter. It moves only as
# part of a torch version bump, so the target tag is always derived from the torch pin
# in pyproject.toml (torch==X.Y.Z+cpu -> vX.Y.Z); there is no way to pass another tag.
# The tag the tree was last synced from is recorded in tools/linter/UPSTREAM_TAG.
#
# Usage:
#   ./tools/sync-linter.sh          # sync tools/linter to the pinned tag (no-op if already there)
#   ./tools/sync-linter.sh --check  # exit 1 if tools/linter is behind the pinned tag
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYPROJECT="${REPO_ROOT}/pyproject.toml"
LINTER_DIR="${REPO_ROOT}/tools/linter"
TAG_FILE="${LINTER_DIR}/UPSTREAM_TAG"

check_only=0
case "${1:-}" in
  "") ;;
  --check) check_only=1 ;;
  *)
    echo "Usage: $0 [--check]" >&2
    echo "The target tag comes from the torch pin in pyproject.toml; it cannot be overridden." >&2
    exit 2
    ;;
esac

torch_ver=$(grep -E 'torch==[0-9]+\.[0-9]+\.[0-9]+' "${PYPROJECT}" | head -1 \
  | sed -E 's/.*torch==([0-9]+\.[0-9]+\.[0-9]+).*/\1/' || true)
if [[ -z "${torch_ver}" ]]; then
  echo "Cannot find a torch==X.Y.Z pin in ${PYPROJECT}" >&2
  exit 1
fi
target="v${torch_ver}"
current="$(cat "${TAG_FILE}" 2>/dev/null || echo "<none>")"

if [[ "${current}" == "${target}" ]]; then
  echo "tools/linter is already at ${target}; nothing to do."
  exit 0
fi

if [[ "${check_only}" -eq 1 ]]; then
  echo "tools/linter is at ${current} but pyproject.toml pins torch ${torch_ver} (${target})." >&2
  echo "Run ./tools/sync-linter.sh as part of the torch bump." >&2
  exit 1
fi

# The sync replaces the whole tree; refuse to overwrite uncommitted edits under it.
dirty="$(git -C "${REPO_ROOT}" status --porcelain -- tools/linter)"
if [[ -n "${dirty}" ]]; then
  echo "tools/linter has uncommitted changes; commit or discard them before syncing." >&2
  exit 1
fi

tmp=$(mktemp -d)
trap 'rm -rf "${tmp}"' EXIT

echo "Fetching pytorch/pytorch tools/linter at ${target}..."
git clone --quiet --depth 1 --filter=blob:none --no-checkout \
  https://github.com/pytorch/pytorch.git "${tmp}"
git -C "${tmp}" sparse-checkout init --cone >/dev/null
git -C "${tmp}" sparse-checkout set tools/linter >/dev/null
git -C "${tmp}" fetch --quiet origin tag "${target}" --depth 1
git -C "${tmp}" checkout --quiet "${target}"

rm -rf "${LINTER_DIR}"
cp -r "${tmp}/tools/linter" "${LINTER_DIR}"
find "${LINTER_DIR}" -name __pycache__ -type d -prune -exec rm -rf {} +
echo "${target}" > "${TAG_FILE}"

echo "Synced tools/linter ${current} -> ${target}. Review the diff and commit it with the torch bump."
