#!/usr/bin/env bash
# Sync `tools/linter`, `.clang-format`, and `.clang-tidy` from upstream PyTorch
# at a tag, by default the tag of the `torch` pin in `pyproject.toml`.
#
# Usage:
#   sync-linter.sh [<tag>]
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
pyproject="${repo_root}/pyproject.toml"

torch_version="$(
  grep \
    --extended-regexp \
    'torch==[0-9]+\.[0-9]+\.[0-9]+' \
    "${pyproject}" \
    | head \
      --lines=1 \
    | sed \
      --regexp-extended \
      's/.*torch==([0-9]+\.[0-9]+\.[0-9]+).*/\1/' \
    || true
)"
if [[ $# -eq 0 && -z "${torch_version}" ]]; then
  echo "No \`torch==X.Y.Z\` pin found in ${pyproject}; pass a tag." >&2
  exit 1
fi
torch_tag="${1:-v${torch_version}}"

pytorch_dir="$(mktemp --directory)"
trap 'rm --recursive --force "${pytorch_dir}"' EXIT

git clone \
  --depth 1 \
  --filter=blob:none \
  --no-checkout \
  https://github.com/pytorch/pytorch.git \
  "${pytorch_dir}"
git -C "${pytorch_dir}" sparse-checkout init \
  --cone
git -C "${pytorch_dir}" sparse-checkout set \
  tools/linter
git -C "${pytorch_dir}" fetch \
  --depth 1 \
  origin \
  tag "${torch_tag}"
git -C "${pytorch_dir}" checkout \
  "${torch_tag}"

rm \
  --recursive \
  --force \
  "${repo_root}/tools/linter"
cp \
  --recursive \
  "${pytorch_dir}/tools/linter" \
  "${repo_root}/tools/linter"
cp \
  "${pytorch_dir}/.clang-format" \
  "${pytorch_dir}/.clang-tidy" \
  "${repo_root}"

echo "Synced tools/linter, .clang-format, and .clang-tidy" \
  "from PyTorch ${torch_tag}"
