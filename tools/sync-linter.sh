#!/usr/bin/env bash
# Sync tools/linter from the PyTorch repository into the current project.
#
# Usage:
#   ./sync-linter.sh              # Sync using default version from torch-rbln/pyproject.toml
#   ./sync-linter.sh v2.10.0      # Sync from the given PyTorch tag
#
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$SCRIPT_DIR/torch-rbln/pyproject.toml" ]; then
  PYPROJECT="$SCRIPT_DIR/torch-rbln/pyproject.toml"
elif [ -f "$SCRIPT_DIR/pyproject.toml" ]; then
  PYPROJECT="$SCRIPT_DIR/pyproject.toml"
elif [ -f "$SCRIPT_DIR/../pyproject.toml" ]; then
  PYPROJECT="$SCRIPT_DIR/../pyproject.toml"
else
  echo "Cannot find torch-rbln pyproject.toml" >&2
  exit 1
fi

# Default version: parse torch version (e.g. 2.9.0) from pyproject.toml and use as tag v2.9.0
TORCH_VER=$(grep -E 'torch==[0-9]+\.[0-9]+\.[0-9]+' "$PYPROJECT" | head -1 | sed -E 's/.*torch==([0-9]+\.[0-9]+\.[0-9]+).*/\1/')
VERSION="${1:-v${TORCH_VER}}"

TMP=$(mktemp -d)

git clone --depth 1 --filter=blob:none --no-checkout \
  https://github.com/pytorch/pytorch.git $TMP

cd $TMP
git sparse-checkout init --cone
git sparse-checkout set tools/linter
git fetch origin tag "$VERSION" --depth 1
git checkout "$VERSION"

cd -

rm -rf tools/linter
cp -r $TMP/tools/linter tools/linter

echo "Synced tools/linter from PyTorch $VERSION"
