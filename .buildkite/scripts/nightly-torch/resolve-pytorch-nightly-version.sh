#!/usr/bin/env bash
# Resolve the PyTorch nightly that the build tests against, and record it in the meta-data key
# `nightly-torch-version`.
#
# Usage:
#   resolve-pytorch-nightly-version.sh
#
# Environment: `TORCH_RBLN_TORCH_VERSION`, the nightly version a manual build pins (unset or empty:
# the latest nightly).
set -euo pipefail

# shellcheck source=.buildkite/scripts/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"

if [[ $# -gt 0 ]]; then
  echo "resolve-pytorch-nightly-version.sh takes no arguments: $*" >&2
  exit "${EXIT_USAGE}"
fi
# `read` trims the whitespace that a value pasted into the New Build dialog may carry.
read -r requested <<< "${TORCH_RBLN_TORCH_VERSION:-}"

echo "--- Resolving PyTorch nightly version"
# A requested version resolves too, to the `+cpu` form (`2.14.0.dev20260810` to
# `2.14.0.dev20260810+cpu`) that the build's version check compares with `torch.__version__`.
requirement=torch
if [[ -n "${requested}" ]]; then
  requirement="torch==${requested}"
fi
target="$(resolve_version pytorch-nightly-cpu "${requirement}")"
echo "Testing against torch==${target}"
buildkite-agent meta-data set nightly-torch-version "${target}"
