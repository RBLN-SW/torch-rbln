#!/bin/bash
# =============================================================================
# export-inference-requirements.sh — Refresh tools/test/requirements-inference.txt
# =============================================================================
#
# Exports vllm-rbln's uv.lock at a ref into the pinned requirements file that
# install-test-deps.sh installs (--no-deps) into the inference venv. torch and
# torch-rbln are left out: the inference venv shares the test venv's copies.
#
# Usage:
#   ./tools/test/export-inference-requirements.sh        (needs uv)
#
# Optional environment (as in install-test-deps.sh):
#   VLLM_RBLN_REPO / VLLM_RBLN_REF / VLLM_RBLN_DIR
#   The default ref is origin/ci/torch-rbln-model-tests; the resolved commit
#   is recorded in the output and is what install-test-deps.sh checks out.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(realpath "$(dirname "$0")")"
PROJECT_ROOT="$(realpath "${SCRIPT_DIR}/../..")"
OUTPUT="${SCRIPT_DIR}/requirements-inference.txt"

repo="${VLLM_RBLN_REPO:-https://github.com/rbln-sw/vllm-rbln.git}"
ref="${VLLM_RBLN_REF:-origin/ci/torch-rbln-model-tests}"
dir="${VLLM_RBLN_DIR:-${PROJECT_ROOT}/vllm-rbln}"

if ! git -C "${dir}" rev-parse --git-dir >/dev/null 2>&1; then
  git clone "${repo}" "${dir}"
fi
(cd "${dir}" && git fetch origin --prune && git checkout --detach "${ref}")
commit="$(git -C "${dir}" rev-parse HEAD)"

tmp="$(mktemp)"
trap 'rm -f "${tmp}"' EXIT

# --locked fails if vllm-rbln's lock is out of date with its pyproject at this
# ref. pip reads the emitted index options from the requirements file.
(cd "${dir}" && uv export \
  --locked \
  --no-hashes \
  --no-annotate \
  --no-emit-project \
  --no-default-groups \
  --no-editable \
  --emit-index-url \
  --no-emit-package torch \
  --no-emit-package torch-rbln \
  --output-file "${tmp}")

{
  echo "# Inference venv requirements, exported from vllm-rbln's uv.lock."
  echo "# Regenerate with tools/test/export-inference-requirements.sh; do not edit."
  echo "# vllm-rbln-repo: ${repo}"
  echo "# vllm-rbln-ref: ${commit}"
  echo "# torch and torch-rbln are omitted: the inference venv shares the test venv's."
  echo
  # Drop uv's banner and the internal mirror; external users install from this file.
  grep -vE '^#|nexus\.mgmt\.rbln\.in' "${tmp}"
} > "${OUTPUT}"

if grep -qE '^(torch|torch-rbln)==' "${OUTPUT}"; then
  echo "export still lists torch or torch-rbln; refusing to write ${OUTPUT}" >&2
  exit 1
fi
echo "Wrote ${OUTPUT} from vllm-rbln ${commit}"
