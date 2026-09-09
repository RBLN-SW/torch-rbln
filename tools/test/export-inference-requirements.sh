#!/bin/bash
# =============================================================================
# export-inference-requirements.sh — Refresh tools/test/requirements-inference.txt
# =============================================================================
#
# The inference venv that install-test-deps.sh builds for test_vllm_llm.py and
# test_optimum_llm.py is installed from a pinned requirements file, not from a
# dependency resolution at install time. This script produces that file from
# vllm-rbln's own uv.lock at a chosen ref, so the stack under test is exactly
# the one vllm-rbln tests itself with, and a change to it is a reviewable diff.
#
# torch and torch-rbln are left out of the export: the inference venv shares
# the test venv's copies, and a listed pin would make pip replace the package
# under test (vllm's own requirement on torch names the release wheel, which a
# locally built or debug torch never matches). install-test-deps.sh refuses a
# file that lists either.
#
# Usage:
#   ./tools/test/export-inference-requirements.sh
#
# Requires ``uv`` (vllm-rbln's pyproject states the minimum version).
#
# Optional environment (shared with install-test-deps.sh):
#   VLLM_RBLN_REPO       Source repo (default rbln-sw/vllm-rbln).
#   VLLM_RBLN_REF        Ref to export from (default origin/ci/torch-rbln-model-tests).
#                        The resolved commit is recorded in the output file and
#                        is what install-test-deps.sh checks out.
#   VLLM_RBLN_DIR        Local checkout path (default ``$PROJECT_ROOT/vllm-rbln``).
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(realpath "$(dirname "$0")")"
readonly SCRIPT_DIR
PROJECT_ROOT="$(realpath "${SCRIPT_DIR}/../..")"
readonly PROJECT_ROOT
OUTPUT="${SCRIPT_DIR}/requirements-inference.txt"
readonly OUTPUT

repo="${VLLM_RBLN_REPO:-https://github.com/rbln-sw/vllm-rbln.git}"
ref="${VLLM_RBLN_REF:-origin/ci/torch-rbln-model-tests}"
dir="${VLLM_RBLN_DIR:-${PROJECT_ROOT}/vllm-rbln}"

# A worktree keeps .git as a file, so ask git rather than test for a directory.
if ! git -C "${dir}" rev-parse --git-dir >/dev/null 2>&1; then
  echo "Cloning ${repo} into ${dir}..."
  git clone "${repo}" "${dir}"
fi
echo "Checking out ${ref} in ${dir}..."
(cd "${dir}" && git fetch origin --prune && git checkout --detach "${ref}")
commit="$(git -C "${dir}" rev-parse HEAD)"
readonly commit

tmp="$(mktemp)"
trap 'rm -f "${tmp}"' EXIT

# --frozen: the lock is vllm-rbln's statement of record; do not re-resolve it.
# --emit-index-url: the +cpu wheels live on the vLLM and PyTorch indexes, and
#   pip reads the index options from the requirements file itself.
(cd "${dir}" && uv export \
  --frozen \
  --no-hashes \
  --no-emit-project \
  --no-default-groups \
  --no-editable \
  --emit-index-url \
  --no-emit-package torch \
  --no-emit-package torch-rbln \
  --output-file "${tmp}")

{
  echo "# Inference-stack requirements for test/models/test_vllm_llm.py and"
  echo "# test/models/test_optimum_llm.py (installed into the inference venv by"
  echo "# tools/test/install-test-deps.sh, --no-deps)."
  echo "#"
  echo "# Exported from vllm-rbln's uv.lock. Do not edit by hand; regenerate with"
  echo "#   tools/test/export-inference-requirements.sh"
  echo "# vllm-rbln-repo: ${repo}"
  echo "# vllm-rbln-ref: ${commit}"
  echo "#"
  echo "# torch and torch-rbln are deliberately absent: the inference venv shares the"
  echo "# test venv's copies (see install-test-deps.sh, step 4)."
  echo
  # uv writes its own two-line banner (with this machine's temp path) first;
  # the header above replaces it. Internal mirrors are for developers' own
  # pip configuration, not for a file external users install from.
  grep -vE '^#|nexus\.mgmt\.rbln\.in' "${tmp}"
} > "${OUTPUT}"

if grep -qE '^(torch|torch-rbln)==' "${OUTPUT}"; then
  echo "export still lists torch or torch-rbln; refusing to write ${OUTPUT}" >&2
  exit 1
fi

echo "Wrote ${OUTPUT} from vllm-rbln ${commit}"
