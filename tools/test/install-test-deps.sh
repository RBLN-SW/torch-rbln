#!/bin/bash
# =============================================================================
# install-test-deps.sh — Install dependencies needed to run torch-rbln tests
# =============================================================================
#
# Target users: External developers who want to run the full test suite without
# manually installing dependencies per test group.
#
# Assumptions:
#   - torch-rbln is already installed (e.g. via tools/dev-setup.sh).
#   - rebel-compiler is installed and usable.
#
# Steps (each can be skipped via env var if a CI runner already provides it):
#   1. Test runner   : pytest, pytest-xdist
#   2. Test infra    : expecttest (for torch.testing._internal)
#   3. Model tests   : torchvision (PyTorch CPU index) + pandas
#   4. vllm-rbln     : git clone + editable install; vllm itself is pulled in
#                      transitively from vllm-rbln's dependency pin so the
#                      version stays in lockstep with vllm-rbln upstream
#                      instead of being re-pinned here.
#
# Usage:
#   ./tools/test/install-test-deps.sh [--dry-run]
#
# Optional environment:
#   UV=1                 Use ``uv pip install`` instead of ``python -m pip``.
#   VLLM_RBLN_SKIP=1     Skip step 4 entirely (test_vllm_llm.py becomes a
#                        clean pytest.importorskip in that case).
#   VLLM_RBLN_REPO       Override the source repo (default rbln-sw/vllm-rbln).
#   VLLM_RBLN_REF        Override the ref (default origin/device_tensor_rebased).
#   VLLM_RBLN_DIR        Override the local checkout path
#                        (default ``$PROJECT_ROOT/vllm-rbln``).
# =============================================================================

set -euo pipefail

readonly SCRIPT_DIR="$(realpath "$(dirname "$0")")"
readonly PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../..")"

# ----- arg parsing ----------------------------------------------------------

DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    -h|--help)
      sed -n '2,30p' "$0"
      exit 0
      ;;
    *) echo "Unknown option: $arg" >&2; exit 1 ;;
  esac
done

# ----- helpers --------------------------------------------------------------

run() {
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "[dry-run] $*"
    return 0
  fi
  "$@"
}

pip_install() {
  if [ "${UV:-0}" = "1" ]; then
    run uv pip install "$@"
  else
    run python -m pip install "$@"
  fi
}

log_step() {
  echo
  echo "=== $* ==="
}

# ----- step 1: pytest -------------------------------------------------------

install_test_runner() {
  log_step "Test runner (pytest, pytest-xdist)"
  pip_install pytest pytest-xdist
}

# ----- step 2: test infra ---------------------------------------------------

install_test_infra() {
  log_step "Test infra (expecttest)"
  pip_install "expecttest>=0.3.0,<0.4.0"
}

# ----- step 3: model-test deps (torchvision + pandas) -----------------------
#
# torchvision needs the PyTorch CPU index (the +cpu wheel must come from
# download.pytorch.org rather than PyPI). pandas is a plain PyPI package used
# only by test/models/test_optimum_llm.py.
#
# transformers and optimum-rbln are NOT installed here — they arrive
# transitively via vllm-rbln in step 4. Environments that opt out of vllm-rbln
# (VLLM_RBLN_SKIP=1) and still want to run those tests must install both
# manually.

install_model_test_deps() {
  log_step "Model-test deps (torchvision CPU + pandas)"
  pip_install "torchvision==0.25.0+cpu" \
    --index-url https://download.pytorch.org/whl/cpu
  pip_install "pandas==2.2.3"
}

# ----- step 4: vllm-rbln ----------------------------------------------------
#
# vllm-rbln's device-tensor flow is not on PyPI yet, so we source-install the
# branch directly. We do NOT install ``vllm`` separately: vllm-rbln pins the
# right vllm version (and the matching vllm-cpu wheel index URL) in its own
# pyproject.toml, and pip pulls vllm transitively when we install vllm-rbln
# editable. This keeps the vllm version in lockstep with vllm-rbln's upstream
# pin instead of duplicating it here.

vllm_wheel_index_from_pyproject() {
  # Read the vllm-cpu index URL out of vllm-rbln's pyproject.toml so we don't
  # have to keep a separate vllm version pinned in this script.
  local pyproject="$1"
  python - "$pyproject" <<'PY'
import sys, tomllib, pathlib
data = tomllib.loads(pathlib.Path(sys.argv[1]).read_text())
for idx in data.get("tool", {}).get("uv", {}).get("index", []):
    if idx.get("name") == "vllm-cpu":
        print(idx["url"])
        sys.exit(0)
sys.exit(f"vllm-cpu index URL not found in {sys.argv[1]}")
PY
}

install_vllm_rbln() {
  if [ "${VLLM_RBLN_SKIP:-0}" = "1" ]; then
    log_step "vllm-rbln (skipped via VLLM_RBLN_SKIP=1)"
    return 0
  fi

  local repo="${VLLM_RBLN_REPO:-https://github.com/rbln-sw/vllm-rbln.git}"
  local ref="${VLLM_RBLN_REF:-origin/chan/remove_cpu_offload}"
  local dir="${VLLM_RBLN_DIR:-$PROJECT_ROOT/vllm-rbln}"

  log_step "vllm-rbln (clone + editable install at $ref)"

  if [ ! -d "$dir/.git" ]; then
    echo "Cloning $repo into $dir..."
    run git clone "$repo" "$dir"
  fi
  echo "Checking out $ref in $dir..."
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "[dry-run] (cd $dir && git fetch origin --prune && git checkout --detach $ref)"
  else
    (cd "$dir" && git fetch origin --prune && git checkout --detach "$ref")
  fi

  local vllm_index
  if [ "$DRY_RUN" -eq 1 ]; then
    vllm_index="<resolved-from-pyproject-at-runtime>"
  else
    vllm_index="$(vllm_wheel_index_from_pyproject "$dir/pyproject.toml")"
    echo "Resolved vllm wheel index from vllm-rbln pyproject: $vllm_index"
  fi

  # The rbln index is needed so pip can resolve vllm-rbln's transitive
  # ``optimum-rbln`` pin; transformers + other ordinary PyPI packages come
  # from the default index.
  pip_install -e "$dir" \
    --extra-index-url "$vllm_index" \
    --extra-index-url https://pypi.rbln.ai/simple/ \
    --extra-index-url https://download.pytorch.org/whl/cpu
}

# ----- main -----------------------------------------------------------------

install_test_runner
install_test_infra
install_model_test_deps
install_vllm_rbln

echo
echo "All test dependencies installed."
