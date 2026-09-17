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
# Steps:
#   1. Test runner   : pytest, pytest-xdist
#   2. Test infra    : expecttest (for torch.testing._internal)
#   3. Model tests   : pandas, transformers 4 (the line test_transformers.py's
#                      models load under)
#
# Usage:
#   ./tools/test/install-test-deps.sh [--dry-run]
#
# Optional environment:
#   UV=1                 Use ``uv pip install`` instead of ``python -m pip``.
# =============================================================================

set -euo pipefail

# ----- arg parsing ----------------------------------------------------------

DRY_RUN=0
for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    -h|--help)
      sed -n '2,30p' "$0"
      exit 0
      ;;
    *) echo "Unknown option: ${arg}" >&2; exit 1 ;;
  esac
done

# ----- helpers --------------------------------------------------------------

run() {
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[dry-run] $*"
    return 0
  fi
  "$@"
}

pip_install() {
  if [[ "${UV:-0}" = "1" ]]; then
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

# ----- step 3: model-test deps (pandas + transformers) ----------------------
#
# pandas is a plain PyPI package used by test/models/test_transformers.py.
#
# transformers is pinned to what the RBLN PyTorch tutorial requires: this suite
# exists to keep that tutorial working, so it has to run what the tutorial runs.
# Two more reasons the pin cannot float: graph mode stops working above 4.52
# (4.53+ fails at runtime, 4.56+ no longer traces through torch.export), and no
# transformers 5 release runs EXAONE-3.5's hub modeling code.

install_model_test_deps() {
  log_step "Model-test deps (pandas + transformers)"
  pip_install "pandas==2.2.3"
  pip_install "transformers==4.49.0"
}

# ----- main -----------------------------------------------------------------

install_test_runner
install_test_infra
install_model_test_deps

echo
echo "All test dependencies installed."
