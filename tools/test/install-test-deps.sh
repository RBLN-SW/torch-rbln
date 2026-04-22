#!/bin/bash
# =============================================================================
# install-test-deps.sh — Install all dependencies needed to run torch-rbln tests
# =============================================================================
#
# Target users: External developers who want to run the full test suite without
# manually installing dependencies per test group.
#
# Assumptions:
#   - torch-rbln is already installed with --no-deps (no project deps pulled in).
#   - rebel-compiler is installed and usable (e.g. in PATH or configured).
#
# This script installs:
#   - Test runner: pytest, pytest-xdist (used by test/run_tests.py).
#   - Test infra: expecttest (used by torch.testing._internal when running
#     `test/ops/` / `test/rbln/`; required in some environments).
#   - Model tests: packages from `test/models/requirements.txt` (torchvision from
#     PyTorch CPU index, transformers from PyPI, optimum-rbln from rbln index plus
#     PyTorch CPU extra index so torchaudio==*+cpu resolves).
#
# For model-test packages, this script checks existing installations (version and
# for torchvision the install source). If already satisfied, it skips; if version
# or source is wrong, it prints guidance and exits.
#
# Usage:
#   ./tools/test/install-test-deps.sh [--dry-run]
#
# Optional environment:
#   PIP_EXTRA_INDEX_URL   Extra index URL (e.g. for private rbln index); used
#                         in addition to the script’s default for optimum-rbln.
#   UV=1                  Use `uv pip install` instead of `python -m pip install`.
# =============================================================================

set -e

readonly SCRIPT_DIR="$(realpath "$(dirname "$0")")"
readonly PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../..")"
readonly MODEL_REQUIREMENTS="$PROJECT_ROOT/test/models/requirements.txt"

DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "Unknown option: $arg" >&2; exit 1 ;;
  esac
done

run_install() {
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "[dry-run] $*"
    return 0
  fi
  if [ -n "${UV:-}" ] && [ "$UV" = "1" ]; then
    uv pip install "$@"
  else
    python -m pip install "$@"
  fi
}

echo "Installing test runner dependencies (pytest, pytest-xdist)..."
run_install pytest pytest-xdist

echo "Installing test infra dependencies (expecttest for torch.testing._internal)..."
run_install "expecttest>=0.3.0,<0.4.0"

if [ ! -f "$MODEL_REQUIREMENTS" ]; then
  echo "No model requirements file at $MODEL_REQUIREMENTS; skipping model-test deps."
  echo "Done."
  exit 0
fi

# Parse test/models/requirements.txt: verify or install with correct index.
# If already installed, verify version (and for torchvision that it is from
# PyTorch CPU index); skip if OK, else print guidance and exit or install if missing.
while IFS= read -r line || [ -n "$line" ]; do
  line="$(echo "$line" | sed 's/#.*$//' | xargs)"
  [ -z "$line" ] && continue

  if [[ "$line" =~ ^([^=]+)==(.+)$ ]]; then
    pkg="${BASH_REMATCH[1]}"
    ver="${BASH_REMATCH[2]}"
    spec="$pkg==$ver"

    current_ver=$(python -m pip show "$pkg" 2>/dev/null | grep "^Version:" | awk '{print $2}')
    if [ -n "$current_ver" ]; then
      # Already installed: verify version and (for torchvision) install source.
      if [ "$pkg" = "torchvision" ]; then
        norm_cur="${current_ver%+cpu}"
        norm_req="${ver%+cpu}"
        loc=$(python -m pip show "$pkg" 2>/dev/null | grep "^Location:" | awk '{print $2}')
        direct_url_file=""
        if [ -n "$loc" ] && [ -d "$loc/${pkg}-${current_ver}.dist-info" ]; then
          direct_url_file="$loc/${pkg}-${current_ver}.dist-info/direct_url.json"
        fi
        if [ -f "$direct_url_file" ]; then
          if grep -q "download.pytorch.org/whl/cpu" "$direct_url_file" 2>/dev/null; then
            if [ "$norm_cur" = "$norm_req" ]; then
              echo "✓ $spec already installed (from PyTorch CPU index)"
              continue
            fi
          else
            echo "ERROR: torchvision is not installed from PyTorch CPU index!" >&2
            echo "  pip install $spec --index-url https://download.pytorch.org/whl/cpu" >&2
            exit 1
          fi
        else
          if [ "$norm_cur" = "$norm_req" ] && [[ "$current_ver" == *"+cpu"* ]]; then
            echo "✓ $spec already installed (from CPU index)"
            continue
          fi
          if [ "$norm_cur" != "$norm_req" ]; then
            echo "ERROR: torchvision version mismatch. Installed: $current_ver, Required: $ver" >&2
            echo "  pip install $spec --index-url https://download.pytorch.org/whl/cpu" >&2
            exit 1
          fi
          echo "ERROR: torchvision installation source unknown; require CPU index." >&2
          echo "  pip install $spec --index-url https://download.pytorch.org/whl/cpu" >&2
          exit 1
        fi
        if [ "$current_ver" != "$ver" ]; then
          echo "ERROR: torchvision version mismatch. Installed: $current_ver, Required: $ver" >&2
          echo "  pip install $spec --index-url https://download.pytorch.org/whl/cpu" >&2
          exit 1
        fi
      fi

      if [ "$current_ver" = "$ver" ]; then
        echo "✓ $spec already installed"
        continue
      fi
      echo "ERROR: $pkg version mismatch. Installed: $current_ver, Required: $ver" >&2
      if [ "$pkg" = "torchvision" ]; then
        echo "  pip install $spec --index-url https://download.pytorch.org/whl/cpu" >&2
      elif [ "$pkg" = "optimum-rbln" ]; then
        echo "  pip install $spec --pre \\" >&2
        echo "    --extra-index-url https://pypi.rbln.ai/simple/ \\" >&2
        echo "    --extra-index-url https://download.pytorch.org/whl/cpu" >&2
      else
        echo "  pip install $spec" >&2
      fi
      exit 1
    fi

    # Not installed: install with correct index.
    if [ "$pkg" = "torchvision" ]; then
      echo "Installing $spec from PyTorch CPU index..."
      run_install "$spec" --index-url https://download.pytorch.org/whl/cpu
    elif [ "$pkg" = "optimum-rbln" ]; then
      echo "Installing $spec (rbln + PyPI + PyTorch CPU extra indexes)..."
      run_install "$spec" --pre \
        --extra-index-url https://pypi.rbln.ai/simple/ \
        --extra-index-url https://download.pytorch.org/whl/cpu
    else
      echo "Installing $spec..."
      run_install "$spec"
    fi
  fi
done < "$MODEL_REQUIREMENTS"

# -----------------------------------------------------------------------------
# vllm + vllm-rbln (for test/models/test_vllm_llm.py)
#
# The test file pytest.importorskip's vllm_rbln, so environments that do not
# install these deps skip cleanly; CI runs that want to exercise the vLLM model
# path on RBLN device tensors install both here.
#
# * vllm is pinned to a CPU wheel on wheels.vllm.ai (0.18.0+cpu today). The
#   version is controlled by the VLLM_VERSION env var.
# * vllm-rbln's device-tensor flow is gated behind the VLLM_RBLN_USE_DEVICE_TENSOR
#   env, introduced on the origin/device_tensor_rebased branch and not yet in a
#   PyPI release. We therefore source-install from GitHub instead of pinning a
#   version. The branch / repo / checkout path are configurable:
#     VLLM_RBLN_REF   default: origin/device_tensor_rebased
#     VLLM_RBLN_REPO  default: https://github.com/rbln-sw/vllm-rbln.git
#     VLLM_RBLN_DIR   default: $PROJECT_ROOT/vllm-rbln
#   Setting VLLM_RBLN_SKIP=1 disables the entire section for environments that
#   cannot or should not install vllm-rbln.
# -----------------------------------------------------------------------------

if [ "${VLLM_RBLN_SKIP:-0}" = "1" ]; then
  echo "VLLM_RBLN_SKIP=1 set — skipping vllm + vllm-rbln install."
  echo "All test dependencies installed."
  exit 0
fi

VLLM_VERSION="${VLLM_VERSION:-0.18.0+cpu}"
VLLM_MINOR="${VLLM_VERSION%+*}"
VLLM_WHEELS_INDEX="https://wheels.vllm.ai/${VLLM_MINOR}/cpu"
VLLM_RBLN_REPO="${VLLM_RBLN_REPO:-https://github.com/rbln-sw/vllm-rbln.git}"
VLLM_RBLN_REF="${VLLM_RBLN_REF:-origin/device_tensor_rebased}"
VLLM_RBLN_DIR="${VLLM_RBLN_DIR:-$PROJECT_ROOT/vllm-rbln}"

current_vllm=$(python -m pip show vllm 2>/dev/null | awk '/^Version:/{print $2}')
if [ -n "$current_vllm" ]; then
  if [ "$current_vllm" = "$VLLM_VERSION" ]; then
    echo "✓ vllm==$VLLM_VERSION already installed"
  else
    echo "ERROR: vllm version mismatch. Installed: $current_vllm, Required: $VLLM_VERSION" >&2
    echo "  pip install vllm==$VLLM_VERSION --extra-index-url $VLLM_WHEELS_INDEX --extra-index-url https://download.pytorch.org/whl/cpu" >&2
    exit 1
  fi
else
  echo "Installing vllm==$VLLM_VERSION from $VLLM_WHEELS_INDEX..."
  run_install "vllm==$VLLM_VERSION" \
    --extra-index-url "$VLLM_WHEELS_INDEX" \
    --extra-index-url https://download.pytorch.org/whl/cpu
fi

if [ ! -d "$VLLM_RBLN_DIR/.git" ]; then
  echo "Cloning vllm-rbln into $VLLM_RBLN_DIR..."
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "[dry-run] git clone $VLLM_RBLN_REPO $VLLM_RBLN_DIR"
  else
    git clone "$VLLM_RBLN_REPO" "$VLLM_RBLN_DIR"
  fi
fi

echo "Updating vllm-rbln to $VLLM_RBLN_REF in $VLLM_RBLN_DIR..."
if [ "$DRY_RUN" -eq 1 ]; then
  echo "[dry-run] (cd $VLLM_RBLN_DIR && git fetch origin && git checkout --detach $VLLM_RBLN_REF)"
else
  (cd "$VLLM_RBLN_DIR" && git fetch origin --prune && git checkout --detach "$VLLM_RBLN_REF")
fi

echo "Installing vllm-rbln (editable) from $VLLM_RBLN_DIR..."
run_install -e "$VLLM_RBLN_DIR" \
  --extra-index-url "$VLLM_WHEELS_INDEX" \
  --extra-index-url https://download.pytorch.org/whl/cpu

echo "All test dependencies installed."
