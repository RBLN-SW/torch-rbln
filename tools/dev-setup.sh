#!/bin/bash
# Quick development setup for common workflows

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

show_help() {
    cat << 'HELP'
Development Setup Helper for torch-rbln

USAGE:
    ./tools/dev-setup.sh <mode> [options]

MODES:
    pypi          Use PyPI rebel-compiler (default, fastest)
    external      Use external rebel-compiler from REBEL_HOME

PYPI MODE (Recommended for most development):
    ./tools/dev-setup.sh pypi [--clean]

    This will:
    - Install dependencies with uv (rebel-compiler from lock)
    - Install torch-rbln in editable mode

    Use --clean to remove build/ and do a fresh build (e.g. after changing C++ code).

EXTERNAL MODE (For rebel-compiler developers):
    export REBEL_HOME=/path/to/rebel_compiler
    ./tools/dev-setup.sh external [--clean]

    This will use build-with-external-rebel.sh

EXAMPLES:
    # Quick setup with PyPI (fastest)
    ./tools/dev-setup.sh pypi

    # PyPI with clean build (removes build/ and rebuilds)
    ./tools/dev-setup.sh pypi --clean

    # Use external rebel-compiler
    export REBEL_HOME=~/rebel_compiler
    ./tools/dev-setup.sh external --clean

HELP
}

check_rebel_index_access() {
    local rbln_ok=1  # assume fail until proven

    set +e
    pip index versions rebel-compiler --index-url https://pypi.rbln.ai/simple/ --timeout 5 < /dev/null &>/dev/null
    rbln_ok=$?
    set -e

    if [[ "${rbln_ok}" -eq 0 ]]; then
        return 0
    fi

    echo ""
    echo "❌ Cannot reach any rbln pypi index (no permission or network error)."
    echo "   Checked with uv (same auth as install):"
    echo "   - pypi.rbln.ai: no access"
    echo ""
    echo "   This usually means credentials for pypi.rbln.ai are missing."
    echo "   Add your RBLN Portal account to ~/.netrc:"
    echo ""
    echo "       machine pypi.rbln.ai"
    echo "       login <your-rbln-portal-id>"
    echo "       password <your-rbln-portal-password>"
    echo ""
    echo "   Then: chmod 600 ~/.netrc  and re-run this script."
    echo "   See README.md → 'Authenticate to the RBLN package index'."
    echo ""
    exit 1
}

mode_pypi() {
    local do_clean=""
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --clean) do_clean=1; shift ;;
            *)
                echo "❌ Unknown pypi option: $1"
                echo "   Usage: ./tools/dev-setup.sh pypi [--clean]"
                exit 1
                ;;
        esac
    done

    echo "📦 Setting up with PyPI rebel-compiler..."
    cd "${PROJECT_ROOT}"

    if [[ -n "${do_clean}" ]]; then
        echo "🧹 Cleaning build artifacts..."
        rm -rf build
        echo "   Removed build/"
    fi

    # If custom env vars are set, show them and prompt before continuing
    if [[ -n "${PYTHONPATH:-}" ]] || [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
        echo ""
        echo "⚠️  Custom environment variables are set (may affect install or runtime):"
        echo "────────────────────────────────────────────────────────────────────────"
        [[ -n "${PYTHONPATH:-}" ]]     && echo "  PYTHONPATH=${PYTHONPATH}"
        [[ -n "${LD_LIBRARY_PATH:-}" ]] && echo "  LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
        echo "────────────────────────────────────────────────────────────────────────"
        echo ""
        if [[ -t 0 ]]; then
            read -r -p "Continue with these env vars? [y/N] " answer
            case "${answer:-n}" in
                [yY]|[yY][eE][sS]) echo "Proceeding with current env." ;;
                *) echo "Stopped. Unset or adjust the variables and re-run if needed."; exit 1 ;;
            esac
        else
            echo "Non-interactive run: proceeding with current env (use a TTY to get y/N prompt)."
        fi
        echo ""
    fi

    if grep -q "rebel-compiler @ file://" pyproject.toml 2>/dev/null; then
        echo "❌ pyproject.toml points rebel-compiler at a local wheel, so the pinned build cannot be installed."
        echo "   Restore both files and re-run:"
        echo "     git restore --source=HEAD -- pyproject.toml uv.lock"
        exit 1
    fi

    echo "Running: uv sync --locked --no-install-project"
    local sync_output
    if sync_output=$(uv sync --locked --no-install-project 2>&1); then
        echo "${sync_output}"
    elif grep -qE "Unauthorized|403 Forbidden|could not be queried|lack of valid authentication" <<<"${sync_output}"; then
        echo ""
        echo "⚠️  Cannot authenticate to a package index. Installing the latest"
        echo "    rebel-compiler from pypi.rbln.ai instead of the locked build."
        echo "────────────────────────────────────────────────────────────────────────"
        echo "${sync_output}"
        echo "────────────────────────────────────────────────────────────────────────"
        echo ""

        check_rebel_index_access

        echo "Running: uv sync --locked --no-install-project --no-install-package rebel-compiler"
        uv sync --locked --no-install-project --no-install-package rebel-compiler

        # Without --no-config, uv pip install applies the constraint and requests the build that just failed.
        echo "Running: uv pip install --no-config --index rbln=https://pypi.rbln.ai/simple/ rebel-compiler"
        uv pip install --no-config --index rbln=https://pypi.rbln.ai/simple/ rebel-compiler
    else
        echo "${sync_output}" >&2
        exit 1
    fi

    echo "Running: uv pip install -e ."
    uv pip install -e . --no-build-isolation

    echo "✅ Setup complete!"
    echo ""
    echo "Verify installation:"
    echo "  python -c 'import torch_rbln; print(torch_rbln.__version__)'"
}

mode_external() {
    local clean_flag=()

    if [[ "$1" = "--clean" ]]; then
        clean_flag=(--clean)
    fi

    if [[ -z "${REBEL_HOME}" ]]; then
        echo "❌ REBEL_HOME is not set"
        echo ""
        echo "Usage:"
        echo "  export REBEL_HOME=/path/to/rebel_compiler"
        echo "  ./tools/dev-setup.sh external ${clean_flag[*]}"
        exit 1
    fi

    echo "🔗 Setting up with external rebel-compiler from REBEL_HOME..."
    echo "REBEL_HOME: ${REBEL_HOME}"

    cd "${PROJECT_ROOT}"
    ./tools/build-with-external-rebel.sh "${clean_flag[@]}"

    echo "✅ Setup complete with external rebel-compiler!"
}

# Main
cd "${PROJECT_ROOT}"

MODE="${1:-pypi}"
shift || true

# pypi and custom modes must not be run with REBEL_HOME set (use external mode instead)
if [[ -n "${REBEL_HOME:-}" ]]; then
    if [[ "${MODE}" = "pypi" ]] || [[ "${MODE}" = "custom" ]]; then
        echo "❌ REBEL_HOME is set (REBEL_HOME=${REBEL_HOME})"
        echo "   pypi and custom modes ignore REBEL_HOME and may cause confusion."
        echo ""
        echo "   To use the compiler at REBEL_HOME, run:"
        echo "     ./tools/dev-setup.sh external [--clean]"
        echo ""
        echo "   To run ${MODE} mode, unset REBEL_HOME first:"
        echo "     unset REBEL_HOME"
        printf '     ./tools/dev-setup.sh %s %s\n' "${MODE}" "$*"
        exit 1
    fi
fi

case "${MODE}" in
    pypi)
        mode_pypi "$@"
        ;;
    external)
        mode_external "$@"
        ;;
    -h|--help|help)
        show_help
        ;;
    *)
        echo "❌ Unknown mode: ${MODE}"
        echo ""
        show_help
        exit 1
        ;;
esac
