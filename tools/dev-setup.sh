#!/bin/bash
# Quick development setup for common workflows

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

show_help() {
    cat << 'HELP'
Development Setup Helper for torch-rbln

USAGE:
    ./tools/dev-setup.sh <mode> [options]

MODES:
    pypi          Use PyPI rebel-compiler (default, fastest)
    external      Use external rebel-compiler from REBEL_HOME

PYPI MODE (Recommended for most development):
    ./tools/dev-setup.sh pypi [--clean] [--extra-index-url <url>]

    This will:
    - Install dependencies with uv (rebel-compiler from lock)
    - Install torch-rbln in editable mode

    Use --clean to remove build/ and do a fresh build (e.g. after changing C++ code).

    If constraints-build-dev.txt exists, rebel-compiler is installed with that
    constraints file plus an extra index. You can provide the index in any of these
    ways (first match wins where applicable):

    - uv built-ins: export UV_INDEX (space-separated URLs) or UV_EXTRA_INDEX_URL;
      see https://docs.astral.sh/uv/reference/environment/
    - This script: --extra-index-url <url> — passed to uv for the constraints-based
      rebel-compiler install only (order with --clean is arbitrary).
    - Interactive: if none of the above apply (TTY), you will be prompted; leave
      empty to use only pypi.rbln.ai.

    pip's PIP_EXTRA_INDEX_URL is not read by uv; use UV_INDEX / UV_EXTRA_INDEX_URL.

EXTERNAL MODE (For rebel-compiler developers):
    export REBEL_HOME=/path/to/rebel_compiler
    ./tools/dev-setup.sh external [--clean]

    This will use build-with-external-rebel.sh

EXAMPLES:
    # Quick setup with PyPI (fastest)
    ./tools/dev-setup.sh pypi

    # PyPI with clean build (removes build/ and rebuilds)
    ./tools/dev-setup.sh pypi --clean

    # PyPI with a private extra index (e.g. for constraints-build-dev.txt)
    ./tools/dev-setup.sh pypi --extra-index-url 'https://example.com/simple/'

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

    if [ "$rbln_ok" -eq 0 ]; then
        return 0
    fi

    echo ""
    echo "❌ Cannot reach any rbln pypi index (no permission or network error)."
    echo "   Checked with uv (same auth as install):"
    echo "   - pypi.rbln.ai: no access"
    echo ""
    exit 1
}

mode_pypi() {
    local do_clean=""
    local arg_extra_index_url=""
    while [ $# -gt 0 ]; do
        case "$1" in
            --clean) do_clean=1; shift ;;
            --extra-index-url)
                if [ -z "${2:-}" ]; then
                    echo "❌ --extra-index-url requires a URL"
                    exit 1
                fi
                arg_extra_index_url="$2"
                shift 2
                ;;
            *)
                echo "❌ Unknown pypi option: $1"
                echo "   Usage: ./tools/dev-setup.sh pypi [--clean] [--extra-index-url <url>]"
                exit 1
                ;;
        esac
    done

    echo "📦 Setting up with PyPI rebel-compiler..."
    cd "$PROJECT_ROOT"

    if [ -n "$do_clean" ]; then
        echo "🧹 Cleaning build artifacts..."
        rm -rf build
        echo "   Removed build/"
    fi

    # If custom env vars are set, show them and prompt before continuing
    if [ -n "${PYTHONPATH:-}" ] || [ -n "${LD_LIBRARY_PATH:-}" ]; then
        echo ""
        echo "⚠️  Custom environment variables are set (may affect install or runtime):"
        echo "────────────────────────────────────────────────────────────────────────"
        [ -n "${PYTHONPATH:-}" ]     && echo "  PYTHONPATH=${PYTHONPATH}"
        [ -n "${LD_LIBRARY_PATH:-}" ] && echo "  LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
        echo "────────────────────────────────────────────────────────────────────────"
        echo ""
        if [ -t 0 ]; then
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

    # Ensure pyproject.toml uses PyPI rebel-compiler (not file:// custom wheel)
    if grep -q "rebel-compiler @ file://" pyproject.toml 2>/dev/null; then
        echo "⚠️  Detected custom rebel-compiler in pyproject.toml"
        echo "   Restoring PyPI version..."
        python ./tools/replace_depends.py \
            --pyproject-path ./pyproject.toml \
            --source "rebel-compiler" \
            --target "rebel-compiler>=0.10.1,<0.20.0" \
            --set-index rbln
        rm -f uv.lock
    fi

    check_rebel_index_access

    echo "Running: uv lock"
    uv lock

    echo "Running: uv sync --no-install-project"
    uv sync --no-install-project

    # Install rebel-compiler: optional extra index + constraints-build-dev.txt; on 401/auth failure fall back to pypi.rbln.ai
    # Extra index sources: --extra-index-url, or uv's UV_INDEX / UV_EXTRA_INDEX_URL (see uv env docs), or prompt (TTY).
    if [ -f constraints-build-dev.txt ]; then
        rebel_extra_index="${arg_extra_index_url}"
        uv_extra_index_env=0
        if [ -z "$rebel_extra_index" ] && { [ -n "${UV_INDEX:-}" ] || [ -n "${UV_EXTRA_INDEX_URL:-}" ]; }; then
            uv_extra_index_env=1
        fi
        if [ -z "$rebel_extra_index" ] && [ "$uv_extra_index_env" -eq 0 ] && [ -t 0 ]; then
            echo ""
            read -r -p "Extra PyPI index URL for rebel-compiler (constraints-build-dev; empty = pypi.rbln.ai only): " rebel_extra_index
            echo ""
        fi

        if [ -n "$rebel_extra_index" ]; then
            echo "Running: uv pip install -c constraints-build-dev.txt rebel-compiler (--extra-index-url / prompt)"
            set +e
            REBEL_OUTPUT=$(uv pip install --extra-index-url "$rebel_extra_index" \
                -c constraints-build-dev.txt rebel-compiler 2>&1)
            REBEL_EXIT=$?
            set -e

            if [ "$REBEL_EXIT" -eq 0 ]; then
                echo "$REBEL_OUTPUT"
            else
                if echo "$REBEL_OUTPUT" | grep -qE "401|Unauthorized|could not be queried|lack of valid authentication"; then
                    echo ""
                    echo "⚠️  Extra index failed due to missing or invalid credentials (e.g. 401 Unauthorized)."
                    echo "    Your environment does not have access to that index."
                    echo "    Falling back to pypi.rbln.ai (rebel-compiler version from lock)."
                    echo ""
                    echo "--- uv output (extra index attempt) ---"
                    echo "$REBEL_OUTPUT"
                    echo "--- end uv output ---"
                    echo ""
                    echo "Running: uv pip install rebel-compiler (pypi.rbln.ai)"
                    uv pip install --extra-index-url https://pypi.rbln.ai/simple/ rebel-compiler
                else
                    echo "$REBEL_OUTPUT" >&2
                    exit 1
                fi
            fi
        elif [ "$uv_extra_index_env" -eq 1 ]; then
            echo "Running: uv pip install -c constraints-build-dev.txt rebel-compiler (UV_INDEX / UV_EXTRA_INDEX_URL)"
            set +e
            REBEL_OUTPUT=$(uv pip install -c constraints-build-dev.txt rebel-compiler 2>&1)
            REBEL_EXIT=$?
            set -e

            if [ "$REBEL_EXIT" -eq 0 ]; then
                echo "$REBEL_OUTPUT"
            else
                if echo "$REBEL_OUTPUT" | grep -qE "401|Unauthorized|could not be queried|lack of valid authentication"; then
                    echo ""
                    echo "⚠️  Extra index failed due to missing or invalid credentials (e.g. 401 Unauthorized)."
                    echo "    Falling back to pypi.rbln.ai (rebel-compiler version from lock)."
                    echo ""
                    echo "--- uv output (extra index attempt) ---"
                    echo "$REBEL_OUTPUT"
                    echo "--- end uv output ---"
                    echo ""
                    echo "Running: uv pip install rebel-compiler (pypi.rbln.ai)"
                    uv pip install --extra-index-url https://pypi.rbln.ai/simple/ rebel-compiler
                else
                    echo "$REBEL_OUTPUT" >&2
                    exit 1
                fi
            fi
        else
            echo "Running: uv pip install rebel-compiler (pypi.rbln.ai) — no extra index (constraints-build-dev not used for pip install)"
            uv pip install --extra-index-url https://pypi.rbln.ai/simple/ rebel-compiler
        fi
    else
        echo "Running: uv pip install rebel-compiler (pypi.rbln.ai)"
        uv pip install --extra-index-url https://pypi.rbln.ai/simple/ rebel-compiler
    fi

    echo "Running: uv pip install -e ."
    uv pip install -e . --no-build-isolation

    echo "✅ Setup complete!"
    echo ""
    echo "Verify installation:"
    echo "  python -c 'import torch_rbln; print(torch_rbln.__version__)'"
}

mode_external() {
    local clean_flag=""

    if [ "$1" = "--clean" ]; then
        clean_flag="--clean"
    fi

    if [ -z "$REBEL_HOME" ]; then
        echo "❌ REBEL_HOME is not set"
        echo ""
        echo "Usage:"
        echo "  export REBEL_HOME=/path/to/rebel_compiler"
        echo "  ./tools/dev-setup.sh external $clean_flag"
        exit 1
    fi

    echo "🔗 Setting up with external rebel-compiler from REBEL_HOME..."
    echo "REBEL_HOME: $REBEL_HOME"

    cd "$PROJECT_ROOT"
    ./tools/build-with-external-rebel.sh $clean_flag

    echo "✅ Setup complete with external rebel-compiler!"
}

# Main
cd "$PROJECT_ROOT"

MODE="${1:-pypi}"
shift || true

# pypi and custom modes must not be run with REBEL_HOME set (use external mode instead)
if [ -n "${REBEL_HOME:-}" ]; then
    if [ "$MODE" = "pypi" ] || [ "$MODE" = "custom" ]; then
        echo "❌ REBEL_HOME is set (REBEL_HOME=$REBEL_HOME)"
        echo "   pypi and custom modes ignore REBEL_HOME and may cause confusion."
        echo ""
        echo "   To use the compiler at REBEL_HOME, run:"
        echo "     ./tools/dev-setup.sh external [--clean]"
        echo ""
        echo "   To run $MODE mode, unset REBEL_HOME first:"
        echo "     unset REBEL_HOME"
        printf '     ./tools/dev-setup.sh %s %s\n' "$MODE" "$*"
        exit 1
    fi
fi

case "$MODE" in
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
        echo "❌ Unknown mode: $MODE"
        echo ""
        show_help
        exit 1
        ;;
esac
