#!/bin/bash
#
# Build torch-rbln using an externally built rebel-compiler.
#
# This script is designed to be run from torch-rbln directory.
# REBEL_HOME must be set to point to an externally built rebel-compiler.
#
# Prerequisites:
#   - REBEL_HOME must be set to a built rebel-compiler directory
#   - rebel-compiler must be already built (rebel_install.sh completed)
#
# Usage:
#   # gcc-13 (default): Use pre-built PyTorch from pip
#   cd /path/to/torch-rbln
#   export REBEL_HOME=/path/to/rebel_compiler
#   ./tools/build-with-external-rebel.sh --clean
#
#   # gcc-12: Requires pre-built torch wheel (TORCH_WHEEL_PATH is mandatory)
#   cd /path/to/torch-rbln
#   export REBEL_HOME=/path/to/rebel_compiler
#   export RBLN_GCC_VERSION=12
#   export TORCH_WHEEL_PATH=/path/to/torch-2.10.0-cp310-cp310-linux_x86_64.whl
#   ./tools/build-with-external-rebel.sh --clean
#
# Arguments:
#   --clean                - Clean build artifacts before building
#   --clean-only           - Only clean build artifacts, do not build
#
# Environment Variables:
#   REBEL_HOME             - Path to rebel-compiler (REQUIRED)
#   TORCH_RBLN_HOME        - Path to torch-rbln (auto-detected from script location)
#   TORCH_RBLN_BUILD_TYPE  - Build type: Release (default) or Debug
#   RBLN_SKIP_VENV         - Set to 1 to skip virtual environment creation
#   RBLN_VENV_PATH         - Custom virtual environment path (default: .venv)
#   RBLN_GCC_VERSION       - GCC version to use: 12 or 13 (default: 13)
#                            - 13: Use pre-built PyTorch from pip (faster)
#                            - 12: MUST set TORCH_WHEEL_PATH (custom torch wheel)
#   TORCH_WHEEL_PATH       - Path to pre-built torch wheel file
#                            REQUIRED for gcc-12 mode, ignored for gcc-13
#                            The wheel must be built with gcc-12 and match Python version
#                            Example: /path/to/torch-2.10.0-cp310-cp310-linux_x86_64.whl
#

set -e

readonly build_type="${TORCH_RBLN_BUILD_TYPE:-Release}"
readonly skip_venv="${RBLN_SKIP_VENV:-0}"
readonly venv_path="${RBLN_VENV_PATH:-.venv}"
readonly gcc_version="${RBLN_GCC_VERSION:-13}"  # 12 or 13
readonly torch_wheel_path="${TORCH_WHEEL_PATH:-}"  # Optional: path to pre-built torch wheel

# Parse command line arguments
do_clean=0
clean_only=0
while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            do_clean=1
            shift
            ;;
        --clean-only)
            do_clean=1
            clean_only=1
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--clean] [--clean-only]"
            exit 1
            ;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

clean_build_artifacts() {
    log_info "Cleaning build artifacts..."

    # Build directories
    local dirs_to_remove=(
        "build"
        "dist"
        "torch_rbln/lib"
        "torch_rbln/include"
        "torch_rbln/out"
        "torch_rbln/test"
        "torch_rbln/bin"
        "torch_rbln.egg-info"
        ".eggs"
    )

    for dir in "${dirs_to_remove[@]}"; do
        if [ -d "$dir" ]; then
            log_info "  Removing directory: $dir"
            rm -rf "$dir"
        fi
    done

    # Generated files
    local files_to_remove=(
        "torch_rbln/_internal/register_ops.py"
        "torch_rbln/_C/__init__.pyi"
    )

    for file in "${files_to_remove[@]}"; do
        if [ -f "$file" ]; then
            log_info "  Removing file: $file"
            rm -f "$file"
        fi
    done

    # Shared library files (.so files in torch_rbln/)
    log_info "  Removing .so files in torch_rbln/..."
    find torch_rbln -maxdepth 1 -name "*.so*" -type f -exec rm -f {} \; 2>/dev/null || true

    # Python cache directories
    log_info "  Removing __pycache__ directories..."
    find . -type d -name "__pycache__" -exec rm -rf {} \; 2>/dev/null || true

    # Python bytecode files
    log_info "  Removing .pyc files..."
    find . -type f -name "*.pyc" -delete 2>/dev/null || true

    # Poetry/pip cache in project
    if [ -f "uv.lock" ] && [ -f "pyproject.toml.backup" ]; then
        log_info "  Restoring original pyproject.toml..."
        cp pyproject.toml.backup pyproject.toml
        rm -f pyproject.toml.backup
    fi

    log_info "Clean completed!"
}

detect_directories() {
    # REBEL_HOME is required for this script
    if [ -z "$REBEL_HOME" ]; then
        log_error "REBEL_HOME is not set."
        log_error "This script requires an externally built rebel-compiler."
        log_error ""
        log_error "Usage:"
        log_error "  export REBEL_HOME=/path/to/rebel_compiler"
        log_error "  ./tools/build-with-external-rebel.sh --clean"
        exit 1
    fi

    # Auto-detect TORCH_RBLN_HOME if not set
    if [ -z "$TORCH_RBLN_HOME" ]; then
        # Check if script is in torch-rbln/tools/
        local script_dir
        script_dir="$(cd "$(dirname "$0")" && pwd)"
        if [ -f "$script_dir/../pyproject.toml" ] && [ -d "$script_dir/../torch_rbln" ]; then
            export TORCH_RBLN_HOME="$(cd "$script_dir/.." && pwd)"
            log_info "Auto-detected TORCH_RBLN_HOME from script location: $TORCH_RBLN_HOME"
        # Check if torch-rbln is a sibling of REBEL_HOME
        elif [ -d "$REBEL_HOME/../torch-rbln" ]; then
            export TORCH_RBLN_HOME="$(cd "$REBEL_HOME/../torch-rbln" && pwd)"
            log_info "Auto-detected TORCH_RBLN_HOME as sibling: $TORCH_RBLN_HOME"
        # Check current directory
        elif [ -f "$PWD/pyproject.toml" ] && [ -d "$PWD/torch_rbln" ]; then
            export TORCH_RBLN_HOME="$PWD"
            log_info "Auto-detected TORCH_RBLN_HOME from current directory: $TORCH_RBLN_HOME"
        else
            log_error "TORCH_RBLN_HOME is not set and could not be auto-detected."
            log_error "Please set TORCH_RBLN_HOME to the torch-rbln repository root."
            log_error "Example: export TORCH_RBLN_HOME=/path/to/torch-rbln"
            exit 1
        fi
    fi
}

check_prerequisites() {
    # Auto-detect directories first
    detect_directories

    # Validate REBEL_HOME
    if [ ! -d "$REBEL_HOME" ]; then
        log_error "REBEL_HOME directory does not exist: $REBEL_HOME"
        exit 1
    fi

    # Check if rebel-compiler is built
    if [ ! -d "$REBEL_HOME/build" ]; then
        log_error "rebel-compiler build directory not found: $REBEL_HOME/build"
        log_error "Please build rebel-compiler first using rebel_install.sh"
        exit 1
    fi

    if [ ! -d "$REBEL_HOME/python/rebel" ]; then
        log_error "rebel-compiler Python package not found: $REBEL_HOME/python/rebel"
        exit 1
    fi

    # Check rebel-compiler Python version compatibility
    check_rebel_python_version

    # Validate TORCH_RBLN_HOME
    if [ ! -d "$TORCH_RBLN_HOME" ]; then
        log_error "TORCH_RBLN_HOME directory does not exist: $TORCH_RBLN_HOME"
        exit 1
    fi

    if [ ! -f "$TORCH_RBLN_HOME/pyproject.toml" ]; then
        log_error "pyproject.toml not found in TORCH_RBLN_HOME: $TORCH_RBLN_HOME"
        exit 1
    fi

    log_info "REBEL_HOME: $REBEL_HOME"
    log_info "TORCH_RBLN_HOME: $TORCH_RBLN_HOME"
    log_info "Build type: $build_type"
}

# Set up compiler (CC/CXX) only for gcc-13: Debian uses gcc-13/g++-13;
# RHEL/CentOS/Fedora use gcc-toolset-13. For gcc-12 we use system default (no setup).
setup_compiler_env() {
    if [ "$gcc_version" != "13" ]; then
        return 0
    fi
    if [ -f /etc/os-release ]; then
        # shellcheck disable=SC1091
        . /etc/os-release
        os_id_like="${ID_LIKE:-}"
    else
        os_id_like=""
    fi
    if [ "$os_id_like" = "debian" ]; then
        export CC=gcc-13
        export CXX=g++-13
        log_info "Using compiler: CC=$CC CXX=$CXX (debian)"
    elif [ -f /opt/rh/gcc-toolset-13/enable ]; then
        # shellcheck disable=SC1091
        . /opt/rh/gcc-toolset-13/enable
        # Use plain gcc/g++ so build_torch_rbln doesn't overwrite with gcc-13/g++-13
        # (toolset provides gcc/g++ in PATH, not gcc-13/g++-13)
        export CC=gcc
        export CXX=g++
        log_info "Using compiler: gcc-toolset-13 (RHEL/CentOS/Fedora), CC=$CC CXX=$CXX"
    else
        export CC=gcc-13
        export CXX=g++-13
        log_info "Using compiler: CC=$CC CXX=$CXX"
    fi
}

check_rebel_python_version() {
    # Get current Python version (e.g., "310" for Python 3.10)
    local current_py_version
    current_py_version=$(python -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')")

    log_info "Current Python version: $current_py_version (Python 3.${current_py_version:1})"

    # Look for rebel._C module matching current Python version
    local expected_pattern="_C.cpython-${current_py_version}-*.so"
    local rebel_c_so

    # scikit-build-core places the built extension under python/build/skbuild
    # (current rebel-compiler layout). Source-tree python/rebel/ only holds .py
    # files and _C/*.pyi stubs, so look in the build output first.
    rebel_c_so=$(find "$REBEL_HOME/python/build/skbuild" -maxdepth 1 -name "$expected_pattern" 2>/dev/null | head -1)

    # Legacy layouts (pre scikit-build-core migration) kept _C.so next to __init__.py.
    if [ -z "$rebel_c_so" ]; then
        rebel_c_so=$(find "$REBEL_HOME/python/rebel" -maxdepth 1 -name "$expected_pattern" 2>/dev/null | head -1)
    fi
    if [ -z "$rebel_c_so" ]; then
        rebel_c_so=$(find "$REBEL_HOME/rebel/python/rebel" -maxdepth 1 -name "$expected_pattern" 2>/dev/null | head -1)
    fi

    if [ -n "$rebel_c_so" ]; then
        log_info "Found matching rebel._C: $(basename "$rebel_c_so")"
        log_info "Python version check passed!"
        return 0
    fi

    # No matching version found - list available versions
    log_error "rebel-compiler was not built for Python $current_py_version!"
    log_error ""

    # Find all available versions
    local available_versions
    available_versions=$(find "$REBEL_HOME/python/build/skbuild" "$REBEL_HOME/python/rebel" "$REBEL_HOME/rebel/python/rebel" -maxdepth 1 -name "_C.cpython-*.so" 2>/dev/null | \
        xargs -I{} basename {} | grep -oP 'cpython-\K\d+' | sort -u)

    if [ -n "$available_versions" ]; then
        log_error "Available rebel-compiler versions:"
        for ver in $available_versions; do
            log_error "  - Python 3.${ver:1} (cpython-$ver)"
        done
        log_error ""
    fi

    log_error "Solutions:"
    log_error "  1. Rebuild rebel-compiler with Python 3.${current_py_version:1}:"
    log_error "     cd $REBEL_HOME"
    log_error "     python3.${current_py_version:1} -m venv .venv && source .venv/bin/activate"
    log_error "     pip install conan~=2.0.0 cmake~=3.18 lit"
    log_error "     ./rebel_install.sh"
    log_error ""
    if [ -n "$available_versions" ]; then
        local first_available
        first_available=$(echo "$available_versions" | head -1)
        log_error "  2. Or use an available Python version for torch-rbln:"
        log_error "     python3.${first_available:1} -m venv .venv"
        log_error "     source .venv/bin/activate"
        log_error "     ./tools/build-with-external-rebel.sh --clean"
    fi
    exit 1
}

setup_virtualenv() {
    if [ "$skip_venv" -eq 1 ]; then
        log_info "Skipping virtual environment creation (RBLN_SKIP_VENV=1)"
        return 0
    fi

    if [ -d "$venv_path" ]; then
        log_warn "Virtual environment already exists: $venv_path"
        log_info "Activating existing virtual environment..."
    else
        log_info "Creating virtual environment: $venv_path"
        python3 -m venv "$venv_path"
    fi

    # Source activate; the venv's activate may run '[ -f .use_external_rebel ] && source activate_rebel'
    # which returns 1 when the file is missing, causing set -e to exit. So temporarily allow non-zero.
    set +e
    # shellcheck disable=SC1091
    source "$venv_path/bin/activate"
    set -e
    if [ -z "${VIRTUAL_ENV:-}" ]; then
        log_error "Failed to activate virtual environment: $venv_path"
        exit 1
    fi
    log_info "Virtual environment activated: $VIRTUAL_ENV"
}

install_dependencies() {
    log_info "Installing build dependencies..."
    pip install --upgrade pip
    # Align with rebel-compiler Quick Install: cmake<4.0, lit.
    # rebel-compiler's own build-system.requires (scikit-build-core, pybind11,
    # cython, setuptools_scm) are installed later in install_rebel_python_deps,
    # *after* `uv sync --no-install-project` — otherwise uv sync would wipe them.
    pip install "cmake>=3.18,<4.0" ninja jinja2 hatchling setuptools-scm editables mypy lit

    # Native lib paths only. Python import is wired up later by the editable
    # rebel-compiler install (scikit-build-core), which drops its own redirect
    # .pth into site-packages — prepending $REBEL_HOME/python to PYTHONPATH here
    # would let the source-tree rebel/ shadow the editable install and break
    # `import rebel._C` under the new layout.
    export LD_LIBRARY_PATH="$REBEL_HOME/build:$LD_LIBRARY_PATH"

    # Source dynamic_linking.env for additional libraries
    if [ -f "$REBEL_HOME/dynamic_linking.env" ]; then
        source "$REBEL_HOME/dynamic_linking.env"
    fi

    # Create activate_rebel script
    create_activate_rebel_script

    # Remove the stale pre-scikit-build-core rebel_compiler.pth if an older run
    # of this script left one behind; it would also shadow the editable install.
    local site_packages
    site_packages=$(python -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || echo "")
    if [ -n "$site_packages" ] && [ -f "$site_packages/rebel_compiler.pth" ]; then
        log_info "  Removing stale rebel_compiler.pth from site-packages"
        rm -f "$site_packages/rebel_compiler.pth"
    fi

    log_info "Dependencies installed"
}

create_activate_rebel_script() {
    local activate_script="$TORCH_RBLN_HOME/$venv_path/bin/activate_rebel"
    local venv_root="$TORCH_RBLN_HOME/$venv_path"
    local flag_file="$venv_root/.use_external_rebel"

    # Extract conan lib path from dynamic_linking.env
    local conan_lib_path=""
    if [ -f "$REBEL_HOME/dynamic_linking.env" ]; then
        conan_lib_path=$(grep -oP 'LD_LIBRARY_PATH=\K[^:]+' "$REBEL_HOME/dynamic_linking.env" | head -1)
    fi

    cat > "$activate_script" << EOF
# Rebel-compiler environment setup (sourced only when .use_external_rebel exists in venv)
# To use PyPI package in this venv instead: remove .venv/.use_external_rebel and re-activate
# rebel-compiler is installed editable via scikit-build-core; Python imports are
# routed by _rebel_compiler_editable.pth in site-packages, so no PYTHONPATH here.
export REBEL_HOME="$REBEL_HOME"
export RBLN_USE_EXTERNAL_REBEL_COMPILER=1
export LD_LIBRARY_PATH="$REBEL_HOME/build:${conan_lib_path}:\$LD_LIBRARY_PATH"
EOF
    chmod +x "$activate_script"

    # Mark this venv as "external rebel"; activate will only source activate_rebel when this exists
    touch "$flag_file"

    # Add conditional auto-source to main activate script
    local activate_file="$venv_root/bin/activate"
    if ! grep -q "activate_rebel" "$activate_file" 2>/dev/null; then
        echo '' >> "$activate_file"
        echo '# Auto-source rebel environment only when .use_external_rebel exists' >> "$activate_file"
        echo '# To use PyPI package: rm .venv/.use_external_rebel and re-activate' >> "$activate_file"
        echo '[ -f "${VIRTUAL_ENV}/.use_external_rebel" ] && source "${VIRTUAL_ENV}/bin/activate_rebel"' >> "$activate_file"
    else
        # Migrate old unconditional source to conditional (so switching to PyPI is possible)
        if ! grep -q '\.use_external_rebel' "$activate_file" 2>/dev/null; then
            sed -i 's|^source "\${VIRTUAL_ENV}/bin/activate_rebel"$|[ -f "${VIRTUAL_ENV}/.use_external_rebel" ] \&\& source "${VIRTUAL_ENV}/bin/activate_rebel"|' "$activate_file" 2>/dev/null || true
            touch "$flag_file"
        fi
    fi

    log_info "Created activate_rebel script (conditional on .use_external_rebel; remove that file to use PyPI package)"
    log_info "To use PyPI rebel in this venv later: rm $venv_path/.use_external_rebel then deactivate and source $venv_path/bin/activate again."
}

modify_pyproject() {
    log_info "Modifying pyproject.toml..."

    # Determine torch dependency based on gcc version
    local torch_dep
    if [ "$gcc_version" = "12" ] && [ -n "$effective_torch_wheel_path" ]; then
        # gcc-12: use wheel file path
        torch_dep="torch @ file://$effective_torch_wheel_path"
        log_info "  Setting torch dependency to wheel: $effective_torch_wheel_path"
    else
        # gcc-13: use PyPI version
        torch_dep="torch==2.10.0+cpu"
        log_info "  Setting torch dependency to PyPI: torch==2.10.0+cpu"
    fi

    python3 << EOF
import re

torch_dep = "$torch_dep"

with open("pyproject.toml", "r", encoding="utf-8") as f:
    content = f.read()

# Remove rebel-compiler dependencies
content = re.sub(
    r'^\s*"rebel-compiler[^"]*",?\s*\n',
    '',
    content,
    flags=re.MULTILINE
)

# Replace torch dependency in [project].dependencies
# Match various formats:
#   "torch @ file://..."
#   "torch==2.10.0+cpu"
#   "torch (==2.10.0+cpu)"  <- parentheses format
#   "torch (>=2.10.0)"
content = re.sub(
    r'^\s*"torch\s*[\(@][^"]*",?\s*\n',
    f'  "{torch_dep}",\n',
    content,
    flags=re.MULTILINE
)

# Replace torch dependency in [build-system].requires (no trailing comma)
content = re.sub(
    r'^\s*"torch\s*[\(@][^"]*"\s*\n',
    f'  "{torch_dep}"\n',
    content,
    flags=re.MULTILINE
)

# Clean up any trailing commas before closing brackets
content = re.sub(r',(\s*\])', r'\1', content)

with open("pyproject.toml", "w", encoding="utf-8") as f:
    f.write(content)

print("pyproject.toml modified successfully")
print(f"  - rebel-compiler dependency removed")
print(f"  - torch dependency set to: {torch_dep}")
EOF
}

configure_uv() {
    log_info "Configuring uv..."

    # Disable keyring to avoid interactive prompts
    # uv uses keyring by default or env vars for auth

    # Check if LDAP credentials are set
    if [ -n "$LDAP_USERNAME" ] && [ -n "$LDAP_PASSWORD" ]; then
        log_info "Configuring uv with LDAP credentials..."
        export UV_INDEX_RBLN_INTERNAL_USERNAME="$LDAP_USERNAME"
        export UV_INDEX_RBLN_INTERNAL_PASSWORD="$LDAP_PASSWORD"
    else
        log_warn "LDAP credentials not set. Set UV_INDEX_RBLN_INTERNAL_USERNAME and UV_INDEX_RBLN_INTERNAL_PASSWORD."
        log_warn "Set LDAP_USERNAME and LDAP_PASSWORD if needed."
    fi
}

validate_torch_wheel() {
    local wheel_path="$1"

    if [ ! -f "$wheel_path" ]; then
        log_error "Torch wheel file not found: $wheel_path"
        exit 1
    fi

    log_info "Validated torch wheel: $wheel_path"
}

# Install rebel-compiler into the venv (scikit-build-core editable). Runs
# after `uv sync --no-install-project` so the install isn't wiped by sync.
#
# Writes _rebel_compiler_editable.pth + the redirect hook to site-packages so
# that `import rebel` resolves to $REBEL_HOME/python/rebel/ while
# `import rebel._C` finds the extension under $REBEL_HOME/python/build/skbuild/.
# With [tool.scikit-build.editable] rebuild=false, the pre-built extension is
# reused rather than recompiled. Runtime deps come from the package's
# pyproject.toml.
#
# --no-build-isolation keeps the editable install deterministic: it reuses the
# scikit-build-core / pybind11 / cython / cmake / ninja pinned in
# install_dependencies instead of fetching a separate PEP 517 build env from
# PyPI each invocation.
install_rebel_python_deps() {
    if [ ! -f "$REBEL_HOME/python/pyproject.toml" ]; then
        log_error "REBEL_HOME/python/pyproject.toml not found — cannot install rebel-compiler."
        log_error "Expected path: $REBEL_HOME/python/pyproject.toml"
        return 1
    fi

    # Fail loudly if the REBEL_HOME checkout isn't scikit-build-core (pre-migration
    # trees built rebel-compiler via a different backend; this script no longer
    # supports them).
    if ! grep -q 'build-backend\s*=\s*"scikit_build_core.build"' "$REBEL_HOME/python/pyproject.toml"; then
        log_error "REBEL_HOME/python/pyproject.toml does not use scikit_build_core.build."
        log_error "This script targets the post-migration rebel-compiler layout; please upgrade"
        log_error "REBEL_HOME to a scikit-build-core build of rebel-compiler."
        return 1
    fi

    # Install rebel-compiler's own build-system.requires into the active venv so
    # `uv pip install --no-build-isolation -e` can reuse them without creating a
    # PEP 517 isolated build env (and its associated PyPI fetches). These pins
    # track rebel-compiler/python/pyproject.toml's [build-system].requires.
    log_info "Installing rebel-compiler build dependencies (scikit-build-core, pybind11, cython)..."
    uv pip install \
        "scikit-build-core>=0.10" \
        "setuptools_scm>=8" \
        "pybind11>=2.10,<=2.10.4" \
        cython \
        "cmake>=3.26,<4.0" \
        ninja || return $?

    log_info "Installing rebel-compiler (editable, --no-build-isolation) from $REBEL_HOME/python..."
    uv pip install --no-build-isolation -e "$REBEL_HOME/python" || return $?

    log_info "rebel-compiler installed"
}

build_torch_rbln() {
    log_info "Building torch-rbln..."

    if [ "$gcc_version" = "13" ] && { [ -z "${CC:-}" ] || [ -z "${CXX:-}" ]; }; then
        log_error "CC and CXX must be set by setup_compiler_env (gcc_version=13)."
        exit 1
    fi
    export REBEL_HOME="$REBEL_HOME"
    export RBLN_USE_EXTERNAL_REBEL_COMPILER=1
    # No PYTHONPATH shim: install_rebel_python_deps (called later in this function)
    # does a scikit-build-core editable install of rebel-compiler, and
    # FindRebel.cmake reads REBEL_HOME directly for headers/libs.

    # Install torch based on gcc version
    if [ "$gcc_version" = "12" ] && [ -n "$effective_torch_wheel_path" ]; then
        # gcc-12: Install from wheel file
        log_info "Installing PyTorch from wheel: $effective_torch_wheel_path"
        pip uninstall -y torch 2>/dev/null || true
        pip install "$effective_torch_wheel_path"
    else
        # gcc-13: Install from PyPI
        local current_torch_version
        current_torch_version=$(pip show torch 2>/dev/null | grep "^Version:" | awk '{print $2}' || echo "")

        if [ "$current_torch_version" != "2.10.0+cpu" ]; then
            log_info "Installing PyTorch 2.10.0+cpu from PyPI..."
            pip uninstall -y torch 2>/dev/null || true
            pip install torch==2.10.0+cpu --index-url https://download.pytorch.org/whl/cpu
        fi
    fi

    # Save torch version before uv sync
    local torch_before
    torch_before=$(pip show torch 2>/dev/null | grep "^Version:" | awk '{print $2}' || echo "")
    log_info "Torch version before uv sync: $torch_before"

    # Update uv.lock and install dependencies (rebel Python deps installed after sync)
    log_info "Updating uv.lock..."
    uv lock

    log_info "Installing dependencies..."
    uv sync --no-install-project || true

    # Check if torch was overwritten by uv sync
    local torch_after
    torch_after=$(pip show torch 2>/dev/null | grep "^Version:" | awk '{print $2}' || echo "")

    if [ "$torch_before" != "$torch_after" ]; then
        log_warn "Torch was changed by uv sync: $torch_before -> $torch_after"
        # Reinstall correct torch
        if [ "$gcc_version" = "12" ] && [ -n "$effective_torch_wheel_path" ]; then
            log_info "Reinstalling torch from wheel..."
            pip uninstall -y torch 2>/dev/null || true
            pip install "$effective_torch_wheel_path"
        else
            log_info "Reinstalling torch from PyPI..."
            pip uninstall -y torch 2>/dev/null || true
            pip install torch==2.10.0+cpu --index-url https://download.pytorch.org/whl/cpu
        fi
    fi

    local torch_final
    torch_final=$(pip show torch 2>/dev/null | grep "^Version:" | awk '{print $2}' || echo "")
    log_info "Final torch version: $torch_final"

    # Install rebel-compiler Python deps after uv sync so they are not overwritten (matches apply-custom-rebel.sh / Quick Install)
    install_rebel_python_deps || return $?

    # Build and install torch-rbln
    log_info "Building torch-rbln with gcc-$gcc_version..."
    CC=$CC CXX=$CXX TORCH_RBLN_BUILD_TYPE="$build_type" uv pip install -e . --no-build-isolation

    log_info "torch-rbln installed successfully!"
}

verify_installation() {
    log_info "Verifying installation..."

    # Check library files
    local libs_ok=1
    [ -f "torch_rbln/lib/libtorch_rbln.so" ] || libs_ok=0
    [ -f "torch_rbln/lib/libc10_rbln.so" ] || libs_ok=0

    if [ $libs_ok -eq 0 ]; then
        log_warn "Some library files not found"
    fi

    # Re-source activate_rebel to ensure LD_LIBRARY_PATH is set correctly
    local activate_rebel_script="$TORCH_RBLN_HOME/$venv_path/bin/activate_rebel"
    if [ -f "$activate_rebel_script" ]; then
        source "$activate_rebel_script"
    fi

    # Test imports
    log_info "Testing imports..."

    local import_result=0
    python -c "
import torch
print(f'  torch: {torch.__version__}')
import rebel
print(f'  rebel: {rebel.__version__}')
import torch_rbln
print(f'  torch_rbln: {torch_rbln.__version__}')
" || import_result=$?

    if [ "$import_result" -ne 0 ]; then
        echo ""
        log_error "=========================================="
        log_error "Import test FAILED! (exit code: $import_result)"
        log_error "=========================================="
        log_error ""
        log_error "Possible causes:"
        log_error "  - Segmentation fault (library version mismatch)"
        log_error "  - LD_LIBRARY_PATH not set correctly"
        log_error "  - rebel-compiler editable install missing or shadowed"
        log_error "    (check \$VIRTUAL_ENV/lib/python*/site-packages/_rebel_compiler_editable.pth;"
        log_error "     re-run: uv pip install --no-build-isolation -e \$REBEL_HOME/python)"
        log_error "  - Python version mismatch with rebel-compiler"
        log_error "  - GCC version mismatch between torch and torch-rbln"
        log_error ""
        log_error "Try manually:"
        log_error "  source $TORCH_RBLN_HOME/$venv_path/bin/activate"
        log_error "  python -c 'import torch; import rebel; import torch_rbln'"
        exit 1
    fi

    echo ""
    log_info "=========================================="
    log_info "Import test PASSED!"
    log_info "=========================================="
}

print_summary() {
    echo ""
    echo -e "${GREEN}=========================================="
    echo "Build completed successfully!"
    echo -e "==========================================${NC}"
    echo ""
    echo "Configuration:"
    echo "  REBEL_HOME:      $REBEL_HOME"
    echo "  TORCH_RBLN_HOME: $TORCH_RBLN_HOME"
    echo "  GCC version:     $gcc_version"
    echo "  Build type:      $build_type"
    if [ "$gcc_version" = "12" ]; then
        echo "  PyTorch:         from wheel ($effective_torch_wheel_path)"
    else
        echo "  PyTorch:         2.10.0+cpu (PyPI)"
    fi
    echo ""
    echo -e "${GREEN}How to use:${NC}"
    echo ""
    echo "  1. Activate the virtual environment:"
    echo "     cd $TORCH_RBLN_HOME"
    echo "     source $venv_path/bin/activate"
    echo ""
    echo "  2. The activate_rebel script is auto-sourced, setting:"
    echo "     - REBEL_HOME"
    echo "     - RBLN_USE_EXTERNAL_REBEL_COMPILER=1"
    echo "     - LD_LIBRARY_PATH (includes native libraries)"
    echo "     (rebel-compiler is editable-installed via scikit-build-core;"
    echo "      Python imports are routed by _rebel_compiler_editable.pth)"
    echo ""
    echo "  3. Example usage:"
    echo "     python -c 'import torch; import rebel; import torch_rbln; print(\"OK\")'"
    echo ""
}

main() {
    log_info "Starting build..."

    # Source .bashrc when present (e.g. CI) so PATH and tooling are consistent
    if [ -f "$HOME/.bashrc" ]; then
        # shellcheck disable=SC1090
        . "$HOME/.bashrc"
    fi

    # Set up compiler for gcc-13 (OS-aware: debian vs RHEL gcc-toolset-13)
    setup_compiler_env

    # Determine effective torch_wheel_path (global for modify_pyproject)
    effective_torch_wheel_path="$torch_wheel_path"

    # Validate gcc version and torch wheel configuration
    if [ "$gcc_version" = "12" ]; then
        if [ -z "$torch_wheel_path" ]; then
            log_error "gcc-12 mode requires TORCH_WHEEL_PATH"
            log_error ""
            log_error "Usage:"
            log_error "  export RBLN_GCC_VERSION=12"
            log_error "  export TORCH_WHEEL_PATH=/path/to/torch-2.10.0-cpXXX-cpXXX-linux_x86_64.whl"
            log_error "  ./tools/build-with-external-rebel.sh --clean"
            exit 1
        fi
        log_info "Mode: gcc-12 + torch wheel"
    else
        if [ -n "$torch_wheel_path" ]; then
            log_warn "TORCH_WHEEL_PATH ignored in gcc-13 mode (use RBLN_GCC_VERSION=12 to use wheel)"
            effective_torch_wheel_path=""
        fi
        log_info "Mode: gcc-13 + PyPI torch"
    fi

    # Check prerequisites
    check_prerequisites
    cd "$TORCH_RBLN_HOME" || exit 1

    # Handle clean options
    if [ "$do_clean" -eq 1 ]; then
        clean_build_artifacts
        [ "$clean_only" -eq 1 ] && exit 0
    fi

    # Validate torch wheel if specified
    [ -n "$effective_torch_wheel_path" ] && validate_torch_wheel "$effective_torch_wheel_path"

    # Build steps
    setup_virtualenv
    install_dependencies
    modify_pyproject
    configure_uv
    build_torch_rbln
    verify_installation
    print_summary
}

main
