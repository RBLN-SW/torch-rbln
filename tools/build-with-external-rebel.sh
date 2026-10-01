#!/bin/bash
#
# Build torch-rbln against the rbln runtime of a rebel-compiler tree.
#
# This script is designed to be run from torch-rbln directory.
# torch-rbln compiles against the runtime headers in $REBEL_HOME/rbln/include, links
# $REBEL_HOME/build/rbln/librbln_rt.so, and at run time uses the rbln package in
# $REBEL_HOME/rbln/python, which maps that library.
#
# Prerequisites:
#   - REBEL_HOME must be set to a rebel-compiler tree built with rbln
#     (rebel_install.sh completed, build/rbln/librbln_rt.so present)
#
# Usage:
#   cd /path/to/torch-rbln
#   export REBEL_HOME=/path/to/rebel_compiler
#   ./tools/build-with-external-rebel.sh --clean
#
# Arguments:
#   --clean                - Clean build artifacts before building
#   --clean-only           - Only clean build artifacts, do not build
#
# Environment Variables:
#   REBEL_HOME             - Path to the rebel-compiler tree (REQUIRED)
#   TORCH_RBLN_HOME        - Path to torch-rbln (auto-detected from script location)
#   TORCH_RBLN_BUILD_TYPE  - Build type: Release (default) or Debug
#   RBLN_SKIP_VENV         - Set to 1 to skip virtual environment creation
#   RBLN_VENV_PATH         - Custom virtual environment path (default: .venv)
#
# The build uses GCC 13 (gcc-13/g++-13 on Debian/Ubuntu, gcc-toolset-13 on
# RHEL/CentOS/Fedora) and installs PyTorch from the PyPI CPU index.
#

set -e

readonly build_type="${TORCH_RBLN_BUILD_TYPE:-Release}"
readonly skip_venv="${RBLN_SKIP_VENV:-0}"
readonly venv_path="${RBLN_VENV_PATH:-.venv}"

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
        if [[ -d "${dir}" ]]; then
            log_info "  Removing directory: ${dir}"
            rm -rf "${dir}"
        fi
    done

    # Generated files
    local files_to_remove=(
        "torch_rbln/_internal/register_ops.py"
        "torch_rbln/_C/__init__.pyi"
    )

    for file in "${files_to_remove[@]}"; do
        if [[ -f "${file}" ]]; then
            log_info "  Removing file: ${file}"
            rm -f "${file}"
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
    if [[ -f "uv.lock" ]] && [[ -f "pyproject.toml.backup" ]]; then
        log_info "  Restoring original pyproject.toml..."
        cp pyproject.toml.backup pyproject.toml
        rm -f pyproject.toml.backup
    fi

    log_info "Clean completed!"
}

detect_directories() {
    # REBEL_HOME is required for this script
    if [[ -z "${REBEL_HOME}" ]]; then
        log_error "REBEL_HOME is not set."
        log_error "This script requires an externally built rebel-compiler."
        log_error ""
        log_error "Usage:"
        log_error "  export REBEL_HOME=/path/to/rebel_compiler"
        log_error "  ./tools/build-with-external-rebel.sh --clean"
        exit 1
    fi

    # Auto-detect TORCH_RBLN_HOME if not set
    if [[ -z "${TORCH_RBLN_HOME}" ]]; then
        # Check if script is in torch-rbln/tools/
        local script_dir
        script_dir="$(cd "$(dirname "$0")" && pwd)"
        if [[ -f "${script_dir}/../pyproject.toml" ]] && [[ -d "${script_dir}/../torch_rbln" ]]; then
            TORCH_RBLN_HOME="$(cd "${script_dir}/.." && pwd)"
            export TORCH_RBLN_HOME
            log_info "Auto-detected TORCH_RBLN_HOME from script location: ${TORCH_RBLN_HOME}"
        # Check if torch-rbln is a sibling of REBEL_HOME
        elif [[ -d "${REBEL_HOME}/../torch-rbln" ]]; then
            TORCH_RBLN_HOME="$(cd "${REBEL_HOME}/../torch-rbln" && pwd)"
            export TORCH_RBLN_HOME
            log_info "Auto-detected TORCH_RBLN_HOME as sibling: ${TORCH_RBLN_HOME}"
        # Check current directory
        elif [[ -f "${PWD}/pyproject.toml" ]] && [[ -d "${PWD}/torch_rbln" ]]; then
            export TORCH_RBLN_HOME="${PWD}"
            log_info "Auto-detected TORCH_RBLN_HOME from current directory: ${TORCH_RBLN_HOME}"
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
    if [[ ! -d "${REBEL_HOME}" ]]; then
        log_error "REBEL_HOME directory does not exist: ${REBEL_HOME}"
        exit 1
    fi
    # Recorded in rbln_home.pth and activate_rebel, which must not depend on the current directory.
    REBEL_HOME="$(realpath "${REBEL_HOME}")"
    export REBEL_HOME

    # Check that the rbln runtime is built
    if [[ ! -f "${REBEL_HOME}/build/rbln/librbln_rt.so" ]]; then
        log_error "rbln runtime not found: ${REBEL_HOME}/build/rbln/librbln_rt.so"
        log_error "Please build rebel-compiler first using rebel_install.sh"
        exit 1
    fi

    if [[ ! -d "${REBEL_HOME}/rbln/python/rbln" ]]; then
        log_error "rbln Python package not found: ${REBEL_HOME}/rbln/python/rbln"
        exit 1
    fi

    # Check rbln Python version compatibility
    check_rbln_python_version

    # Validate TORCH_RBLN_HOME
    if [[ ! -d "${TORCH_RBLN_HOME}" ]]; then
        log_error "TORCH_RBLN_HOME directory does not exist: ${TORCH_RBLN_HOME}"
        exit 1
    fi

    if [[ ! -f "${TORCH_RBLN_HOME}/pyproject.toml" ]]; then
        log_error "pyproject.toml not found in TORCH_RBLN_HOME: ${TORCH_RBLN_HOME}"
        exit 1
    fi

    log_info "REBEL_HOME: ${REBEL_HOME}"
    log_info "TORCH_RBLN_HOME: ${TORCH_RBLN_HOME}"
    log_info "Build type: ${build_type}"
}

# Debian/Ubuntu: gcc-13/g++-13. RHEL/CentOS/Fedora: gcc-toolset-13.
setup_compiler_env() {
    if [[ -f /etc/os-release ]]; then
        # shellcheck disable=SC1091
        . /etc/os-release
        os_id_like="${ID_LIKE:-}"
    else
        os_id_like=""
    fi
    if [[ "${os_id_like}" = "debian" ]]; then
        export CC=gcc-13
        export CXX=g++-13
        log_info "Using compiler: CC=${CC} CXX=${CXX} (debian)"
    elif [[ -f /opt/rh/gcc-toolset-13/enable ]]; then
        # shellcheck disable=SC1091
        . /opt/rh/gcc-toolset-13/enable
        # Use plain gcc/g++ so build_torch_rbln doesn't overwrite with gcc-13/g++-13
        # (toolset provides gcc/g++ in PATH, not gcc-13/g++-13)
        export CC=gcc
        export CXX=g++
        log_info "Using compiler: gcc-toolset-13 (RHEL/CentOS/Fedora), CC=${CC} CXX=${CXX}"
    else
        export CC=gcc-13
        export CXX=g++-13
        log_info "Using compiler: CC=${CC} CXX=${CXX}"
    fi
}

check_rbln_python_version() {
    # Get current Python version (e.g., "310" for Python 3.10)
    local current_py_version
    current_py_version=$(python -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')")

    log_info "Current Python version: ${current_py_version} (Python 3.${current_py_version:1})"

    # The rbln runtime extension is built for one Python next to the package sources.
    local runtime_dir="${REBEL_HOME}/rbln/python/rbln/runtime"
    local runtime_so
    runtime_so=$(find "${runtime_dir}" -maxdepth 1 -name "_runtime.cpython-${current_py_version}-*.so" 2>/dev/null | head -1 || true)

    if [[ -n "${runtime_so}" ]]; then
        log_info "Found matching rbln.runtime extension: $(basename "${runtime_so}")"
        log_info "Python version check passed!"
        return 0
    fi

    log_error "The rbln runtime extension was not built for Python ${current_py_version}!"
    log_error ""

    local available_versions
    # shellcheck disable=SC2038
    available_versions=$(find "${runtime_dir}" -maxdepth 1 -name "_runtime.cpython-*.so" 2>/dev/null | \
        xargs -I{} basename {} | grep -oP 'cpython-\K\d+' | sort -u || true)

    if [[ -n "${available_versions}" ]]; then
        log_error "Available rbln.runtime builds:"
        for ver in ${available_versions}; do
            log_error "  - Python 3.${ver:1} (cpython-${ver})"
        done
        log_error ""
    fi

    log_error "Solutions:"
    log_error "  1. Rebuild rebel-compiler with Python 3.${current_py_version:1}:"
    log_error "     cd ${REBEL_HOME}"
    log_error "     python3.${current_py_version:1} -m venv .venv && source .venv/bin/activate"
    log_error "     pip install conan~=2.0.0 cmake~=3.18 lit"
    log_error "     ./rebel_install.sh"
    log_error ""
    if [[ -n "${available_versions}" ]]; then
        local first_available
        first_available=$(echo "${available_versions}" | head -1)
        log_error "  2. Or use an available Python version for torch-rbln:"
        log_error "     python3.${first_available:1} -m venv .venv"
        log_error "     source .venv/bin/activate"
        log_error "     ./tools/build-with-external-rebel.sh --clean"
    fi
    exit 1
}

setup_virtualenv() {
    if [[ "${skip_venv}" -eq 1 ]]; then
        log_info "Skipping virtual environment creation (RBLN_SKIP_VENV=1)"
        return 0
    fi

    if [[ -d "${venv_path}" ]]; then
        log_warn "Virtual environment already exists: ${venv_path}"
        log_info "Activating existing virtual environment..."
    else
        log_info "Creating virtual environment: ${venv_path}"
        python3 -m venv "${venv_path}"
    fi

    # Source activate; the venv's activate may run '[ -f .use_external_rebel ] && source activate_rebel'
    # which returns 1 when the file is missing, causing set -e to exit. So temporarily allow non-zero.
    set +e
    # shellcheck disable=SC1091
    source "${venv_path}/bin/activate"
    set -e
    if [[ -z "${VIRTUAL_ENV:-}" ]]; then
        log_error "Failed to activate virtual environment: ${venv_path}"
        exit 1
    fi
    log_info "Virtual environment activated: ${VIRTUAL_ENV}"
}

install_dependencies() {
    log_info "Installing build dependencies..."
    pip install --upgrade pip
    # Align with rebel-compiler Quick Install: cmake<4.0, lit.
    pip install "cmake>=3.18,<4.0" ninja jinja2 hatchling setuptools-scm editables mypy lit

    # Create activate_rebel script
    create_activate_rebel_script

    log_info "Dependencies installed"
}

create_activate_rebel_script() {
    local activate_script="${TORCH_RBLN_HOME}/${venv_path}/bin/activate_rebel"
    local venv_root="${TORCH_RBLN_HOME}/${venv_path}"
    local flag_file="${venv_root}/.use_external_rebel"

    cat > "${activate_script}" << EOF
# Sourced only when .use_external_rebel exists in the venv. REBEL_HOME is what a rebuild of
# torch-rbln compiles against; rbln itself is importable through rbln_home.pth in site-packages.
export REBEL_HOME="${REBEL_HOME}"
EOF
    chmod +x "${activate_script}"

    # Mark this venv as "external rebel"; activate will only source activate_rebel when this exists
    touch "${flag_file}"

    # Add conditional auto-source to main activate script
    local activate_file="${venv_root}/bin/activate"
    if ! grep -q "activate_rebel" "${activate_file}" 2>/dev/null; then
        # shellcheck disable=SC2129
        echo '' >> "${activate_file}"
        echo '# Auto-source rebel environment only when .use_external_rebel exists' >> "${activate_file}"
        # Literal ${VIRTUAL_ENV} is expanded at activation time, not now; keep the single quotes.
        # shellcheck disable=SC2016
        echo '[ -f "${VIRTUAL_ENV}/.use_external_rebel" ] && source "${VIRTUAL_ENV}/bin/activate_rebel"' >> "${activate_file}"
    else
        # Migrate old unconditional source to conditional
        if ! grep -q '\.use_external_rebel' "${activate_file}" 2>/dev/null; then
            # Literal ${VIRTUAL_ENV} in the replacement is expanded at activation time; keep single quotes.
            # shellcheck disable=SC2016
            sed -i 's|^source "\${VIRTUAL_ENV}/bin/activate_rebel"$|[ -f "${VIRTUAL_ENV}/.use_external_rebel" ] \&\& source "${VIRTUAL_ENV}/bin/activate_rebel"|' "${activate_file}" 2>/dev/null || true
            touch "${flag_file}"
        fi
    fi

    log_info "Created activate_rebel script (conditional on .use_external_rebel)"
}

modify_pyproject() {
    log_info "Modifying pyproject.toml..."

    local torch_dep="torch==2.13.0+cpu"
    log_info "  Setting torch dependency to PyPI: ${torch_dep}"

    python3 << EOF
import re

torch_dep = "${torch_dep}"

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
#   "torch==2.13.0+cpu"
#   "torch (==2.13.0+cpu)"  <- parentheses format
#   "torch (>=2.13.0)"
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
    if [[ -n "${LDAP_USERNAME}" ]] && [[ -n "${LDAP_PASSWORD}" ]]; then
        log_info "Configuring uv with LDAP credentials..."
        export UV_INDEX_RBLN_INTERNAL_USERNAME="${LDAP_USERNAME}"
        export UV_INDEX_RBLN_INTERNAL_PASSWORD="${LDAP_PASSWORD}"
    else
        log_warn "LDAP credentials not set. Set UV_INDEX_RBLN_INTERNAL_USERNAME and UV_INDEX_RBLN_INTERNAL_PASSWORD."
        log_warn "Set LDAP_USERNAME and LDAP_PASSWORD if needed."
    fi
}

# Make the rbln package of REBEL_HOME importable in place, after `uv sync` so the sync keeps it.
# rbln ships no packaging metadata, so its third-party imports are installed here.
install_rbln_package() {
    local site_packages
    site_packages=$(python -c "import sysconfig; print(sysconfig.get_path('purelib'))") || return $?

    log_info "Adding ${REBEL_HOME}/rbln/python to ${site_packages}/rbln_home.pth"
    echo "${REBEL_HOME}/rbln/python" > "${site_packages}/rbln_home.pth"

    uv pip install numpy ml_dtypes || return $?

    python -c "import rbln.runtime" || {
        log_error "rbln.runtime does not import from ${REBEL_HOME}/rbln/python"
        return 1
    }
    log_info "rbln installed"
}

build_torch_rbln() {
    log_info "Building torch-rbln..."

    if [[ -z "${CC:-}" ]] || [[ -z "${CXX:-}" ]]; then
        log_error "CC and CXX must be set by setup_compiler_env."
        exit 1
    fi
    # FindRebel.cmake reads REBEL_HOME for the runtime headers and libraries.
    export REBEL_HOME="${REBEL_HOME}"

    local current_torch_version
    current_torch_version=$(pip show torch 2>/dev/null | grep "^Version:" | awk '{print $2}' || true)

    if [[ "${current_torch_version}" != "2.13.0+cpu" ]]; then
        log_info "Installing PyTorch 2.13.0+cpu from PyPI..."
        pip uninstall -y torch 2>/dev/null || true
        pip install torch==2.13.0+cpu --index-url https://download.pytorch.org/whl/cpu
    fi

    # Save torch version before uv sync
    local torch_before
    torch_before=$(pip show torch 2>/dev/null | grep "^Version:" | awk '{print $2}' || true)
    log_info "Torch version before uv sync: ${torch_before}"

    # Update uv.lock and install dependencies (rbln is made importable after sync)
    log_info "Updating uv.lock..."
    uv lock

    log_info "Installing dependencies..."
    uv sync --no-install-project || true

    # Check if torch was overwritten by uv sync
    local torch_after
    torch_after=$(pip show torch 2>/dev/null | grep "^Version:" | awk '{print $2}' || true)

    if [[ "${torch_before}" != "${torch_after}" ]]; then
        log_warn "Torch was changed by uv sync: ${torch_before} -> ${torch_after}"
        log_info "Reinstalling torch from PyPI..."
        pip uninstall -y torch 2>/dev/null || true
        pip install torch==2.13.0+cpu --index-url https://download.pytorch.org/whl/cpu
    fi

    local torch_final
    torch_final=$(pip show torch 2>/dev/null | grep "^Version:" | awk '{print $2}' || true)
    log_info "Final torch version: ${torch_final}"

    # shellcheck disable=SC2310
    install_rbln_package || return $?

    # Build and install torch-rbln
    log_info "Building torch-rbln with gcc-13..."
    CC=${CC} CXX=${CXX} TORCH_RBLN_BUILD_TYPE="${build_type}" uv pip install -e . --no-build-isolation

    log_info "torch-rbln installed successfully!"
}

verify_installation() {
    log_info "Verifying installation..."

    # Check library files
    local libs_ok=1
    [[ -f "torch_rbln/lib/libtorch_rbln.so" ]] || libs_ok=0
    [[ -f "torch_rbln/lib/libc10_rbln.so" ]] || libs_ok=0

    if [[ ${libs_ok} -eq 0 ]]; then
        log_warn "Some library files not found"
    fi

    # Re-source activate_rebel to ensure REBEL_HOME is set correctly
    local activate_rebel_script="${TORCH_RBLN_HOME}/${venv_path}/bin/activate_rebel"
    if [[ -f "${activate_rebel_script}" ]]; then
        # shellcheck disable=SC1090
        source "${activate_rebel_script}"
    fi

    # Test imports
    log_info "Testing imports..."

    local import_result=0
    python -c "
import torch
print(f'  torch: {torch.__version__}')
import rbln.runtime
print(f'  rbln: {rbln.runtime.__file__}')
import torch_rbln
print(f'  torch_rbln: {torch_rbln.__version__}')
" || import_result=$?

    if [[ "${import_result}" -ne 0 ]]; then
        echo ""
        log_error "=========================================="
        log_error "Import test FAILED! (exit code: ${import_result})"
        log_error "=========================================="
        log_error ""
        log_error "Possible causes:"
        log_error "  - Segmentation fault (library version mismatch)"
        log_error "  - RBLN ABI mismatch: REBEL_HOME was rebuilt after torch-rbln; rebuild torch-rbln"
        log_error "  - rbln not importable (check \$VIRTUAL_ENV/lib/python*/site-packages/rbln_home.pth)"
        log_error "  - Python version mismatch with the rbln runtime extension"
        log_error "  - GCC version mismatch between torch and torch-rbln"
        log_error ""
        log_error "Try manually:"
        log_error "  source ${TORCH_RBLN_HOME}/${venv_path}/bin/activate"
        log_error "  python -m torch_rbln.diagnose"
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
    echo "  REBEL_HOME:      ${REBEL_HOME}"
    echo "  TORCH_RBLN_HOME: ${TORCH_RBLN_HOME}"
    echo "  Compiler:        ${CC} / ${CXX}"
    echo "  Build type:      ${build_type}"
    echo "  PyTorch:         2.13.0+cpu (PyPI)"
    echo ""
    echo -e "${GREEN}How to use:${NC}"
    echo ""
    echo "  1. Activate the virtual environment:"
    echo "     cd ${TORCH_RBLN_HOME}"
    echo "     source ${venv_path}/bin/activate"
    echo ""
    echo "  2. The activate_rebel script is auto-sourced, setting REBEL_HOME for rebuilds."
    echo "     rbln is imported from ${REBEL_HOME}/rbln/python through rbln_home.pth."
    echo ""
    echo "  3. Example usage:"
    echo "     python -c 'import torch; import torch_rbln; print(\"OK\")'"
    echo ""
}

main() {
    log_info "Starting build..."

    # Source .bashrc when present (e.g. CI) so PATH and tooling are consistent
    if [[ -f "${HOME}/.bashrc" ]]; then
        # shellcheck disable=SC1091
        . "${HOME}/.bashrc"
    fi

    setup_compiler_env

    # Check prerequisites
    check_prerequisites
    cd "${TORCH_RBLN_HOME}" || exit 1

    # Handle clean options
    if [[ "${do_clean}" -eq 1 ]]; then
        clean_build_artifacts
        [[ "${clean_only}" -eq 1 ]] && exit 0
    fi

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
