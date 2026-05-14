#!/bin/bash
#
# Build a local rebel-compiler checkout and wire its wheel into this repo (uv).
#
# Prerequisites:
#   - REBEL_HOME: path to a rebel-compiler tree that provides the same entry points
#     as upstream (e.g. rebel_install.sh at the repo root, python/gen_requirements.py).
#   - RBLN_USE_EXTERNAL_REBEL_COMPILER=1
#
# Conan remotes, profiles, and CMake dependency resolution are owned by rebel-compiler's
# rebel_install.sh; this script only prepares a Python toolchain (uv venv + pip deps),
# invokes that installer with torch-rbln integration flags, builds the Python wheel,
# and updates this project's dependency pin.

set -e

readonly use_external_rebel_compiler="${RBLN_USE_EXTERNAL_REBEL_COMPILER:-0}"
readonly use_new_hash="${RBLN_USE_NEW_HASH:-0}"
readonly build_type="${TORCH_RBLN_BUILD_TYPE:-Release}"
readonly rebel_deploy="${REBEL_DEPLOY:-ON}"
readonly rebel_prod="${REBEL_PROD:-0}"
readonly artifactory_username="${RBLN_ARTIFACTORY_USERNAME:-}"
readonly artifactory_password="${RBLN_ARTIFACTORY_PASSWORD:-}"
readonly skip_uv_lock="${SKIP_UV_LOCK:-0}"
readonly conan_remote="${RBLN_CONAN_REMOTE_NAME:-rebel}"
readonly source_dir="$(realpath "$(dirname "$0")/..")"

# Parse command line arguments
use_latest_umd=0
build_only=0
clean_build=0
rebel_all_core=1
rebel_install_args=()
while [[ $# -gt 0 ]]; do
  case ${1} in
    -r|--rblnthunk_use_latest)
      use_latest_umd=1
      shift
      ;;
    --build-only)
      build_only=1
      shift
      ;;
    --clean)
      clean_build=1
      shift
      ;;
    --no-allcore)
      rebel_all_core=0
      shift
      ;;
    *)
      rebel_install_args+=("$1")
      shift
      ;;
  esac
done

[ "$rebel_all_core" -eq 1 ] && rebel_install_args=(-a "${rebel_install_args[@]}")

main() {
  local old_venv="${VIRTUAL_ENV:-}"

  if [ "$use_external_rebel_compiler" -eq 0 ] || [ -z "${REBEL_HOME:-}" ]; then
    echo "Error: Build against an external rebel-compiler checkout:" >&2
    echo "  export RBLN_USE_EXTERNAL_REBEL_COMPILER=1" >&2
    echo "  export REBEL_HOME=/path/to/rebel_compiler" >&2
    echo "  ./tools/apply-custom-rebel.sh" >&2
    return 1
  fi

  cd "$REBEL_HOME" || return $?
  REBEL_HOME="$(pwd)"
  export REBEL_HOME

  if [ ! -f "$REBEL_HOME/rebel_install.sh" ]; then
    echo "Error: REBEL_HOME must contain rebel_install.sh (got: $REBEL_HOME)" >&2
    return 1
  fi
  if [ ! -x "$REBEL_HOME/rebel_install.sh" ]; then
    echo "Error: rebel_install.sh is not executable (chmod +x \"$REBEL_HOME/rebel_install.sh\")" >&2
    return 1
  fi

  # Remove existing build dir if requested (avoids Ninja vs Unix Makefiles cache mismatch)
  if [ "$clean_build" -eq 1 ]; then
    echo "Cleaning existing build directory (--clean)..."
    rm -rf ./build
  fi

  # Update the submodules
  git submodule update --init ./ || return $?

  # Create a virtual environment using uv (--clear replaces any stale .venv from a prior run)
  uv venv --clear .venv && . .venv/bin/activate || return $?

  # Tooling needed before rebel_install.sh
  uv pip install conan~=2.1.0 cmake~=3.18 build lit auditwheel wheel ninja setuptools || return $?

  # Optional non-interactive Conan auth when the remote already exists (rebel_install.sh
  # registers the remote on first use; it may prompt for LDAP interactively otherwise).
  if [ -n "$artifactory_username" ]; then
    export LDAP_USERNAME="${LDAP_USERNAME:-$artifactory_username}"
    conan remote login "$conan_remote" "$artifactory_username" \
      ${artifactory_password:+-p "$artifactory_password"} || true
  fi

  # Build rebel-compiler (Conan profile/remote setup lives in rebel_install.sh)
  ./rebel_install.sh "${rebel_install_args[@]}" \
    -n \
    -DCMAKE_INSTALL_PREFIX=./build/install \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    "-DINSTALL_DEV=$( [ "$rebel_deploy" = ON ] && echo OFF || echo ON )" \
    "-DRBLNTHUNK_USE_LATEST=$( [ "$use_latest_umd" -eq 1 ] && echo ON || echo OFF )" \
    "-DCMAKE_BUILD_TYPE=$build_type" \
    "-DREBEL_DEPLOY=$rebel_deploy" || return $?

  # Install the Python dependencies.
  # Newer rebel-compiler (scikit-build-core) declares deps in pyproject.toml and no
  # longer ships python/gen_requirements.py. Fall back to the legacy path if present.
  if [ -f ./python/gen_requirements.py ]; then
    python ./python/gen_requirements.py
  fi
  if [ -f ./python/requirements/core.txt ]; then
    uv pip install -r ./python/requirements/core.txt || return $?
  fi

  if [ "$build_only" -eq 1 ]; then
    deactivate
    [ -n "$old_venv" ] && . "$old_venv/bin/activate"
    echo "Build-only complete. REBEL_HOME=$REBEL_HOME is ready for build-with-external-rebel.sh"
    return 0
  fi

  local new_hash
  new_hash="$(create_temp_commit_if_needed)"
  trap "remove_temp_commit_if_needed $new_hash" SIGINT

  # Build the rebel-compiler package
  build_package
  local ret=$?

  trap - SIGINT
  remove_temp_commit_if_needed "$new_hash"

  # If package building fails, abort
  [ $ret -eq 0 ] || return $ret

  # Deactivate the virtual environment
  deactivate

  # Restore the old virtual environment
  [ -n "$old_venv" ] && . "$old_venv/bin/activate"

  # Find the wheel file
  whls=()
  while read -r whl; do whls+=("$whl"); done < \
    <(find "$PWD/wheelhouse" -type f -name "rebel_compiler*.whl")
  if [ "${#whls[@]}" -ne 1 ]; then
    echo "Error: Expected to find exactly one wheel file, but found ${#whls[@]}." >&2
    return 1
  fi

  cd "$source_dir" || return $?

  # Pin rebel-compiler in pyproject.toml to the built wheel.
  python ./tools/replace_depends.py \
    --pyproject-path ./pyproject.toml \
    --source "rebel-compiler" \
    --target "rebel-compiler @ file://${whls[0]}" || return $?
  uv pip uninstall rebel-compiler 2>/dev/null || true

  if [ "$skip_uv_lock" -eq 1 ]; then
    # Fast development mode: skip lock and install directly
    echo "⚡ Fast mode: Installing rebel-compiler without uv lock (SKIP_UV_LOCK=1)"
    echo "   Use for rapid development. Run full build before deployment."
    uv pip install --force-reinstall "${whls[0]}"
  else
    # Standard mode: update lock file for reproducible builds
    echo "🔒 Standard mode: Updating uv.lock for reproducible build"
    uv lock && uv sync --no-install-project
  fi
}

create_temp_commit_if_needed() {
  [ "$use_new_hash" -eq 0 ] && return 0
  git commit -m "Temp" --allow-empty --no-verify 1>/dev/null
  git rev-parse HEAD
}

remove_temp_commit_if_needed() {
  [ "$use_new_hash" -eq 0 ] && return 0
  git reset "$1~1"
}

build_package() {
  rm -rf ./python/dist && mkdir -p ./python/dist
  rm -rf ./wheelhouse && mkdir -p ./wheelhouse
  git clean -fdx ./python/tvm || true

  local build_env="REBEL_PROD=$rebel_prod"
  if [ "$use_latest_umd" -eq 1 ]; then
    build_env="$build_env RBLNTHUNK_USE_LATEST=1"
  fi

  eval "$build_env python -m build ./python --wheel" || return $?
  for wheel in ./python/dist/*.whl; do
    auditwheel repair "$wheel" || return $?
  done
}

main
