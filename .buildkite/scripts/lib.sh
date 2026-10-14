#!/usr/bin/env bash
# Helpers the scripts under `.buildkite/scripts/` source.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "${BASH_SOURCE[0]##*/} must be sourced, not executed" >&2
  exit 2
fi

# Exit codes for the failures a script detects itself: 1 for a failed check or test, 2 for a usage
# error, 3 for an environment the step cannot work in (a missing artifact, a wrong NPU count). A
# command that fails on its own exits with its own status.
# shellcheck disable=SC2034
EXIT_FAILURE=1
# shellcheck disable=SC2034
EXIT_USAGE=2
EXIT_ENVIRONMENT=3

# `SCRIPTS_DIR` is absolute so that the helpers find their Python scripts after a caller changes
# directory.
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# check_npu_environment <NPU count>: exit 3 unless the kernel driver, its device nodes, and the RBLN
# runtime with its user-mode driver are present, and `rbln-smi` and the runtime both count <NPU
# count> NPUs. The runtime probe runs in the environment `install-wheel.sh` installed.
check_npu_environment() {
  local expected=$1 smi_json smi_count driver_version device_nodes probe loaded umd physical
  echo "--- Checking NPU environment"
  if ! command -v rbln-smi >/dev/null 2>&1; then
    echo "rbln-smi is not on PATH, so the NPUs allocated to this pod cannot be listed" >&2
    exit "${EXIT_ENVIRONMENT}"
  fi
  if ! rbln-smi; then
    echo "rbln-smi failed" >&2
    exit "${EXIT_ENVIRONMENT}"
  fi
  if ! smi_json="$(rbln-smi --json)"; then
    echo "rbln-smi --json failed" >&2
    exit "${EXIT_ENVIRONMENT}"
  fi
  if ! smi_count="$(
    python3 -c 'import json, sys; print(len(json.load(sys.stdin)["devices"]))' <<<"${smi_json}"
  )"; then
    echo "rbln-smi --json gave no device list" >&2
    exit "${EXIT_ENVIRONMENT}"
  fi
  if [[ ! -r /sys/module/rebellions/version ]]; then
    echo "The rebellions kernel driver is not loaded" >&2
    exit "${EXIT_ENVIRONMENT}"
  fi
  driver_version="$(</sys/module/rebellions/version)"
  if ! device_nodes="$(compgen -G '/dev/rbln[0-9]*')"; then
    echo "The pod has no /dev/rbln* device node" >&2
    exit "${EXIT_ENVIRONMENT}"
  fi
  echo "Kernel driver: rebellions ${driver_version}; device nodes: ${device_nodes//$'\n'/ }"
  probe="$(mktemp)"
  if ! uv run --no-sync python "${SCRIPTS_DIR}/test/probe-rbln-runtime.py" npu "${probe}"; then
    echo "The NPU probe (import torch_rbln, physical_device_count) failed" >&2
    exit "${EXIT_ENVIRONMENT}"
  fi
  read -r loaded umd physical < "${probe}"
  echo "RBLN runtime loaded: ${loaded} (user-mode driver: ${umd});" \
    "NPUs: rbln-smi ${smi_count}, runtime ${physical}, expected ${expected}"
  if [[ "${loaded}" != True ]]; then
    echo "The RBLN runtime did not load its user-mode driver" >&2
    exit "${EXIT_ENVIRONMENT}"
  fi
  if [[ "${smi_count}" != "${expected}" || "${physical}" != "${expected}" ]]; then
    echo "The pod should expose ${expected} NPUs;" \
      "rbln-smi lists ${smi_count}, the runtime counts ${physical}" >&2
    exit "${EXIT_ENVIRONMENT}"
  fi
}

# single_wheel <directory> <exit code>: print the path of the one wheel in <directory>; any other
# count exits with <exit code>.
single_wheel() {
  local wheels
  shopt -s nullglob
  wheels=("$1"/*.whl)
  shopt -u nullglob
  if [[ "${#wheels[@]}" -ne 1 ]]; then
    echo "Expected exactly one wheel in $1/, found ${#wheels[@]}: ${wheels[*]}" >&2
    exit "$2"
  fi
  echo "${wheels[0]}"
}

# read_meta_data <key>: print the build's meta-data value of <key>, which an earlier step set.
read_meta_data() {
  local value
  if ! value="$(buildkite-agent meta-data get "$1")"; then
    echo "Could not read the meta-data key $1; the step that sets it did not run or failed" >&2
    exit "${EXIT_ENVIRONMENT}"
  fi
  echo "${value}"
}

# index_setting <name> <key>: print the <key> (`url`, `publish-url`) that `pyproject.toml` gives the
# uv index <name>.
index_setting() {
  local value
  if ! value="$(
    uv run \
      --quiet \
      --no-config \
      "${SCRIPTS_DIR}/read-index-setting.py" \
      "$1" \
      "$2"
  )"; then
    echo "Could not read the uv index $1 from pyproject.toml" >&2
    exit "${EXIT_ENVIRONMENT}"
  fi
  if [[ -z "${value}" ]]; then
    echo "pyproject.toml gives the uv index $1 no $2" >&2
    exit "${EXIT_ENVIRONMENT}"
  fi
  echo "${value}"
}

# resolve_version <index> <requirement>: print the version that the uv index <index> resolves
# <requirement> to, pre-releases included. Without `--no-config`, the constraints in
# `pyproject.toml` would pin the answer to the version already in use; the option drops the index
# declarations too, so `UV_INDEX` brings the index back under the name its credentials carry.
resolve_version() {
  local index_url resolved version
  index_url="$(index_setting "$1" url)" || return
  if ! resolved="$(
    UV_INDEX="$1=${index_url}" uv pip compile \
      - \
      --no-config \
      --no-deps \
      --no-header \
      --no-annotate \
      --prerelease allow \
      <<< "$2"
  )"; then
    echo "Could not resolve $2 from the $1 index" >&2
    exit "${EXIT_ENVIRONMENT}"
  fi
  version="$(sed --quiet 's/^[^=]*==//p' <<< "${resolved}")"
  if [[ -z "${version}" ]]; then
    echo "The $1 index resolved $2 to no version: ${resolved}" >&2
    exit "${EXIT_ENVIRONMENT}"
  fi
  echo "${version}"
}

# index_credential_var <index> <USERNAME|PASSWORD>: print the name of the variable that holds the
# credential, `UV_INDEX_<INDEX>_USERNAME` or `UV_INDEX_<INDEX>_PASSWORD`.
index_credential_var() {
  local var
  var="UV_INDEX_$(tr '[:lower:]-' '[:upper:]_' <<<"$1")_$2"
  if [[ -z "${!var:-}" ]]; then
    echo "${var} is not set; check the step's secrets" >&2
    exit "${EXIT_ENVIRONMENT}"
  fi
  echo "${var}"
}

# index_has_wheel <index> <project> <wheel>: print `true` when the project page that an upload to
# the uv index <index> lands in lists <wheel>, and `false` when it does not or the project has no
# page. The index's `url` may be a group that serves other repositories too, so the check reads the
# repository of its `publish-url`. An upload checks here first because `uv` skips only an
# identical file, and a rebuild of a version is not one.
index_has_wheel() {
  local publish_url url username_var password_var
  publish_url="$(index_setting "$1" publish-url)" || return
  url="${publish_url%/}/simple/$2/"
  username_var="$(index_credential_var "$1" USERNAME)" || return
  password_var="$(index_credential_var "$1" PASSWORD)" || return
  if ! INDEX_USERNAME="${!username_var}" \
    INDEX_PASSWORD="${!password_var}" \
    uv run \
      --quiet \
      --no-config \
      "${SCRIPTS_DIR}/publishing/check-index-listing.py" \
      "${url}" \
      "$3"; then
    echo "Cannot check whether ${url} lists $3" >&2
    exit "${EXIT_ENVIRONMENT}"
  fi
}
