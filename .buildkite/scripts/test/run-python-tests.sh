#!/usr/bin/env bash
# Run a Python test suite against the installed wheel.
#
# Usage:
#   run-python-tests.sh --build-type <Release|Debug>
#                       --artifact-build-id <build ID>
#                       --artifact-dir <name>
#                       --exclude-periodic <true|false>
#                       --test-suite <core|distributed|ops|models>
#                       --npu-product <name>
#                       --num-npus <0|1|2|4|8>
#                       --num-workers <n>
set -euo pipefail

build_type=""
artifact_build_id=""
artifact_dir=""
exclude_periodic=""
test_suite=""
npu_product=""
num_npus=""
num_workers=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-type)
      build_type="$2"
      shift 2
      ;;
    --artifact-build-id)
      artifact_build_id="$2"
      shift 2
      ;;
    --artifact-dir)
      artifact_dir="$2"
      shift 2
      ;;
    --exclude-periodic)
      case "${2-}" in
        true | false) exclude_periodic="$2" ;;
        *)
          echo "--exclude-periodic must be true or false, not '${2-}'" >&2
          exit 2
          ;;
      esac
      shift 2
      ;;
    --test-suite)
      test_suite="$2"
      shift 2
      ;;
    --npu-product)
      npu_product="$2"
      shift 2
      ;;
    --num-npus)
      case "${2-}" in
        0 | 1 | 2 | 4 | 8) num_npus="$2" ;;
        *)
          echo "--num-npus must be 0, 1, 2, 4, or 8, not '${2-}'" >&2
          exit 2
          ;;
      esac
      shift 2
      ;;
    --num-workers)
      if [[ ! "${2-}" =~ ^[1-9][0-9]*$ ]]; then
        echo "--num-workers must be a positive integer, not '${2-}'" >&2
        exit 2
      fi
      num_workers="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${build_type}" ]]; then
  echo "--build-type is required" >&2
  exit 2
fi

if [[ -z "${artifact_build_id}" ]]; then
  echo "--artifact-build-id is required" >&2
  exit 2
fi

if [[ -z "${artifact_dir}" ]]; then
  echo "--artifact-dir is required" >&2
  exit 2
fi

if [[ -z "${exclude_periodic}" ]]; then
  echo "--exclude-periodic is required" >&2
  exit 2
fi

if [[ -z "${test_suite}" ]]; then
  echo "--test-suite is required" >&2
  exit 2
fi

if [[ -z "${npu_product}" ]]; then
  echo "--npu-product is required" >&2
  exit 2
fi

if [[ -z "${num_npus}" ]]; then
  echo "--num-npus is required" >&2
  exit 2
fi

if [[ -z "${num_workers}" ]]; then
  echo "--num-workers is required" >&2
  exit 2
fi

case "${num_npus}" in
  0)
    selection="((not npu) or npu(count=0))"
    pytest_env=("RBLN_FORCE_NPU_NAME=${npu_product}")
    ;;
  *)
    selection="npu(count=${num_npus})"
    pytest_env=()
    ;;
esac
if [[ "${exclude_periodic}" == "true" ]]; then
  selection="(not periodic) and ${selection}"
fi

case "${test_suite}" in
  core) test_dir="test/rbln" ;;
  distributed | ops | models) test_dir="test/${test_suite}" ;;
  *)
    echo "--test-suite must be core, distributed, ops, or models, not" \
      "'${test_suite}'" >&2
    exit 2
    ;;
esac

echo "--- Checking NPU environment"
shopt \
  -s \
  nullglob
device_nodes=(/dev/rbln[0-9]*)
shopt \
  -u \
  nullglob
if [[ "${#device_nodes[@]}" -ne "${num_npus}" ]]; then
  echo "The number of /dev/rbln* device nodes is ${#device_nodes[@]}, not" \
    "the requested ${num_npus}" >&2
  exit 1
fi
for device_node in "${device_nodes[@]}"; do
  card_name="$(<"/sys/class/rebellions/${device_node#/dev/}/card_name")"
  if [[ "${card_name}" != "${npu_product}" ]]; then
    echo "${device_node} is ${card_name}, not the requested" \
      "${npu_product}" >&2
    exit 1
  fi
done
if ! python3 \
  -c 'import ctypes; ctypes.CDLL("librbln-thunk.so")'; then
  echo "Could not load librbln-thunk.so, the RBLN user-mode driver" >&2
  exit 1
fi

echo "--- Downloading ${artifact_dir}"
buildkite-agent artifact download \
  --build "${artifact_build_id}" \
  "${artifact_dir}/*" \
  .
wheels=("${artifact_dir}"/*.whl)
wheel="${wheels[0]}"

echo "--- Installing locked environment"
# The downloaded lock, not the checkout's, pins the `torch` the wheel needs.
cp \
  "${artifact_dir}/pyproject.toml" \
  "${artifact_dir}/uv.lock" \
  .
uv sync \
  --extra runtime \
  --group test \
  --locked \
  --no-default-groups \
  --no-install-project
if [[ "${build_type}" == "Debug" ]]; then
  libnuma_status="$(
    dpkg-query \
      --show \
      --showformat='${db:Status-Status}' \
      libnuma1 \
      || true
  )"
  if [[ "${libnuma_status}" != "installed" ]]; then
    echo "--- Installing libnuma1"
    apt-get update
    apt-get install \
      --no-install-recommends \
      --yes \
      libnuma1
  fi
fi

echo "--- Installing $(basename "${wheel}")"
uv pip install \
  --python .venv/bin/python \
  "${wheel}[runtime]"

# `pytest` prepends the repository root to `sys.path`, where `./torch_rbln`
# would shadow the installed wheel.
mv \
  ./torch_rbln \
  ./torch_rbln.bak

failed_pytest_runs=()

# An op test also passes when the op fell back to CPU, comparing CPU against CPU, so
# the op suite records which ops fell back in each test.
serial_report=()
parallel_report=()
if [[ "${test_suite}" == "ops" ]]; then
  serial_report=(--fallback-report="test-reports/${test_suite}-serial.json")
  parallel_report=(--fallback-report="test-reports/${test_suite}-parallel.json")
fi

echo "--- Running ${test_suite} suite, ${test_dir}, serial tests"
pytest_exit_code=0
env \
  "${pytest_env[@]}" \
  uv run \
    --no-sync \
    pytest \
      "${test_dir}" \
      -m "${selection} and serial" \
      "${serial_report[@]}" \
      --numprocesses=1 \
  || pytest_exit_code=$?
# Exit 5 means `pytest` selected no tests, which is not a failure.
if [[ "${pytest_exit_code}" -ne 0 && "${pytest_exit_code}" -ne 5 ]]; then
  failed_pytest_runs+=("serial tests (pytest exit code ${pytest_exit_code})")
fi

echo "--- Running ${test_suite} suite, ${test_dir}, parallel tests"
pytest_exit_code=0
env \
  "${pytest_env[@]}" \
  uv run \
    --no-sync \
    pytest \
      "${test_dir}" \
      -m "${selection} and (not serial)" \
      "${parallel_report[@]}" \
      --numprocesses="${num_workers}" \
  || pytest_exit_code=$?
if [[ "${pytest_exit_code}" -ne 0 && "${pytest_exit_code}" -ne 5 ]]; then
  failed_pytest_runs+=("parallel tests (pytest exit code ${pytest_exit_code})")
fi

if [[ "${test_suite}" == "ops" ]]; then
  echo "--- Uploading CPU fallback reports"
  # The reports do not decide the step's result, so a failed upload does not fail it.
  if ! buildkite-agent artifact upload \
    "test-reports/${test_suite}-*.json"; then
    echo "Could not upload the CPU fallback reports" >&2
  fi
fi

if [[ "${#failed_pytest_runs[@]}" -gt 0 ]]; then
  echo "--- Reporting failed pytest runs"
  for failed_pytest_run in "${failed_pytest_runs[@]}"; do
    echo "Failed: ${failed_pytest_run}" >&2
  done
  exit 1
fi
