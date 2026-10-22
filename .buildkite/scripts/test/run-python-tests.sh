#!/usr/bin/env bash
# Run a Python test suite against the installed wheel.
#
# Usage:
#   run-python-tests.sh --build-type <Release|Debug>
#                       --artifact-build-id <build ID>
#                       --artifact-dir <name>
#                       --exclude-periodic <true|false>
#                       --suite <core|distributed|ops|models>
set -euo pipefail

build_type=""
artifact_build_id=""
artifact_dir=""
exclude_periodic=""
suite=""
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
    --suite)
      suite="$2"
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

if [[ -z "${suite}" ]]; then
  echo "--suite is required" >&2
  exit 2
fi

# `-m` rejects an empty `()`, so `selection` carries its own `and`.
selection=""
if [[ "${exclude_periodic}" == "true" ]]; then
  selection="(not periodic) and "
fi

case "${suite}" in
  core) test_dir="test/rbln" ;;
  distributed | ops | models) test_dir="test/${suite}" ;;
  *)
    echo "--suite must be core, distributed, ops, or models, not '${suite}'" >&2
    exit 2
    ;;
esac

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
  --locked \
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

# pytest prepends the repository root to `sys.path`, where `./torch_rbln`
# would shadow the installed wheel.
mv \
  ./torch_rbln \
  ./torch_rbln.bak

if [[ "${suite}" == "models" ]]; then
  echo "--- Installing model-test dependencies"
  uv run \
    --no-sync \
    bash \
      tools/test/install-test-deps.sh
fi

failed_pytest_runs=()

echo "--- Running ${suite} suite, ${test_dir}, serial tests"
pytest_exit_code=0
uv run \
  --no-sync \
  pytest \
    "${test_dir}" \
    -m "${selection}serial" \
    --numprocesses=1 \
  || pytest_exit_code=$?
# Exit 5 means pytest selected no tests, which is not a failure.
if [[ "${pytest_exit_code}" -ne 0 && "${pytest_exit_code}" -ne 5 ]]; then
  failed_pytest_runs+=("serial tests (pytest exit code ${pytest_exit_code})")
fi

echo "--- Running ${suite} suite, ${test_dir}, parallel tests"
pytest_exit_code=0
uv run \
  --no-sync \
  pytest \
    "${test_dir}" \
    -m "${selection}(not serial)" \
    --numprocesses=32 \
  || pytest_exit_code=$?
if [[ "${pytest_exit_code}" -ne 0 && "${pytest_exit_code}" -ne 5 ]]; then
  failed_pytest_runs+=("parallel tests (pytest exit code ${pytest_exit_code})")
fi

if [[ "${#failed_pytest_runs[@]}" -gt 0 ]]; then
  echo "--- Reporting failed pytest runs"
  for failed_pytest_run in "${failed_pytest_runs[@]}"; do
    echo "Failed: ${failed_pytest_run}" >&2
  done
  exit 1
fi
