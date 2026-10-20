#!/usr/bin/env bash
# Run a Python test suite against the installed wheel.
#
# Usage:
#   run-python-tests.sh --build-type <Release|Debug>
#                       --artifact-build-id <build ID>
#                       --artifact-dir <name>
#                       --test-mode <ci|release>
#                       --suite <core|distributed|ops|models>
set -euo pipefail

build_type=""
artifact_build_id=""
artifact_dir=""
test_mode=""
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
    --test-mode)
      test_mode="$2"
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

if [[ -z "${test_mode}" ]]; then
  echo "--test-mode is required" >&2
  exit 2
fi

if [[ -z "${suite}" ]]; then
  echo "--suite is required" >&2
  exit 2
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

# `test/run_tests.py` runs `python -m pytest`, which puts the current
# directory first on `sys.path`, where `./torch_rbln` would shadow the
# installed wheel.
mv \
  ./torch_rbln \
  ./torch_rbln.bak

echo "--- Running ${suite} suite"
uv run \
  --no-sync \
  python \
    test/run_tests.py \
    --suite "${suite}" \
    --test_mode "${test_mode}" \
    --workers 32
