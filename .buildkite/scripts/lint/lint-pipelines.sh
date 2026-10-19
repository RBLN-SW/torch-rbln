#!/usr/bin/env bash
# Lint the pipeline files under `.buildkite/`.
#
# Usage:
#   lint-pipelines.sh
set -euo pipefail

if [[ $# -gt 0 ]]; then
  echo "lint-pipelines.sh takes no arguments: $*" >&2
  exit 2
fi

failed_checks=()

echo "--- Installing lint tools"
uv sync \
  --locked \
  --only-group lint

echo "--- Running yamllint"
uv run \
  --no-sync \
  yamllint \
    --strict \
    .buildkite/ \
  || failed_checks+=(yamllint)

echo "--- Dry-running pipeline uploads"
# The dry runs use each file's defaults, not this job's `TORCH_RBLN_*` values.
unset_values=()
for name in "${!TORCH_RBLN_@}"; do
  unset_values+=("--unset=${name}")
done
for pipeline in .buildkite/*.yml .buildkite/pipelines/*.yml; do
  required="$(
    sed \
      --quiet \
      --regexp-extended \
      's/^  (TORCH_RBLN_[A-Z0-9_]+): \$\{\1\?\}$/\1=placeholder/p' \
      "${pipeline}"
  )"
  placeholders=()
  if [[ -n "${required}" ]]; then
    mapfile \
      -t \
      placeholders \
      <<<"${required}"
  fi
  if env \
    "${unset_values[@]}" \
    "${placeholders[@]}" \
    buildkite-agent pipeline upload \
      --dry-run \
      --format json \
      "${pipeline}" \
    >/dev/null; then
    echo "Passed: ${pipeline}"
  else
    failed_checks+=("${pipeline}")
  fi
done

if [[ "${#failed_checks[@]}" -gt 0 ]]; then
  echo "--- Reporting failed checks"
  for check in "${failed_checks[@]}"; do
    echo "Failed: ${check}" >&2
  done
  exit 1
fi
