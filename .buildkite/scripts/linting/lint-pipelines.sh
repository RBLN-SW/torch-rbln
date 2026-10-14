#!/usr/bin/env bash
# Lint `.buildkite/`: `yamllint` on the YAML, `shellcheck` on the scripts, and a dry-run upload of
# every pipeline file. Every check runs even when an earlier one fails.
#
# Usage:
#   lint-pipelines.sh
set -euo pipefail

# shellcheck source=.buildkite/scripts/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"

if [[ $# -gt 0 ]]; then
  echo "lint-pipelines.sh takes no arguments: $*" >&2
  exit "${EXIT_USAGE}"
fi
failed=()

echo "--- Installing lint tools"
uv venv
uv pip install --group lint

echo "--- Running yamllint"
uv run \
  --no-sync \
  yamllint \
    --strict \
    .buildkite/ \
  || failed+=(yamllint)

echo "--- Running shellcheck"
# A pull request's `lintrunner` checks only the changed scripts; a change to `lib.sh` can break the
# scripts that source it.
find .buildkite/scripts \
  -type f \
  -name '*.sh' \
  -exec uv run --no-sync shellcheck {} + \
  || failed+=(shellcheck)

echo "--- Dry-running pipeline uploads"
# Each file must upload on its own defaults, as it does when a pipeline's steps upload it directly,
# so the dry runs leave out this job's `TORCH_RBLN_*` values.
unset_values=()
for name in "${!TORCH_RBLN_@}"; do
  unset_values+=("--unset=${name}")
done
for pipeline in .buildkite/*.yml .buildkite/pipelines/*.yml; do
  if env "${unset_values[@]}" buildkite-agent pipeline upload \
    --dry-run \
    --format json \
    "${pipeline}" \
    > /dev/null; then
    echo "${pipeline}: ok"
  else
    failed+=("the dry run of ${pipeline}")
  fi
done

if [[ "${#failed[@]}" -gt 0 ]]; then
  echo "Failed: ${failed[*]}" >&2
  exit "${EXIT_FAILURE}"
fi
