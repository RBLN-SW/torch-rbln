#!/usr/bin/env bash
# Check a pull request's title against Conventional Commits.
#
# Usage:
#   check-pr-title.sh
#
# Environment: `BUILDKITE_PULL_REQUEST`, and `GH_REPO` and `GH_TOKEN` for `gh`.
set -euo pipefail

# shellcheck source=.buildkite/scripts/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"

if [[ $# -gt 0 ]]; then
  echo "check-pr-title.sh takes no arguments: $*" >&2
  exit "${EXIT_USAGE}"
fi
if [[ -z "${GH_REPO:-}" ]]; then
  echo "GH_REPO is not set; check the step env" >&2
  exit "${EXIT_ENVIRONMENT}"
fi

echo "--- Reading pull request title"
if ! title="$(
  gh pr view "${BUILDKITE_PULL_REQUEST:?}" \
    --json title \
    --jq .title
)"; then
  echo "Cannot read pull request #${BUILDKITE_PULL_REQUEST} of ${GH_REPO}" >&2
  exit "${EXIT_ENVIRONMENT}"
fi
echo "Pull request #${BUILDKITE_PULL_REQUEST}: ${title}"

echo "--- Checking title against Conventional Commits"
types='feat|fix|refactor|perf|docs|test|build|ci|chore'
pattern="^(${types})(\([^()]+\))?!?: [a-z](.*[^.])?$"
if [[ "${title}" =~ ${pattern} ]]; then
  echo "The title follows Conventional Commits"
  exit 0
fi
{
  echo "'${title}' is not a Conventional Commits title"
  echo "  Format:        <type>(<optional scope>): <description>"
  echo "  Allowed types: ${types//|/, }"
  echo "  The description starts with a lowercase letter and does not end"
  echo "  with a period."
  echo "Edit the title, then push or rebuild the pull request's last build"
  echo "to run this check again."
} >&2
exit "${EXIT_FAILURE}"
