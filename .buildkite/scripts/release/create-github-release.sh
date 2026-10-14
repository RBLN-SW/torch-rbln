#!/usr/bin/env bash
# Create the GitHub Release of the tag that `create-release-tag.sh` recorded when it is a final or
# post release and has none yet. The notes start at the newest final or post release below the tag.
#
# Usage:
#   create-github-release.sh
#
# Environment: `GH_REPO` and `GH_TOKEN` for `gh`.
set -euo pipefail

# shellcheck source=.buildkite/scripts/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"

if [[ $# -gt 0 ]]; then
  echo "create-github-release.sh takes no arguments: $*" >&2
  exit "${EXIT_USAGE}"
fi
if [[ -z "${GH_REPO:-}" ]]; then
  echo "GH_REPO is not set; check the step env" >&2
  exit "${EXIT_ENVIRONMENT}"
fi
tag="$(read_meta_data release-tag)"

echo "--- Looking up GitHub Release ${tag}"
kind="$(uv run --no-config "${SCRIPTS_DIR}/release/classify-release-tag.py" "${tag}")"
if [[ "${kind}" != final && "${kind}" != post ]]; then
  echo "${tag} gets no GitHub Release; only final and post releases do"
  exit 0
fi
# `releases/tags/<tag>` returns published releases only, so a draft for the tag does not stop this
# step from creating the release.
lookup_error="$(mktemp)"
if gh api \
  "repos/{owner}/{repo}/releases/tags/${tag}" \
  --silent \
  2> "${lookup_error}"; then
  echo "GitHub Release ${tag} already exists; nothing to do"
  exit 0
fi
if ! grep \
  --quiet \
  --fixed-strings \
  '(HTTP 404)' \
  "${lookup_error}"; then
  cat "${lookup_error}" >&2
  echo "Could not look up the GitHub Release of ${tag}" >&2
  exit "${EXIT_ENVIRONMENT}"
fi

echo "--- Finding release notes baseline"
tags="$(
  gh api \
    --paginate \
    "repos/{owner}/{repo}/git/matching-refs/tags/v" \
    --jq '.[].ref | ltrimstr("refs/tags/")'
)"
baseline="$(
  uv run \
    --no-config \
    "${SCRIPTS_DIR}/release/find-notes-baseline.py" \
    "${tag}" \
    <<< "${tags}"
)"
echo "Release notes baseline: ${baseline:-<none>}"

echo "--- Creating GitHub Release ${tag}"
notes=(--generate-notes)
if [[ -n "${baseline}" ]]; then
  notes+=(--notes-start-tag "${baseline}")
fi
gh release create \
  "${tag}" \
  --verify-tag \
  --title "${tag}" \
  "${notes[@]}"
echo "Created GitHub Release ${tag}"
