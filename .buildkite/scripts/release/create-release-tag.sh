#!/usr/bin/env bash
# Tag the commit the build runs on with `TORCH_RBLN_VERSION_TAG` through the GitHub API, and
# record the tag in the meta-data key `release-tag`. A re-run leaves a tag at that commit alone; a
# tag at another commit is an error.
#
# Usage:
#   create-release-tag.sh
#
# Environment: `TORCH_RBLN_VERSION_TAG`, the optional `TORCH_RBLN_RELEASE_WHEELS`, and `GH_REPO` and
# `GH_TOKEN` for `gh`.
set -euo pipefail

# shellcheck source=.buildkite/scripts/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"

if [[ $# -gt 0 ]]; then
  echo "create-release-tag.sh takes no arguments: $*" >&2
  exit "${EXIT_USAGE}"
fi
if [[ -z "${TORCH_RBLN_VERSION_TAG:-}" ]]; then
  echo "TORCH_RBLN_VERSION_TAG is required" >&2
  exit "${EXIT_USAGE}"
fi
if [[ -z "${GH_REPO:-}" ]]; then
  echo "GH_REPO is not set; check the step env" >&2
  exit "${EXIT_ENVIRONMENT}"
fi
tag="${TORCH_RBLN_VERSION_TAG}"
release_wheels="${TORCH_RBLN_RELEASE_WHEELS:-true}"

echo "--- Checking inputs"
kind="$(
  uv run \
    --no-config \
    "${SCRIPTS_DIR}/release/classify-release-tag.py" \
    "${tag}"
)"
echo "Release kind of ${tag}: ${kind}"
# The wheel steps run unless the value is `false`, so a mistyped value stops the release before it
# tags anything.
case "${release_wheels}" in
  true | false) ;;
  *)
    echo "TORCH_RBLN_RELEASE_WHEELS is '${release_wheels}'; expected true or false" >&2
    exit "${EXIT_USAGE}"
    ;;
esac
if [[ "${kind}" == development && "${release_wheels}" == true ]]; then
  echo "A release publishes no development version; start the build with" \
    "TORCH_RBLN_RELEASE_WHEELS=false to create ${tag} alone" >&2
  exit "${EXIT_USAGE}"
fi

# tag_commit <tag>: print the commit the tag points at, or nothing if there is no such tag.
tag_commit() {
  local object type sha
  if ! object="$(
    gh api \
      --paginate \
      "repos/{owner}/{repo}/git/matching-refs/tags/$1" \
      --jq ".[] | select(.ref == \"refs/tags/$1\") | \"\(.object.type) \(.object.sha)\""
  )"; then
    echo "Could not read the tag $1 in ${GH_REPO}" >&2
    exit "${EXIT_ENVIRONMENT}"
  fi
  read -r type sha <<< "${object}"
  if [[ "${type}" == tag ]]; then
    if ! sha="$(
      gh api \
        "repos/{owner}/{repo}/git/tags/${sha}" \
        --jq .object.sha
    )"; then
      echo "Could not read the tag object of $1 in ${GH_REPO}" >&2
      exit "${EXIT_ENVIRONMENT}"
    fi
  fi
  echo "${sha}"
}

echo "--- Creating tag ${tag}"
# `BUILDKITE_COMMIT` may name the commit symbolically (`HEAD`); the checkout has resolved it.
sha="$(git rev-parse HEAD)"
echo "Tag ${tag} -> ${sha}"

existing="$(tag_commit "${tag}")"
if [[ -z "${existing}" ]]; then
  gh api \
    --method POST \
    "repos/{owner}/{repo}/git/refs" \
    --raw-field "ref=refs/tags/${tag}" \
    --raw-field "sha=${sha}" \
    --silent
  echo "Created tag ${tag} at ${sha}"
elif [[ "${existing}" == "${sha}" ]]; then
  echo "Tag ${tag} already points at ${sha}; skipping tag creation"
else
  echo "Tag ${tag} already exists at ${existing}, not at ${sha}; pick a new version" >&2
  exit "${EXIT_FAILURE}"
fi

buildkite-agent meta-data set release-tag "${tag}"
