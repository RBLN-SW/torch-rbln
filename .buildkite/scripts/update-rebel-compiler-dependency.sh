#!/usr/bin/env bash
# Propose a `rebel-compiler` pin update as a pull request.
#
# Usage:
#   update-rebel-compiler-dependency.sh
set -euo pipefail

package="rebel-compiler"
head_branch="update-rebel-compiler"
base_branch="main"

if [[ $# -gt 0 ]]; then
  echo "update-rebel-compiler-dependency.sh takes no arguments: $*" >&2
  exit 2
fi

: "${GH_REPO:?is not set}"
: "${GH_TOKEN:?is not set}"

# The head branch is one commit ahead of this build's commit, so a build on
# another branch would carry that branch's commits into the pull request.
if [[ "${BUILDKITE_BRANCH:-}" != "${base_branch}" ]]; then
  echo "BUILDKITE_BRANCH is '${BUILDKITE_BRANCH:-}', not ${base_branch}" >&2
  exit 2
fi

echo "--- Resolving ${package} version"
index_url="$(
  .buildkite/scripts/read-index-setting.py \
    --index rbln-nightly \
    --setting url
)"
# Without `--no-config`, the `constraint-dependencies` of `pyproject.toml`
# would hold the result at the current pin.
if ! pinned_requirement="$(
  uv pip compile \
    - \
    --no-config \
    --index "rbln-nightly=${index_url}" \
    --no-deps \
    --no-header \
    --no-annotate \
    --prerelease allow \
    <<<"${package}"
)"; then
  echo "Could not resolve ${package} from the rbln-nightly index" >&2
  exit 1
fi
target="$(
  sed \
    --quiet \
    's/^[^=]*==//p' \
    <<<"${pinned_requirement}"
)"
if [[ -z "${target}" ]]; then
  echo "The rbln-nightly index resolved ${package} to no version:" \
    "${pinned_requirement}" >&2
  exit 1
fi
echo "Resolved ${package} to ${target}"

echo "--- Comparing ${target} with current pin"
current="$(
  uv run \
    --no-config \
    tools/pin_rebel_compiler.py \
      --current
)"
echo "Current pin: ${current}"
if [[ "${target}" == "${current}" ]]; then
  echo "Already pinned to ${target}; nothing to do"
  exit 0
fi
is_later="$(
  .buildkite/scripts/compare-versions.py \
    --candidate "${target}" \
    --baseline "${current}"
)"
if [[ "${is_later}" != "true" ]]; then
  echo "${target} is not newer than the current pin ${current}; nothing to do"
  exit 0
fi

echo "--- Pinning ${package} to ${target}"
uv run \
  --no-config \
  tools/pin_rebel_compiler.py \
    "${target}"
uv lock

echo "--- Proposing ${target}"
.buildkite/scripts/propose-dependency-update.py \
  --package "${package}" \
  --package-version "${target}" \
  --head-branch "${head_branch}" \
  --base-branch "${base_branch}"
