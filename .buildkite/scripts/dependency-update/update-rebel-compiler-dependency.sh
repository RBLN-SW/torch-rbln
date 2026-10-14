#!/usr/bin/env bash
# Propose a `rebel-compiler` pin update as a pull request. The pin is the requested version, or the
# newest build on the nightly index when it is a production build newer than the current pin. The
# change goes to the proposal branch as one force-pushed commit, so an unmerged proposal is
# replaced rather than stacked on.
#
# Usage:
#   update-rebel-compiler-dependency.sh
#
# Environment: `TORCH_RBLN_REBEL_COMPILER_VERSION`, the version a manual build requests (unset or
# empty: the newest build on the index), `BUILDKITE_BRANCH`, the `UV_INDEX_*` index credentials,
# and `GH_REPO` and `GH_TOKEN` for `gh`.
set -euo pipefail

# shellcheck source=.buildkite/scripts/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/../lib.sh"

package=rebel-compiler
proposal_branch=update-rebel-compiler
base_branch=main
if [[ $# -gt 0 ]]; then
  echo "update-rebel-compiler-dependency.sh takes no arguments: $*" >&2
  exit "${EXIT_USAGE}"
fi
if [[ -z "${GH_REPO:-}" ]]; then
  echo "GH_REPO is not set; check the step env" >&2
  exit "${EXIT_ENVIRONMENT}"
fi
if [[ -z "${BUILDKITE_BRANCH:-}" ]]; then
  echo "BUILDKITE_BRANCH is not set" >&2
  exit "${EXIT_ENVIRONMENT}"
fi
if [[ -z "${GH_TOKEN:-}" ]]; then
  echo "GH_TOKEN is not set; check the step secrets" >&2
  exit "${EXIT_ENVIRONMENT}"
fi
# The proposal commits on top of this build's commit and targets `main`, so a build on another
# branch would carry that branch's commits into it.
if [[ "${BUILDKITE_BRANCH}" != "${base_branch}" ]]; then
  echo "The dependency update runs on ${base_branch}; this build is on '${BUILDKITE_BRANCH}'" >&2
  exit "${EXIT_USAGE}"
fi

echo "--- Resolving ${package} version"
# `read` trims the whitespace that a value pasted into the New Build dialog may carry.
read -r requested <<< "${TORCH_RBLN_REBEL_COMPILER_VERSION:-}"
if [[ -n "${requested}" ]]; then
  target="${requested}"
  echo "Using the requested ${package} version: ${target}"
else
  target="$(resolve_version rbln-nightly "${package}")"
  echo "Newest ${package} on the index: ${target}"
fi

echo "--- Comparing ${target} with current pin"
current="$(
  uv run \
    --no-config \
    tools/pin_rebel_compiler.py \
      --current
)"
echo "Current pin: ${current}"
if [[ "${target}" == "${current}" ]]; then
  echo "Already pinned to that build; nothing to do"
  exit 0
fi
if [[ -z "${requested}" ]]; then
  # A production build carries the local version `prod` (a tagged release) or `g<sha>.prod`.
  production='^(prod|g[0-9a-f]{7,40}\.prod)$'
  if [[ "${target}" != *+* || ! "${target#*+}" =~ ${production} ]]; then
    echo "${target} is not a production build; leaving the pin alone"
    exit 0
  fi
  newer="$(
    uv run \
      --no-config \
      "${SCRIPTS_DIR}/dependency-update/compare-versions.py" \
      "${target}" \
      "${current}"
  )"
  if [[ "${newer}" != true ]]; then
    echo "${target} is not newer than the current pin ${current}; leaving it alone"
    exit 0
  fi
fi
title="chore: update ${package} to ${target}"

echo "--- Pinning ${target} and re-resolving uv.lock"
uv run \
  --no-config \
  tools/pin_rebel_compiler.py \
  "${target}"
uv lock
git config \
  --global \
  --add \
  safe.directory \
  "${PWD}"
if git diff --quiet -- pyproject.toml uv.lock; then
  echo "Pinning ${target} left pyproject.toml and uv.lock unchanged" >&2
  exit "${EXIT_FAILURE}"
fi

echo "--- Committing proposal"
git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
git add pyproject.toml uv.lock
git commit --message "${title}"

echo "--- Pushing proposal to ${proposal_branch}"
# `gh` hands `git` the token as a credential helper, so the token stays out of the remote URL.
gh auth setup-git --hostname github.com
remote="https://github.com/${GH_REPO}.git"
# A branch that already proposes the same change on the same `main` commit keeps its head, because
# a new commit would start every check of its pull request again. `git ls-remote --exit-code`
# exits 2 when the branch does not exist yet.
proposed_tree=""
proposed_parent=""
lookup_status=0
git ls-remote \
  --exit-code \
  "${remote}" \
  "refs/heads/${proposal_branch}" \
  > /dev/null \
  || lookup_status=$?
case "${lookup_status}" in
  0)
    git fetch \
      --quiet \
      --no-tags \
      "${remote}" \
      "refs/heads/${proposal_branch}"
    proposed_tree="$(git rev-parse 'FETCH_HEAD^{tree}')"
    proposed_parent="$(git rev-parse 'FETCH_HEAD^')"
    ;;
  2) ;;
  *)
    echo "Could not look up ${proposal_branch} in ${GH_REPO}" >&2
    exit "${EXIT_ENVIRONMENT}"
    ;;
esac
tree="$(git rev-parse 'HEAD^{tree}')"
parent="$(git rev-parse 'HEAD^')"
if [[ "${proposed_tree}" == "${tree}" && "${proposed_parent}" == "${parent}" ]]; then
  echo "${proposal_branch} already proposes ${target} on this commit; nothing to push"
else
  git push --force "${remote}" "HEAD:refs/heads/${proposal_branch}"
fi

echo "--- Opening or updating pull request"
body="Updates the \`${package}\` dependency to \`${target}\` in"
body+=" \`pyproject.toml\` and \`uv.lock\`."
body+=$'\n\n'
body+="This pull request was opened by the dependency update pipeline. If"
body+=" the new version needs code changes to pass CI, push them to this"
body+=" branch and merge before the pipeline runs again: each run"
body+=" force-pushes a fresh proposal over a branch that differs from it."
number="$(
  gh api \
    --method GET \
    "repos/{owner}/{repo}/pulls" \
    --field "head={owner}:${proposal_branch}" \
    --field "base=${base_branch}" \
    --field state=open \
    --jq '.[0].number // empty'
)"
if [[ -n "${number}" ]]; then
  gh pr edit \
    "${number}" \
    --title "${title}" \
    --body "${body}"
  echo "Updated pull request #${number}"
else
  gh pr create \
    --head "${proposal_branch}" \
    --base "${base_branch}" \
    --title "${title}" \
    --body "${body}"
fi
