#!/usr/bin/env bash
# Propose a `torch-rbln` pin update to a repository that pins it, as a pull
# request.
#
# Usage:
#   update-torch-rbln-pins.sh vllm-rbln|rebel-compiler
#
# The version is TORCH_RBLN_PIN_VERSION, which must be a release on pypi.org,
# or the latest release there when it is empty.
set -euo pipefail

package="torch-rbln"
head_branch="update-torch-rbln"
base_branch="dev"

if [[ $# -ne 1 ]]; then
  echo "update-torch-rbln-pins.sh takes one argument, the repository: $*" >&2
  exit 2
fi
target_repo="$1"

: "${GH_REPO:?is not set}"
: "${GH_TOKEN:?is not set}"

# pin_pattern captures the text before the pin, the pin, and the text after it.
case "${target_repo}" in
  vllm-rbln)
    # The range in pyproject.toml admits a whole patch line, and uv.lock pins
    # the version in it.
    pin_file="pyproject.toml"
    # "torch-rbln>=0.12.1.dev0,<0.12.2 ; platform_machine == 'x86_64'"
    pin_pattern='("torch-rbln)([<>=!~][^ ;"]*)()'
    proposed_files=("${pin_file}" uv.lock)
    title_prefix="other(deps): update ${package} to "
    ;;
  rebel-compiler)
    pin_file=".buildkite/pr.yml"
    # TORCH_RBLN: "0.12.1a1"
    pin_pattern='^(  TORCH_RBLN: ")([^"]+)(")$'
    proposed_files=("${pin_file}")
    title_prefix="update(ci): update TORCH_RBLN version to "
    ;;
  *)
    echo "Unknown repository: ${target_repo}" >&2
    exit 2
    ;;
esac

echo "--- Resolving ${package} version"
# A release of vllm-rbln requires the pinned version from pypi.org, so the pin
# is always a release there.
release_options=(--package "${package}")
if [[ -n "${TORCH_RBLN_PIN_VERSION:-}" ]]; then
  .buildkite/scripts/check_pin_version.py \
    --version "${TORCH_RBLN_PIN_VERSION}"
  release_options+=(--version "${TORCH_RBLN_PIN_VERSION}")
fi
target="$(
  .buildkite/scripts/pypi_release.py \
    "${release_options[@]}"
)"
echo "Resolved ${package} to ${target}"

echo "--- Cloning ${GH_REPO} at ${base_branch}"
checkout_dir="$(mktemp --directory)/${target_repo}"
auth_header="AUTHORIZATION: basic $(printf 'x-access-token:%s' "${GH_TOKEN}" | base64 --wrap=0)"
clone_options=(--depth 1 --branch "${base_branch}")
if [[ "${target_repo}" == "rebel-compiler" ]]; then
  # The repository is large and only one file changes.
  clone_options+=(--filter blob:none --sparse)
fi
git \
  -c "http.https://github.com/.extraheader=${auth_header}" \
  clone \
  "${clone_options[@]}" \
  "https://github.com/${GH_REPO}.git" \
  "${checkout_dir}"
if [[ "${target_repo}" == "rebel-compiler" ]]; then
  git \
    -C "${checkout_dir}" \
    -c "http.https://github.com/.extraheader=${auth_header}" \
    sparse-checkout set \
      --no-cone \
      "/${pin_file}"
fi

echo "--- Comparing ${target} with current pin"
if [[ "$(grep --count --extended-regexp "${pin_pattern}" "${checkout_dir}/${pin_file}")" != "1" ]]; then
  echo "${pin_file} does not pin ${package} exactly once" >&2
  exit 1
fi
if [[ "${target_repo}" == "vllm-rbln" ]]; then
  current="$(
    .buildkite/scripts/project_pin.py \
      --project "${checkout_dir}" \
      --package "${package}" \
      locked
  )"
else
  current="$(
    grep \
      --only-matching \
      --extended-regexp \
      "${pin_pattern}" \
      "${checkout_dir}/${pin_file}" \
      | sed \
        --regexp-extended \
        "s/${pin_pattern}/\2/"
  )"
fi
echo "Current pin: ${current}"
if [[ "${target}" == "${current}" ]]; then
  echo "Already pinned to ${target}; nothing to do"
  exit 0
fi
is_later="$(
  .buildkite/scripts/compare_versions.py \
    --candidate "${target}" \
    --baseline "${current}"
)"
if [[ "${is_later}" != "true" ]]; then
  echo "${target} is not newer than the current pin ${current}; nothing to do"
  exit 0
fi

echo "--- Pinning ${package} to ${target}"
pin="${target}"
if [[ "${target_repo}" == "vllm-rbln" ]]; then
  pin=""
  admitted="$(
    .buildkite/scripts/project_pin.py \
      --project "${checkout_dir}" \
      --package "${package}" \
      allows "${target}"
  )"
  if [[ "${admitted}" != "true" ]]; then
    # A new patch line: the range moves to it, and stays until the next one.
    pin="$(
      .buildkite/scripts/project_pin.py \
        --project "${checkout_dir}" \
        --package "${package}" \
        patch-line "${target}"
    )"
  fi
fi
if [[ -n "${pin}" ]]; then
  sed \
    --in-place \
    --regexp-extended \
    "s/${pin_pattern}/\1${pin}\3/" \
    "${checkout_dir}/${pin_file}"
fi
if [[ "${target_repo}" == "vllm-rbln" ]]; then
  # Only torch-rbln and what its new version needs may move. The range also
  # admits the development builds on the nightly index, so name the version.
  (
    cd "${checkout_dir}"
    uv lock \
      --upgrade-package "${package}==${target}"
  )
fi
git \
  -C "${checkout_dir}" \
  --no-pager \
  diff

echo "--- Proposing ${target}"
file_options=()
for file in "${proposed_files[@]}"; do
  file_options+=(--file "${file}")
done
.buildkite/scripts/propose_dependency_update.py \
  --package "${package}" \
  --package-version "${target}" \
  --head-branch "${head_branch}" \
  --base-branch "${base_branch}" \
  --repo-dir "${checkout_dir}" \
  "${file_options[@]}" \
  --title "${title_prefix}${target}"
