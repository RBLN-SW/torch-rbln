#!/usr/bin/env -S uv run --isolated --only-group ci --locked python
"""Tag the build's commit with TORCH_RBLN_VERSION_TAG.

The script also creates a GitHub Release for a final release or its
post-release.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from typing import TYPE_CHECKING

from github import Auth, Github, GithubException, UnknownObjectException
from github.Consts import DEFAULT_BASE_URL
from github.GithubObject import NotSet
from packaging.version import InvalidVersion, Version
from requests import RequestException


if TYPE_CHECKING:
    from collections.abc import Iterable

    from github.Repository import Repository


# PyGithub does not wrap the exceptions of requests, its HTTP library.
GITHUB_ERRORS = (GithubException, RequestException)


def repository() -> Repository:
    github = Github(
        auth=Auth.Token(os.environ["GH_TOKEN"]), base_url=os.environ.get("GITHUB_API_URL", DEFAULT_BASE_URL)
    )
    return github.get_repo(os.environ["GH_REPO"])


def parse_tag(tag: str) -> Version | None:
    try:
        version = Version(tag.removeprefix("v"))
    except InvalidVersion:
        return None
    return version if tag.startswith("v") and str(version) == tag[1:] else None


def shipped_version(tag: str) -> Version | None:
    version = parse_tag(tag)
    if version is None or version.is_prerelease or version.local:
        return None
    return version


def tag_commit(repo: Repository, tag: str) -> str | None:
    try:
        ref = repo.get_git_ref(f"tags/{tag}")
    except UnknownObjectException:
        return None
    if ref.object.type == "tag":
        return repo.get_git_tag(ref.object.sha).object.sha
    return ref.object.sha


def notes_baseline(tag: str, tags: Iterable[str]) -> str | None:
    current = Version(tag.removeprefix("v"))
    below = [(version, name) for name in tags if (version := shipped_version(name)) is not None and version < current]
    return max(below)[1] if below else None


def has_published_release(repo: Repository, tag: str) -> bool:
    try:
        repo.get_release(tag)
    except UnknownObjectException:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    tag = os.environ.get("TORCH_RBLN_VERSION_TAG", "")
    if not tag:
        parser.error("TORCH_RBLN_VERSION_TAG is required")
    version = parse_tag(tag)
    if version is None:
        parser.error(f"'{tag}' is not an accepted tag: expected v and a PEP 440 version in its normalized form")
    if version.dev not in (None, 0):
        parser.error(f"'{tag}' is not an accepted tag: setuptools-scm derives later versions only from a .dev0 tag")
    # The wheel steps run only when the value is true, so a typo would
    # tag the commit without wheels.
    release_wheels = os.environ.get("TORCH_RBLN_RELEASE_WHEELS") or "true"
    if release_wheels not in ("true", "false"):
        parser.error(f"TORCH_RBLN_RELEASE_WHEELS is '{release_wheels}'; expected true or false")
    for name in ("GH_REPO", "GH_TOKEN"):
        if not os.environ.get(name):
            raise SystemExit(f"{name} is not set")

    # The release uses the checked-out commit because BUILDKITE_COMMIT
    # can be a symbolic name such as HEAD.
    head_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE, text=True, check=True
    ).stdout.strip()
    try:
        repo = repository()
        tagged_sha = tag_commit(repo, tag)
        if tagged_sha is None:
            repo.create_git_ref(f"refs/tags/{tag}", head_sha)
    except GITHUB_ERRORS as error:
        raise SystemExit(f"Could not read or create the tag {tag} in {os.environ['GH_REPO']}: {error}") from None
    if tagged_sha is None:
        print(f"Created tag {tag} at {head_sha}")
    elif tagged_sha == head_sha:
        print(f"Tag {tag} already points at {head_sha}; skipping tag creation")
    else:
        raise SystemExit(f"Tag {tag} already exists at {tagged_sha}, not at {head_sha}; pick a new version")

    if shipped_version(tag) is None:
        print(
            f"Skipping the GitHub Release for {tag}: only a final release, or a"
            " post-release of one, without a local version label has one"
        )
        return
    try:
        release_exists = has_published_release(repo, tag)
    except GITHUB_ERRORS as error:
        raise SystemExit(f"Could not look up the GitHub Release for {tag}: {error}") from None
    if release_exists:
        print(f"GitHub Release {tag} already exists; nothing to do")
        return
    try:
        tags = [ref.ref.removeprefix("refs/tags/") for ref in repo.get_git_matching_refs("tags/v")]
    except GITHUB_ERRORS as error:
        raise SystemExit(f"Could not list the tags in {os.environ['GH_REPO']}: {error}") from None
    baseline = notes_baseline(tag, tags)
    print(f"Release notes baseline: {baseline or '<none>'}")

    try:
        # create_git_release() takes no previous_tag_name, so the notes
        # are generated first.
        notes = repo.generate_release_notes(tag, previous_tag_name=baseline or NotSet)
        repo.create_git_release(tag, name=tag, message=notes.body)
    except GITHUB_ERRORS as error:
        raise SystemExit(f"Could not create the GitHub Release for {tag}: {error}") from None
    print(f"Created GitHub Release {tag}")


if __name__ == "__main__":
    main()
