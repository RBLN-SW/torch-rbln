#!/usr/bin/env -S uv run --isolated --only-group ci --locked python
"""Propose pyproject.toml and uv.lock as a pull request."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from github import Auth, Github, GithubException, InputGitAuthor, InputGitTreeElement, UnknownObjectException
from github.Consts import DEFAULT_BASE_URL
from requests import RequestException


if TYPE_CHECKING:
    from github.GitCommit import GitCommit
    from github.GitRef import GitRef
    from github.GitTree import GitTree
    from github.Repository import Repository


# PyGithub does not wrap the exceptions of requests, its HTTP library.
GITHUB_ERRORS = (GithubException, RequestException)
PROPOSED_FILES = ("pyproject.toml", "uv.lock")
BOT_AUTHOR = InputGitAuthor("github-actions[bot]", "41898282+github-actions[bot]@users.noreply.github.com")


def repository() -> Repository:
    github = Github(
        auth=Auth.Token(os.environ["GH_TOKEN"]), base_url=os.environ.get("GITHUB_API_URL", DEFAULT_BASE_URL)
    )
    return github.get_repo(os.environ["GH_REPO"])


def branch_ref(repo: Repository, branch: str) -> GitRef | None:
    try:
        return repo.get_git_ref(f"heads/{branch}")
    except UnknownObjectException:
        return None


def proposal_tree(repo: Repository, build_commit: GitCommit) -> GitTree:
    return repo.create_git_tree(
        [
            InputGitTreeElement(path, "100644", "blob", content=Path(path).read_text(encoding="utf-8"))
            for path in PROPOSED_FILES
        ],
        base_tree=build_commit.tree,
    )


def commit_proposal(repo: Repository, branch: str, title: str, build_commit: GitCommit, tree: GitTree) -> str | None:
    ref = branch_ref(repo, branch)
    if ref is not None:
        branch_head = repo.get_git_commit(ref.object.sha)
        if branch_head.tree.sha == tree.sha and [parent.sha for parent in branch_head.parents] == [build_commit.sha]:
            return None
    commit = repo.create_git_commit(title, tree, [build_commit], author=BOT_AUTHOR, committer=BOT_AUTHOR)
    if ref is None:
        repo.create_git_ref(f"refs/heads/{branch}", commit.sha)
    else:
        ref.edit(commit.sha, force=True)
    return commit.sha


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--package", required=True)
    parser.add_argument("--package-version", required=True)
    parser.add_argument("--head-branch", required=True)
    parser.add_argument("--base-branch", required=True)
    args = parser.parse_args()
    for name in ("GH_REPO", "GH_TOKEN"):
        if not os.environ.get(name):
            msg = f"{name} is not set"
            raise SystemExit(msg)
    title = f"chore: update {args.package} to {args.package_version}"
    body = (
        f"Updates the `{args.package}` dependency to `{args.package_version}`.\n\n"
        "The next update force-pushes this branch and drops any commit pushed to it, "
        "so merge a fix pushed here before then."
    )
    build_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],  # noqa: S607
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    ).stdout.strip()

    try:
        repo = repository()
        build_commit = repo.get_git_commit(build_sha)
        tree = proposal_tree(repo, build_commit)
        commit = commit_proposal(repo, args.head_branch, title, build_commit, tree)
    except GITHUB_ERRORS as error:
        msg = f"Could not propose {args.package_version} on {args.head_branch} in {os.environ['GH_REPO']}: {error}"
        raise SystemExit(msg) from None
    if commit is None:
        print(f"{args.head_branch} already proposes {args.package_version} on {build_sha}; nothing to push")
    else:
        print(f"Pushed {commit} to {args.head_branch}")

    try:
        pull = next(
            iter(repo.get_pulls(state="open", head=f"{repo.owner.login}:{args.head_branch}", base=args.base_branch)),
            None,
        )
        if pull is None:
            pull = repo.create_pull(base=args.base_branch, head=args.head_branch, title=title, body=body)
            print(f"Opened pull request #{pull.number}")
        else:
            pull.edit(title=title, body=body)
            print(f"Updated pull request #{pull.number}")
    except GITHUB_ERRORS as error:
        msg = f"Could not open or update the pull request from {args.head_branch}: {error}"
        raise SystemExit(msg) from None


if __name__ == "__main__":
    main()
