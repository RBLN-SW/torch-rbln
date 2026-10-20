#!/usr/bin/env -S uv run --isolated --only-group ci --locked python
"""Check that the pull request title follows Conventional Commits."""

from __future__ import annotations

import argparse
import os
import re
from typing import TYPE_CHECKING

from github import Auth, Github, GithubException
from github.Consts import DEFAULT_BASE_URL
from requests import RequestException


if TYPE_CHECKING:
    from github.Repository import Repository


# PyGithub does not wrap the exceptions of requests, its HTTP library.
GITHUB_ERRORS = (GithubException, RequestException)
COMMIT_TYPES = ("feat", "fix", "refactor", "perf", "docs", "test", "build", "ci", "chore")
TITLE_PATTERN = re.compile(rf"({'|'.join(COMMIT_TYPES)})(\([^()]+\))?!?: [a-z](.*[^.])?")
TITLE_RULES = f"""\
  format: `<type>[optional scope]: <description>`
  type: {", ".join(f"`{name}`" for name in COMMIT_TYPES)}
  description: must start with a lowercase letter and must not end with a period"""


def repository() -> Repository:
    github = Github(
        auth=Auth.Token(os.environ["GH_TOKEN"]), base_url=os.environ.get("GITHUB_API_URL", DEFAULT_BASE_URL)
    )
    return github.get_repo(os.environ["GH_REPO"])


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    for name in ("GH_REPO", "GH_TOKEN"):
        if not os.environ.get(name):
            raise SystemExit(f"{name} is not set")
    pull_request = os.environ.get("BUILDKITE_PULL_REQUEST", "")
    try:
        pull_request_number = int(pull_request)
    except ValueError:
        raise SystemExit(f"BUILDKITE_PULL_REQUEST is '{pull_request}', not a pull request number") from None
    try:
        title = repository().get_pull(pull_request_number).title
    except GITHUB_ERRORS as error:
        raise SystemExit(
            f"Could not read pull request #{pull_request_number} in {os.environ['GH_REPO']}: {error}"
        ) from None
    print(f"Pull request #{pull_request_number}: {title}")
    if not TITLE_PATTERN.fullmatch(title):
        raise SystemExit(
            f'"{title}" is not a Conventional Commits title\n{TITLE_RULES}\nEdit the title, then retry this job'
        )


if __name__ == "__main__":
    main()
