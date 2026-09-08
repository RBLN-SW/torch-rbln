# Release Process

This document describes how PyTorch RBLN releases are prepared, validated, and published. For CI/CD workflow internals, see [Workflows](WORKFLOWS.md); for pull request and merge policies, see [Contributing Guide](CONTRIBUTING.md).

## Overview

PyTorch RBLN maintains a single long-lived branch, `main`. Every change reaches it through a squash-merged pull request. Release checks validate `main` daily against the release test suite, and a version tag on a validated commit triggers artifact build and publication.

```text
                   ┌─ pre-merge checks
                   │
feature        ●───●
              ╱     ╲ squash merge
main  ───────●───────●───────────────────────●───►
                     │                       │
                     └─ post-merge checks    ├─ release checks
                                             └─ version tag ───► publish
```

## Branch and Tag Protection

Separate [GitHub repository rulesets](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets) apply to the `main` branch and to the version tags.

`main` accepts changes only through pull requests, which must have the required approvals and passing status checks. Repository admins can bypass the merge requirements, but not even they can push to `main` directly.

Version tags are created by the release automation. The ruleset prevents them from being moved or deleted, so a published version always points at the same commit.

## Versioning

Each PyTorch RBLN release version matches the corresponding [RBLN SDK](https://docs.rbln.ai/) version.

Versions are derived from git tags using [setuptools-scm](https://setuptools-scm.readthedocs.io/). Tags use the format `v<major>.<minor>.<patch>` with an optional `rc<N>` (release candidate) or `.post<N>` (post-release) suffix:

| Source                     | Example tag         | Resulting version            |
|----------------------------|---------------------|------------------------------|
| On a release candidate     | `v0.10.0rc0`        | `0.10.0rc0`                  |
| On a release tag           | `v0.10.0`           | `0.10.0`                     |
| On a post-release          | `v0.10.0.post0`     | `0.10.0.post0`               |
| 5 commits past a candidate | `v0.10.0rc0` + 5    | `0.10.0rc1.dev5+g1a2b3c4`    |
| 5 commits past a release   | `v0.10.0` + 5       | `0.10.1.dev5+g1a2b3c4`       |
| 5 commits past a post      | `v0.10.0.post0` + 5 | `0.10.0.post1.dev5+g1a2b3c4` |

Development builds between tags carry a `.devN+g<sha>` suffix and sort below the release they anticipate.

## Release Lifecycle

### 1. Integration on `main`

Pre-merge checks validate each pull request, and post-merge checks repeat that validation on the resulting `main` commit, giving it a test result of its own.

`main` is expected to stay release-ready. When a post-merge run fails, the change is reverted or fixed forward rather than left for the next release.

### 2. Release validation

Release checks run the release test suite against the current `main` commit and record the outcome there. They run daily and can also be started manually.

### 3. Tagging and publication

The release manager runs the release automation to tag the validated commit. The tag triggers the CD workflow, which builds wheels for all supported Python versions and publishes them to the internal package index first, then to public PyPI.

### 4. Release notes

The release automation creates a [GitHub Release](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository) for the tag, with auto-generated notes covering the commits since the previous final release. It skips release candidates.

## Debug Builds

Debug wheels — built against a debug variant of PyTorch — are produced alongside release wheels during release checks. They surface issues that only manifest under debug-mode assertions.

Debug wheels are uploaded to the internal package index with a `.debug` or `+debug` version suffix (for example, `0.10.0+debug`). They are **not** published to public PyPI.

## Hotfix

When a critical issue in a released version cannot wait for the next release, a hotfix takes the ordinary path: a pull request to `main`. After it merges, the release manager starts release checks on `main` and, once they pass, cuts a patch release (e.g. `v0.10.0` → `v0.10.1`) following the tagging and publication steps above.

## Related Documentation

- [Workflows](WORKFLOWS.md) — CI/CD workflow details, triggers, and concurrency
- [Contributing Guide](CONTRIBUTING.md) — PR requirements and merge policy
- [Third-Party Update](THIRD_PARTY_UPDATE.md) — Dependency versioning (PyTorch, rebel-compiler)
