# Release Process

This document describes how torch-rbln releases are prepared, validated, and published. For CI/CD workflow internals, see [Workflows](WORKFLOWS.md); for PR and merge policies, see [Contributing Guide](CONTRIBUTING.md).

## Overview

Every release follows the same pipeline: changes integrate on `dev`, a nightly sync updates `rc` with the latest `dev` state so a current `rc → main` pull request is always ready to merge, and a version tag on `main` triggers artifact build and publication. Because the release candidate stays current, promotions to `main` happen on a steady cadence rather than as large, infrequent merges.

```
         feature PRs        nightly sync       RC PR         version tag
feature ─────────────► dev ──────────────► rc ───────► main ─────────────► publish
                        ▲                                │
                        └────────── backmerge ───────────┘
```

## Branch Model

The release pipeline is built on five kinds of branches, each with a specific lifecycle:

| Branch    | Lifecycle                                                    |
|-----------|--------------------------------------------------------------|
| `feature` | Branch from `dev` → PR to `dev`                              |
| `dev`     | Long-lived. Everyday integration point for all feature work. |
| `rc`      | Branch from `dev` → PR to `main`                             |
| `hotfix`  | Branch from `main` → PR to `main`                            |
| `main`    | Long-lived. Always release-ready. Tags are created here.     |

The diagram below shows which workflow runs at each branch transition. The `(CI)`, `(Release)`, and `(CD)` labels refer to the workflows described in [Workflows](WORKFLOWS.md):

```
                  (CI)             (Release)           (CI)
                PR to dev          PR to main        PR to dev
                   │                   │                │
feature        ○───●                   │                │
              ╱     ╲                  │                │     (CI)
             ╱       ╲                 │                │  push to dev
dev     ────○─────────●────────○───────┼────────────────┼────●─────────────────
                push to dev     ╲      │                │   ╱
                    (CI)         ╲     │                │  ╱
rc                                ○────●                │ ╱
                (Release)               ╲               │╱
                PR to main               ╲         ●────●  backmerge
                    │                     ╲       ╱
hotfix         ○────●                      ╲     ╱
              ╱      ╲                      ╲   ╱
             ╱        ╲                      ╲ ╱
main    ────○──────────●──────────────────────●───────────────────────────●────
                  push to main           push to main                     │
                   (Release)              (Release)                      tag
                                                                         (CD)
```

## Branch and Tag Protection

The branches and tags above are protected by [GitHub repository rulesets](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets), so changes reach them through pull requests and the release automation rather than direct pushes. This keeps releases reproducible and ensures only CI-validated changes are promoted. Force pushes are blocked on every protected ref, and only the actors noted below can create or delete them.

| Ref(s)            | How changes are made    |
|-------------------|-------------------------|
| `dev`, `main`     | Pull requests only      |
| `rc`, `backmerge` | Release automation only |
| `v*` tags         | Release automation only |

**`dev` and `main`** accept changes only through pull requests, which must pass required reviews and status checks — the CI workflow for `dev`, the Release workflow for `main`. Repository admins can bypass these merge requirements when needed (for example, to land an urgent fix), but no one — admins included — can push to these branches directly.

**`rc` and `backmerge`** are maintained entirely by the release automation: the nightly job syncs `rc` from `dev`, and the backmerge automation updates `backmerge` from `main`. No one else can update them, so contributors never push to these branches.

**`v*` tags** are created only by the release automation, and pushing a `v*` tag triggers the CD workflow. No one else can create, move, or delete a `v*` tag, so every published version corresponds to a tag created on a validated `main` commit.

### Merge methods

The merge method depends on the branch pair:

| Pull request      | Merge method          |
|-------------------|-----------------------|
| `feature → dev`   | Squash and merge      |
| `rc → main`       | Create a merge commit |
| `backmerge → dev` | Create a merge commit |

`rc → main` and `backmerge → dev` use a merge commit so the release commit on `main` and its version tag stay reachable from `dev`, which setuptools-scm relies on to derive correct development versions. Every other pull request is squash-merged.

## Versioning

torch-rbln releases align with the [RBLN SDK](https://docs.rbln.ai/) — each release version matches the corresponding SDK version.

Versions are derived from git tags using [setuptools-scm](https://setuptools-scm.readthedocs.io/). Tags use the format `v<major>.<minor>.<patch>[rc<N>]`, where the optional `rc<N>` suffix marks a release candidate build.

| Source                 | Example Tag      | Resulting Version      |
|------------------------|------------------|------------------------|
| On a release tag       | `v0.10.0`        | `0.10.0`               |
| On a release candidate | `v0.10.0rc0`     | `0.10.0rc0`            |
| 5 commits past a tag   | `v0.10.0rc0` + 5 | `0.10.1.dev5+g1a2b3c4` |

Development builds between tags carry a `.devN+g<commit>` suffix so they sort below the next release.

## Release Lifecycle

### 1. Integration on `dev`

All changes land on `dev` through pull requests. The CI workflow validates each PR with linting and `test_set_ci`-marked tests. See [Contributing Guide](CONTRIBUTING.md) for PR requirements and merge policy.

### 2. Nightly sync to `rc`

A nightly job syncs `rc` with the latest `dev` and keeps an open pull request from `rc` to `main`. Because the sync runs every night, the `rc → main` PR stays continuously up to date, giving the release manager a fresh release candidate to merge on a regular schedule. The Release workflow runs the full test suite on this PR whenever `rc` changes, surfacing regressions before they reach `main`.

### 3. Promotion to `main`

When the release candidate is ready, the release manager merges the `rc → main` pull request. The Release workflow runs again on `main`, building and linting across all supported Python versions.

### 4. Tagging and publication

The release manager creates a `v<version>` tag on `main` — either through the [GitHub Releases UI](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository) or the command line:

```bash
git tag v<version>  # e.g. git tag v0.10.0
git push origin v<version>
```

The tag triggers the CD workflow, which builds wheels for all supported Python versions and publishes them to the internal package index first, then to public PyPI.

### 5. Release notes

The release manager writes release notes for the new version on the [GitHub Releases](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository) page. If step 4 used the GitHub Releases UI, the notes go in alongside the tag; otherwise, they are added afterward.

### 6. Backmerge to `dev`

Every merge into `main` — whether from `rc` or a hotfix — triggers an automated backmerge pull request from `main` back to `dev`, keeping `dev` in sync with `main`.

## Debug Builds

Debug wheels — built against a debug variant of PyTorch — are produced alongside release wheels during release validation and nightly runs. They surface issues that only manifest under debug-mode assertions, catching bugs that would otherwise ship unnoticed.

Debug wheels are uploaded to the internal package index for testing, using a `.debug` or `+debug` version suffix (for example, `0.10.0+debug`) to distinguish them from release wheels. They are **not** published to public PyPI.

## Hotfix

When a critical issue in a released version cannot wait for the next release cycle:

1. Branch from `main`.
2. Open a pull request back to `main` — the Release workflow validates the change.
3. Merge, then create a new patch-version tag on `main` (e.g. `v0.10.0` → `v0.10.1`) following the tagging and publication steps above.
4. An automated backmerge PR propagates the fix back to `dev`.

## Related Documentation

- [Workflows](WORKFLOWS.md) — CI/CD workflow details, triggers, and concurrency
- [Contributing Guide](CONTRIBUTING.md) — PR requirements and merge policy
- [Third-Party Update](THIRD_PARTY_UPDATE.md) — Dependency versioning (PyTorch, rebel-compiler)
