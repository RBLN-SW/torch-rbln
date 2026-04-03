# CI/CD Workflows

This document describes the GitHub Actions workflows that power the automated testing and deployment pipeline for `torch-rbln`. All workflow definitions live in [`.github/workflows/`](../.github/workflows/).

## Overview

| Workflow                               | Trigger                       | Purpose                             | Test Scope                                                         |
|----------------------------------------|-------------------------------|-------------------------------------|--------------------------------------------------------------------|
| CI (`ci.yaml`)                         | PRs and pushes to `dev`       | Fast feedback on every change       | Linting + `test_set_ci`-marked tests                               |
| Release (`release.yaml`)               | PRs and pushes to `main`      | Pre-release validation              | Linting + all tests except `test_set_experimental`/`test_set_perf` |
| CD (`cd.yaml`)                         | Version tags (`v*`)           | Build and publish release artifacts | Deployment pipeline                                                |
| Check PR Title (`check_pr_title.yaml`) | PR opened, edited, or updated | Enforce Conventional Commits format | —                                                                  |

CI, Release, and CD workflows delegate to a shared [Event Dispatch](#event-dispatch-mechanism) mechanism that sends events to infrastructure with physical RBLN NPU devices.

---

## Triggers and Concurrency

| Workflow       | `pull_request`                  | `push`             | `workflow_dispatch`  | Cancel in-progress?              |
|----------------|---------------------------------|--------------------|----------------------|----------------------------------|
| CI             | To any branch **except** `main` | To `dev`           | Yes (by ref and SHA) | Yes (except `workflow_dispatch`) |
| Release        | To `main`                       | To `main`          | Yes (by ref and SHA) | Yes (except `workflow_dispatch`) |
| CD             | —                               | Tags matching `v*` | Yes (by ref and SHA) | **No**                           |
| Check PR Title | On open, edit, sync, reopen     | —                  | —                    | Yes                              |

CI and Release runs are grouped by PR number (or SHA for pushes); a new push cancels the in-progress run. Manually triggered runs (`workflow_dispatch`) are never cancelled. CD runs are also never cancelled — once a deployment starts, it runs to completion. Check PR Title runs are grouped by PR number and always cancel the in-progress run.

---

## CI Workflow

**File:** [`.github/workflows/ci.yaml`](../.github/workflows/ci.yaml)

The CI workflow provides fast feedback during everyday development and is the gatekeeper for merging into `dev`. It runs linting and the `test_set_ci`-marked test suite:

```bash
python test/run_tests.py  # -m "test_set_ci"
```

This selects tests marked with `@pytest.mark.test_set_ci` — the core set of tests that should always pass. Linting runs via `lintrunner` (see [Linting](LINTING.md)) to catch style violations before tests execute. See the [Test Guide](TEST_GUIDE.md) for details on test markers and parallel/serial worker splitting.

---

## Release Workflow

**File:** [`.github/workflows/release.yaml`](../.github/workflows/release.yaml)

The Release workflow runs when code is promoted from `dev` to `main` for release. It covers a broader test suite than CI, but neither is a strict superset of the other — experimental tests can run in CI but are excluded from Release:

```bash
python test/run_tests.py --test_mode=release  # -m "not (test_set_experimental or test_set_perf)"
```

### Test Coverage by Workflow

| Test Mode          | Marker                               | pytest Expression                                   | Workflow         | Description                                                                 |
|--------------------|--------------------------------------|-----------------------------------------------------|------------------|-----------------------------------------------------------------------------|
| CI tests           | `@pytest.mark.test_set_ci`           | `-m "test_set_ci"`                                  | CI               | Core tests that run on every PR — must always pass                          |
| Release tests      | *(no marker)*                        | `-m "not (test_set_experimental or test_set_perf)"` | Release          | Extended coverage included at release time — too slow or niche for every PR |
| Performance tests  | `@pytest.mark.test_set_perf`         | `-m "test_set_perf"`                                | *(manual)*       | Benchmarks — not included in any automated workflow                         |
| Experimental tests | `@pytest.mark.test_set_experimental` | excluded from Release                               | CI or *(manual)* | Early-stage features — can opt into CI with `@pytest.mark.test_set_ci`      |

> **Note:**
> - **Marker overlap:** CI mode (`-m "test_set_ci"`) and Release mode (`-m "not (test_set_experimental or test_set_perf)"`) overlap but neither is a strict superset of the other. Release includes all `test_set_ci`-marked tests *plus* unmarked tests, but excludes `test_set_experimental`. A test marked with both `@pytest.mark.test_set_ci` and `@pytest.mark.test_set_experimental` will run in CI but **not** in Release.
> - **Linting:** `lintrunner` runs in both CI and Release workflows.

---

## CD Workflow

**File:** [`.github/workflows/cd.yaml`](../.github/workflows/cd.yaml)

The CD workflow builds and publishes release artifacts after code has passed both CI and Release testing. It dispatches a `torch-rbln-cd` event and focuses on artifact generation and deployment rather than test execution.

---

## Check PR Title Workflow

**File:** [`.github/workflows/check_pr_title.yaml`](../.github/workflows/check_pr_title.yaml)

The Check PR Title workflow enforces the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) format on all pull request titles. It triggers whenever a PR is opened, edited, synchronized, or reopened.

The workflow uses [`amannn/action-semantic-pull-request`](https://github.com/amannn/action-semantic-pull-request) to validate the title against the following format:

```
<type>(<optional scope>): <description>
```

**Allowed types:** `feat`, `fix`, `refactor`, `perf`, `docs`, `test`, `build`, `ci`, `chore`

If the title is invalid, the workflow posts a sticky comment on the PR explaining the required format using [`marocchino/sticky-pull-request-comment`](https://github.com/marocchino/sticky-pull-request-comment). The comment is automatically deleted once the title is corrected.

---

## Event Dispatch Mechanism

**File:** [`.github/workflows/dispatch_event.yaml`](../.github/workflows/dispatch_event.yaml)

CI, Release, and CD workflows delegate to an internal hardware-backed automation flow via GitHub [repository dispatch](https://docs.github.com/en/actions/writing-workflows/choosing-when-your-workflow-runs/events-that-trigger-workflows#repository_dispatch).
This is necessary because testing `torch-rbln` requires access to physical RBLN NPU hardware hosted on dedicated infrastructure.

The dispatch payload includes:

| Field            | Description                                                                    |
|------------------|--------------------------------------------------------------------------------|
| `event_name`     | GitHub event name that triggered the workflow (`push`, `pull_request`, etc.)   |
| `event_type`     | Dispatch type: `torch-rbln-ci`, `torch-rbln-release`, or `torch-rbln-cd`       |
| `torch_rbln_ref` | Git reference (branch name or tag, e.g. `refs/heads/main`, `refs/tags/v1.0.0`) |
| `torch_rbln_sha` | Git commit SHA for the exact revision to build and test                        |

The event is dispatched to a separate repository (configured via `vars.TORCH_RBLN_DISPATCH_REPOSITORY`) using [`peter-evans/repository-dispatch`](https://github.com/peter-evans/repository-dispatch), which triggers the corresponding workflow on infrastructure with RBLN NPU devices.

---

## Workflow Lifecycle: From PR to Release

```
                  (CI)              (Release)          (CI)
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

### Branch roles

| Branch    | Lifecycle                                                    |
|-----------|--------------------------------------------------------------|
| `feature` | Branch from `dev` → PR to `dev`                              |
| `dev`     | Long-lived. Everyday integration point for all feature work. |
| `rc`      | Branch from `dev`. Standing PR to `main`                     |
| `hotfix`  | Branch from `main` → PR to `main`                            |
| `main`    | Long-lived. Always release-ready. Tags are created here.     |

### Flow summary

1. **Feature development:** Developers branch from `dev`, open a PR back to `dev`. The CI workflow validates every PR with linting and `test_set_ci`-marked tests.
2. **Integration:** Merging to `dev` triggers a CI push build, continuously verifying integration stability.
3. **Release candidate:** A nightly job creates an `rc` branch from `dev` and maintains a standing PR to `main`. The Release workflow runs the full test suite on this PR daily.
4. **Release:** When ready, the release manager merges the `rc` PR into `main`. Tagging `main` triggers the CD workflow for artifact build and deployment.
5. **Backmerge:** After a `main` merge, a backmerge PR is automatically created from `main` to `dev`, ensuring hotfixes and release-only changes flow back.
6. **Hotfix:** For urgent fixes, branch from `main`, PR back to `main`. The backmerge mechanism propagates the fix to `dev`.

---

## Related Documentation

- [Contributing Guide](CONTRIBUTING.md) — PR requirements and merge policy
- [Test Guide](TEST_GUIDE.md) — Test infrastructure, markers, and `run_tests.py` usage
- [Linting](LINTING.md) — Code style and pre-commit hooks
