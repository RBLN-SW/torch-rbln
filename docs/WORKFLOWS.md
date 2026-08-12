# CI/CD Workflows

This document describes the GitHub Actions workflows that power the automated testing and deployment pipeline for `torch-rbln`. All workflow definitions live in [`.github/workflows/`](../.github/workflows/).

## Overview

| Workflow                               | Trigger                       | Purpose                             | Test Scope                                                         |
|----------------------------------------|-------------------------------|-------------------------------------|--------------------------------------------------------------------|
| CI (`ci.yaml`)                         | PRs and pushes to `dev`       | Fast feedback on every change       | Linting + `test_set_ci`-marked tests                               |
| Release (`release.yaml`)               | PRs and pushes to `main`      | Pre-release validation              | Linting + all tests except `test_set_experimental`/`test_set_perf` |
| CD (`cd.yaml`)                         | Version tags (`v*`)           | Build and publish release artifacts | Deployment pipeline                                                |
| Build (`build.yaml`)                   | PRs; manual dispatch          | Build and publish the wheel         | —                                                                  |
| Check PR Title (`check-pr-title.yaml`) | PR opened, edited, or updated | Enforce Conventional Commits format | —                                                                  |
| Lint (`lint.yaml`)                     | PRs; pushes to `dev`          | Lint the source tree and workflows  | —                                                                  |
| Nightly PyTorch (`nightly-torch.yaml`) | Daily cron; manual dispatch   | Track the PyTorch nightly wheel     | No-NPU smoke tests (`RBLN_DUMMY_DEVICE=1`)                         |

CI, Release, and CD workflows delegate to a shared [Event Dispatch](#event-dispatch-mechanism) mechanism that sends events to infrastructure with physical RBLN NPU devices.

---

## Triggers and Concurrency

| Workflow       | `pull_request`                  | `push`             | Cancel in-progress? |
|----------------|---------------------------------|--------------------|---------------------|
| CI             | To any branch **except** `main` | To `dev`           | Yes                 |
| Release        | To `main`                       | To `main`          | Yes                 |
| CD             | —                               | Tags matching `v*` | **No**              |
| Build          | All PRs                         | —                  | PRs only            |
| Check PR Title | On open, edit, sync, reopen     | —                  | Yes                 |
| Lint           | To `main` or `dev`              | To `dev`           | Yes                 |
| Nightly PyTorch| —                               | —                  | **No**              |

CI, Release, and Lint runs are grouped by PR number (or SHA for pushes); a new push cancels the in-progress run. CD runs are never cancelled — once a deployment starts, it runs to completion. Check PR Title runs are grouped by PR number and always cancel the in-progress run. Build also runs on manual `workflow_dispatch`; its PR runs are grouped by PR number and cancel superseded commits, while dispatch runs are grouped by run ID and always complete.

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

The Release workflow runs when code is promoted from `dev` to `main` for release. It builds and lints across all supported Python versions. It covers a broader test suite than CI, but neither is a strict superset of the other — experimental tests can run in CI but are excluded from Release:

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

## Build Workflow

**File:** [`.github/workflows/build.yaml`](../.github/workflows/build.yaml)

The Build workflow builds the `torch-rbln` wheel and publishes it to the internal package index, on every pull request and on manual `workflow_dispatch`. Unlike CI, Release, and CD, it builds in a container instead of dispatching to RBLN NPU hardware.

The entrypoint fans out a `python_version` × `build_type` matrix to the reusable [`_build-wheel.yaml`](../.github/workflows/_build-wheel.yaml), which pins `rebel-compiler`, builds the wheel, verifies the built artifact in a clean environment, publishes it, and then checks the published version resolves from the index. PRs build `Release`; a manual dispatch can select `Release`, `Debug`, or both via `build_types`.

---

## Check PR Title Workflow

**File:** [`.github/workflows/check-pr-title.yaml`](../.github/workflows/check-pr-title.yaml)

The Check PR Title workflow enforces the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) format on all pull request titles. It triggers whenever a PR is opened, edited, synchronized, or reopened.

The workflow uses [`amannn/action-semantic-pull-request`](https://github.com/amannn/action-semantic-pull-request) to validate the title against the following format:

```
<type>(<optional scope>): <description>
```

**Allowed types:** `feat`, `fix`, `refactor`, `perf`, `docs`, `test`, `build`, `ci`, `chore`

If the title is invalid, the workflow uses [`marocchino/sticky-pull-request-comment`](https://github.com/marocchino/sticky-pull-request-comment) to post a sticky comment on the PR explaining the required format. The comment is automatically deleted once the title is corrected.

---

## Lint Workflow

**File:** [`.github/workflows/lint.yaml`](../.github/workflows/lint.yaml)

The Lint workflow runs on pull requests to `main` or `dev` and on pushes to `dev`, fanning out to two reusable workflows:

- [`_lint-source.yaml`](../.github/workflows/_lint-source.yaml) runs `lintrunner` over the source tree (see [Linting](LINTING.md)). A pull request to `main` (release validation) lints every tracked file across Python 3.10–3.13; pull requests and pushes to `dev` lint only the changed files on 3.12, for fast feedback.
- [`_lint-workflows.yaml`](../.github/workflows/_lint-workflows.yaml) runs `actionlint`, `yamllint`, and `zizmor` on the workflow files.

A final `Lint` job aggregates both, so branch protection has one stable check even as the source matrix varies by branch.

---

## Event Dispatch Mechanism

**File:** [`.github/workflows/_dispatch-event.yaml`](../.github/workflows/_dispatch-event.yaml)

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

## Automated Dependency Updates

**File:** [`.github/workflows/update-rebel-compiler-dependency.yaml`](../.github/workflows/update-rebel-compiler-dependency.yaml)

This workflow runs on a daily schedule and tracks the latest `rebel-compiler` production build. When a newer one is available, it creates or updates a pull request against `dev` for a maintainer to review and merge.

It can also be run manually via `workflow_dispatch`, optionally pinning a specific `rebel_compiler_version` instead of resolving the latest.

---

## Nightly PyTorch Workflow

**File:** [`.github/workflows/nightly-torch.yaml`](../.github/workflows/nightly-torch.yaml)

Everyday CI builds against the release pin (`torch==2.11.0+cpu`). This workflow additionally builds and smoke-tests `torch-rbln` against the **latest PyTorch nightly CPU wheel** every day at 14:00 KST (05:00 UTC), so an upstream breaking change surfaces within a day instead of at the next `torch` bump. Tracking PyTorch `main` in CI is the outstanding prerequisite for enlisting the repository in PyTorch's Cross-Repository CI Relay (CRCR).

Scheduled runs use the default branch (`dev`); a manual `workflow_dispatch` tests whichever ref it is started from, and can pin an explicit `torch_version` and `python_version` instead of the defaults (latest nightly, Python 3.12). An explicit `torch_version` is still resolved against the nightly index, so it may be given with or without the `+cpu` local suffix and fails fast if the index does not serve it.

Steps:

1. **Resolve** the latest nightly version from `https://download.pytorch.org/whl/nightly/cpu`, before checkout so the repository's release-pinned uv configuration cannot influence the result.
2. **Repoint** the `torch` pin via [`tools/replace_depends.py`](../tools/replace_depends.py) — it rewrites `[project].dependencies` and `[build-system].requires`, and points `[tool.uv.sources].torch` at the `pytorch-nightly-cpu` index declared in `pyproject.toml`. The edit is local to the run and never committed.
3. **Build** the wheel with the same container, compiler setup, and `constraints-build-dev.txt` build constraint as [`_build-wheel.yaml`](../.github/workflows/_build-wheel.yaml), but **without publishing** it to the internal package index.
4. **Test** on a CPU-only runner with no NPU attached, using `RBLN_DUMMY_DEVICE=1` (see [Configuration](CONFIGURATION.md#rbln_dummy_device)):
   - a smoke script that installs the built wheel into a clean venv and checks the versions, the dummy device topology, a host↔device round-trip, and one eager op;
   - the no-NPU test suites `test/rbln/test_dummy_device.py` and `test/distributed/test_no_device.py` (each manages `RBLN_DUMMY_DEVICE` itself, so it is not set for this step).
5. **Report** the resolved `torch` version, the built `torch-rbln` version, and the outcome to the job summary, with an error annotation on failure.

---

## Related Documentation

- [Release Process](RELEASE_PROCESS.md) — Branch model, versioning, tagging, and publication
- [Contributing Guide](CONTRIBUTING.md) — PR requirements and merge policy
- [Test Guide](TEST_GUIDE.md) — Test infrastructure, markers, and `run_tests.py` usage
- [Linting](LINTING.md) — `lintrunner` and the workflow linters
