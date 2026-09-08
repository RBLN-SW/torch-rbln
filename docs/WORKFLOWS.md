# CI/CD Workflows

This document describes the GitHub Actions workflows that power the automated testing and deployment pipeline for `torch-rbln`. All workflow definitions live in [`.github/workflows/`](../.github/workflows/).

## Overview

The workflows below each have a trigger of their own. Files prefixed with `_` are reusable workflows they call, never triggered directly.

| Workflow                                                                     | Purpose                                 |
|------------------------------------------------------------------------------|-----------------------------------------|
| Pre-merge checks (`pre-merge-checks.yaml`)                                   | Gate the proposed change                |
| Post-merge checks (`post-merge-checks.yaml`)                                 | Confirm the integrated commit           |
| Release checks (`release-checks.yaml`)                                       | Decide whether `main` is fit to release |
| PR title check (`check-pr-title.yaml`)                                       | Enforce Conventional Commits format     |
| Build (`build.yaml`)                                                         | Build and publish wheels                |
| CD (`cd.yaml`)                                                               | Build and publish release artifacts     |
| `rebel-compiler` dependency update (`update-rebel-compiler-dependency.yaml`) | Propose a `rebel-compiler` version bump |
| Nightly PyTorch (`nightly-torch.yaml`)                                       | Track the PyTorch nightly wheel         |

---

## Triggers and Concurrency

| Workflow                           | Trigger                     | Grouped by                       | `cancel-in-progress` |
|------------------------------------|-----------------------------|----------------------------------|----------------------|
| Pre-merge checks                   | All PRs                     | PR number                        | `true`               |
| Post-merge checks                  | Pushes to `main`            | Commit SHA                       | `false`              |
| Release checks                     | Daily 00:15 KST; manual run | Commit SHA                       | `false`              |
| PR title check                     | All PRs, including edits    | PR number                        | `true`               |
| Build                              | All PRs; manual run         | PR number; run ID on manual runs | `true` on PRs        |
| CD                                 | Version tags (`v*`)         | Commit SHA                       | `false`              |
| `rebel-compiler` dependency update | Daily 09:00 KST; manual run | Workflow name                    | `false`              |
| Nightly PyTorch                    | Daily 14:00 KST; manual run | Workflow name                    | `false`              |

Cancelling is safe when a newer commit makes the earlier result obsolete, which is why the pull request workflows cancel. Keying on the commit SHA already isolates commits from each other, so the only runs sharing a group are repeats of one commit, and `false` lets the run in flight finish. Grouping the dependency-tracking workflows by workflow name keeps a run from overlapping its predecessor.

---

## Checks

These workflows decide whether a change is acceptable. Pre-merge, post-merge, and release checks are the validation workflows, one at each stage of a change's lifecycle; they run their NPU tests through the shared [event dispatch mechanism](#event-dispatch-mechanism).

### Pre-merge checks

**File:** [`.github/workflows/pre-merge-checks.yaml`](../.github/workflows/pre-merge-checks.yaml)

Pre-merge checks provide fast feedback and gate merging into `main`. They run the CI-mode suite against the pull request head:

```bash
python test/run_tests.py  # -m "test_set_ci"
```

See the [Test Guide](TEST_GUIDE.md) for what the markers select and how tests are split across workers.

### Post-merge checks

**File:** [`.github/workflows/post-merge-checks.yaml`](../.github/workflows/post-merge-checks.yaml)

Pull requests are merged without rebasing onto the latest `main`, so pre-merge checks cannot answer whether the integrated result holds. Post-merge checks run the CI-mode suite on the `main` commit a merge produces.

They also lint the whole tree rather than the changed files alone (see [Linting](#linting)), and their result is the health signal for `main` in the [Release Process](RELEASE_PROCESS.md).

### Release checks

**File:** [`.github/workflows/release-checks.yaml`](../.github/workflows/release-checks.yaml)

Release checks decide whether the current `main` commit is fit to release (see [Release Process](RELEASE_PROCESS.md)). A manual run can validate the tip of `main` ahead of a release. They run a broader test suite than pre-merge and post-merge checks:

```bash
python test/run_tests.py --test_mode=release  # -m "not (test_set_experimental or test_set_perf)"
```

### PR title check

**File:** [`.github/workflows/check-pr-title.yaml`](../.github/workflows/check-pr-title.yaml)

The PR title check enforces the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) format on all pull request titles.

The workflow uses [`amannn/action-semantic-pull-request`](https://github.com/amannn/action-semantic-pull-request) to validate the title against the following format:

```text
<type>(<optional scope>): <description>
```

**Allowed types:** `feat`, `fix`, `refactor`, `perf`, `docs`, `test`, `build`, `ci`, `chore`

If the title is invalid, the workflow uses [`marocchino/sticky-pull-request-comment`](https://github.com/marocchino/sticky-pull-request-comment) to post a sticky comment on the PR explaining the required format. The comment is automatically deleted once the title is corrected.

### Linting

**File:** [`.github/workflows/_lint.yaml`](../.github/workflows/_lint.yaml)

Each validation workflow calls `_lint.yaml` and sets the scope for its stage. That workflow fans out in turn:

- [`_lint-source.yaml`](../.github/workflows/_lint-source.yaml) runs `lintrunner` over the source tree (see [Linting](LINTING.md)), once per combination of Python version and build type.
- [`_lint-workflows.yaml`](../.github/workflows/_lint-workflows.yaml) runs `actionlint`, `yamllint`, and `zizmor` on the workflow files. It is invariant to those dimensions, so it runs once.

| Workflow          | Source files      | Python version | Build type     |
|-------------------|-------------------|----------------|----------------|
| Pre-merge checks  | Changed in the PR | 3.12           | Release        |
| Post-merge checks | Every tracked     | 3.12           | Release        |
| Release checks    | Every tracked     | 3.10–3.14      | Release, Debug |

`mypy` results are interpreter-specific, so release checks confirm every supported Python version. The build type selects `NDEBUG`, which decides whether `clang-tidy` sees the debug-only code paths, so linting `Release` alone never covers them. `clang-tidy` also compiles against the Python headers, so it varies along both dimensions; the matrix is therefore a full cross product rather than a separate sweep per linter.

---

## Artifacts

These workflows produce `torch-rbln` wheels.

### Build

**File:** [`.github/workflows/build.yaml`](../.github/workflows/build.yaml)

Build runs in a container and dispatches nothing to RBLN NPU hardware.

For each `python_version` and `build_type` combination, the entrypoint calls the reusable [`_build-wheel.yaml`](../.github/workflows/_build-wheel.yaml), which pins `rebel-compiler`, builds the wheel, verifies it in a clean environment, publishes it, and checks that the published version resolves from the index. Pull request runs build `Release` on Python 3.12; a manual run can widen both dimensions.

### CD

**File:** [`.github/workflows/cd.yaml`](../.github/workflows/cd.yaml)

CD builds and publishes release artifacts for a tagged commit. It dispatches a `torch-rbln-cd` event.

---

## Event Dispatch Mechanism

**File:** [`.github/workflows/_dispatch-event.yaml`](../.github/workflows/_dispatch-event.yaml)

The validation workflows and CD delegate to another repository through GitHub [repository dispatch](https://docs.github.com/en/actions/writing-workflows/choosing-when-your-workflow-runs/events-that-trigger-workflows#repository_dispatch). That repository runs the work on physical RBLN NPU hardware. The caller sends `torch-rbln-ci`, `torch-rbln-release`, or `torch-rbln-cd` as the dispatch event type, with this payload:

| Field            | Description                                                    |
|------------------|----------------------------------------------------------------|
| `event_name`     | GitHub event that triggered the workflow                       |
| `torch_rbln_ref` | Git reference to check out, e.g. `main` or `refs/tags/v0.10.0` |
| `torch_rbln_sha` | Git commit SHA to build and test                               |

A [`peter-evans/repository-dispatch`](https://github.com/peter-evans/repository-dispatch) step sends the event to the repository named by `vars.TORCH_RBLN_DISPATCH_REPOSITORY`.

---

## Dependency Tracking

These workflows track the latest published build of a dependency on a daily schedule.

### `rebel-compiler` dependency update

**File:** [`.github/workflows/update-rebel-compiler-dependency.yaml`](../.github/workflows/update-rebel-compiler-dependency.yaml)

When a newer `rebel-compiler` production build appears, this workflow creates or updates a pull request against `main` for a maintainer to review and merge.

It can also be run manually via `workflow_dispatch`, optionally pinning a specific `rebel_compiler_version` instead of resolving the latest.

### Nightly PyTorch

**File:** [`.github/workflows/nightly-torch.yaml`](../.github/workflows/nightly-torch.yaml)

Everyday CI builds against the release pin (`torch==2.11.0+cpu`). This workflow additionally builds and smoke-tests `torch-rbln` against the **latest PyTorch nightly CPU wheel** every day at 14:00 KST (05:00 UTC), so an upstream breaking change surfaces within a day instead of at the next `torch` bump. Tracking PyTorch `main` in CI is the outstanding prerequisite for enlisting the repository in PyTorch's Cross-Repository CI Relay (CRCR).

Scheduled runs use the default branch (`main`); a manual `workflow_dispatch` tests whichever ref it is started from, and can pin an explicit `torch_version` and `python_version` instead of the defaults (latest nightly, Python 3.12). An explicit `torch_version` is still resolved against the nightly index, so it may be given with or without the `+cpu` local suffix and fails fast if the index does not serve it.

Steps:

1. **Resolve** the latest nightly version from `https://download.pytorch.org/whl/nightly/cpu`, before checkout so the repository's release-pinned uv configuration cannot influence the result.
2. **Repoint** the `torch` pin via [`tools/replace_depends.py`](../tools/replace_depends.py) — it rewrites `[project].dependencies` and `[build-system].requires`, and points `[tool.uv.sources].torch` at the `pytorch-nightly-cpu` index declared in `pyproject.toml`. The edit is local to the run and never committed.
3. **Build** the wheel with the same container, compiler setup, and `constraints-build-dev.txt` build constraint as [`_build-wheel.yaml`](../.github/workflows/_build-wheel.yaml), but **without publishing** it to the internal package index.
4. **Test** on a CPU-only runner with no NPU attached, using `RBLN_DUMMY_DEVICE=1` (see [Configuration](CONFIGURATION.md#rbln_dummy_device)):
   - a smoke script that installs the built wheel into a clean venv and checks the versions, the dummy device topology, a host↔device round-trip, and one eager op;
   - the no-NPU test suites `test/rbln/test_dummy_device.py` and `test/distributed/test_no_device.py` (each manages `RBLN_DUMMY_DEVICE` itself, so it is not set for this step), and `test/internal/test_device_arch.py`, whose architecture gates only see a host with no NPU here.
5. **Report** the resolved `torch` version, the built `torch-rbln` version, and the outcome to the job summary, with an error annotation on failure.

---

## Related Documentation

- [Release Process](RELEASE_PROCESS.md) — Release lifecycle, versioning, tagging, and publication
- [Contributing Guide](CONTRIBUTING.md) — PR requirements and merge policy
- [Test Guide](TEST_GUIDE.md) — Test infrastructure, markers, and `run_tests.py` usage
- [Linting](LINTING.md) — `lintrunner` and the workflow linters
