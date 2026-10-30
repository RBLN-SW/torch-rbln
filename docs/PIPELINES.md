# CI/CD Pipelines

CI/CD runs as Buildkite pipelines defined in [`.buildkite/`](../.buildkite).
Each pipeline's settings hold its build sources and a step that uploads its
entry file. The entry file's steps upload step files, and their steps run
scripts.

```text
.buildkite/
├── pipeline.<name>.yml  # Entry file of each pipeline
├── pipelines/           # Step files that the entry files upload
└── scripts/             # Scripts that the steps run
```

## Pipelines

| Pipeline                                                              | Build source             | Description                                                              | Commit status       |
| --------------------------------------------------------------------- | ------------------------ | ------------------------------------------------------------------------ | ------------------- |
| [PR title lint](../.buildkite/pipeline.pr-title-lint.yml)             | Pull requests            | Checks that the pull request title follows Conventional Commits          | `PR title lint`     |
| [Pre-merge checks](../.buildkite/pipeline.pre-merge-checks.yml)       | Pull requests            | Lints, builds, and tests the pull request                                | `Pre-merge checks`  |
| [Post-merge checks](../.buildkite/pipeline.post-merge-checks.yml)     | Pushes to `main`         | Lints, builds, and tests the merged commit                               | `Post-merge checks` |
| [Release checks](../.buildkite/pipeline.release-checks.yml)           | Daily schedule on `main` | Checks the wheels a release ships, then promotes them                    | `Release checks`    |
| [Release](../.buildkite/pipeline.release.yml)                         | Manual                   | Tags a commit and publishes its wheels                                   | —                   |
| [Dependency updates](../.buildkite/pipeline.dependency-updates.yml)   | Daily schedule on `main` | Opens a pull request that updates `rebel-compiler` to the latest version | —                   |
| [Nightly `torch` test](../.buildkite/pipeline.nightly-torch-test.yml) | Daily schedule on `main` | Tests against nightly `torch`                                            | —                   |

A build created by a GitHub event or a schedule reports its commit status. A
manual build reports one only with `TORCH_RBLN_COMMIT_STATUS=true`.

## Checks

Pre-merge, post-merge, and release checks all run these jobs:

- **Source linting** runs the linters once per Python version, CPU architecture,
  and build type.
- **Wheel builds** build a wheel per Python version, CPU architecture, and build
  type. They check each wheel for `manylinux` compatibility and valid metadata,
  and verify that it installs and imports in a clean environment.
- **NPU tests** run the C++ tests and the Python test suites for each NPU
  product. Only release checks run the Python tests marked
  [`periodic`](TEST_GUIDE.md#ci-integration).

All three publish their `Release` wheels to the `rbln-internal` index and run
their NPU tests against the x86_64 wheels for Python 3.12, but they differ in
coverage:

|                | Pre-merge checks               | Post-merge checks              | Release checks                                                                                          |
| -------------- | ------------------------------ | ------------------------------ | ------------------------------------------------------------------------------------------------------- |
| Source linting | Python 3.12, x86_64, `Release` | Python 3.12, x86_64, `Release` | Python 3.10–3.14, x86_64, `Release` and `Debug`                                                         |
| Wheel builds   | Python 3.12, x86_64, `Release` | Python 3.12, x86_64, `Release` | Python 3.10–3.14, x86_64 and aarch64, `Release`; Python 3.12, x86_64, `Debug`                           |
| NPU tests      | RBLN-CA25, RBLN-CR13           | RBLN-CA25                      | `Release` wheel: RBLN-CA25, RBLN-CR13; `Debug` wheel: RBLN-CA25, without the `models` Python test suite |
| Periodic tests | Not run                        | Not run                        | Run                                                                                                     |

### Linting

Source linting covers every tracked file in post-merge and release checks. In
pre-merge checks, it covers the files that the pull request changes, or every
tracked file if the pull request changes the lint configuration.

[Pipeline linting](LINTING.md#pipeline-linting) runs a dry-run upload of each
pipeline file with the `TORCH_RBLN_*` environment variables unset, except for
the required ones (`${VAR?}`), which receive placeholder values. Each entry file
must therefore pass with its own defaults.

### Python tests

Each Python test suite runs as one step per NPU count, and each step runs the
tests whose [`npu` marker](TEST_GUIDE.md#markers) declares that count. An
NPU-less step runs the tests that do not require an NPU. For the dummy devices
those tests create, it sets `RBLN_FORCE_NPU_NAME` to the step's NPU product and
`LD_LIBRARY_PATH` to the `UMD_PATH` environment variable, because only NPU pods
have the RBLN user-mode driver.

Before running `pytest`, each step checks that the pod has exactly the requested
NPUs of the step's product and that it can load the RBLN user-mode driver. A
failed check fails the step, so a misallocated pod cannot pass with skipped
tests or on the wrong NPUs.

The `ops` step also writes a
[CPU fallback report](TEST_GUIDE.md#report-cpu-fallback-per-test) for each
`pytest` run and uploads both as artifacts. A failed upload does not fail the
step.

The `ops` step runs its parallel tests with `--dist=worksteal`. Op test
durations differ by orders of magnitude, and the default `load` scheduling can
leave long tests queued on one worker while the others sit idle. The step also
sets `OMP_NUM_THREADS=1` for both runs: it runs one worker per CPU, and each
worker's OpenMP pool would otherwise size itself to every CPU it can see.

### Release checks

The `Debug` wheel builds and runs against a debug wheel of the pinned `torch`
from the `rbln-internal` index. When the index lacks that wheel, release checks
on `main` build and publish it first; on any other branch, the `Debug` wheel
build fails because it cannot lock an unpublished wheel.

The aarch64 wheel builds soft fail, as they do in the release pipeline.

On `main`, once every other step passes, the promotion steps publish the
`Release` wheels to the `rbln-nightly` index.

## Release

The release pipeline tags the build's commit, which can be on any branch.

| Variable                    | Default    | Description                             |
| --------------------------- | ---------- | --------------------------------------- |
| `TORCH_RBLN_VERSION_TAG`    | (required) | The tag to create                       |
| `TORCH_RBLN_RELEASE_WHEELS` | `true`     | Whether to build and publish the wheels |

The pipeline does the following:

1. Tags the commit and creates a GitHub Release for a final release or a
   post-release.
2. Builds the `Release` wheels that release checks build.
3. Publishes the wheels to the `rbln-internal` index and promotes them to the
   `rbln-nightly` index, without testing them.

A retry or a new build on the same commit keeps the tag, the GitHub Release, and
the wheels that already exist. A tag that already points at another commit fails
the build.

## Dependency updates

When the latest `rebel-compiler` on the `rbln-nightly` index is newer than the
pin, the pipeline force-pushes the
[pin update](THIRD_PARTY_UPDATE.md#rebel-compiler) to the
`update-rebel-compiler` branch, dropping any change pushed there, and opens or
updates its pull request. It runs only on `main`.

## Nightly `torch` test

The nightly `torch` test builds against nightly `torch` instead of the `torch`
pin and runs the `core` Python test suite for RBLN-CR13.

| Variable                    | Default    | Description                             |
| --------------------------- | ---------- | --------------------------------------- |
| `TORCH_RBLN_PYTHON_VERSION` | `3.12`     | The Python version                      |
| `TORCH_RBLN_TORCH_VERSION`  | The latest | The nightly `torch` version to build on |

## Agent queues

The entry files do not hard-code queue names. The upload step in each pipeline's
settings sets these environment variables in its `env`, and an upload without
them fails.

| Variable                      | Description                                       |
| ----------------------------- | ------------------------------------------------- |
| `TORCH_RBLN_QUEUE_AMD64_CA25` | AMD64 agents with RBLN-CA25 for NPU tests         |
| `TORCH_RBLN_QUEUE_AMD64_CR13` | AMD64 agents with RBLN-CR13 for NPU tests         |
| `TORCH_RBLN_QUEUE_AMD64`      | AMD64 agents for x86_64 builds and source linting |
| `TORCH_RBLN_QUEUE_ARM64`      | Arm64 agents for aarch64 builds                   |
| `TORCH_RBLN_QUEUE_DEFAULT`    | Agents for NPU-less steps and every other step    |
