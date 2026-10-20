# CI/CD Pipelines

CI/CD runs as Buildkite pipelines defined in [`.buildkite/`](../.buildkite). Each pipeline's settings hold its build sources and a step that uploads its entry file. The entry file's steps upload step files, and their steps run scripts.

```text
.buildkite/
├── pipeline.<name>.yml  # Entry file of each pipeline
├── pipelines/           # Step files that the entry files upload
└── scripts/             # Scripts that the steps run
```

## Pipelines

| Pipeline                                                            | Build source             | Description                                                              | Commit status       |
|---------------------------------------------------------------------|--------------------------|--------------------------------------------------------------------------|---------------------|
| [PR title lint](../.buildkite/pipeline.pr-title-lint.yml)           | Pull requests            | Checks that the pull request title follows Conventional Commits          | `PR title lint`     |
| [Pre-merge checks](../.buildkite/pipeline.pre-merge-checks.yml)     | Pull requests            | Lints, builds, and tests the pull request                                | `Pre-merge checks`  |
| [Post-merge checks](../.buildkite/pipeline.post-merge-checks.yml)   | Pushes to `main`         | Lints, builds, and tests the merged commit                               | `Post-merge checks` |
| [Release checks](../.buildkite/pipeline.release-checks.yml)         | Daily schedule on `main` | Checks the wheels a release ships, then promotes them                    | `Release checks`    |
| [Release](../.buildkite/pipeline.release.yml)                       | Manual                   | Tags a commit and publishes its wheels                                   | —                   |
| [Dependency updates](../.buildkite/pipeline.dependency-updates.yml) | Daily schedule on `main` | Opens a pull request that updates `rebel-compiler` to the latest version | —                   |
| [Nightly torch test](../.buildkite/pipeline.nightly-torch-test.yml) | Daily schedule on `main` | Tests against nightly `torch`                                            | —                   |

A build created by a GitHub event or a schedule reports its commit status. A manual build reports one only with `TORCH_RBLN_COMMIT_STATUS=true`.

## Checks

Pre-merge, post-merge, and release checks all run these jobs:

- **Source linting** runs the linters once per Python version, CPU architecture, and build type.
- **Wheel builds** build a wheel per Python version, CPU architecture, and build type.
- **NPU tests** run the C++ tests and the Python test suites on NPUs. The [test mode](TEST_GUIDE.md#8-cicd-integration) selects which Python tests run.

All three publish their `Release` wheels to the `rbln-internal` index and run their NPU tests against the x86_64 wheels for Python 3.12, but they differ in coverage:

|                | Pre-merge checks               | Post-merge checks              | Release checks                                                                                          |
|----------------|--------------------------------|--------------------------------|---------------------------------------------------------------------------------------------------------|
| Source linting | Python 3.12, x86_64, `Release` | Python 3.12, x86_64, `Release` | Python 3.10–3.14, x86_64, `Release` and `Debug`                                                         |
| Wheel builds   | Python 3.12, x86_64, `Release` | Python 3.12, x86_64, `Release` | Python 3.10–3.14, x86_64, `Release`; Python 3.12, x86_64, `Debug`                                       |
| NPU tests      | RBLN-CA25, RBLN-CR13           | RBLN-CA25                      | `Release` wheel: RBLN-CA25, RBLN-CR13; `Debug` wheel: RBLN-CA25, without the `models` Python test suite |
| Test mode      | `ci`                           | `ci`                           | `release`                                                                                               |

### Linting

Source linting covers every tracked file in post-merge and release checks. In pre-merge checks, it covers the files that the pull request changes, or every tracked file if the pull request changes the lint configuration.

[Pipeline linting](LINTING.md#pipeline-linting) runs a dry-run upload of each pipeline file with the `TORCH_RBLN_*` variables unset, except for required variables (`${VAR?}`), which receive placeholder values. Each entry file must therefore pass with its own defaults.

### Release checks

The `Debug` wheel builds and runs against a debug wheel of the pinned `torch` from the `rbln-internal` index. When the index lacks that wheel, release checks on `main` build and publish it first; on any other branch, the `Debug` wheel build fails because it cannot lock an unpublished wheel.

On `main`, once every other step passes, the promotion steps publish the `Release` wheels to the `rbln-nightly` index.

## Release

The release pipeline tags the build's commit, which can be on any branch.

| Input                       | Default    | Description                             |
|-----------------------------|------------|-----------------------------------------|
| `TORCH_RBLN_VERSION_TAG`    | (required) | The tag to create                       |
| `TORCH_RBLN_RELEASE_WHEELS` | `true`     | Whether to build and publish the wheels |

The pipeline does the following:

1. Tags the commit and creates a GitHub Release for a final release or a post-release.
2. Builds the `Release` wheels that release checks build.
3. Publishes the wheels to the `rbln-internal` index and promotes them to the `rbln-nightly` index, without testing them.

A retry or a new build on the same commit keeps the tag, the GitHub Release, and the wheels that already exist. A tag that already points at another commit fails the build.

## Dependency updates

When the latest `rebel-compiler` on the `rbln-nightly` index is newer than the pin, the pipeline force-pushes the [pin update](THIRD_PARTY_UPDATE.md#rebel-compiler) to the `update-rebel-compiler` branch, dropping any change pushed there, and opens or updates its pull request. It runs only on `main`.

## Nightly torch test

The nightly torch test builds against nightly `torch` instead of the `torch` pin and runs the `core` Python test suite on RBLN-CR13.

| Input                       | Default    | Description                             |
|-----------------------------|------------|-----------------------------------------|
| `TORCH_RBLN_PYTHON_VERSION` | `3.12`     | The Python version                      |
| `TORCH_RBLN_TORCH_VERSION`  | The latest | The nightly `torch` version to build on |

## Agent queues

The entry files do not hard-code queue names. The upload step in each pipeline's settings sets these variables in its `env`, and an upload without them fails.

| Variable                      | Description                                       |
|-------------------------------|---------------------------------------------------|
| `TORCH_RBLN_QUEUE_AMD64_CA25` | AMD64 agents with RBLN-CA25 for NPU tests         |
| `TORCH_RBLN_QUEUE_AMD64_CR13` | AMD64 agents with RBLN-CR13 for NPU tests         |
| `TORCH_RBLN_QUEUE_AMD64`      | AMD64 agents for x86_64 builds and source linting |
| `TORCH_RBLN_QUEUE_DEFAULT`    | Agents for every other step                       |
