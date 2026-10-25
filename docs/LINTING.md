# Linting

## Source linting

[Source linting](PIPELINES.md#linting) runs `lintrunner`. To run the same checks
locally, install the dependencies and initialize `lintrunner` once:

```bash
uv sync --locked --no-install-project
uv run --no-sync lintrunner init
```

For C++ changes, `clang-tidy` also needs the compilation database that CMake
writes when it configures. Configuring needs the runtime headers from the
`rebel-compiler` that `uv sync` installed:

```bash
uv run --no-sync cmake -GNinja -B build -S . \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=torch_rbln \
  -DBUILD_TESTING=ON
```

Release checks also lint a `Debug` configuration. To reproduce a `clang-tidy`
finding reported only there, configure with `-DCMAKE_BUILD_TYPE=Debug`.

Lint the files changed since `origin/main` and apply the fixes:

```bash
uv run --no-sync lintrunner -m origin/main -a
```

## Linters

Most linters come from upstream PyTorch. Their adapters in `tools/linter/` pin
the tool versions, and [syncing from upstream](THIRD_PARTY_UPDATE.md#linter)
overwrites `tools/linter/` along with `.clang-format` and `.clang-tidy`, so do
not edit these files by hand. The other linters run tools from the `lint`
dependency group through `tools/linter_rbln/command_linter.py`.

## Pipeline linting

[Pipeline linting](PIPELINES.md#linting) dry-runs the upload of each pipeline
file. Source linting checks the pipeline files as YAML.
