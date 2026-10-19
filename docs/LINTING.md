# Linting

## Source linting

[Source linting](PIPELINES.md#linting) runs `lintrunner` over the source tree.

Install dependencies and initialize `lintrunner` once:

```bash
uv sync --locked --no-install-project
uv run --no-sync lintrunner init
```

For C++ changes, `clang-tidy` needs a compile database and the Rebel runtime headers, which `uv sync` installs with the pinned `rebel-compiler`. Configure CMake:

```bash
uv run --no-sync cmake -GNinja -B build -S . \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=torch_rbln \
  -DBUILD_TESTING=ON
```

Release checks also lint a `Debug` configuration; reconfigure with `-DCMAKE_BUILD_TYPE=Debug` to reproduce a `clang-tidy` finding reported only under `Debug`.

To lint and auto-fix changed files:

```bash
uv run --no-sync lintrunner -m origin/main -a
```

## Pipeline linting

[Pipeline linting](PIPELINES.md#linting) runs [`yamllint`](https://github.com/adrienverge/yamllint) over `.buildkite/`, then a dry-run upload of each pipeline file.

The `uv sync` command under [Source linting](#source-linting) also installs `yamllint`. To run it locally:

```bash
uv run --no-sync yamllint --strict .buildkite/
```
