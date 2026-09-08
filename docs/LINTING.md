# Linting

## Source linting

[Source linting](WORKFLOWS.md#linting) runs `lintrunner` over the source tree.

Install dependencies and initialize `lintrunner` once:

```bash
uv sync --no-install-project
uv run --no-sync lintrunner init
```

For C++ changes, `clang-tidy` needs the Rebel runtime headers and a compile database. Install the build-pinned rebel-compiler, then configure CMake:

```bash
uv pip install --constraint constraints-build-dev.txt rebel-compiler
uv run --no-sync cmake -GNinja -B build -S . \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=torch_rbln
```

Release checks also lint a `Debug` configuration; reconfigure with `-DCMAKE_BUILD_TYPE=Debug` to reproduce a `clang-tidy` finding reported only under `Debug`.

To lint and auto-fix changed files:

```bash
uv run --no-sync lintrunner -m origin/main -a
```

## Workflow linting

[Workflow linting](WORKFLOWS.md#linting) runs [`actionlint`](https://github.com/rhysd/actionlint), [`yamllint`](https://github.com/adrienverge/yamllint), and [`zizmor`](https://github.com/zizmorcore/zizmor) on the workflow files.

Install dependencies once:

```bash
uv venv
uv pip install --group lint
```

To run them locally:

```bash
uv run --no-sync actionlint
uv run --no-sync yamllint --strict .github/
uv run --no-sync zizmor --offline .github/
```
