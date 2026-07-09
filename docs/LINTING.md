# Linting

A Git `pre-commit` hook runs linting automatically on commit. Initialize `lintrunner` once:

```bash
source .venv/bin/activate
lintrunner init
```

After initialization, linting runs on every `git commit`. To manually lint and auto-fix:

```bash
lintrunner -m main -a
```

## Workflow linting

[Workflow linting](WORKFLOWS.md#lint-workflows) runs [`actionlint`](https://github.com/rhysd/actionlint), [`yamllint`](https://github.com/adrienverge/yamllint), and [`zizmor`](https://github.com/zizmorcore/zizmor) on the workflow files. To run them locally:

```bash
uv pip install --group lint

actionlint
yamllint --strict .github/
zizmor --offline .github/
```
