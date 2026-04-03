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
