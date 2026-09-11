# How to update the version of third party packages

This document explains how to update the versions of third party packages in torch-rbln.
The current third party packages are:

- **PyTorch** (for Debug CI builds)
- **`rebel-compiler`** (build-only; see [PyTorch RBLN — Overview](https://docs.rbln.ai/latest/software/rbln_pytorch/overview.html) and [Installation](https://docs.rbln.ai/latest/software/rbln_pytorch/installation.html) for the package and setup; for usage and debugging workflows, see [Running and debugging with PyTorch RBLN](https://docs.rbln.ai/latest/software/rbln_pytorch/tutorial_running_n_debugging.html))

## Common

In order to update the version of third party packages, you edit the following file.

```
  pyproject.toml / [project] & [build-system] & [dependency-groups].build
```

## Pytorch

torch-rbln uses torch packages. The current version of torch is
fixed by ```pyproject.toml``` in the root directory.

For Debug CI builds, PyTorch is cloned from the repository specified by the
```pytorch-repo``` and ```pytorch-ref``` inputs in the GitHub Actions workflow
(defaults: ```https://github.com/pytorch/pytorch.git```, tag derived from ```pyproject.toml```).

### Version update checklist

When updating the PyTorch version (e.g. 2.10.0 → 2.11.0), change **all** of the
following together:

| # | File | What to change |
|---|------|----------------|
| 1 | ```pyproject.toml``` `[project].dependencies` | `torch==X.Y.Z+cpu` |
| 2 | ```pyproject.toml``` `[build-system].requires` | `torch==X.Y.Z+cpu` |
| 3 | Upstream files (see below) | Sync from the new tag |
| 4 | ```tools/linter``` | Run ```./tools/sync-linter.sh``` (see below) |

CI Debug builds automatically derive the PyTorch git tag (`vX.Y.Z`) from the
```pyproject.toml``` torch version, so no additional workflow files need updating.

For local Debug builds, clone PyTorch manually:

```
  git clone --filter=blob:none https://github.com/pytorch/pytorch.git third_party/pytorch
  cd third_party/pytorch && git checkout vX.Y.Z
```

torch-rbln directly takes the following files from Pytorch upstream.

```
  test/test_ops.py : the first line has this file's upstream hash code.
  aten/src/ATen/native/native_functions.yaml : the first line has this file's upstream hash code.
  aten/src/ATen/native/tags.yaml : the first line has this file's upstream hash code.
```

### Upstream files (grammar/syntax only)

These files are not identical to the ones in PyTorch upstream. We use their grammar and
syntax, not their full implementation—they describe how to generate the actual
implementation of operations for each device backend. When updating the PyTorch version,
you must manually bring in the updated versions of these files from the upstream
repository.

### Linter (verbatim copy of PyTorch upstream)

```tools/linter``` (adapters, clang_tidy, dictionary, etc.) is a verbatim copy of
```tools/linter``` from PyTorch upstream at the tag recorded in
```tools/linter/UPSTREAM_TAG```. It is not edited in-tree, and it moves only as part
of a torch version bump: the target tag is always derived from the torch pin in
```pyproject.toml``` (```torch==X.Y.Z+cpu``` → ```vX.Y.Z```) and cannot be overridden.

After changing the pin (checklist items 1–2), run

```
  ./tools/sync-linter.sh
```

from anywhere. It fetches ```tools/linter``` at the pinned tag, replaces the tree,
updates ```UPSTREAM_TAG```, and is a no-op when the tree is already there. It refuses
to run over uncommitted changes under ```tools/linter```. Commit the result together
with the pin change.

## Rebel compiler

**`rebel-compiler`** is a **build-only** dependency in this repo. For installation, versioning, and runtime use of the compiler package, follow the **RBLN SDK** documentation:

- [PyTorch RBLN — Installation (quickstart)](https://docs.rbln.ai/latest/software/rbln_pytorch/installation.html#install)
- [PyTorch RBLN — Running and debugging](https://docs.rbln.ai/latest/software/rbln_pytorch/tutorial_running_n_debugging.html)

To bump the **pinned build dependency** in torch-rbln, update the version specifier in **`pyproject.toml`** in both **`[build-system].requires`** and **`[dependency-groups].build`**, and keep them aligned with each other.

> **Note:** The development build constraint is updated automatically by the [`rebel-compiler` dependency update workflow](WORKFLOWS.md#rebel-compiler-dependency-update).
