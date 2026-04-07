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

When updating the PyTorch version (e.g. 2.9.0 → 2.10.0), change **all** of the
following together:

| # | File | What to change |
|---|------|----------------|
| 1 | ```pyproject.toml``` `[project].dependencies` | `torch==X.Y.Z+cpu` |
| 2 | ```pyproject.toml``` `[build-system].requires` | `torch==X.Y.Z+cpu` |
| 3 | Upstream files (see below) | Sync from the new tag |

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

### Linter (migrated from PyTorch upstream)

The full contents of ```third_party/pytorch/tools/linter``` have been migrated into
```tools/linter``` in the torch-rbln repository. The linter is no longer taken from
the PyTorch submodule; it is maintained in-tree under ```tools/linter``` (adapters,
clang_tidy, dictionary, etc.). When updating the PyTorch version, there is no need
to sync or copy the linter from upstream—use the in-repo ```tools/linter``` only.

To refresh ```tools/linter``` from PyTorch upstream (e.g. after a version bump),
run ```sync-linter.sh``` from the repo root or from the parent of ```torch-rbln```.
The script clones PyTorch at a given tag and copies ```tools/linter``` into the
current tree. The **default tag** is derived from the torch version in
```pyproject.toml``` (e.g. ```torch==2.9.0+cpu``` → tag ```v2.9.0```). You can
override it by passing a tag: ```./sync-linter.sh v2.10.0```.

## Rebel compiler

**`rebel-compiler`** is a **build-only** dependency in this repo. For installation, versioning, and runtime use of the compiler package, follow the **RBLN SDK** documentation:

- [PyTorch RBLN — Installation (quickstart)](https://docs.rbln.ai/latest/software/rbln_pytorch/installation.html#install)
- [PyTorch RBLN — Running and debugging](https://docs.rbln.ai/latest/software/rbln_pytorch/tutorial_running_n_debugging.html)

To bump the **pinned build dependency** in torch-rbln, update the version specifier in **`pyproject.toml`** in both **`[build-system].requires`** and **`[dependency-groups].build`**, and keep them aligned with each other.
