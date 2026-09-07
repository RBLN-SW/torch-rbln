# PyTorch RBLN
<div align="center">
<picture>
  <source srcset="https://raw.githubusercontent.com/RBLN-SW/torch-rbln/main/docs/img/torch-rbln-white.png" media="(prefers-color-scheme: dark)">
  <source srcset="https://raw.githubusercontent.com/RBLN-SW/torch-rbln/main/docs/img/torch-rbln-black.png" media="(prefers-color-scheme: light)">
  <img src="https://raw.githubusercontent.com/RBLN-SW/torch-rbln/main/docs/img/torch-rbln-black.png" alt="PyTorch RBLN" width="90%">
</picture>

[![PyPI version](https://badge.fury.io/py/torch-rbln.svg)](https://badge.fury.io/py/torch-rbln)
[![License](https://img.shields.io/github/license/RBLN-SW/torch-rbln)](https://github.com/RBLN-SW/torch-rbln/blob/main/LICENSE)
[![Documentation](https://img.shields.io/badge/docs-available-brightgreen)](https://docs.rbln.ai/latest/software/rbln_pytorch/overview.html)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-1.4-4baaaa.svg)](https://github.com/RBLN-SW/torch-rbln/blob/main/docs/CODE_OF_CONDUCT.md)
</div>

## About

PyTorch RBLN (`torch-rbln`) is a PyTorch extension that allows natural use of Rebellions NPU compute within PyTorch. By implementing **eager mode**, which operates in a **define-by-run** fashion, it supports the full lifecycle of **model development, deployment, and serving** in the PyTorch ecosystem. It is also convenient for **debugging** and related workflows.

The same interface style as **CPU** and **GPU** applies — the **`rbln`** device, **`torch.rbln`**, and **`torch.compile`** — so developers and customers can target RBLN NPUs with familiar APIs. Operations on `rbln` tensors are integrated via PyTorch’s **[out-of-tree extension](https://docs.pytorch.org/tutorials/unstable/python_extension_autoload.html)** path; execution is coordinated with the RBLN compiler and runtime (**`rebel-compiler`**).

PyTorch RBLN is currently in **beta** and under active development. APIs may change between releases, backward compatibility is not guaranteed, and production use is not recommended yet. For the full notice, architecture, supported operators, and tutorials, see **[PyTorch RBLN — Overview](https://docs.rbln.ai/latest/software/rbln_pytorch/overview.html)** in the RBLN SDK documentation. For wheels, **`rebel-compiler`**, and building from source, see **[Installation](https://docs.rbln.ai/latest/software/rbln_pytorch/installation.html)**.

## Getting started

### Prerequisites

- **Python** 3.10–3.14 — see [Installation — Requirements](https://docs.rbln.ai/latest/software/rbln_pytorch/installation.html#requirements) (source build).
- **`rebel-compiler`** — required; **not** installed with **`torch-rbln`**. Use an **[RBLN Portal account](https://docs.rbln.ai/latest/supports/contact_us.html#rbln-portal)** and the RBLN package index as described in [Install pre-built wheels](https://docs.rbln.ai/latest/software/rbln_pytorch/installation.html#install).

### Install pre-built wheels

**`torch-rbln`** (public wheel). Install **`torch`** from the PyTorch CPU index first, then **`torch-rbln`** from PyPI.

```bash
pip3 install torch==2.11.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip3 install torch-rbln
```

For **`rebel-compiler`** and the rest of the setup, see **Prerequisites** above and [Installation](https://docs.rbln.ai/latest/software/rbln_pytorch/installation.html#install).

### Build from source

1. Install **[uv](https://docs.astral.sh/uv/getting-started/installation/)** (see [Installation — Prerequisites](https://docs.rbln.ai/latest/software/rbln_pytorch/installation.html#prerequisites) in the SDK docs).
2. Configure access to the RBLN package index (see [Authenticate to the RBLN package index](#authenticate-to-the-rbln-package-index) below).
3. Follow **[Build from source](https://docs.rbln.ai/latest/software/rbln_pytorch/installation.html#build-from-source-advanced)** (venv, **`rebel-compiler`**, editable build, manual steps).

```bash
git clone https://github.com/RBLN-SW/torch-rbln.git
cd torch-rbln
uv venv .venv && source .venv/bin/activate
./tools/dev-setup.sh pypi
```

**`rebel-compiler`** must be available in the same environment before the **`torch-rbln`** build finishes (see Prerequisites).

#### Authenticate to the RBLN package index

**`rebel-compiler`** is installed from the RBLN package index (`pypi.rbln.ai`), which requires an **[RBLN Portal account](https://docs.rbln.ai/latest/supports/contact_us.html#rbln-portal)**. Without credentials, `./tools/dev-setup.sh pypi` fails with:

```
❌ Cannot reach any rbln pypi index (no permission or network error).
```

Add your RBLN Portal credentials to `~/.netrc` so `pip`/`uv` can authenticate:

```
machine pypi.rbln.ai
login <your-rbln-portal-id>
password <your-rbln-portal-password>
```

Then restrict its permissions (tools refuse a world-readable `.netrc`):

```bash
chmod 600 ~/.netrc
```

Re-run `./tools/dev-setup.sh pypi` once the file is in place.

## Documentation

**RBLN SDK (hosted)**

- [Overview](https://docs.rbln.ai/latest/software/rbln_pytorch/overview.html) — design, components, and entry points into the PyTorch RBLN docs
- [Installation](https://docs.rbln.ai/latest/software/rbln_pytorch/installation.html) — pre-built wheels, **`rebel-compiler`**, build from source
- [Running and debugging with PyTorch RBLN](https://docs.rbln.ai/latest/software/rbln_pytorch/tutorial_running_n_debugging.html) — basic usage and debugging
- [Running a LLM model: Llama3.2-1B](https://docs.rbln.ai/latest/software/rbln_pytorch/tutorial_llama.html) — `transformers` example
- [Supported Ops](https://docs.rbln.ai/latest/software/rbln_pytorch/supported_ops.html) — operator coverage
- [APIs](https://docs.rbln.ai/latest/software/rbln_pytorch/api.html) — Python API reference
- [Troubleshooting](https://docs.rbln.ai/latest/software/rbln_pytorch/troubleshoot.html) — `librbln` / `torch_rbln.diagnose`, core dumps, logging, dtype / CPU, memory (maintained in the RBLN SDK docs)

**This repository**

- [Configuration](https://github.com/RBLN-SW/torch-rbln/blob/main/docs/CONFIGURATION.md) — environment variables and runtime options
- [Test Guide](https://github.com/RBLN-SW/torch-rbln/blob/main/docs/TEST_GUIDE.md) — local test runs
- [Linting](https://github.com/RBLN-SW/torch-rbln/blob/main/docs/LINTING.md) — code style and lint
- [Third-party update](https://github.com/RBLN-SW/torch-rbln/blob/main/docs/THIRD_PARTY_UPDATE.md) — PyTorch pin, upstream files, `rebel-compiler` version bumps in `pyproject.toml`
- [Release Process](https://github.com/RBLN-SW/torch-rbln/blob/main/docs/RELEASE_PROCESS.md) — branch model, versioning, tagging, and publication

## Contributing

See [docs/CONTRIBUTING.md](https://github.com/RBLN-SW/torch-rbln/blob/main/docs/CONTRIBUTING.md).

## License

Apache License 2.0 — see [LICENSE](https://github.com/RBLN-SW/torch-rbln/blob/main/LICENSE) and [NOTICE](https://github.com/RBLN-SW/torch-rbln/blob/main/NOTICE).

## Contact

- **Community:** [discuss.rebellions.ai](https://discuss.rebellions.ai/)
- **Email:** [support@rebellions.ai](mailto:support@rebellions.ai)
