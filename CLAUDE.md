# CLAUDE.md — torch-rbln

## Project Overview

**torch-rbln** is a PyTorch extension that integrates Rebellions NPU (neural processing unit) compute into PyTorch via the `PrivateUse1` backend mechanism. It enables eager mode execution (define-by-run), graph compilation via `torch.compile(..., backend="rbln")`, and distributed training on RBLN NPUs using familiar PyTorch APIs (`torch.rbln`, `torch.device("rbln")`).

The project is in **beta** with active development. APIs may change between releases.

**Repository:** `RBLN-SW/torch-rbln`
**License:** Apache 2.0
**Python:** 3.10–3.13
**PyTorch:** 2.9.1+cpu

---

## Repository Structure

```
torch-rbln/
├── torch_rbln/              # Main Python package
│   ├── __init__.py          # Entry point: backend registration, native lib loading
│   ├── device/              # RBLN device implementation (device.py, context_holder.py)
│   ├── _internal/           # Op registration, kernels, monkey patches, utilities
│   │   ├── register_ops.py           # (generated) operator registration
│   │   ├── register_custom_ops.py    # Eager-mode custom op implementations
│   │   ├── register_backward_ops.py  # Backward pass implementations
│   │   ├── ops_utils.py              # CPU fallback, output tensor management
│   │   ├── monkey_patches.py         # torch.compile patching
│   │   └── kernels/                  # Custom kernels (SDPA, transpose)
│   ├── csrc/rbln/           # C++ Python bindings (Module.cpp)
│   │   └── distributed/c10d/rbln/  # ProcessGroupRBLN (distributed ops)
│   ├── diagnose.py          # Runtime diagnostics
│   ├── memory.py            # Memory management
│   └── lib/                 # Built native libraries (libc10_rbln.so, libtorch_rbln.so)
│
├── c10/rbln/                # C++ c10 device backend (allocator, generator, guards, fallback)
├── aten/src/ATen/native/    # ATen operator implementations
│   ├── RBLNRegisterOps.cpp  # Central C++ operator registration (~80+ ops)
│   ├── native_functions.yaml # YAML operator definitions
│   └── rbln/                # Copy, resize, tensor factories, utilities
│
├── test/                    # Test suite (see Testing section)
├── tools/                   # Build tools, code generation, linter adapters
│   ├── dev-setup.sh         # Development environment setup
│   ├── codegen/             # Code generation from YAML op definitions
│   ├── linter/              # Lintrunner adapter scripts
│   └── test/                # Test dependency installation
│
├── docs/                    # Developer documentation
│   ├── CONTRIBUTING.md      # Contribution guidelines
│   ├── LINTING.md           # Linting setup
│   ├── TEST_GUIDE.md        # Comprehensive testing guide
│   ├── CONFIGURATION.md     # Runtime configuration / env vars
│   └── WORKFLOWS.md         # CI/CD pipeline documentation
│
├── cmake/                   # CMake find modules (Torch, Rebel, GTest, etc.)
├── third_party/             # rebel-compiler headers
├── .github/workflows/       # CI/CD workflows
├── pyproject.toml           # Build config, dependencies, tool settings
├── CMakeLists.txt           # C++ build (C++17, Ninja)
├── hatch_build.py           # Custom hatchling hook for CMake build
└── .lintrunner.toml         # Linter configuration (20+ linters)
```

---

## Build & Development Setup

### Initial Setup

```bash
git clone https://github.com/RBLN-SW/torch-rbln.git
cd torch-rbln
uv venv .venv && source .venv/bin/activate
./tools/dev-setup.sh pypi          # Standard setup
./tools/dev-setup.sh pypi --clean  # Clean rebuild (removes build/)
```

### Build System

- **Python build:** Hatchling with custom CMake hook (`hatch_build.py`)
- **C++ build:** CMake 3.18+ with Ninja generator, C++17 standard
- **Version:** Git-tag based via setuptools-scm (`tools/get_version.py`)
- **Code generation:** `tools/run_codegen.py` generates `torch_rbln/_internal/register_ops.py` from YAML operator definitions

### Package Indexes

Three package indexes are used (configured in `pyproject.toml`):
1. **PyPI** (default) — `https://pypi.org/simple/`
2. **PyTorch CPU** (explicit) — `https://download.pytorch.org/whl/cpu` (for `torch`)
3. **Rebellions** (explicit) — `https://pypi.rbln.ai/simple/` (for `rebel-compiler`)

### Key Build Environment Variables

| Variable | Description |
|----------|-------------|
| `TORCH_RBLN_BUILD_TYPE` | CMake build type (default: `Release`) |
| `TORCH_RBLN_DEPLOY` | Enable deploy mode (`ON`) |
| `REBEL_HOME` | Path to external rebel-compiler installation |

---

## Testing

### Quick Reference

```bash
# Install test dependencies
./tools/test/install-test-deps.sh

# Run CI tests (default)
python test/run_tests.py

# Run release tests
python test/run_tests.py --test_mode=release

# Run specific suites
python test/run_tests.py --suite=core        # test/internal/, test/rbln/
python test/run_tests.py --suite=distributed # test/distributed/
python test/run_tests.py --suite=ops         # test/ops/
python test/run_tests.py --suite=models      # test/models/ (installs deps first)

# Run individual tests with pytest
python -m pytest test/rbln/test_rbln_apis.py -s -v
python -m pytest test/rbln/ -m "test_set_ci"
python -m pytest test/rbln/ --numprocesses=16

# C++ tests
ctest --test-dir ./build
ctest --test-dir ./build -R "Allocator"
```

### Test Markers

| Marker | Purpose |
|--------|---------|
| `@pytest.mark.test_set_ci` | Core tests that run on every PR (must always pass) |
| `@pytest.mark.test_set_experimental` | Early-stage feature tests (opt-in in CI) |
| `@pytest.mark.test_set_perf` | Performance benchmarks (manual only) |
| `@pytest.mark.single_worker` | Must run serially (distributed, memory stats) |
| `@pytest.mark.no_dynamo_reset` | Skip autouse TorchDynamo reset fixture |

### Test Conventions

- Tests use `pytest` with `pytest-xdist` for parallel execution (default: 16 workers)
- PyTorch internal test framework (`torch.testing._internal`) for device-type instantiation and dtype parametrization
- **Auto-use fixtures** (in `test/conftest.py`): deterministic seeding, TorchDynamo reset, `TORCH_RBLN_DISABLE_FALLBACK=compile_error`
- Test classes inherit from `torch.testing._internal.common_utils.TestCase`
- Tests use `@dtypes(*SUPPORTED_DTYPES)` and `@parametrize()` decorators
- `SUPPORTED_DTYPES = [torch.float16]` — the primary supported dtype
- Tolerances: `ATOL=0.01, RTOL=0.01` for float16 comparisons
- Test class names get a `PRIVATEUSE1` suffix appended by `instantiate_device_type_tests`
- Use `run_in_isolated_process()` for tests requiring clean state
- All test files start with comment: `# Owner(s): ["module: PrivateUse1"]`

### Test File Structure

```python
# Owner(s): ["module: PrivateUse1"]
import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import TestCase

@pytest.mark.test_set_ci
class TestExample(TestCase):
    @dtypes(*SUPPORTED_DTYPES)
    def test_something(self):
        ...

instantiate_device_type_tests(TestExample, globals(), only_for="privateuse1")

if __name__ == "__main__":
    run_tests()
```

---

## Linting & Code Style

### Setup

```bash
source .venv/bin/activate
lintrunner init              # Initialize once (downloads tools)
```

### Running Linters

```bash
lintrunner -m main -a        # Lint changed files vs main, auto-fix
```

### Style Rules

- **Line length:** 120 characters (all formatters)
- **Python formatting:** Black (line-length=120, target-version=py39)
- **Import sorting:** isort (profile=black)
- **Python linting:** Ruff, Flake8 (with bugbear, comprehensions, logging-format, simplify, TorchFix plugins)
- **Type checking:** mypy (disallow_untyped_defs=True)
- **C++ formatting:** clang-format
- **C++ linting:** clang-tidy
- **CMake linting:** cmakelint

The project uses **lintrunner** to coordinate 20+ specialized linters (see `.lintrunner.toml`). A pre-commit hook runs linting automatically on `git commit`.

---

## Git & PR Conventions

### Branch Strategy

- `dev` — integration branch; PRs target here
- `main` — always release-ready; tagged for releases
- Feature branches: created from `dev`
- Hotfix branches: created from `main`

### Commit Messages

**Conventional Commits v1.0.0** format is required for PR titles (enforced by CI):

```
<type>(<optional scope>): <description>
```

Allowed types: `feat`, `fix`, `refactor`, `perf`, `docs`, `test`, `build`, `ci`, `chore`

Subject must start lowercase and not end with a period. Examples:
```
feat: add support for new operator
fix: correct tensor copy for non-contiguous layouts
refactor: simplify CPU fallback path
```

Individual commit messages within a PR branch don't require strict Conventional Commits but should be readable. All PRs are **squash-merged**.

### PR Requirements

- Title follows Conventional Commits
- Linked to a corresponding issue
- All CI tests pass
- Linting passes
- At least one approval from relevant team
- Description includes: purpose, related issue, affected modules, testing method

---

## Architecture & Key Patterns

### Initialization Flow

When torch-rbln loads (via `torch.backends` entry point — no explicit import needed):

1. `torch_backends_entry_point()` called
2. Native extensions loaded (`libc10_rbln.so`, `libtorch_rbln.so`)
3. `PrivateUse1` backend renamed to `"rbln"`
4. Device module registered (`torch.rbln.*`)
5. Operators registered (C++ dispatch + Python eager ops)
6. Monkey patches applied (`torch.compile`)
7. Distributed c10d bindings initialized

### Operator Registration (Two Layers)

**C++ layer** (`aten/src/ATen/native/RBLNRegisterOps.cpp`):
- Uses `TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)` to register ops with PyTorch dispatch
- Implements copy, resize, tensor factories natively
- Provides CPU fallback for unsupported ops

**Python layer** (`torch_rbln/_internal/register_custom_ops.py`):
- Wraps ops in `torch.nn.Module`, compiles with `torch.compile(..., backend="rbln")`
- Handles CPU fallback via `is_cpu_fallback_cases()` / `cpu_fallback_path()`
- Manages output tensors with `out_tensor_context()` / `finalize_output_tensor()`

### CPU Fallback

Unsupported operations gracefully fall back to CPU execution by default. This behavior is controlled by:
- `TORCH_RBLN_DISABLE_FALLBACK` env var (categories: `compile_error`, `non_blocking_copy`, `unsupported_op`, `all`)
- `RBLNFallbackConfig` in C++ (`c10/rbln/RBLNFallbackConfig.cpp`)
- `is_cpu_fallback_cases()` in Python (`torch_rbln/_internal/ops_utils.py`)

### Adding New Operators

1. **Native C++ ops:** Register in `aten/src/ATen/native/RBLNRegisterOps.cpp` using `TORCH_LIBRARY_IMPL`
2. **Eager-mode Python ops:** Add to `torch_rbln/_internal/register_custom_ops.py` following the existing pattern (Module class + compile + fallback)
3. **Backward passes:** Add to `torch_rbln/_internal/register_backward_ops.py`
4. **YAML definitions:** Update `aten/src/ATen/native/native_functions.yaml` and regenerate with `tools/run_codegen.py`

### Key Design Patterns

| Pattern | Location | Purpose |
|---------|----------|---------|
| PrivateUse1 dispatch | `RBLNRegisterOps.cpp` | Route PyTorch ops to RBLN backend |
| CPU fallback | `ops_utils.py`, `RBLNCPUFallback.cpp` | Graceful degradation for unsupported ops |
| Eager compilation | `register_custom_ops.py` | Compile individual ops at runtime via torch.compile |
| Out-tensor optimization | `context_holder.py`, `ops_utils.py` | Reuse user-provided output tensors, avoid copies |
| Monkey patching | `monkey_patches.py` | Lazy backend registration, compiled function wrapping |
| Reentrancy guard | `torch_compile_patch_helpers.py` | Prevent infinite recursion in nested dispatches |

---

## Runtime Configuration

Key environment variables (see `docs/CONFIGURATION.md` for full details):

| Variable | Description | Default |
|----------|-------------|---------|
| `TORCH_RBLN_LOG_LEVEL` | Log verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) | `WARNING` |
| `TORCH_RBLN_LOG_PATH` | Log file path (debug builds only) | `./torch_rbln.log` |
| `TORCH_RBLN_DEPLOY` | Skip NaN/Inf validation for production | Off |
| `TORCH_RBLN_DISABLE_FALLBACK` | Comma-separated fallback categories to disable | None |
| `TORCH_RBLN_EAGER_MALLOC` | Enable eager memory allocation | Off |
| `RBLN_NPUS_PER_DEVICE` | NPUs per logical device (1, 2, 4, 8, 16, 32) | 1 |
| `RBLN_DEVICE_MAP` | Explicit NPU-to-device mapping | Auto |
| `TORCH_RBLN_USE_TP_FAILOVER` | Auto-retry with tp_size=1 on TP failure | Off |
| `TORCH_RBLN_USE_DEVICE_TP` | Eager ops follow device group TP size | Off |

---

## Dependencies

### Core Runtime
- `torch == 2.9.1+cpu`
- `rebel-compiler >= 0.10.2, < 0.20.0` (separate install from RBLN index)
- `PyYAML >= 6.0`, `scipy >= 1.14.0`, `libcst >= 1.2.0`

### Build
- `cmake >= 3.18`, `ninja >= 1.11.1.3`
- `hatchling >= 1.25`, `setuptools-scm >= 8.0`
- `mypy >= 1.13.0`

### Test (install via `./tools/test/install-test-deps.sh`)
- `pytest`, `pytest-xdist`, `expecttest`
- `torchvision == 0.24.1+cpu`, `transformers == 4.57.6`
- `optimum-rbln == 0.10.1a1`, `pandas == 2.2.3`

---

## CI/CD

| Workflow | Trigger | Scope |
|----------|---------|-------|
| **CI** (`ci.yaml`) | PRs to `dev`, push to `dev` | Linting + `test_set_ci` tests |
| **Release** (`release.yaml`) | PRs to `main`, push to `main` | All tests except experimental/perf |
| **CD** (`cd.yaml`) | Version tags (`v*`) | Build + publish artifacts |
| **PR Title** (`check_pr_title.yaml`) | PR open/edit/sync | Conventional Commits validation |

All test workflows dispatch to remote infrastructure with physical RBLN NPU hardware.

---

## Important Notes for AI Assistants

- **Do not modify generated files** — `torch_rbln/_internal/register_ops.py` is generated by `tools/run_codegen.py` from YAML definitions. Edit the YAML or codegen templates instead.
- **C++ changes require rebuild** — After editing files in `c10/`, `aten/`, or `torch_rbln/csrc/`, rebuild with `./tools/dev-setup.sh pypi --clean` or `uv pip install -e . --no-build-isolation`.
- **rebel-compiler is external** — It is not part of this repo. It provides `librbln.so`, `rebel.core.torch_compile`, and `rebel.core.torch_eager`.
- **Tests require RBLN hardware** — Most tests need physical NPU devices. Tests that don't will be marked accordingly.
- **float16 is the primary dtype** — Most operations target `torch.float16`.
- **English only** — All issues, comments, code, and documentation must be in English.
- **120-char line limit** — Enforced across Python (Black/Ruff) and C++ (clang-format).
- **Squash merge only** — PRs are always squash-merged into the target branch.
