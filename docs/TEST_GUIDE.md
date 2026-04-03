# Test Guide

This document is the comprehensive guide for testing the `torch-rbln` project — the RBLN NPU backend for PyTorch, registered via the [`PrivateUse1`](https://pytorch.org/docs/stable/notes/extending.html#privateuse1) extension mechanism.

It covers the test infrastructure, how to run and write tests, available fixtures and markers, and how to debug failures.

---

## Prerequisites

Make sure you are in a virtual environment with `torch-rbln` installed. If you installed with `--no-deps`, install the test dependencies first:

```bash
./tools/test/install-test-deps.sh
```

This installs:
- **Test runner:** [`pytest`](https://docs.pytest.org/), [`pytest-xdist`](https://pytest-xdist.readthedocs.io/) (parallel execution)
- **Test infra:** [`expecttest`](https://github.com/ezyang/expecttest) (required by `torch.testing._internal`)
- **Model-test dependencies:** torchvision (from PyTorch CPU index), transformers, optimum-rbln, pandas

Model tests may also require access to external model artifacts. In many OSS
environments they are best treated as optional/manual rather than baseline
bring-up tests.

---

## 1. Test Frameworks

### Python Tests — pytest

Python tests use [`pytest`](https://docs.pytest.org/) — the de facto standard Python testing framework. pytest provides powerful features including test discovery, fixtures, parametrization, markers, and a rich plugin ecosystem. The project also uses [`pytest-xdist`](https://pytest-xdist.readthedocs.io/) for parallel test execution across multiple workers.

In addition to pytest, tests extensively use the **PyTorch internal test framework** (`torch.testing._internal`) for device-type instantiation, dtype parametrization, and operator-level testing (`OpInfo`).

### C++ Tests — Google Test

C++ tests use [**Google Test**](https://google.github.io/googletest/) (gtest) — Google's C++ testing and mocking framework. Tests are built with CMake and discovered at runtime via `gtest_discover_tests()`. Each test executable is linked against `torch_rbln` and `GTest::gtest_main`.

---

## 2. Test Directory Structure

```
test/
├── run_tests.py  # Unified test runner (CI/release, suite selection, parallel/serial splitting)
├── conftest.py   # Global pytest fixtures (deterministic seeding, Dynamo reset, env-var isolation)
├── filters.py    # Op-test filters (PrivateUse1 dispatch key parsing, skipped tests, unsupported ops)
├── utils.py      # Shared test utilities (seed helpers, device-count skip markers, SUPPORTED_DTYPES)
│
├── rbln/                                  # RBLN backend-specific tests
│   ├── test_custom_kernel.py              # RBLN custom kernels
│   ├── test_device_mapping.py             # Device mapping and topology APIs
│   ├── test_find_and_load_tvm_library.py  # Error handling when librbln.so cannot be found
│   ├── test_graph_eager_mode.py           # Numerical agreement between torch.compile graph mode and eager mode
│   ├── test_internal_op_utils.py          # Internal op utilities
│   ├── test_llama_ops.py                  # Core ops used in LLaMA-family models
│   ├── test_multi_device.py               # Multi-device tensor movement and cross-device operations
│   ├── test_non_zero_storage_offset.py    # Correct handling of tensors with non-zero storage offsets
│   ├── test_op_caching.py                 # Operator caching / graph-reuse behavior
│   ├── test_rbln_apis.py                  # RBLN Python APIs
│   ├── test_registered_ops.py             # All natively registered and fallback ops from RBLNRegisterOps.cpp / register_ops.py
│   ├── test_sdpa_decode_overflow.py       # SDPA decode-phase overflow detection and fallback behavior
│   ├── test_tensor_copy.py                # Tensor copy operations across directions (H2D, D2H, D2D) and dtypes
│   ├── test_tensor_memory.py              # Device memory allocation and lifetime management
│   ├── test_torch_compile_patch.py        # torch.compile patch helpers
│   └── _closed_tests/
│       └── test_host_precision_cast.py    # Host-side precision-cast correctness (float16 vs. custom_float16)
│
├── internal/                              # Internal subsystem tests
│   └── test_memory_stats.py               # Memory statistics APIs
│
├── distributed/                           # Distributed tests
│   ├── benchmark_collective_ops.py        # Collective-communication latency / throughput benchmarks
│   ├── test_process_group.py              # ProcessGroupRBLN collective ops
│   └── test_tp_pp.py                      # Tensor-parallel (TP) and pipeline-parallel (PP) end-to-end tests
│
├── ops/                                   # PyTorch operator compatibility tests
│   └── test_ops.py                        # Adapted from upstream PyTorch test_ops.py — validates op correctness on RBLN
│
├── models/                                # Model-level integration tests
│   ├── requirements.txt                   # Extra dependencies for model tests
│   ├── test_optimum_llm.py                # LLM inference via optimum-rbln
│   └── test_transformers.py               # Transformers model profiling and trace-pattern analysis
│
└── cpp/                                   # C++ unit tests (Google Test)
    ├── CMakeLists.txt
    ├── core/                              # Core C++ tests
    │   ├── RBLNAllocatorTest.cpp          # Device memory allocator
    │   ├── RBLNCopyOpTest.cpp             # Copy operations
    │   ├── RBLNDeviceGuardTest.cpp        # Device guard semantics
    │   ├── RBLNFunctionsTest.cpp          # Miscellaneous RBLN functions
    │   └── RBLNTensorUtilsTest.cpp        # Tensor utility helpers
    └── c10d/
        └── ProcessGroupRBLNTest.cpp       # ProcessGroup C++ tests
```

### Key Components

| File           | Purpose                                                                                                                                              |
|----------------|------------------------------------------------------------------------------------------------------------------------------------------------------|
| `run_tests.py` | Unified test runner — handles suite selection, CI/release marker filtering, and parallel/serial worker splitting                                     |
| `conftest.py`  | Auto-use fixtures applied to every test: deterministic seeding, TorchDynamo reset, and `TORCH_RBLN_DISABLE_FALLBACK` env-var isolation               |
| `filters.py`   | Parses `native_functions.yaml` for `PrivateUse1` dispatch keys; defines which upstream op tests to skip                                              |
| `utils.py`     | Shared helpers — `SUPPORTED_DTYPES`, `set_deterministic_seeds()`, `requires_logical_devices`, `requires_physical_devices`, `run_in_isolated_process` |

---

## 3. Running Tests

### Python Tests

#### Using `run_tests.py`

`test/run_tests.py` is the primary entry point. It wraps pytest with the correct marker expressions and parallel/serial splitting.

```bash
# All CI tests (default)
python test/run_tests.py

# All release tests
python test/run_tests.py --test_mode=release

# Specific suites
python test/run_tests.py --suite=core
python test/run_tests.py --suite=distributed
python test/run_tests.py --suite=models
python test/run_tests.py --suite=ops

# Multiple suites
python test/run_tests.py --suite=core --suite=ops

# Adjust parallel workers (default: 16)
python test/run_tests.py --suite=core --workers=8
```

Each suite maps to specific test directories:

| Suite         | Directories                    | Notes                                                         |
|---------------|--------------------------------|---------------------------------------------------------------|
| `core`        | `test/internal/`, `test/rbln/` | RBLN backend and internal subsystem tests                     |
| `distributed` | `test/distributed/`            | Collective ops, tensor-parallel, and pipeline-parallel tests  |
| `models`      | `test/models/`                 | Automatically installs model-test dependencies before running |
| `ops`         | `test/ops/`                    | PyTorch operator compatibility tests via PrivateUse1          |

**How `run_tests.py` maps flags to markers:**

```text
--test_mode=ci (default)
            → -m "test_set_ci and single_worker"  (serial, --numprocesses=1)
              + -m "test_set_ci and not single_worker"  (parallel, --numprocesses=16)

--test_mode=release
            → -m "not (test_set_experimental or test_set_perf) and single_worker"  (serial, --numprocesses=1)
              + -m "not (test_set_experimental or test_set_perf) and not single_worker"  (parallel, --numprocesses=16)
```

#### Running Individual Tests with pytest

```bash
# By directory
python -m pytest test/rbln/
python -m pytest test/ops/

# By file
python -m pytest test/rbln/test_rbln_apis.py

# By class (note: PrivateUse1 suffix appended by instantiate_device_type_tests)
python -m pytest test/rbln/test_rbln_apis.py::TestDeviceManagementAPIsPRIVATEUSE1

# By method
python -m pytest test/rbln/test_rbln_apis.py::TestDeviceManagementAPIsPRIVATEUSE1::test_device_count_rbln

# Filter by marker
python -m pytest test/rbln/ -m "test_set_ci"
python -m pytest test/rbln/ -m "test_set_ci and not single_worker"
python -m pytest test/distributed/ -m "test_set_perf"

# Filter by keyword
python -m pytest test/models/test_transformers.py -k "batch_size_2"
python -m pytest test/ -m "test_set_ci" -k "not float32"

# Parallel execution
python -m pytest test/rbln/ --numprocesses=16

# Serial execution (required for some tests)
python -m pytest test/internal/test_memory_stats.py --numprocesses=1

# Collect tests without running (verify selection)
python -m pytest test/rbln/ --co -q

# Show stdout/stderr + verbose + stop on first failure
python -m pytest test/rbln/test_graph_eager_mode.py -s -v -x
```

#### Installing Test Dependencies

```bash
# Install everything needed to run the full suite
./tools/test/install-test-deps.sh

# Preview what would be installed (dry-run)
./tools/test/install-test-deps.sh --dry-run

# Use uv instead of pip
UV=1 ./tools/test/install-test-deps.sh
```

### C++ Tests

C++ tests use [Google Test](https://google.github.io/googletest/). They are compiled during the editable install and run via CTest:

```bash
# Build the C++ library and tests
uv pip install -e . --no-build-isolation

# Run all C++ tests via CTest
ctest --test-dir ./build

# Run tests matching a pattern
ctest --test-dir ./build -R "Allocator"

# Verbose output
ctest --test-dir ./build --verbose
```

---

## 4. pytest Configuration

All pytest configuration lives in `pyproject.toml` under `[tool.pytest.ini_options]`.

### Default Options

```ini
addopts = "--verbose -rEfX --tb=native -p no:warnings --assert=plain --max-worker-restart=0"
testpaths = ["test"]
xfail_strict = true
```

| Option                   | Effect                                                                |
|--------------------------|-----------------------------------------------------------------------|
| `--verbose`              | Show each test name and result                                        |
| `-rEfX`                  | Extra summary for Error, failed, and Xpassed tests                    |
| `--tb=native`            | Standard Python traceback format                                      |
| `-p no:warnings`         | Disable warnings plugin for clean output                              |
| `--assert=plain`         | Disable assertion rewriting (avoids import-order conflicts)           |
| `--max-worker-restart=0` | Fail immediately on worker crash instead of restarting                |
| `xfail_strict = true`    | `@pytest.mark.xfail` tests that unexpectedly pass become failures     |

### Markers

| Marker                  | Description                                                                  | Usage                                       |
|-------------------------|------------------------------------------------------------------------------|---------------------------------------------|
| `test_set_ci`           | CI pipeline tests. Selected by default in `run_tests.py`.                    | `pytest -m "test_set_ci"`                   |
| `test_set_experimental` | Experimental feature tests. Excluded in `--test_mode=release`.               | `pytest -m "not test_set_experimental"`     |
| `test_set_perf`         | Performance / benchmark tests. Not included in CI or release by default.     | `pytest -m "test_set_perf"`                 |
| `single_worker`         | Tests that must run serially. `run_tests.py` splits execution automatically. | `pytest -m "test_set_ci and single_worker"` |
| `no_dynamo_reset`       | Skips the autouse `reset_dynamo` fixture (TorchDynamo cache reset).          | `@pytest.mark.no_dynamo_reset`              |

For guidance on which marker to apply when writing a new test, see [Which Marker Should I Use?](#which-marker-should-i-use) in Section 8.

### Global Fixtures

These fixtures are defined in `test/conftest.py` and apply automatically to every test unless noted otherwise.

| Fixture                          | Scope    | Autouse | Description                                                                                                   |
|----------------------------------|----------|---------|---------------------------------------------------------------------------------------------------------------|
| `set_seeds`                      | function | **Yes** | Sets `torch.manual_seed(0)`, `np.random.seed(0)`, `random.seed(0)` for deterministic reproducibility          |
| `reset_dynamo`                   | function | **Yes** | Calls `torch._dynamo.reset()` before each test. Skipped if test is marked with `@pytest.mark.no_dynamo_reset` |
| `disable_compile_error_fallback` | function | **Yes** | Appends `compile_error` to `TORCH_RBLN_DISABLE_FALLBACK` env var                                              |
| `enable_deploy_mode`             | function | No      | Sets `TORCH_RBLN_DEPLOY=ON`. Apply with `@pytest.mark.usefixtures("enable_deploy_mode")`                      |
| `enable_eager_malloc`            | function | No      | Sets `TORCH_RBLN_EAGER_MALLOC=1`. Apply with `@pytest.mark.usefixtures("enable_eager_malloc")`                |

---

## 5. PrivateUse1 Test Framework

`torch-rbln` registers the RBLN NPU as a PyTorch `PrivateUse1` backend. This section explains how the PyTorch testing infrastructure supports custom backends and how `torch-rbln` leverages it.

### Device-Type Test Instantiation

The PyTorch test framework (`torch.testing._internal`) automatically generates device-specific test classes from templates. The key mechanism is `instantiate_device_type_tests()`:

1. Write a template class inheriting from `TestCase` with methods that accept a `device` parameter.
2. Call `instantiate_device_type_tests(TestClass, globals(), only_for="privateuse1")` at module scope.
3. The framework generates concrete classes like `TestFooPRIVATEUSE1` with methods suffixed `_rbln`.

**All test files in this project use `only_for="privateuse1"`** to generate tests exclusively for the RBLN backend, avoiding unnecessary CPU/CUDA variants.

```python
# This call at the bottom of every test file:
instantiate_device_type_tests(TestRegisteredNativeOps, globals(), only_for="privateuse1")

# Generates: TestRegisteredNativeOpsPRIVATEUSE1
# With methods: test_copy_from_rbln, test_empty_memory_format_rbln, etc.
```

### Key Decorators

#### From `torch.testing._internal`

| Decorator                      | Description                                                                          |
|--------------------------------|--------------------------------------------------------------------------------------|
| `@parametrize("name", values)` | Generates independent test cases per value. Composes with device-type instantiation. |
| `@dtypes(*dtypes)`             | Generates dtype-specialized variants (e.g., `test_foo_rbln_float16`).                |
| `@ops(op_db)`                  | Instantiates the test for every OpInfo × dtype × device combination.                 |
| `@suppress_warnings`           | Suppresses warnings during test execution.                                           |

#### From `test/utils.py`

| Decorator / Utility                    | Description                                                                                                                     |
|----------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|
| `@requires_logical_devices(N)`         | `pytest.mark.skipif` — skips if `torch.rbln.device_count() < N`.                                                                |
| `@requires_physical_devices(N)`        | `pytest.mark.skipif` — skips if `torch.rbln.physical_device_count() < N`.                                                       |
| `SUPPORTED_DTYPES`                     | `[torch.float16]` — the baseline dtype list for RBLN backend tests.                                                             |
| `set_deterministic_seeds(seed)`        | Sets torch, numpy, and random seeds. Must be called in each spawned process for reproducibility.                                |
| `run_in_isolated_process(func, *args)` | Runs `func` in a fresh `spawn`-method process. Useful when tests need clean process state. `func` and `args` must be picklable. |

#### pytest Markers as Decorators

| Decorator                                         | Description                                            |
|---------------------------------------------------|--------------------------------------------------------|
| `@pytest.mark.test_set_ci`                        | Include in CI pipeline runs.                           |
| `@pytest.mark.test_set_perf`                      | Mark as performance/benchmark test.                    |
| `@pytest.mark.single_worker`                      | Force serial execution.                                |
| `@pytest.mark.no_dynamo_reset`                    | Opt out of the autouse `reset_dynamo` fixture.         |
| `@pytest.mark.usefixtures("enable_deploy_mode")`  | Enable `TORCH_RBLN_DEPLOY=ON` for the test class.      |
| `@pytest.mark.usefixtures("enable_eager_malloc")` | Enable `TORCH_RBLN_EAGER_MALLOC=1` for the test class. |

### Op Filtering for RBLN

`test/filters.py` provides `custom_instantiate_device_type_tests()` — a wrapper around `instantiate_device_type_tests()` that automatically skips tests for operators not supported on the RBLN backend:

- Parses `aten/src/ATen/native/native_functions.yaml` for ops with `PrivateUse1` dispatch keys
- Skips tests for ops not in the parsed list
- Skips unsupported dtypes (only `SUPPORTED_DTYPES` are allowed, with exceptions for CPU-fallback dtype tests like SDPA)
- Skips specific known-failing tests listed in `_skipped_tests`
- Skips forward AD tests for ops listed in `_ops_without_forward_ad_support`
- In debug builds (`torch.version.debug`), runs only a representative subset of ops for faster iteration

```python
# Used in test/ops/test_ops.py instead of plain instantiate_device_type_tests:
from test.filters import custom_instantiate_device_type_tests

custom_instantiate_device_type_tests(TestCommon, globals(), only_for="privateuse1")
```

---

## 6. Writing Tests — Templates and Best Practices

### Minimal Test Template

The simplest pattern for a new RBLN backend test:

```python
# Owner(s): ["module: PrivateUse1"]

"""Test suite for <feature description>."""

import pytest
import torch
from torch.testing._internal.common_device_type import dtypes, instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase

from test.utils import SUPPORTED_DTYPES


@pytest.mark.test_set_ci
class TestMyFeature(TestCase):
    """Test <feature description> on the RBLN PrivateUse1 backend."""

    @dtypes(*SUPPORTED_DTYPES)
    def test_basic_op(self, dtype):
        x = torch.randn([4, 4], dtype=dtype, device="rbln")
        y = torch.randn([4, 4], dtype=dtype, device="rbln")
        result = x + y
        expected = x.cpu() + y.cpu()
        self.assertEqual(result.cpu(), expected)


instantiate_device_type_tests(TestMyFeature, globals(), only_for="privateuse1")

if __name__ == "__main__":
    run_tests()
```

> **Key points:**
> - Inherit from `TestCase` (from `torch.testing._internal.common_utils`).
> - Mark with `@pytest.mark.test_set_ci` for inclusion in CI pipelines.
> - Use `@dtypes(*SUPPORTED_DTYPES)` to parametrize over RBLN-supported dtypes.
> - Call `instantiate_device_type_tests(..., only_for="privateuse1")` at module scope to generate RBLN-only test classes.
> - End with `if __name__ == "__main__": run_tests()`.

### Test Template with Fixtures

When tests need specific environment configurations, use `@pytest.mark.usefixtures`:

```python
@pytest.mark.test_set_ci
@pytest.mark.usefixtures("enable_deploy_mode")
class TestDeployFeature(TestCase):
    """Tests that require TORCH_RBLN_DEPLOY=ON."""

    @dtypes(*SUPPORTED_DTYPES)
    @parametrize("shape", [(2, 3), (4, 5, 6), (8, 8)])
    def test_registered_op(self, dtype, shape):
        x = torch.randn(shape, dtype=dtype, device="rbln")
        result = torch.ops.aten.clone(x)
        self.assertEqual(x.cpu(), result.cpu())


instantiate_device_type_tests(TestDeployFeature, globals(), only_for="privateuse1")
```

### Multi-Device Test Template

For tests that require multiple RBLN devices:

```python
from test.utils import requires_logical_devices, SUPPORTED_DTYPES


@pytest.mark.test_set_ci
@pytest.mark.usefixtures("enable_deploy_mode")
class TestMultiDevice(TestCase):

    @requires_logical_devices(2)
    @dtypes(*SUPPORTED_DTYPES)
    def test_tensor_move_between_devices(self, dtype):
        """Test moving tensors between RBLN devices."""
        device0 = torch.device("rbln:0")
        device1 = torch.device("rbln:1")

        x = torch.randn([4, 4], dtype=dtype, device=device0)
        y = x.to(device1)

        self.assertEqual(y.device, device1)
        self.assertEqual(x.cpu(), y.cpu())


instantiate_device_type_tests(TestMultiDevice, globals(), only_for="privateuse1")
```

### Deploy Mode / Eager Mode Test Template

Tests that require `TORCH_RBLN_DEPLOY=ON` or `TORCH_RBLN_EAGER_MALLOC=1`:

```python
@pytest.mark.test_set_ci
@pytest.mark.single_worker
@pytest.mark.usefixtures("enable_eager_malloc")
class TestMemoryStats(TestCase):
    """Tests for memory statistics APIs. Requires eager malloc and serial execution."""

    def test_memory_allocated(self):
        device = "rbln:0"
        initial = torch.rbln.memory_allocated(device)
        x = torch.empty([1024], dtype=torch.float16, device=device)
        allocated = torch.rbln.memory_allocated(device)
        self.assertGreater(allocated, initial)
        del x
        # Memory may not be freed immediately due to caching allocator
```

### Custom Kernel Test Template

For testing custom RBLN operators registered via `torch.library.custom_op`:

```python
@pytest.mark.test_set_ci
@pytest.mark.no_dynamo_reset  # Preserve custom op registrations across tests
class TestCustomKernel(TestCase):

    @dtypes(*SUPPORTED_DTYPES)
    def test_custom_op(self, device, dtype):
        # Call the custom operator
        result = torch.ops.rbln_custom_ops.my_custom_op(
            torch.randn([4, 4], dtype=dtype, device=device)
        )
        self.assertEqual(result.shape, (4, 4))


instantiate_device_type_tests(TestCustomKernel, globals(), only_for="privateuse1")
```

> **Note:** `@pytest.mark.no_dynamo_reset` is essential when tests register custom ops — `torch._dynamo.reset()` would clear the registrations.

### C++ Test Template (Google Test)

C++ tests register the `privateuse1` backend as `"rbln"` and use Google Test fixtures:

```cpp
#include <c10/core/Allocator.h>
#include <c10/rbln/RBLNFunctions.h>
#include <gtest/gtest.h>

class RBLNMyFeatureTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    c10::register_privateuse1_backend("rbln");
    ASSERT_TRUE(c10::is_privateuse1_backend_registered());
    ASSERT_EQ(c10::get_privateuse1_backend(true), "rbln");
    ASSERT_GE(c10::rbln::get_device_count(), 1);
  }

  void SetUp() override {
    c10::rbln::set_device_index(0);
  }
};

TEST_F(RBLNMyFeatureTest, BasicOperation) {
  auto* allocator = c10::GetAllocator(c10::kPrivateUse1);
  EXPECT_NE(allocator, nullptr);

  const auto data = allocator->allocate(1024);
  EXPECT_NE(data.get(), nullptr);
  EXPECT_TRUE(data.device().is_privateuseone());
}
```

To add a new C++ test:
1. Create the `.cpp` file under `test/cpp/`.
2. Add it to `test/cpp/CMakeLists.txt` in the `sources` list.
3. Rebuild with `uv pip install -e . --no-build-isolation`.

### Parametrization Best Practices

Use `@parametrize` from `torch.testing._internal.common_utils` — **not** `pytest.mark.parametrize`. The PyTorch decorator integrates with `instantiate_device_type_tests()` and produces correctly suffixed test names.

#### Prefer `@parametrize` over `for` loops

```python
# Bad — loop hides which batch_size failed
def test_linear(self, device):
    for batch_size in [1, 2, 4]:
        out = model(torch.randn(batch_size, 8, device=device))
        self.assertEqual(out.shape[0], batch_size)

# Good — each batch_size is a separate test case
@parametrize("batch_size", [1, 2, 4])
def test_linear(self, device, batch_size):
    out = model(torch.randn(batch_size, 8, device=device))
    self.assertEqual(out.shape[0], batch_size)
```

#### Stack decorators for cross-product parametrization

```python
@parametrize("dtype", [torch.float16, torch.float32])
@parametrize("batch_size", [1, 2, 4])
@parametrize("seq_len", [16, 128])
def test_forward(self, device, dtype, batch_size, seq_len):
    ...
# Generates 2 × 3 × 2 = 12 independent tests per device type
```

#### Verify parametrized test names

```bash
python -m pytest test/rbln/test_registered_ops.py --co -q
# Expected:
# test_clone_shape_(2, 3)_rbln
# test_clone_shape_(5, 10)_rbln
# ...
```

---

## 7. Debugging Test Failures

### Useful Debugging Options

| Option                   | What It Does                                                                      |
|--------------------------|-----------------------------------------------------------------------------------|
| `-s`, `--capture=no`     | Disables output capture — lets you see `print()` statements and logs in real time |
| `-v`, `--verbose`        | Increases verbosity                                                               |
| `-x`, `--exitfirst`      | Stops on the first failure                                                        |
| `--maxfail=N`            | Stops after N failures                                                            |
| `--collect-only`, `--co` | Collects tests without running them (useful for verifying selection)              |
| `--durations=N`          | Reports the N slowest tests                                                       |

#### Examples

```bash
# Full output + verbose + stop on first failure
python -m pytest test/rbln/test_graph_eager_mode.py -s -v -x

# List collected tests without running them
python -m pytest test/rbln/ --co

# Show the 10 slowest tests
python -m pytest test/rbln/ --durations=10
```

### Common Failure Scenarios

#### Multi-Device Tests Skipped

```
SKIPPED: Requires at least 2 logical devices, found 1
```

**Cause:** The test requires multiple RBLN devices (via `@requires_logical_devices` or `@requires_physical_devices`), but fewer are available.

**Fix:** Run on a machine with enough devices. Check with `torch.rbln.device_count()` or `torch.rbln.physical_device_count()`.

#### Missing Model-Test Dependencies

```
ModuleNotFoundError: No module named 'optimum.rbln'
```

**Cause:** The extra packages required by model tests are not installed.

**Fix:** Run the dependency installer:
```bash
./tools/test/install-test-deps.sh
```

#### Numerical Tolerance Mismatch

```
AssertionError: Tensor-likes are not close!
```

**Cause:** The RBLN result exceeds the test's `atol`/`rtol` tolerance compared to the CPU reference.

**Fix:** Check whether the op has known precision limitations and adjust tolerances if appropriate.

#### `pytest-xdist` Worker Crash

```
INTERNALERROR> worker 'gw0' crashed
```

**Cause:** A parallel worker process crashed. The project sets `--max-worker-restart=0`, so any crash fails the suite immediately.

**Fix:** Re-run the offending test file in single-worker mode to get a clearer traceback:
```bash
python -m pytest test/rbln/test_problem_file.py --numprocesses=1 -s -v
```

#### TorchDynamo Internal Error

```
torch._dynamo.exc.InternalTorchDynamoError: ...
```

**Cause:** An error during `torch.compile` graph capture.

**Fix:** The `reset_dynamo` fixture calls `torch._dynamo.reset()` before each test, so running the test in isolation usually resolves stale-state issues. If the error persists, check whether the test is marked with `@pytest.mark.no_dynamo_reset` and inspect the captured graph for unsupported patterns.

---

## 8. CI/CD Integration

Tests in this project run automatically as part of the CI/CD pipeline via GitHub Actions. Understanding how the automated workflows interact with the test infrastructure helps you write tests that work correctly in both local and CI environments.

For the full workflow architecture, see [Workflows](WORKFLOWS.md).

### How Tests Run in CI

The [CI workflow](WORKFLOWS.md#ci-workflow) triggers on every pull request (except those targeting `main`) and on pushes to `dev`. It runs `run_tests.py` in CI mode:

```bash
python test/run_tests.py  # -m "test_set_ci"
```

This means only tests marked with `@pytest.mark.test_set_ci` are selected. **If you write a new test and want it to run in CI, you must add the `@pytest.mark.test_set_ci` marker.**

### How Tests Run in Release

The [Release workflow](WORKFLOWS.md#release-workflow) triggers on pull requests to `main` and on pushes to `main`. It runs `run_tests.py` in release mode:

```bash
python test/run_tests.py --test_mode=release  # -m "not (test_set_experimental or test_set_perf)"
```

This selects a broader set of tests — everything except tests marked `test_set_experimental` or `test_set_perf`.

### Which Marker Should I Use?

When writing a new test, choose the marker based on when the test should run:

| Test Type          | Marker                               | When It Runs                              | Guideline                                                                    |
|--------------------|--------------------------------------|-------------------------------------------|------------------------------------------------------------------------------|
| CI tests           | `@pytest.mark.test_set_ci`           | Every PR to `dev`                         | Default choice — most tests should use this marker                           |
| Release tests      | *(no marker)*                        | PRs to `main`                             | For tests too slow or resource-intensive for every PR, but needed at release |
| Performance tests  | `@pytest.mark.test_set_perf`         | Manual only (`pytest -m "test_set_perf"`) | Benchmarks and latency/throughput measurements                               |
| Experimental tests | `@pytest.mark.test_set_experimental` | CI (with `test_set_ci` marker) or manual  | Early-stage features — excluded from Release to avoid blocking releases      |

> **How this works:** CI mode runs `pytest -m "test_set_ci"`, selecting only `test_set_ci`-marked tests. Release mode runs `pytest -m "not (test_set_experimental or test_set_perf)"`, which includes all `test_set_ci`-marked tests *plus* unmarked tests, but excludes `test_set_experimental` and `test_set_perf`. The two modes overlap but neither is a strict superset of the other — a test marked with both `@pytest.mark.test_set_ci` and `@pytest.mark.test_set_experimental` will run in CI but not in Release. In practice, **most tests should be marked `@pytest.mark.test_set_ci`**. Omit the marker only when a test is intentionally too slow for per-commit CI but still valuable for pre-release validation.
>
> **Experimental tests in CI:** A test marked with both `@pytest.mark.test_set_ci` and `@pytest.mark.test_set_experimental` will run in CI but be excluded from Release. This is useful for validating in-progress features on every PR without risking release stability.

### Marker Set Relationships

Test set markers operate on two orthogonal dimensions:

- **Stability stage**: `test_set_experimental` ($E$) vs. release-eligible ($R$) — intended to be mutually exclusive by convention
- **Execution scope**: `test_set_ci` ($C$), `test_set_perf` ($P$), or unmarked — determines which workflow selects the test

In the intended design, these dimensions satisfy three invariants:
* $E \cap R = \emptyset$
* $C = (C \cap E) \;\cup\; (C \cap R)$
* $P = (P \cap E) \;\cup\; (P \cap R)$

The first expresses the design intent that experimental and release-eligible tests form **disjoint partitions** — test authors should not mark a test as both. The second and third express that every CI or perf test should belong to exactly one stability stage; there is no third category that escapes both partitions.

```text
┌──── All Tests ───────────────────────────────────────────────────┐
│  E (experimental)  ┃    R (release-eligible)                     │
│                    ┃                                             │
│  ┌─────────────────╂─────────────────────────────────────────┐   │
│  │     C ∩ E       ┃               C ∩ R                     │ C │
│  │    CI only      ┃           CI + Release                  │   │
│  └─────────────────╂─────────────────────────────────────────┘   │
│  ┌─────────────────╂─────────────────────────────────────────┐   │
│  │     P ∩ E       ┃               P ∩ R                     │ P │
│  └─────────────────╂─────────────────────────────────────────┘   │
│   unmarked ∩ E     ┃         unmarked ∩ R                        │
│                  E ∩ R = ∅                                       │
└──────────────────────────────────────────────────────────────────┘
```

| Region              | Markers                                   | CI | Perf | Release |
|---------------------|-------------------------------------------|:--:|:----:|:-------:|
| $C \cap E$          | `test_set_ci` + `test_set_experimental`   | ✅ | ❌   | ❌      |
| $C \cap R$          | `test_set_ci`                             | ✅ | ❌   | ✅      |
| $P \cap E$          | `test_set_perf` + `test_set_experimental` | ❌ | ✅   | ❌      |
| $P \cap R$          | `test_set_perf`                           | ❌ | ✅   | ❌      |
| unmarked $\cap\; E$ | `test_set_experimental`                   | ❌ | ❌   | ❌      |
| unmarked $\cap\; R$ | *(none)*                                  | ❌ | ❌   | ✅      |

> **Reading the table:** CI mode selects $C$ (`-m "test_set_ci"`), so only rows containing $C$ run. Release mode selects $\neg E \land \neg P$ (`-m "not (test_set_experimental or test_set_perf)"`), so it picks up $C \cap R$ and unmarked $\cap\; R$ — everything stable that isn't a benchmark. The diagonal pattern (✅ in CI ↔ ❌ in Release for $C \cap E$) is the mechanism that lets experimental features get CI validation without blocking releases.

### Remote Execution

All CI and release tests execute on remote infrastructure with access to
physical RBLN NPU devices. Results are reported back as PR status checks. This
means:

- Tests that require NPU hardware **will** have device access in CI.
- Test failures appear directly on the pull request as failed status checks.
- You can manually trigger a workflow run via `workflow_dispatch` if you need to re-run tests for a specific commit.
