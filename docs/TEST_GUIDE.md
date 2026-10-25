# Test Guide

This guide covers how to run, write, and debug tests for `torch-rbln`, the RBLN
NPU backend for PyTorch registered through the
[`PrivateUse1`](https://docs.pytorch.org/tutorials/advanced/privateuseone.html)
extension mechanism.

---

## Prerequisites

Install `torch-rbln` in the project environment, then install the test
dependencies into it:

```bash
uv sync \
  --group test \
  --inexact \
  --locked \
  --no-default-groups \
  --no-install-project
```

This installs:

- **Test runner**: [`pytest`](https://docs.pytest.org/),
  [`pytest-xdist`](https://pytest-xdist.readthedocs.io/) (parallel execution)
- **Test infra**: [`expecttest`](https://github.com/ezyang/expecttest) (required
  by `torch.testing._internal`)
- **Op-test dependencies**: [`pyyaml`](https://pyyaml.org/)
- **Model-test dependencies**: `transformers` 4.49.0 (the version the RBLN
  PyTorch tutorial pins)

Model tests may also need external model artifacts.

---

## Test frameworks

- **Python**: [`pytest`](https://docs.pytest.org/) with
  [`pytest-xdist`](https://pytest-xdist.readthedocs.io/) for parallel workers,
  plus the PyTorch test framework (`torch.testing._internal`) for device-type
  instantiation, dtype parametrization, and `OpInfo` op tests.
- **C++**: [GoogleTest](https://google.github.io/googletest/), built with CMake
  and discovered with `gtest_discover_tests()`. Each test executable links
  against `torch_rbln` and `GTest::gtest_main`.

---

## Directory layout

<!-- rumdl-disable MD013 -->
```text
test/
├── conftest.py   # Global pytest fixtures (deterministic seeding, Dynamo reset, env-var isolation)
├── filters.py    # Op-test filters (PrivateUse1 dispatch key parsing, skipped tests, unsupported ops)
├── utils.py      # Shared test utilities (seed helpers, SUPPORTED_DTYPES)
│
├── rbln/                                  # RBLN backend-specific tests
│   ├── test_abi_check.py                  # rebel ABI handshake between this build and the loaded librbln.so
│   ├── test_custom_kernel.py              # RBLN custom kernels
│   ├── test_custom_op_compile.py          # Paged-attention custom ops inside a larger compiled graph (engine graph mode)
│   ├── test_device_mapping.py             # Device mapping and topology APIs
│   ├── test_graph_eager_mode.py           # Numerical agreement between torch.compile graph mode and eager mode
│   ├── test_internal_op_utils.py          # Internal op utilities
│   ├── test_llama_ops.py                  # Core ops used in LLaMA-family models
│   ├── test_memory_stats.py               # Memory statistics APIs
│   ├── test_multi_device.py               # Multi-device tensor movement and cross-device operations
│   ├── test_non_zero_storage_offset.py    # Correct handling of tensors with non-zero storage offsets
│   ├── test_op_caching.py                 # Operator caching / graph-reuse behavior
│   ├── test_privateuse1_contract.py       # PrivateUse1 backend contract conformance (one test per upstream clause)
│   ├── test_rbln_apis.py                  # RBLN Python APIs
│   ├── test_rbln_runtime_lib.py           # librbln.so resolution order and single-copy/version checks
│   ├── test_registered_ops.py             # All natively registered and fallback ops from RBLNRegisterOps.cpp / register_ops.py
│   ├── test_rsd_kv_cache.py               # Persistent KV cache under RSD: two programs + eager access on a buffer sharded across NPUs
│   ├── test_sdpa_decode_overflow.py       # SDPA decode-phase overflow detection and fallback behavior
│   ├── test_tensor_copy.py                # Tensor copy operations across directions (H2D, D2H, D2D) and dtypes
│   ├── test_tensor_memory.py              # Device memory allocation and lifetime management
│   ├── test_torch_compile_patch.py        # torch.compile patch helpers
│   └── _closed_tests/
│       └── test_host_precision_cast.py    # Host-side precision-cast correctness (float16 vs. custom_float16)
│
├── distributed/                           # Distributed tests
│   ├── test_process_group.py              # ProcessGroupRBLN collective ops
│   └── test_tp_pp.py                      # Tensor-parallel (TP) and pipeline-parallel (PP) end-to-end tests
│
├── ops/                                   # PyTorch operator compatibility tests
│   └── test_ops.py                        # Adapted from upstream PyTorch test_ops.py — validates op correctness on RBLN
│
├── models/                                # Model-level integration tests
│   └── test_transformers.py               # Causal-LM logits vs an fp32 CPU reference (transformers)
│
└── cpp/                                   # C++ unit tests (GoogleTest)
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
<!-- rumdl-enable MD013 -->

---

## Run Python tests

Run `pytest` through `uv run --no-sync`, which uses the project environment
without syncing it, so the editable build stays in place.

### Run a suite the way CI does

CI tests each suite with two `pytest` runs on its directory: the tests marked
`serial` on one worker, then the rest in parallel. For the `core` suite in the
pre-merge checks:

```bash
uv run --no-sync pytest \
  test/rbln \
  -m "(not periodic) and serial" \
  --numprocesses=1

uv run --no-sync pytest \
  test/rbln \
  -m "(not periodic) and (not serial)" \
  --numprocesses=32
```

Release checks drop `(not periodic) and`. CI also splits a suite into steps by
NPU count, each with both runs (see [Python tests](PIPELINES.md#python-tests)).
The `core` suite runs `test/rbln`; `distributed`, `models`, and `ops` run
`test/<suite>`, and `models` needs the
[model-test dependencies](#prerequisites). CI counts a `pytest` run that
collects nothing (exit code 5) as no failure.

### Run individual tests

```bash
# By directory
uv run --no-sync pytest \
  test/rbln/

# By file
uv run --no-sync pytest \
  test/rbln/test_rbln_apis.py

# By class (note: PrivateUse1 suffix appended by instantiate_device_type_tests)
uv run --no-sync pytest \
  test/rbln/test_rbln_apis.py::TestDeviceManagementAPIsPRIVATEUSE1

# By method
uv run --no-sync pytest \
  test/rbln/test_rbln_apis.py::TestDeviceManagementAPIsPRIVATEUSE1::test_device_count_rbln

# Filter by marker
uv run --no-sync pytest \
  test/rbln/ \
  -m "not periodic"

# Filter by keyword
uv run --no-sync pytest \
  test/models/test_transformers.py \
  -k "b2_s1024"

# Show stdout/stderr + verbose + stop on first failure
uv run --no-sync pytest \
  test/rbln/test_graph_eager_mode.py \
  --capture=no \
  --verbose \
  --exitfirst
```

---

## Run C++ tests

C++ tests build during the editable install and run with CTest:

```bash
# Build the C++ library and tests
uv pip install \
  -e . \
  --no-build-isolation

# Run all C++ tests via CTest
ctest \
  --test-dir ./build

# Run tests matching a pattern
ctest \
  --test-dir ./build \
  -R "Allocator"

# Verbose output
ctest \
  --test-dir ./build \
  --verbose
```

---

## `pytest` configuration

All `pytest` configuration lives in `pyproject.toml` under
`[tool.pytest.ini_options]`.

### Default options

<!-- rumdl-disable MD013 -->
```toml
[tool.pytest.ini_options]
addopts = "--verbose -rEfX --tb=native -p no:warnings --assert=plain --max-worker-restart=0 --strict-markers"
testpaths = ["test"]
xfail_strict = true
```
<!-- rumdl-enable MD013 -->

| Option                   | Effect                                                            |
| ------------------------ | ----------------------------------------------------------------- |
| `--verbose`              | Show each test name and result                                    |
| `-rEfX`                  | Extra summary for Error, failed, and Xpassed tests                |
| `--tb=native`            | Standard Python traceback format                                  |
| `-p no:warnings`         | Disable warnings plugin for clean output                          |
| `--assert=plain`         | Disable assertion rewriting (avoids import-order conflicts)       |
| `--max-worker-restart=0` | Fail immediately on worker crash instead of restarting            |
| `--strict-markers`       | A marker not registered under `markers` fails collection          |
| `xfail_strict = true`    | `@pytest.mark.xfail` tests that unexpectedly pass become failures |

### Markers

The suite registers these markers in `pyproject.toml`:

| Marker            | Description                                                                                     |
| ----------------- | ----------------------------------------------------------------------------------------------- |
| `periodic`        | Runs only in release checks.                                                                    |
| `serial`          | Runs on a single worker and fails at setup when run in parallel.                                |
| `npu(count=N)`    | Requires N NPUs, where N is 1, 2, 4, or 8. When N is 0, runs only without an NPU.               |
| `no_dynamo_reset` | Keeps the TorchDynamo cache from earlier tests by skipping the autouse `reset_dynamo` fixture.  |
| `torch_rbln_only` | Has an outcome that does not depend on `rebel-compiler`, so `rebel-compiler`'s CI deselects it. |

To make a test require NPUs, mark it with `npu(count=N)`, not with
`deviceCountAtLeast` from `torch.testing._internal`, which breaks device-type
test generation in every file collected after it and silently drops most tests.
Follow these rules for `npu(count=N)`:

- Pass `count` as a keyword, and round any other count up to 1, 2, 4, or 8.
- Use `count=0` only for a test that asserts the no-device state.
- Leave a test that runs with or without an NPU unmarked. CI runs it only in an
  NPU-less step (see [Python tests](PIPELINES.md#python-tests)).

For more information about when to use `periodic` and `torch_rbln_only`, see
[CI integration](#ci-integration).

### Fixtures

These fixtures are defined in `test/conftest.py` and apply automatically to
every test unless noted otherwise.

| Fixture                          | Scope    | Autouse | Description                                                                                                                                  |
| -------------------------------- | -------- | ------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `set_seeds`                      | function | **Yes** | Sets `torch.manual_seed(0)`, `np.random.seed(0)`, `random.seed(0)` for deterministic reproducibility                                         |
| `reset_dynamo`                   | function | **Yes** | Calls `torch._dynamo.reset()` before each test. Skipped if test is marked with `@pytest.mark.no_dynamo_reset`                                |
| `disable_compile_error_fallback` | function | **Yes** | Appends `compile_error` to `TORCH_RBLN_DISABLE_FALLBACK` env var                                                                             |
| `reset_caching_allocator`        | function | **Yes** | Drains in-flight device work then releases the RBLN caching-allocator blocks in teardown; fails loud and skips the flush if the drain faults |
| `restore_current_device`         | function | **Yes** | Restores the selected RBLN device and lazy-init flag after each test so a `set_device()` can't leak into later tests                         |
| `restore_current_stream`         | function | **Yes** | Restores every device's current stream after each test so a `set_stream()` can't leak into later tests                                       |
| `enable_deploy_mode`             | function | No      | Sets `TORCH_RBLN_DEPLOY=ON`. Apply with `@pytest.mark.usefixtures("enable_deploy_mode")`                                                     |
| `enable_eager_malloc`            | function | No      | Sets `TORCH_RBLN_EAGER_MALLOC=1`. Apply with `@pytest.mark.usefixtures("enable_eager_malloc")`                                               |

---

## `PrivateUse1` test framework

### Device-type test instantiation

The PyTorch test framework (`torch.testing._internal`) automatically generates
device-specific test classes from templates. The key mechanism is
`instantiate_device_type_tests()`:

1. Write a template class inheriting from `TestCase` with methods that accept a
   `device` parameter.
2. Call
   `instantiate_device_type_tests(TestClass, globals(), only_for="privateuse1")`
   at module scope.
3. The framework generates concrete classes like `TestFooPRIVATEUSE1`, with
   `_rbln` added to each method name before any dtype suffix.

**New device-type tests use `only_for="privateuse1"`** to generate tests
exclusively for the RBLN backend, avoiding unnecessary CPU/CUDA variants.

Since that leaves exactly one device type, the instantiation earns its keep
through the axes it expands — `@dtypes`, `@parametrize`, or a `device` argument
— and it is required for any test that uses one.

**Do not wrap a test that never touches a device**. The call deletes the
template class and rebuilds it per device type, and the RBLN base is only in
that list when `torch.rbln.is_available()` is true. On a host without an NPU the
template is gone and nothing replaces it, so the file collects zero tests — no
failure, no skip. This costs nothing for a test that requires an NPU. A wrapped
test that never touches a device, however, never runs in CI: only an NPU-less
step selects it, and there the template generates no tests. Tests asserting
behavior with no device present (`test_dummy_device.py`, `test_no_device.py`) or
through a subprocess (`test_import_rbln_devices_seal.py`) are plain `TestCase`.

```python
# This call at the bottom of a device-type test file:
instantiate_device_type_tests(TestRegisteredNativeOps, globals(), only_for="privateuse1")

# Generates: TestRegisteredNativeOpsPRIVATEUSE1
# With methods such as test_copy_from_shape0_rbln_float16
```

### PyTorch decorators

From `torch.testing._internal`:

| Decorator                      | Description                                                                          |
| ------------------------------ | ------------------------------------------------------------------------------------ |
| `@parametrize("name", values)` | Generates independent test cases per value. Composes with device-type instantiation. |
| `@dtypes(*dtypes)`             | Generates dtype-specialized variants (e.g., `test_foo_rbln_float16`).                |
| `@ops(op_db)`                  | Instantiates the test for every OpInfo × dtype × device combination.                 |
| `@suppress_warnings`           | Suppresses warnings during test execution.                                           |

### Project helpers

From `test/utils.py`:

| Utility                                | Description                                                                                                                     |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `SUPPORTED_DTYPES`                     | Baseline dtype list for RBLN backend tests, derived from `torch_rbln._internal.ops_utils.SupportedDtypes.dispatch`.             |
| `set_deterministic_seeds(seed)`        | Sets `torch`, `numpy`, and `random` seeds. Must be called in each spawned process for reproducibility.                          |
| `run_in_isolated_process(func, *args)` | Runs `func` in a fresh `spawn`-method process. Useful when tests need clean process state. `func` and `args` must be picklable. |

### Op filtering

`test/filters.py` provides `custom_instantiate_device_type_tests()` — a wrapper
around `instantiate_device_type_tests()` that automatically skips tests for
operators not supported on the RBLN backend:

- Parses `aten/src/ATen/native/native_functions.yaml` for ops with `PrivateUse1`
  dispatch keys
- Skips tests for ops not in the parsed list
- Skips unsupported dtypes (only `SUPPORTED_DTYPES` are allowed, with exceptions
  for CPU-fallback dtype tests like SDPA)
- Skips specific known-failing tests listed in `_skipped_tests`
- Skips forward AD tests for ops listed in `_ops_without_forward_ad_support`
- In debug builds (`torch.version.debug`), runs only a representative subset of
  ops for faster iteration

```python
# Used in test/ops/test_ops.py instead of plain instantiate_device_type_tests:
from test.filters import custom_instantiate_device_type_tests

custom_instantiate_device_type_tests(TestCommon, globals(), only_for="privateuse1")
```

### Contract conformance tests

`test/rbln/test_privateuse1_contract.py` is organized differently from the rest
of the suite, on purpose: PyTorch does not merely _offer_ the `torch.rbln`
module and the RBLN accelerator hooks, it calls into them from paths that never
asked for an NPU. Each test pins exactly one clause upstream states and cites
its source, so a `torch` upgrade or a new call site fails on the clause rather
than on a downstream symptom.

Conventions specific to it:

- **Every probe runs in a fresh subprocess** (`run_probe`). It cannot use
  `run_in_isolated_process()` from `test/utils.py`, which needs a picklable
  callable, runs after `torch_rbln` is already imported, and captures no output
  — the probes must set `RBLN_*` _before_ the import and inspect stdout/stderr.
- **A clause not satisfied yet is a `strict=True` xfail** naming the work that
  closes it, rather than being omitted. `xfail_strict = true` is the project
  default, so an unexpected pass fails the suite and signals that the marker
  should be removed.
- **The remap verdict needs a reference measurement**. With one usable NPU a
  live mapping and a frozen one report the same device count, so `Probe.remap`
  is conclusive only when a pool scanned in a clean process shows two or more;
  `/dev/rbln*` is not that count, since a container can show more device nodes
  than the runtime can use. Scenarios name devices from this job's own selection
  instead of assuming id 0 is theirs.

---

## Write tests

### Minimal template

The simplest pattern for a new RBLN backend test:

```python
# Owner(s): ["module: PrivateUse1"]

"""Test suite for <feature description>."""

import pytest
import torch
from torch.testing._internal.common_device_type import dtypes, instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase

from test.utils import SUPPORTED_DTYPES


@pytest.mark.npu(count=1)
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

> **Key points**:
>
> - Inherit from `TestCase` (from `torch.testing._internal.common_utils`).
> - Add `@pytest.mark.periodic` only when the test meets one of the
> [criteria](#when-to-use-the-periodic-marker).
> - Mark the class with `@pytest.mark.npu(count=N)` as described in
> [Markers](#markers).
> - Use `@dtypes(*SUPPORTED_DTYPES)` to parametrize over RBLN-supported dtypes.
> - Call `instantiate_device_type_tests(..., only_for="privateuse1")` at module
>   scope to generate RBLN-only test classes.
> - End with `if __name__ == "__main__": run_tests()`.

### Template with fixtures

When tests need specific environment configurations, use
`@pytest.mark.usefixtures`:

```python
import pytest
import torch
from torch.testing._internal.common_device_type import dtypes, instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, TestCase

from test.utils import SUPPORTED_DTYPES


@pytest.mark.npu(count=1)
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

### Multi-device template

For tests that require multiple RBLN devices:

```python
import pytest
import torch
from torch.testing._internal.common_device_type import dtypes, instantiate_device_type_tests
from torch.testing._internal.common_utils import TestCase

from test.utils import SUPPORTED_DTYPES


@pytest.mark.npu(count=2)
@pytest.mark.usefixtures("enable_deploy_mode")
class TestMultiDevice(TestCase):

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

### Eager malloc template

For tests that need `TORCH_RBLN_EAGER_MALLOC=1`:

```python
import pytest
import torch
from torch.testing._internal.common_utils import TestCase


@pytest.mark.serial
@pytest.mark.npu(count=1)
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

> **Rule — never process-cache an env flag that tests toggle per-test**. xdist
> workers are long-lived processes that run many tests in sequence. A
> function-scoped fixture (`enable_deploy_mode`, `enable_eager_malloc`,
> `monkeypatch.setenv`, `patch.dict`) sets an env var for one test, but if
> C++/Python reads it _once_ and caches it in a `static`/`@lru_cache` for the
> process lifetime, the **first** value latches into the whole worker and leaks
> into every later test on it — an intermittent, order-dependent failure that
> only shows up when xdist happens to schedule the toggling test first (e.g. the
> `test_runtime_hidden_d2h` / eager-malloc flake). Read such flags **live**
> (`std::getenv` per call — see `is_deploy_mode()` /
> `is_nan_inf_check_disabled()` in `DispatchShim.cpp`), and if a Python
> `@lru_cache` gate must respond to per-test env, `cache_clear()` it in the
> fixture's teardown.
>
> The same applies to any other process-global state a test mutates. Restore it
> in the test's own teardown (e.g. `TestTorchCompileMonkeyPatch.tearDown`
> re-installs the original `torch.compile` wrappers it patched), or — for the
> selected RBLN device and the current stream — rely on the autouse
> `restore_current_device` / `restore_current_stream` guards in
> `test/conftest.py` that repair them after each test.

### Custom kernel template

For testing custom RBLN operators registered via `torch.library.custom_op`:

```python
import pytest
import rebel  # noqa: F401
import torch
from torch.testing._internal.common_device_type import dtypes, instantiate_device_type_tests
from torch.testing._internal.common_utils import TestCase

from test.utils import SUPPORTED_DTYPES


@pytest.mark.npu(count=1)
@pytest.mark.no_dynamo_reset  # Keep the compile cache across tests
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

### C++ template

C++ tests register the `privateuse1` backend as `"rbln"` and use GoogleTest
fixtures:

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

CI runs all the tests of a binary in one process, whereas `ctest` runs each test
in its own process, so a test must not depend on starting in a fresh process.

### Parametrization

Use `@parametrize` from `torch.testing._internal.common_utils` — **not**
`pytest.mark.parametrize`. The PyTorch decorator integrates with
`instantiate_device_type_tests()` and produces correctly suffixed test names.

#### Prefer parametrization over loops

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

#### Stack decorators for a cross product

```python
@parametrize("dtype", [torch.bfloat16, torch.float16])
@parametrize("batch_size", [1, 2, 4])
@parametrize("seq_len", [16, 128])
def test_forward(self, device, dtype, batch_size, seq_len):
    ...
# Generates 2 × 3 × 2 = 12 independent tests per device type
```

#### Verify test names

```bash
uv run --no-sync pytest \
  test/rbln/test_registered_ops.py \
  --collect-only \
  --quiet
```

The output lists each generated test, such as
`<TestCaseFunction test_clone_shape0_rbln_float16>`.

---

## Troubleshoot test failures

### Multi-device tests skipped

```text
SKIPPED (NPUs required: 2, available: 1)
```

**Cause**: The test is marked `npu(count=N)`, but fewer than N NPUs are
available, so `test/conftest.py` skips it.

**Fix**: Run the test on a machine with at least N NPUs.

### Missing model-test dependencies

```text
ModuleNotFoundError: No module named 'transformers'
```

**Cause**: The extra packages required by model tests are not installed.

**Fix**: Install them as described in [Prerequisites](#prerequisites).

### Numerical tolerance mismatch

```text
AssertionError: Tensor-likes are not close!
```

**Cause**: The RBLN result exceeds the test's `atol`/`rtol` tolerance compared
to the CPU reference.

**Fix**: Check whether the op has known precision limitations and adjust
tolerances if appropriate.

### Worker crash

```text
worker 'gw0' crashed while running 'test/rbln/test_problem_file.py::<test>'
```

**Cause**: A parallel worker process crashed. The project sets
`--max-worker-restart=0`, so xdist does not replace the worker and stops the
`pytest` run.

**Fix**: To see the crash's traceback and output, rerun the test file without
xdist:

```bash
uv run --no-sync pytest \
  test/rbln/test_problem_file.py \
  --numprocesses=0 \
  --capture=no \
  --verbose
```

### TorchDynamo internal error

```text
torch._dynamo.exc.InternalTorchDynamoError: ...
```

**Cause**: An error during `torch.compile` graph capture.

**Fix**: The `reset_dynamo` fixture calls `torch._dynamo.reset()` before each
test, so running the test in isolation usually resolves stale-state issues. If
the error persists, check whether the test is marked with
`@pytest.mark.no_dynamo_reset` and inspect the captured graph for unsupported
patterns.

---

## CI integration

The `periodic` and `torch_rbln_only` markers each take a test out of some CI
runs, so use them only as the following sections describe.

### When to use the `periodic` marker

Only when one of these holds:

- It reruns, in a second configuration, a path that every check covers, such as
  the `TORCH_RBLN_C10D_ASYNC=1` subtests in `test/distributed/`.
- It is the rest of a parametrize sweep whose representative case every check
  runs, such as the non-representative shapes in `test_transformers.py`.
- It needs hardware or time too costly for every pull request, such as
  `test_v2v_cross_chiplet.py`.

Do not mark the representative case `periodic`; otherwise the pre-merge and
post-merge checks stop covering its path.

### When to use the `torch_rbln_only` marker

`rebel-compiler` runs this suite on its own pull requests, to see whether a
compiler or runtime change regresses `torch-rbln`. It has a hard per-job time
budget, so it should run the tests that answer that question and not the rest.

A test belongs in that lane when its outcome depends on `librbln` or the `rbln`
compile backend: device allocation, copies and layouts, compiled graphs and
their artifacts, the custom op schemas, device topology and runtime state,
numerics on device, the runtime counters the profiler reads. That is the
default, and it is most of this suite — a new test needs no marker to be
included.

Mark a test `torch_rbln_only` when a different `rebel-compiler` would not change
its outcome:

- it pins an upstream **`torch`** clause or a torch-facing contract, so a
  `torch` upgrade is what breaks it (`test_privateuse1_contract.py`);
- the thing under test is a `torch-rbln` gate that the test drives itself, with
  the runtime stubbed or forced (`test_runtime_unavailable.py`);
- it tests packaging — where `librbln.so` is found, not what it does once loaded
  (`test_rbln_runtime_lib.py`). Import behavior is the counter-example: whether
  a remap after import is still accepted is the runtime's doing, so
  `test_import_rbln_devices_seal.py` stays in the lane;
- it asserts an absence at the `torch` level (`test_amp_autocast.py`);
- it tests only a `torch-rbln` surface whose behavior `rebel-compiler` covers on
  its own side (`test_file_offloading.py`);
- it tests the harness itself (`test_isolation_guards.py`).

Mark it on the module (`pytestmark = pytest.mark.torch_rbln_only`) with a
comment saying which of those it is; the reason is the reviewable part, not the
marker. `torch-rbln`'s own lanes run these tests as before — the marker only
removes them from someone else's gate.

Cost is part of the judgment, in one direction only: a file that costs nothing
stays in the lane even when its subject is arguably ours, because the risk of
dropping a signal outweighs the seconds. `test_torch_compile_patch.py` is mostly
helper unit tests and still runs there — two of its cases drive a real compile
through the rbln backend, and the whole file is a fraction of a second.
