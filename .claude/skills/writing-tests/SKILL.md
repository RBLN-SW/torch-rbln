---
name: writing-tests
description: Use when adding or modifying tests under test/ — where the file goes, which marker, if any, a test needs, how to assert an op reached the device rather than the host fallback, the process-state traps in this suite, and how to prove a new test fails before the fix.
---

# Writing tests in torch-rbln

## 0. Read the guide first

`docs/TEST_GUIDE.md` is the contract for this suite, not background reading. Before writing, read the sections that apply to what you are adding: "Directory layout" for where the file goes, "PrivateUse1 test framework" for the device-type framework, "Write tests" for the templates and parametrization rules, and "Markers" and "CI integration" for which markers the test needs. A test that ignores it gets sent back in review even when it passes.

This skill covers what goes wrong on top of that — the traps the guide does not spell out.

## 1. Design before writing

Answer these before writing a line. If you cannot, ask instead of guessing.

1. What is the module under test for?
2. What is its input/output contract?
3. What failure is this test guarding against?
4. What is the cheapest level that catches that failure? Prefer an op-level test over a model-level one.

## 2. Where the file goes

| Directory           | Holds                                                                   |
| ------------------- | ----------------------------------------------------------------------- |
| `test/rbln/`        | Backend behavior — ops, device mapping, memory, copies, `torch.compile` |
| `test/ops/`         | Upstream op-compatibility tests. Adapted from PyTorch; change minimally |
| `test/distributed/` | ProcessGroup, TP/PP                                                     |
| `test/models/`      | Whole-model integration; needs extra dependencies and model artifacts   |
| `test/cpp/`         | Google Test, built by CMake                                             |

Extend the nearest existing file before adding a new one. Shared helpers go in `test/utils.py` or the `conftest.py` that already exists — do not add a `helpers_*.py`.

## 3. The file has a required shape

```python
# Owner(s): ["module: PrivateUse1"]
...
instantiate_device_type_tests(TestMyFeature, globals(), only_for="privateuse1")

if __name__ == "__main__":
    run_tests()
```

The `TESTOWNERS` and `TEST_HAS_MAIN` linters reject a Python test file missing either end. C++ tests are GTest under `test/cpp/` and none of this applies to them.

Check the `collected N items` line every time you run a file. CI treats a run that collects no tests as passing, so a green check does not prove that your test ran.

**Instantiate only what has an axis.** `instantiate_device_type_tests` expands `@dtypes`, `@parametrize`, or a `device` argument, and is required for a test that uses one. Do not wrap a test that never touches a device: the call deletes the template class and rebuilds it per device type, and on a host with no NPU nothing replaces it — the file then collects zero tests, with no failure and no skip. An import guard, a subprocess check, or a contract assertion is a plain `TestCase`; follow a neighbor like `test_dummy_device.py` or `test_import_rbln_devices_seal.py`. The "PrivateUse1 test framework" section of `docs/TEST_GUIDE.md` covers the mechanism.

## 4. Markers

Add a marker only when its condition holds:

- `periodic`: the test meets one of the `periodic` criteria in `docs/TEST_GUIDE.md`. The test then runs only in the release checks.
- `serial`: the test uses a resource shared across processes. It runs on one worker and fails at setup on more.
- `npu(count=N)`: the test requires N NPUs, rounded up to 1, 2, 4, or 8. For `count=0`, it requires that no NPU be present.
- `no_dynamo_reset`: the test is compile-heavy. The marker keeps the compile cache across tests, which the `reset_dynamo` autouse fixture would otherwise clear.

`serial` serializes but does not isolate: every `serial` test shares one worker process, so state leaks between them. Choose the isolation by the state, not by what a nearby test uses:

- **A resource shared across processes**, such as a device, a fixed port, or a file the runtime writes: `serial`.
- **State that restores**, such as an env var the code reads live or the selected device: a fixture or `monkeypatch`.
- **State that does not restore**, such as an import-time effect, a `static` or `@lru_cache` latch, or a committed device mapping: `run_in_isolated_process` (`test/utils.py`), or a `subprocess` as in `test_import_rbln_devices_seal.py`.

## 5. Process state leaks between tests

This is where tests in this repository most often lie.

- **A flag read into `static` or `@lru_cache` state latches for the whole process.** A `monkeypatch.setenv` after that first read changes nothing, and the value you did set can outlive the test on an xdist worker and silently change an unrelated later one — that is a real CI flake this suite has already paid for, and the "Write tests" section of `docs/TEST_GUIDE.md` records the rule: such flags are read live. Use `run_in_isolated_process` (`test/utils.py`) whenever the behavior under test depends on an env var, a singleton, or a fresh device count.
- **The device mapping freezes on first real use.** Queries like `device_count()` only plan it; an allocation, a `synchronize()`, or a collective commits it and later edits to `RBLN_DEVICES` are ignored. So a test that uses a device changes what every later test on that worker sees, and the autouse `restore_current_device` fixture restores the selected index, not that commit.
- **The suite force-disables the `compile_error` fallback** (`test/conftest.py`), so a compile failure raises here even though it would fall back to CPU in production. Do not re-enable it to make a test pass.
- **A failed drain poisons the rest of the worker.** Teardown synchronizes every device after each test; if that synchronize raises, your test errors in teardown and every remaining test on that worker skips rather than run against a faulted device. So a `RuntimeError: RBLN device drain failed` plus a wall of skips points at the test that ran before them, not at them. Do not paper over it with a retry.

## 6. Assertions

Assertion rewriting is off (`--assert=plain` in `pyproject.toml`), so a bare `assert a == b` prints no values when it fails. Use `self.assertEqual`, `torch.testing.assert_close`, or pass a message.

**To assert an op ran on the device, assert it — do not infer it from the values.** An input that is not fp16 or bf16 takes the host path on every route; in eager, a single-tensor op whose last dimension is not a multiple of 64 does too (`compile_and_run_view_aware` in `ops_utils.py`). The result then matches the CPU reference exactly; the test passes without ever reaching the kernel. `test/rbln/test_dispatch_shim_precheck.py` is the worked example from both directions: `_C._warmcache_size()` before and after, because a CPU fallback never primes the warm cache. `_C._dispatch_fallback_by_op()` gives per-op fallback counts for the same purpose, for the ops the C++ shim prechecks — not for the alignment route.

`xfail_strict = true`: an `xfail` that starts passing fails the suite. `xfail_rebel` / `xfail_atom` (`test/utils.py`) are the strict, architecture-conditional markers; `_REBEL_XFAILS` in `test/conftest.py` applies `xfail_rebel` by test name and detects its own stale keys. A failure that is not architecture-specific does not get an xfail.

## 7. Tolerances are derived, not tuned

Comparing against a CPU reference, state where the tolerance comes from: the dtype's mantissa, the accumulation length, the op family. `MATMUL_TOLERANCES` in `test/rbln/test_registered_ops.py` is the worked example. Widening a tolerance until a test passes is how a real numerical regression gets committed.

Exact comparison of generated token IDs is not a numerical test. At bf16, a truncated model produces near-tie logits and ordinary run-to-run nondeterminism flips the argmax, so the test fails on a healthy backend. Compare logits with a tolerance, or compare tokens with a stated allowance.

If a test is flaky, find out what varies. Do not re-run it until it is green.

## 8. Prove the test fails

A test that passes without the change covers nothing.

Run the new test against the code as it was before your change. The reliable way is a worktree at the base commit with the test files copied in:

```bash
git worktree add ../wt-base $(git merge-base HEAD origin/main)
git -C ../wt-base checkout $(git rev-parse HEAD) -- <your new test files>
# Set up and build ../wt-base as `AGENTS.md` describes, with no venv activated.
uv run --directory ../wt-base --no-sync python \
  -c 'import torch_rbln; print(torch_rbln.__file__)'  # must be ../wt-base

uv run --directory ../wt-base --no-sync pytest \
  <your new test> \
  --exitfirst  # must fail
```

A stash is a shortcut with conditions: the change is Python-only, nothing else runs from this tree, and you stash only the production paths you touched — `git stash push -- <paths>`, check `git stash list` before and after, and pop. The tree here often carries unrelated edits, and losing one costs more than the check is worth. For a native change the stash is never enough: the built `.so` is the same until you rebuild, so a stash without a rebuild on both sides tests one binary twice.

Watch it fail and read the failure; a test can fail for the wrong reason. If you cannot run it, say so explicitly instead of assuming it is red.

## 9. Reject these

- **An assertion that a constant would also satisfy.** `assert result >= 0` on a bridged call passes whether the wrapper forwards the runtime's value or returns a hardcoded `0`. Assert the value that distinguishes them.
- **A counter assertion whose condition the test does not pin.** Asserting that a hidden host copy happened is fine when the test forces it — `test_fallback_with_transfer` in `test/rbln/test_profiler.py` uses a non-contiguous int32 tensor, which cannot be borrowed, so the copy is a consequence of the input and the comment says so. It is wrong when the count depends on a runtime choice the test does not control: the same shape asserted on a fresh contiguous tensor turned into a CI flake once an allocation mode changed, because zero was also correct. Pin the condition, or assert the invariant instead — that every counted event is attributed.
- **A test that covers less than the change claims.** If the PR body says a capability set is derived, the test derives it; hardcoding the two dtypes you expect proves nothing about the derivation.
- Asserting a statically defined value against itself
- Testing that a function was called rather than what it did
- A negative test for logic that was removed
- Duplicating the implementation's logic inside the test
- Mocking a device operation that a real tensor would perform
- A placeholder test that is skipped
- A test that repeats an axis another test already covers, or restates the implementation — length alone is not the fault
