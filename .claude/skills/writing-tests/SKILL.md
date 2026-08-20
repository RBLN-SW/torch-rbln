---
name: writing-tests
description: Use when adding or modifying tests under test/ — where the file goes, the markers that decide whether it runs at all, how to assert an op reached the device rather than the host fallback, the process-state traps in this suite, and how to prove a new test fails before the fix.
---

# Writing tests in torch-rbln

## 0. Read the guide first

`docs/TEST_GUIDE.md` is the contract for this suite, not background reading. Before writing, read the sections that apply to what you are adding: §2 for where the file goes, §5 for the PrivateUse1 device-type framework, §6 for the templates and parametrization rules, §8 for which marker the test needs. A test that ignores it gets sent back in review even when it passes.

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
| `test/internal/`    | Internal subsystems reached through `torch_rbln._internal`              |
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

**Instantiate only what has an axis.** `instantiate_device_type_tests` expands `@dtypes`, `@parametrize`, or a `device` argument, and is required for a test that uses one. Do not wrap a test that never touches a device: the call deletes the template class and rebuilds it per device type, and on a host with no NPU nothing replaces it — the file then collects zero tests, with no failure and no skip. An import guard, a subprocess check, or a contract assertion is a plain `TestCase`; follow a neighbour like `test_dummy_device.py` or `test_import_rbln_devices_seal.py`. `docs/TEST_GUIDE.md` §5 covers the mechanism.

## 4. Markers decide whether the test runs at all

| Marker                 | Effect                                                                  |
| ---------------------- | ----------------------------------------------------------------------- |
| `test_set_ci`          | Runs on every PR to `dev`. **Without it your test runs in no PR check.** |
| *(none)*               | Release only — PRs to `main`                                            |
| `test_set_perf`        | Manual only                                                             |
| `test_set_experimental`| Excluded from release                                                    |
| `single_worker`        | Runs in the serial pass, which is still xdist at `--numprocesses=1`      |
| `no_dynamo_reset`      | Opts out of the autouse `torch._dynamo.reset()`                         |

Default to `test_set_ci`. Omit it only for a test too slow for per-commit CI, and say so in the PR.

Mark `single_worker` when the test mutates state shared with other tests on the same worker — device selection, an environment variable, a global cache, or a fixed port. Several dozen tests already carry it; read one near your target before deciding you do not need it.

## 5. Process state leaks between tests

This is where tests in this repository most often lie.

- **A flag read into `static` or `@lru_cache` state latches for the whole process.** A `monkeypatch.setenv` after that first read changes nothing, and the value you did set can outlive the test on an xdist worker and silently change an unrelated later one — that is a real CI flake this suite has already paid for, and `docs/TEST_GUIDE.md` §6 records the rule: such flags are read live. Use `run_in_isolated_process` (`test/utils.py`) whenever the behavior under test depends on an env var, a singleton, or a fresh device count.
- **The device mapping freezes on first real use.** Queries like `device_count()` only plan it; an allocation, a `synchronize()`, or a collective commits it and later edits to `RBLN_DEVICES` are ignored. So a test that uses a device changes what every later test on that worker sees, and the autouse `restore_current_device` fixture restores the selected index, not that commit.
- **The suite force-disables the `compile_error` fallback** (`test/conftest.py`), so a compile failure raises here even though it would fall back to CPU in production. Do not re-enable it to make a test pass.
- **A failed drain poisons the rest of the worker.** Teardown synchronizes every device after each test; if that synchronize raises, your test errors in teardown and every remaining test on that worker skips rather than run against a faulted device. So a `RuntimeError: RBLN device drain failed` plus a wall of skips points at the test that ran before them, not at them. Do not paper over it with a retry.

## 6. Assertions

Assertion rewriting is off (`--assert=plain` in `pyproject.toml`), so a bare `assert a == b` prints no values when it fails. Use `self.assertEqual`, `torch.testing.assert_close`, or pass a message.

**To assert an op ran on the device, assert it — do not infer it from the values.** An input that is not fp16 or bf16, or whose last dimension is not a multiple of 64, takes the host path, and the result then matches the CPU reference exactly; the test passes without ever reaching the kernel. `test/rbln/test_dispatch_shim_precheck.py` is the worked example from both directions: `_C._warmcache_size()` before and after, because a CPU fallback never primes the warm cache. `_C._dispatch_fallback_by_op()` gives per-op fallback counts for the same purpose.

`xfail_strict = true`: an `xfail` that starts passing fails the suite. `xfail_rebel` / `xfail_atom` (`test/utils.py`) are the strict, architecture-conditional markers; `_REBEL_XFAILS` in `test/conftest.py` applies `xfail_rebel` by test name and detects its own stale keys. A failure that is not architecture-specific does not get an xfail.

## 7. Tolerances are derived, not tuned

Comparing against a CPU reference, state where the tolerance comes from: the dtype's mantissa, the accumulation length, the op family. `MATMUL_TOLERANCES` in `test/rbln/test_registered_ops.py` is the worked example. Widening a tolerance until a test passes is how a real numerical regression gets committed.

Exact comparison of generated token IDs is not a numerical test. At bf16, a truncated model produces near-tie logits and ordinary run-to-run nondeterminism flips the argmax, so the test fails on a healthy backend. Compare logits with a tolerance, or compare tokens with a stated allowance.

If a test is flaky, find out what varies. Do not re-run it until it is green.

## 8. Prove the test fails

A test that passes without the change covers nothing.

```bash
git stash list                              # note what is already there
git stash push -- <the production files you changed>
uv run --no-sync pytest <your new test> -x   # must fail
git stash pop                               # verify it restored what you stashed
```

Stash only the paths your change touched, and check `git stash list` before and after — the working tree here often carries unrelated edits, and losing one of those costs more than the check is worth. If the tree is dirty enough that this is risky, run the test against the parent commit in a separate worktree instead.

For a C++ change the stash is not enough — rebuild after the stash, and rebuild again after the pop, or you are testing the same binary twice.

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
- A test diff larger than the code change it covers
