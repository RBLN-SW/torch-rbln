---
name: debugging
description: Use when investigating a bug, a failing test, or unexpected behavior in torch-rbln — the build and process-state traps that make a reproduction lie, how to tell a backend defect from a compiler defect, and the evidence rules that stop a wrong cause from looking confirmed.
---

# Debugging torch-rbln

Two things make a session go wrong here: reproducing something other than the reported bug, and confirming a cause that is not the cause. This skill is about those. It does not restate general debugging method.

## 1. Before you trust the reproduction

Get a reproducer before changing anything, and narrow it to the smallest input that still fails.

**Narrowing has a floor.** Shrink past it and the op stops reaching the device: `SUPPORTED_DTYPES` is fp16 and bf16 only, and a last dimension that is not a multiple of 64 takes the host path. Either way the result then matches CPU exactly, which reads as "the bug went away". Keep the dtype at fp16 or bf16, keep the last dimension aligned, and check after each narrowing step that the op still ran on the device.

**If you cannot reproduce it, stop and report that.** An untested fix for an unverified report costs a reviewer more than no patch.

**The minimal reproducer is a deliverable, not a step.** When the defect goes to another team, it ships as a standalone script that runs without installing a model, a dataset, or the test suite — a reproduction that requires the whole environment will not be run by the people who have to fix it. Keep narrowing until the script is short enough to paste.

**Three things tell you what actually happened**, and none of them are the returned values:

- `TORCH_RBLN_LOG_LEVEL=INFO` prints ```aten::<op>` op ran on CPU instead of RBLN`` for every fallback. This is the only signal that says whether the device path was taken.
- `torch.rbln.explain()` attributes hidden overhead — dispatch, CPU fallback, host copies — and `with_stack=True` gives the call site. `docs/EXPLAIN.md` is the reference; read it before writing your own instrumentation.
- `python -m torch_rbln.diagnose` (with `TORCH_RBLN_DIAGNOSE=1`) diagnoses library loading when the failure is `Cannot find libraries` or a missing `librbln.so`, rather than anything about your change.

Each of these makes a run exercise something other than what you think:

- **You are probably running the last build.** Any change under `aten/`, `c10/`, `torch_rbln/csrc/`, or `native_functions.yaml` needs `uv pip install -e . --no-build-isolation` before it means anything. This applies to the reproduction as much as the fix — a "fixed" result from a stale `.so` is the most common false conclusion in this repository.
- **`torch_rbln/_internal/register_ops.py` is generated.** Reading it is fine and often the fastest way to see what the eager path actually does. Editing it to test a hypothesis works until the next build, so treat any result from an edited copy as provisional and move the change to the YAML or the generator before concluding anything.
- **A flag may be latched, and the device mapping freezes on first use.** The per-dispatch flags are read live (`DispatchShim.cpp`), but anything cached in `static` or `@lru_cache` state keeps the value it saw first. The device mapping is planned by queries like `device_count()`, which claim nothing, and committed — frozen for the process — by the first real use: an allocation, a `synchronize()`, a collective. Editing `RBLN_DEVICES` after that is ignored. Check how the flag you are toggling is actually read, or use `run_in_isolated_process`.
- **The test suite is not production.** `test/conftest.py` force-disables the `compile_error` fallback, resets Dynamo between tests, and drains the allocator in teardown. A bug that appears only under pytest, or only outside it, is about that difference.
- **xdist workers carry state, and share the device.** A failure that appears only in a full run is usually a leak from an earlier test on that worker, but contention for the shared device produces the same shape. Run the suspected pair together in one process to tell them apart; if it is a leak, fix the leak rather than the second test.

## 2. When the log came from somewhere else

If the failure is in a log the user pasted from another machine — CI, a customer setup, a colleague's box — reason from the log and the source. Do not run local probes for host state; this box is not that box, and what they report is unrelated noise. When host-side evidence is genuinely needed, say what command to run there and ask for the output.

## 3. Make the evidence discriminate

Evidence that fits your hypothesis is easy to find and proves nothing — a wrong cause has confirming evidence too. Force the evidence to choose between explanations.

- **State what would be true if you are wrong**, and check that specifically.
- **Predict the output before you run.** A surprise means your model of the system is wrong, and that is the finding.
- **The cause must explain every symptom.** A leftover unexplained detail usually means a second cause or the wrong one.
- **Separate a flake from a regression before chasing it.** Run the same commit three times. If the failing configuration moves between runs — a different matrix leg, a different worker — it is nondeterminism, not your change. Say which one you established, and do not report a fix for a flake you only saw once.
- **For a numerical bug, compare against CPU before theorizing about the kernel** — at the smallest shape that still reaches the device, not the smallest that still runs. Half the "wrong result" reports are dtype or tolerance, not the op; an exact match is the other half, and it usually means the comparison never left the host.

Report which parts you verified and which you inferred. Never state a hypothesis as a fact.

## 4. Decide which layer owns it

Before writing a fix, place the defect:

1. **torch-rbln Python** — the shim, the device API, a `torch.compile` patch
2. **torch-rbln C++** — allocator, guard, copy, native kernel, process group
3. **rebel-compiler / librbln** — the compiler and runtime, a separate repository; the exact build pin is `constraints-build-dev.txt`

Signals for layer 3: the same graph produces different results across compiler versions; an error code from the runtime; a failure that disappears with `TORCH_RBLN_*` knobs that only change how much work is handed to the compiler; a fault inside `librbln`. Confirm by installing a different rebel-compiler version and re-running — a version bisect is worth the time, because it converts a guess into a fact. Install it into the venv; do not commit the pin change. Building against a compiler source checkout, when you need to test a candidate fix, swaps this repository's `pyproject.toml` and rebuilds the venv — ask first and use a throwaway worktree.

**Dump what the external component received before you bisect it.** A bisect tells you which commit flipped the behavior, not why the input it now produces is invalid — and "commit X introduced it" is not a fix you can write. Capture the IR or graph handed to the compiler, from a working build and a failing one, and diff them. The difference is the root cause; the bisect is the shortcut to a pair of builds worth diffing.

The switches that produce those artifacts belong to the runtime, not to this repository, and are not documented here — setting one is not always enough on its own. Ask the compiler team which switch yields the artifact you need on the pinned version rather than guessing at names.

**Check what the runtime already does before you build anything.** rebel-compiler ships its headers and `librbln.so` in the wheel; read its source and the pinned version's changes rather than inferring behavior from symptoms. A gate the runtime already implements, re-implemented here, is machinery that fights a problem that no longer exists — and it is the shape that survives ten review rounds before someone reads the runtime and deletes all of it.

**When the cause is layer 3, do not fix it here.** Reduce it to a minimal reproducer, report it against rebel-compiler, and say so. If the failure is architecture-specific, `xfail_rebel` / `xfail_atom` (`test/utils.py`) mark it strictly so it fails again once the bug is fixed; `_REBEL_XFAILS` in `test/conftest.py` is the table that applies `xfail_rebel` by test name. A failure that is not architecture-specific does not get parked. A guard added in the backend to make a compiler bug invisible removes the only pressure to fix it and stays in this code permanently. If you think a local workaround is warranted anyway, stop and ask.

## 5. Fix the cause, not its surroundings

Change the thing that is wrong — not a caller, not a wrapper, not a guard around the symptom.

Never widen an exception, add a default, route an op to CPU fallback, or loosen a tolerance to make the failure go away. Reaching for any of those means you are fixing without a cause.

## 6. If the fix does not work, discard it

Revert it and form another hypothesis. Do not stack a second patch on a failed first one. A failed fix is information: the cause was probably wrong.

## 7. Verify against the reproducer

Rebuild, run the reproducer again, run the tests around what you touched, and read the output. A fix that has not been run is not a fix. Check both execution paths — eager and `torch.compile(backend="rbln")` — because a fix in a generated shim does not necessarily reach the graph path, and vice versa.

## 8. Report the gaps

State explicitly what you could not build, run, or reproduce; hardware or models you did not have; and anything you inferred rather than verified.

Tests that broke after your change are your regressions. Debug them. Do not stash or revert to check whether they also fail on `dev`.
