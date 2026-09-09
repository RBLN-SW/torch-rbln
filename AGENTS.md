# Agent instructions for torch-rbln

torch-rbln is an out-of-tree PyTorch backend (`PrivateUse1`) for Rebellions NPUs. C++ lives in `aten/`, `c10/`, and `torch_rbln/csrc/`; Python in `torch_rbln/`; tests in `test/`.

These documents own their subjects. Read them rather than duplicating them here:

| Subject                                                    | Document                     |
| ---------------------------------------------------------- | ---------------------------- |
| Test infrastructure, markers, fixtures, templates          | `docs/TEST_GUIDE.md`         |
| Environment variables and runtime options                  | `docs/CONFIGURATION.md`      |
| Lint setup and commands                                    | `docs/LINTING.md`            |
| PR process, issue labels, merge policy                     | `docs/CONTRIBUTING.md`       |
| CI and release lanes                                       | `docs/WORKFLOWS.md`          |
| PyTorch pin, vendored upstream files, rebel-compiler bumps | `docs/THIRD_PARTY_UPDATE.md` |

## Terms

The codebase already has a word for each of these. Use it, and do not reach for a synonym because the sentence reads better.

- **eager mode** — op-by-op dispatch through the generated shim. Each op is compiled and cached individually, so "eager" does not mean the compiler is uninvolved.
- **graph mode** — the user's `torch.compile(backend="rbln")` over a whole FX graph.
- **deploy mode** — `TORCH_RBLN_DEPLOY=ON`, which skips the host-side NaN/Inf scan. It is not a third execution path.
- **logical device** — what `rbln:N` and `device_count()` refer to: one or more **physical NPUs** grouped by `RBLN_NPUS_PER_DEVICE` or `RBLN_DEVICE_MAP`. Always say which of the two you mean.
- **fallback** — two separate controls. `TORCH_RBLN_DISABLE_FALLBACK` covers unsupported ops, compile errors, and non-blocking copies; `TORCH_RBLN_DEV_DISABLE_OP_CPU_FALLBACK` covers the per-dispatch checks (`dtype`, `scalar`, `storage_offset`, `nan_inf`, …). Name the one you mean.
- **shim** — the generated Python wrapper in `register_ops.py`. **kernel** — a hand-written implementation, C++ or Python. Not interchangeable.
- **suite** — a `run_tests.py` tree: `core`, `distributed`, `models`, `ops`. **lane** — what a marker selects: the CI lane, the release lane, the perf lane.

## Which file do I edit

Two questions come before any change, and getting either wrong wastes the whole session.

**Is this even our layer?** Three layers own this stack; only two are in this repository.

1. **torch-rbln Python** (`torch_rbln/`) — shims, device and memory APIs, `torch.compile` patches
2. **torch-rbln C++** (`aten/`, `c10/`, `torch_rbln/csrc/`) — allocator, guard, copy, kernels, process group
3. **rebel-compiler / librbln** — the compiler and runtime; a separate repository. `pyproject.toml` carries the version range, `constraints-build-dev.txt` the exact dev build pin.

- Say which layer the defect is in, and what the minimal set of changes is, before writing any of them.
- **Do not work around a layer-3 defect in layer 1 or 2.** A guard that makes a compiler or runtime bug invisible hides it from the team that can fix it and stays here permanently.
- When it is layer 3: reduce it to a minimal reproducer, report it to the rebel-compiler team, and say so in the PR. If the failure is architecture-specific, `xfail_rebel` / `xfail_atom` (`test/utils.py`) mark it strictly, so it fails again once the bug is fixed. A failure that is not architecture-specific does not get parked — it stays visible.
- **The boundary runs both ways — before building machinery here, check what the pinned runtime already does.**
- **Bump the pin when your change needs it; leave it alone when only your experiment did.** Raising `constraints-build-dev.txt` in the PR that requires a newer compiler is normal here — say in the body what the change needs from it. What does not belong is a pin you moved while trying versions, or a local-wheel path some tool wrote into `pyproject.toml`. A scheduled workflow also proposes bumps on its own, so an unexplained one reads as noise. Moving the *range* in `pyproject.toml` is a separate procedure — `docs/THIRD_PARTY_UPDATE.md` owns it and the two places that stay aligned.
- To settle which layer it is, install a different `rebel-compiler` version and rebuild. Building against a compiler source checkout is heavier and mutates this tree — ask first, and use a worktree you can throw away.

**Is this file generated?** These are build outputs; the edit disappears on the next build.

| File                                                                             | Real source                                                    |
| --------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| `torch_rbln/_internal/register_ops.py`                                           | `aten/src/ATen/native/native_functions.yaml`, `tools/codegen/` |
| `torch_rbln/_internal/_abi_snapshot.py`                                          | `cmake/abi_snapshot.py.in`, configured at build time           |
| `torch_rbln/lib/`, `torch_rbln/include/`, `torch_rbln/_C/__init__.pyi`, `build/` | `cmake --install`                                              |

`register_ops.py` is the one that catches people: every eager shim (`add_rbln`, `mul_rbln`, …), 1400 lines of ordinary-looking Python, gitignored. Fixing an op there works until the next build.

`native_functions.yaml`, `tags.yaml`, and `test/ops/test_ops.py` are adapted from upstream PyTorch; the two YAML files name the upstream tag they came from on their first line. Keep the diff no larger than the change requires, and keep that line true when you re-sync one. A torch version bump moves three things together — `[project].dependencies`, `[build-system].requires`, and these files re-synced from the new tag (`docs/THIRD_PARTY_UPDATE.md`). `tools/linter` is maintained in-tree and refreshed by `sync-linter.sh`; do not copy it from upstream.

## Commands

Run everything through `uv`. Never use the system `python3` or a bare `pip`.

```bash
uv run --no-sync pytest test/rbln/test_tensor_copy.py -x   # narrowest target covering the change
uv run --no-sync python test/run_tests.py --suite=core     # a whole suite, the way CI runs it
uv run --no-sync lintrunner -m origin/main -a              # lint and auto-fix what you changed
uv pip install -e . --no-build-isolation                   # rebuild — see below
```

**A change under `aten/`, `c10/`, `torch_rbln/csrc/`, `CMakeLists.txt`, or `native_functions.yaml` does not reach your interpreter until you rebuild.** The build needs `rebel-compiler` present at the pin, installed with `uv pip install --constraint constraints-build-dev.txt rebel-compiler`. A link error naming `rbln::` symbols means the installed one is older than the pin — install the pinned version rather than changing code to match what you have. An `AttributeError` for a missing `torch_rbln._C` attribute means the opposite: the built extension is older than the Python tree, which is what a branch switch leaves behind. An `RBLN ABI mismatch` is a third shape — the snapshot this build recorded and the loaded `librbln.so` disagree.

Skip the rebuild and pytest loads the previously built `torch_rbln/_C` and `torch_rbln/lib/*.so`: the run tests the old code and its result means nothing. A bare `ninja -C build` stages nothing into `torch_rbln/`; the install step does. The build needs GCC 13 or newer (`CC=gcc-13 CXX=g++-13`), enforced by `cmake/FindTorch.cmake`.

Verify the rebuild actually took before trusting any result. An install can skip the build step and leave the previous binary in place while reporting success; grepping the built artifact for a string only your change introduces is the cheap check.

`lintrunner -m <base>` lints against a merge base; use `origin/main`, which is what CI does too. It covers every file the PR touches, not the ones you edited this session — that difference is what gets a PR rejected by CI for a file you never opened.

## Evidence

- **Read the source, do not infer it.** rebel-compiler ships its headers and `librbln.so` in the wheel; PyTorch is in the venv. An answer from an issue description, a changelog, or a plausible reading of a symptom is a guess; do not state it as fact.
- **Separate the symptom from the cause.** A narrowed symptom is not a cause. If the cause is not identified, say so.
- **Mark what you verified and what you inferred**, in the same message.
- **Reproduce before you fix**, and say so if you could not. The minimal reproducer is a deliverable: a standalone script another team can run without installing a model or a suite. Not being able to reproduce — no device here, a log from another machine — is a limit to state, not a reason to stop: narrow the cause from the log and the source, write the candidate fix, run what this host can run, and hand over the command that settles it on a device. What you may not do is call it fixed before that command has run.
- **State what you could not do** — tests you could not run, hardware you did not have, assumptions you could not check. An empty list is a claim; write it only if it is true.
- **If you cannot finish, do not produce a plausible partial result.** Stop and say what blocked you.
- **Tests that break after your change are yours to investigate.** Comparing base against head is how you do that — in a separate worktree, same venv and same pin, never by stashing or reverting in the tree you work from. An editable checkout is the runtime for whatever is running from it, and a stashed C++ change tests the same binary until you rebuild. A failure that reproduces on the base commit is reported as pre-existing, with that commit; everything else is your regression.

## Measuring

- **Print the base before every measurement**: the commit of torch-rbln, of rebel-compiler, and of any consumer in the loop, plus the venv and the env vars that matter. A number without its base compares to nothing later.
- **Name a target by commit plus working-tree state**, never by branch. "main" does not say which uncommitted changes were in the tree.
- **Fix the environment and change nothing but the thing under test.** Thread counts, allocator mode, and log level move the mean more than most changes do; log level is not cosmetic here. Change the harness and the comparison resets.
- **A debug-only flag is not the production baseline.** A result measured with one set does not describe what ships.

## Know where you are

Before the first edit, record the commit and dirty state (`git log -1`, `git status --short`), the venv and the `rebel-compiler` installed in it, and whether this host has an NPU. The answer decides what a result here can mean:

| Host                        | What a run here proves                                                                                                                                                                                                     |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Source only, no toolchain   | Nothing ran. Read, reason, write the patch, and say so.                                                                                                                                                                    |
| Build host, no NPU          | Build, lint, codegen, and `RBLN_DUMMY_DEVICE=1` for tensor construction, host/device copies, and compile-only `torch.compile` (`docs/CONFIGURATION.md`). `is_available()` is `True` there and says nothing about hardware. |
| NPU host                    | Numerics, performance, fallback attribution, distributed. Only a result from here settles a claim about the device path.                                                                                                    |

Match the verification to the change: a docs change needs lint; Python logic needs the narrowest pytest target and the suite around it; a codegen or native change needs the rebuild, proof that it took, and those tests; a perf change needs a measurement with its base printed. Read the collected count, not only the exit code — `run_tests.py` counts a run that collected nothing as neither pass nor fail, and a device-type test on a host without an NPU collects nothing.

## Before saying a change is done

Do these without being asked. `.claude/skills/finishing-a-change/SKILL.md` is the full list.

1. Rebuild, if the change touched the native side.
2. `lintrunner -a`, and read the output.
3. Run the tests covering the change; report the command and its result. "Should pass" is not a result.
4. **State the regression and performance impact** — what else takes this path, and what the change costs on it. If you did not measure, say you did not measure.
5. Read your own diff for verbose comments, leftover scratch files, and code with no caller.
6. Write the PR body to claim exactly what the code does, and no more.

## Operators

Three registration paths, and the choice decides what else must change.

- **YAML path** — a `func:` entry with a `PrivateUse1: <name>_rbln` dispatch in `native_functions.yaml`; codegen emits the shim. This is the default.
- **Hand-written Python kernel** — the same YAML entry plus the `use_custom_kernel_rbln` tag; codegen emits only the registration and you supply the kernel.
- **C++ path** — a kernel in `aten/src/ATen/native/rbln/` registered with `m.impl(...)` in `RBLNRegisterOps.cpp`, for ops touching storage, strides, or the allocator.

An op registered only in C++ is invisible to op-test discovery: add it to `_ops_with_rbln_native_kernel` in `test/filters.py` or its OpInfo tests never run.

Every op runs both eager and inside a user's graph, and must produce the same numbers on both; `test/rbln/test_graph_eager_mode.py` is the guard. Say which paths you exercised.

Read `.claude/skills/adding-an-op/SKILL.md` before starting.

## Patching upstream PyTorch

`torch_rbln/_internal/monkey_patches.py` replaces upstream symbols at import through `apply_all_patches()`: `torch.compile`, `torch._dynamo.reset`, and — on torch below 2.13 — `GuardBuilder.id_match_unchecked` and `torch.Tensor.__repr__`. A new patch follows the existing shape; `test/rbln/test_torch_compile_patch.py` asserts it.

- **Inert for everyone else.** A non-RBLN backend must reach the original untouched, the way the `torch.compile` wrapper early-returns on `is_rbln_backend`.
- **Narrow the global surface.** The `__repr__` patch is scoped by a thread-local to guard build only, so a user's `repr(tensor)` is unchanged. Do not replace a global unconditionally.
- **Idempotent, with the original saved.** Guard re-application with the module-level flag and keep the original in a module global.
- **Every patch is undone by `remove_all_patches()`**, including the caches it warmed — `clear_rbln_compile_cache()` and `warm_cache.clear()` are part of the teardown.
- **A backport of an upstream fix is version-gated and cites the issue**, so it becomes a no-op once upstream lands it — see the `torch.__version__ >= (2, 13)` early return.
- Registration is lazy: the dynamo backend registers on the first `torch.compile` call, not at import. Keep it that way — see the import rule below.

## Contracts that bite

- **Nothing on the import path may claim a device.** `import torch_rbln` runs at `import torch` through the backend autoload entry point. The device mapping is *planned* by queries like `device_count()`, which claim nothing, and *committed* — frozen for the process — by the first real use: an allocation, a `synchronize()`, a collective. Selecting a device does not commit (`docs/CONFIGURATION.md`). A launcher may therefore set `RBLN_DEVICES` after import, including inside a forked worker, as long as nothing has used a device yet; an arch query at import takes that away. `test/rbln/test_import_rbln_devices_seal.py` is the guard.
- **The rebel ABI handshake decides whether this build and the loaded runtime may talk.** `rebel-compiler` declares the interface it implements and the oldest consumer it accepts; the build freezes the first of those into `_abi_snapshot.py`, and `import torch_rbln` checks that the loaded `librbln.so` still accepts that snapshot before any other rebel call. We keep no number of our own — there is nothing to bump, and the snapshot is generated, so a mismatch is a build or an install to correct, never a file to edit. It moves only on a rebuild, so a stale build gives a stale verdict, and a wheel already shipped cannot follow a later raise of the minimum. `TORCH_RBLN_SKIP_ABI_CHECK` hides the diagnosis, not the incompatibility: use it to unblock a machine while a matching wheel builds, never to make an import go green. `docs/CONFIGURATION.md` has the contract.
- **Upstream's own test is the contract**, not what the hardware happens to do. For `PrivateUse1`, `torch.accelerator`, AMP, the profiler, and pinned memory, read the corresponding test in the PyTorch repository at the pinned version and satisfy what it asserts. **When the contract is ambiguous, follow CUDA** — that is what vLLM, lmcache, and transformers are written against.
- **CPU fallback is a product feature, not a way to make code work.** Do not add a case to `is_cpu_fallback_cases()`, route an op to `fallback_rbln`, or change a `TORCH_RBLN_DISABLE_FALLBACK` category to move a test outcome — that turns a correctness bug into a silent performance cliff. The suite force-disables `compile_error` (`test/conftest.py`) so compile failures surface.
- **A fallback where the device path was under test is not a pass.** A CPU fallback compares CPU against CPU; an export that fell back to `jit.trace` gives plausible output from an untested path. Check the fallback log or the counters before calling such a run successful. An expected fallback is a result in its own right: fp32 never runs on the NPU, `_ops_with_cpu_fallback_tests` in `test/filters.py` tests the fallback itself, and dummy mode has no device to fall back from. A minimal reproducer for a device-path bug keeps the fallback disabled — drop it and the bug disappears; a reproducer for a fallback defect keeps it on, because the fallback is the subject.
- **A new variable of ours takes the `TORCH_RBLN_` prefix.** The bare `RBLN_*` namespace is shared: `RBLN_DEVICES` and its `RBLN_VISIBLE_DEVICES` alias belong to the runtime, `RBLN_DUMMY_DEVICE` is validated by it, and the ids in `RBLN_DEVICE_MAP` index the pool `RBLN_DEVICES` leaves visible rather than system ids. Do not add to that namespace or reinterpret what is in it.
- **A new `TORCH_RBLN_*` variable** is read in one place — `env_utils.py`, or its C++ config object — and documented in `docs/CONFIGURATION.md`. Read it live, the way the per-dispatch flags in `DispatchShim.cpp` do. A value cached in `static` or `@lru_cache` state latches for the whole process, so a test that sets it later changes nothing and what it did set leaks into later tests on that worker — this has already cost a CI flake.

## Production code

- **No `print`.** Use the logging facility.
- **No test-only entry points.** A hook or knob that exists so a test can reach internal state does not belong in shipped code. Restructure the test, or the code.
- Code either succeeds or fails with a clear error. No `except Exception` or bare `except` to get past a problem; no `except: pass`; no unrequested fallback, default, or silent recovery.
- Do not route a case that cannot happen into an `else`. Use `raise`, `assert`, or `RBLN_CHECK`.
- Delete removed code completely: no renamed `_var` leftovers, no dead re-exports, no `# removed`.
- Everything in the repository is English — code, comments, log and error messages, test names, docs, commits, PR titles and bodies. Reply to the user in the language they write in.

## What not to publish

This repository is public. Absolute measurements stay out of it: throughput, latency, memory footprint, and accuracy from an internal run do not belong in code, comments, tests, docs, commit messages, or PR descriptions. Benchmark scripts and `test_set_perf` tests live here; their results do not.

A relative change is fine. Say that something got a third faster, not what the two numbers were, and keep the raw figures to an internal channel.

`docs/CONTRIBUTING.md` asks a `perf` change for "benchmarks or measurement methodology". The methodology and the harness belong in the PR; the numbers they produced do not.

## Fix the cause, keep the diff reviewable

These pull against each other, and both are real.

- **Fix the cause, not the report.** A change that blocks the symptom and leaves the mechanism comes back under another name. When a class of bug repeats, revisit the earlier fixes rather than adding one more.
- **A diff nobody can review does not land.** When the patch keeps growing, stop and re-check the approach — that usually means the fix is in the wrong place, not that the problem is large.
- **Do not land code with no caller.** Machinery belongs in the change that consumes it and measures it. Check who actually calls the thing before building for the general case.
- **One tuning constant per trade-off**, and the PR carries the measurement that picks its value.
- Reuse before you write: no new file when the code fits in one that exists, no abstraction for a single use. A helper earns its place by naming a condition the call site cannot, or by owning a resource lifetime — not by shortening one call. Delete the scratch files you made while iterating.

## Prose

Everything written in words — a comment, a docstring, a doc page, a PR body, a bug report — earns its length or gets cut. A sentence stays if it changes what the reader decides; the rest costs them more than it saved you.

- Do not restate the code, describe your changes, address the reader, or comment code you did not otherwise change. When a comment narrates the next line, delete it and let the name carry the meaning. Assume the reader knows PyTorch dispatch and RBLN hardware.
- Keep what a reader needs in order to decide something: a runtime contract nothing validates, why a retry or fallback is sound, which direction a heuristic is conservative in, and why a threshold sits where it does — the reason for the value, not the arithmetic that produces it. These may be long; explain them properly. `torch_rbln/_internal/compile_cache.py` and the fixtures in `test/conftest.py` are the level.
- Your change makes comments stale. Re-read every comment around the lines you touched.
- A bug report or an issue is four things: the symptom with numbers, the reproducer, where to look, and what you verified. The triggers you tried, the workarounds that failed, why nobody caught it earlier, and mechanisms you have not confirmed stay in your own notes.
- Do not flatten a corner case into a tidy rule. An honest boundary is worth more than "X always fails", because the boundary is what someone acts on.

## Tests

Read `docs/TEST_GUIDE.md` before adding a test — it is the contract for this suite, and review checks against it. `.claude/skills/writing-tests/SKILL.md` covers what goes wrong on top of it. The rules broken most often:

- A test must fail without the change it covers. Verify that; do not assume it.
- **A test with no `@pytest.mark.test_set_ci` never runs in the pre-merge checks.** It runs only in the release lane — the release checks on `main`.
- **Pick the isolation the state needs.** `single_worker` for a resource shared across processes — a device, a fixed port; a fixture or `monkeypatch` for state that restores — an env var that is read live, the selected device; `run_in_isolated_process` (`test/utils.py`) for state that does not — an import-time effect, a `static` or `@lru_cache` latch, a committed device mapping. `single_worker` only serializes: it is one xdist worker, not a fresh process, and undoes no leak.
- Assertion rewriting is off (`--assert=plain`), so a bare `assert a == b` prints no values.
- Do not re-run a flaky test until it goes green, and do not loosen a tolerance to make one pass.
- Do not assert an optimization the runtime is free to change, or write an assertion a hardcoded constant would also satisfy.
- A test diff larger than the change is not wrong by itself — a one-line allocator fix can need a dtype, layout, and process matrix. Say what each axis guards, and cut the tests that repeat another axis or restate the implementation.

## When a rule is in the way

A decision recorded in the tree holds until its reason is gone: an `xfail` with a reason, a skip that names its issue, a table entry with a why. Do not clear one because you could. An uninstalled dependency or an `importorskip` records no decision by itself. What makes installing one a question is the venv, not the skip — an editable checkout is the runtime for whoever else uses it, and a model dependency can move the torch or `rebel-compiler` pin. Install into a throwaway venv or worktree, or say exactly what the install changed.

Not all rules carry the same weight. Absolute: do not edit a generated file, do not work around a layer-3 defect, do not silence a fallback to move an outcome, do not move a pin the change does not need, do not push without asking, and do not widen the task past what was asked. The rest — the shape of a helper, a comment, a test — are review defaults: apply them, and when you depart from one, write the reason where the reviewer will see it and move on. Do not stop to ask about those.

Stop and ask only for what crosses an absolute rule or changes the scope: a workaround for a defect whose root fix is out of scope, an edit that has to touch a generated file, a pin you need to move, a fix in a layer the task did not name. Do not decide those alone, and do not work around them quietly.

## Commits and PRs

`type(scope): summary`, matching the existing history — `fix(profiler): register the kineto bridge unconditionally so import doesn't seal RBLN_DEVICES`.

Explain intent; the diff already shows what changed. Claim exactly what the code does — a check that only runs under a condition is not "the index is range-checked". Fill in **Affected Modules** and **How to Test**, link the issue, and name the layer you changed.

- Every branch comes from `main` and targets `main`, hotfixes included. Tags are cut on `main`, and everything is squash-merged (`docs/RELEASE_PROCESS.md`, `docs/CONTRIBUTING.md`).
- **A fix for a regression the feature itself introduced belongs in the same PR.** Split it out and whichever lands first puts mainline in the regressed state.
- **No internal names.** A working label from the conversation that produced the change — a version number, a phase, a step — means nothing later and fossilizes in the history. Name the mechanism.

## Skills

- Adding or changing an operator: `.claude/skills/adding-an-op/SKILL.md`
- Adding or changing tests: `.claude/skills/writing-tests/SKILL.md`
- Investigating a bug or a test failure: `.claude/skills/debugging/SKILL.md`
- Before claiming a change is done: `.claude/skills/finishing-a-change/SKILL.md`
