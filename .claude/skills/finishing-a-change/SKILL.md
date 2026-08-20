---
name: finishing-a-change
description: Use before claiming a change is done, fixed, ready, or passing, and before writing a commit or PR — rebuilds, runs lint and the tests, reviews the diff against the repository rules, and reports what was not done.
---

# Finishing a change

By the time you get here the session is long and the instructions you read at the start have faded. Work through this list rather than from memory.

## 1. Rebuild if you touched the native side

```bash
uv pip install -e . --no-build-isolation
```

Required after a change under `aten/`, `c10/`, `torch_rbln/csrc/`, `cmake/`, `CMakeLists.txt`, `hatch_build.py`, `native_functions.yaml`, or `tools/codegen/` — the build runs codegen before compiling, so a generator change takes effect only here. Without it every test result below is about the previous build.

Confirm it took. An install can report success while skipping the build, and two error shapes tell you which side is stale: a link error naming `rbln::` symbols means the installed `rebel-compiler` is older than the pin, and an `AttributeError` for a missing `torch_rbln._C` attribute means the built extension is older than the Python tree — what a branch switch leaves behind.

## 2. Run lint

```bash
uv run --no-sync lintrunner -m origin/dev -a
```

Read the output. Passing means you saw it pass. C++ linting needs the compile database — see `docs/LINTING.md`. `-a` auto-fixes; re-read the diff afterwards, since it will have edited your files.

Lint against the merge base, not the files you happened to touch. CI lints the whole PR diff, so a file an earlier commit on this branch introduced still has to pass — a missing owner header or formatting drift in one of those is a rejected PR for a file you never opened.

A merge-base run covers uncommitted edits to tracked files but **not files that are still untracked**: a brand-new test lints clean because it was never looked at. `git add -N` the new files first, or pass them explicitly.

`-a` rewrites files in place across that whole range, including unrelated edits you have in the tree. If the tree carries work that is not part of this change, run it without `-a` first and read what it wants to change.

Never run `ruff`, `black`, or `isort` on their own; `.lintrunner.toml` coordinates them with configs that differ from each tool's defaults, so a direct run produces different output than CI. If lint reports a malformed mypy status file, the daemon state is corrupt from an aborted run: `dmypy stop`, remove `.dmypy.json`, and run again.

## 3. Run the tests

Run the narrowest target that covers the change, then the suite around it:

```bash
uv run --no-sync pytest test/rbln/test_<module>.py -x
uv run --no-sync python test/run_tests.py --suite=core
```

Include the command and its result in your reply. "Should pass" is not a result. If you could not run them, say that instead of implying you did.

## 4. State the regression and performance impact

Say what else takes the path you changed, and what the change costs on it. A copy path, a dispatch shim, or an allocator change is on the hot path for every model; a reviewer will ask, so answer first. If you did not measure, say you did not measure — do not offer "should be negligible" as a result.

## 5. Read your own diff

```bash
git diff                    # unstaged
git diff --staged           # staged
git status --short          # untracked files you may have meant to add
git diff origin/dev...HEAD  # what the PR will actually show
```

General:

- **Files nobody asked for** — docs, examples, scripts, changelogs
- **Scratch files** left over from iterating, anywhere in the tree
- **Single-use helpers** that should be inlined
- **Abstractions** built for one call site
- **Comments that narrate the code**, describe your changes, or address the reader
- **Comments added to code you did not otherwise change**
- **Comments your change made stale** — read the ones around every line you touched
- **Code with no caller** — machinery whose justification is in a later change belongs there
- **A second tuning constant for a trade-off that already has one**, or a threshold with no measurement behind it
- **Swallowed failures** — `except Exception`, bare `except`, `except: pass`, an unrequested fallback
- **`else` branches for cases that cannot happen** — should be `raise`, `assert`, or `RBLN_CHECK`
- **Dead leftovers** — renamed `_var`, unused re-exports, `# removed` comments
- **Anything not in English**

torch-rbln specific:

- **No stray dependency pin** — `constraints-build-dev.txt` and the `pyproject.toml` rebel-compiler range are not touched by a change that is about something else.
- **No edits to generated files** — `torch_rbln/_internal/register_ops.py`, `_abi_snapshot.py`, `torch_rbln/lib/`, `torch_rbln/include/`, `torch_rbln/_C/__init__.pyi`. `git status` will not warn you; they are gitignored.
- **No new CPU fallback case** added to make something pass, and no `TORCH_RBLN_DISABLE_FALLBACK` category changed to move a test outcome.
- **No workaround for a rebel-compiler or librbln defect.** If there is one, it is declared in the PR body with the upstream issue, and you asked before writing it.
- **A new `TORCH_RBLN_*` variable** is read only in `env_utils.py` (or its C++ config object) and is documented in `docs/CONFIGURATION.md`.
- **An op registered only in C++** is listed in `_ops_with_rbln_native_kernel` in `test/filters.py`.
- **New or changed tests follow `docs/TEST_GUIDE.md`** — placement, template shape, marker, and the existing fixtures and helpers rather than new ones. Re-read the relevant section against your diff; this is checked in review.
- **New test files** carry `# Owner(s): ["module: PrivateUse1"]` and the `run_tests()` main block, and are marked `test_set_ci` unless you can say why not.
- **No widened tolerance** without a derivation, and no test made green by re-running it.
- **Nothing on the import path uses a device** — the first allocation, `synchronize()`, or collective commits the mapping and freezes it for the process.
- **PR body claims match the diff.** A conditional check is not described as unconditional; a derived set is not described as derived if it is hardcoded.
- **`native_functions.yaml` and `test/ops/test_ops.py`** stay close to upstream.

## 6. Report what you did not do

State it explicitly, every time:

- Tests you could not run, and why
- Hardware, devices, or models you did not have
- Assumptions you could not verify
- Parts of the request you left out

An empty list is a claim. Only write it if it is true.

## 7. Write the commit and PR

- PR title in `type(scope): summary` — CI enforces it. Individual commits only need to be readable (`docs/CONTRIBUTING.md`)
- Explain intent, not the diff. Symptom, reproducer, where to look, what you verified — the debugging path that got you there is not part of it
- Say which layer you changed and which execution paths you exercised — eager, `torch.compile(backend="rbln")`, or both
- Fill in **Affected Modules** and **How to Test** in the template, and link the issue
- English
