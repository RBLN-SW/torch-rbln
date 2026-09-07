---
name: adding-an-op
description: Use when adding, changing, or removing an aten operator on the RBLN backend — the three registration paths, which one a given op belongs on, and what else has to change once you pick.
---

# Adding an operator

An op reaches the RBLN backend by one of three routes. Picking the wrong one produces code that works locally and then disappears, or an op whose tests never run.

## 1. Pick the path

**Generated shim (default).** Add a `func:` entry with a `PrivateUse1: <name>_rbln` dispatch to `aten/src/ATen/native/native_functions.yaml`. `tools/codegen/` emits the dispatch wrapper into `torch_rbln/_internal/register_ops.py`, which routes the op through `torch.compile` with the rebel backend. Use this whenever the op is a normal tensor computation.

**Hand-written Python kernel.** Same YAML entry, plus the `use_custom_kernel_rbln` tag. Codegen then emits only the registration and you supply the kernel in `torch_rbln/_internal/register_custom_ops.py` or `torch_rbln/_internal/kernels/`. Use this when the op needs a shape rewrite, a fused pattern, or host-side logic before it reaches the compiler — `pow.Tensor_Scalar_out` is the worked example.

**C++ kernel.** Implement under `aten/src/ATen/native/rbln/` and register with `m.impl(...)` inside `TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)` in `aten/src/ATen/native/RBLNRegisterOps.cpp`. Use this for ops that touch storage, strides, device guards, or the allocator — copy, resize, factories, advanced indexing.

Backward is not one of these routes. It is used in a few places only and adding it is not recommended; if a change genuinely needs it, read `torch_rbln/_internal/register_backward_ops.py` first and ask.

Not sure between two? Ask. A misplaced kernel is expensive to move later.

## 2. Never edit `register_ops.py`

It is generated and gitignored. Editing it works until the next build wipes it. The sources are the YAML entry and the generator under `tools/codegen/generators/`.

`tools/codegen/config.py` decides per-op behavior — empty-tensor handling, dtype checks, contiguity, kwarg filtering — from name sets. Membership only matters for a set an accessor actually reads: the entries of `EMPTY_TENSOR_HANDLING_MAP` (`REDUCTION_OPS`, `BROADCASTABLE_OPS`, and the per-op sets), `SKIP_DTYPE_CHECK_OPS`, `SINGLE_MEM_LOC_OUT_VIOLATION_OPS`, `SPECIAL_KWARGS_FILTER_OPS`. Others are defined and never used — `UNARY_OPS` is one, so adding a unary op to it changes nothing. Grep for the set before assuming it does, and add the op only where a reduction or a broadcast really needs different handling.

## 3. What else changes

- **C++-only registration is invisible to op tests.** `test/filters.py` discovers ops by parsing `native_functions.yaml`; an op registered only in `RBLNRegisterOps.cpp` must be added to `_ops_with_rbln_native_kernel` or its upstream OpInfo tests never run.
- **The dispatch-shim sets in `tools/codegen/config.py`.** `codegen.py:_registration_line` emits `_register_cpp_shim(...)` for an overload listed in one of the `SHIM_*` sets and `aten_impl.impl(...)` for everything else. What the shim buys is that neither hot path enters Python: it runs the pre-check (dtype, all-scalar, contiguity and offset) in C++ and on failure calls `cpu_fallback_rbln` directly, and on a warm-cache hit it drives the rebel runtime from C++ with no pybind crossing. Python is entered only on a cache miss, to compile — see the header comment in `torch_rbln/csrc/rbln/DispatchShim.h`.
- **Which path your op belongs on follows from that pre-check.** The sets cover the families it can decide — binary and unary pointwise, compare, where, reduction, matmul — 42 of the 50 registrations today. The 8 on the Python path are the ones it cannot: in-place (`masked_fill_`), backward kernels, and fused or custom ops (`_softmax`, SDPA). Add a plain pointwise or reduction op to the matching set; leave an in-place, a backward, or an op with typed non-tensor arguments on the Python path. For a single typed argument — `where`'s bool condition — `CPP_SHIM_SKIP_DTYPE_ARGS` exempts that index instead.
- **A public API name that differs from the schema name** goes in `_ops_with_public_api_name_mismatch` in the same file.
- **`native_functions.yaml` is adapted from upstream PyTorch.** Copy the upstream entry and add only the `PrivateUse1` dispatch and RBLN tags. The `NATIVEFUNCTIONS` linter round-trips the file, so reformatting it fails lint.
- **Every tag must be declared in `tags.yaml`**, which is a trimmed copy of upstream's — so an entry copied verbatim can carry a tag we do not have, and parsing then fails with `illegal tag <name>` before codegen emits anything. Add the tag to `tags.yaml` or drop it from the entry.
- **Removing an op** means removing its YAML entry, its filter entries, and its tests together. A leftover filter entry keeps a test alive for a kernel that no longer exists.

## 4. Rebuild, then read what was generated

```bash
uv pip install -e . --no-build-isolation
```

Codegen runs during the build. Until it does, your YAML change has no effect at all — the shim on disk is still the old one.

For a YAML-path op you can skip the C++ rebuild while iterating and run codegen alone:

```bash
uv run --no-sync python tools/run_codegen.py aten/src/ATen/native/native_functions.yaml aten/src/ATen/native/tags.yaml torch_rbln/_internal/register_ops.py
```

That rewrites the generated file in place, so back it up first if the tree is one you care about. A C++-path op still needs the full install. After the build, read the generated function in `torch_rbln/_internal/register_ops.py` and confirm it is what you intended: the wrong category mapping produces a wrapper that compiles and returns wrong results on empty or broadcast inputs.

## 5. Verify

### First: prove it ran on the device

Nothing below means anything until this holds. The failure is silent — an op routed to the host returns the CPU result, so the comparison you were about to trust agrees exactly.

1. **Give it an input the device path accepts: fp16 or bf16, with a last dimension that is a multiple of 64.** `SUPPORTED_DTYPES` is those two only, and `compile_and_run_view_aware` applies its own alignment fallback after `is_cpu_fallback_cases` (see the 64-element alignment comment in `ops_utils.py`). Miss either condition and the op runs on the host however it is registered. Measured on `rsqrt`: `(4, 64)` fp16 and bf16 differed from CPU at the rounding level, while fp32 — and fp16 at `(4, 8)` — matched exactly, because neither reached the kernel. `torch.rand` defaults to fp32, which is how this is usually missed.
2. **Check the log, not the numbers.** With `TORCH_RBLN_LOG_LEVEL=INFO` every fallback prints ```aten::<op>` op ran on CPU instead of RBLN``. No such line on an aligned input is the proof. `TORCH_RBLN_DISABLE_FALLBACK=compile_error` does not cover the alignment route and will not raise for you.
3. **Run an established peer on the same input** — a unary against `rsqrt`, a binary against `add`. The same order of difference from CPU says your op took the same path.

### Then: correctness

- Eager and inside `torch.compile(backend="rbln")` produce the same numbers.
- Every dtype in `SUPPORTED_DTYPES` that the op claims to support.
- Empty tensors, broadcasting, non-contiguous inputs, and a non-zero storage offset — the four the generated wrappers handle by category and therefore get wrong by category.
- The `out=` variant writes into the passed tensor.
- `test/rbln/test_registered_ops.py` and the op tests in `test/ops/test_ops.py` still pass.

Follow `.claude/skills/writing-tests/SKILL.md` for the tests themselves.

## 6. Do not

- Do not route the op to CPU fallback to make it pass. An op that falls back is an op that is not implemented; say so instead.
- Do not add a special case to `is_cpu_fallback_cases()` for a shape or dtype the kernel gets wrong. That hides a correctness bug behind a performance cliff.
- Do not widen a tolerance to make a new op's test agree with the CPU reference. Find out why they disagree first.
