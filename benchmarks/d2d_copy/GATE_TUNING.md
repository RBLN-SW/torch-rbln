# `strided_v2v_copy` gate — experiment playbook

This document explains the experiments used to derive the heuristic in
`aten/src/ATen/native/rbln/RBLNCopy.cpp::should_use_strided_v2v_copy` on the
development machine, and tells you how to re-run them on a different NPU to
check that the gate's thresholds still hold there. It builds on the generic
benchmark suite described in `README.md` (same directory); commands assume you
run from this directory with the venv active.

## 1. What we're measuring

This branch routes same-device same-shape same-dtype `aten::copy_` calls
through `strided_v2v_copy` *when a gate says it's profitable*. Outside the
gate, copy falls back to the historical host-bounce path
(`get_cpu_copy_of_rbln_tensor` → CPU tensor → re-upload).

For any view geometry, `strided_v2v_copy` walks dims `[0, inner_start)` and
issues one v2v entry per outer index — i.e. `outer_count` entries of
`inner_block_bytes` each. Its cost is roughly:

```
cost_engine ≈ outer_count * per_entry_overhead  +  total_bytes / device_bw
cost_host   ≈ d2h(view_span) + h2d(numel)  + (small per-call overhead)
```

Two regimes dominate:

- **Engine wins** when `outer_count` is small (entry overhead is amortized),
  the inner block is fat (DMA bandwidth dominates), or the view is sparse
  (host bounce must transfer the whole storage span both ways).
- **Host wins** when `outer_count` is huge with tiny inner blocks over a
  compact span (per-entry overhead × outer_count exceeds one bounce of the
  span).

The gate in `RBLNCopy.cpp` is:

```cpp
constexpr int64_t kStridedV2VOuterAlways = 1024;           // outer ≤ this → engine
constexpr int64_t kStridedV2VOuterMax = 256 * 1024;        // hard cap for span override
constexpr size_t  kStridedV2VFatInnerBytes = 256;          // fat inner block → engine
constexpr size_t  kStridedV2VLargeViewSpanBytes = 1 << 20; // sparse view → engine

// engine ⟺ outer ≤ 1024
//        ∨ inner ≥ 256 B
//        ∨ (outer ≤ 256K ∧ span ≥ 1 MB)
```

The experiments below justify those four constants on this hardware. Re-run them
on the target device, then update the constants if the crossover moves.

## 2. Setup

Everything assumes the venv is active (`source .venv/bin/activate`) and editable
install is current. The benches all `import rebel` to force the runtime to load
before the first allocation, so make sure the device is reachable and idle.
**Don't run two benches on the same NPU concurrently** — timing gets noisy when
two processes share the same chip.

You will rebuild `libtorch_rbln.so` three times — once per build configuration
described in §3. Each build is a one-line patch in
`aten/src/ATen/native/rbln/RBLNCopy.cpp` followed by:

```bash
uv pip install -e . --no-build-isolation
```

(roughly 30–60 s incremental).

## 3. Build configurations

All three patches live inside `should_use_strided_v2v_copy(...)` in
`RBLNCopy.cpp`. Save the original, then swap one line.

| Label    | Patch                                     | Meaning                                              |
| -------- | ----------------------------------------- | ---------------------------------------------------- |
| `engine` | Make the function body `return true;`     | Every same-shape same-dtype rbln→rbln copy → engine  |
| `host`   | Make the function body `return false;`    | Every same-shape same-dtype rbln→rbln copy → bounce  |
| `gated`  | Leave the file as-shipped on this branch  | The four-constant heuristic (current default)        |

A convenient workflow is to copy the as-shipped `RBLNCopy.cpp` to
`/tmp/RBLNCopy_gated.cpp`, hand-edit two variants to
`/tmp/RBLNCopy_engine.cpp` and `/tmp/RBLNCopy_host.cpp`, then `cp` the
appropriate version into place before each rebuild.

Sanity check after each build:

```bash
python -m pytest test/rbln/test_copy_v2v.py test/rbln/test_tensor_copy.py -x
```

Both files should be green under every build config. If `test_copy_v2v.py`
fails under the `host` build, the v2v engine kernels (cat / index_select /
index_copy / repeat_interleave) themselves have regressed and you should stop
and investigate before continuing — the host build only changes `aten::copy_`,
not those native kernels.

## 4. Experiments

### 4.1 Engine reachability probe — `bench_v2v_engine.py`

Run once on the as-shipped build. If `index_copy_` is in the same ballpark as
`aten::copy_` rather than ~300× faster, the engine is not actually dispatching
to `rbln_memcpy_v2v_multi` on the target device — likely a runtime build
mismatch. The gate cannot be tuned around a broken backend; stop here.

### 4.2 Worst-case regression check — `bench_worst_case.py`

Stride patterns where the inner contiguous block is tiny
(`inner_block_elems = 1`) and `outer_count` is huge. These are where
unconditionally routing to the engine loses to host by 10–60×. Run under all
three builds and confirm the gated build matches host.

Reference numbers (dev machine, rebel-compiler 0.10.5.dev81):

| Shape                           | host build | gated build |
| ------------------------------- | ---------- | ----------- |
| `transpose 64×64` (8 KB)        | ~7 μs      | ~8 μs       |
| `transpose 256×256` (128 KB)    | ~37 μs     | ~37 μs      |
| `transpose 1024×1024` (2 MB)    | ~2.8 ms    | ~2.8 ms     |
| `transpose 2048×2048` (8 MB)    | ~11 ms     | ~12 ms      |
| `permute (128,128,128) (1,0,2)` | ~710 μs    | ~565 μs (engine) |

Pass criterion: every line under the gated build within ~20% of the better
of engine/host. A large positive delta means the gate misrouted a worst-case
shape — tune `kStridedV2VFatInnerBytes`, `kStridedV2VOuterAlways`, or
`kStridedV2VOuterMax`.

### 4.3 Crossover sweep — `bench_crossover_tighter.py`

The main numeric experiment (see README for the grid). Run once per build
config, saving JSON each time:

```bash
# 1. With engine build active:
python bench_crossover_tighter.py --label engine --out /tmp/tighter_engine.json

# 2. Rebuild with host patch, then:
python bench_crossover_tighter.py --label host --out /tmp/tighter_host.json

# 3. Rebuild with the as-shipped (gated) source, then:
python bench_crossover_tighter.py --label gated --out /tmp/tighter_gated.json
```

Each run takes about 5–10 minutes on the dev machine. Cells whose total view
exceeds 16 MB or whose parent storage exceeds 256 MB are skipped automatically;
adjust `--max-total-mb` if the target device has different memory.

### 4.4 KV-shape variation sweep — `bench_kv_shapes.py`

The original motivation: vllm-rbln sub-block KV writes. For each cell the
script prints which gate clause fires (`gate_decision()` is a Python mirror of
the C++ predicate). Note the printed gate column is informational only — if
you tune the C++ constants, update the mirror to match. The timing column is
always ground truth.

```bash
# Under each build:
python bench_kv_shapes.py --label engine --out /tmp/kv_engine.json
python bench_kv_shapes.py --label host   --out /tmp/kv_host.json
python bench_kv_shapes.py --label gated  --out /tmp/kv_gated.json
```

## 5. Analysis

### 5.1 `analyze_sweep.py` — engine-vs-host crossover characterization

```bash
python analyze_sweep.py /tmp/tighter_engine.json /tmp/tighter_host.json
```

This is what the gate *ought* to encode. Look for:

- A clean monotone in the "wins by outer bucket" table: outer ≤ 64 wins
  everywhere, outer > 1024 loses everywhere, with a transition band in
  between. If the transition is sharper or softer than on the dev machine,
  consider moving `kStridedV2VOuterAlways` / `kStridedV2VOuterWithLargeWork`.
- A clean monotone in "wins by inner_bytes bucket": there is some inner_bytes
  threshold above which engine wins regardless of outer. That threshold is
  what `kStridedV2VFatInnerBytes` should be set to.

### 5.2 `analyze_gate.py` — gated-vs-oracle and gated-vs-host

Combines the three JSON outputs and computes totals under each strategy,
an oracle (per-cell min of engine/host), win/regression counts vs always-host,
and worst overpayments vs oracle.

```bash
python analyze_gate.py /tmp/tighter_engine.json /tmp/tighter_host.json /tmp/tighter_gated.json gated
```

Reference numbers on the dev machine (340-cell tighter sweep,
rebel-compiler 0.10.5.dev81):

```
oracle (min eng/host) :       ~ 43000 μs   (1.00× oracle)
always-engine         :       ~ 47000 μs   (1.09× oracle)
always-host           :      ~7673000 μs   ( 177× oracle)
gated                 :       ~ 46000 μs   (1.05× oracle)

Gated wins >100μs vs always-host: 222 cells
Gated regresses >100μs vs always-host: 1 cell (225 μs, outer=16K inner=2B sf=256)
```

Pass criteria on a new device:

1. `gated` total < `always-host` total (the gate should be net helpful).
2. `gated` total <= `always-engine` total (the gate shouldn't *worsen* the
   engine's average outcome — it should at minimum match it, and ideally beat
   it by avoiding worst-case cells).
3. Zero regressions of >100 μs vs always-host in cells where `outer ≤ 1024`.
   Occasional ~hundreds-of-μs regressions in tiny-inner cells routed to
   engine by the span override are acceptable; if you see >1 ms deltas,
   the constants need work.

Also worth eyeballing the KV runs:

```bash
python analyze_gate.py /tmp/kv_engine.json /tmp/kv_host.json /tmp/kv_gated.json gated
```

On the dev machine the gated KV sweep matches the engine sweep cell-for-cell
(all 66 cells have `outer = 2 × num_kv_heads ≤ 16`, so the always-engine
clause fires unconditionally), which means the sub-block KV workload retains
the engine speedup regardless of `n_tokens` / `head_size`.

## 6. What to do with the results

- If the dev-machine constants (1024 / 256 B / 1 MB / 256K) reproduce the
  same shape of crossover on the target device → the gate is portable, ship
  as-is.
- If the outer-count knee has moved → adjust `kStridedV2VOuterAlways` to the
  new knee position, re-run `bench_worst_case.py` to confirm no regression,
  re-run `analyze_gate.py` to confirm the ratio improved.
- If the inner-bytes knee has moved → adjust `kStridedV2VFatInnerBytes` only;
  pick the smallest value at which engine still beats host across the
  `(outer > 1024, inner_bytes ≥ X)` band in `analyze_sweep.py`.
- If host bounce of sparse views costs differently (e.g. faster host link) →
  adjust `kStridedV2VLargeViewSpanBytes`; the span override should fire when
  span/payload ratio is large enough to dominate host-bounce time.
- `kStridedV2VOuterMax` bounds the span override: per-entry cost × outer
  must stay below the host bounce of span bytes. Derive it from the worst
  transpose timing under `bench_worst_case.py`.
- If `bench_v2v_engine.py` shows the engine isn't reaching the fast path at
  all → stop and check the rebel-compiler / runtime build; the gate cannot
  be tuned around a broken backend.

The gate is not expected to beat the per-cell oracle, but on the dev machine
it sits within 5% of it. Anything within ~1.5× oracle with no >1 ms
regressions and good behavior on the worst-case patterns is acceptable.
