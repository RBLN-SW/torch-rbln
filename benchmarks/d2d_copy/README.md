# d2d copy benchmark suite

Microbenchmarks and analysis tools for device-to-device (`rbln` → `rbln`)
copy performance. Use these to characterize the crossover between two
implementations of strided d2d copies:

- **engine** — `strided_v2v_copy`: split the copy into the largest jointly
  contiguous inner block × an outer iteration; enqueue `outer_count` v2v
  entries and dispatch through `rbln_memcpy_v2v_multi`.
- **host bounce** — copy device → CPU → device
  (`get_cpu_copy_of_rbln_tensor` → `tensor_copy_from_cpu_to_rbln`).

Cost model:

```
cost_engine ≈ outer_count * per_entry_overhead + total_bytes / device_bw
cost_host   ≈ d2h(view_span) + h2d(numel)     + small per-call overhead
```

The engine wins when `outer_count` is small or the inner block is fat; host
bounce wins for huge `outer_count` with tiny inner blocks (e.g. transpose).
Where exactly the crossover sits is hardware-dependent — the sweep scripts
below measure it.

Commands assume an active venv and a current editable build, run from this
directory. Do not run two benches on the same NPU concurrently — timings
become noisy.

## Benchmark scripts

| Script                       | Purpose                                                                  | Output |
| ---------------------------- | ------------------------------------------------------------------------ | ------ |
| `bench_v2v_engine.py`        | Probe — does the v2v engine reach the device fast path at all?           | stdout |
| `bench_worst_case.py`        | Pathological patterns: transpose / permute / strided-inner               | stdout |
| `bench_crossover_tighter.py` | Sweep over `(outer_count, inner_elems, stride_factor)`                   | JSON   |
| `bench_kv_shapes.py`         | vllm-rbln sub-block KV copy across `(kv_heads, head_size, block, n_tok)` | JSON   |
| `analyze_sweep.py`           | Crossover characterization from a pair of sweep JSONs                    | stdout |

### `bench_v2v_engine.py`

Compares `aten::copy_` on a sub-block KV slice against `index_copy_` on the
same geometry. The native `index_copy_` kernel always drives the engine, so
this answers "can this device reach the v2v fast path?" independently of
`copy_`'s routing. Reference (dev machine): `index_copy_` ≈ 160 μs vs ≈ 50 ms
when `copy_` falls back to host-bounce on the same 64 MB view.

```bash
python bench_v2v_engine.py
```

### `bench_worst_case.py`

Stride patterns whose largest contiguous block is one element (`outer_count` =
numel). On the dev machine `strided_v2v_copy` is 100–10000× slower than host
bounce here; the host path takes tens of μs.

```bash
python bench_worst_case.py
```

### `bench_crossover_tighter.py`

Main sweep. For each `(outer, inner_elems, stride_factor)` cell, times
`dst.copy_(view)` where `view = parent[:, 0, :]` pins the geometry exactly:
`outer_count = outer`, inner block = `inner_elems` elements, view span ≈
`stride_factor × numel`. 316 cells, ~5–10 min per run.

```bash
python bench_crossover_tighter.py --label <build-label> --out /tmp/sweep_<label>.json
```

Run once per build configuration under test (e.g. a build forced to engine vs
forced to host bounce), then compare:

```bash
python analyze_sweep.py /tmp/sweep_engine.json /tmp/sweep_host.json
```

The analyzer prints per-cell ratios, wins/losses, and win-rate bucketed by
outer count and inner block size — the data needed to place engine/host
routing thresholds.

### `bench_kv_shapes.py`

The workload that motivated all of this: vllm-rbln sub-block KV cache writes
`kv[:, dst, :, :, :n, :] = kv[:, src, :, :, :n, :]`. Sweeps
`num_kv_heads ∈ {2,4,8}`, `head_size ∈ {64,128}`, `block_size ∈ {1024,4096}`,
`n_tokens` from sub-block to full block. Reports the geometry each cell
presents (`outer_count`, `inner_bytes`, `view_span`) alongside timing.

```bash
python bench_kv_shapes.py --label <build-label> --out /tmp/kv_<label>.json
```

KV cells have `outer_count = 2 × num_kv_heads ≤ 16` regardless of `n_tokens`
— they sit deep in engine-wins territory.
