# `torch.rbln.explain()` — the hidden-overhead explainer

## 1. What it is (and what it is NOT)

`torch.rbln.explain()` answers one question: **"What did my code make the RBLN backend do that I never asked for and cannot see?"**

A normal PyTorch op — `a.copy_(b)`, `torch.cat`, an indexing write, a forward pass — can silently round-trip the host (NPU→CPU→NPU), fall back to a CPU kernel, or recompile a graph. None of that shows up in your Python code; it just makes things slower for reasons that are invisible. `explain()` counts those hidden events, attributes the **cause**, prints a one-line **note** (a *fix* on the rows you can act on, a heads-up on the runtime-internal ones), and (opt-in) points at the **where** in your source.

Think of it as `torch._dynamo.explain()` or JAX's `transfer_guard` for the RBLN device — **a hidden-overhead explainer, not a timing profiler.**

> **It is NOT a timing profiler.** It does not tell you how long your forward pass took, which layer is slow, or your tok/s. A region with zero hidden overhead can still be slow (it may be legitimately device-compute-bound). For wall-clock timing use `torch.profiler` / `nsys`. `explain()` only surfaces the *hidden host overhead* — the part you didn't write and can't see.

**Zero-cost when idle (ON==OFF):** counters sit on already-slow points (a host DMA / fallback branch), never the fast device path, and reads are lazy — so leaving `explain()` in your code costs nothing when you're not profiling. (Inside a region the (B) timer adds one sub-µs clock read per librbln boundary call — negligible.)

## 2. Quick start

```python
import torch
import torch_rbln  # registers the rbln device + torch.rbln

with torch.rbln.explain() as p:
    model(x)            # any rbln-device work

print(p.report())       # [clean]/[overhead] marker + torch.profiler-style table
p.verdict()             # {'clean': bool, 'reasons': [...], ...}  -> CI gate
p.dump()                # full dict (every number) -> programmatic use
```

That is the whole loop: **wrap → glance at the `[clean]`/`[overhead]` marker → read the table → act on the Note column** (for the rows you can act on — some rows are runtime-internal notifications, see §below). The marker is a **fact** ("did anything hidden fire"), not a RED/AMBER/GREEN severity grade — cost lives in the table's Bytes/Note, which you judge.

## 3. The API

| Call | Returns | Use |
| --- | --- | --- |
| `explain(with_stack=False)` | `RBLNExplain` (context manager) | wrap a region you place yourself |
| `explain_steady(fn, *, warmup=2, return_cold=False, as_diff=False, trace=False)` | `RBLNExplain` / `(cold, warm)` / `RBLNDiff` | auto-place two regions around `fn` to separate first-call cost from steady-state |
| `p.report()` | `str` | the human-readable, marker-first report (print this) |
| `p.verdict()` | `dict` | `clean` flag + `reasons` (facts); for CI gating |
| `p.dump()` | `dict` | every signal as raw numbers; for programmatic checks |
| `p.help(signal=None)` | `str` | full remedy prose for a signal (or all fired signals) |
| `p.diff(other)` | `RBLNDiff` | compare two regions YOU placed (early vs later) |
| `p.start()` / `p.stop()` | — | manual region boundaries (instead of `with`) |
| `with_stack=True` | — | also capture the Python call-site of each fallback/recompile/bounce (opt-in) |

`profile` / `RBLNProfile` are back-compat aliases of `explain` / `RBLNExplain`. `with_stack=`
matches `torch.profiler.profile`'s parameter name; the older `trace=` still works as an alias.

## 4. Reading the report

A real report (an integer-metadata region) annotated line by line:

```
[overhead: 2 signals]  RBLN EXPLAIN   (wall 66.430ms | device mem 128.00 Mb peak (reserved))   (1)
  ! host oversubscription: 8 threads on 1 core(s) -> ... (tune affinity)  (2)
  rebel runtime: 4.367ms in librbln (6.6% of wall) -- acquire 1051.0(3000) | ...  (3)

---------------------  -----  -------  ---------------------------------------
Signal                 Count    Bytes  Note                               (4)
---------------------  -----  -------  ---------------------------------------
host_bounce/d2d_copy    1000  9.77 Kb  try: make contiguous, or v2v engine
dispatch/cpu_fallback   2000       --  try: graph mode, or a supported dtype
---------------------  -----  -------  ---------------------------------------
    physical d2h (real transfer): 0 - served on host, no device crossing          (5)
    cpu_fallback: aten::sub.out 1000, aten::mul.out 500, aten::clamp.out 500  (sum 10.885ms wall)  (6)
      why: dtype-not-fp16 2000                                                       (7)
      no fast-path handler (optimization candidates): aten::sub.out, aten::mul.out, aten::clamp.out  (8)
      where? -> rerun with explain(with_stack=True)                                  (9)
  (full fix detail -> p.help(signal) or dump()['remedies'])
```

1. **Marker + header.** `[clean]` (nothing hidden fired) / `[overhead: N signals]` (N distinct hidden signals fired). The count is a **fact, not a severity grade** — there is no RED/AMBER/GREEN, and N counts *how many kinds* of signal fired, not how bad they are. explain does not colour-judge how bad an event is (a host bounce can be free); cost is read from the table's `Bytes`/`Note` and the `physical d2h` line, which *you* judge. `wall` is the region's wall-clock (a reference, not the point of the tool). `device mem … peak` is the device-memory high-water mark — the rebel `BufferAllocator`'s **total physical device bytes, including cached/idle buffers held for reuse** (a *reserved footprint*, not just live tensors); see §6.
2. **(E) Host oversubscription** — a *resource fact*. The worker may run on far fewer CPU cores than it has threads (here 8 threads pinned to 1 core), so its tiny serial host ops get preempted and per-op latency inflates. This is an **environment amplifier**, not a per-op bug — see §6.
3. **(B) Rebel-runtime time** — how much of the region was spent *inside* librbln (the runtime: `borrow`/`v2v`/`h2v`/`acquire`/`return`/`v2h`), each shown as `µs(calls)`. The `%` is of the **wrapped region's** wall, so it reads as "runtime vs torch-side dispatch" only when the region is host-bound; a region that includes the device forward shows a tiny `%` because device compute dominates the wall (not because dispatch is cheap). See §6.
4. **The signal table.** Each fired signal: `Count` (how many times), `Bytes` (how much moved through host, if applicable), `Note` (terse what-to-change for the rows you can act on; for runtime-internal rows it is a notification, not a to-do). This is the core.
5. **Real device→host qualifier.** The torch-side `host_bounce` is a *copy-path* count — it is blind to whether the data actually crossed the device boundary. The runtime's transfer witness here says `physical d2h: 0` → **the bounce was served on the host, no device crossing → low cost, not a leak.** (If it said `>0`, real bytes crossed.)
6. **`cpu_fallback` by op** + `(sum … wall)`: which ops ran on CPU and the total fallback wall time (a *weak, noisy* A/B signal — inflated under oversubscription, not a per-op absolute; see §10).
7. **`why`**: the reason each fallback fired — `dtype-not-fp16` (the op's dtype is outside the device dispatch policy), `nan/inf input`, or `all-scalar inputs`.
8. **(A) `no fast-path handler`**: of the ops that fell back, which ones have **no CPU fast-path handler** — i.e. your **optimization candidates** (registered ops already bypass the slow boxed path).
9. **`where?`**: rerun with `explain(with_stack=True)` to get the exact Python call-site of each fallback/recompile/bounce.

### A gallery of report shapes

The example above is one shape. Below is the range you'll actually see, from cleanest to richest. Skim them once — recognizing the *shape* of a report is most of reading it.

**(G1) Clean.** Nothing hidden happened. (Note the 5.6 s wall: that is real decode time. `[clean]` means "no hidden host overhead", NOT "fast" — `explain` says nothing about device-compute time.)

```
[clean]  RBLN EXPLAIN   (wall 5.620s | device mem 2.46 Gb peak (reserved))
  (clean) no hidden overhead
```

**(G2) A host-served bounce — cheap.** An `int64 → int32` device cast round-trips host memory (a `d2d_copy` bounce), but `physical d2h (real transfer): 0` says it **never actually crossed the device boundary** — the runtime served it on the host. A host round-trip did occur, but the bytes (8 b) and cost are tiny. This is exactly why the marker is a fact, not a grade: `[overhead: 1 signal]` flags that *something* fired; the `physical d2h: 0` line tells you it was cheap. This is the int-cast in the sampler (`sampled.to(torch.int32)`).

```
[overhead: 1 signal]  RBLN EXPLAIN   (wall 190.000us | device mem 128.00 Mb peak (reserved))

--------------------  -----  -----  -----------------------------------
Signal                Count  Bytes  Note
--------------------  -----  -----  -----------------------------------
host_bounce/d2d_copy      1    8 b  try: make contiguous, or v2v engine
--------------------  -----  -----  -----------------------------------
    physical d2h (real transfer): 0 - served on host, no device crossing
```

**(G3) A REAL device→host crossing — real bytes.** The contrast to G2: here the source was device-only and the layout couldn't be planned on-device, so the runtime did a **genuine DMA** — `physical d2h (real transfer): 1 copies, 512 b`. Same `[overhead: 1 signal]` marker as G2, but the `physical d2h` line now shows a real crossing — that line, not a colour, is how you tell the expensive kind from the cheap one. (The `Bytes` cell's meaning is spelled out in the `Note`: `(bytes: physical d2h)` here vs `(bytes: host path)` for a host-served row.)

```
[overhead: 1 signal]  RBLN EXPLAIN   (wall 510.000us | device mem 4.91 Gb peak (reserved))

----------------  -----  -----  ---------------------------------------------------------
Signal            Count  Bytes  Note
----------------  -----  -----  ---------------------------------------------------------
runtime/v2v_slow      1  512 b  fyi: real device->host DMA (costly) (bytes: physical d2h)
----------------  -----  -----  ---------------------------------------------------------
    state: src_device_only_real_d2h 1
    reject: dtype_mismatch 1 -> align the src/dst dtype
    physical d2h (real transfer): 1 copies, 512 b
```

> One `runtime/v2v_slow` event with sublines: the `state:` axis (where the bytes went), the `reject:` axis (WHY the fast plan was rejected — here `dtype_mismatch`, which you CAN act on; an internal reason would read `internal_fallback … runtime-internal`), and the authoritative `physical d2h` transfer. They are the same event seen three ways — do not add their counts.

> G2 vs G3 is the single most important distinction: a `host_bounce` is a *copy-path* count (blind to residency); the `physical d2h` line is the *DMA witness* for the copy the row is qualifying. `0` = served on host (cheap); `>0` = real crossing (the thing to fix). (It witnesses **synchronous / host-served** transfers; a deliberate `non_blocking=True` copy on **pinned** host memory takes the async DMA path and is intentionally not itemized — see the note in §6.)

**(G4) A real vLLM decode step (device-tensor mode) — the capstone.** This is a 400-step steady-decode window of a Llama-1B run with `VLLM_RBLN_USE_DEVICE_TENSOR=1`, captured with `with_stack=True`. It shows nearly every signal at once: the attention-metadata integer math falling back (`sub`/`mul`/`clamp`, `dtype-not-fp16`), the `positions[idx]` gather going host-slow (`v2v_slow` + the `d2d_copy` bounce), all **host-served** (`physical d2h: 0`), under a worker pinned to one core (oversubscription), with the rebel-runtime share and the exact source lines:

```
[overhead: 3 signals]  RBLN EXPLAIN   (wall 5.791s | device mem 5.06 Gb peak (reserved))
  ! host oversubscription: 8 threads on 1 core(s) -> per-op host latency may be inflated (tune affinity / OMP_NUM_THREADS)
  rebel runtime: 24.673ms in librbln (0.4% of wall) -- h2v 7289.0(3600) | acquire 5455.0(2400) | v2v_multi 4140.0(400) | return 3488.0(4400) | borrow 2250.0(2000) | v2h 2051.0(1600)

---------------------  -----  --------  ---------------------------------------------------------
Signal                 Count     Bytes  Note
---------------------  -----  --------  ---------------------------------------------------------
host_bounce/d2d_copy    1600  12.59 Kb  try: make contiguous, or v2v engine
runtime/v2v_slow         796       0 b  fyi: served on host, no DMA (low cost) (bytes: host path)
dispatch/cpu_fallback   1600        --  try: graph mode, or a supported dtype
---------------------  -----  --------  ---------------------------------------------------------
    state: src_not_on_device 796
    reject: internal_fallback 796 (runtime-internal, not user-fixable)
    physical d2h (real transfer): 0 - served on host, no device crossing
    at host_bounce/d2d_copy: flash_attention.py:1126(build) <- rbln_model_runner.py:1423(_prepare_inputs)
    cpu_fallback: aten::sub.out 800, aten::mul.out 400, aten::clamp.out 400   (sum 41.084ms wall)
      why: dtype-not-fp16 1600
      at aten::sub.out: rbln_model_runner.py:1276(_prepare_inputs) <- rbln_model_runner.py(execute_model)
      at aten::mul.out: flash_attention.py:1148(build) <- rbln_model_runner.py:1423(_prepare_inputs)
      at aten::clamp.out: flash_attention.py:1147(build) <- rbln_model_runner.py:1423(_prepare_inputs)
```

How to read G4 in 30 seconds:

- **`[overhead: 3 signals]`**, driven by `host_bounce` (1600) + `v2v_slow` (796) + `cpu_fallback` (1600). The bounce and v2v are **host-served** (`physical d2h: 0`) — no DMA, so the cost is host-side machinery, not data movement. (No colour claims this is severe; the `physical d2h: 0` line is what tells you the crossing cost is nil.)
- **What**: `cpu_fallback` is the integer metadata math (`sub`/`mul`/`clamp`, all `dtype-not-fp16`); `v2v_slow`/`bounce` is the `positions[idx]` gather. Per 400 steps: 4 fallbacks + ~2 gathers + 4 bounces per step.
- **Where**: the `at …` lines pin it to `flash_attention.py:1126/1147/1148` (the metadata builder) and `rbln_model_runner.py:1276` (logits). All of it is one cluster.
- **Context**: the `! oversubscription` warning flags that these per-op costs may be inflated by CPU contention (worker on 1 core / 8 threads) — check affinity before micro-optimizing. The `rebel runtime` share here is tiny (0.4%) because this region wraps the device forward; the takeaway is still that the runtime is cheap per call, so the lever is the dispatched-op count / dtype, not the runtime.
- **Act**: the (A) candidates (`sub`/`mul`/`clamp`) want fast-path handlers or CPU placement; the gather wants device residency or a host fast-path. (See the per-op cost: `sum 41.084ms` is the fallback wall over the window — a coarse magnitude, not a per-op absolute; cf. §10.)

(The same step under `VLLM_RBLN_USE_DEVICE_TENSOR=0`, where the metadata stays on CPU as native ops, is **G1 — fully `[clean]`**. That A/B is exactly how you localize a device-tensor regression.)

## 5. Signal reference

Every signal is a thing the backend did behind a normal op. The table gives the meaning, the intent (why you should care), and the lever.

| Signal (report label) | Side | What happened | Why it matters / intent | Lever |
| --- | --- | --- | --- | --- |
| `host_bounce/d2d_copy` (dump key `copy_d2d_host_bounce`) | torch | a non-contiguous device→device `copy_` round-tripped the host | a host round-trip you didn't ask for; the data left the device | make the copy contiguous, or route through the on-device v2v engine |
| `host_bounce/copy_h2d_staging` | torch | a CPU source was staged to host before the h2v push | keep the source on-device, or stage once and reuse | keep src on-device / stage once |
| `host_bounce/copy_h2d_noncontig_dst` | torch | host→device write into a non-contiguous device dst | write into a contiguous buffer first, then h2d | contiguous staging, then h2d |
| `host_bounce/strided_v2v_cpu_fallback` | torch | a strided v2v (cat/index/copy_) fell back to a host CPU op | the device v2v engine rejected the geometry | lower outer_count / fatter contiguous inner block |
| `host_bounce/v2v_batch_to_per_entry` | torch | a batched v2v was rejected to per-entry copies | batch geometry exceeded the per-dst limit | check `kMaxV2VMultiCopies` / batch geometry |
| `runtime/v2v_slow` | rebel | a device v2v fell to the host slow path; ONE event row with `state:` / `reject:` / `physical d2h` sublines (the same event, three views — don't add their counts) | a device copy did not stay on device | see the sublines below |
| ↳ `state: src_not_on_device` | rebel | the source was host-latest | (state axis) | keep the source on device |
| ↳ `state: src_device_only_real_d2h` | rebel | device-only source forced a real d2h | (state axis) a real device→host transfer happened | — |
| ↳ `state: src_synced_host_served` | rebel | src on host+device; host memcpy, no transfer | (state axis) usually benign | — |
| ↳ `reject: <user reason>` (e.g. `dtype_mismatch`) | rebel | WHY the fast plan was rejected, when you can act | user-actionable | align dtype / alignment / ≤30 chunks |
| ↳ `reject: internal_fallback` | rebel | WHY the fast plan was rejected, runtime-internal (non-deterministic detail) | a notification, NOT a to-do | not user-fixable; see `dump['…']['debug']` if recurring |
| `dispatch/cpu_fallback` | torch | an op ran on CPU (fp16-only NPU can't run it) | the per-op tax for non-device dtypes (e.g. int metadata) | graph mode / native rbln kernel / fix dtype — see (A) candidates |
| `dispatch/recompile` | torch | a graph (re)compiled (warm-cache miss) | a cold first compile is expected; **repeated** recompiles in a steady loop are not | stabilize shapes (pad/bucket) for warm-cache reuse, or graph mode |

And the auxiliary readouts (in `dump()`):

- `cpu_fallback_by_op`, `recompile_by_op` — per-op counts (what fell back / recompiled).
- `cpu_fallback_reasons` — the `dtype-not-fp16 / nan-inf / all-scalar` breakdown.
- `cpu_fallback_unaccelerated` (A) — fallback ops with no fast-path handler.
- `real_host_sync_d2h` / `real_host_sync_h2d` — count+bytes of **real** synchronous/host-served physical transfers (vs the copy-path bounce count). Deliberate pinned `non_blocking` async transfers are out of scope (§6).
- `trace_by_op` (A WHERE) — op/site → call-site, only when `with_stack=True`.
- `host_threads` (E), `rebel_runtime` (B), `device_memory` — resource facts (§6).

## 6. What the marker means — and what stays a fact (not a grade)

The `[clean]` / `[overhead: N signals]` marker is a **fact**: did any *hidden host overhead* — overhead you issued as a normal op and cannot see — fire in the region, and how many *kinds*? That is all it claims. The count is not a severity grade (there is no RED/AMBER/GREEN): a host bounce can be free, a real d2h can be costly, and the tool will not guess which by colour or by N. `[overhead: N signals]` means "there are N kinds of thing in the table to look at"; the table's `Bytes`/`Note` and the `physical d2h` line carry the **cost**, and **who can act on it** is a separate axis in the row's `Note`/sublines (some of it is runtime-internal, not yours to fix).

What flips the marker to `[overhead: N signals]` (each counts as one of the N):

- a **host bounce** or a runtime **`v2v_slow`** fired — a device copy did not stay on device. The `v2v_slow` sublines decompose it: the `reject:` axis says whether the cause is **user-actionable** (e.g. `dtype_mismatch`) or **runtime-internal** (`internal_fallback`, a notification, not a to-do), and `physical d2h` is the real-transfer witness (`0` = served on host, cheap; `>0` = a real crossing). `v2v_slow` includes `src_synced_host_served` (host memcpy, no transfer), so an overhead marker driven **only** by that is usually low-cost (see §11).
- a **`cpu_fallback`** or **`recompile`** fired (ran on CPU / recompiled).

`[clean]` means none of the above fired (it does **not** mean fast — see §10).

These are deliberately **facts, NOT marker drivers** (they inform, they don't accuse):

| Fact | What it tells you | Why it's a fact, not a finding |
| --- | --- | --- |
| `device_memory` (peak) | rebel `BufferAllocator` physical device footprint incl. cached buffers (reserved, not live) | a number, not an unwanted event |
| `host_threads` (E) — oversubscription | environment amplifier of *all* host overhead | it's your deployment's CPU config, not a bug in any op |
| `rebel_runtime` (B) — time in librbln | splits host cost into runtime vs torch-dispatch | tells you *where* to look, doesn't itself accuse |
| `real_host_sync_h2d` push | the lazy push that feeds a device graph | a push to feed a graph is *expected*, unlike an unwanted d2h pull |

This separation is the point: an `[overhead]` finding is something to **inspect and account for** (a fix where the row says you can, a cost to acknowledge where it's runtime-internal); a fact is context that helps you decide *how*.

> **`device_memory`** is the rebel `BufferAllocator`'s **reserved footprint** (physical bytes held, including idle cached buffers), a high-water gauge — *not* live-tensor bytes. For the live-vs-reserved split use `torch.rbln.memory_stats()`.

> **Transfer counters** (`physical d2h`, `host->device push`) count **synchronous / async-fallback** transfers only. A deliberate pinned `non_blocking=True` copy takes the async DMA path and is intentionally not itemized (it never flips the marker); use `torch.profiler` / `nsys` to account for that volume.

Two facts are worth dwelling on, because they decide your whole strategy:

- **(E) Oversubscription.** If `explain` warns `host oversubscription: N threads on M cores`, the per-op host costs you see **may be inflated by CPU contention** (idle OMP/numba threads spinning on the same cores), rather than caused by any single op. In practice the same op was measured ~5× slower in a worker pinned to one core with 8 threads than in a clean process. **The first lever is then affinity / `OMP_NUM_THREADS` / thread config — not your code.** Chasing per-op micro-optimizations while oversubscribed wastes effort. (It is a heuristic — many idle threads on few cores; it says *may*, not *will*.)
- **(B) Rebel-runtime share.** The `%` is of the **wrapped region's wall**, so read it accordingly. Wrap a *host-bound* region (e.g. just the metadata builder): a small share — say 6% — means the other ~94% of that region is **torch-side dispatch** (dispatcher, boxing, TensorIterator, Python), so the lever is the dispatched-op count, not the runtime. But if you wrap a step that includes the device forward, expect a *tiny* share (e.g. 0.4%) **simply because device compute dominates the wall** — that does NOT mean dispatch is 94%, it means you wrapped the forward. Either way each librbln boundary call is sub-µs, so a *large* rebel share (in a host-bound region) is the only case where the runtime itself is the lever.

## 7. WHERE: `explain(with_stack=True)`

By default `explain()` captures no call-sites (so it adds nothing). With `with_stack=True`, the first time each op falls back / recompiles / bounces, its Python call-site is captured and shown under the offending op:

```python
with torch.rbln.explain(with_stack=True) as p:
    model(x)
print(p.report())
#   cpu_fallback: aten::sub.out 800 ...
#       at aten::sub.out: rbln_model_runner.py:1276(_prepare_inputs) <- ...
```

Use it the moment the report says `where? -> rerun with explain(with_stack=True)`. It turns "something fell back 800 times" into "*this line* fell back". (`trace=True` remains as a back-compat alias.)

## 8. One-time vs recurring: `diff` and `explain_steady`

**`explain` cannot tell a one-time cost from a recurring one on its own.** It observes a bounded region; it has no idea whether that region is your first step or your thousandth, or whether each run does the same work. So it never labels a signal "cold/one-time" by itself — a wrong "ignore this, it's one-time" is worse than no label.

To make that distinction, place two regions YOURSELF and compare:

```python
with torch.rbln.explain() as early:   # first use (cold compile expected here)
    model(x)
for _ in range(5):
    model(x)                          # warm up
with torch.rbln.explain() as later:   # steady state
    model(x)
print(early.diff(later).report())
#   "gone in B"  -> did not recur (one-time, e.g. cold compile)
#   "*** PERSISTS" + op + call-site -> recurring overhead (the real target)
```

`explain_steady(fn, warmup=2, as_diff=True)` automates exactly this (cold = first call it makes, warm = a later call). Its labels mean literally "first call I made" vs "a later call I made" — valid as one-time-vs-recurring only if (1) `fn` was not already compiled before, and (2) every `fn()` does the same work. Only you can ensure that.

## 9. Cases & corner cases (seen in practice)

**(a) Host-served bounce — cheap.** An `int64 → int32` device cast bounces (`d2d_copy`), but the qualifier reads `physical d2h (real transfer): 0 — served on host`. The copy went *through* host memory but **never crossed the device boundary** (no DMA). The marker reads `[overhead: 1 signal]` (a host round-trip did happen) but it is low-cost. The `real_*_sync` counters are how you tell a true device crossing from a host-served copy (for synchronous/host-served copies; deliberate pinned `non_blocking` transfers are out of scope, §6) — this is exactly why the marker is a fact, not a severity grade.

**(c) The "`[clean]` but slow" trap.** Wrapping only the compiled forward (`model_executable`) often shows `[clean]` — the compiled graph is clean by construction. But the real hidden overhead is usually in the **glue** (metadata builder, sampler, padding) that runs *outside* that narrow wrap. **Wrap the whole step**, not just the forward, or `explain` will honestly report "nothing hidden here" about a region that wasn't where the cost was.

**(d) The oversubscription rabbit hole.** A decode step looked like it had a large per-op host cost. The cause was not any op — the worker was pinned to **1 core with 8 threads**, so every tiny serial host op was preempted (~5× inflation). The (E) warning surfaces this directly; without it you can burn days chasing per-op fixes that affinity config would solve.

## 10. Pitfalls / how to read it honestly

- **`[clean]` ≠ fast.** `[clean]` means "no hidden host overhead in this region." The region can still be device-compute-bound. Use a timer for speed.
- **Narrow wrap hides the cost.** If you wrap only the compiled forward, the glue/sampler overhead is outside the region. Wrap the full step.
- **`cpu_fallback_ns` (the `sum … wall`) is a weak signal.** It is noisy and inflated several-fold under a pinned-core worker; the first fallback in a region is much more expensive than steady ones. Treat it as a coarse A/B over many ops, never a per-op absolute.
- **e2e wall is noisy.** Run-to-run host cost can swing ±10%+ (thermal, scheduling). The *mechanism* signals (which op fell back, the bounce count, the bytes) are far more stable than the wall — trust those, and use interleaved A/B for any wall comparison.
- **A region is a delta.** Each region reports only what happened inside it (memory is a level/high-water, not a delta). A fresh region does not inherit the previous region's counts.

## 11. Practical playbook

| You see… | It means… | Do… |
| --- | --- | --- |
| `! host oversubscription` | host ops inflated by CPU contention | fix affinity / `OMP_NUM_THREADS` first — before any per-op work |
| `cpu_fallback` + `no fast-path handler: <ops>` | those ops run on CPU with the full boxed tax | add a `fast_paths/*.cpp` handler for them, or keep that metadata on CPU |
| `host_bounce` or `v2v_slow` with `physical d2h > 0` | real bytes crossed the host | make the copy contiguous / route through the v2v engine |
| `host_bounce` or `v2v_slow` with `physical d2h: 0` | host-served, no device crossing | low priority; the machinery overhead is small |
| `dispatch/recompile` recurring (via `diff`) | shapes aren't stabilizing | pad/bucket to a fixed shape so the warm cache hits |
| `rebel runtime` is a large % | the runtime itself is the cost | optimize the runtime path |
| `rebel runtime` is a small % | torch-side dispatch dominates | reduce dispatched-op count; runtime micro-opt won't help |
| `where? -> with_stack=True` | you need the source location | rerun with `explain(with_stack=True)` |
| overhead marker only from `src_synced_host_served` | benign host memcpy | usually no action |

## 12. CI gating

`dump()` / `verdict()` are stable dicts — assert on them. Gate on the raw numbers you care about (more precise than any marker):

```python
with torch.rbln.explain() as p:
    run_one_decode_step()
d = p.dump()
assert d["hidden_host_bounce"]["total_count"] == 0, "a host round-trip regressed"
assert not d.get("cpu_fallback_unaccelerated"), f"un-accelerated fallback ops: {d['cpu_fallback_unaccelerated']}"

# or a coarse "did anything hidden fire?" gate via the factual flag:
assert p.verdict()["clean"], f"hidden overhead regressed: {p.verdict()['reasons']}"
```

The counters this gate reads are zero-cost (cold-path atomics, lazy reads); the only in-region cost is the (B) runtime timer's sub-µs clock read per librbln boundary call, which is negligible (see §1). So the gate barely perturbs the measured region.

## 13. Scope — what `explain` deliberately does NOT do

- It is not a timing profiler (no per-layer wall, no tok/s) — that's `torch.profiler` / `nsys`.
- It does not surface command-stream counts, device-idle time, or total host-traffic bytes — those are generic utilization metrics that live in the TVM runtime (where they read ~0 here) and would mislead; they are intentionally excluded.
- It does not guess your run lifecycle (one-time vs recurring) — you supply that via `diff`.

The single hidden signal still on the roadmap is a side-effect host-sync on the TVM graph-exec path (a runtime h2v during graph execution that no current counter sees); everything else here is the complete, honest surface today.
