# `torch.rbln.explain()` — the hidden-overhead explainer

## 1. What it is (and what it is NOT)

`torch.rbln.explain()` answers one question: **"What did my code make the RBLN backend do that I never asked for and cannot see?"**

A normal PyTorch op — `a.copy_(b)`, `torch.cat`, an indexing write, a forward pass — can silently round-trip the host (NPU→CPU→NPU), fall back to a CPU kernel, or recompile a graph. None of that shows up in your Python code; it just makes things slower for reasons that are invisible. `explain()` counts those hidden events, attributes the **cause**, prints a one-line **fix**, and (opt-in) points at the **where** in your source.

Think of it as `torch._dynamo.explain()` or JAX's `transfer_guard` for the RBLN device — **a hidden-overhead explainer, not a timing profiler.**

> **It is NOT a timing profiler.** It does not tell you how long your forward pass took, which layer is slow, or your tok/s. A region with zero hidden overhead can still be slow (it may be legitimately device-compute-bound). For wall-clock timing use `torch.profiler` / `nsys`. `explain()` only surfaces the *hidden host overhead* — the part you didn't write and can't see.

**Zero-cost when idle (ON==OFF):** the counters (host_bounce / cpu_fallback / recompile / v2v_slow) sit on already-slow points (a host DMA, a fallback branch) — never the fast device path — and reads are lazy, so leaving `explain()` in your code costs nothing when you are not profiling. The one exception is the **(B) runtime timer**: it is gated to the explain region — *outside* any region it is a single relaxed atomic load per librbln boundary call; *inside* a region it adds a sub-µs clock read per boundary call. That is negligible (boundary calls are a tiny fraction of any region), so profiling barely perturbs the region — just not literally zero while a region is open.

## 2. Quick start

```python
import torch
import torch_rbln  # registers the rbln device + torch.rbln

with torch.rbln.explain() as p:
    model(x)            # any rbln-device work

print(p.report())       # verdict-first, torch.profiler-style table
p.verdict()             # {'status': 'GREEN'|'AMBER'|'RED', ...}  -> CI gate
p.dump()                # full dict (every number) -> programmatic use
```

That is the whole loop: **wrap → read the verdict → read the table → act on the Fix column.**

## 3. The API

| Call | Returns | Use |
| --- | --- | --- |
| `explain(trace=False)` | `RBLNExplain` (context manager) | wrap a region you place yourself |
| `explain_steady(fn, *, warmup=2, return_cold=False, as_diff=False, trace=False)` | `RBLNExplain` / `(cold, warm)` / `RBLNDiff` | auto-place two regions around `fn` to separate first-call cost from steady-state |
| `p.report()` | `str` | the human-readable, verdict-first report (print this) |
| `p.verdict()` | `dict` | `status` + reasons; for CI gating |
| `p.dump()` | `dict` | every signal as raw numbers; for programmatic checks |
| `p.help(signal=None)` | `str` | full remedy prose for a signal (or all fired signals) |
| `p.diff(other)` | `RBLNDiff` | compare two regions YOU placed (early vs later) |
| `p.start()` / `p.stop()` | — | manual region boundaries (instead of `with`) |
| `trace=True` | — | also capture the Python call-site of each fallback/recompile/bounce (opt-in) |

`profile` / `RBLNProfile` are back-compat aliases of `explain` / `RBLNExplain`.

## 4. Reading the report

A real report (an integer-metadata region) annotated line by line:

```
[BAD ]  RBLN EXPLAIN — RED   (wall 66.43 ms · mem 128.00 MB peak)        (1)
  ! host oversubscription: 8 threads on 1 core(s) -> ... (tune affinity)  (2)
  rebel runtime: 4367.3 us in librbln (6.6% of wall) -- acquire 1051.0(3000) · ...  (3)

--------------------------------  -----  -------  ----------------------------------
Signal                            Count    Bytes  Fix                                (4)
--------------------------------  -----  -------  ----------------------------------
host_bounce/copy_d2d_host_bounce   1000  9.77 KB  make contiguous, or v2v engine
dispatch/cpu_fallback              2000        -  graph mode / native kernel / dtype
--------------------------------  -----  -------  ----------------------------------
    real device->host (runtime): 0 — host_bounce above was served on host          (5)
    cpu_fallback: aten::sub.out 1000, aten::mul.out 500, aten::clamp.out 500  (Σ 10885 µs wall)  (6)
      why: dtype-not-fp16 2000                                                       (7)
      no fast-path handler (optimization candidates): aten::sub.out, aten::mul.out, aten::clamp.out  (8)
      where? -> rerun with explain(trace=True)                                       (9)
  (full fix detail -> p.help(signal) or dump()['remedies'])
```

1. **Verdict + level.** `[ OK ]` GREEN / `[WARN]` AMBER / `[BAD ]` RED. `wall` is the region's wall-clock (a reference, not the point of the tool). `mem … peak` is the device-memory high-water mark — the rebel `BufferAllocator`'s **total physical device bytes, including cached/idle buffers held for reuse** (a *reserved footprint*, not just live tensors); see §6.
2. **(E) Host oversubscription** — a *resource fact*. The worker may run on far fewer CPU cores than it has threads (here 8 threads pinned to 1 core), so its tiny serial host ops get preempted and per-op latency inflates. This is an **environment amplifier**, not a per-op bug — see §6.
3. **(B) Rebel-runtime time** — how much of the region was spent *inside* librbln (the runtime: `borrow`/`v2v`/`h2v`/`acquire`/`return`/`v2h`), each shown as `µs(calls)`. The `%` is of the **wrapped region's** wall, so it reads as "runtime vs torch-side dispatch" only when the region is host-bound; a region that includes the device forward shows a tiny `%` because device compute dominates the wall (not because dispatch is cheap). See §6.
4. **The signal table.** Each fired signal: `Count` (how many times), `Bytes` (how much moved through host, if applicable), `Fix` (terse what-to-change). This is the core.
5. **Real device→host qualifier.** The torch-side `host_bounce` is a *copy-path* count — it is blind to whether the data actually crossed the device boundary. The runtime's authoritative witness here says `0` real d2h → **the bounce was served on the host, no device crossing → low cost, not a leak.** (If it said `>0`, real bytes crossed.)
6. **`cpu_fallback` by op** + `(Σ … µs wall)`: which ops ran on CPU and the total fallback wall time (a *weak, noisy* A/B signal — inflated under oversubscription, not a per-op absolute; see §10).
7. **`why`**: the reason each fallback fired — `dtype-not-fp16` (the op's dtype is outside the device dispatch policy), `nan/inf input`, or `all-scalar inputs`.
8. **(A) `no fast-path handler`**: of the ops that fell back, which ones have **no CPU fast-path handler** — i.e. your **optimization candidates** (registered ops already bypass the slow boxed path).
9. **`where?`**: rerun with `explain(trace=True)` to get the exact Python call-site of each fallback/recompile/bounce.

### A gallery of report shapes

The example above is one shape. Below is the range you'll actually see, from cleanest to richest. Skim them once — recognizing the *shape* of a report is most of reading it.

**(G1) Clean — GREEN.** Nothing hidden happened. (Note the 5.6 s wall: that is real decode time. GREEN means "no hidden host overhead", NOT "fast" — `explain` says nothing about device-compute time.)

```
[ OK ]  RBLN EXPLAIN — GREEN   (wall 5619.52 ms · mem 2.46 GB peak)
  (clean) no hidden overhead
```

**(G2) One CPU fallback, no host crossing — AMBER.** An integer op (`int32 + int32`) can't run on the fp16-only NPU, so it ran on CPU — but no data crossed the host, so it's AMBER, not RED. The (A) line flags that `add.out` has no fast-path handler.

```
[WARN]  RBLN EXPLAIN — AMBER   (wall 0.28 ms · mem 128.00 MB peak)

----------------------  -----  -----  ----------------------------------
Signal                  Count  Bytes  Fix
----------------------  -----  -----  ----------------------------------
dispatch/cpu_fallback       1      -  graph mode / native kernel / dtype
----------------------  -----  -----  ----------------------------------
    cpu_fallback: aten::add.out 1
      why: dtype-not-fp16 1
      no fast-path handler (optimization candidates): aten::add.out
```

**(G3) A host-served bounce — RED, but cheap.** An `int64 → int32` device cast round-trips host memory (a `copy_d2d_host_bounce`), but `real device->host (runtime): 0` says it **never actually crossed the device boundary** — the runtime served it on the host. RED (a host round-trip did occur) but the bytes (8 B) and cost are tiny. This is the int-cast in the sampler (`sampled.to(torch.int32)`).

```
[BAD ]  RBLN EXPLAIN — RED   (wall 0.19 ms · mem 128.00 MB peak)

--------------------------------  -----  -----  ----------------------------------
Signal                            Count  Bytes  Fix
--------------------------------  -----  -----  ----------------------------------
host_bounce/copy_d2d_host_bounce      1    8 B  make contiguous, or v2v engine
--------------------------------  -----  -----  ----------------------------------
    real device->host (runtime): 0 — host_bounce above was served on host
```

**(G4) A REAL device→host crossing — RED, real bytes.** The contrast to G3: here the source was device-only and the layout couldn't be planned on-device, so the runtime did a **genuine DMA** — `real device->host (runtime): 1 copies, 512 B`. This is the expensive kind; the count/bytes are non-zero because data actually left the device.

```
[BAD ]  RBLN EXPLAIN — RED   (wall 0.51 ms · mem 4.91 GB peak)

------------------------------------------  -----  -----  --------------------------------
Signal                                      Count  Bytes  Fix
------------------------------------------  -----  -----  --------------------------------
runtime/v2v_slow:src_device_only_real_d2h       1      -  establish device residency first
------------------------------------------  -----  -----  --------------------------------
    real device->host (runtime): 1 copies, 512 B
```

> G3 vs G4 is the single most important distinction: a `host_bounce` is a *copy-path* count (blind to residency); the `real device->host` line is the *authoritative DMA witness*. `0` = served on host (cheap); `>0` = real crossing (the thing to fix).

**(G5) Recompiles in the steady loop — AMBER.** A decode step recompiles 33 graphs — `neg.out` recompiles **every token** (RoPE `rotate_half`). A cold first compile is expected; recurring recompiles are not. Confirm recurrence with `diff` (§8) before acting.

```
[WARN]  RBLN EXPLAIN — AMBER   (wall 234.10 ms · mem 4.91 GB peak)

----------------------  -----  -----  -------------------------------
Signal                  Count  Bytes  Fix
----------------------  -----  -----  -------------------------------
dispatch/recompile         33      -  stabilize shapes, or graph mode
----------------------  -----  -----  -------------------------------
    recompile: aten::neg.out 32, aten::mul.out 1
```

**(G6) A device-feeding push — a FACT, region still GREEN.** A matmul whose operands are host-latest pushes them to the device to feed the graph: `host->device push (runtime): 3 pushes, 352 KB`. That is **expected** (a graph must read device data), so it does NOT turn the verdict RED — it's surfaced as a fact so device-tensor glue cost is *visible*, not accused.

```
[ OK ]  RBLN EXPLAIN — GREEN   (wall 1.20 ms · mem 5.00 GB peak)
  host->device push (runtime): 3 pushes, 352.00 KB
  (clean) no hidden overhead
```

**(G7) A real vLLM decode step (device-tensor mode) — the capstone.** This is a 400-step steady-decode window of a Llama-1B run with `VLLM_RBLN_USE_DEVICE_TENSOR=1`, captured with `trace=True`. It shows nearly every signal at once: the attention-metadata integer math falling back (`sub`/`mul`/`clamp`, `dtype-not-fp16`), the `positions[idx]` gather going host-slow (`v2v_slow` + the `copy_d2d_host_bounce`), all **host-served** (`real d2h: 0`), under a worker pinned to one core (oversubscription), with the rebel-runtime share and the exact source lines:

```
[BAD ]  RBLN EXPLAIN — RED   (wall 5790.68 ms · mem 5.06 GB peak)
  ! host oversubscription: 8 threads on 1 core(s) -> per-op host latency may be inflated (tune affinity / OMP_NUM_THREADS)
  rebel runtime: 24673.0 us in librbln (0.4% of wall) -- h2v 7289.0(3600) · acquire 5455.0(2400) · v2v_multi 4140.0(400) · return 3488.0(4400) · borrow 2250.0(2000) · v2h 2051.0(1600)

--------------------------------------  -----  --------  ----------------------------------
Signal                                  Count     Bytes  Fix
--------------------------------------  -----  --------  ----------------------------------
host_bounce/copy_d2d_host_bounce         1600  12.59 KB  make contiguous, or v2v engine
runtime/v2v_slow:src_not_on_device        796         -  establish device residency first
dispatch/cpu_fallback                    1600         -  graph mode / native kernel / dtype
--------------------------------------  -----  --------  ----------------------------------
    real device->host (runtime): 0 — host_bounce above was served on host
    cpu_fallback: aten::sub.out 800, aten::mul.out 400, aten::clamp.out 400   (Σ 41084 µs wall)
      why: dtype-not-fp16 1600
      at aten::sub.out: rbln_model_runner.py:1276(_prepare_inputs) <- rbln_model_runner.py(execute_model)
      at aten::mul.out: flash_attention.py:1148(build) <- rbln_model_runner.py:1423(_prepare_inputs)
      at aten::clamp.out: flash_attention.py:1147(build) <- rbln_model_runner.py:1423(_prepare_inputs)
      at host_bounce/copy_d2d_host_bounce: flash_attention.py:1126(build) <- rbln_model_runner.py:1423(_prepare_inputs)
```

How to read G7 in 30 seconds:

- **RED**, driven by `host_bounce` (1600) + `v2v_slow` (796). Both are **host-served** (`real d2h: 0`) — no DMA, so the cost is host-side machinery, not data movement.
- **What**: `cpu_fallback` is the integer metadata math (`sub`/`mul`/`clamp`, all `dtype-not-fp16`); `v2v_slow`/`bounce` is the `positions[idx]` gather. Per 400 steps: 4 fallbacks + ~2 gathers + 4 bounces per step.
- **Where**: the `at …` lines pin it to `flash_attention.py:1126/1147/1148` (the metadata builder) and `rbln_model_runner.py:1276` (logits). All of it is one cluster.
- **Context**: the `! oversubscription` warning flags that these per-op costs may be inflated by CPU contention (worker on 1 core / 8 threads) — check affinity before micro-optimizing. The `rebel runtime` share here is tiny (0.4%) because this region wraps the device forward; the takeaway is still that the runtime is cheap per call, so the lever is the dispatched-op count / dtype, not the runtime.
- **Act**: the (A) candidates (`sub`/`mul`/`clamp`) want fast-path handlers or CPU placement; the gather wants device residency or a host fast-path. (See the per-op cost: `Σ 41084 µs` is the fallback wall over the window — a coarse magnitude, not a per-op absolute; cf. §10.)

(The same step under `VLLM_RBLN_USE_DEVICE_TENSOR=0`, where the metadata stays on CPU as native ops, is **G1 — fully GREEN**. That A/B is exactly how you localize a device-tensor regression.)

## 5. Signal reference

Every signal is a thing the backend did behind a normal op. The table gives the meaning, the intent (why you should care), and the lever.

| Signal (report label) | Side | What happened | Why it matters / intent | Lever |
| --- | --- | --- | --- | --- |
| `host_bounce/copy_d2d_host_bounce` | torch | a non-contiguous device→device `copy_` round-tripped the host | a host round-trip you didn't ask for; the data left the device | make the copy contiguous, or route through the on-device v2v engine |
| `host_bounce/copy_h2d_staging` | torch | a CPU source was staged to host before the h2v push | keep the source on-device, or stage once and reuse | keep src on-device / stage once |
| `host_bounce/copy_h2d_noncontig_dst` | torch | host→device write into a non-contiguous device dst | write into a contiguous buffer first, then h2d | contiguous staging, then h2d |
| `host_bounce/strided_v2v_cpu_fallback` | torch | a strided v2v (cat/index/copy_) fell back to a host CPU op | the device v2v engine rejected the geometry | lower outer_count / fatter contiguous inner block |
| `host_bounce/v2v_batch_to_per_entry` | torch | a batched v2v was rejected to per-entry copies | batch geometry exceeded the per-dst limit | check `kMaxV2VMultiCopies` / batch geometry |
| `runtime/v2v_slow:src_not_on_device` | rebel | a device v2v fell to the host slow path because the source was host-latest | the copy could not stay on device | establish device residency before the copy |
| `runtime/v2v_slow:src_device_only_real_d2h` | rebel | device-only source forced a real d2h (layout unplannable) | a real device→host transfer happened | fix dtype/alignment, keep ≤30 chunks |
| `runtime/v2v_slow:src_synced_host_served` | rebel | src already on host+device; a host memcpy, no transfer | usually benign | (no action needed) |
| `dispatch/cpu_fallback` | torch | an op ran on CPU (fp16-only NPU can't run it) | the per-op tax for non-device dtypes (e.g. int metadata) | graph mode / native rbln kernel / fix dtype — see (A) candidates |
| `dispatch/recompile` | torch | a graph (re)compiled (warm-cache miss) | a cold first compile is expected; **repeated** recompiles in a steady loop are not | stabilize shapes (pad/bucket) for warm-cache reuse, or graph mode |

And the auxiliary readouts (in `dump()`):

- `cpu_fallback_by_op`, `recompile_by_op` — per-op counts (what fell back / recompiled).
- `cpu_fallback_reasons` — the `dtype-not-fp16 / nan-inf / all-scalar` breakdown.
- `cpu_fallback_unaccelerated` (A) — fallback ops with no fast-path handler.
- `real_host_sync_d2h` / `real_host_sync_h2d` — the authoritative count+bytes of **real** physical transfers (vs the copy-path bounce count).
- `trace_by_op` (A WHERE) — op/site → call-site, only when `trace=True`.
- `host_threads` (E), `rebel_runtime` (B), `device_memory` — resource facts (§6).

## 6. Verdict vs. facts — what turns it RED, and what doesn't

The **verdict** (`GREEN`/`AMBER`/`RED`) is driven ONLY by *hidden host overhead you can act on*:

- **RED** — a host bounce fired, or a runtime `v2v_slow` event fired (any reason). A device copy did not stay on device. *Caveat:* `v2v_slow` includes `src_synced_host_served`, which is benign (host memcpy, no transfer), so a RED driven **only** by that reason is usually not actionable (see §11). Use the `real device→host` count to tell a true crossing from a host-served copy.
- **AMBER** — only `cpu_fallback` and/or `recompile` fired (ran on CPU / recompiled, but no host bounce or v2v_slow).
- **GREEN** — none of the above.

These are deliberately **facts, NOT verdict drivers** (they inform, they don't accuse):

| Fact | What it tells you | Why it's not a verdict |
| --- | --- | --- |
| `device_memory` (peak) | rebel `BufferAllocator` physical device footprint incl. cached buffers (reserved, not live) | a number, not an unwanted event |
| `host_threads` (E) — oversubscription | environment amplifier of *all* host overhead | it's your deployment's CPU config, not a bug in any op |
| `rebel_runtime` (B) — time in librbln | splits host cost into runtime vs torch-dispatch | tells you *where* to fix, doesn't itself accuse |
| `real_host_sync_h2d` push | the lazy push that feeds a device graph | a push to feed a graph is *expected*, unlike an unwanted d2h pull |

This separation is the point: a RED is something to fix; a fact is context that helps you decide *how*.

> **What `device_memory` actually counts.** It reads the rebel runtime's `BufferAllocator` gauge (`rbln_prof_get_memory`): the high-water of the **total physical device bytes the allocator holds — regular *and* cached buffers** — decremented only when a buffer is physically freed, *not* when a tensor is released back to the cache. So it is a **reserved / footprint** number that includes idle cached buffers, not a *live-tensor* number; the "live device bytes" phrasing in the source comment is imprecise. It is also distinct from `torch.rbln.memory_stats()`, which is the c10-layer caching-allocator view one level above (separate `allocated` = live / `reserved` / `active` / `cached` stats). If you need the live-vs-reserved split, read `memory_stats()`; explain surfaces only the reserved footprint. So when dt=1 shows a higher `mem peak` than dt=0, that is a *reserved* difference (the device-tensor glue holds more buffers), not "2× more live tensors".

Two facts are worth dwelling on, because they decide your whole strategy:

- **(E) Oversubscription.** If `explain` warns `host oversubscription: N threads on M cores`, the per-op host costs you see **may be inflated by CPU contention** (idle OMP/numba threads spinning on the same cores), rather than caused by any single op. In practice the same op was measured ~5× slower in a worker pinned to one core with 8 threads than in a clean process. **The first lever is then affinity / `OMP_NUM_THREADS` / thread config — not your code.** Chasing per-op micro-optimizations while oversubscribed wastes effort. (It is a heuristic — many idle threads on few cores; it says *may*, not *will*.)
- **(B) Rebel-runtime share.** The `%` is of the **wrapped region's wall**, so read it accordingly. Wrap a *host-bound* region (e.g. just the metadata builder): a small share — say 6% — means the other ~94% of that region is **torch-side dispatch** (dispatcher, boxing, TensorIterator, Python), so the lever is the dispatched-op count, not the runtime. But if you wrap a step that includes the device forward, expect a *tiny* share (e.g. 0.4%) **simply because device compute dominates the wall** — that does NOT mean dispatch is 94%, it means you wrapped the forward. Either way each librbln boundary call is sub-µs, so a *large* rebel share (in a host-bound region) is the only case where the runtime itself is the lever.

## 7. WHERE: `explain(trace=True)`

By default `explain()` captures no call-sites (so it adds nothing). With `trace=True`, the first time each op falls back / recompiles / bounces, its Python call-site is captured and shown under the offending op:

```python
with torch.rbln.explain(trace=True) as p:
    model(x)
print(p.report())
#   cpu_fallback: aten::sub.out 800 ...
#       at aten::sub.out: rbln_model_runner.py:1276(_prepare_inputs) <- ...
```

Use it the moment the report says `where? -> rerun with explain(trace=True)`. It turns "something fell back 800 times" into "*this line* fell back".

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

**(a) Host-served bounce — RED but cheap.** An `int64 → int32` device cast bounces (`copy_d2d_host_bounce`), but the qualifier reads `real device->host (runtime): 0 — served on host`. The copy went *through* host memory but **never crossed the device boundary** (no DMA). It's RED (a host round-trip did happen) but low-cost. The `real_*_sync` counters are how you tell a true device crossing from a host-served copy.

**(b) Integer metadata fallback.** Attention/sampling metadata math (`positions[idx]`, `cs - pidx*len`, `clamp(...)`, `query_start_loc[1:]-1`) on device tensors shows up as `cpu_fallback` with `why: dtype-not-fp16` — the fp16-only NPU can't run integer ops, so they run on CPU. The (A) line lists which lack a fast-path handler = the candidates to accelerate (or to keep on CPU).

**(c) The "GREEN but slow" trap.** Wrapping only the compiled forward (`model_executable`) often shows GREEN — the compiled graph is clean by construction. But the real hidden overhead is usually in the **glue** (metadata builder, sampler, padding) that runs *outside* that narrow wrap. **Wrap the whole step**, not just the forward, or `explain` will honestly report "nothing hidden here" about a region that wasn't where the cost was.

**(d) The oversubscription rabbit hole.** A decode step looked like it had a large per-op host cost. The cause was not any op — the worker was pinned to **1 core with 8 threads**, so every tiny serial host op was preempted (~5× inflation). The (E) warning surfaces this directly; without it you can burn days chasing per-op fixes that affinity config would solve.

**(e) "Is the runtime the problem?"** The device-tensor metadata cost ~600 µs/step. (B) showed only ~65 µs of that was inside librbln (~10%; each boundary call ~0.2 µs — the runtime is fast); the other ~90% was torch-side dispatch + per-op device-buffer alloc. Conclusion: optimizing the runtime would recover ~10% at most; the lever was the dispatched-op count, not the runtime. (B) answers "rebel or torch?" in one line. (An isolated microbench of the same cluster put rebel at ~6–7% of the cluster wall — consistent: small either way.)

**(f) Device-tensor glue push.** With device-tensor mode, metadata is pushed to the device to feed the next graph. That shows as `host->device push (runtime): N pushes` — a **fact, not RED**, because feeding a graph is expected. It's there so you can *see* the glue cost, not to flag it as a bug.

## 10. Pitfalls / how to read it honestly

- **GREEN ≠ fast.** GREEN means "no hidden host overhead in this region." The region can still be device-compute-bound. Use a timer for speed.
- **Narrow wrap hides the cost.** If you wrap only the compiled forward, the glue/sampler overhead is outside the region. Wrap the full step.
- **`cpu_fallback_ns` (the `Σ … µs wall`) is a weak signal.** It is noisy and inflated several-fold under a pinned-core worker; the first fallback in a region is much more expensive than steady ones. Treat it as a coarse A/B over many ops, never a per-op absolute.
- **e2e wall is noisy.** Run-to-run host cost can swing ±10%+ (thermal, scheduling). The *mechanism* signals (which op fell back, the bounce count, the bytes) are far more stable than the wall — trust those, and use interleaved A/B for any wall comparison.
- **A region is a delta.** Each region reports only what happened inside it (memory is a level/high-water, not a delta). A fresh region does not inherit the previous region's counts.

## 11. Practical playbook

| You see… | It means… | Do… |
| --- | --- | --- |
| `! host oversubscription` | host ops inflated by CPU contention | fix affinity / `OMP_NUM_THREADS` first — before any per-op work |
| `cpu_fallback` + `no fast-path handler: <ops>` | those ops run on CPU with the full boxed tax | add a `fast_paths/*.cpp` handler for them, or keep that metadata on CPU |
| `host_bounce/*` with `real d2h > 0` | real bytes crossed the host | make the copy contiguous / route through the v2v engine |
| `host_bounce/*` with `real d2h: 0` | host-served, no device crossing | low priority; the machinery overhead is small |
| `dispatch/recompile` recurring (via `diff`) | shapes aren't stabilizing | pad/bucket to a fixed shape so the warm cache hits |
| `rebel runtime` is a large % | the runtime itself is the cost | optimize the runtime path |
| `rebel runtime` is a small % | torch-side dispatch dominates | reduce dispatched-op count; runtime micro-opt won't help |
| `where? -> trace=True` | you need the source location | rerun with `explain(trace=True)` |
| RED only from `src_synced_host_served` | benign host memcpy | usually no action |

## 12. CI gating

`dump()` / `verdict()` are stable dicts — assert on them:

```python
with torch.rbln.explain() as p:
    run_one_decode_step()
d = p.dump()
assert d["hidden_host_bounce"]["total_count"] == 0, "a host round-trip regressed"
assert not d.get("cpu_fallback_unaccelerated"), f"un-accelerated fallback ops: {d['cpu_fallback_unaccelerated']}"
```

The counters this gate reads are zero-cost (cold-path atomics, lazy reads); the only in-region cost is the (B) runtime timer's sub-µs clock read per librbln boundary call, which is negligible (see §1). So the gate barely perturbs the measured region.

## 13. Scope — what `explain` deliberately does NOT do

- It is not a timing profiler (no per-layer wall, no tok/s) — that's `torch.profiler` / `nsys`.
- It does not surface command-stream counts, device-idle time, or total host-traffic bytes — those are generic utilization metrics that live in the TVM runtime (where they read ~0 here) and would mislead; they are intentionally excluded.
- It does not guess your run lifecycle (one-time vs recurring) — you supply that via `diff`.

The single hidden signal still on the roadmap is a side-effect host-sync on the TVM graph-exec path (a runtime h2v during graph execution that no current counter sees); everything else here is the complete, honest surface today.
