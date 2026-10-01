# `torch.rbln.explain()` — the hidden-overhead explainer

## 1. What it is (and what it is NOT)

`torch.rbln.explain()` answers one question: **"What did my code make the RBLN backend do that I never asked for and cannot see?"**

A normal PyTorch op — `a.copy_(b)`, `torch.cat`, an indexing write, a forward pass — can silently round-trip the host (NPU→CPU→NPU), fall back to a CPU kernel, or recompile a graph. None of that shows up in your Python code; it just makes things slower for reasons that are invisible. `explain()` counts those hidden events, attributes the **cause**, prints a one-line **note** (a *fix* to try), and (opt-in) points at the **where** in your source.

Think of it as `torch._dynamo.explain()` or JAX's `transfer_guard` for the RBLN device — **a hidden-overhead explainer, not a timing profiler.**

> **It is NOT a timing profiler.** It does not tell you how long your forward pass took, which layer is slow, or your tok/s. A region with zero hidden overhead can still be slow (it may be legitimately device-compute-bound). For wall-clock timing use `torch.profiler` / `nsys`. `explain()` only surfaces the *hidden host overhead* — the part you didn't write and can't see.

**Zero-cost when idle (ON==OFF):** counters sit on already-slow points (a host round-trip / fallback branch), never the fast device path, and reads are lazy — so leaving `explain()` in your code costs nothing when you're not profiling. (Inside a region the (B) timer adds one sub-µs clock read per runtime copy call — negligible.)

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

That is the whole loop: **wrap → glance at the `[clean]`/`[overhead]` marker → read the table → act on the Note column**. The marker is a **fact** ("did anything hidden fire"), not a RED/AMBER/GREEN severity grade — cost lives in the table's Bytes/Note, which you judge.

## 3. The API

| Call | Returns | Use |
| --- | --- | --- |
| `explain(with_stack=False)` | `RBLNExplain` (context manager) | wrap a region you place yourself |
| `explain_steady(fn, *, warmup=2, return_cold=False, as_diff=False, with_stack=False)` | `RBLNExplain` / `(cold, warm)` / `RBLNDiff` | auto-place two regions around `fn` to separate first-call cost from steady-state |
| `p.report()` | `str` | the human-readable, marker-first report (print this) |
| `p.verdict()` | `dict` | `clean` flag + `reasons` (facts); for CI gating |
| `p.dump()` | `dict` | every signal as raw numbers; for programmatic checks |
| `p.help(signal=None)` | `str` | full note prose for a signal (or all fired signals) |
| `p.diff(other)` | `RBLNDiff` | compare two regions YOU placed (early vs later) |
| `p.start()` / `p.stop()` | — | manual region boundaries (instead of `with`) |
| `with_stack=True` | — | also capture the Python call-site of each fallback/recompile/bounce (opt-in) |

`profile` / `RBLNProfile` are back-compat aliases of `explain` / `RBLNExplain`. `with_stack=`
matches `torch.profiler.profile`'s parameter name; the older `trace=` still works as an alias.

## 4. Reading the report

An example report (an integer-metadata region) annotated line by line:

```
[overhead: 2 signals]  RBLN EXPLAIN   (region wall 66.430ms | device mem 128.00 MB peak, reserved)   (1)

  ! host oversubscription: 8 threads / 1 core -> host latency may be inflated        (2)
    (tune affinity / OMP_NUM_THREADS)
  runtime: 4.367ms in copy calls (6.6% of region wall)                               (3)
    v2h 1.9ms/3000  h2v 1.6ms/3000  v2v 817us/1000

---------------------  -----  -------  -------------------------------------
Signal                 Count    Bytes  Note                                          (4)
---------------------  -----  -------  -------------------------------------
host_bounce/d2d_copy   1,000  9.77 KB  try: cast/broadcast at the producer           (5)
dispatch/cpu_fallback  2,000       --  try: graph mode, or a supported dtype
---------------------  -----  -------  -------------------------------------

  dispatch/cpu_fallback:  (sum 10.885ms wall)                                        (6)
    aten::sub.out    1,000
    aten::mul.out      500
    aten::clamp.out    500
    why: dtype-not-fp16 2,000                                                        (7)
    candidates (no fast-path handler): sub.out, mul.out, clamp.out                   (8)

  where? -> rerun with explain(with_stack=True)                                      (9)
  (detail: p.help(signal) | raw: p.dump())
```

1. **Marker + header.** `[clean]` (nothing hidden fired) / `[overhead: N signals]` (N distinct hidden signals fired). The count is a **fact, not a severity grade** — there is no RED/AMBER/GREEN, and N counts *how many kinds* of signal fired, not how bad they are. explain does not colour-judge how bad an event is (a bounce of a few bytes is cheap); cost is read from the table's `Bytes` and the runtime line, which *you* judge. `region wall` is the wrapped region's wall-clock (a reference, not the point of the tool — hence the `region` prefix, so the first number on screen doesn't invite a timing read). `device mem … peak, reserved` is the device-memory high-water mark — the caching allocator's **reserved bytes, including cached/idle blocks held for reuse** (a *reserved footprint*, not host process RSS, not just live tensors), summed over the devices this process allocated on (`over N devices` when more than one); see §6.
2. **(E) Host oversubscription** — *context*. The worker may run on far fewer CPU cores than it has threads (here 8 threads on 1 core), so its tiny serial host ops get preempted and per-op latency inflates. This is an **environment amplifier**, not a per-op bug — see §6.
3. **(B) Runtime copy time** — how much of the region was spent *inside* the runtime's copy calls (`v2v`, `v2v_multi`, `v2h`, `h2v`, `v2h_multi`, `h2v_multi`), each shown as `time/calls` (top-4 by time, then `+N more`). A copy runs in order on its stream, so its time includes waiting for the work queued before it (a compiled run, say). The `%` is of the **region wall**, so it reads as "runtime vs torch-side dispatch" only when the region is host-bound. See §6.
4. **The signal table.** Fixed category order (`host_bounce` → `dispatch`); NOT sorted by cost (bytes ≠ cost, and a stable order is what makes A/B comparison of two reports work). Each fired signal: `Count` (how many times), `Bytes` (the volume that went through the host, for the bounce rows), `Note`.
5. **The Note is the fix to try** (`try: …`), a starting point rather than a guaranteed fix; `p.help(signal)` prints the full prose.
6. **Detail blocks, grouped under their parent signal** (in table order, blank-line separated — no flat pile where ownership must be inferred). `dispatch/cpu_fallback:` shows its per-op counts, each with its call-site under `with_stack=True`, and the `(sum … wall)` total fallback wall time (a *weak, noisy* A/B signal — inflated under oversubscription, not a per-op absolute; see §10).
7. **`why`**: the reason each fallback fired — `dtype-not-fp16` (the op's dtype is outside the device dispatch policy), `nan/inf input`, or `all-scalar inputs`.
8. **(A) `candidates (no fast-path handler)`**: of the ops that fell back, which ones have **no CPU fast-path handler** — i.e. your **optimization candidates** (registered ops already bypass the slow boxed path).
9. **`where?`**: rerun with `explain(with_stack=True)` to get the exact Python call-site of each fallback/recompile/bounce.

### A gallery of report shapes

The example above is one shape. Below is the range you'll actually see, from cleanest to richest. Skim them once — recognizing the *shape* of a report is most of reading it.

**(G1) Clean.** Nothing hidden happened. The report names *what it checked* (so `[clean]` is trustworthy, not silent) and reminds you clean ≠ fast. (Note the large region wall: `[clean]` means "no hidden host overhead" — `explain` says nothing about device-compute time.)

```
[clean]  RBLN EXPLAIN   (region wall 5.620s | device mem 2.46 GB peak, reserved)
  checked: host_bounce, cpu_fallback, recompile -- none fired
  note: clean = no hidden host overhead, not "fast"
```

**(G2) A small bounce.** An `int64 → int32` device cast is not a byte copy, so it round-trips the host: the source is read to the host, cast there and written back (the `v2h` and `h2v` on the runtime line). The bytes (8 B) and the cost are tiny. This is exactly why the marker is a fact, not a grade: `[overhead: 1 signal]` flags that *something* fired; the `Bytes` column tells you how much. This is the int-cast in the sampler (`sampled.to(torch.int32)`).

```
[overhead: 1 signal]  RBLN EXPLAIN   (region wall 190.000us | device mem 128.00 MB peak, reserved)

  runtime: 41.000us in copy calls (21.6% of region wall)
    v2h 23us/1  h2v 18us/1

--------------------  -----  -----  -----------------------------------
Signal                Count  Bytes  Note
--------------------  -----  -----  -----------------------------------
host_bounce/d2d_copy      1    8 B  try: cast/broadcast at the producer
--------------------  -----  -----  -----------------------------------

  where? -> rerun with explain(with_stack=True)
  (detail: p.help(signal) | raw: p.dump())
```

**(G3) A decode step in device-tensor mode — the capstone.** Captured with `with_stack=True`, it shows nearly every signal at once: the attention-metadata integer math falling back (`sub`/`mul`/`clamp`, `dtype-not-fp16`), a cast in the metadata builder bouncing through the host (`d2d_copy`), under a worker pinned to one core (oversubscription), with the runtime copy share and the exact source lines:

```
[overhead: 2 signals]  RBLN EXPLAIN   (region wall 5.791s | device mem 5.06 GB peak, reserved)

  ! host oversubscription: 8 threads / 1 core -> host latency may be inflated
    (tune affinity / OMP_NUM_THREADS)
  runtime: 24.673ms in copy calls (0.4% of region wall)
    v2h 9.4ms/3200  h2v 8.3ms/3200  v2v 6.9ms/2400

---------------------  -----  --------  -------------------------------------
Signal                 Count     Bytes  Note
---------------------  -----  --------  -------------------------------------
host_bounce/d2d_copy   1,600  12.59 KB  try: cast/broadcast at the producer
dispatch/cpu_fallback  1,600        --  try: graph mode, or a supported dtype
---------------------  -----  --------  -------------------------------------

  host_bounce/d2d_copy:
    at /.../backends/flash_attention.py:1126(build) <- /.../worker/rbln_model_runner.py:1423(_prepare_inputs)

  dispatch/cpu_fallback:  (sum 41.084ms wall)
    aten::sub.out    800  at /.../worker/rbln_model_runner.py:1276(_prepare_inputs)
    aten::mul.out    400  at /.../backends/flash_attention.py:1148(build)
    aten::clamp.out  400  at /.../backends/flash_attention.py:1147(build)
    why: dtype-not-fp16 1,600
    candidates (no fast-path handler): sub.out, mul.out, clamp.out

  (detail: p.help(signal) | raw: p.dump())
```

How to read G3 in 30 seconds:

- **`[overhead: 2 signals]`**, driven by `host_bounce` (1600) + `cpu_fallback` (1600). The bounce moved 12.59 KB through the host in total — small; the cost is the round-trip machinery per step, not the data volume.
- **What**: `cpu_fallback` is the integer metadata math (`sub`/`mul`/`clamp`, all `dtype-not-fp16`); each fallback reads its inputs to the host and writes its result back, which is most of the `v2h`/`h2v` calls on the runtime line. The bounce is a cast in the metadata builder. Per 400 steps: 4 fallbacks + 4 bounces per step.
- **Where**: the `at …` lines pin it to `flash_attention.py:1126/1147/1148` (the metadata builder) and `rbln_model_runner.py:1276` (logits). All of it is one cluster.
- **Context**: the `! oversubscription` warning flags that these per-op costs may be inflated by CPU contention (worker on 1 core / 8 threads) — check affinity before micro-optimizing. The runtime copy share here is tiny (0.4%) because this region wraps the device forward; the takeaway is still that each copy is cheap, so the lever is the dispatched-op count / dtype, not the runtime.
- **Act**: the (A) candidates (`sub`/`mul`/`clamp`) want fast-path handlers or CPU placement; the cast wants to happen where the data is produced. (See the per-op cost: `sum 41.084ms` is the fallback wall over the window — a coarse magnitude, not a per-op absolute; cf. §10.)

(The same step under `VLLM_RBLN_USE_DEVICE_TENSOR=0`, where the metadata stays on CPU as native ops, is **G1 — fully `[clean]`**. That A/B is exactly how you localize a device-tensor regression.)

## 5. Signal reference

Every signal is a thing the backend did behind a normal op. The table gives the meaning, the intent (why you should care), and the lever.

| Signal (report label) | Side | What happened | Why it matters / intent | Lever |
| --- | --- | --- | --- | --- |
| `host_bounce/d2d_copy` (dump key `copy_d2d_host_bounce`) | torch | a device→device `copy_` that casts dtype, broadcasts, or crosses devices non-contiguously round-tripped the host | a host round-trip you didn't ask for; the data left the device | cast/broadcast where the data is produced, or make a cross-device copy contiguous |
| `host_bounce/h2d_staging` (dump key `copy_h2d_staging`) | torch | a CPU source was converted on the host (dtype, shape or contiguity differs from the dst) before the h2v write | an extra host pass over the source on every copy | prepare the source in the dst's layout once and reuse it, or keep it on-device |
| `host_bounce/h2d_noncontig_dst` (dump key `copy_h2d_noncontig_dst`) | torch | host→device write into a non-contiguous device dst: the dst was read to the host, written there, and copied back | a device→host read you didn't ask for | write into a contiguous buffer first, then h2d |
| `host_bounce/strided_v2v_cpu_fallback` | torch | the runtime refused a strided device copy (cat/index/copy_), so it ran as a host CPU op | the copy left the device | the warning logged with it carries the runtime's error |
| `host_bounce/host_batch_to_per_entry` | torch | a batched h2v/v2h call failed and was replayed entry by entry | lost batching; the batch hit a runtime error | the warning logged with it carries the runtime's error |
| `host_bounce/op_arg_through_host` | torch | the compiler could not lay out an op's input or result as torch holds it (a last axis narrower than 64 elements, for one), so the op ran on the device with that arg encoded or decoded on the host | a host round-trip and a stream sync on every call of the op | a last axis of 64 elements or more keeps the arg on the device |
| `dispatch/cpu_fallback` | torch | an op ran on CPU (fp16-only NPU can't run it); its inputs were read to the host and its results written back | the per-op tax for non-device dtypes (e.g. int metadata) | graph mode / native rbln kernel / fix dtype — see (A) candidates |
| `dispatch/recompile` | torch | a graph (re)compiled (warm-cache miss) | a cold first compile is expected; **repeated** recompiles in a steady loop are not | stabilize shapes (pad/bucket) for warm-cache reuse, or graph mode |

The two `*_batch_to_per_entry` rows are lost batching, not added host round-trips, so they do not count toward `hidden_host_bounce.total_count` or the marker; they show in the table when something else fired, and always in `dump()`.

And the auxiliary readouts (in `dump()`):

- `cpu_fallback_by_op`, `recompile_by_op` — per-op counts (what fell back / recompiled).
- `cpu_fallback_reasons` — the `dtype-not-fp16 / nan-inf / all-scalar` breakdown.
- `cpu_fallback_unaccelerated` (A) — fallback ops with no fast-path handler.
- `trace_by_op` (A WHERE) — op/site → call-site, only when `with_stack=True`.
- `host_threads` (E), `runtime` (B: `total_ns`, `wall_fraction`, `by_primitive` → `{ns, calls}`), `device_memory` (`current_bytes`, `peak_bytes`, `devices`) — context (§6).

## 6. What the marker means — and what stays context (not a grade)

The `[clean]` / `[overhead: N signals]` marker is a **fact**: did any *hidden host overhead* — overhead you issued as a normal op and cannot see — fire in the region, and how many *kinds*? That is all it claims. The count is not a severity grade (there is no RED/AMBER/GREEN): a bounce of a few bytes is cheap, one of a KV block is not, and the tool will not guess which by colour or by N. `[overhead: N signals]` means "there are N kinds of thing in the table to look at"; the table's `Bytes` and the runtime line carry the **cost**.

What flips the marker to `[overhead: N signals]` (each counts as one of the N):

- a **host bounce** fired — a copy that should have stayed on one side round-tripped the host.
- a **`cpu_fallback`** or **`recompile`** fired (ran on CPU / recompiled).

`[clean]` means none of the above fired (it does **not** mean fast — see §10).

These are deliberately **context, NOT marker drivers** (they inform, they don't accuse):

| Context | What it tells you | Why it's context, not a finding |
| --- | --- | --- |
| `device_memory` (peak) | caching-allocator footprint incl. cached blocks (reserved, not live) | a number, not an unwanted event |
| `host_threads` (E) — oversubscription | environment amplifier of *all* host overhead | it's your deployment's CPU config, not a bug in any op |
| `runtime` (B) — time in the runtime's copy calls | splits host cost into runtime copies vs torch-dispatch | tells you *where* to look, doesn't itself accuse |

This separation is the point: an overhead finding is something to **inspect and account for**; context helps you decide *how*.

> **`device_memory`** is the caching allocator's **reserved footprint** (bytes held, including idle cached blocks), a high-water gauge — *not* live-tensor bytes, and it leaves out what another process holds on the NPU. For the live-vs-reserved split use `torch.rbln.memory_stats()`; for the whole NPU use `torch.rbln.mem_get_info()` or `rbln-smi`.

> **A CPU fallback's own traffic** — its inputs read to the host, its results written back — is part of the `cpu_fallback` finding and shows as `v2h`/`h2v` calls on the runtime line, not as a host bounce. Count the fallbacks, not the copies.

Two facts are worth dwelling on, because they decide your whole strategy:

- **(E) Oversubscription.** If `explain` warns `host oversubscription: N threads on M cores`, the per-op host costs you see **may be inflated by CPU contention** (idle OMP/numba threads spinning on the same cores), rather than caused by any single op. In practice the same op was measured ~5× slower in a worker pinned to one core with 8 threads than in a clean process. **The first lever is then affinity / `OMP_NUM_THREADS` / thread config — not your code.** Chasing per-op micro-optimizations while oversubscribed wastes effort. (It is a heuristic — many idle threads on few cores; it says *may*, not *will*.)
- **(B) Runtime copy share.** The `%` is of the **wrapped region's wall**, so read it accordingly. Wrap a *host-bound* region (e.g. just the metadata builder): a small share — say 6% — means the other ~94% of that region is **torch-side dispatch** (dispatcher, boxing, TensorIterator, Python), so the lever is the dispatched-op count, not the runtime. But if you wrap a step that includes the device forward, expect a *tiny* share (e.g. 0.4%) **simply because device compute dominates the wall** — that does NOT mean dispatch is 94%, it means you wrapped the forward. Mind that a copy waits for the work queued before it on its stream: a copy right after a compiled run carries that run's remaining device time, so a large share in a region that runs compiled work may be device time, not copy cost. Only a large share in a host-bound region makes the runtime copies themselves the lever.

## 7. WHERE: `explain(with_stack=True)`

By default `explain()` captures no call-sites (so it adds nothing). With `with_stack=True`, the first time each op falls back / recompiles / bounces, its Python call-site is captured and shown under the offending op:

```python
with torch.rbln.explain(with_stack=True) as p:
    model(x)
print(p.report())
#   dispatch/cpu_fallback:  (sum ... wall)
#     aten::sub.out  800  at .../rbln_model_runner.py:1276(_prepare_inputs)
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
#   ">> persists" + op + call-site -> recurring overhead (the real target)
```

`explain_steady(fn, warmup=2, as_diff=True)` automates exactly this (cold = first call it makes, warm = a later call). Its labels mean literally "first call I made" vs "a later call I made" — valid as one-time-vs-recurring only if (1) `fn` was not already compiled before, and (2) every `fn()` does the same work. Only you can ensure that.

## 9. Cases & corner cases (seen in practice)

**(a) A small bounce — cheap.** An `int64 → int32` device cast bounces (`d2d_copy`) with 8 B. The marker reads `[overhead: 1 signal]` (a host round-trip did happen), but the `Bytes` column says it moved almost nothing — this is exactly why the marker is a fact, not a severity grade. The same row at megabytes per step is a real cost.

**(b) The "`[clean]` but slow" trap.** Wrapping only the compiled forward (`model_executable`) often shows `[clean]` — the compiled graph is clean by construction. But the real hidden overhead is usually in the **glue** (metadata builder, sampler, padding) that runs *outside* that narrow wrap. **Wrap the whole step**, not just the forward, or `explain` will honestly report "nothing hidden here" about a region that wasn't where the cost was.

**(c) The oversubscription rabbit hole.** A decode step looked like it had a large per-op host cost. The cause was not any op — the worker was pinned to **1 core with 8 threads**, so every tiny serial host op was preempted (~5× inflation). The (E) warning surfaces this directly; without it you can burn days chasing per-op fixes that affinity config would solve.

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
| `host_bounce` with large `Bytes` | real volume round-tripped the host | follow the row's fix: cast/broadcast at the producer, contiguous dst |
| `host_bounce` with a few bytes | a small round-trip | low priority unless it recurs every step (check with `diff`) |
| `dispatch/recompile` recurring (via `diff`) | shapes aren't stabilizing | pad/bucket to a fixed shape so the warm cache hits |
| `runtime` is a large % of a host-bound region | the runtime copies themselves are the cost | batch or avoid the copies |
| `runtime` is a small % | torch-side dispatch dominates | reduce dispatched-op count; runtime micro-opt won't help |
| `where? -> with_stack=True` | you need the source location | rerun with `explain(with_stack=True)` |

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

The counters this gate reads are zero-cost (cold-path atomics, lazy reads); the only in-region cost is the (B) timer's sub-µs clock read per runtime copy call, which is negligible (see §1). So the gate barely perturbs the measured region.

## 13. Scope — what `explain` deliberately does NOT do

- It is not a timing profiler (no per-layer wall, no tok/s) — that's `torch.profiler` / `nsys`.
- It does not draw a timeline or surface device-idle time — that is `torch.profiler`'s job. With `RBLN_PROFILER=1` (read when the `torch.profiler` session starts, `ProfilerActivity.PrivateUse1` included), the exported trace carries a row per NPU (`NPU <id>`) with a lane per kind of work (`compute`, `copy`, `host`, `collective`), and an arrow from each `rbln launch` marker on the host thread to the work that launch ran.
- It does not guess your run lifecycle (one-time vs recurring) — you supply that via `diff`.
