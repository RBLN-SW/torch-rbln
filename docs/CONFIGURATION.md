# Configuration

## Logging

`torch-rbln` provides structured logging via `spdlog` to help diagnose runtime behavior, including CPU fallback operations and device execution traces.

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TORCH_RBLN_LOG_LEVEL` | Controls log verbosity | `WARNING` |
| `TORCH_RBLN_LOG_PATH` | Log file path (debug builds only) | `./torch_rbln.log` |

```bash
export TORCH_RBLN_LOG_LEVEL=INFO
export TORCH_RBLN_LOG_PATH=./torch_rbln.log
```

A log file is always created in debug builds. Its path can be configured via `TORCH_RBLN_LOG_PATH` environment variable.

### Log Levels

| Level | Description | Use Case |
|-------|---------|----------|
| `DEBUG` | Detailed internal states, function entry/exit, parameter values | Deep debugging during development (debug builds only) |
| `INFO` | Runtime information, CPU fallback notifications | General development and troubleshooting |
| `WARNING` (default) | Important warnings that may affect execution | Production monitoring |
| `ERROR` | Errors and critical failures | Error tracking and alerting |

### Debug vs Release Builds

| Feature | Debug Build | Release Build |
|---------|-------------|---------------|
| Minimum log level | `DEBUG` | `INFO` |
| Log file | ✅ Written to `TORCH_RBLN_LOG_PATH` | ❌ Not available |
| Source location | ✅ Included | ❌ Omitted |
| Thread ID | ✅ Included | ❌ Omitted |


## Deploy Mode

Skip NaN/Inf validation checks to reduce runtime overhead in production:

```bash
export TORCH_RBLN_DEPLOY=ON
```

**Runtime behavior.** `TORCH_RBLN_DEPLOY` and `TORCH_RBLN_DEV_DISABLE_OP_CPU_FALLBACK` are read on each operator dispatch, so changes affect subsequent dispatches without restarting the process. Modify these variables only when no RBLN operator dispatch is in progress; concurrent environment mutation and dispatch are unsupported.

## Fallback Control

By default, `torch-rbln` falls back to CPU execution when it encounters unsupported operations or compilation errors.
The `TORCH_RBLN_DISABLE_FALLBACK` environment variable allows you to selectively disable these fallbacks so that errors are raised instead.
The variable is read on every fallback check, so changes take effect immediately without restarting the process. This makes it possible to toggle fallback behavior dynamically at runtime — for example, tightening checks for a specific code path and relaxing them afterward. The error a disabled fallback raises is never caught by another fallback: a compiled graph that raises it does not run on CPU instead.

```bash
export TORCH_RBLN_DISABLE_FALLBACK=compile_error,unsupported_op
```

A fallback takes the place of a graph or op that could not be built, loaded or accepted, never of one that ran and failed: an error from running a graph, an eager op or a copy, such as a failed device job, is raised whatever this variable holds.

The value is a **comma-separated list** of fallback categories to disable:

| Category             | Fallback behavior (default)                              | When disabled                                         |
|----------------------|----------------------------------------------------------|-------------------------------------------------------|
| `compile_error`      | A graph or eager op the compiler cannot build, or a graph that cannot load, runs on CPU | Raises the compilation error directly                 |
| `host_round_trip`    | An input a compiled function writes in place, which the device holds otherwise than torch holds the tensor, goes through the host on every run | Raises when the function is first run, naming the input |
| `non_blocking_copy`  | Non-blocking copy silently falls back to blocking copy   | Raises an error instead of degrading to blocking copy |
| `strided_copy_error` | A batched strided copy the runtime refuses falls back to CPU execution | Raises the underlying error directly                  |
| `unsupported_op`     | Unsupported RBLN ops silently fall back to CPU execution | Raises an error listing the unsupported operator      |
| `all`                | —                                                        | Disables **all** of the above fallbacks               |

**Examples:**

```bash
# Disable all fallbacks (strict mode — every unsupported path raises an error)
export TORCH_RBLN_DISABLE_FALLBACK=all

# Disable only unsupported-op fallback
export TORCH_RBLN_DISABLE_FALLBACK=unsupported_op

# Disable compile-error and non-blocking-copy fallbacks
export TORCH_RBLN_DISABLE_FALLBACK=compile_error,non_blocking_copy

# Enable all fallbacks (default)
unset TORCH_RBLN_DISABLE_FALLBACK
```

## Op CPU Fallback Control

> **⚠️ WARNING: This is a development-only option. Do NOT use in production or deploy environments. Disabling CPU fallback cases can cause silent numerical corruption, hangs, or crashes on unsupported inputs.**

Selectively disables individual CPU fallback checks in eager-mode operator dispatch. When set, the specified checks are skipped and the operation is sent directly to the RBLN device even if it would normally fall back to CPU. A runtime warning is emitted once per process the first time this variable is observed.

```bash
# Disable only the NaN/Inf check (e.g. to benchmark without the scan overhead)
export TORCH_RBLN_DEV_DISABLE_OP_CPU_FALLBACK=nan_inf

# Disable only the trace/debugger check (e.g. to run under pdb without CPU fallback)
export TORCH_RBLN_DEV_DISABLE_OP_CPU_FALLBACK=trace

# Disable only the reentrant check (e.g. for debugging; risks infinite recursion)
export TORCH_RBLN_DEV_DISABLE_OP_CPU_FALLBACK=reentrant

# Disable multiple checks
export TORCH_RBLN_DEV_DISABLE_OP_CPU_FALLBACK=dtype,scalar

# Disable all CPU fallback checks (dangerous)
export TORCH_RBLN_DEV_DISABLE_OP_CPU_FALLBACK=all

# Re-enable all checks (default)
unset TORCH_RBLN_DEV_DISABLE_OP_CPU_FALLBACK
```

The value is a **comma-separated list** of fallback case names to disable:

| Case             | Default behavior (enabled)                                       | When disabled                                                 |
|------------------|------------------------------------------------------------------|---------------------------------------------------------------|
| `dispatch_mode`  | Falls back to CPU when a non-infra `TorchDispatchMode` is active | Skips the check — risks infinite recursion                    |
| `trace`          | Falls back to CPU when a Python trace is active (e.g. pdb, coverage) | Skips the check — compile may run under tracer              |
| `reentrant`      | Falls back to CPU when already inside RBLN compile op (e.g. print/repr, nested op); logs a warning | Skips the check — risks infinite recursion                 |
| `dtype`          | Falls back on unsupported or mismatched tensor dtypes            | Sends such tensors to RBLN — may produce wrong results        |
| `scalar`         | Falls back when all input tensors are 0-dim scalars              | Sends scalar ops to RBLN — may fail in rebel-compiler         |
| `nan_inf`        | Falls back when inputs contain NaN or Inf (non-deploy mode only) | Skips the NaN/Inf scan — invalid values reach the device      |
| `all`            | —                                                                | Disables **all** of the above checks                          |

## Float32 Precision

The NPU computes floats in dlfloat16 (1 sign, 6 exponent and 9 mantissa bits, about three decimal digits). `RBLN_FLOAT32_PRECISION` picks how eager ops over fp32 tensors run, and `rebel.v2.set_float32_precision()` sets it for the rest of the process:

| Value              | Eager fp32 ops                                                               |
|--------------------|-------------------------------------------------------------------------------|
| `device` (default) | Run on the NPU, which computes them in dlfloat16; a warning says so once per process |
| `exact`            | Fall back to CPU, which computes them as fp32                                 |

```bash
export RBLN_FLOAT32_PRECISION=exact
```

```python
from rebel import v2
v2.set_float32_precision("exact")
```

The same precision applies to what `torch.compile` builds for an rbln device: under `exact` the compiler keeps the fp32 ops of a graph on the host. fp16 and bf16 ops run on the NPU under either.

Ops that round to an integer follow the device, not the CPU. `ceil` and `floor` round the input after it is rounded to dlfloat16 (to nearest, an fp16 tie to even and an fp32 tie away from zero, which moves a value by up to half a dlfloat16 step), and `div` with a `rounding_mode` rounds the quotient the NPU computes, which is within two dlfloat16 steps of the exact one. So a value or quotient that close to an integer can come out one away from the CPU's result.

## Compile Cache

An eager op is compiled the first time a process runs it for a profile of its arguments (the shape and dtype of each tensor and every other value), and `torch.compile(backend="rbln")` compiles each graph Dynamo hands over. Both keep the compiled functions on disk, in a directory the processes of a host share: a process takes a function another one made rather than compiling it again. A restarted process traces a graph again to find its function, but takes an eager op of torch or torch-rbln without tracing it. While one process compiles a function, the others that need it wait and take it. The directory only saves compiles: a process that cannot create, read or write it warns once, naming it, and keeps its compiled functions in memory.

| Variable | Description |
|----------|-------------|
| `TORCH_RBLN_COMPILE_CACHE_DIR` | The directory. Set but empty, torch-rbln names no directory: the functions go where `RBLN_COMPILE_CACHE_DIR` says when that is set, and stay in memory otherwise. |
| `RBLN_COMPILE_CACHE_DIR` | The directory of the `rebel.v2` compiles that name none, used when `TORCH_RBLN_COMPILE_CACHE_DIR` is unset. |
| `RBLN_COMPILE_CACHE_DISK` | The bytes of compiled functions a directory keeps, 20 GiB by default. Past them, the process storing a function drops those no process used for the longest. |

With neither directory variable set, the directory is `$XDG_CACHE_HOME/torch_rbln/compile_cache`, or `~/.cache/torch_rbln/compile_cache`. To keep every compiled function in memory only, leave `RBLN_COMPILE_CACHE_DIR` unset and set `TORCH_RBLN_COMPILE_CACHE_DIR` to empty (`export TORCH_RBLN_COMPILE_CACHE_DIR=`). The `cache_dir` option of `torch.compile` names the directory of that graph's compile instead. A function is found again only by the same build of the compiler, so the directory needs no clearing after an upgrade; the functions of the earlier build are the least recently used and go first.

## Debugging Device Errors

Eager ops and graph runs are queued on the current stream and run after the call that queues them returns. An error of the device work they do, such as a failed device job, is raised by a later call that waits for the stream (`torch.rbln.synchronize()`, a copy to the CPU, a later op), far from the op that caused it.

`RBLN_LAUNCH_BLOCKING=1` runs each operation queued on a stream before the call that queues it returns, and waits for its device work, so its error is raised by that call, with the Python traceback of the op that caused it. A process group then runs each collective on the thread that calls it, under `TORCH_RBLN_C10D_ASYNC=1` too, so the collective is done when the call returns; its error is raised by the work's `wait()`, which a collective without `async_op` calls itself. The variable is read once, at the first operation a process queues, so set it before the process starts. The host no longer runs ahead of the device, which slows a model down: use it to find the op, not in production. It is a `rebel.v2` runtime variable, so it does the same for every library running on `rebel.v2`, optimum-rbln and vLLM included.

```bash
RBLN_LAUNCH_BLOCKING=1 python train.py
```

## Eager Op Buckets

An eager op is compiled for the shapes of its tensors, so code whose leading extent changes from call to call, such as a sampler over a changing batch, compiles each op once per extent. `TORCH_RBLN_EAGER_BUCKETS` (default off, read at every dispatch) makes an elementwise op compile once per bucket of extents instead: 1, 2, 3, 4, 6, 8, 12, 16, 24, … (each power of two and the point halfway to the next). A growing extent then compiles about twice per doubling.

```bash
export TORCH_RBLN_EAGER_BUCKETS=1
```

A call is bucketed when all of these hold:

- The op is tagged `pointwise`, not `nondeterministic_seeded`, and has one result.
- Its result has two or more rows along its leading axis.
- Every tensor it reads is contiguous, and every input of the result's rank and leading extent is on the device, with rows of a multiple of 64 elements. The compiler computes an elementwise op on the host, which rounds as the CPU does, when its last axis is not a multiple of 64 elements (for an op over two tensors of one shape, when its element count is not), and on the device, which rounds as the device does, otherwise. A bucket of other rows could move a call from one to the other and change its result. A one-dimensional tensor therefore never buckets.
- It has no `out=`, or its `out=` shares memory with an input (as an in-place op's does) or is not contiguous. Such an `out=` takes a copy of the result, as it does unbucketed. A call that writes any other `out=` runs the function of its own extent.

When the compiler refuses a bucket, or the bucket's function cannot compute by rows, the calls of that bucket run the function of their own extent on the device.

A bucketed call returns the same values as an unbucketed one. Its run computes every row of the bucket, fewer than 1.5 times the call's rows, from whatever memory lies after each input's rows, and drops the rows past the call's. This costs:

- Memory: the result keeps the storage of the bucket's rows, so its `untyped_storage()` is up to 1.5 times its size, and `torch.save` writes those rows as well. The next bucketed op reads the result in place.
- A copy per call: an input whose allocation has no room for the bucket's rows is copied into a tensor of the bucket's rows at every call, and the copy blocks the caller until the work queued before it is done. An input that the previous bucketed op made always has room; any other tensor has room only when the caching allocator's rounding of its allocation left enough.

## Deferred Eager Ops

Each eager op runs as its own compiled function, a device job of its own. `TORCH_RBLN_DEFER_OPS=N` defers them instead: the ops a stream runs gather into a segment of up to N, which runs as one function, compiled from the ops' modules, at the next point that needs it. Unset or `0`, each op runs as it is called. The variable is read at every op.

```bash
export TORCH_RBLN_DEFER_OPS=32
```

A deferred op allocates its results when it is called, so shapes, views and later ops can use them at once; until the segment runs, nothing reads or writes its tensors. It reads and writes the memory its tensors view when it is called, whatever later changes their views (`set_`, `.data =`, `t_()`). The ops deferred on a stream are queued, ahead of the work that needs them, by:

- anything queued on that stream: a copy to, from or within the device (`.cpu()`, `.item()`, `.to()`, `copy_`, and the copy a CPU fallback makes of its inputs), a kernel other than a compiled op (`cat`, `index_select`, a fill), a `torch.compile` graph, or a collective;
- `torch.rbln.synchronize()`, a stream's `synchronize()` and `query()`, an event recorded on the stream, and `torch.rbln.empty_cache()` for every stream of its device;
- `torch.rbln.set_stream()` leaving the stream, and the start and end of a `torch.profiler` session that records the NPU (`RBLN_PROFILER=1`);
- the segment reaching N ops, and an op whose tensor shares memory with a tensor of the segment through another view when either of the two is written.

A segment runs op by op the first time it is seen, and is compiled the second time; later runs of it take the compiled function from a cache, without entering Python. One the compiler refuses or fails on keeps running op by op. The process keeps as many segments as `TORCH_RBLN_COMPILED_OPS` keeps compiled ops, letting go of the least recently run.

The function of a segment writes the tensors its ops were given as `out=` (an in-place op's tensor among them), the inputs its ops update in place, and the other results a tensor still holds when it runs. It does not round the values between its ops to the tensors' dtype, so a result can differ in its last bits from the result of the same ops run one by one.

An op is deferred only when its function does all its work on the device over tensors it takes in place; any other runs as it is called, after the deferred ones. An error of a deferred op is raised no earlier than the call that runs its segment, and the ops after it in the segment do not run; an error of its device work, such as a failed device job, is raised by a later call that waits for the stream, as for any op (see Debugging Device Errors). Two things leave deferring nothing to fuse:

- Outside deploy mode, the NaN/Inf scan reads every op's inputs on the host before it runs, which runs what is deferred at each op: set `TORCH_RBLN_DEPLOY=ON` (or skip the scan with `TORCH_RBLN_DEV_DISABLE_OP_CPU_FALLBACK=nan_inf`).
- `RBLN_LAUNCH_BLOCKING` runs every op as it is called.

Each stream defers its own ops, as CUDA orders work only within a stream: an op on one stream that reads what another wrote needs that order given, by an event or `wait_stream`, which runs the writer's deferred ops when it records the event.

## Device Mapping

Each RBLN device is exactly one NPU: `rbln:N` is the N-th NPU that `RBLN_DEVICES` selects.

### RBLN_DEVICES

Selects which NPUs this process can see, as a comma-separated list of ids — the `CUDA_VISIBLE_DEVICES` analogue. `RBLN_VISIBLE_DEVICES` is an alias of it. Unset means every NPU on the host.

It is owned by the runtime (`rebel-compiler`), not by torch-rbln: with `RBLN_DEVICES=4,5,6,7`, `rbln:0` is NPU `4` as `rbln-smi` numbers it, and `rbln:3` is NPU `7`. An id the host does not have is left out with a warning.

### RBLN_NPUS_PER_DEVICE and RBLN_DEVICE_MAP

These grouped several NPUs into one device on the v1 runtime. torch-rbln on rebel.v2 refuses them rather than ignoring them: with either set, `device_count()` is `0`, `is_available()` is `False`, and the first device use raises an error that names the variable. Unset the variable and select the NPUs with `RBLN_DEVICES`. Spreading one model over several NPUs is the compiler's job (`rebel.v2` with `devices=N`), not a device's; a `torch.compile(backend="rbln")` graph runs on the one NPU of its device, and its `devices` option accepts only `1`.

### When the mapping takes effect

The mapping is resolved in two stages:

| Stage | What happens | Triggered by |
|-------|--------------|--------------|
| **Plan** | The variables above are read and the NPU of each device is listed. No NPU is claimed. | `torch.rbln.is_available()`, `torch.rbln.device_count()`, other queries |
| **Commit** | The NPU of each planned device is opened, which holds a context on it. The mapping is then frozen. | First actual device use — an allocation, `synchronize()`, a collective. Selecting a device with `set_device()` does **not** commit: it is bookkeeping and claims nothing. |

Until commit, editing the variables still changes the mapping. After commit it is fixed for the process lifetime: later changes are ignored rather than rejected, and unsetting a variable does not widen the pool back to every device. This matches `torch.cuda`, which likewise refuses to cache its device count "prior to CUDA initialization, because the number of devices can change due to changes to `CUDA_VISIBLE_DEVICES`".

A launcher may therefore assign `RBLN_DEVICES` after import — including inside a `fork()`ed worker — as long as it does so before the process first uses a device.

### Viewing Devices

```python
import torch
torch.rbln.device_summary()
```

```
[RBLN] Devices:
+--------+-----+
| Device | NPU |
+--------+-----+
| rbln:0 | 4   |
| rbln:1 | 5   |
+--------+-----+
```

### RBLN_DUMMY_DEVICE

Development / compile-only mode for hosts **without an NPU**. When set, the
allocator and memory transfers are served from host memory, so device tensors
can be constructed and a model traced/compiled (e.g. on a CI or compiler box)
even though no hardware is present.

`RBLN_DUMMY_DEVICE` is a **boolean** flag (shared with the rebel runtime, which
validates it at startup): `1/true/t/yes/y/on` enable it, `0/false/f/no/n/off`
and unset disable it, and any other value (e.g. `4`) aborts at startup. There is
one host-backed device per id `RBLN_DEVICES` names, or one without it; no NPU
backs them, so the ids are not checked against the host.

```bash
# 1 host-backed device
export RBLN_DUMMY_DEVICE=1

# 2 host-backed devices, rbln:0 and rbln:1
export RBLN_DUMMY_DEVICE=1 RBLN_DEVICES=0,1
```

- Forced regardless of physical NPU presence; checked before any runtime query,
  so a host with no SDK/driver still works.
- `device_count()` reports the dummy device count; `physical_device_count()` stays `0`.
- **Scope**: device tensor construction, host/device copies, and compile-only
  `torch.compile` (building artifacts). Anything that must run on the NPU raises a
  clear error — a compiled graph, or an eager op on a device dtype (fp16/bf16, and fp32
  under the default float32 precision) — since there is no NPU to run it. Host-side ops
  (e.g. int64, or fp32 under `RBLN_FLOAT32_PRECISION=exact`) fall back to CPU exactly as
  they would on a real device.
  Distributed collectives still require real hardware. Memory-stat APIs
  (`memory_stats`, `memory_allocated`, ...) count the host-backed blocks the caching
  allocator hands out, with the same rounding and caching as on a device, while
  `mem_get_info()` and `mem_get_info_per_chiplet()` raise: there is no device DRAM to report.
- `torch.rbln.is_available()` returns `True` in this mode — treat it as a
  development flag, not real hardware availability.

## rebel.v2 ABI Check

`torch-rbln` compiles against the runtime headers of the rebel-compiler tree named by
`REBEL_HOME` (`rebel/v2/include/rebel/v2/**/*.h`) and runs on the `librebel_v2_rt.so` that the
`rebel.v2` package maps when it is imported. Both have to come from the same headers. Their ABI id is a
SHA-256 over every header's relative path and SHA-256, computed by the runtime's own
`rebel/v2/cmake/RebelV2Abi.cmake`:

- the runtime exports the id of the headers it was built with as `rebel_v2_abi_id()`;
- the `torch-rbln` build runs the same script over the headers it compiles against and records
  the id in the generated `torch_rbln/_internal/_abi_snapshot.py`. Editing, adding or removing a
  header regenerates it on the next build.

`import torch_rbln` compares the two ids before it loads its own native libraries. When they
differ, the import fails with an `RBLN ABI mismatch` message naming the runtime, both ids and the
include directory the build used. Rebuild `torch-rbln` with `REBEL_HOME` set to the tree the
runtime was built from, or run it with the `rebel.v2` package of the tree it was built against.

Cases that leave no verdict to reach warn and continue instead:

| case | why it cannot decide |
|------|----------------------|
| `torch-rbln` recorded no ABI id | `_abi_snapshot.py` is missing from this install |
| no handle can be taken on the mapped runtime | its symbols cannot be read |
| the runtime exports no `rebel_v2_abi_id` | it cannot say which headers it was built with |

Run `python -m torch_rbln.diagnose` to see where `rbln` imports from, the runtime it maps, both
ids and the verdict for the current environment.

### TORCH_RBLN_SKIP_ABI_CHECK

Skips the check entirely.

```bash
export TORCH_RBLN_SKIP_ABI_CHECK=1   # accepts 1 / ON / TRUE / YES
```

This is an escape hatch for unblocking a machine while a matching build is made. It
suppresses the diagnosis, not the incompatibility: the mismatch it hides is what would
otherwise surface as an `undefined symbol` import crash or as corruption inside the
runtime.
