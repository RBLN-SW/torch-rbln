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

## Fallback Control

By default, `torch-rbln` falls back to CPU execution when it encounters unsupported operations or compilation errors.
The `TORCH_RBLN_DISABLE_FALLBACK` environment variable allows you to selectively disable these fallbacks so that errors are raised instead.
The variable is read on every fallback check, so changes take effect immediately without restarting the process. This makes it possible to toggle fallback behavior dynamically at runtime — for example, tightening checks for a specific code path and relaxing them afterward.

```bash
export TORCH_RBLN_DISABLE_FALLBACK=compile_error,unsupported_op
```

The value is a **comma-separated list** of fallback categories to disable:

| Category             | Fallback behavior (default)                              | When disabled                                         |
|----------------------|----------------------------------------------------------|-------------------------------------------------------|
| `compile_error`      | `torch.compile` failures fall back to CPU execution      | Raises the compilation error directly                 |
| `non_blocking_copy`  | Non-blocking copy silently falls back to blocking copy   | Raises an error instead of degrading to blocking copy |
| `strided_copy_error` | Batched strided copy failures fall back to CPU execution | Raises the underlying error directly                  |
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

Selectively disables individual CPU fallback checks in eager-mode operator dispatch. When set, the specified checks are skipped and the operation is sent directly to the RBLN device even if it would normally fall back to CPU. A runtime warning is emitted every time this variable is detected.

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
| `storage_offset` | Falls back when a contiguous tensor has `storage_offset != 0`    | Sends offset tensors to RBLN — may read wrong data            |
| `nan_inf`        | Falls back when inputs contain NaN or Inf (non-deploy mode only) | Skips the NaN/Inf scan — invalid values reach the device      |
| `all`            | —                                                                | Disables **all** of the above checks                          |

## Device Mapping

By default, each physical NPU is mapped 1:1 to a logical device (**Direct Mapping**). To group multiple physical NPUs into a single logical device for RSD (Rebellions Scalable Design), use one of the following environment variables.

### RBLN_NPUS_PER_DEVICE

Groups physical NPUs uniformly. Must be one of: `1`, `2`, `4`, `8`, `16`, `32`.

```bash
export RBLN_NPUS_PER_DEVICE=2
```

**Examples** (4 physical devices):
- `RBLN_NPUS_PER_DEVICE=2` → `rbln:0` = NPUs [0, 1], `rbln:1` = NPUs [2, 3]
- `RBLN_NPUS_PER_DEVICE=4` → `rbln:0` = NPUs [0, 1, 2, 3]

With 6 physical devices and `RBLN_NPUS_PER_DEVICE=4`:
- `rbln:0` = NPUs [0, 1, 2, 3]; NPUs [4, 5] remain unused (warning displayed)

### RBLN_DEVICE_MAP

Explicit mapping for fine-grained control. Each group must contain a supported size (`1`, `2`, `4`, `8`, `16`, `32`).

```bash
export RBLN_DEVICE_MAP="[0,1],[2,3,4,5]"
```

This maps `rbln:0` → NPUs [0, 1] and `rbln:1` → NPUs [2, 3, 4, 5].

### Priority

`RBLN_DEVICE_MAP` > `RBLN_NPUS_PER_DEVICE` > default (1:1)

### Viewing Device Topology

```python
import torch
torch.rbln.device_summary()
```

```
[RBLN] Device Topology Initialized:
+-------------------+-------------------+----------------------+
| Logical Device    | Physical NPU IDs  | Status               |
+-------------------+-------------------+----------------------+
| rbln:0            | [ 0, 1 ]          | Active (Aggregated)  |
| rbln:1            | [ 2, 3 ]          | Active (Aggregated)  |
+-------------------+-------------------+----------------------+
```

### RBLN_DUMMY_DEVICE

Development / compile-only mode for hosts **without an NPU**. When set, the
allocator and memory transfers are served from host memory, so device tensors
can be constructed and a model traced/compiled (e.g. on a CI or compiler box)
even though no hardware is present.

`RBLN_DUMMY_DEVICE` is a **boolean** flag (shared with the rebel runtime, which
validates it at startup): `1/true/t/yes/y/on` enable it, `0/false/f/no/n/off`
and unset disable it, and any other value (e.g. `4`) aborts at startup. The
host-backed logical-device layout comes from `RBLN_DEVICE_MAP` (its group count
and sizes). Without `RBLN_DEVICE_MAP`, `RBLN_NPUS_PER_DEVICE=N` yields a single
logical device of size N (TP=N); with neither set, one device (TP=1).

```bash
# 1 host-backed logical device
export RBLN_DUMMY_DEVICE=1

# Preserve an RSD/TP layout for compilation (group count + sizes honored)
export RBLN_DUMMY_DEVICE=1 RBLN_DEVICE_MAP="[0,1],[2,3]"   # 2 logical devices, TP=2
```

- Forced regardless of physical NPU presence; checked before any runtime query,
  so a host with no SDK/driver still works.
- `device_count()` reports the dummy logical device count (the `RBLN_DEVICE_MAP`
  group count when set, otherwise 1); `physical_device_count()` stays `0`.
  `RBLN_DEVICE_MAP` is validated as far as is possible without hardware — group
  sizes against the allowed sizes (1, 2, 4, 8, 16, 32) and duplicate physical ids
  are rejected — but physical-id ranges are **not** checked, so a map valid under
  dummy is not guaranteed to be valid on a specific machine.
- **Scope**: device tensor construction, host/device copies, and `torch.compile`
  tracing/compilation. Out of scope (still require an NPU and will fail): actual
  kernel / compiled-graph **execution**, distributed collectives, and
  runtime-only introspection such as `get_memory_info`.
- `torch.rbln.is_available()` returns `True` in this mode — treat it as a
  development flag, not real hardware availability.

## Tensor Parallel Configuration

The following environment variables control tensor parallel behavior for `torch.compile` operations and eager mode ops.

### TORCH_RBLN_USE_TP_FAILOVER

Enables automatic tensor parallel failover. When a RuntimeError occurs during execution with `num_devices > 1`, the system automatically retries with `num_devices=1` on the root NPU of the device group.

This is useful for models that don't support tensor parallelism, allowing them to run on a single NPU within an aggregated device group without manual intervention.

```bash
export TORCH_RBLN_USE_TP_FAILOVER=ON   # enable
export TORCH_RBLN_USE_TP_FAILOVER=OFF  # disable (default: OFF)
```

**Behavior:**
- When set to ON and a RuntimeError occurs with `num_devices > 1`:
  1. The system logs a warning message indicating the failover attempt
  2. The model is recompiled with `num_devices=1`
  3. Execution continues on the root NPU of the device group
- When set to OFF or unset (default), RuntimeErrors are propagated as-is

**Example scenario:**
With `RBLN_NPUS_PER_DEVICE=4` (4 NPUs per logical device):
- Initial compilation attempts `num_devices=4`
- If the model doesn't support TP, a RuntimeError occurs
- With failover enabled, the system retries with `num_devices=1` on NPU 0

### TORCH_RBLN_USE_DEVICE_TP

Controls whether eager mode operations use the device group's `num_devices` instead of `num_devices=1`.

By default, eager mode ops (operations outside of `torch.compile`) use `num_devices=1`. When this environment variable is set to ON, eager mode ops will follow the logical device size defined by `RBLN_NPUS_PER_DEVICE` or `RBLN_DEVICE_MAP`, matching the behavior of `torch.compile` operations.

```bash
export TORCH_RBLN_USE_DEVICE_TP=ON   # use device group num_devices
export TORCH_RBLN_USE_DEVICE_TP=OFF  # use num_devices=1 for eager ops (default: OFF)
```

**Behavior:**
- When set to ON: Eager mode ops use the device group's `num_devices` (e.g., `num_devices=4` with `RBLN_NPUS_PER_DEVICE=4`)
- When set to OFF or unset (default): Eager mode ops use `num_devices=1`

**Use case:**
This is useful when you want consistent tensor parallel behavior across both eager and compiled operations, particularly in mixed execution scenarios.
