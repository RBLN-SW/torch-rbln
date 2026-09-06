# Extending the eager dispatch catalog: what happens when a dtype is enabled

torch-rbln's eager path decides per op whether to run on the device (compile the op with
rebel and run it) or on the host (CPU fallback). The dtype catalog is the first gate. This
note follows one op — `a + b` on two `int32` tensors — through every layer when `int32` is
enabled with the environment variables below, so that a compiler/runtime engineer can see
which piece has to become cheap for the dtype to be worth enabling by default.

```
TORCH_RBLN_DISPATCH_DTYPES=int32         # admit int32 into the eager dispatch catalog
TORCH_RBLN_DISPATCH_STRICT=int32         # optional: never take the performance fallbacks for it
```

## 1. Catalog (C++, `c10/rbln/RBLNSupportedDtypes.h`, `RBLNFunctions.cpp`)

`kDispatchDtypes` is `{fp16, bf16}`. `is_dispatch_dtype_rt(s)` returns true for the catalog
**or** for a dtype named in `TORCH_RBLN_DISPATCH_DTYPES`. The env string is compared on every
call and re-parsed only when it changes (a short strcmp on the hot path; no latching).
`_C._dispatch_dtypes()` returns the union, `_C._dispatch_strict_dtypes()` the strict set.

## 2. Shim precheck (C++, `DispatchShim.cpp::quick_fallback_check`)

Ops registered on the C++ shim (`_register_cpp_shim`: `add.out`, `mul.out`, `where.self_out`,
comparisons, `_softmax.out` on dev, ...) hit this before any Python. The dtype check calls
`is_dispatch_dtype_rt`; without the env an int32 operand returns reason 1 (dtype) and the op
goes straight to `cpu_fallback_rbln` (the boxed fallback) with no Python involved. With
`int32` in the catalog the check passes and the shim calls the Python wrapper (`add_rbln`),
or serves a warm-cache hit from C++ once the op has compiled for this shape.

Also here: the NaN/Inf scan (non-deploy only) covers every floating dtype the policy admits
(fp16, bf16, and float32/float64 when extended); integer dtypes cannot hold NaN. Without this
a warm-cache hit would run the device kernel on NaN/Inf input without the Python scan ever
seeing it.

## 3. Python gate (`torch_rbln/_internal/ops_utils.py::is_cpu_fallback_cases`)

The generated wrapper (`register_ops.py`, codegen output) calls this with the args. It
returns "fallback" for, in order: a tracer (`sys.gettrace`), an active TorchDispatchMode,
dtype outside `SupportedDtypes.dispatch` **or mixed dtypes**, **an extension-dtype operand on a
different device than the others** (new, closed unless the catalog is extended: before the
extension such a call went to the boxed fallback, which tolerates mixed devices and puts the
result on the first tensor argument's device — vllm-rbln's speculative-decoding glue relies on
that; the compile path needs every operand on rbln; stock dtypes keep their existing rules),
all-scalar inputs, a contiguous tensor with a storage offset, NaN/Inf, re-entrancy.

`SupportedDtypes.dispatch` / `.strict` are **snapshots taken at import** of the C++ getters,
so the env must be set before `import torch_rbln`.

## 4. Compile path (`ops_utils.py::compile_and_run_view_aware`)

Performance fallbacks live here, not in the gate:

* **64-alignment**: a last dim not divisible by 64 makes the compiler wrap the device function
  in host pad/depad ops; for small tensors that orchestration dwarfs the compute, so
  the op is sent to `cpu_fallback_path` instead. `TORCH_RBLN_DISPATCH_STRICT=int32` disables
  this for int32 operands: the op is compiled and run as a first-class device op, pad
  penalty included. (The shim's own align-penalty routing is compiled out in v0.4.0.)
* `div` with `rounding_mode` on catalog dtypes: known-wrong device kernel, always host.

Otherwise the op is compiled through `compile_rbln_cached` (per (op, shape, dtype)
signature, in-memory cache only — every new shape costs a rebel compile, in every
process) and run with `out=` bound to the caller's tensor.

## 5. Compiler (rebel)

The eager compile passes `input_attrs` for each rbln tensor (device="rbln", key=data_ptr).
* `int32` is device-native (`isDeviceSupportedDtype`), so the input's physical dtype is int32
  and no cast is inserted; `int64` is not — it is computed as int32 on the device and
  `int64 -> int32` casts wrap the graph; floats that are not bf16 are computed as DLFloat16.
* Unaligned last dims get `contrib_aligned_pad`/depad **host** ops around the device function;
  `SCHEDULE_HOST_OPS_STAT,enabled,bailed,device_ops,host_ops,...` in the compile log shows how
  the op was split (`1,0,1,0` = one device op, no host ops).
* Some int ops are placed on the host regardless of alignment (reductions such as `sum`,
  int gathers), and `int64` graphs run entirely on the host.

## 6. Runtime (rebel vmem)

Each rbln tensor is a vmem entry with a user view (host) and a physical view (device) and a
per-entry sync state. On a graph call the runtime applies the compiled input config to the
entry (`SetDeviceAllocConfiguration`): a matching config hash or byte layout reuses the
device view in place; otherwise the entry is synced to the host, reallocated and uploaded.
For int32 the user and physical dtypes are equal, so a device-latest int32 tensor produced
by one op is consumed by the next with no transfer. Every call still pays a fixed cost of I/O
address patching and input re-validation when the input address differs from the last call.

## 7. What it costs today (relative; raw figures stay internal)

| case | host path vs device path |
|---|---|
| `i + 1`, int32 `[3, 200064]`, device-latest | device path is tens of times cheaper: the host path pays a D2H of the tensor plus the CPU op, the device path pays nothing but the launch |
| `g + 1`, int32 `[4, 3]`, host-latest | device path is tens of times *more* expensive: a launch costs far more than the host op; even a fallback that goes through Python costs several times the direct host op |
| `i.sum(-1)`, int32 `[3, 200064]` | still leaves the device: the compiler places the reduction on the host |
| first call of any new (op, shape, dtype) | a rebel compile, per process |

So enabling int32 pays off for large, aligned tensors that already live on the device and
are consumed by eager code, and costs for small glue (`[B, 3]`-shaped speculative-decoding
tensors, where the Python round trip alone makes them several times slower even when they
end up on the host). In that workload the second dominates, which is why the default
catalog stays fp16/bf16.

## 8. What would make a dtype worth enabling by default

1. Pad-free execution of small / unaligned shapes (compiler), or a shim-side "tiny op stays
   on the host" rule that does not go through Python.
2. Lower launch latency for eager device ops, or fusion of eager chains.
3. A persistent (on-disk) cache for eager-compiled programs so per-shape compiles do not
   recur per process.
4. Device placement of int reductions / gathers in the compiler.

`TORCH_RBLN_DISPATCH_STRICT` exists so each of these can be measured as "device path cost
for dtype X" rather than inferred: with it set, `torch.rbln.explain()` and
`ops_utils.cpu_fallback_counts()` list exactly what still leaves the device and why.
