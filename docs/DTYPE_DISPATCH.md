# Extending the eager dispatch catalog

torch-rbln's eager path decides per op whether to compile it for the device or run it on the
host (CPU fallback). The dtype catalog is the first gate: fp16 and bf16. Two variables widen it
for one process, for measurement. Both are unset by default, and a default build behaves
exactly as before.

```
TORCH_RBLN_DISPATCH_DTYPES=<names>       admit these dtypes into the eager dispatch catalog
TORCH_RBLN_DISPATCH_STRICT=<names|all>   never take the performance fallbacks for these dtypes
```

Names are torch dtype names or the usual aliases (`float32`/`fp32`/`f32`, `float64`, `float16`,
`bfloat16`, `int32`, `int64`, `int16`, `int8`, `uint8`, `bool`), comma-separated. An unknown name
is an error that names the variable and the token. Set them before `import torch_rbln`: the C++
shim reads them live, the Python gate snapshots them at import.

## What admitting a dtype does

An op whose operands are all of an admitted dtype passes the shim precheck and the Python gate
and reaches the compile path, where it is compiled per (op, shape, dtype) and run with `out=`
bound to the caller's tensor. Every other reason to stay on the host still applies: a tracer or
dispatch mode, mixed dtypes, all-scalar inputs, a storage offset, NaN/Inf in a floating input,
re-entrancy. The NaN/Inf scan covers every floating dtype that can be admitted.

**Mixed devices.** Before the extension, a call with an operand outside the catalog went to the
C++ boxed fallback, which tolerates a CPU operand next to an rbln one and places the result on
the first tensor argument's device. Extended, such a call would reach the compile path and fail.
So when any non-bool operand is of an extension dtype (a bool tensor is a condition or mask) and
the operands span more than one device, the Python gate sends the op to the host and keeps the
first-argument rule. Stock dtypes keep their existing behaviour.

**Strict.** Two fallbacks are performance choices, not safety: the 64-alignment fallback in the
compile path (host pad/depad orchestration dwarfs the compute on small tensors) and the
align-penalty routing in the shim. `TORCH_RBLN_DISPATCH_STRICT` skips both for the listed
dtypes, so the op reaches the compiler whatever its shape. What the compiler does with it is its
decision; it may place a small or unaligned op on the host inside the compiled program.
`SCHEDULE_HOST_OPS_STAT` in the compile log says how an op was split; `torch.rbln.explain()` and
`ops_utils.cpu_fallback_counts()` list what left the device through torch-rbln's own fallbacks.

## Boundaries to know before enabling a dtype

- Integer arithmetic saturates on the device; the CPU wraps. `INT32_MAX + 1`, `INT32_MIN - 1`,
  `65536 * 65536` and `-INT32_MIN` give different answers on the two paths. Code that relies on
  wrap-around (hashes, counters, bit tricks) must not run under an extended integer catalog.
- Floats other than bf16 compute in the device's reduced precision; int64 is narrowed to 32 bits
  on the device, so values outside the int32 range are wrong.
- Every new (op, shape, dtype) costs a compile in every process; eager programs are not cached
  on disk.
- A device launch costs far more than the host op on small tensors; admitting a dtype for
  glue-sized tensors makes them slower even when the result is right.

The default catalog stays fp16/bf16 because of these. The variables exist so the trade-off can
be measured on a workload instead of argued.
