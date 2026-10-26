# Eager dispatch: dtypes and fallbacks

torch-rbln's eager path decides per op whether to compile it for the device or
run it on the host (CPU fallback). The direction is to hand ops to the compiler
and runtime and let them optimize, rather than second-guess them in torch-rbln.
torch-rbln-side performance fallbacks are being removed in steps; what remains
of them is transitional. Fallbacks that protect correctness stay.

## What reaches the device

An op whose operands are all of a dispatch dtype reaches the compile path, where
it is compiled per (op, shape, dtype) and run with `out=` bound to the caller's
tensor. The dispatch dtypes are fp16, bf16 and fp32. The device may cast fp32
ops to a custom float format, so an fp32 op can run in reduced precision.

Every new (op, shape, dtype) costs a compile in every process, and eager
programs are not cached on disk.

## What stays on the host

- Ops registered directly to the CPU fallback, which have no device
  implementation.
- Safety and correctness checks: a tracer or dispatch mode, mixed dtypes across
  operands, all-scalar inputs, a storage offset, NaN/Inf in a floating input,
  re-entrancy. The NaN/Inf scan covers every floating dtype that can be
  dispatched.
- Mixed devices for extension dtypes (below).
- Unaligned last dims, for now (below).

`torch.rbln.explain()` and `ops_utils.cpu_fallback_counts()` list what left the
device through torch-rbln's own fallbacks.

## Alignment fallback

An op whose tensor arguments have a last dim that is not a multiple of 64 runs
on the host. On the device such a shape is costly today:

- the compiler wraps the op in host pad/depad steps, which cost more than the
  compute on small tensors;
- a loop whose shapes keep changing, such as a decode loop with a growing
  attention length, compiles once per shape;
- some unaligned ops do not yet run correctly.

This fallback is transitional and goes away once the compiler handles these
shapes.

To hand unaligned ops to the compiler anyway, set
`TORCH_RBLN_DEV_DISABLE_OP_CPU_FALLBACK=align` (development only; see
`docs/CONFIGURATION.md`). What the compiler does with them is its decision:
`SCHEDULE_HOST_OPS_STAT` in the compile log says how an op was split.

## Extra dtypes

```text
TORCH_RBLN_DISPATCH_DTYPES=<names>    admit more dtypes into the dispatch catalog
```

Names are torch dtype names or the usual aliases, comma-separated: `float64`,
`int32`, `int64`, `int16`, `int8`, `uint8`, `bool` (the dispatch dtypes are
accepted and already admitted). An unknown name is an error that names the
variable and the token. Set it before `import torch_rbln`: the C++ shim reads it
live, the Python gate snapshots it at import.

**Mixed devices.** Before the extension, a call with an operand outside the
catalog went to the C++ boxed fallback, which tolerates a CPU operand next to an
rbln one and places the result on the first tensor argument's device. Extended,
such a call would reach the compile path and fail. So when any non-bool operand
is of an extension dtype (a bool tensor is a condition or mask) and the operands
span more than one device, the Python gate sends the op to the host and keeps
the first-argument rule. Dispatch dtypes keep their existing behavior.

## Boundaries to know before admitting a dtype

- Integer arithmetic saturates on the device; the CPU wraps. `INT32_MAX + 1`,
  `INT32_MIN - 1`, `65536 * 65536` and `-INT32_MIN` give different answers on
  the two paths. Code that relies on wrap-around (hashes, counters, bit tricks)
  must not run under an extended integer catalog.
- float64 computes in the device's reduced precision, like fp32; int64 is
  narrowed to 32 bits on the device, so values outside the int32 range are
  wrong.
- A device launch costs far more than the host op on small tensors; admitting a
  dtype for glue-sized tensors makes them slower even when the result is right.

These are why the other dtypes are not in the default catalog yet.
