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
ops to a custom float format, so an fp32 op can run in reduced precision and
range: very large magnitudes saturate and very small ones flush to zero.

Every new (op, shape, dtype) costs a compile in every process, and eager
programs are not cached on disk.

## What stays on the host

- Ops registered directly to the CPU fallback, which have no device
  implementation.
- Safety and correctness checks: a tracer or dispatch mode, mixed dtypes across
  operands, all-scalar inputs, a storage offset, NaN/Inf in a floating input,
  re-entrancy. The NaN/Inf scan covers every floating dtype that can be
  dispatched.
- Unaligned last dims, for now (below).

`torch.rbln.explain()` lists, per op, every op-level CPU fallback torch-rbln
takes, with their wall time: the C++ shim, the boxed fallback for ops without a
device kernel, the Python path, and a compile that fails all run their fallback
inside one scope (`c10::rbln::CpuFallbackScope`), which counts, logs and times
it. An eager op whose compile fails is listed under that op.
A strided copy that bounces through the host and a fallback inside the runtime
are not ops falling back; `explain()` reports them separately, as a host bounce
and as a runtime signal.

## Alignment fallback

The common eager Python path falls back to CPU for tensor arguments whose last
dim is not a multiple of 64. This check does not cover TensorList inputs such as
`cat`/`stack`, separate C++ kernels, or graph mode. It also skips ops whose
device inputs the compiler pads itself: elementwise arithmetic and unary math,
comparisons, `maximum`/`minimum`, `max`/`min`/`mean`, `softmax` and the matmul
family (the list is `_ALIGN_EXEMPT_OPS` in `ops_utils.py`). For other ops such a
shape is costly on the device today:

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

Names are torch dtype names, comma-separated, as in `torch.<name>`: `float64`,
`int32`, `int64`, `int16`, `int8`, `uint8`, `bool`, or a torch alias such as
`double`, `int` or `long` (the dispatch dtypes are accepted and already
admitted). It is read once, at `import torch_rbln`, and the same set then
governs both the Python gate and the C++ shim; changing it later has no effect.
A name that is not a torch dtype, or a dtype outside that list, fails the
import with an error that names the variable and the token.

## Boundaries to know before admitting a dtype

- Integer arithmetic saturates on the device; the CPU wraps. `INT32_MAX + 1`,
  `INT32_MIN - 1`, `65536 * 65536` and `-INT32_MIN` give different answers on
  the two paths. Code that relies on wrap-around (hashes, counters, bit tricks)
  must not run under an extended integer catalog.
- float64 computes in the device's reduced precision, like fp32; int64 is
  narrowed to 32 bits on the device, so values outside the int32 range are
  wrong.
- An admitted dtype takes the compile path, which places the result of a
  mixed-device call such as `where(cpu_mask, a, b)` on the device. Before, the
  C++ boxed fallback put it on the first argument's device; code that relied on
  that must move its operands explicitly.
- A device launch costs far more than the host op on small tensors; admitting a
  dtype for glue-sized tensors makes them slower even when the result is right.

These are why the other dtypes are not in the default catalog yet.
