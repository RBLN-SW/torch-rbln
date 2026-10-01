#pragma once

#include <pybind11/pybind11.h>
#include <torch/csrc/utils/pybind.h>
#include <torch_rbln/csrc/rbln/OpFunction.h>

#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace torch_rbln::shim {

// Install a C++ boxed dispatch shim for `op_name` on PrivateUse1, and register
// `py_fn` as the Python impl invoked on the non-fallback path.
//
// The shim runs a cheap pre-check in C++ (dtype, scalar-all, contig+offset). On
// pre-check fail it calls into `at::native::rbln::cpu_fallback_rbln` directly —
// the Python layer is never entered for that call. On pre-check pass, if a
// matching warm-cache entry exists the shim runs its OpFunction from C++. Only
// on warm-cache miss does the shim unbox
// the jit stack, call `py_fn` respecting the op schema's kwarg-only markers,
// and rebox the return onto the stack.
//
// `skip_dtype_args` lists positional argument indices whose dtype should not be
// checked against float16. Used for ops with typed non-fp16 inputs (e.g.
// `aten::where.self_out`'s cond at index 0 is bool). These args are still
// skipped from the all-scalar check too.
//
// Called from generated `register_ops.py` at module-init time in place of the
// usual `aten_impl.impl(...)` Python registration. The registered C++ library
// is kept alive for the process lifetime.
void register_cpp_shim(
    const std::string& op_name,
    pybind11::object py_fn,
    const std::vector<size_t>& skip_dtype_args = {});

// DIAG: dispatch path counters/timing populated inside generic_shim_boxed.
// Returns (n_total, n_fallback, n_warm_hit, n_miss, ns_warm_hit, ns_miss).
//   n_total      - every shim invocation
//   n_fallback   - quick_fallback_check=true → cpu_fallback_rbln
//   n_warm_hit   - warm-cache hit fast path (op function run from C++)
//   n_miss       - cold/miss path (Python compile via py_fn)
//   ns_warm_hit  - cumulative ns inside warm-cache hit path (~all in the run)
//   ns_miss      - cumulative ns inside miss path (Python compile + first run)
//   ns_fallback  - cumulative ns inside cpu_fallback_rbln (the COST of fallbacks,
//                  so the report can separate many-cheap from few-expensive)
std::tuple<uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t> diag_dump_dispatch_paths();
void diag_reset_dispatch_paths();

// DIAG/PROFILER: per-op CPU-fallback attribution (op_name -> fallback count),
// non-zero entries only. Recorded on the already-slow fallback branch only (one
// relaxed atomic via a pointer cached on the op's ShimEntry; warm/fast path
// untouched), read lazily under registry_mutex. Lets torch.rbln.profile() turn
// the aggregate cpu_fallback count into "which ops fell back".
std::vector<std::pair<std::string, uint64_t>> diag_dump_fallback_by_op();
void diag_reset_fallback_by_op();

// DIAG/PROFILER: per-op warm-cache MISS (recompile) attribution (op_name ->
// count), non-zero only. Same low-overhead scheme as the fallback counter
// (one relaxed atomic on the already-slow miss branch). Lets the profiler turn
// the aggregate recompile_miss count into "which op-shapes keep recompiling".
std::vector<std::pair<std::string, uint64_t>> diag_dump_recompile_by_op();
void diag_reset_recompile_by_op();

// DIAG/PROFILER: cpu_fallback reason histogram, counts for reason codes 1..3:
// [dtype-not-fp16, nan/inf input, all-scalar]. Recorded on the fallback branch
// only (reason already computed by quick_fallback_check) -> ON==OFF preserved.
std::vector<uint64_t> diag_dump_fallback_reasons();
void diag_reset_fallback_reasons();

// DIAG/PROFILER (A) WHERE: opt-in Python call-site capture. OFF by default
// (default explain() pays nothing). When enabled, the slow fallback/miss branch
// captures the user's call-site ONCE per op (deduped, GIL-safe). dump returns
// (op_name -> "file:line(func) <- ..."); reset clears between regions.
void diag_set_trace_enabled(bool on);
std::vector<std::pair<std::string, std::string>> diag_dump_trace_by_op();
void diag_reset_trace_by_op();

// DIAG: per-segment timers inside the warm-cache hit path. Returns
// (n_hits, ns_lookup, ns_run, ns_finalize); ns_run covers binding the tensors
// and queueing the run. Counts/accumulates only when the hit path returns
// true, so per-segment averages reflect successful warm-path calls only.
std::tuple<uint64_t, uint64_t, uint64_t, uint64_t> diag_dump_warm_segments();
void diag_reset_warm_segments();

// Called by the Python wrapper after a successful miss-path compile and run to
// install a warm-cache entry keyed by the CacheKey that the shim built on the
// way in (stored in a thread-local so Python doesn't need to re-build it).
//
// `inputs` are the tensors `function` ran over; each must be one of the call's
// own tensor arguments, whose positions the entry keeps. Returns true if an
// install actually happened. Safe to call when no pending context exists —
// returns false.
bool install_warmcache_from_pending(std::shared_ptr<OpFunction> function, const std::vector<at::Tensor>& inputs);

} // namespace torch_rbln::shim
