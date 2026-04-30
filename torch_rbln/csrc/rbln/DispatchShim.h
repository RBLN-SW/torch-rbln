#pragma once

#include <pybind11/pybind11.h>
#include <torch/csrc/utils/pybind.h>

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

namespace torch_rbln::shim {

// Install a C++ boxed dispatch shim for `op_name` on PrivateUse1, and register
// `py_fn` as the Python impl invoked on the non-fallback path.
//
// The shim runs a cheap pre-check in C++ (dtype, scalar-all, contig+offset). On
// pre-check fail it calls into `at::native::rbln::cpu_fallback_rbln` directly —
// the Python layer is never entered for that call. On pre-check pass, if a
// matching warm-cache entry exists the shim drives rebel's PyRblnSyncRuntime
// directly from C++ (no pybind). Only on warm-cache miss does the shim unbox
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
//   n_warm_hit   - warm-cache hit fast path (rebel runtime driven from C++)
//   n_miss       - cold/miss path (Python compile via py_fn)
//   ns_warm_hit  - cumulative ns inside warm-cache hit path (~all in rebel run)
//   ns_miss      - cumulative ns inside miss path (Python compile + first run)
std::tuple<uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t>
diag_dump_dispatch_paths();
void diag_reset_dispatch_paths();
uint64_t diag_dump_align_fastpath_count();

// DIAG: per-segment timers inside the warm-cache hit path. Returns
// (n_hits, ns_lookup, ns_io_build, ns_gil, ns_prep_in, ns_prep_out, ns_run,
//  ns_finalize). Counts/accumulates only when the hit path returns true; early
// failures (find miss, ptr==0, runtime soft-fail) are excluded so per-segment
// averages reflect successful warm-path calls only.
std::tuple<uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t>
diag_dump_warm_segments();
void diag_reset_warm_segments();

// Called by the Python wrapper after a successful miss-path compile to install
// a warm-cache entry keyed by the CacheKey that the shim built on the way in
// (stored in a thread-local so Python doesn't need to re-build it).
//
// Returns true if an install actually happened (pending key was valid and
// accepted). Safe to call when no pending context exists — returns false.
//
// `runtime_raw_ptr` is the opaque pointer to rebel::PyRblnSyncRuntime,
// extracted via the pybind-simple-layout offset trick in warm_cache.py.
// `out_profiles` is a list of (shape, dtype_str, is_rbln) per output tensor.
bool install_warmcache_from_pending(
    pybind11::object dyn_runtime,
    pybind11::int_ runtime_raw_ptr,
    uint32_t num_inputs,
    uint32_t num_outputs,
    const std::vector<std::tuple<std::vector<int64_t>, std::string, bool>>& out_profiles);

} // namespace torch_rbln::shim
