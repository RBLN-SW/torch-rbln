#pragma once

#include <pybind11/pybind11.h>

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
void register_cpp_shim(const std::string& op_name,
                       pybind11::object py_fn,
                       const std::vector<size_t>& skip_dtype_args = {});

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
    uintptr_t runtime_raw_ptr,
    uint32_t num_inputs,
    uint32_t num_outputs,
    const std::vector<std::tuple<std::vector<int64_t>, std::string, bool>>& out_profiles);

} // namespace torch_rbln::shim
