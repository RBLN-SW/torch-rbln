#pragma once

#include <ATen/native/CPUFallback.h>
#include <c10/rbln/RBLNMacros.h>

namespace at::native::rbln {

/**
 * @brief CPU fallback for RBLN operators.
 *
 * @param op The operator handle.
 * @param stack The JIT stack.
 * @param error_on_views If true, an error is raised on view operators.
 * @param cpu_dispatch_key The dispatch key for CPU.
 */
C10_RBLN_API void cpu_fallback_rbln(const c10::OperatorHandle& op, torch::jit::Stack* stack,
                                    bool error_on_views = false,
                                    c10::DispatchKey cpu_dispatch_key = c10::DispatchKey::CPU);

// DIAG: per-stage cumulative ns. Returns (calls, ns_setup, ns_dispatch, ns_writeback, ns_release).
C10_RBLN_API std::tuple<uint64_t, uint64_t, uint64_t, uint64_t, uint64_t>
diag_dump_cpu_fallback_stages();
C10_RBLN_API void diag_reset_cpu_fallback_stages();

} // namespace at::native::rbln
