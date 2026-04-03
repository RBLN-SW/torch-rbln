#pragma once

#include <ATen/native/CPUFallback.h>

namespace at::native::rbln {

/**
 * @brief CPU fallback for RBLN operators.
 *
 * @param op The operator handle.
 * @param stack The JIT stack.
 * @param error_on_views If true, an error is raised on view operators.
 * @param cpu_dispatch_key The dispatch key for CPU.
 */
void cpu_fallback_rbln(const c10::OperatorHandle& op, torch::jit::Stack* stack, bool error_on_views = false,
                       c10::DispatchKey cpu_dispatch_key = c10::DispatchKey::CPU);

} // namespace at::native::rbln
