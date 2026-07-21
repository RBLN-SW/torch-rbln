#ifndef TORCH_RBLN_PROFILER_KINETO_RBLN_KINETO_ADAPTER_H
#define TORCH_RBLN_PROFILER_KINETO_RBLN_KINETO_ADAPTER_H

namespace rbln::profiler::kineto {

// Registers the rbln IActivityProfiler factory with libkineto. Explicit (not
// static-init) -- static-init in a pybind module can be dead-stripped.
void register_rbln_kineto_profiler();

} // namespace rbln::profiler::kineto

#endif // TORCH_RBLN_PROFILER_KINETO_RBLN_KINETO_ADAPTER_H
