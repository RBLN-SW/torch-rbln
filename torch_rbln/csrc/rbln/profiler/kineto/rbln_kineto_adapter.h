#ifndef TORCH_RBLN_PROFILER_KINETO_RBLN_KINETO_ADAPTER_H
#define TORCH_RBLN_PROFILER_KINETO_RBLN_KINETO_ADAPTER_H

namespace rbln {
namespace profiler {
namespace kineto {

// Registers the rbln IActivityProfiler factory with libkineto; call once at _C
// module init. Explicit (not static-init) -- static-init in a pybind module can
// be dead-stripped.
void register_rbln_kineto_profiler();

} // namespace kineto
} // namespace profiler
} // namespace rbln

#endif // TORCH_RBLN_PROFILER_KINETO_RBLN_KINETO_ADAPTER_H
