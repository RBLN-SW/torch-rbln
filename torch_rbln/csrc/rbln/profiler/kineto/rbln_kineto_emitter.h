#ifndef TORCH_RBLN_PROFILER_KINETO_RBLN_KINETO_EMITTER_H
#define TORCH_RBLN_PROFILER_KINETO_RBLN_KINETO_EMITTER_H

// C-ABI -> libkineto assembly

#include <kineto/ActivityType.h>
#include <kineto/GenericTraceActivity.h>
#include <kineto/IActivityProfiler.h>
#include <kineto/TraceSpan.h>

#include <cstdint>
#include <vector>

#include <rebel/runtime/api/rbln_kineto_api.h>

namespace rbln::profiler::kineto {

// Sort-priority base for rbln device rows: keeps them below host CPU rows in the
// Perfetto UI; each sorts at base + pid so multi-node rows keep node order.
constexpr int64_t kRblnDeviceSortIndex = 5000000;

// One projection's output, emitted through the logger in processTrace.
struct ProjectedKinetoTrace {
  std::vector<::libkineto::DeviceInfo> device_infos;
  std::vector<::libkineto::ResourceInfo> resource_infos;
  std::vector<::libkineto::GenericTraceActivity> activities;
};

// Assembles 'out' (cleared first) from the C export, converting each slice's
// steady_clock time to system time by adding clock_offset_ns. All strings are
// copied, so 'exp' can be discarded once this returns.
void convert_export_to_kineto(
    const RblnKinetoExport* exp,
    int64_t clock_offset_ns,
    const ::libkineto::TraceSpan& span,
    ProjectedKinetoTrace* out);

} // namespace rbln::profiler::kineto

#endif // TORCH_RBLN_PROFILER_KINETO_RBLN_KINETO_EMITTER_H
