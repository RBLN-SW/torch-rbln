#ifndef TORCH_RBLN_PROFILER_KINETO_RBLN_KINETO_EMITTER_H
#define TORCH_RBLN_PROFILER_KINETO_RBLN_KINETO_EMITTER_H

// Runtime activities -> libkineto assembly

#include <kineto/ActivityType.h>
#include <kineto/GenericTraceActivity.h>
#include <kineto/IActivityProfiler.h>
#include <kineto/TraceSpan.h>
#include <rbln/runtime/activity.h>

#include <cstdint>
#include <vector>

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

// Assembles 'out' (cleared first) from what the runtime recorded: a row per device, a lane
// per kind of activity on it, and an arrow from each launch to what it ran. Times move from
// steady_clock to system time by adding clock_offset_ns.
void convert_activities_to_kineto(
    const ::rbln::runtime::Activities& recorded,
    int64_t clock_offset_ns,
    const ::libkineto::TraceSpan& span,
    ProjectedKinetoTrace* out);

} // namespace rbln::profiler::kineto

#endif // TORCH_RBLN_PROFILER_KINETO_RBLN_KINETO_EMITTER_H
