// C-ABI -> libkineto assembly

#include <torch_rbln/csrc/rbln/profiler/kineto/rbln_kineto_emitter.h>

#include <c10/rbln/RBLNLogging.h>

#include <cstdint>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>

namespace rbln::profiler::kineto {

namespace {

// Map the producer-set activity kind to a libkineto ActivityType.
::libkineto::ActivityType kind_to_activity_type(RblnKinetoActivityKind kind) {
  using ::libkineto::ActivityType;
  switch (kind) {
    case RBLN_KINETO_KIND_COMPUTE:
      return ActivityType::CONCURRENT_KERNEL;
    case RBLN_KINETO_KIND_DMA:
      return ActivityType::GPU_MEMCPY;
    case RBLN_KINETO_KIND_SYNC:
      return ActivityType::PRIVATEUSE1_DRIVER;
    case RBLN_KINETO_KIND_RUNTIME:
    default:
      return ActivityType::PRIVATEUSE1_RUNTIME;
  }
}

const char* slice_annotation(const RblnKinetoSlice& s, const char* name) {
  for (uint32_t a = 0; a < s.annotations_count; ++a) {
    const RblnKinetoAnnotation& ann = s.annotations[a];
    if (ann.name && ann.value && std::string(ann.name) == name)
      return ann.value;
  }
  return nullptr;
}

// Kernels would multiply the arrows; only the workload slices carry one.
bool is_dst_slice(const RblnKinetoSlice& s) { return s.kind == RBLN_KINETO_KIND_RUNTIME; }

// Region -> device ac2g arrows, keyed by launch_id so they bind across the async gap.
// Requires out->activities[i] to be slice i (the caller projects them in order).
void add_flow_arrows(const RblnKinetoExport* exp, int64_t clock_offset_ns,
                     const ::libkineto::TraceSpan& span, ProjectedKinetoTrace* out) {
  if (exp->host_launches_count == 0) {
    RBLN_LOG_INFO("rbln flow arrows: no host launches captured; 0 arrows");
    return;
  }

  // Step 1: assign one flow id per host launch.
  std::unordered_map<int64_t, uint32_t> flow_id_by_launch; // launch_id -> shared flow id
  flow_id_by_launch.reserve(exp->host_launches_count);
  for (uint32_t i = 0; i < exp->host_launches_count; ++i)
    flow_id_by_launch.emplace(exp->host_launches[i].launch_id, i + 1);

  // Step 2: put that flow id on the dst slices (1:N fan).
  std::vector<bool> wired(exp->host_launches_count, false);
  uint64_t arrows = 0, skipped_no_launch_id = 0, skipped_no_launch = 0;
  for (uint32_t i = 0; i < exp->slices_count; ++i) {
    const RblnKinetoSlice& s = exp->slices[i];
    if (!is_dst_slice(s))
      continue;
    const char* launch_id_str = slice_annotation(s, RBLN_KINETO_ANN_LAUNCH_ID);
    if (!launch_id_str) {
      ++skipped_no_launch_id;
      continue;
    }
    const int64_t launch_id = std::strtoll(launch_id_str, nullptr, 10);
    auto flow_id_it = flow_id_by_launch.find(launch_id);
    if (flow_id_it == flow_id_by_launch.end()) {
      ++skipped_no_launch;
      continue;
    }

    out->activities[i].flow.id = flow_id_it->second;
    out->activities[i].flow.type = ::libkineto::kLinkAsyncCpuGpu;
    out->activities[i].flow.start = 0;
    wired[flow_id_it->second - 1] = true;
    ++arrows;
  }

  // Step 3: put it on the source marker, only for launches whose slices got it, so none dangles.
  uint64_t sources = 0, unwired = 0;
  for (uint32_t i = 0; i < exp->host_launches_count; ++i) {
    if (!wired[i]) {
      ++unwired;
      continue;
    }
    const RblnKinetoHostLaunch& host_launch = exp->host_launches[i];
    ::libkineto::GenericTraceActivity src(
        span, ::libkineto::ActivityType::PRIVATEUSE1_RUNTIME,
        std::string(host_launch.name ? host_launch.name : ""));
    const int64_t ts = host_launch.steady_ns + clock_offset_ns;
    src.startTime = ts;
    src.endTime = ts;
    src.device = host_launch.pid;
    src.resource = host_launch.tid;
    src.flow.id = i + 1;
    src.flow.type = ::libkineto::kLinkAsyncCpuGpu;
    src.flow.start = 1;
    src.addMetadata(RBLN_KINETO_ANN_LAUNCH_ID, std::to_string(host_launch.launch_id));
    out->activities.push_back(std::move(src));
    ++sources;
  }
  if (unwired)
    RBLN_LOG_WARN("rbln flow arrows: {} of {} host launches had no RUNTIME slice; no arrow drawn",
                  unwired, exp->host_launches_count);

  RBLN_LOG_INFO(
      "rbln flow arrows: {} sources, {} arrows; skipped {} slices w/o launch_id, {} w/o host-launch match",
      sources, arrows, skipped_no_launch_id, skipped_no_launch);
}

} // namespace

void convert_export_to_kineto(
    const RblnKinetoExport* exp,
    int64_t clock_offset_ns,
    const ::libkineto::TraceSpan& span,
    ProjectedKinetoTrace* out) {
  if (!out)
    return;
  out->device_infos.clear();
  out->resource_infos.clear();
  out->activities.clear();
  if (!exp)
    return;

  out->device_infos.reserve(exp->devices_count);
  for (uint32_t i = 0; i < exp->devices_count; ++i) {
    const RblnKinetoDevice& dev = exp->devices[i];
    out->device_infos.emplace_back(
        /*id=*/dev.pid,
        /*sortIndex=*/kRblnDeviceSortIndex + dev.pid,
        /*name=*/std::string(dev.name ? dev.name : ""),
        /*label=*/std::string());
  }

  out->resource_infos.reserve(exp->lanes_count);
  for (uint32_t i = 0; i < exp->lanes_count; ++i) {
    const RblnKinetoLane& lane = exp->lanes[i];
    out->resource_infos.emplace_back(
        /*deviceId=*/lane.device_pid,
        /*id=*/lane.resource_tid,
        /*sortIndex=*/lane.resource_tid,
        /*name=*/std::string(lane.name ? lane.name : ""));
  }

  // Reserve capacity for all activities (slices + source markers).
  out->activities.reserve(exp->slices_count + exp->host_launches_count);
  for (uint32_t i = 0; i < exp->slices_count; ++i) {
    const RblnKinetoSlice& s = exp->slices[i];
    const ::libkineto::ActivityType act_type = kind_to_activity_type(s.kind);
    ::libkineto::GenericTraceActivity act(span, act_type, std::string(s.name ? s.name : ""));
    act.startTime = s.start_steady_ns + clock_offset_ns;
    act.endTime = s.end_steady_ns + clock_offset_ns;
    act.device = s.device_pid;
    act.resource = s.resource_tid;

    std::string joined;
    for (uint32_t c = 0; c < s.categories_count; ++c) {
      if (!s.categories[c])
        continue;
      if (!joined.empty())
        joined += ", ";
      joined += s.categories[c];
    }
    if (!joined.empty())
      act.addMetadataQuoted("categories", joined);
    for (uint32_t a = 0; a < s.annotations_count; ++a) {
      const RblnKinetoAnnotation& ann = s.annotations[a];
      const std::string ann_name = ann.name ? ann.name : "";
      const std::string ann_val = ann.value ? ann.value : "";
      if (ann.value_is_quoted) {
        act.addMetadataQuoted(ann_name, ann_val);
      } else {
        act.addMetadata(ann_name, ann_val);
      }
    }
    out->activities.push_back(std::move(act));
  }

  add_flow_arrows(exp, clock_offset_ns, span, out);
}

} // namespace rbln::profiler::kineto
