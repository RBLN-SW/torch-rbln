// C-ABI -> libkineto assembly

#include <torch_rbln/csrc/rbln/profiler/kineto/rbln_kineto_emitter.h>

#include <cstdint>
#include <string>

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

  out->activities.reserve(exp->slices_count);
  for (uint32_t i = 0; i < exp->slices_count; ++i) {
    const RblnKinetoSlice& s = exp->slices[i];
    const ::libkineto::ActivityType act_type = kind_to_activity_type(s.kind);
    ::libkineto::GenericTraceActivity act(span, act_type, std::string(s.name ? s.name : ""));
    act.startTime = s.start_steady_ns + clock_offset_ns;
    act.endTime = s.end_steady_ns + clock_offset_ns;
    act.device = s.device_pid;
    act.resource = s.resource_tid;
    act.id = static_cast<int32_t>(s.corr_id);

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
}

} // namespace rbln::profiler::kineto
