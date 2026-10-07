// Runtime activities -> libkineto assembly

#include <torch_rbln/csrc/rbln/profiler/kineto/rbln_kineto_emitter.h>

#include <c10/rbln/RBLNLogging.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rbln::profiler::kineto {

namespace {

using ::rebel::v2::runtime::Activity;

constexpr std::array<const char*, 4> kLaneNames = {"compute", "copy", "host", "collective"};

::libkineto::ActivityType activity_type(Activity::Kind kind) {
  using ::libkineto::ActivityType;
  switch (kind) {
    case Activity::Kind::kDevice:
      return ActivityType::CONCURRENT_KERNEL;
    case Activity::Kind::kCopy:
      return ActivityType::GPU_MEMCPY;
    case Activity::Kind::kCollective:
      return ActivityType::PRIVATEUSE1_DRIVER;
    case Activity::Kind::kHost:
    default:
      return ActivityType::PRIVATEUSE1_RUNTIME;
  }
}

constexpr uint32_t kRblnFlowIdBase = 0xF0000000u;
constexpr uint32_t kRblnFlowIdSpan = 0x01000000u;

// An arrow from each launch to the activities it ran, and a zero-length marker at the launch
// for its tail; a launch that ran nothing gets no marker, so none dangles.
void add_flow_arrows(
    const ::rebel::v2::runtime::Activities& recorded,
    int64_t clock_offset_ns,
    const ::libkineto::TraceSpan& span,
    ProjectedKinetoTrace* out) {
  const auto& launches = recorded.launches;
  const size_t assignable = std::min<size_t>(launches.size(), kRblnFlowIdSpan);
  if (assignable < launches.size()) {
    RBLN_LOG_WARN(
        "rbln flow arrows: {} of {} launches exceed the flow-id field; no arrow drawn",
        launches.size() - assignable,
        launches.size());
  }
  std::unordered_map<uint64_t, uint32_t> flow_of;
  for (size_t i = 0; i < assignable; ++i) {
    flow_of.emplace(launches[i].id, kRblnFlowIdBase + static_cast<uint32_t>(i));
  }
  std::vector<bool> wired(assignable, false);
  for (size_t i = 0; i < recorded.activities.size(); ++i) {
    auto it = flow_of.find(recorded.activities[i].launch);
    if (it == flow_of.end()) {
      continue;
    }
    auto& act = out->activities[i];
    act.flow.id = it->second;
    act.flow.type = ::libkineto::kLinkAsyncCpuGpu;
    act.flow.start = 0;
    wired[it->second - kRblnFlowIdBase] = true;
  }
  for (size_t i = 0; i < assignable; ++i) {
    if (!wired[i]) {
      continue;
    }
    const auto& launch = launches[i];
    ::libkineto::GenericTraceActivity src(span, ::libkineto::ActivityType::PRIVATEUSE1_RUNTIME, "rbln launch");
    src.startTime = launch.ns + clock_offset_ns;
    src.endTime = src.startTime;
    src.device = launch.pid;
    src.resource = launch.tid;
    src.flow.id = kRblnFlowIdBase + static_cast<uint32_t>(i);
    src.flow.type = ::libkineto::kLinkAsyncCpuGpu;
    src.flow.start = 1;
    src.addMetadata("launch_id", std::to_string(launch.id));
    out->activities.push_back(std::move(src));
  }
}

} // namespace

void convert_activities_to_kineto(
    const ::rebel::v2::runtime::Activities& recorded,
    int64_t clock_offset_ns,
    const ::libkineto::TraceSpan& span,
    ProjectedKinetoTrace* out) {
  out->device_infos.clear();
  out->resource_infos.clear();
  out->activities.clear();

  std::set<uint32_t> devices;
  std::set<std::pair<uint32_t, Activity::Kind>> lanes;
  for (const auto& a : recorded.activities) {
    devices.insert(a.device);
    lanes.emplace(a.device, a.kind);
  }
  for (const auto device : devices) {
    const auto pid = static_cast<int64_t>(device);
    out->device_infos.push_back(::libkineto::DeviceInfo{
        /*id=*/pid,
        /*sortIndex=*/kRblnDeviceSortIndex + pid,
        /*name=*/"NPU " + std::to_string(device),
        /*label=*/std::string()});
  }
  for (const auto& [device, kind] : lanes) {
    const auto tid = static_cast<int64_t>(kind);
    out->resource_infos.push_back(::libkineto::ResourceInfo{
        /*id=*/tid,
        /*sortIndex=*/tid,
        /*deviceId=*/static_cast<int64_t>(device),
        /*name=*/kLaneNames.at(static_cast<size_t>(kind))});
  }

  out->activities.reserve(recorded.activities.size() + recorded.launches.size());
  for (const auto& a : recorded.activities) {
    ::libkineto::GenericTraceActivity act(span, activity_type(a.kind), a.name);
    act.startTime = a.start_ns + clock_offset_ns;
    act.endTime = a.end_ns + clock_offset_ns;
    act.device = static_cast<int64_t>(a.device);
    act.resource = static_cast<int64_t>(a.kind);
    if (a.launch != 0) {
      act.addMetadata("launch_id", std::to_string(a.launch));
    }
    out->activities.push_back(std::move(act));
  }

  add_flow_arrows(recorded, clock_offset_ns, span, out);
}

} // namespace rbln::profiler::kineto
