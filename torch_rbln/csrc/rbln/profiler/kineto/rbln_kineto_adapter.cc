// libkineto plugin coupling a torch.profiler scope with one rbln profiling
// scope, through the rbln_kineto_* C-ABI (rebel/runtime/api/rbln_kineto_api.h).
// start()/stop() drive the exporter via the C API

#include "torch_rbln/csrc/rbln/profiler/kineto/rbln_kineto_adapter.h"

#include <kineto/ActivityType.h>
#include <kineto/Config.h>
#include <kineto/GenericTraceActivity.h>
#include <kineto/IActivityProfiler.h>
#include <kineto/TraceSpan.h>
#include <kineto/libkineto.h>
#include <kineto/output_base.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "rebel/runtime/api/rbln_kineto_api.h"
#include "torch_rbln/csrc/rbln/profiler/kineto/rbln_kineto_emitter.h"

namespace rbln {
namespace profiler {
namespace kineto {

namespace {

// GenericTraceActivity keeps the span by pointer, and the activities live longer
// than this session (they travel into torch's trace result), so the span must
// live at least as long: static, not a session member/local (which would dangle).
const ::libkineto::TraceSpan& default_trace_span() {
  static ::libkineto::TraceSpan span(0, 0, "rbln_trace");
  return span;
}

int64_t now_ns(std::chrono::steady_clock::time_point tp) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(tp.time_since_epoch()).count();
}
int64_t now_ns(std::chrono::system_clock::time_point tp) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(tp.time_since_epoch()).count();
}

} // namespace

class RblnActivityProfilerSession : public ::libkineto::IActivityProfilerSession {
 public:
  RblnActivityProfilerSession() = default;
  ~RblnActivityProfilerSession() override = default;
  RblnActivityProfilerSession(const RblnActivityProfilerSession&) = delete;
  RblnActivityProfilerSession& operator=(const RblnActivityProfilerSession&) = delete;

  void start() override {
    status_ = ::libkineto::TraceStatus::RECORDING;
    // (steady, system) anchor at scope open; exported slices carry steady_clock ns,
    // so we add (system - steady) at projection time to convert them to system time.
    anchor_steady_ns_ = now_ns(std::chrono::steady_clock::now());
    anchor_system_ns_ = now_ns(std::chrono::system_clock::now());
    start_ts_ns_ = anchor_system_ns_;
    rbln_kineto_begin_scope();
  }

  void stop() override {
    status_ = ::libkineto::TraceStatus::PROCESSING;
    clock_offset_ns_ = anchor_system_ns_ - anchor_steady_ns_;
    projected_ = ProjectedKinetoTrace{};
    int32_t exported = 0;
    rbln_kineto_end_scope_and_export(&sink_thunk, this, &exported);
    status_ = ::libkineto::TraceStatus::READY;
  }

  void on_export(const RblnKinetoExport* exp) {
    convert_export_to_kineto(exp, clock_offset_ns_, default_trace_span(), &projected_);
  }

  std::vector<std::string> errors() override {
    return {};
  }

  void processTrace(::libkineto::ActivityLogger& logger) override {
    // Register device/resource rows through the logger, not the getDeviceInfo()
    // getter: that getter returns only one DeviceInfo, but a trace can have
    // multiple device rows, so the single getter can't fit them.
    for (const auto& dev : projected_.device_infos) {
      logger.handleDeviceInfo(dev, start_ts_ns_);
    }
    for (const auto& res : projected_.resource_infos) {
      logger.handleResourceInfo(res, start_ts_ns_);
    }
    for (const auto& act : projected_.activities) {
      act.log(logger);
    }
  }

  std::unique_ptr<::libkineto::DeviceInfo> getDeviceInfo() override {
    return {};
  }

  std::vector<::libkineto::ResourceInfo> getResourceInfos() override {
    return {};
  }

  std::unique_ptr<::libkineto::CpuTraceBuffer> getTraceBuffer() override {
    auto buf = std::make_unique<::libkineto::CpuTraceBuffer>();
    for (const auto& act : projected_.activities) {
      buf->emplace_activity(act);
    }
    return buf;
  }

 private:
  static void sink_thunk(const RblnKinetoExport* exp, void* user_data) {
    static_cast<RblnActivityProfilerSession*>(user_data)->on_export(exp);
  }

  ProjectedKinetoTrace projected_;
  int64_t anchor_steady_ns_ = 0;
  int64_t anchor_system_ns_ = 0;
  int64_t clock_offset_ns_ = 0;
  int64_t start_ts_ns_ = 0; // metadata-event timestamp for device/resource rows
};

class RblnActivityProfiler : public ::libkineto::IActivityProfiler {
 public:
  RblnActivityProfiler()
      : supported_{::libkineto::ActivityType::PRIVATEUSE1_RUNTIME, ::libkineto::ActivityType::PRIVATEUSE1_DRIVER} {}
  ~RblnActivityProfiler() override = default;

  const std::string& name() const override {
    static const std::string kName = "RblnProfiler";
    return kName;
  }

  const std::set<::libkineto::ActivityType>& availableActivities() const override {
    return supported_;
  }

  std::unique_ptr<::libkineto::IActivityProfilerSession> configure(
      const std::set<::libkineto::ActivityType>& /*activity_types*/,
      const ::libkineto::Config&) override {
    // Create a session only when rbln profiling is actually running
    // (rbln_kineto_is_active()), not based on the requested activity types: pytorch
    // always includes PRIVATEUSE1_RUNTIME/DRIVER in its default set, so the request
    // can't tell us whether the user actually wants rbln profiling.
    int32_t active = 0;
    rbln_kineto_is_active(&active);
    if (!active)
      return nullptr;
    return std::make_unique<RblnActivityProfilerSession>();
  }

  std::unique_ptr<::libkineto::IActivityProfilerSession> configure(
      int64_t /*ts_ms*/,
      int64_t /*duration_ms*/,
      const std::set<::libkineto::ActivityType>& activity_types,
      const ::libkineto::Config& config) override {
    return configure(activity_types, config);
  }

 private:
  std::set<::libkineto::ActivityType> supported_;
};

namespace {

std::unique_ptr<::libkineto::IActivityProfiler> create_rbln_profiler() {
  return std::make_unique<RblnActivityProfiler>();
}

} // namespace

void register_rbln_kineto_profiler() {
  ::libkineto::api().registerProfilerFactory(&create_rbln_profiler);
}

} // namespace kineto
} // namespace profiler
} // namespace rbln
