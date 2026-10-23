#include <c10/rbln/DeviceMappingManager.h>
#include <c10/rbln/RBLNLogging.h>
#include <rebel/v2/runtime/device.h>
#include <rebel/v2/runtime/flags.h>

#include <array>
#include <atomic>
#include <cstdlib>
#include <exception>
#include <limits>
#include <string>
#include <vector>

namespace c10::rbln {

// Every failure here uses RBLN_CHECK_QUIET, not RBLN_CHECK: mapping init is reached
// from is_available()/device_count(), which torch requires never to throw, so callers
// catch. RBLN_CHECK would still log c10::Error::what() -- stack trace included -- to
// the console of every co-tenant process that merely asked whether an NPU exists.
// The message rides the exception; the point of use logs it when it matters.

namespace {

std::atomic<RblnDeviceMappingInitializedCallback> g_device_mapping_initialized_cb{nullptr};

constexpr std::array<const char*, 2> kGroupingVariables = {"RBLN_NPUS_PER_DEVICE", "RBLN_DEVICE_MAP"};

std::string envValue(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr ? value : "";
}

void rejectGroupingVariables() {
  std::vector<std::string> set;
  for (const char* name : kGroupingVariables) {
    if (const auto value = envValue(name); !value.empty()) {
      set.push_back(fmt::format("{}={}", name, value));
    }
  }
  RBLN_CHECK_QUIET(
      set.empty(),
      "{} {} set, but a torch device is one NPU on rebel.v2: rbln:N is the N-th NPU RBLN_DEVICES selects. "
      "Unset {} and select the NPUs with RBLN_DEVICES.",
      fmt::join(set, " and "),
      set.size() == 1 ? "is" : "are",
      set.size() == 1 ? "it" : "them");
}

} // namespace

void register_rbln_device_mapping_initialized_callback(RblnDeviceMappingInitializedCallback cb) {
  g_device_mapping_initialized_cb.store(cb, std::memory_order_release);
}

DeviceMappingManager& DeviceMappingManager::getInstance() {
  static DeviceMappingManager instance;
  // Fire optional hook once after singleton exists. Must not use call_once for the callback: the
  // callback may call back into getInstance() on the same thread while call_once's init function
  // is still running -> self-deadlock (appears as hang / infinite loop).
  static std::atomic<bool> mapping_ready_hook_done{false};
  if (!mapping_ready_hook_done.exchange(true, std::memory_order_acq_rel)) {
    if (auto cb = g_device_mapping_initialized_cb.load(std::memory_order_acquire)) {
      cb();
    }
  }
  return instance;
}

// Construction must not touch the environment or the runtime, so planning is lazy
// (ensurePlanned): a function-local static whose constructor throws is re-attempted on the
// next call ([stmt.dcl]/4), and c10::call_once leaves its flag unset on a throwing
// initializer (c10/util/CallOnce.h), which would re-run the whole mapping init -- the NPU
// probe included -- on every query under a rejected RBLN_* config.
DeviceMappingManager::DeviceMappingManager() = default;

bool dummyDeviceEnabled() {
  return ::rebel::v2::runtime::flags::kDummyDevice.value();
}

void DeviceMappingManager::commit() {
  // Committed is the common case: every allocation re-enters here, so answer it without
  // the mutex. Failed still has to take the lock -- it rethrows commit_error_ below.
  if (plan_state_.load(std::memory_order_acquire) == PlanState::Committed) {
    return;
  }
  const std::lock_guard<std::mutex> guard(plan_mutex_);
  if (plan_state_.load(std::memory_order_relaxed) == PlanState::Committed) {
    return;
  }
  // A part-way failure left NPUs open with no way to release them, so a retry against a
  // rebuilt plan would hold NPUs the new plan does not name.
  RBLN_CHECK(plan_state_.load(std::memory_order_relaxed) != PlanState::Failed, "{}", commit_error_);
  ensurePlannedLocked();
  // Loud here (RBLN_CHECK, not the quiet variant): this is the point of use. Its
  // logging goes to spdlog only, so it is safe to throw from under the lock.
  RBLN_CHECK(plan_error_.empty(), "{}", plan_error_);

  // A dummy device has no NPU behind it to open.
  if (!dummyDeviceEnabled()) {
    for (size_t index = 0; index < npus_.size(); ++index) {
      try {
        ::rebel::v2::runtime::Device::open(npus_[index]);
      } catch (const std::exception& e) {
        // No rollback: NPUs opened by earlier iterations stay open. Name them, so the
        // hold is visible.
        const std::string already_open = index > 0
            ? fmt::format(" Note: rbln:0..{} were already opened by this process and remain held.", index - 1)
            : std::string();
        commit_error_ = fmt::format(
            "cannot open NPU {} for rbln:{} ({}); the device may be in use by another process. Free the "
            "device or adjust RBLN_DEVICES.{}",
            npus_[index],
            index,
            e.what(),
            already_open);
        plan_state_.store(PlanState::Failed, std::memory_order_release);
        RBLN_CHECK(false, "{}", commit_error_);
      }
    }
  }

  // An empty plan registered nothing, so the freeze has nothing to protect and would only
  // make the 0-device state permanent. Leave it Open; to_device_id() still raises.
  if (npus_.empty()) {
    RBLN_LOG_DEBUG("Nothing to commit: 0 device(s) planned; the plan stays open");
    return;
  }

  plan_state_.store(PlanState::Committed, std::memory_order_release);
  RBLN_LOG_DEBUG("Committed {} device(s) to the runtime", npus_.size());
}

uint32_t DeviceMappingManager::npu(c10::DeviceIndex device_index) const {
  ensurePlanned();
  RBLN_CHECK(
      device_index >= 0 && static_cast<size_t>(device_index) < npus_.size(),
      "rbln:{} is not one of the {} device(s) of this process",
      static_cast<int>(device_index),
      npus_.size());
  return npus_[static_cast<size_t>(device_index)];
}

bool DeviceMappingManager::hasFailedCommit() const noexcept {
  return plan_state_.load(std::memory_order_acquire) == PlanState::Failed;
}

std::vector<uint32_t> DeviceMappingManager::seenNpus() const {
  if (plan_state_.load(std::memory_order_acquire) != PlanState::Open) {
    return npus_;
  }
  return visibleNpus();
}

bool DeviceMappingManager::isCommitted() const {
  return plan_state_.load(std::memory_order_acquire) == PlanState::Committed;
}

std::string DeviceMappingManager::envSignature() {
  // Any change to one of these changes the plan. Values are length-prefixed so that
  // e.g. RBLN_DEVICES="0|" and RBLN_DEVICES="0", RBLN_DEVICE_MAP="" cannot collide.
  // RBLN_DEVICES counts as the text its flag reads, under its name or its alias.
  //
  // RBLN_DUMMY_DEVICE is deliberately absent: it is startup-only, since the runtime
  // registers it FlagMutability::kSealed and is_dummy_device() caches for the process.
  std::string signature;
  auto add = [&signature](const std::string& text) {
    signature += std::to_string(text.size());
    signature += ':';
    signature += text;
  };
  add(::rebel::v2::runtime::flags::kDevices.raw().value_or(""));
  for (const char* name : kGroupingVariables) {
    add(envValue(name));
  }
  return signature;
}

void DeviceMappingManager::ensurePlanned() const {
  // A frozen plan is immutable, so skip the mutex: every allocation asks this repeatedly.
  // plan_error_ is necessarily empty here -- commit() refuses to freeze with one set.
  if (plan_state_.load(std::memory_order_acquire) != PlanState::Open) {
    return;
  }
  std::string error;
  {
    const std::lock_guard<std::mutex> guard(plan_mutex_);
    ensurePlannedLocked();
    error = plan_error_;
  }
  // Rethrown here rather than inside the lock so that every query method reports a
  // rejected RBLN_* config identically, and get_device_count_nothrow() can map it to 0.
  RBLN_CHECK_QUIET(error.empty(), "{}", error);
}

void DeviceMappingManager::ensurePlannedLocked() const {
  // Frozen once commit() has run: re-planning under live allocations would silently move
  // devices out from under them. A part-way failure counts too -- it still left NPUs open.
  if (plan_state_.load(std::memory_order_relaxed) != PlanState::Open) {
    return;
  }
  const auto signature = envSignature();
  if (planned_ && signature == plan_signature_) {
    return;
  }

  npus_.clear();
  plan_error_.clear();
  plan_signature_ = signature;
  planned_ = true;

  try {
    buildPlan();
  } catch (const std::exception& e) {
    // Remembered, not rethrown from here: query methods raise it, the nothrow ones map
    // it to "0 devices". Storing it is what keeps a rejected config from re-running
    // validation per call. msg() is the message alone; what() embeds a backtrace.
    const auto* c10_error = dynamic_cast<const c10::Error*>(&e);
    plan_error_ = c10_error != nullptr ? c10_error->msg() : e.what();
    npus_.clear();
  }
}

std::vector<uint32_t> DeviceMappingManager::visibleNpus() {
  std::vector<uint32_t> npus;
  const auto& visible = ::rebel::v2::runtime::flags::kDevices;
  if (visible.isSet()) {
    for (const uint32_t npu : visible.value()) {
      if (!::rebel::v2::runtime::Device::exists(npu)) {
        RBLN_LOG_WARN("RBLN_DEVICES names NPU {}, which this host does not have; leaving it out", npu);
        continue;
      }
      npus.push_back(npu);
    }
    return npus;
  }
  uint32_t count = 0;
  try {
    count = ::rebel::v2::runtime::Device::count();
  } catch (const std::exception& e) {
    RBLN_CHECK_QUIET(
        false,
        "cannot count NPUs; the RBLN kernel driver may not be loaded or the device may be unavailable: {}",
        e.what());
  }
  for (uint32_t npu = 0; npu < count; ++npu) {
    npus.push_back(npu);
  }
  return npus;
}

void DeviceMappingManager::planNpus() const {
  // Dummy: host-backed, no NPU to probe. One device per id RBLN_DEVICES names, else one.
  if (dummyDeviceEnabled()) {
    const auto& visible = ::rebel::v2::runtime::flags::kDevices;
    npus_ = visible.isSet() ? visible.value() : std::vector<uint32_t>{0};
    RBLN_LOG_INFO(
        "RBLN_DUMMY_DEVICE active: {} host-backed device(s), 0 physical NPU. "
        "Tensor construction/compilation run on host memory; execution still needs an NPU.",
        npus_.size());
    return;
  }

  // Without the runtime nothing can execute, so report 0 devices. Planning does not need it;
  // commit() does, and every caller treats "no runtime" as "no device".
  if (!::rebel::v2::runtime::Device::available()) {
    RBLN_LOG_INFO("RBLN driver not loaded; planning 0 device(s). Device access will fail at the point of use.");
    return;
  }

  // No NPU: plan 0 devices instead of failing (like torch.cuda.device_count() == 0 on a
  // CPU-only host). Device use fails at the point of use, so a model can still be
  // traced/compiled without an NPU.
  npus_ = visibleNpus();
  RBLN_LOG_DEBUG("Planned {} device(s) on NPU(s) {}", npus_.size(), fmt::join(npus_, ","));
}

void DeviceMappingManager::buildPlan() const {
  RBLN_LOG_DEBUG("Planning RBLN device mapping");
  rejectGroupingVariables();
  planNpus();
  constexpr auto kMaxDevices = static_cast<size_t>(std::numeric_limits<c10::DeviceIndex>::max());
  RBLN_CHECK_QUIET(
      npus_.size() <= kMaxDevices,
      "RBLN_DEVICES selects {} NPUs, more than the {} devices torch can index",
      npus_.size(),
      kMaxDevices);
}

void DeviceMappingManager::initialize() {
  ensurePlanned();
}

} // namespace c10::rbln
