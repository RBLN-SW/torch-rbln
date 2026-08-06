#include <c10/rbln/DeviceMappingManager.h>
#include <c10/rbln/RBLNFunctions.h>
#include <c10/rbln/RBLNLogging.h>
#include <rebel/runtime/api/rbln_runtime_api.h>

#include <atomic>
#include <cctype>
#include <cstdlib>
#include <exception>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

namespace c10::rbln {

// Every failure here uses RBLN_CHECK_QUIET, not RBLN_CHECK: mapping init is reached
// from is_available()/device_count(), which torch requires never to throw, so callers
// catch. RBLN_CHECK would still log c10::Error::what() -- stack trace included -- to
// the console of every co-tenant process that merely asked whether an NPU exists.
// The message rides the exception; the point of use logs it when it matters.

namespace {

std::atomic<RblnDeviceMappingInitializedCallback> g_device_mapping_initialized_cb{nullptr};

// Parse an int from a user env value; turn std::stoi's context-free throw into
// an actionable error naming the variable and the bad value.
int parseEnvInt(const std::string& value, const char* var_name) {
  size_t pos = 0;
  int result = 0;
  try {
    result = std::stoi(value, &pos);
  } catch (const std::exception&) {
    RBLN_CHECK_QUIET(false, "{}='{}' is not a valid integer", var_name, value);
  }
  // std::stoi parses only a leading prefix ("1abc" -> 1); reject any trailing
  // non-whitespace so a malformed value errors instead of being truncated.
  while (pos < value.size() && (std::isspace(static_cast<unsigned char>(value[pos])) != 0)) {
    ++pos;
  }
  RBLN_CHECK_QUIET(pos == value.size(), "{}='{}' is not a valid integer", var_name, value);
  return result;
}

} // namespace

void register_rbln_device_mapping_initialized_callback(RblnDeviceMappingInitializedCallback cb) {
  g_device_mapping_initialized_cb.store(cb, std::memory_order_release);
}

DeviceMappingManager& DeviceMappingManager::getInstance() {
  static DeviceMappingManager instance;
  // Fire optional hook once after singleton exists. Must not use call_once for the callback: the
  // callback may call Python _get_device_topology() -> getInstance() on the same thread while
  // call_once's init function is still running -> self-deadlock (appears as hang / infinite loop).
  static std::atomic<bool> mapping_ready_hook_done{false};
  if (!mapping_ready_hook_done.exchange(true, std::memory_order_acq_rel)) {
    if (auto cb = g_device_mapping_initialized_cb.load(std::memory_order_acquire)) {
      cb();
    }
  }
  return instance;
}

// Nothing here: construction must not touch the environment or the runtime. Planning is
// lazy (ensurePlanned) so that a throwing configuration cannot escape the constructor.
// It used to: a function-local static whose constructor throws is re-attempted on the
// next call ([stmt.dcl]/4), and c10::call_once leaves its flag unset on a throwing
// initializer (c10/util/CallOnce.h), so a malformed RBLN_* config re-ran the whole
// mapping init -- rbln_register_device_id() included -- on every single query.
DeviceMappingManager::DeviceMappingManager() = default;

bool DeviceMappingManager::isValidDeviceGroupSize(size_t size) const {
  for (const auto& base_size : BASE_SIZES) {
    if (static_cast<size_t>(base_size) == size) {
      return true;
    }
  }
  return false;
}

std::string DeviceMappingManager::getValidSizesString() const {
  std::stringstream ss;
  bool first = true;
  for (const auto& base_size : BASE_SIZES) {
    if (!first) {
      ss << ", ";
    }
    ss << base_size;
    first = false;
  }
  return ss.str();
}

void DeviceMappingManager::validateDeviceGroups(const std::vector<std::vector<int>>& groups) const {
  // Hardware-independent checks shared by the real (RBLN_DEVICE_MAP / RBLN_NPUS_PER_DEVICE)
  // and dummy paths. The physical-id *range* check needs a physical device count and so
  // stays in the real path (initializeFromDeviceMap).
  constexpr auto kMaxDeviceIndex = static_cast<size_t>(std::numeric_limits<c10::DeviceIndex>::max());
  RBLN_CHECK_QUIET(
      groups.size() <= kMaxDeviceIndex,
      "RBLN_DEVICE_MAP/RBLN_NPUS_PER_DEVICE requests {} logical devices, exceeding the maximum of {}",
      groups.size(),
      kMaxDeviceIndex);

  std::unordered_set<int> used_physical_ids;
  for (size_t i = 0; i < groups.size(); ++i) {
    RBLN_CHECK_QUIET(
        isValidDeviceGroupSize(groups[i].size()),
        "Logical device rbln:{} has {} physical NPU(s); valid sizes are {}.",
        i,
        groups[i].size(),
        getValidSizesString());
    for (int phy_id : groups[i]) {
      RBLN_CHECK_QUIET(
          used_physical_ids.insert(phy_id).second,
          "Physical NPU {} is assigned to more than one logical device",
          phy_id);
    }
  }
}

RblnNpuMappingEnvDisplay getRblnNpuMappingEnvDisplay() {
  const char* map_env = std::getenv("RBLN_DEVICE_MAP");
  const char* npus_env = std::getenv("RBLN_NPUS_PER_DEVICE");
  return {
      (map_env && map_env[0] != '\0' ? std::string(map_env) : "-"),
      (npus_env && npus_env[0] != '\0' ? std::string(npus_env) : "-"),
  };
}

bool dummyDeviceEnabled() {
  const char* env = std::getenv("RBLN_DUMMY_DEVICE");
  if (env == nullptr || env[0] == '\0') {
    return false;
  }
  // Truthy spellings of the RBLN_DUMMY_DEVICE boolean flag (non-boolean values
  // are already rejected by the runtime at startup).
  std::string s(env);
  const auto first = s.find_first_not_of(" \t\n\r\f\v");
  if (first == std::string::npos) {
    return false;
  }
  s = s.substr(first, s.find_last_not_of(" \t\n\r\f\v") - first + 1);
  for (char& c : s) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return s == "1" || s == "true" || s == "t" || s == "yes" || s == "y" || s == "on";
}

std::vector<std::vector<int>> DeviceMappingManager::parseDeviceMap(const std::string& device_map_str) {
  // Grammar (whitespace-insensitive): groups := group (',' group)* ;
  //   group := '[' number (',' number)* ']' . Empty elements/groups and trailing
  // commas at either level are rejected so a malformed map errors loudly instead
  // of being silently truncated (e.g. "[0,]" / "[]" / "[0],,[1]" / "[0],[1],").
  std::vector<std::vector<int>> result;
  const size_t len = device_map_str.length();
  size_t pos = 0;

  auto skip_spaces = [&]() {
    while (pos < len && device_map_str[pos] == ' ') {
      pos++;
    }
  };

  skip_spaces();
  // An empty / whitespace-only value means "no explicit mapping"; callers fall
  // back to a default layout. Any non-empty value must be fully well-formed.
  if (pos >= len) {
    return result;
  }

  while (true) {
    // ---- one group: '[' number (',' number)* ']' ----
    RBLN_CHECK_QUIET(
        device_map_str[pos] == '[',
        "Invalid RBLN_DEVICE_MAP format. Expected '[' at position {} (format: \"[0,1],[2,3]\")",
        pos);
    pos++; // consume '['

    std::vector<int> group;
    std::string num_str;
    bool expect_number = true; // a number is required after '[' and after each ','

    while (true) {
      skip_spaces();
      RBLN_CHECK_QUIET(
          pos < len, "Invalid RBLN_DEVICE_MAP format. Unterminated group; expected ']' before end of value");
      const char ch = device_map_str[pos];
      if (ch == ']') {
        // "[]" or a trailing ',' before ']' leaves expect_number set.
        RBLN_CHECK_QUIET(
            !expect_number, "Invalid RBLN_DEVICE_MAP format. Empty group or trailing ',' at position {}", pos);
        break;
      }
      if (ch == ',') {
        RBLN_CHECK_QUIET(
            !expect_number, "Invalid RBLN_DEVICE_MAP format. Empty list element (unexpected ',') at position {}", pos);
        group.push_back(parseEnvInt(num_str, "RBLN_DEVICE_MAP"));
        num_str.clear();
        expect_number = true;
        pos++;
        continue;
      }
      RBLN_CHECK_QUIET(
          ch >= '0' && ch <= '9', "Invalid RBLN_DEVICE_MAP format. Unexpected character '{}' at position {}", ch, pos);
      num_str += ch;
      expect_number = false;
      pos++;
    }
    // Commit the final number (group is non-empty: expect_number is false at ']').
    group.push_back(parseEnvInt(num_str, "RBLN_DEVICE_MAP"));
    result.emplace_back(std::move(group));
    pos++; // consume ']'

    // ---- separator: end of string, or ',' followed by another group ----
    skip_spaces();
    if (pos >= len) {
      break;
    }
    RBLN_CHECK_QUIET(
        device_map_str[pos] == ',', "Invalid RBLN_DEVICE_MAP format. Expected ',' between groups at position {}", pos);
    pos++; // consume ','
    skip_spaces();
    RBLN_CHECK_QUIET(pos < len, "Invalid RBLN_DEVICE_MAP format. Trailing ',' with no group after it");
  }

  return result;
}

void DeviceMappingManager::planLogicalDevice(int logical_device_index, const std::vector<int>& physical_ids) const {
  // Bookkeeping only. rbln_register_device_id() lives in commit(); see the class
  // comment for why an availability query must not reach it.
  assigned_devices_.insert(static_cast<c10::DeviceIndex>(logical_device_index));

  DeviceMapping mapping;
  mapping.logical_device = static_cast<c10::DeviceIndex>(logical_device_index);
  mapping.physical_device_ids = physical_ids;
  device_mapping_table_.emplace_back(std::move(mapping));

  RBLN_LOG_DEBUG(
      "Planned logical device {} with physical NPU IDs: {}", logical_device_index, fmt::join(physical_ids, ","));
}

void DeviceMappingManager::commit() {
  {
    const std::lock_guard<std::mutex> guard(plan_mutex_);
    if (committed_) {
      return;
    }
    ensurePlannedLocked();
    // Loud here (RBLN_CHECK, not the quiet variant): this is the point of use. Its
    // logging goes to spdlog only, so it is safe to throw from under the lock.
    RBLN_CHECK(plan_error_.empty(), "{}", plan_error_);

    for (const auto& mapping : device_mapping_table_) {
      // Kept as DeviceIndex; widened only at each use. Binding it to an `int` local trips
      // bugprone-signed-char-misuse, since DeviceIndex is a signed char.
      const auto logical_device = mapping.logical_device;
      // rbln_register_device_id takes int*, so a non-const copy is required.
      std::vector<int> physical_ids = mapping.physical_device_ids;
      const int rc = rbln_register_device_id(
          static_cast<int>(logical_device), physical_ids.data(), static_cast<int>(physical_ids.size()));
      // No rollback is possible: the runtime exposes no unregister call, so devices claimed
      // by earlier iterations stay claimed. Name them, so the leak is visible.
      const std::string already_claimed = (logical_device > 0)
          ? fmt::format(
                " Note: rbln:0..{} were already registered by this process and remain claimed.",
                static_cast<int>(logical_device) - 1)
          : std::string();
      RBLN_CHECK(
          rc == 0,
          "rbln_register_device_id failed for rbln:{} on physical NPU(s) [{}] (rc={}); the device(s) may be in use by "
          "another process or hold stale allocations. Free the device(s) or adjust RBLN_DEVICES.{}",
          static_cast<int>(logical_device),
          fmt::join(physical_ids, ","),
          rc,
          already_claimed);
    }

    committed_ = true;
    RBLN_LOG_DEBUG("Committed {} logical device(s) to the runtime", device_mapping_table_.size());
  }
}

bool DeviceMappingManager::isCommitted() const {
  const std::lock_guard<std::mutex> guard(plan_mutex_);
  return committed_;
}

void DeviceMappingManager::collectUnusedDevices(
    const std::vector<bool>& physical_device_used,
    int physical_device_count) const {
  for (int i = 0; i < physical_device_count; ++i) {
    if (!physical_device_used[i]) {
      unused_physical_devices_.push_back(i);
    }
  }
}

void DeviceMappingManager::initializeFromDeviceMap(const std::string& device_map_str, int physical_device_count) const {
  RBLN_LOG_INFO("Using RBLN_DEVICE_MAP mode");
  std::vector<std::vector<int>> device_groups = parseDeviceMap(device_map_str);

  RBLN_CHECK_QUIET(!device_groups.empty(), "RBLN_DEVICE_MAP must contain at least one logical device mapping");

  // Shape / duplicate-id / count validation shared with the dummy path.
  validateDeviceGroups(device_groups);

  RblnNpuMappingEnvDisplay env_display = getRblnNpuMappingEnvDisplay();
  std::vector<bool> physical_device_used(physical_device_count, false);
  int logical_device_index = 0;

  for (const auto& group : device_groups) {
    // Physical-id range check + usage tracking for unused-device collection (needs
    // hardware, so it stays here rather than in validateDeviceGroups).
    for (int phy_id : group) {
      if (phy_id < 0 || phy_id >= physical_device_count) {
        std::string map_display = env_display.device_map;
        if (map_display.size() > 80) {
          map_display = map_display.substr(0, 77) + "...";
        }
        RBLN_CHECK_QUIET(
            false,
            "Physical NPU {} out of range (this process has {} physical NPU(s), valid 0..{}). "
            "Env RBLN_DEVICE_MAP={}, RBLN_NPUS_PER_DEVICE={}.",
            phy_id,
            physical_device_count,
            physical_device_count - 1,
            map_display,
            env_display.npus_per_device);
      }
      physical_device_used[phy_id] = true;
    }

    // Register this logical device with its physical NPU indices
    planLogicalDevice(logical_device_index, group);
    logical_device_index++;
  }

  device_count_ = static_cast<c10::DeviceIndex>(logical_device_index);

  // Collect unused physical NPU indices
  collectUnusedDevices(physical_device_used, physical_device_count);
}

void DeviceMappingManager::initializeFromNpusPerDevice(int npus_per_device, int physical_device_count) const {
  if (npus_per_device == 1) {
    RBLN_LOG_INFO("Using default 1:1 mapping (RBLN_NPUS_PER_DEVICE=1)");
  } else {
    RBLN_LOG_INFO("Using RBLN_NPUS_PER_DEVICE mode (RBLN_NPUS_PER_DEVICE={})", npus_per_device);
  }

  // Track which physical NPUs are used
  std::vector<bool> physical_device_used(physical_device_count, false);

  int logical_device_index = 0;
  int current_phy_index = 0;

  while (current_phy_index < physical_device_count) {
    std::vector<int> physical_ids;

    // Group physical NPUs into one logical device
    for (int i = 0; i < npus_per_device && current_phy_index < physical_device_count; ++i) {
      physical_ids.push_back(current_phy_index);
      physical_device_used[current_phy_index] = true;
      current_phy_index++;
    }

    // Only register if we have a complete set of NPUs for one logical device (size == npus_per_device)
    // Incomplete mappings (remaining physical NPUs < npus_per_device) will be marked as unused
    if (static_cast<int>(physical_ids.size()) == npus_per_device) {
      planLogicalDevice(logical_device_index, physical_ids);
      logical_device_index++;
    } else {
      // Incomplete logical device mapping: mark these physical NPUs as unused
      // Note: physical_device_used was already set to true, but we'll reset it
      // so they get collected in the unused_physical_devices_ vector below
      for (int phy_id : physical_ids) {
        physical_device_used[phy_id] = false;
      }
      RBLN_LOG_DEBUG(
          "Incomplete logical device mapping: {} physical NPU(s) (expected {}), marking as unused",
          physical_ids.size(),
          npus_per_device);
    }
  }

  device_count_ = static_cast<c10::DeviceIndex>(logical_device_index);

  if (device_count_ == 0) {
    RblnNpuMappingEnvDisplay env_display = getRblnNpuMappingEnvDisplay();
    RBLN_CHECK_QUIET(
        false,
        "No logical device (this process has {} physical NPU(s), need {} per logical device). "
        "Env RBLN_DEVICE_MAP={}, RBLN_NPUS_PER_DEVICE={}.",
        physical_device_count,
        npus_per_device,
        env_display.device_map,
        npus_per_device);
  }

  // Collect unused physical NPU indices
  collectUnusedDevices(physical_device_used, physical_device_count);
}

void DeviceMappingManager::initializeDummyDevices() const {
  // Layout from RBLN_DEVICE_MAP (TP shape preserved), else RBLN_NPUS_PER_DEVICE as
  // one logical device of size N, else a single device. IDs are shape markers; no
  // NPU backs them, so they are not range-checked against hardware.
  std::vector<std::vector<int>> groups;
  if (const char* map_env = std::getenv("RBLN_DEVICE_MAP"); map_env != nullptr && map_env[0] != '\0') {
    groups = parseDeviceMap(std::string(map_env));
  }
  if (groups.empty()) {
    int npus_per_device = 1;
    const auto env_display = getRblnNpuMappingEnvDisplay();
    if (env_display.npus_per_device != "-" && !env_display.npus_per_device.empty()) {
      npus_per_device = parseEnvInt(env_display.npus_per_device, "RBLN_NPUS_PER_DEVICE");
      RBLN_CHECK_QUIET(npus_per_device > 0, "RBLN_NPUS_PER_DEVICE must be a positive integer, got {}", npus_per_device);
    }
    std::vector<int> group;
    group.reserve(static_cast<size_t>(npus_per_device));
    for (int i = 0; i < npus_per_device; ++i) {
      group.push_back(i);
    }
    groups = {group};
  }

  // Shape / duplicate-id / count validation shared with the real path; the physical-id
  // range check is skipped (there is no NPU to range them against).
  validateDeviceGroups(groups);

  RBLN_LOG_INFO(
      "RBLN_DUMMY_DEVICE active: {} host-backed logical device(s), 0 physical NPU. "
      "Tensor construction/compilation run on host memory; execution still needs an NPU.",
      groups.size());

  for (size_t i = 0; i < groups.size(); ++i) {
    planLogicalDevice(static_cast<int>(i), groups[i]);
  }
  device_count_ = static_cast<c10::DeviceIndex>(groups.size());
  buildDeviceTopology();
}

std::string DeviceMappingManager::envSignature() {
  // Any change to one of these changes the plan. Values are length-prefixed so that
  // e.g. RBLN_DEVICES="0|" and RBLN_DEVICES="0", RBLN_DEVICE_MAP="" cannot collide.
  std::string signature;
  for (const char* name : {"RBLN_DEVICES", "RBLN_DEVICE_MAP", "RBLN_NPUS_PER_DEVICE", "RBLN_DUMMY_DEVICE"}) {
    const char* value = std::getenv(name);
    const std::string text = (value != nullptr) ? std::string(value) : std::string();
    signature += std::to_string(text.size());
    signature += ':';
    signature += text;
  }
  return signature;
}

void DeviceMappingManager::ensurePlanned() const {
  std::string error;
  {
    const std::lock_guard<std::mutex> guard(plan_mutex_);
    ensurePlannedLocked();
    error = plan_error_;
  }
  // Rethrown here rather than inside the lock so that every query method reports a
  // malformed RBLN_* config identically, and get_device_count_nothrow() can map it to 0.
  RBLN_CHECK_QUIET(error.empty(), "{}", error);
}

void DeviceMappingManager::ensurePlannedLocked() const {
  // Frozen after commit(): re-planning under live allocations would silently move
  // logical devices out from under them.
  if (committed_) {
    return;
  }
  const auto signature = envSignature();
  if (planned_ && signature == plan_signature_) {
    return;
  }

  device_count_ = 0;
  assigned_devices_.clear();
  device_mapping_table_.clear();
  unused_physical_devices_.clear();
  plan_error_.clear();
  plan_signature_ = signature;
  planned_ = true;

  try {
    buildPlan();
  } catch (const std::exception& e) {
    // Remembered, not rethrown from here: query methods raise it, the nothrow ones map
    // it to "0 devices". Storing it is what keeps a malformed config from re-running
    // validation (and, before the plan/commit split, device registration) per call.
    plan_error_ = e.what();
    device_count_ = 0;
    assigned_devices_.clear();
    device_mapping_table_.clear();
    unused_physical_devices_.clear();
    buildDeviceTopology();
  }
}

void DeviceMappingManager::buildPlan() const {
  RBLN_LOG_DEBUG("Planning RBLN device mapping");

  // Without the runtime nothing can execute, so report 0 devices as before. Planning
  // itself no longer needs it; commit() does, and every caller already treats "no
  // runtime" as "no device".
  if (!rbln_runtime_available()) {
    RBLN_LOG_INFO(
        "RBLN runtime not loaded; planning 0 logical device(s). Device access will fail at the point of use.");
    buildDeviceTopology();
    return;
  }

  // Dummy: host-backed, no NPU. Physical ids are shape markers, so no probe is needed.
  if (dummyDeviceEnabled()) {
    initializeDummyDevices();
    return;
  }

  // The runtime is the only authority on how many NPUs RBLN_DEVICES leaves visible.
  // Calling it seals RBLN_DEVICES, so an availability query freezes the value for this
  // process and everything it forks -- see rebellions-sw/fsw-inference#475. Working
  // around that here (counting /dev/rbln* and parsing RBLN_DEVICES ourselves) was tried
  // and rejected: it creates a second source of truth for something the runtime owns.
  int physical_device_count = 0;
  RBLN_CHECK_QUIET(
      !rbln_get_device_count(&physical_device_count),
      "rbln_get_device_count failed; the RBLN kernel driver may not be loaded or the device is unavailable");
  RBLN_LOG_DEBUG("Found {} physical NPU(s)", physical_device_count);

  // No physical NPU: plan 0 logical devices instead of failing (like
  // torch.cuda.device_count() == 0 on a CPU-only host). Device use fails at the point of
  // use, so a model can still be traced/compiled without an NPU.
  if (physical_device_count <= 0) {
    RBLN_LOG_INFO(
        "No physical NPU detected; planning 0 logical device(s). Device access will fail at the point of use.");
    buildDeviceTopology();
    return;
  }

  // RBLN_DEVICE_MAP takes priority over RBLN_NPUS_PER_DEVICE.
  const RblnNpuMappingEnvDisplay env_display = getRblnNpuMappingEnvDisplay();
  if (env_display.device_map != "-" && !env_display.device_map.empty()) {
    initializeFromDeviceMap(env_display.device_map, physical_device_count);
  } else {
    // Unset RBLN_NPUS_PER_DEVICE means 1 (a 1:1 mapping).
    int npus_per_device = 1;
    if (env_display.npus_per_device != "-" && !env_display.npus_per_device.empty()) {
      npus_per_device = parseEnvInt(env_display.npus_per_device, "RBLN_NPUS_PER_DEVICE");
      RBLN_CHECK_QUIET(npus_per_device > 0, "RBLN_NPUS_PER_DEVICE must be a positive integer");
      RBLN_CHECK_QUIET(
          isValidDeviceGroupSize(static_cast<size_t>(npus_per_device)),
          "RBLN_NPUS_PER_DEVICE must be one of the valid sizes: {}. Got {} which is invalid.",
          getValidSizesString(),
          npus_per_device);
    }
    initializeFromNpusPerDevice(npus_per_device, physical_device_count);
  }

  buildDeviceTopology();
}

void DeviceMappingManager::initialize() {
  ensurePlanned();
}

std::vector<int> DeviceMappingManager::getPhysicalDeviceIds(c10::DeviceIndex logical_device_index) const {
  ensurePlanned();
  RBLN_CHECK_QUIET(
      logical_device_index >= 0 && logical_device_index < static_cast<c10::DeviceIndex>(device_mapping_table_.size()),
      "Invalid logical device index: {}",
      static_cast<int>(logical_device_index));

  const auto& mapping = device_mapping_table_[logical_device_index];
  return mapping.physical_device_ids;
}

void DeviceMappingManager::buildDeviceTopology() const {
  device_topology_.entries_.clear();
  device_topology_.unused_physical_device_ids_.clear();

  // Build entries for all logical devices
  for (c10::DeviceIndex i = 0; i < device_count_; ++i) {
    DeviceTopologyEntry entry;
    entry.logical_device_index_ = static_cast<int>(static_cast<unsigned char>(i));
    // Read the table directly: the public getter plans on demand, and this runs while
    // the plan lock is already held.
    entry.physical_device_ids_ = device_mapping_table_[static_cast<size_t>(i)].physical_device_ids;
    entry.is_aggregated_ = entry.physical_device_ids_.size() > 1;
    device_topology_.entries_.emplace_back(std::move(entry));
  }

  // Copy unused physical NPU IDs
  device_topology_.unused_physical_device_ids_ = unused_physical_devices_;
}

} // namespace c10::rbln
