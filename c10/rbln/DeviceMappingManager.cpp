#include <c10/rbln/DeviceMappingManager.h>
#include <c10/rbln/RBLNFunctions.h>
#include <c10/rbln/RBLNLogging.h>
#include <rebel/runtime/api/rbln_runtime_api.h>

#include <atomic>
#include <cctype>
#include <cstdlib>
#include <exception>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

namespace c10::rbln {

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
    RBLN_CHECK(false, "{}='{}' is not a valid integer", var_name, value);
  }
  // std::stoi parses only a leading prefix ("1abc" -> 1); reject any trailing
  // non-whitespace so a malformed value errors instead of being truncated.
  while (pos < value.size() && (std::isspace(static_cast<unsigned char>(value[pos])) != 0)) {
    ++pos;
  }
  RBLN_CHECK(pos == value.size(), "{}='{}' is not a valid integer", var_name, value);
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

DeviceMappingManager::DeviceMappingManager() {
  initialize();
}

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
  // Mirror rebel's ParseBool truthy set (it already rejected non-boolean values
  // at startup, so only these and the falsy spellings can reach here).
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
    RBLN_CHECK(
        device_map_str[pos] == '[',
        "Invalid RBLN_DEVICE_MAP format. Expected '[' at position {} (format: \"[0,1],[2,3]\")",
        pos);
    pos++; // consume '['

    std::vector<int> group;
    std::string num_str;
    bool expect_number = true; // a number is required after '[' and after each ','

    while (true) {
      skip_spaces();
      RBLN_CHECK(pos < len, "Invalid RBLN_DEVICE_MAP format. Unterminated group; expected ']' before end of value");
      const char ch = device_map_str[pos];
      if (ch == ']') {
        // "[]" or a trailing ',' before ']' leaves expect_number set.
        RBLN_CHECK(!expect_number, "Invalid RBLN_DEVICE_MAP format. Empty group or trailing ',' at position {}", pos);
        break;
      }
      if (ch == ',') {
        RBLN_CHECK(
            !expect_number, "Invalid RBLN_DEVICE_MAP format. Empty list element (unexpected ',') at position {}", pos);
        group.push_back(parseEnvInt(num_str, "RBLN_DEVICE_MAP"));
        num_str.clear();
        expect_number = true;
        pos++;
        continue;
      }
      RBLN_CHECK(
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
    RBLN_CHECK(
        device_map_str[pos] == ',', "Invalid RBLN_DEVICE_MAP format. Expected ',' between groups at position {}", pos);
    pos++; // consume ','
    skip_spaces();
    RBLN_CHECK(pos < len, "Invalid RBLN_DEVICE_MAP format. Trailing ',' with no group after it");
  }

  return result;
}

void DeviceMappingManager::registerLogicalDevice(int logical_device_index, const std::vector<int>& physical_ids) {
  // Register the logical device with its physical NPU indices
  // Need a non-const copy for rbln_register_device_id which requires int*
  std::vector<int> physical_ids_copy = physical_ids;
  const int rc = rbln_register_device_id(
      logical_device_index, physical_ids_copy.data(), static_cast<int>(physical_ids_copy.size()));
  RBLN_CHECK(
      rc == 0,
      "rbln_register_device_id failed for rbln:{} on physical NPU(s) [{}] (rc={}); the device(s) may be in use by "
      "another process or hold stale allocations. Free the device(s) or adjust RBLN_DEVICES.",
      logical_device_index,
      fmt::join(physical_ids_copy, ","),
      rc);
  assigned_devices_.insert(static_cast<c10::DeviceIndex>(logical_device_index));

  // Store mapping information
  DeviceMapping mapping;
  mapping.logical_device = static_cast<c10::DeviceIndex>(logical_device_index);
  mapping.physical_device_ids = physical_ids;
  device_mapping_table_.emplace_back(std::move(mapping));

  // Log the registration
  RBLN_LOG_DEBUG(
      "Registered logical device {} with physical NPU IDs: {}", logical_device_index, fmt::join(physical_ids, ","));
}

void DeviceMappingManager::collectUnusedDevices(
    const std::vector<bool>& physical_device_used,
    int physical_device_count) {
  for (int i = 0; i < physical_device_count; ++i) {
    if (!physical_device_used[i]) {
      unused_physical_devices_.push_back(i);
    }
  }
}

void DeviceMappingManager::initializeFromDeviceMap(const std::string& device_map_str, int physical_device_count) {
  RBLN_LOG_INFO("Using RBLN_DEVICE_MAP mode");
  std::vector<std::vector<int>> device_groups = parseDeviceMap(device_map_str);

  RBLN_CHECK(!device_groups.empty(), "RBLN_DEVICE_MAP must contain at least one logical device mapping");

  RblnNpuMappingEnvDisplay env_display = getRblnNpuMappingEnvDisplay();
  std::vector<bool> physical_device_used(physical_device_count, false);
  int logical_device_index = 0;

  for (const auto& group : device_groups) {
    RBLN_CHECK(!group.empty(), "Each logical device mapping in RBLN_DEVICE_MAP must contain at least one physical NPU");

    // Validate mapping size: physical NPUs per logical device must be one of the allowed base sizes
    RBLN_CHECK(
        isValidDeviceGroupSize(group.size()),
        "Each logical device mapping in RBLN_DEVICE_MAP must contain a valid number of physical NPUs. "
        "Valid sizes are: {}. Mapping with {} physical NPU(s) is invalid.",
        getValidSizesString(),
        group.size());

    // Validate physical NPU IDs (each mapping lists physical NPU indices for one logical device)
    for (int phy_id : group) {
      if (phy_id < 0 || phy_id >= physical_device_count) {
        std::string map_display = env_display.device_map;
        if (map_display.size() > 80) {
          map_display = map_display.substr(0, 77) + "...";
        }
        RBLN_CHECK(
            false,
            "Physical NPU {} out of range (this process has {} physical NPU(s), valid 0..{}). "
            "Env RBLN_DEVICE_MAP={}, RBLN_NPUS_PER_DEVICE={}.",
            phy_id,
            physical_device_count,
            physical_device_count - 1,
            map_display,
            env_display.npus_per_device);
      }
      RBLN_CHECK(
          !physical_device_used[phy_id], "Physical NPU {} is already assigned to another logical device", phy_id);
      physical_device_used[phy_id] = true;
    }

    // Register this logical device with its physical NPU indices
    registerLogicalDevice(logical_device_index, group);
    logical_device_index++;
  }

  device_count_ = static_cast<c10::DeviceIndex>(logical_device_index);

  // Collect unused physical NPU indices
  collectUnusedDevices(physical_device_used, physical_device_count);
}

void DeviceMappingManager::initializeFromNpusPerDevice(int npus_per_device, int physical_device_count) {
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
      registerLogicalDevice(logical_device_index, physical_ids);
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
    RBLN_CHECK(
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

std::optional<std::vector<std::vector<int>>> DeviceMappingManager::getDummyDeviceGroups() {
  if (!dummyDeviceEnabled()) {
    return std::nullopt;
  }
  // RBLN_DEVICE_MAP, when present, defines the layout (and RSD/TP shape) even in
  // dummy mode — group sizes are preserved so torch.compile's auto TP sizing
  // still works; ids are not range-validated against hardware (there is none).
  // Otherwise default to a single host-backed logical device (TP=1).
  const char* map_env = std::getenv("RBLN_DEVICE_MAP");
  if (map_env != nullptr && map_env[0] != '\0') {
    auto groups = parseDeviceMap(std::string(map_env));
    if (!groups.empty()) {
      return groups;
    }
  }
  return std::vector<std::vector<int>>{{0}};
}

void DeviceMappingManager::initializeDummyDevices(const std::vector<std::vector<int>>& groups) {
  constexpr auto kMaxDeviceIndex = static_cast<size_t>(std::numeric_limits<c10::DeviceIndex>::max());
  RBLN_CHECK(
      groups.size() <= kMaxDeviceIndex,
      "RBLN_DUMMY_DEVICE/RBLN_DEVICE_MAP requests {} logical devices, exceeding the maximum of {}",
      groups.size(),
      kMaxDeviceIndex);
  RBLN_LOG_INFO(
      "RBLN_DUMMY_DEVICE active: presenting {} host-backed logical device(s), 0 physical NPU. "
      "Tensor construction and compilation run on host memory; actual execution (kernels / compiled "
      "graphs) still requires an NPU.",
      groups.size());
  std::unordered_set<int> used_physical_ids;
  for (size_t i = 0; i < groups.size(); ++i) {
    // Validate group size (TP shape) and reject duplicate physical ids — both
    // are enforceable without hardware, so a config that parses under dummy
    // matches what the real path accepts. Only the physical-id *range* check is
    // skipped (there is no NPU to range-check against).
    RBLN_CHECK(
        isValidDeviceGroupSize(groups[i].size()),
        "RBLN_DEVICE_MAP group for rbln:{} has {} physical NPU(s); valid sizes are {}.",
        i,
        groups[i].size(),
        getValidSizesString());
    for (int phy_id : groups[i]) {
      RBLN_CHECK(
          used_physical_ids.insert(phy_id).second,
          "Physical NPU {} is assigned to more than one logical device in RBLN_DEVICE_MAP",
          phy_id);
    }
    const auto logical_index = static_cast<c10::DeviceIndex>(i);
    assigned_devices_.insert(logical_index);
    DeviceMapping mapping;
    mapping.logical_device = logical_index;
    // Shape markers only (RSD/TP sizing); no real NPU backs these ids.
    mapping.physical_device_ids = groups[i];
    device_mapping_table_.emplace_back(std::move(mapping));
  }
  device_count_ = static_cast<c10::DeviceIndex>(groups.size());
  buildDeviceTopology();
}

void DeviceMappingManager::initialize() {
  c10::call_once(init_flag_, [this]() {
    RBLN_LOG_DEBUG("Initializing RBLN device mapping");

    // Explicit opt-in (RBLN_DUMMY_DEVICE), forced regardless of physical NPU
    // presence. Checked before any runtime query so a host with no SDK/driver
    // can still construct device tensors and compile.
    if (auto dummy_groups = getDummyDeviceGroups()) {
      initializeDummyDevices(*dummy_groups);
      return;
    }

    int device_count = 0;
    // A failed query (SDK/driver not loadable) stays fatal; "query succeeded,
    // found 0 NPUs" is handled below.
    RBLN_CHECK(
        !rbln_get_device_count(&device_count),
        "rbln_get_device_count failed; the RBLN SDK/driver may not be loadable — ensure the SDK is installed "
        "and the kernel driver is loaded");
    const int physical_device_count = device_count;
    RBLN_LOG_DEBUG("Found {} physical NPU(s)", physical_device_count);

    // No physical NPU: register 0 logical devices instead of failing (like
    // torch.cuda.device_count()==0 on a CPU-only host). Device use fails at the
    // point of use, so a model can still be traced/compiled without an NPU.
    if (physical_device_count == 0) {
      RBLN_LOG_INFO(
          "No physical NPU detected; initializing with 0 logical device(s). "
          "Device access will fail at the point of use.");
      device_count_ = 0;
      buildDeviceTopology();
      return;
    }

    // Check RBLN NPU mapping env (RBLN_DEVICE_MAP takes priority over RBLN_NPUS_PER_DEVICE)
    RblnNpuMappingEnvDisplay env_display = getRblnNpuMappingEnvDisplay();

    if (env_display.device_map != "-" && !env_display.device_map.empty()) {
      // RBLN_DEVICE_MAP mode: use explicit mapping
      initializeFromDeviceMap(env_display.device_map, physical_device_count);
    } else {
      // RBLN_NPUS_PER_DEVICE mode: map physical NPUs to logical devices by count
      // If RBLN_NPUS_PER_DEVICE is not set, default to 1 (1:1 mapping)
      int npus_per_device = 1;
      if (env_display.npus_per_device != "-" && !env_display.npus_per_device.empty()) {
        npus_per_device = parseEnvInt(env_display.npus_per_device, "RBLN_NPUS_PER_DEVICE");
        RBLN_CHECK(npus_per_device > 0, "RBLN_NPUS_PER_DEVICE must be a positive integer");
        // Validate: must be one of the allowed base sizes
        RBLN_CHECK(
            isValidDeviceGroupSize(static_cast<size_t>(npus_per_device)),
            "RBLN_NPUS_PER_DEVICE must be one of the valid sizes: {}. Got {} which is invalid.",
            getValidSizesString(),
            npus_per_device);
      }
      initializeFromNpusPerDevice(npus_per_device, physical_device_count);
    }

    // Build and cache the device mapping summary
    buildDeviceTopology();
  });
}

std::vector<int> DeviceMappingManager::getPhysicalDeviceIds(c10::DeviceIndex logical_device_index) const {
  RBLN_CHECK(
      logical_device_index >= 0 && logical_device_index < static_cast<c10::DeviceIndex>(device_mapping_table_.size()),
      "Invalid logical device index: {}",
      static_cast<int>(logical_device_index));

  const auto& mapping = device_mapping_table_[logical_device_index];
  return mapping.physical_device_ids;
}

void DeviceMappingManager::buildDeviceTopology() {
  device_topology_.entries_.clear();
  device_topology_.unused_physical_device_ids_.clear();

  // Build entries for all logical devices
  for (c10::DeviceIndex i = 0; i < device_count_; ++i) {
    DeviceTopologyEntry entry;
    entry.logical_device_index_ = static_cast<int>(static_cast<unsigned char>(i));
    entry.physical_device_ids_ = getPhysicalDeviceIds(i);
    entry.is_aggregated_ = entry.physical_device_ids_.size() > 1;
    device_topology_.entries_.emplace_back(std::move(entry));
  }

  // Copy unused physical NPU IDs
  device_topology_.unused_physical_device_ids_ = unused_physical_devices_;
}

} // namespace c10::rbln
