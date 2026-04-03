#pragma once

#include <c10/core/Device.h>
#include <c10/rbln/RBLNMacros.h>
#include <c10/util/CallOnce.h>
#include <array>
#include <string>
#include <unordered_set>
#include <vector>

namespace c10::rbln {

// Terminology (unified across Torch API and this implementation)
//
// - Logical device: The device index visible to Torch (e.g. rbln:0, rbln:1). One logical device
//   may map to one or more physical NPUs. Used in device_count(), device index checks, and
//   user-facing messages as "logical device(s)".
//
// - Physical NPU: The NPU hardware index (0, 1, 2, ...) as reported by the runtime.
//   RBLN_DEVICE_MAP and RBLN_NPUS_PER_DEVICE define how many physical NPUs form one
//   logical device (NPU count mapping).

// Type Definitions

/**
 * @brief Entry in device topology representing a single logical device.
 */
class DeviceTopologyEntry {
 public:
  /**
   * @brief Get the logical device index.
   * @return The logical device index (rbln:0, rbln:1, ...).
   */
  int getLogicalDeviceIndex() const {
    return logical_device_index_;
  }

  /**
   * @brief Get the physical NPU IDs mapped to this logical device.
   * @return Vector of physical NPU indices.
   */
  const std::vector<int>& getPhysicalDeviceIds() const {
    return physical_device_ids_;
  }

  /**
   * @brief Check if this is an aggregated device.
   * @return True if aggregated, false otherwise.
   */
  bool isAggregated() const {
    return is_aggregated_;
  }

  // Friend class for construction
  friend class DeviceMappingManager;

 private:
  int logical_device_index_;
  std::vector<int> physical_device_ids_;
  bool is_aggregated_;
};

/**
 * @brief Complete device topology: all logical devices and unused physical NPU IDs.
 */
class DeviceTopology {
 public:
  /**
   * @brief Get the device topology entries (one per logical device).
   * @return Vector of device topology entries.
   */
  const std::vector<DeviceTopologyEntry>& getEntries() const {
    return entries_;
  }

  /**
   * @brief Get the unused physical NPU IDs (not assigned to any logical device).
   * @return Vector of unused physical NPU indices.
   */
  const std::vector<int>& getUnusedPhysicalDeviceIds() const {
    return unused_physical_device_ids_;
  }

  // Friend class for construction
  friend class DeviceMappingManager;

 private:
  std::vector<DeviceTopologyEntry> entries_;
  std::vector<int> unused_physical_device_ids_;
};

/**
 * @brief Internal: one logical device -> list of physical NPU indices.
 */
struct DeviceMapping {
  c10::DeviceIndex logical_device = 0; // Logical device index (rbln:N)
  std::vector<int> physical_device_ids; // Physical NPU indices
};

// Constants

/**
 * @brief Valid sizes for physical NPUs per logical device (for aggregation).
 *
 * This matches the base_sizes defined in rebel/core/compilation/_impl.py (line 770):
 *   base_sizes = [1, 2, 4, 8, 16, 32]
 */
constexpr std::array<int, 6> BASE_SIZES = {1, 2, 4, 8, 16, 32};
constexpr size_t BASE_SIZES_COUNT = BASE_SIZES.size();

// DeviceMappingManager Class

/**
 * @brief Manages RBLN NPU count mapping: logical devices <-> physical NPUs.
 *
 * Handles initialization from RBLN_DEVICE_MAP or RBLN_NPUS_PER_DEVICE, and provides
 * access to the mapping. Initialized once on first use.
 */
class C10_RBLN_API DeviceMappingManager {
 public:
  // Singleton Access

  /**
   * @brief Get the singleton instance of DeviceMappingManager.
   * @return Reference to the singleton instance.
   */
  static DeviceMappingManager& getInstance();

  // Initialization

  /**
   * @brief Initialize device mapping from environment variables.
   *
   * This function is automatically called during singleton construction,
   * so explicit calls are typically not necessary. It is thread-safe and
   * will only initialize once, even if called multiple times. It reads
   * RBLN_DEVICE_MAP or RBLN_NPUS_PER_DEVICE environment variables to set
   * up the device mapping.
   */
  void initialize();

  // Public Query Methods

  /**
   * @brief Get the number of logical devices.
   * @return The number of logical devices (rbln:0 .. rbln:N-1).
   */
  c10::DeviceIndex getLogicalDeviceCount() const {
    return device_count_;
  }

  /**
   * @brief Check if a logical device index is assigned.
   * @param device_index The logical device index (rbln:N).
   * @return True if the device is assigned, false otherwise.
   */
  bool isDeviceAssigned(c10::DeviceIndex device_index) const {
    return assigned_devices_.find(device_index) != assigned_devices_.end();
  }

  /**
   * @brief Get physical NPU indices mapped to a logical device.
   * @param logical_device_index The logical device index.
   * @return Vector of physical NPU indices.
   */
  std::vector<int> getPhysicalDeviceIds(c10::DeviceIndex logical_device_index) const;

  /**
   * @brief Get the list of unused physical NPU IDs.
   * @return Vector of unused physical NPU indices.
   */
  std::vector<int> getUnusedPhysicalDeviceIds() const {
    return unused_physical_devices_;
  }

  /**
   * @brief Get the device mapping table.
   * @return Reference to the device mapping table.
   */
  const std::vector<DeviceMapping>& getDeviceMappingTable() const {
    return device_mapping_table_;
  }

  /**
   * @brief Get the cached device topology.
   * @return Reference to the cached device topology.
   */
  const DeviceTopology& getDeviceTopology() const {
    return device_topology_;
  }

  // Deleted Methods

  DeviceMappingManager(const DeviceMappingManager&) = delete;
  DeviceMappingManager& operator=(const DeviceMappingManager&) = delete;

 private:
  // Construction/Destruction

  DeviceMappingManager();
  ~DeviceMappingManager() = default;

  // Private Helper Methods

  /**
   * @brief Parse RBLN_DEVICE_MAP environment variable.
   * @param device_map_str Format: "[0,1],[2,3,4,5]" (each bracket is one logical device mapping)
   * @return Vector of vectors: each inner vector is the physical NPU indices for one logical device
   */
  std::vector<std::vector<int>> parseDeviceMap(const std::string& device_map_str);

  /**
   * @brief Register one logical device with its physical NPU indices.
   */
  void registerLogicalDevice(int logical_device_index, const std::vector<int>& physical_ids);

  /**
   * @brief Collect unused physical NPU indices based on usage tracking.
   */
  void collectUnusedDevices(const std::vector<bool>& physical_device_used, int physical_device_count);

  /**
   * @brief Initialize RBLN NPU mapping from RBLN_DEVICE_MAP environment variable.
   */
  void initializeFromDeviceMap(const std::string& device_map_str, int physical_device_count);

  /**
   * @brief Initialize RBLN NPU mapping from RBLN_NPUS_PER_DEVICE environment variable.
   */
  void initializeFromNpusPerDevice(int npus_per_device, int physical_device_count);

  /**
   * @brief Check if the number of physical NPUs per logical device is valid (must be in BASE_SIZES).
   */
  bool isValidDeviceGroupSize(size_t size) const;

  /**
   * @brief Get a string representation of valid sizes for error messages.
   */
  std::string getValidSizesString() const;

  /**
   * @brief Build and cache the device topology.
   */
  void buildDeviceTopology();

  // Member Variables

  c10::DeviceIndex device_count_ = 0;
  std::unordered_set<c10::DeviceIndex> assigned_devices_;
  std::vector<DeviceMapping> device_mapping_table_;
  std::vector<int> unused_physical_devices_;
  DeviceTopology device_topology_;
  c10::once_flag init_flag_;
};

/** Invoked once after device mapping topology is built (e.g. torch_rbln._C registers a Python logger). */
using RblnDeviceMappingInitializedCallback = void (*)();

C10_RBLN_API void register_rbln_device_mapping_initialized_callback(RblnDeviceMappingInitializedCallback cb);

/**
 * @brief RBLN NPU mapping env vars: current process values for error messages.
 *
 * Used to display RBLN_DEVICE_MAP and RBLN_NPUS_PER_DEVICE when reporting
 * NPU count / mapping configuration errors. Unset or empty is represented as "-".
 */
struct RblnNpuMappingEnvDisplay {
  std::string device_map; // RBLN_DEVICE_MAP value
  std::string npus_per_device; // RBLN_NPUS_PER_DEVICE value
};

/**
 * @brief Get current process's RBLN NPU mapping env (RBLN_DEVICE_MAP, RBLN_NPUS_PER_DEVICE) for display.
 */
C10_RBLN_API RblnNpuMappingEnvDisplay getRblnNpuMappingEnvDisplay();

} // namespace c10::rbln
