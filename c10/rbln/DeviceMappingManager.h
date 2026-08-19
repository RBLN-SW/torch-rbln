#pragma once

#include <c10/core/Device.h>
#include <c10/rbln/RBLNMacros.h>
#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>
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

  // Initialization is two stages:
  //
  //   plan   Parse RBLN_DEVICES (alias RBLN_VISIBLE_DEVICES) / RBLN_DEVICE_MAP /
  //          RBLN_NPUS_PER_DEVICE, validate, compute the logical->physical table.
  //          Claims no NPU. Run by availability and enumeration queries.
  //   commit rbln_register_device_id() per planned logical device, which opens a context
  //          on every mapped NPU. Deferred to the first actual device use.
  //
  // torch calls is_available()/device_count() from paths that never asked for an NPU, so
  // claiming hardware there takes NPUs from co-tenants;
  // ATen/detail/AcceleratorHooksInterface.h: isAvailable() "should NOT initialize the
  // context on any device".
  //
  // Commit is also where the runtime freezes its own RBLN_DEVICES mapping:
  // rbln_register_device_id() reaches Context::Create, which latches it
  // (rebellions-sw/rebel_compiler#12904). Both layers therefore freeze together, so a
  // launcher may still assign RBLN_DEVICES after import -- fork()ed workers included --
  // until the first device use.

  /**
   * @brief Plan the device mapping from the environment (no NPU is claimed).
   *
   * Idempotent, and safe to call concurrently. Every query method plans on demand, so
   * explicit calls are typically unnecessary.
   *
   * Mutating the RBLN_* environment concurrently with a query is NOT supported: before the
   * mapping commits, a change makes the next query rebuild the plan, clearing the containers
   * a getter reads. Assign the RBLN_* variables from one thread, before the queries.
   */
  void initialize();

  /**
   * @brief Claim the planned logical devices with the runtime, and freeze the plan.
   *
   * Idempotent. Called from to_device_id(), the shared precursor to every
   * device-touching runtime call, so the claim happens exactly when the process first
   * commits to using a device. Raises if the plan is invalid or a registration fails.
   */
  void commit();

  /**
   * @brief Whether the planned devices have been claimed with the runtime.
   *
   * Once committed the plan is frozen: later RBLN_* environment edits are ignored
   * rather than silently changing the mapping under live allocations.
   */
  bool isCommitted() const;

  // Public Query Methods

  /**
   * @brief Get the number of logical devices.
   * @return The number of logical devices (rbln:0 .. rbln:N-1).
   */
  c10::DeviceIndex getLogicalDeviceCount() const {
    ensurePlanned();
    return device_count_;
  }

  /**
   * @brief Check if a logical device index is assigned.
   * @param device_index The logical device index (rbln:N).
   * @return True if the device is assigned, false otherwise.
   */
  bool isDeviceAssigned(c10::DeviceIndex device_index) const {
    ensurePlanned();
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
    ensurePlanned();
    return unused_physical_devices_;
  }

  /**
   * @brief Get the device mapping table.
   * @return Reference to the device mapping table.
   */
  const std::vector<DeviceMapping>& getDeviceMappingTable() const {
    ensurePlanned();
    return device_mapping_table_;
  }

  /**
   * @brief Get the cached device topology.
   * @return Reference to the cached device topology.
   */
  const DeviceTopology& getDeviceTopology() const {
    ensurePlanned();
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
  static std::vector<std::vector<int>> parseDeviceMap(const std::string& device_map_str);

  /**
   * @brief Record one logical device -> physical NPU mapping. Bookkeeping only.
   *
   * Deliberately does NOT call rbln_register_device_id(); commit() does that later.
   */
  void planLogicalDevice(int logical_device_index, const std::vector<int>& physical_ids) const;

  /**
   * @brief Build the plan if it is missing or stale. Idempotent, and serialized on
   * plan_mutex_; see initialize() for the concurrent-environment caveat.
   *
   * A failed plan is remembered (not retried): rethrowing a stored error keeps a
   * malformed RBLN_* config from re-running validation on every query. Until commit()
   * the plan is rebuilt whenever the RBLN_* environment changes, matching
   * torch/cuda/__init__.py: "Do not cache the device count prior to CUDA
   * initialization, because the number of devices can change due to changes to
   * CUDA_VISIBLE_DEVICES setting prior to CUDA initialization." A vLLM worker assigns
   * RBLN_DEVICES after import, and this is what lets that assignment take effect.
   */
  void ensurePlanned() const;

  /**
   * @brief ensurePlanned() for callers that already hold plan_mutex_.
   */
  void ensurePlannedLocked() const;

  /**
   * @brief Build the plan from the current environment. Caller holds plan_mutex_.
   */
  void buildPlan() const;

  /**
   * @brief The RBLN_* environment the plan depends on, as a comparable string.
   */
  static std::string envSignature();

  /**
   * @brief Collect unused physical NPU indices based on usage tracking.
   */
  void collectUnusedDevices(const std::vector<bool>& physical_device_used, int physical_device_count) const;

  /**
   * @brief Initialize RBLN NPU mapping from RBLN_DEVICE_MAP environment variable.
   */
  void initializeFromDeviceMap(const std::string& device_map_str, int physical_device_count) const;

  /**
   * @brief Initialize RBLN NPU mapping from RBLN_NPUS_PER_DEVICE environment variable.
   */
  void initializeFromNpusPerDevice(int npus_per_device, int physical_device_count) const;

  /**
   * @brief Register host-backed logical devices (RBLN_DUMMY_DEVICE) with no NPU.
   *
   * Layout comes from RBLN_DEVICE_MAP / RBLN_NPUS_PER_DEVICE (or a single
   * device); physical ids are shape markers only and are not range-checked.
   */
  void initializeDummyDevices() const;

  /**
   * @brief Check if the number of physical NPUs per logical device is valid (must be in BASE_SIZES).
   */
  bool isValidDeviceGroupSize(size_t size) const;

  /**
   * @brief Get a string representation of valid sizes for error messages.
   */
  std::string getValidSizesString() const;

  /**
   * @brief Hardware-independent validation of a logical->physical device group layout,
   * shared by the RBLN_DEVICE_MAP (real) and RBLN_DUMMY_DEVICE paths: bounds the logical
   * device count, requires a valid group size, and rejects a physical id assigned to more
   * than one logical device. The physical-id range check needs a physical device count and
   * stays in the real path.
   */
  void validateDeviceGroups(const std::vector<std::vector<int>>& groups) const;

  /**
   * @brief Build and cache the device topology.
   */
  void buildDeviceTopology() const;

  // Member Variables

  // Guards every member below. The plan is a lazy cache rebuilt on demand, hence
  // `mutable` on state a const query may refresh.
  mutable std::mutex plan_mutex_;
  mutable bool planned_ = false;
  // Both non-Open states freeze the plan for good; Failed means commit() threw part-way and
  // the devices it claimed cannot be released. Atomic because a frozen plan can never
  // change, so the allocation path's five queries read it without plan_mutex_. Written under
  // the lock; release/acquire pairs with those writes.
  enum class PlanState : std::uint8_t { Open, Committed, Failed };
  mutable std::atomic<PlanState> plan_state_{PlanState::Open};
  std::string commit_error_; // Failed: the error every later commit() rethrows
  mutable std::string plan_signature_; // envSignature() the current plan was built from
  mutable std::string plan_error_; // non-empty: the plan is invalid, rethrown on query

  mutable c10::DeviceIndex device_count_ = 0;
  mutable std::unordered_set<c10::DeviceIndex> assigned_devices_;
  mutable std::vector<DeviceMapping> device_mapping_table_;
  mutable std::vector<int> unused_physical_devices_;
  mutable DeviceTopology device_topology_;
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

/**
 * @brief Whether RBLN_DUMMY_DEVICE is enabled (truthy spellings only; the rebel
 * runtime already rejects non-boolean values). Runtime-free; never throws.
 * The logical device count comes from RBLN_DEVICE_MAP, not this flag.
 */
C10_RBLN_API bool dummyDeviceEnabled();

} // namespace c10::rbln
