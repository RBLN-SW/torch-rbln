#pragma once

#include <c10/core/Device.h>
#include <c10/rbln/RBLNMacros.h>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace c10::rbln {

/**
 * @brief Maps each torch device to its NPU: rbln:N is the N-th NPU that RBLN_DEVICES (or its
 * alias RBLN_VISIBLE_DEVICES) names, in its order, or the N-th NPU of the host without it.
 *
 * A torch device is exactly one NPU. RBLN_NPUS_PER_DEVICE and RBLN_DEVICE_MAP, which grouped
 * several NPUs into one device on the v1 runtime, are rejected rather than ignored.
 */
class C10_RBLN_API DeviceMappingManager {
 public:
  /**
   * @brief Get the singleton instance of DeviceMappingManager.
   * @return Reference to the singleton instance.
   */
  static DeviceMappingManager& getInstance();

  /**
   * @brief The NPUs the process sees, by system number: those of this host that
   * RBLN_DEVICES (or its alias RBLN_VISIBLE_DEVICES) names, in its order, or every NPU
   * without it. Read from the environment now; plans nothing and claims no NPU.
   */
  static std::vector<uint32_t> visibleNpus();

  /**
   * @brief The NPUs the process sees: those the plan froze once it left the open state,
   * else visibleNpus().
   */
  std::vector<uint32_t> seenNpus() const;

  // Initialization is two stages:
  //
  //   plan   Read RBLN_DEVICES (alias RBLN_VISIBLE_DEVICES), reject the grouping variables,
  //          and list the NPU of each device. Claims no NPU. Run by availability and
  //          enumeration queries.
  //   commit Opens every planned NPU, which holds a context on it. Deferred to the first
  //          actual device use.
  //
  // torch calls is_available()/device_count() from paths that never asked for an NPU, so
  // claiming hardware there takes NPUs from co-tenants;
  // ATen/detail/AcceleratorHooksInterface.h: isAvailable() "should NOT initialize the
  // context on any device".
  //
  // Commit also freezes the NPUs RBLN_DEVICES lets the process see, so a launcher may still
  // assign RBLN_DEVICES after import -- fork()ed workers included -- until the first device use.

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
   * @brief Claim the planned devices with the runtime, and freeze the plan.
   *
   * Idempotent. Called from to_device_id(), the shared precursor to every
   * device-touching runtime call, so the claim happens exactly when the process first
   * commits to using a device. Raises if the plan is invalid or an NPU cannot be opened.
   */
  void commit();

  /**
   * @brief Whether a commit failed part-way, leaving the process unable to use a device.
   *
   * Nothrow, for the availability predicate: the plan still reports its device count, but
   * every later device use rethrows the stored error, so the backend is not usable.
   */
  bool hasFailedCommit() const noexcept;

  /**
   * @brief Whether the planned devices have been claimed with the runtime.
   *
   * Once committed the plan is frozen: later RBLN_* environment edits are ignored
   * rather than silently changing the mapping under live allocations.
   */
  bool isCommitted() const;

  /**
   * @brief The number of devices (rbln:0 .. rbln:N-1).
   */
  c10::DeviceIndex getDeviceCount() const {
    ensurePlanned();
    return static_cast<c10::DeviceIndex>(npus_.size());
  }

  /**
   * @brief The NPU behind each device, in device order, by system number. Under
   * RBLN_DUMMY_DEVICE no NPU backs them, and the numbers are those RBLN_DEVICES names.
   */
  std::vector<uint32_t> npus() const {
    ensurePlanned();
    return npus_;
  }

  /**
   * @brief The system number of the NPU behind rbln:device_index.
   */
  uint32_t npu(c10::DeviceIndex device_index) const;

  DeviceMappingManager(const DeviceMappingManager&) = delete;
  DeviceMappingManager& operator=(const DeviceMappingManager&) = delete;

 private:
  DeviceMappingManager();
  ~DeviceMappingManager() = default;

  /**
   * @brief Build the plan if it is missing or stale. Idempotent, and serialized on
   * plan_mutex_; see initialize() for the concurrent-environment caveat.
   *
   * A failed plan is remembered (not retried): rethrowing a stored error keeps a
   * rejected RBLN_* config from re-running validation on every query. Until commit()
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
   * @brief List the NPU of each device into npus_. Caller holds plan_mutex_.
   */
  void planNpus() const;

  /**
   * @brief The RBLN_* environment the plan depends on, as a comparable string.
   */
  static std::string envSignature();

  // Guards every member below. The plan is a lazy cache rebuilt on demand, hence
  // `mutable` on state a const query may refresh.
  mutable std::mutex plan_mutex_;
  mutable bool planned_ = false;
  // Both non-Open states freeze the plan for good; Failed means commit() threw part-way and
  // the NPUs it opened cannot be released. Atomic because a frozen plan can never change, so
  // the allocation path's queries read it without plan_mutex_. Written under the lock;
  // release/acquire pairs with those writes.
  enum class PlanState : std::uint8_t { Open, Committed, Failed };
  mutable std::atomic<PlanState> plan_state_{PlanState::Open};
  std::string commit_error_; // Failed: the error every later commit() rethrows
  mutable std::string plan_signature_; // envSignature() the current plan was built from
  mutable std::string plan_error_; // non-empty: the plan is invalid, rethrown on query

  mutable std::vector<uint32_t> npus_; // the NPU of each device, by system number
};

/** Invoked once after the device mapping singleton exists (e.g. torch_rbln._C registers a Python logger). */
using RblnDeviceMappingInitializedCallback = void (*)();

C10_RBLN_API void register_rbln_device_mapping_initialized_callback(RblnDeviceMappingInitializedCallback cb);

/**
 * @brief Whether RBLN_DUMMY_DEVICE is enabled (truthy spellings only; the rebel
 * runtime already rejects non-boolean values). Runtime-free; never throws.
 */
C10_RBLN_API bool dummyDeviceEnabled();

} // namespace c10::rbln
