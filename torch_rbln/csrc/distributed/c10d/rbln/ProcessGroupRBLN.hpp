/**
 * @file ProcessGroupRBLN.hpp
 * @brief ProcessGroupRBLN implementation for RBLN distributed computing
 *
 * This file provides the ProcessGroupRBLN class which implements distributed
 * communication operations using the RBLN (Rebellions Neural Processing Unit)
 * backend. It supports collective operations like allreduce, broadcast, scatter,
 * and point-to-point communication through RCCL (RBLN Collective Communication Library).
 */

#pragma once

// Standard library includes
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// PyTorch distributed includes
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>

// ATen includes
#include <ATen/ThreadLocalState.h>

// Forward declarations
namespace rbln {
class Rccl;
}
struct rccl_unique_id;

namespace c10d {

// ============================================================================
// Constants and Configuration
// ============================================================================

/// @brief Backend name for RBLN ProcessGroup
constexpr const char* RBLN_BACKEND_NAME = "rbln-ccl";

/// @brief Default timeout for RCCL operations
constexpr const std::chrono::milliseconds RCCL_DEFAULT_TIMEOUT{5000};

/// @brief Default number of worker threads for async operations
constexpr int DEFAULT_NUM_WORKERS = 1;

// ============================================================================
// ProcessGroupRBLN Class
// ============================================================================

/**
 * @brief ProcessGroupRBLN implements RBLN bindings for c10d distributed computing
 *
 * This class provides distributed communication operations using the RBLN NPU
 * backend. It supports collective operations (allreduce, broadcast, scatter)
 * and point-to-point communication (send, recv) through RCCL.
 *
 * The class uses an asynchronous work queue pattern where operations are
 * enqueued and processed by worker threads to avoid blocking the main thread.
 */
class TORCH_API ProcessGroupRBLN : public Backend {
 public:
  // ============================================================================
  // Constructor and Destructor
  // ============================================================================

  /**
   * @brief Construct a new ProcessGroupRBLN object
   * @param store Shared store for coordination (used for rccl_unique_id broadcast in autoport init)
   * @param rank Process rank in the group
   * @param size Total number of processes in the group
   * @param group_id Group identifier (used for store keys, e.g. rbln_rccl_uid_ + group_id)
   * @param global_ranks_in_group Global ranks of processes in this group
   * @param options Backend options including timeout settings
   * @param glooBackend Optional Gloo backend for non-float16 allreduce/reduce_scatter fallback
   */
  explicit ProcessGroupRBLN(
      const c10::intrusive_ptr<Store>& store,
      int rank,
      int size,
      int group_id = -1,
      const std::vector<int>& global_ranks_in_group = std::vector<int>(),
      const c10::intrusive_ptr<Options>& options =
          c10::make_intrusive<Options>(RBLN_BACKEND_NAME, RCCL_DEFAULT_TIMEOUT),
      c10::intrusive_ptr<Backend> glooBackend = c10::intrusive_ptr<Backend>());

  /**
   * @brief Destroy the ProcessGroupRBLN object
   *
   * Ensures all pending work is completed and worker threads are properly terminated.
   */
  virtual ~ProcessGroupRBLN() override;

  // ============================================================================
  // Collective Communication Operations
  // ============================================================================

  /**
   * @brief Broadcast tensors from root rank to all other ranks
   * @param tensors Tensors to broadcast (in-place operation)
   * @param opts Broadcast options including root rank and tensor selection
   * @return Work object for tracking operation completion
   */
  c10::intrusive_ptr<Work> broadcast(
      std::vector<at::Tensor>& tensors,
      const BroadcastOptions& opts = BroadcastOptions()) override;

  /**
   * @brief Scatter tensors from root rank to all ranks
   * @param outputs Output tensors for each rank
   * @param inputs Input tensors (only used on root rank)
   * @param opts Scatter options including root rank
   * @return Work object for tracking operation completion
   */
  c10::intrusive_ptr<Work> scatter(
      std::vector<at::Tensor>& outputs,
      std::vector<std::vector<at::Tensor>>& inputs,
      const ScatterOptions& opts = ScatterOptions()) override;

  /**
   * @brief Gather tensors from all ranks into lists
   * @param outputs Output tensor lists for each input tensor (outputs.size() = inputs.size())
   * @param inputs Input tensors to be gathered from current rank
   * @param opts Allgather options
   * @return Work object for tracking operation completion
   *
   * @note Each element in outputs[i] contains the gathered tensors from all ranks
   *       for the corresponding input tensor inputs[i]. Each output list size must be
   *       world_size (outputs[i].size() = world_size).
   */
  c10::intrusive_ptr<Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputs,
      std::vector<at::Tensor>& inputs,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  /**
   * @brief Single tensor all-gather operation (base implementation)
   * @param output_tensor Output tensor to hold gathered data from all ranks
   * @param input_tensor Input tensor to be gathered
   * @param opts Allgather options
   * @return Work object for tracking operation completion
   */
  c10::intrusive_ptr<Work> _allgather_base(
      at::Tensor& output_tensor,
      at::Tensor& input_tensor,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  /**
   * @brief Coalesced all-gather: gather multiple input tensors into flat output tensors
   * @param outputTensors Output tensors (outputTensors[i].numel() == inputTensors[i].numel() * world_size)
   * @param inputTensors Input tensors to gather from all ranks
   * @param opts Allgather options
   * @return Work object for tracking operation completion
   */
  c10::intrusive_ptr<Work> allgather_into_tensor_coalesced(
      std::vector<at::Tensor>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  /**
   * @brief All-reduce operation across all ranks
   * @param tensors Tensors to reduce (in-place operation)
   * @param opts Allreduce options including reduction operation type
   * @return Work object for tracking operation completion
   */
  c10::intrusive_ptr<Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override;

  /**
   * @brief Reduce-scatter operation across all ranks
   * @param outputs Output tensors (one per rank)
   * @param inputs Input tensor lists (inputs[0] must have world_size tensors)
   * @param opts ReduceScatter options including reduction operation type
   * @return Work object for tracking operation completion
   */
  c10::intrusive_ptr<Work> reduce_scatter(
      std::vector<at::Tensor>& outputs,
      std::vector<std::vector<at::Tensor>>& inputs,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

  /**
   * @brief Single tensor reduce-scatter operation (base implementation)
   * @param outputTensor Output tensor to hold reduced and scattered data for current rank
   * @param inputTensor Input tensor containing data from all ranks (size: output_size * world_size)
   * @param opts ReduceScatter options including reduction operation type
   * @return Work object for tracking operation completion
   */
  c10::intrusive_ptr<Work> _reduce_scatter_base(
      at::Tensor& outputTensor,
      at::Tensor& inputTensor,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

  // ============================================================================
  // Point-to-Point Communication Operations
  // ============================================================================

  /**
   * @brief Send tensors to a specific destination rank
   * @param tensors Tensors to send
   * @param dstRank Destination rank
   * @param tag Message tag
   * @return Work object for tracking operation completion
   */
  c10::intrusive_ptr<Work> send(std::vector<at::Tensor>& tensors, int dstRank, int tag) override;

  /**
   * @brief Receive tensors from a specific source rank
   * @param tensors Tensors to receive into
   * @param srcRank Source rank
   * @param tag Message tag
   * @return Work object for tracking operation completion
   */
  c10::intrusive_ptr<Work> recv(std::vector<at::Tensor>& tensors, int srcRank, int tag) override;

  // ============================================================================
  // Synchronization Operations
  // ============================================================================

  /**
   * @brief Barrier synchronization across all ranks
   * @param opts Barrier options
   * @return Work object for tracking operation completion
   */
  c10::intrusive_ptr<Work> barrier(const BarrierOptions& opts = BarrierOptions()) override;

  // ============================================================================
  // Public Accessors
  // ============================================================================

  /**
   * @brief Get the backend name
   * @return Backend name string
   */
  const std::string getBackendName() const noexcept {
    return backendName_;
  }

  // ============================================================================
  // Sequence Number Management
  // ============================================================================

  /**
   * @brief Set sequence number for the group
   *
   * Agrees on an initial sequence number for the whole group by having rank 0
   * create it and broadcast it to other ranks using the store.
   */
  void setSequenceNumberForGroup() override;

  /**
   * @brief Get the current sequence number for the group
   * @return Current sequence number
   *
   * Retrieves the current sequence number for the whole group, which should be
   * in sync. If the returned number is not consistent across the group, it
   * may indicate that there is some sort of collective desynchronization.
   */
  uint64_t getSequenceNumberForGroup() override;

  /**
   * @brief Broadcast rccl_unique_id via TCP Store (rank 0 sets, others get).
   *
   * Used by the autoport init path (RCCL_PORT_GEN set). Call after PrepareContextAndExportMem on all ranks;
   * rank 0 generates the id via this group's rccl_->GetUniqueIdForBroadcast and sets it in the store;
   * other ranks get from the store. Uses this group's group_id_ for the store key. Result is written to
   * rcclID (no copy on return).
   *
   * @param rcclID Output: on rank 0 filled by GetUniqueIdForBroadcast; on others filled from the store
   */
  void broadcastUniqueId(struct rccl_unique_id* rcclID);

 private:
  // ============================================================================
  // Private Helper Methods
  // ============================================================================

  /**
   * @brief Generate next unique tag for operations
   * @return Next tag value
   */
  uint32_t nextTag() noexcept;

  /**
   * @brief Generate next unique sequence number for operations
   * @return Next sequence number
   */
  uint64_t nextSeq() noexcept;

  /**
   * @brief Enqueue work for asynchronous execution
   * @param work Work object to enqueue
   */
  void enqueue(c10::intrusive_ptr<Work> work);

  /**
   * @brief Execute work synchronously or enqueue for async execution based on sync_mode_
   * @param work Work object to execute or enqueue (must be RBLNWork type)
   */
  void enqueueOrExecute(c10::intrusive_ptr<Work> work);

  /**
   * @brief Worker thread main loop
   * @param workerIndex Index of the worker thread
   */
  void runLoop(int workerIndex);

  // ============================================================================
  // Work Queue Management
  // ============================================================================

  /// @brief Queue of pending work items
  std::deque<c10::intrusive_ptr<Work>> workQueue_;

  /// @brief Currently executing work items (one per worker thread)
  std::vector<c10::intrusive_ptr<Work>> workInProgress_;

  /// @brief Mutex for protecting work queue and in-progress work
  std::mutex workMutex_;

  /// @brief Condition variable for notifying when work is available
  std::condition_variable workProduceCV_;

  /// @brief Condition variable for notifying when work is completed
  std::condition_variable workConsumeCV_;

  /// @brief Flag to signal worker threads to stop
  bool stop_{false};

  /// @brief Worker threads for processing async work
  std::vector<std::thread> threads_;

  // ============================================================================
  // RBLN Backend State
  // ============================================================================

  /// @brief Shared store for coordination (used for rccl_unique_id broadcast).
  c10::intrusive_ptr<Store> store_;

  /// @brief Group ID for this process group (used for store key).
  int group_id_;

  /// @brief RCCL instance for collective communication
  std::shared_ptr<::rbln::Rccl> rccl_;

  /// @brief Gloo backend for FP32 operations
  c10::intrusive_ptr<Backend> glooBackend_;

  /// @brief Current tag counter for operation identification
  uint32_t tag_{0};

  /// @brief Current sequence number for operation ordering
  uint64_t seq_{0};

  /// @brief Backend name identifier
  std::string backendName_{RBLN_BACKEND_NAME};

  /// @brief List of global ranks in group
  std::vector<int> global_ranks_in_group_;

  /// @brief Global rank of the current process
  int global_rank_;

  /// @brief Device ID mapped from global rank to local rank in default group
  int device_id_;

  /// @brief Flag to control sync/async execution mode
  /// If true, work is executed synchronously (default)
  /// If false, work is enqueued for async execution
  bool sync_mode_{true};
};

} // namespace c10d
