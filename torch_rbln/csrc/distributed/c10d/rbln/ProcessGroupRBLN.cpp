/**
 * @file ProcessGroupRBLN.cpp
 * @brief Implementation of ProcessGroupRBLN for RBLN distributed computing
 *
 * This file contains the implementation of ProcessGroupRBLN class and its
 * associated helper classes for distributed communication operations using
 * the RBLN NPU backend and RCCL library.
 */

// Standard library includes
#include <cstdlib>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>

// C10 includes
#include <c10/rbln/RBLNFunctions.h>
#include <c10/rbln/RBLNLogging.h>
#include <c10/util/error.h>

// ATen includes
#include <ATen/ThreadLocalState.h>
#include <ATen/record_function.h>

// PyTorch distributed includes
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

// RBLN includes
#include <aten/src/ATen/native/rbln/RBLNTensorUtils.h>
#include <rebel/runtime/distributed/rbln_rccl.h>
#include <torch/torch.h>
#include <torch_rbln/csrc/distributed/c10d/rbln/ProcessGroupRBLN.hpp>
#include <torch_rbln/csrc/distributed/c10d/rbln/RcclUniqueIdForC10d.hpp>

#include <cstring>

namespace c10d {

// ============================================================================
// Anonymous Namespace for Helper Functions and Classes
// ============================================================================

namespace {

// ============================================================================
// RCCL Alignment Constants
// ============================================================================

// RCCL alignment constraints vary by operation type
constexpr size_t RCCL_ALLGATHER_ALIGNMENT = 128ULL; // AllGather: 128 bytes
constexpr size_t RCCL_ALLREDUCE_ALIGNMENT = 512ULL; // AllReduce: 512 bytes
constexpr size_t RCCL_REDUCE_SCATTER_ALIGNMENT = RCCL_ALLREDUCE_ALIGNMENT;

// AllReduce size limit: data_size / world_size <= 1MB (temporary driver limitation)
constexpr size_t RCCL_ALLREDUCE_MAX_BYTES_PER_RANK = 1ULL * 1024ULL * 1024ULL;
// ReduceScatter size limit: aligned_data_size * world_size <= 32MB (driver 3.0.0_rc2 limitation)
constexpr size_t RCCL_REDUCE_SCATTER_MAX_BYTES_PER_WORLD = 32ULL * 1024ULL * 1024ULL;
constexpr size_t RCCL_MAX_BYTES_PER_REDUCE_SCATTER_OP = 2ULL * 1024ULL * 1024ULL;
constexpr size_t RCCL_MAX_COMMAND_BUFFER_SUB_COMMANDS_COUNT = 15ULL;
// AllGather size limit: output size <= 1GB (can be performed in a single allgather operation)
constexpr size_t RCCL_ALLGATHER_MAX_OUTPUT_BYTES = 1ULL * 1024ULL * 1024ULL * 1024ULL;

/**
 * @brief Get element size for RCCL operations based on dtype
 * @param dtype The RBLN dtype to get element size for
 * @return Element size in bytes
 */
size_t getRcclElementSize(::rbln::DataType dtype) {
  switch (dtype) {
    case ::rbln::DataType::Float16:
      return 2;
    default:
      RBLN_CHECK(false, "Unsupported dtype for RCCL element size calculation");
  }
  return 0;
}

// ============================================================================
// Type Conversion Utilities
// ============================================================================

/**
 * @brief Convert PyTorch ReduceOp to RCCL RcclReduceOp
 * @param reduceOp PyTorch reduction operation type
 * @return Corresponding RCCL reduction operation type
 * @throws c10::Error for unsupported operations
 */
::rbln::Rccl::RcclReduceOp convertReduceOp(const ReduceOp& reduceOp) {
  switch (reduceOp) {
    case ReduceOp::SUM:
      return ::rbln::Rccl::RcclReduceOp::RCCL_REDUCE_OP_SUM;
    case ReduceOp::PRODUCT:
      return ::rbln::Rccl::RcclReduceOp::RCCL_REDUCE_OP_MUL;
    case ReduceOp::MIN:
      return ::rbln::Rccl::RcclReduceOp::RCCL_REDUCE_OP_MIN;
    case ReduceOp::MAX:
      return ::rbln::Rccl::RcclReduceOp::RCCL_REDUCE_OP_MAX;
    case ReduceOp::AVG:
      return ::rbln::Rccl::RcclReduceOp::RCCL_REDUCE_OP_AVG;
    case ReduceOp::BAND:
      RBLN_CHECK(false, "Bitwise operations (BAND) are not supported by RCCL");
      break;
    case ReduceOp::BOR:
      RBLN_CHECK(false, "Bitwise operations (BOR) are not supported by RCCL");
      break;
    case ReduceOp::BXOR:
      RBLN_CHECK(false, "Bitwise operations (BXOR) are not supported by RCCL");
      break;
    case ReduceOp::PREMUL_SUM:
      RBLN_CHECK(false, "PREMUL_SUM is not supported by RCCL");
      break;
    case ReduceOp::UNUSED:
    default:
      RBLN_CHECK(false, "Unsupported reduce operation for RCCL");
  }
  // This should never be reached due to the checks above
  return ::rbln::Rccl::RcclReduceOp::RCCL_REDUCE_OP_SUM;
}

/**
 * @brief Convert PyTorch ScalarType to RCCL DataType
 * @param scalarType PyTorch scalar type
 * @return Corresponding RCCL data type
 * @throws c10::Error for unsupported scalar types
 *
 * note RCCL only supports INT8 and CF16 types. For types larger than
 * the native RCCL type (e.g., int64 -> INT8), callers must adjust the
 * count parameter to be in bytes using getRcclCount().
 */
::rbln::Rccl::RcclDataType convertDataType(const at::ScalarType& scalarType) {
  switch (scalarType) {
    case at::ScalarType::Float:
    case at::ScalarType::Double:
    case at::ScalarType::Half:
    case at::ScalarType::BFloat16:
      return ::rbln::Rccl::RcclDataType::RBLN_CUSTOM_FP16_TYPE;
    case at::ScalarType::Char:
    case at::ScalarType::Byte:
    case at::ScalarType::Bool:
    case at::ScalarType::Short:
    case at::ScalarType::Int:
    case at::ScalarType::Long:
      return ::rbln::Rccl::RcclDataType::RCCL_INT8_TYPE;
    default:
      RBLN_CHECK(false, "Unsupported scalar type for RCCL: {}", c10::str(scalarType));
  }
  // This should never be reached due to the check above
  return ::rbln::Rccl::RcclDataType::RCCL_INT8_TYPE;
}

/**
 * @brief Calculate RCCL count based on data type mapping
 * @param numel Number of elements in the tensor
 * @param scalarType PyTorch scalar type
 * @return Count to pass to RCCL operations
 *

 * CL uses INT8 for integer types and CF16 for floating types.
 * This function adjusts the count accordingly:
 * - For INT8-mapped types: count = numel * element_size (bytes)
 * - For CF16-mapped types: count = numel * element_size / 2 (half-precision elements)
 */
int64_t getRcclCount(int64_t numel, const at::ScalarType& scalarType) {
  size_t elem_size = c10::elementSize(scalarType);
  switch (scalarType) {
    case at::ScalarType::Float:
    case at::ScalarType::Double:
    case at::ScalarType::Half:
    case at::ScalarType::BFloat16:
      // CUSTOM_FLOAT16: 2 bytes per element
      return numel * static_cast<int64_t>(elem_size) / 2;
    case at::ScalarType::Char:
    case at::ScalarType::Byte:
    case at::ScalarType::Bool:
    case at::ScalarType::Short:
    case at::ScalarType::Int:
    case at::ScalarType::Long:
      // INT8: 1 byte per element - use total byte count
      return numel * static_cast<int64_t>(elem_size);
    default:
      RBLN_CHECK(false, "Unsupported scalar type for RCCL count: {}", c10::str(scalarType));
  }
  return numel;
}

/**
 * @brief Check if a size is aligned to a given alignment
 * @param size The size to check
 * @param alignment The alignment to check against
 * @return True if the size is aligned, false otherwise
 */
bool isAlignedTo(size_t size, size_t alignment) {
  return (size % alignment) == 0;
}

c10::intrusive_ptr<c10::ivalue::Future> createFutureAsOutput(
    const std::vector<std::vector<at::Tensor>>& outputTensors) {
  if (outputTensors.size() > 1) {
    return c10::make_intrusive<c10::ivalue::Future>(
        c10::ListType::create(c10::ListType::create(c10::TensorType::get())));
  }
  return c10::make_intrusive<c10::ivalue::Future>(c10::ListType::create(c10::TensorType::get()));
}

void returnFutureWithOutput(
    c10::intrusive_ptr<c10::ivalue::Future>& future,
    const std::vector<std::vector<at::Tensor>>& outputTensors) {
  if (outputTensors.empty()) {
    future->markCompleted(c10::IValue(std::vector<at::Tensor>()));
    return;
  }
  if (outputTensors.size() > 1) {
    future->markCompleted(c10::IValue(outputTensors));
    return;
  }
  future->markCompleted(c10::IValue(outputTensors[0]));
}

// ============================================================================
// V-Memory Configuration Functions for RCCL
// ============================================================================

/**
 * @brief Get memory info for a tensor, returns nullopt if not tracked
 *
 * Common helper to retrieve VMemoryManager metadata for RCCL operations.
 * Returns nullopt for zero-ptr or untracked tensors (e.g., CPU tensors).
 *
 * @param tensor The tensor to query
 * @return Optional MemoryInfo, or nullopt if tensor is not device-managed
 */
std::optional<rbln::MemoryInfo> getRcclMemoryInfo(const at::Tensor& tensor) {
  auto vmem = reinterpret_cast<uint64_t>(tensor.data_ptr());
  if (vmem == 0) {
    return std::nullopt;
  }
  rbln::MemoryInfo mem_info;
  if (rbln::rbln_get_memory_info(vmem, mem_info) != RBLNRetCode_SUCCESS) {
    return std::nullopt;
  }
  return mem_info;
}

/**
 * @brief Prepare tensor for RCCL byte-copy operations (no dtype transform)
 *
 * Uses SINGLE_DEVICE_NO_TRANSFORM mode for raw byte transfers.
 * Suitable for: AllGather, Broadcast, Send, Recv, Scatter.
 *
 * @param tensor The tensor to prepare
 */
void ensureRcclRawMemory(const at::Tensor& tensor) {
  auto mem_info = getRcclMemoryInfo(tensor);
  if (!mem_info) {
    return;
  }
  RBLN_CHECK(
      rbln::rbln_set_raw_memory_alloc(mem_info->key_vaddr, tensor.storage().nbytes()) == RBLNRetCode_SUCCESS,
      "Failed to allocate RCCL raw memory");
}

/**
 * @brief Prepare tensor for RCCL reduce operations (CustomFloat16 transform)
 *
 * Uses WITH_TRANSFORM mode with Float16 -> CustomFloat16 conversion.
 * This enables NPU arithmetic and supports offset access for chunked operations.
 * Suitable for: AllReduce, ReduceScatter.
 *
 * @param tensor The tensor to prepare
 */
void ensureRcclReduceMemory(const at::Tensor& tensor) {
  auto mem_info = getRcclMemoryInfo(tensor);
  if (!mem_info) {
    return;
  }

  auto vmem = reinterpret_cast<uint64_t>(tensor.data_ptr());
  const auto rbln_dtype = c10::rbln::to_rbln_dtype(tensor.scalar_type());

  // For view tensors, use storage size to cover entire underlying memory
  std::vector<int64_t> shape;
  if (vmem != mem_info->key_vaddr) {
    auto storage_size = tensor.storage().nbytes();
    shape = {static_cast<int64_t>(storage_size / tensor.element_size())};
  } else {
    // storage base equals tensor address: require contiguous layout
    RBLN_CHECK(
        tensor.is_contiguous(),
        "ensureRcclReduceMemory: tensor must be contiguous when storage_offset == 0; call .flatten()");
    shape = std::vector<int64_t>(tensor.sizes().begin(), tensor.sizes().end());
  }

  RBLN_CHECK(
      rbln::rbln_set_memory_info(mem_info->key_vaddr, rbln_dtype, rbln::DataType::CustomFloat16, shape) ==
          RBLNRetCode_SUCCESS,
      "Failed to set RCCL reduce memory layout");
}

/**
 * @brief Register contiguous RCCL raw buffer with shape expanded by world_size.
 *
 * Used when allgather output tensors are contiguous in memory (one buffer per rank).
 * Registers the buffer at key_vaddr with shape [world_size, ...tensor_shape] so the
 * runtime treats it as a single tensor holding all ranks' data. No dtype transform.
 *
 * @param tensors Vector of contiguous output tensors (e.g. output)
 */
void ensureContiguousRcclRawMemory(const std::vector<at::Tensor>& tensors) {
  TORCH_CHECK(!tensors.empty(), "ensureContiguousRcclRawMemory: tensors must be non-empty");
  RBLN_LOG_DEBUG("ensureContiguousRcclRawMemory: tensors.size={}", tensors.size());
  for (const auto& tensor : tensors) {
    auto mem_info = getRcclMemoryInfo(tensor);
    TORCH_CHECK(mem_info, "ensureContiguousRcclRawMemory: tensor must be tracked by VMemoryManager");
    TORCH_CHECK(
        rbln::rbln_set_raw_memory_alloc(mem_info->key_vaddr, tensor.storage().nbytes()) == RBLNRetCode_SUCCESS,
        "Failed to allocate RCCL raw memory");
  }
}

/**
 * @brief Register contiguous RCCL reduce buffer with shape expanded by world_size.
 *
 * Used when inputs_[i] are contiguous in memory (one per rank). Registers the buffer
 * at key_vaddr with shape [world_size, ...tensor_shape] so the runtime treats it
 * as a single tensor holding all ranks' data.
 *
 * @param tensors Vector of contiguous tensors (e.g. inputs_[i])
 */
void ensureContiguousRcclReduceMemory(const std::vector<at::Tensor>& tensors) {
  TORCH_CHECK(!tensors.empty(), "ensureContiguousRcclReduceMemory: tensors must be non-empty");
  RBLN_LOG_DEBUG("ensureContiguousRcclReduceMemory: tensors.size={}", tensors.size());
  for (const auto& tensor : tensors) {
    auto mem_info = getRcclMemoryInfo(tensor);
    TORCH_CHECK(mem_info, "ensureContiguousRcclReduceMemory: tensor must be tracked by VMemoryManager");
    const auto rbln_dtype = c10::rbln::to_rbln_dtype(tensor.scalar_type());
    std::vector<int64_t> shape = std::vector<int64_t>(tensor.sizes().begin(), tensor.sizes().end());
    TORCH_CHECK(
        rbln::rbln_set_memory_info(mem_info->key_vaddr, rbln_dtype, rbln::DataType::CustomFloat16, shape) ==
            RBLNRetCode_SUCCESS,
        "Failed to set RCCL reduce memory layout");
  }
}

// ============================================================================
// RBLNWork Base Class
// ============================================================================

/**
 * @brief Base class for asynchronous work operations
 *
 * This class provides the foundation for all asynchronous work operations
 * in ProcessGroupRBLN. It handles work execution, completion tracking,
 * and result management.
 */
class RBLNWork : public Work {
 public:
  explicit RBLNWork(
      std::vector<std::vector<at::Tensor>> outputTensors,
      OpType opType,
      uint64_t seq,
      const char* profilingTitle = nullptr,
      const std::optional<std::vector<at::Tensor>>& inputTensors = std::nullopt);

  ~RBLNWork() override = default;

  static void execute(const c10::intrusive_ptr<RBLNWork>& work);

  virtual void run() = 0;

  std::vector<at::Tensor> result() override;

  c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;
  uint64_t getSequencenumber() const override;

  inline at::ThreadLocalState getTLS() const {
    return tls_;
  }

 protected:
  friend class ProcessGroupRBLN;

 private:
  void finishWorkRBLN();
  void finishWorkRBLNError(const std::exception_ptr& eptr);
  inline void recordRBLNWorkProfilingInfo(
      const char* profilingTitle,
      const std::optional<std::vector<at::Tensor>>& inputTensors);

  const std::vector<std::vector<at::Tensor>> outputTensors_;
  c10::intrusive_ptr<at::ivalue::Future> future_;
  std::function<void()> recordFunctionBeforeCallback_;
  uint64_t seq_;
  at::ThreadLocalState tls_;
};

RBLNWork::RBLNWork(
    std::vector<std::vector<at::Tensor>> outputTensors,
    OpType opType,
    uint64_t seq,
    const char* profilingTitle,
    const std::optional<std::vector<at::Tensor>>& inputTensors)
    : Work(-1, opType, profilingTitle, inputTensors),
      outputTensors_(std::move(outputTensors)),
      future_(createFutureAsOutput(outputTensors_)),
      recordFunctionBeforeCallback_(nullptr),
      seq_(seq) {
  recordRBLNWorkProfilingInfo(profilingTitle, inputTensors);
}

void RBLNWork::recordRBLNWorkProfilingInfo(
    const char* profilingTitle,
    const std::optional<std::vector<at::Tensor>>& inputTensors) {
  if (profilingTitle != nullptr) {
    auto recordingFunction = std::make_shared<at::RecordFunction>(at::RecordScope::USER_SCOPE);
    if (recordingFunction->isActive()) {
      recordingFunction->_setAsync();
      std::vector<c10::IValue> inputs;
      if (inputTensors) {
        inputs.reserve(inputTensors->size());
        for (const auto& tensor : *inputTensors) {
          inputs.emplace_back(tensor);
        }
      }
      recordingFunction->before(profilingTitle, c10::ArrayRef<const c10::IValue>(inputs.data(), inputs.size()));
      std::function<void()> end_handler = [recordingFunction]() { recordingFunction->end(); };
      recordFunctionBeforeCallback_ = at::wrapPropagateTLSState(end_handler);
    }
  }
}

// static
void RBLNWork::execute(const c10::intrusive_ptr<RBLNWork>& work) {
  if (work->recordFunctionBeforeCallback_) {
    work->recordFunctionBeforeCallback_();
  }
  try {
    work->run();
  } catch (...) {
    work->finishWorkRBLNError(std::current_exception());
    return;
  }

  // FIXME: We need to call it here since Future completion requires all
  // the work to be synchronized to RBLN.
  work->synchronize();
  work->finishWorkRBLN();
}

std::vector<at::Tensor> RBLNWork::result() {
  RBLN_CHECK(isCompleted(), "Work needs to be completed before calling result(). Should call wait() before result().");
  RBLN_CHECK(outputTensors_.size() <= 1, "work result does not support list of lists, use .getFuture() and value()");
  return outputTensors_.empty() ? std::vector<at::Tensor>() : outputTensors_.at(0);
}

c10::intrusive_ptr<c10::ivalue::Future> RBLNWork::getFuture() {
  return future_;
}

uint64_t RBLNWork::getSequencenumber() const {
  return seq_;
}

void RBLNWork::finishWorkRBLN() {
  if (future_) {
    returnFutureWithOutput(future_, outputTensors_);
  }
  finish();
}

void RBLNWork::finishWorkRBLNError(const std::exception_ptr& eptr) {
  if (future_) {
    future_->setError(eptr);
  }
  finish(eptr);
}

// ============================================================================
// Specific RBLNWork Implementations
// ============================================================================

/// Store key prefix for rccl_unique_id broadcast (default group).
constexpr const char* STORE_KEY_PREFIX_UNIQUE_ID = "rbln_rccl_uid_";

/**
 * @brief RBLNWork for RCCL initialization
 *
 * Initializes the RCCL communication library for multi-rank runs (size > 1).
 * Two paths are supported:
 * - **Autoport (RCCL_PORT_GEN set):** PrepareContextAndExportMem on all ranks,
 *   then rank 0 generates rccl_unique_id and broadcasts it via the TCP Store;
 *   all ranks call InitWithUniqueId with that id.
 * - **Default (RCCL_PORT_GEN unset):** Init() uses GetGlobalContext, ExportMem,
 *   and InitDefaultGroup (legacy single-context path).
 */
class InitRBLNWork : public RBLNWork {
 public:
  InitRBLNWork(
      ProcessGroupRBLN* pg,
      int rank,
      int size,
      int group_id,
      int device_id,
      const std::vector<int>& global_ranks_in_group,
      uint64_t seq,
      std::shared_ptr<::rbln::Rccl> rccl)
      : RBLNWork(std::vector<std::vector<at::Tensor>>{}, OpType::UNKNOWN, seq, "rbln:init", std::nullopt),
        pg_(pg),
        rank_(rank),
        size_(size),
        group_id_(group_id),
        device_id_(device_id),
        global_ranks_in_group_(global_ranks_in_group),
        rccl_(std::move(rccl)) {}
  ~InitRBLNWork() override = default;

  void run() override {
    RECORD_FUNCTION("rbln::init", std::vector<c10::IValue>());

    if (size_ == 1) {
      return;
    }

    RBLN_CHECK(rccl_ != nullptr, "InitRBLNWork: rccl_ is null");
    RBLN_CHECK(pg_ != nullptr, "InitRBLNWork: pg_ is null");

    RBLN_LOG_INFO("RCCL init rank={} size={} group_id={} device_id={}", rank_, size_, group_id_, device_id_);

    const char* port_gen_env = std::getenv("RCCL_PORT_GEN");
    const bool use_autoport = (port_gen_env != nullptr && port_gen_env[0] != 0);
    const bool is_default_group = global_ranks_in_group_.empty();

    if (use_autoport && is_default_group) {
      RBLNRetCode ret = rccl_->PrepareContextAndExportMem(device_id_);
      RBLN_CHECK(
          ret == RBLNRetCode_SUCCESS,
          "RCCL PrepareContextAndExportMem failed with error code: {} (rank={})",
          static_cast<int>(ret),
          rank_);
      struct rccl_unique_id rcclID = {};
      pg_->broadcastUniqueId(&rcclID);
      ret = rccl_->InitWithUniqueId(rank_, size_, group_id_, global_ranks_in_group_, device_id_, &rcclID);
      RBLN_CHECK(
          ret == RBLNRetCode_SUCCESS,
          "RCCL InitWithUniqueId failed with error code: {} (rank={})",
          static_cast<int>(ret),
          rank_);
    } else {
      RBLNRetCode ret = rccl_->Init(rank_, size_, group_id_, global_ranks_in_group_, device_id_);
      RBLN_CHECK(
          ret == RBLNRetCode_SUCCESS, "RCCL Init failed with error code: {} (rank={})", static_cast<int>(ret), rank_);
    }
    RBLN_LOG_INFO("RCCL init done rank={}", rank_);
  }

 private:
  ProcessGroupRBLN* pg_;
  int rank_;
  int size_;
  int group_id_;
  int device_id_;
  std::vector<int> global_ranks_in_group_;
  std::shared_ptr<::rbln::Rccl> rccl_;
};

/**
 * @brief RBLNWork for point-to-point send operations
 *
 * Handles sending tensors to a specific destination rank.
 */
class SendRBLNWork : public RBLNWork {
 public:
  SendRBLNWork(at::Tensor& tensor, int dstRank, uint32_t utag, uint64_t seq, std::shared_ptr<::rbln::Rccl> rccl)
      : RBLNWork(
            std::vector<std::vector<at::Tensor>>{},
            OpType::SEND,
            seq,
            "rbln:send",
            std::optional<std::vector<at::Tensor>>({tensor})),
        tensor_(tensor),
        dstRank_(dstRank),
        utag_(utag),
        rccl_(std::move(rccl)) {}

  ~SendRBLNWork() override = default;

  void run() override {
    RECORD_FUNCTION("rbln::send", std::vector<c10::IValue>({tensor_}));

    ensureRcclRawMemory(tensor_);
    auto ptr = tensor_.const_data_ptr();
    const auto numel = tensor_.numel();
    const auto elem_size = tensor_.element_size();
    const auto size = numel * elem_size;
    RBLN_LOG_DEBUG(
        "Sending tensor: numel={}, elem_size={}, size={}, rcclDataType={}, dstRank={}",
        numel,
        elem_size,
        size,
        static_cast<int>(convertDataType(tensor_.scalar_type())),
        dstRank_);
    RBLNRetCode ret =
        rccl_->Send((void*)ptr, size, ::rbln::Rccl::RcclDataType::RCCL_INT8_TYPE, dstRank_, nullptr); // NOLINT
    RBLN_CHECK(ret == RBLNRetCode_SUCCESS, "RCCL Send failed with error code: {}", static_cast<int>(ret));
  }

 private:
  at::Tensor tensor_;
  int dstRank_;
  uint32_t utag_;
  std::shared_ptr<::rbln::Rccl> rccl_;
};

/**
 * @brief RBLNWork for point-to-point receive operations
 *
 * Handles receiving tensors from a specific source rank.
 */
class RecvRBLNWork : public RBLNWork {
 public:
  RecvRBLNWork(at::Tensor& tensor, int srcRank, uint32_t utag, uint64_t seq, std::shared_ptr<::rbln::Rccl> rccl)
      : RBLNWork(
            std::vector<std::vector<at::Tensor>>{},
            OpType::RECV,
            seq,
            "rbln:recv",
            std::optional<std::vector<at::Tensor>>({tensor})),
        tensor_(tensor),
        srcRank_(srcRank),
        utag_(utag),
        rccl_(std::move(rccl)) {}

  ~RecvRBLNWork() override = default;

  void run() override {
    RECORD_FUNCTION("rbln::recv", std::vector<c10::IValue>({tensor_}));

    ensureRcclRawMemory(tensor_);
    auto ptr = tensor_.mutable_data_ptr();
    const auto numel = tensor_.numel();
    const auto elem_size = tensor_.element_size();
    const auto size = numel * elem_size;
    RBLN_LOG_DEBUG(
        "Receiving tensor: numel={}, elem_size={}, size={}, rcclDataType={}, srcRank={}",
        numel,
        elem_size,
        size,
        static_cast<int>(convertDataType(tensor_.scalar_type())),
        srcRank_);
    RBLNRetCode ret =
        rccl_->Recv((void*)ptr, size, ::rbln::Rccl::RcclDataType::RCCL_INT8_TYPE, srcRank_, nullptr); // NOLINT
    RBLN_CHECK(ret == RBLNRetCode_SUCCESS, "RCCL Recv failed with error code: {}", static_cast<int>(ret));
  }

 private:
  at::Tensor tensor_;
  int srcRank_;
  uint32_t utag_;
  std::shared_ptr<::rbln::Rccl> rccl_;
};

// ============================================================================
// Collective Communication RBLNWork Classes
// ============================================================================

/**
 * @brief RBLNWork for broadcast operations
 *
 * Handles broadcasting tensors from root rank to all other ranks.
 */
class BroadcastRBLNWork : public RBLNWork {
 public:
  BroadcastRBLNWork(
      std::vector<at::Tensor>& inputs,
      int rootRank,
      int rootTensor,
      uint32_t tag,
      uint64_t seq,
      std::shared_ptr<::rbln::Rccl> rccl,
      int rank,
      int worldSize)
      : RBLNWork(
            std::vector<std::vector<at::Tensor>>{inputs},
            OpType::BROADCAST,
            seq,
            "rbln:broadcast",
            std::optional<std::vector<at::Tensor>>(inputs)),
        inputs_(inputs),
        rootRank_(rootRank),
        rootTensor_(rootTensor),
        tag_(tag),
        rccl_(std::move(rccl)),
        rank_(rank),
        worldSize_(worldSize) {}

  ~BroadcastRBLNWork() override = default;

  void run() override {
    // Single input tensor is guaranteed by ProcessGroupRBLN::broadcast check
    RBLN_CHECK(inputs_.size() == 1, "BroadcastRBLNWork expects exactly one input tensor, got {}", inputs_.size());
    auto& input = inputs_[0];

    RECORD_FUNCTION("rbln::broadcast", std::vector<c10::IValue>{input});
    RBLN_LOG_DEBUG(
        "Executing BroadcastRBLNWork::run(): worldSize={}, rank={}, rootRank={}", worldSize_, rank_, rootRank_);

    // For single device, no actual broadcast is needed
    if (worldSize_ == 1) {
      RBLN_LOG_DEBUG("Single device broadcast: no operation needed");
      // In single device case, data is already in place, just return success
      return;
    }

    // Execute the broadcast operation for multi-device
    ensureRcclRawMemory(input);
    auto numel = input.numel();
    auto elem_size = input.element_size();
    auto size = numel * elem_size;

    if (size == 0) {
      RBLN_LOG_DEBUG("Skipping broadcast for zero-sized tensor");
      return;
    }

    RBLNRetCode ret = RBLNRetCode_INVALID;
    if (rank_ == rootRank_) {
      ret = rccl_->Broadcast(
          input.data_ptr(),
          input.data_ptr(),
          size, // NOLINT
          ::rbln::Rccl::RcclDataType::RCCL_INT8_TYPE,
          rootRank_,
          nullptr);
    } else {
      ret = rccl_->Broadcast(
          nullptr, input.data_ptr(), size, ::rbln::Rccl::RcclDataType::RCCL_INT8_TYPE, rootRank_, nullptr); // NOLINT
    }
    RBLN_CHECK(ret == RBLNRetCode_SUCCESS, "RCCL Broadcast failed with error code: {}", static_cast<int>(ret));
    RBLN_LOG_DEBUG("RCCL Broadcast completed successfully");
  }

 private:
  std::vector<at::Tensor> inputs_;
  int rootRank_;
  int rootTensor_;
  uint32_t tag_;
  std::shared_ptr<::rbln::Rccl> rccl_;
  int rank_;
  int worldSize_;
};

class AllgatherRBLNWork : public RBLNWork {
 public:
  AllgatherRBLNWork(
      std::vector<std::vector<at::Tensor>>& outputs,
      std::vector<at::Tensor>& inputs,
      uint32_t tag,
      uint64_t seq,
      std::shared_ptr<::rbln::Rccl> rccl,
      int rank,
      int worldSize,
      int device_id)
      : RBLNWork(
            std::vector<std::vector<at::Tensor>>{outputs},
            OpType::ALLGATHER,
            seq,
            "rbln:all_gather",
            std::optional<std::vector<at::Tensor>>(inputs)),
        outputs_(outputs),
        inputs_(inputs),
        tag_(tag),
        rccl_(std::move(rccl)),
        rank_(rank),
        worldSize_(worldSize),
        device_id_(device_id) {}

  ~AllgatherRBLNWork() override = default;

  void run() override {
    RECORD_FUNCTION("rbln::all_gather", std::vector<c10::IValue>(inputs_.begin(), inputs_.end()));
    RBLN_LOG_DEBUG(
        "Executing AllgatherRBLNWork::run(): worldSize={}, rank={}, tensor_count={}",
        worldSize_,
        rank_,
        inputs_.size());

    if (worldSize_ == 1) {
      for (const auto i : c10::irange(inputs_.size())) {
        outputs_[i][0].copy_(inputs_[i]);
      }
      return;
    }

    int idx = 0;
    for (const auto& input : inputs_) {
      auto& output = outputs_[idx++];
      if (output.empty()) {
        RBLN_CHECK(false, "Received an empty list");
      }
      RBLN_CHECK(
          static_cast<int>(output.size()) == worldSize_,
          "output tensor count must equal worldSize, got {} expected {}",
          output.size(),
          worldSize_);
      ensureRcclRawMemory(input);

      auto is_tensor_vector_contiguous = [](std::vector<at::Tensor>& tensors, size_t size) {
        char* ptr = static_cast<char*>(tensors[0].data_ptr());
        for (size_t i = 1; i < tensors.size(); ++i) {
          char* begin = static_cast<char*>(tensors[i].data_ptr());
          ptr += size;
          if (begin != ptr) {
            return false;
          }
        }
        return true;
      };
      // Compute once per input, shared by fast path / chunked / remainder
      const auto numel = input.numel();
      const auto elem_size = input.element_size();
      const size_t input_size = numel * elem_size;
      const size_t align_unit_numel = RCCL_ALLGATHER_ALIGNMENT / elem_size;
      const size_t aligned_up_numel = (numel + align_unit_numel - 1) / align_unit_numel * align_unit_numel;
      const size_t aligned_up_output_size = aligned_up_numel * elem_size * worldSize_;
      const bool needs_chunking = aligned_up_output_size > RCCL_ALLGATHER_MAX_OUTPUT_BYTES;
      const size_t max_input_chunk_numel = RCCL_ALLGATHER_MAX_OUTPUT_BYTES / elem_size / worldSize_;
      const size_t input_chunk_numel =
          needs_chunking ? (max_input_chunk_numel / align_unit_numel) * align_unit_numel : aligned_up_numel;
      const size_t remainder_input_numel = input_chunk_numel > 0 ? (numel % input_chunk_numel) : numel;
      const bool needs_padding = (remainder_input_numel > 0) || !isAlignedTo(input_size, RCCL_ALLGATHER_ALIGNMENT);

      bool is_output_contig =
          is_tensor_vector_contiguous(output, input_size) && isAlignedTo(input_size, rbln::RBLN_DEVICE_MEM_ALLOC_UNIT);
      if (is_output_contig && (needs_chunking || !isAlignedTo(input_size, RCCL_ALLGATHER_ALIGNMENT))) {
        is_output_contig = false;
      }
      if (is_output_contig) {
        ensureContiguousRcclRawMemory(output);
        std::vector<void*> output_ptrs;
        output_ptrs.reserve(output.size());
        for (const auto& t : output) {
          output_ptrs.push_back(t.data_ptr());
        }
        if (rccl_->CheckChunksDeviceAddrsContiguous(
                output_ptrs, static_cast<int>(input_size), ::rbln::Rccl::RcclDataType::RCCL_INT8_TYPE, "CheckRecv") !=
            RBLNRetCode_SUCCESS) {
          RBLN_LOG_DEBUG("RCCL AllGather: recv device addrs not contiguous, using single buffer");
          is_output_contig = false;
        }
      }
      auto fastPath = [&]() {
        if (needs_chunking || !isAlignedTo(input_size, RCCL_ALLGATHER_ALIGNMENT)) {
          return false;
        }
        std::vector<void*> recv_ptrs;
        at::Tensor gathered_output;
        if (is_output_contig) {
          for (const auto& t : output) {
            recv_ptrs.push_back(t.data_ptr());
          }
        } else {
          auto& t = output[0];
          std::vector<int64_t> sizes{static_cast<int64_t>(output.size())};
          sizes.insert(sizes.end(), t.sizes().begin(), t.sizes().end());
          gathered_output = torch::empty(sizes, t.options());
          ensureRcclRawMemory(gathered_output);
          recv_ptrs.push_back(gathered_output.data_ptr());
        }
        int ret = rccl_->AllGather(
            input.data_ptr(),
            recv_ptrs,
            static_cast<int>(input_size),
            ::rbln::Rccl::RcclDataType::RCCL_INT8_TYPE,
            0,
            nullptr);
        RBLN_CHECK(ret == 0, "RCCL AllGather failed with error code: {}", ret);
        if (!is_output_contig) {
          for (const auto j : c10::irange(output.size())) {
            output[j].copy_(gathered_output[static_cast<int64_t>(j)]);
          }
        }
        RBLN_LOG_DEBUG(
            "RCCL AllGather (fast path) completed successfully for {} outputs",
            is_output_contig ? "contiguous" : "non-contiguous");
        return true;
      };

      size_t input_offset = 0;
      size_t output_offset = 0;

      auto processChunked = [&]() {
        if (input_chunk_numel == 0) {
          return;
        }
        std::vector<int64_t> output_chunk_sizes{
            static_cast<int64_t>(worldSize_), static_cast<int64_t>(input_chunk_numel)};
        at::Tensor output_chunk_buffer = torch::empty(output_chunk_sizes, input.options());
        ensureRcclRawMemory(output_chunk_buffer);

        auto input_flatten = input.flatten();
        size_t remaining = numel;
        while (remaining >= input_chunk_numel) {
          auto input_chunk = input_flatten.slice(0, input_offset, input_offset + input_chunk_numel);
          ensureRcclRawMemory(input_chunk);
          int ret = rccl_->AllGather(
              input_chunk.data_ptr(),
              std::vector<void*>{output_chunk_buffer.data_ptr()},
              static_cast<int>(input_chunk_numel * elem_size),
              ::rbln::Rccl::RcclDataType::RCCL_INT8_TYPE,
              0,
              nullptr);
          RBLN_CHECK(ret == 0, "RCCL AllGather (chunked) failed with error code: {}", ret);
          for (const auto j : c10::irange(output.size())) {
            const auto& gathered_chunk = output_chunk_buffer[static_cast<int64_t>(j)];
            auto output_tensor_flatten = output[j].flatten();
            auto output_slice = output_tensor_flatten.slice(0, output_offset, output_offset + input_chunk_numel);
            output_slice.copy_(gathered_chunk);
          }
          input_offset += input_chunk_numel;
          output_offset += input_chunk_numel;
          remaining -= input_chunk_numel;
        }
        RBLN_LOG_DEBUG("RCCL AllGather (chunked) completed successfully");
      };

      auto processRemainder = [&]() {
        if (!needs_padding || remainder_input_numel == 0) {
          if (needs_padding) {
            RBLN_LOG_DEBUG("RCCL AllGather (aligned padding) completed successfully");
          }
          return;
        }
        const size_t aligned_up_remainder_input_numel =
            (remainder_input_numel + align_unit_numel - 1) / align_unit_numel * align_unit_numel;

        std::vector<int64_t> padded_input_chunk_sizes = {static_cast<int64_t>(aligned_up_remainder_input_numel)};
        at::Tensor padded_input_chunk = torch::empty(padded_input_chunk_sizes, input.options());
        auto input_remainder = input.flatten().slice(0, input_offset, numel);
        padded_input_chunk.slice(0, 0, remainder_input_numel).copy_(input_remainder);

        std::vector<int64_t> remainder_output_chunk_sizes{
            static_cast<int64_t>(worldSize_), static_cast<int64_t>(aligned_up_remainder_input_numel)};
        at::Tensor remainder_output_chunk_buffer = torch::empty(remainder_output_chunk_sizes, input.options());

        ensureRcclRawMemory(padded_input_chunk);
        ensureRcclRawMemory(remainder_output_chunk_buffer);
        int ret = rccl_->AllGather(
            padded_input_chunk.data_ptr(),
            std::vector<void*>{remainder_output_chunk_buffer.data_ptr()},
            static_cast<int>(aligned_up_remainder_input_numel * elem_size),
            ::rbln::Rccl::RcclDataType::RCCL_INT8_TYPE,
            0,
            nullptr);
        RBLN_CHECK(ret == 0, "RCCL AllGather (padded remainder) failed with error code: {}", ret);
        for (const auto j : c10::irange(output.size())) {
          const auto& gathered_remainder = remainder_output_chunk_buffer[static_cast<int64_t>(j)];
          auto output_tensor_flatten = output[j].flatten();
          auto output_slice = output_tensor_flatten.slice(0, output_offset, output_offset + remainder_input_numel);
          auto gathered_remainder_slice = gathered_remainder.slice(0, 0, remainder_input_numel);
          output_slice.copy_(gathered_remainder_slice);
        }
        RBLN_LOG_DEBUG("RCCL AllGather (padded remainder) completed successfully");
      };

      if (fastPath()) {
        continue;
      }
      processChunked();
      processRemainder();
    }
  }

 private:
  std::vector<std::vector<at::Tensor>> outputs_;
  std::vector<at::Tensor> inputs_;
  uint32_t tag_;
  std::shared_ptr<::rbln::Rccl> rccl_;
  int rank_;
  int worldSize_;
  int device_id_;
};

class AllreduceRBLNWork : public RBLNWork {
 public:
  AllreduceRBLNWork(
      std::vector<at::Tensor>& inputs,
      const ReduceOp& reduceOp,
      uint32_t tag,
      uint64_t seq,
      std::shared_ptr<::rbln::Rccl> rccl,
      int worldSize,
      int device_id,
      c10::intrusive_ptr<Backend> glooBackend)
      : RBLNWork(
            std::vector<std::vector<at::Tensor>>{inputs},
            OpType::ALLREDUCE,
            seq,
            "rbln:all_reduce",
            std::optional<std::vector<at::Tensor>>(inputs)),
        inputs_(inputs),
        reduceOp_(reduceOp),
        tag_(tag),
        rccl_(std::move(rccl)),
        worldSize_(worldSize),
        device_id_(device_id),
        glooBackend_(std::move(glooBackend)),
        rcclReduceOp_(convertReduceOp(reduceOp)),
        rcclDataType_(convertDataType(inputs[0].scalar_type())) {}

  ~AllreduceRBLNWork() override = default;

  void run() override {
    RECORD_FUNCTION("rbln::all_reduce", std::vector<c10::IValue>(inputs_.begin(), inputs_.end()));
    RBLN_LOG_DEBUG("Executing AllreduceRBLNWork::run(): worldSize={}, tensor_count={}", worldSize_, inputs_.size());
    // For single device, no actual allreduce is needed
    if (worldSize_ == 1) {
      // In single device case, data is already reduced (no communication needed)
      return;
    }

    // Check if input is not CUSTOM_FLOAT16
    const auto rbln_dtype = c10::rbln::to_rbln_dtype(inputs_[0].scalar_type());
    if (rbln_dtype != ::rbln::DataType::Float16) {
      if (glooBackend_) {
        // Use Gloo backend for non-CUSTOM_FLOAT16 allreduce
        // Copy RBLN tensor to CPU
        std::vector<at::Tensor> cpu_tensors;
        for (size_t i = 0; i < inputs_.size(); ++i) {
          for (auto& input : inputs_) {
            cpu_tensors.emplace_back(input.to("cpu"));
          }
        }

        // Call Gloo allreduce on CPU tensor
        AllreduceOptions opts;
        opts.reduceOp = reduceOp_;
        auto work = glooBackend_->allreduce(cpu_tensors, opts);
        work->wait();
        RBLN_LOG_DEBUG("non-CUSTOM_FLOAT16 allreduce by Gloo Backend is completed");

        size_t i = 0;
        for (auto& input : inputs_) {
          input.copy_(cpu_tensors[i++]);
        }
        return;
      }
      RBLN_CHECK(
          false,
          "non-CUSTOM_FLOAT16 allreduce requires a Gloo backend. "
          "Pass gloo_backend when creating ProcessGroupRBLN, or convert tensors to custom float16 before allreduce.");
    }

    rcclDataType_ = ::rbln::Rccl::RcclDataType::RBLN_CUSTOM_FP16_TYPE;

    for (const auto& input : inputs_) {
      const auto rbln_dtype = c10::rbln::to_rbln_dtype(input.scalar_type());
      RBLN_CHECK(rbln_dtype == ::rbln::DataType::Float16, "RCCL AllReduce only supports float16 dtype");

      const size_t elem_size = getRcclElementSize(rbln_dtype);
      const size_t align_numel = RCCL_ALLREDUCE_ALIGNMENT / elem_size;
      const size_t numel = input.numel();
      const size_t total_bytes = numel * elem_size;
      const size_t max_bytes = RCCL_ALLREDUCE_MAX_BYTES_PER_RANK * worldSize_;
      const size_t max_chunk_numel = (max_bytes / elem_size / align_numel) * align_numel;

      // Fast path: small aligned tensor
      if (total_bytes <= max_bytes && isAlignedTo(total_bytes, RCCL_ALLREDUCE_ALIGNMENT)) {
        ensureRcclReduceMemory(input);
        RBLNRetCode ret = rccl_->AllReduce(
            input.data_ptr(), input.data_ptr(), static_cast<int>(numel), rcclDataType_, rcclReduceOp_, 0, nullptr);
        RBLN_CHECK(ret == RBLNRetCode_SUCCESS, "RCCL AllReduce failed: {}", static_cast<int>(ret));
        continue;
      }

      // Calculate aligned portion and remainder
      const size_t aligned_numel = (numel / align_numel) * align_numel;
      const size_t remainder_numel = numel - aligned_numel;
      const bool needs_padding = (remainder_numel > 0);

      auto input_flat = input.flatten();
      char* input_ptr = static_cast<char*>(input_flat.data_ptr());

      // Allocate device memory for the entire storage once (supports offset access for chunks)
      ensureRcclReduceMemory(input_flat);

      // Process aligned portion directly (no copy needed)
      size_t offset = 0;
      size_t remaining = aligned_numel;

      while (remaining > 0) {
        const size_t chunk = std::min(remaining, max_chunk_numel);
        void* ptr = input_ptr + (offset * elem_size);

        RBLNRetCode ret = rccl_->AllReduce(ptr, ptr, static_cast<int>(chunk), rcclDataType_, rcclReduceOp_, 0, nullptr);
        RBLN_CHECK(ret == RBLNRetCode_SUCCESS, "RCCL AllReduce (chunked) failed: {}", static_cast<int>(ret));

        remaining -= chunk;
        offset += chunk;
      }

      // Handle remainder with padding (only copy the unaligned tail)
      if (needs_padding) {
        const auto device = c10::Device(c10::kPrivateUse1, static_cast<c10::DeviceIndex>(device_id_));
        const auto options = c10::TensorOptions().device(device).dtype(c10::kHalf);
        auto padded_tensor = at::empty({static_cast<int64_t>(align_numel)}, options);

        // Copy only the remainder elements to padded tensor
        auto remainder_slice = input_flat.slice(0, aligned_numel, numel);
        auto padded_slice = padded_tensor.slice(0, 0, remainder_numel);
        padded_slice.copy_(remainder_slice);

        // AllReduce the padded remainder
        ensureRcclReduceMemory(padded_tensor);
        RBLNRetCode ret = rccl_->AllReduce(
            padded_tensor.data_ptr(),
            padded_tensor.data_ptr(),
            static_cast<int>(align_numel),
            rcclDataType_,
            rcclReduceOp_,
            0,
            nullptr);
        RBLN_CHECK(ret == RBLNRetCode_SUCCESS, "RCCL AllReduce (padded remainder) failed: {}", static_cast<int>(ret));

        // Copy back only the remainder
        remainder_slice.copy_(padded_slice);
      }
    }
  }

 private:
  std::vector<at::Tensor> inputs_;
  ReduceOp reduceOp_;
  uint32_t tag_;
  std::shared_ptr<::rbln::Rccl> rccl_;
  int device_id_;
  int worldSize_;
  c10::intrusive_ptr<Backend> glooBackend_;
  ::rbln::Rccl::RcclReduceOp rcclReduceOp_;
  ::rbln::Rccl::RcclDataType rcclDataType_;
};

class ReduceScatterRBLNWork : public RBLNWork {
 public:
  ReduceScatterRBLNWork(
      std::vector<at::Tensor>& outputs,
      std::vector<std::vector<at::Tensor>>& inputs,
      const ReduceOp& reduceOp,
      uint32_t tag,
      uint64_t seq,
      std::shared_ptr<::rbln::Rccl> rccl,
      int rank,
      int worldSize,
      int device_id,
      c10::intrusive_ptr<Backend> glooBackend)
      : RBLNWork(
            std::vector<std::vector<at::Tensor>>{outputs},
            OpType::REDUCE_SCATTER,
            seq,
            "rbln:reduce_scatter",
            std::optional<std::vector<at::Tensor>>(outputs)),
        outputs_(outputs),
        inputs_(inputs),
        reduceOp_(reduceOp),
        tag_(tag),
        rccl_(std::move(rccl)),
        rank_(rank),
        worldSize_(worldSize),
        device_id_(device_id),
        glooBackend_(std::move(glooBackend)),
        rcclReduceOp_(convertReduceOp(reduceOp)),
        rcclDataType_(convertDataType(inputs[0][0].scalar_type())) {}

  ~ReduceScatterRBLNWork() override = default;

  void run() override {
    RECORD_FUNCTION("rbln::reduce_scatter", std::vector<c10::IValue>(outputs_.begin(), outputs_.end()));
    RBLN_LOG_DEBUG(
        "Executing ReduceScatterRBLNWork::run(): worldSize={}, rank={}, tensor_count={}",
        worldSize_,
        rank_,
        inputs_.size());

    // For single device, no actual reduce_scatter is needed
    if (worldSize_ == 1) {
      // In single device case, just copy input to output
      outputs_[0].copy_(inputs_[0][0]);
      return;
    }

    // For reduce_scatter, send buffer should contain concatenated data from all ranks' inputs
    // Each rank sends all inputs_[i] tensors concatenated, receives a portion of the reduced result
    // The total send size is recv_numel * worldSize

    // Execute reduce_scatter operation
    for (const auto i : c10::irange(outputs_.size())) {
      auto& output = outputs_[i];
      // For reduce_scatter, each rank needs to concatenate all inputs_[i] tensors
      // inputs_[i] contains worldSize_ tensors, each of size recv_numel
      RBLN_CHECK(static_cast<int>(inputs_[i].size()) == worldSize_, "inputs_[i] must contain worldSize_ tensors");
      // all input shapes should be the same as output shape
      for (const auto& input_tensor : inputs_[i]) {
        RBLN_CHECK(input_tensor.sizes() == output.sizes(), "input shape should be the same as output shape");
      }
      std::vector<void*> input_ptrs;
      for (const auto& t : inputs_[i]) {
        input_ptrs.push_back(t.data_ptr());
      }

      ensureRcclReduceMemory(output);
      const auto rbln_dtype = c10::rbln::to_rbln_dtype(inputs_[i][0].scalar_type());

      // Check if input is not FLOAT16
      if (rbln_dtype != ::rbln::DataType::Float16) {
        if (glooBackend_) {
          // Use Gloo backend for non-FLOAT16 reduce_scatter
          std::vector<at::Tensor> cpu_inputs;
          cpu_inputs.reserve(inputs_[i].size());
          for (auto& tensor : inputs_[i]) {
            cpu_inputs.emplace_back(tensor.to("cpu"));
          }
          std::vector<std::vector<at::Tensor>> cpu_inputs_wrapped = {cpu_inputs};
          std::vector<at::Tensor> cpu_outputs = {output.to("cpu")};

          ReduceScatterOptions opts;
          opts.reduceOp = reduceOp_;
          auto work = glooBackend_->reduce_scatter(cpu_outputs, cpu_inputs_wrapped, opts);
          work->wait();

          output.copy_(cpu_outputs[0]);
          continue;
        }
        RBLN_CHECK(
            false,
            "non-CUSTOM_FLOAT16 reduce_scatter requires a Gloo backend. "
            "Pass gloo_backend when creating ProcessGroupRBLN, or convert tensors to CUSTOM_FLOAT16 before reduce_scatter.");
      }

      rcclDataType_ = ::rbln::Rccl::RcclDataType::RBLN_CUSTOM_FP16_TYPE;

      auto elem_size = getRcclElementSize(rbln_dtype);
      auto recv_numel = output.numel();
      auto send_numel = recv_numel * worldSize_;

      // Check if all inputs are contiguous and can be used directly
      bool inputs_contiguous = true;
      for (const auto j : c10::irange(inputs_[i].size() - 1)) {
        char* current_end = static_cast<char*>(inputs_[i][j].data_ptr()) + inputs_[i][j].numel() * elem_size;
        char* next_begin = static_cast<char*>(inputs_[i][j + 1].data_ptr());
        if (current_end != next_begin) {
          inputs_contiguous = false;
          break;
        }
      }

      const size_t align_unit_numel = RCCL_REDUCE_SCATTER_ALIGNMENT / elem_size;
      const size_t aligned_up_numel = (send_numel + align_unit_numel - 1) / align_unit_numel * align_unit_numel;
      const size_t total_send_bytes = send_numel * elem_size;

      // current rccl implementation constraint: divide by worldSize_ twice
      const size_t max_send_numel_by_buf_constraint =
          RCCL_REDUCE_SCATTER_MAX_BYTES_PER_WORLD / worldSize_ / worldSize_ / elem_size;
      const size_t max_send_numel_by_sub_command_count_constraint = RCCL_MAX_COMMAND_BUFFER_SUB_COMMANDS_COUNT /
          (worldSize_ - 1) * RCCL_MAX_BYTES_PER_REDUCE_SCATTER_OP / worldSize_ / worldSize_ / elem_size;
      const size_t max_send_numel =
          std::min(max_send_numel_by_buf_constraint, max_send_numel_by_sub_command_count_constraint);
      const size_t total_aligned_up_send_bytes = aligned_up_numel * elem_size * worldSize_;
      const bool needs_chunking = (total_aligned_up_send_bytes > RCCL_REDUCE_SCATTER_MAX_BYTES_PER_WORLD) ||
          (total_aligned_up_send_bytes / RCCL_MAX_BYTES_PER_REDUCE_SCATTER_OP * (worldSize_ - 1) >
           RCCL_MAX_COMMAND_BUFFER_SUB_COMMANDS_COUNT);

      // ========================================================================
      // Step 1: Fast path - Process small aligned contiguous tensors directly
      // ========================================================================
      // Conditions: inputs are contiguous, no chunking needed, aligned, and send device addrs contiguous
      bool send_device_addrs_contig = false;
      if (inputs_contiguous && !needs_chunking && isAlignedTo(total_send_bytes, RCCL_REDUCE_SCATTER_ALIGNMENT)) {
        ensureContiguousRcclReduceMemory(inputs_[i]);
        send_device_addrs_contig =
            (rccl_->CheckChunksDeviceAddrsContiguous(
                 input_ptrs, static_cast<int>(recv_numel), rcclDataType_, "CheckSend") == RBLNRetCode_SUCCESS);
        if (!send_device_addrs_contig) {
          RBLN_LOG_DEBUG("RCCL ReduceScatter: send device addrs not contiguous, using chunked path");
        }
      }
      if (inputs_contiguous && !needs_chunking && isAlignedTo(total_send_bytes, RCCL_REDUCE_SCATTER_ALIGNMENT) &&
          send_device_addrs_contig) {
        // Step 1-1: Execute ReduceScatter in a single operation
        RBLNRetCode ret = rccl_->ReduceScatter(
            input_ptrs, output.data_ptr(), static_cast<int>(recv_numel), rcclDataType_, rcclReduceOp_, 0, nullptr);
        RBLN_CHECK(ret == RBLNRetCode_SUCCESS, "RCCL ReduceScatter failed with fast path: {}", static_cast<int>(ret));
        RBLN_LOG_DEBUG("RCCL ReduceScatter (fast path) completed successfully");
        continue;
      }

      // ========================================================================
      // Step 2: Chunked path - Process large tensors in chunks
      // ========================================================================
      // Step 2-1: Calculate chunk size based on alignment and size constraints
      // aligned_down_send_numel can be zero if send_numel is less than align_unit_numel
      const size_t aligned_down_send_numel = (send_numel / align_unit_numel) * align_unit_numel;
      size_t send_chunk_numel = 0;
      if (max_send_numel >= aligned_down_send_numel) {
        send_chunk_numel = aligned_down_send_numel;
      } else {
        send_chunk_numel = max_send_numel;
      }
      // Step 2-2: Calculate remainder size after chunking
      const size_t remainder_send_numel =
          send_chunk_numel > 0 ? send_numel - (send_numel / send_chunk_numel) * send_chunk_numel : send_numel;
      const bool needs_padding = (remainder_send_numel > 0);
      void* out_ptr = nullptr;
      size_t input_numel_offset = 0;
      size_t out_numel_offset = 0;

      // Step 2-3: Process aligned chunks
      if (send_chunk_numel > 0) {
        // Step 2-3-1: Create temporary tensor for concatenated chunk data
        std::vector<int64_t> send_chunk_sizes = {static_cast<int64_t>(send_chunk_numel)};
        at::Tensor send_tensor = torch::empty(send_chunk_sizes, inputs_[i][0].options());

        size_t remaining = send_numel;
        const size_t recv_chunk_numel = send_chunk_numel / worldSize_;
        // Step 2-3-2: Process each chunk iteratively
        while (remaining >= send_chunk_numel) {
          // Step 2-3-2-1: Concatenate chunk from all input tensors into send_tensor
          size_t send_tensor_offset = 0;
          for (const auto& input_tensor : inputs_[i]) {
            auto input_tensor_flatten = input_tensor.flatten();
            auto input_send_tensor_slice =
                input_tensor_flatten.slice(0, input_numel_offset, input_numel_offset + recv_chunk_numel);
            auto send_tensor_slice = send_tensor.slice(0, send_tensor_offset, send_tensor_offset + recv_chunk_numel);
            send_tensor_slice.copy_(input_send_tensor_slice);
            send_tensor_offset += recv_chunk_numel;
          }

          // Step 2-3-2-2: Calculate output pointer offset for this chunk
          out_ptr = static_cast<char*>(output.data_ptr()) + (out_numel_offset * elem_size);

          // Step 2-3-2-3: Execute ReduceScatter for this chunk
          ensureRcclReduceMemory(send_tensor);
          RBLNRetCode ret = rccl_->ReduceScatter(
              std::vector<void*>{send_tensor.data_ptr()},
              out_ptr,
              static_cast<int>(recv_chunk_numel),
              rcclDataType_,
              rcclReduceOp_,
              0,
              nullptr);
          RBLN_CHECK(ret == RBLNRetCode_SUCCESS, "RCCL ReduceScatter (chunked) failed: {}", static_cast<int>(ret));
          RBLN_LOG_DEBUG("RCCL ReduceScatter (chunked) completed successfully");

          // Step 2-3-2-4: Update offsets for next chunk
          input_numel_offset += recv_chunk_numel;
          remaining -= send_chunk_numel;
          out_numel_offset += recv_chunk_numel;
        }
      }

      // ========================================================================
      // Step 3: Handle remainder with padding
      // ========================================================================
      // Step 3-1: Process unaligned remainder portion if exists
      if (needs_padding) {
        // Step 3-1-1: Calculate output pointer for remainder
        out_ptr = static_cast<char*>(output.data_ptr()) + (out_numel_offset * elem_size);
        size_t remainder_recv_numel = recv_numel - input_numel_offset;
        RBLN_CHECK(
            remainder_send_numel == remainder_recv_numel * worldSize_,
            "remainder_send_numel != remainder_recv_numel * worldSize_");

        // Step 3-1-2: Calculate aligned size for padded remainder tensor
        size_t aligned_up_remainder_send_numel =
            (remainder_send_numel + align_unit_numel - 1) / align_unit_numel * align_unit_numel;
        // Step 3-1-3: Create temporary tensor for padded remainder
        at::Tensor remainder_tensor =
            torch::empty({static_cast<int64_t>(aligned_up_remainder_send_numel)}, inputs_[i][0].options());

        // Step 3-1-4: Copy remainder portions from all input tensors to padded tensor
        size_t offset = 0;
        for (const auto& input_tensor : inputs_[i]) {
          auto input_tensor_flatten = input_tensor.flatten();
          auto input_send_tensor_slice = input_tensor_flatten.slice(0, input_numel_offset, recv_numel);
          auto remainder_tensor_slice = remainder_tensor.slice(0, offset, offset + remainder_recv_numel);
          remainder_tensor_slice.copy_(input_send_tensor_slice);
          offset += remainder_recv_numel;
        }

        // Step 3-1-5: Execute ReduceScatter on padded remainder
        ensureRcclReduceMemory(remainder_tensor);
        RBLNRetCode ret = rccl_->ReduceScatter(
            std::vector<void*>{remainder_tensor.data_ptr()},
            out_ptr,
            static_cast<int>(remainder_recv_numel),
            rcclDataType_,
            rcclReduceOp_,
            0,
            nullptr);
        RBLN_CHECK(
            ret == RBLNRetCode_SUCCESS, "RCCL ReduceScatter (padded remainder) failed: {}", static_cast<int>(ret));
        RBLN_LOG_DEBUG("RCCL ReduceScatter (padded remainder) completed successfully");
      }
    }

    RBLN_LOG_DEBUG("RCCL ReduceScatter completed successfully");
  }

 private:
  std::vector<at::Tensor> outputs_;
  std::vector<std::vector<at::Tensor>> inputs_;
  ReduceOp reduceOp_;
  uint32_t tag_;
  std::shared_ptr<::rbln::Rccl> rccl_;
  int rank_;
  int worldSize_;
  int device_id_;
  c10::intrusive_ptr<Backend> glooBackend_;
  ::rbln::Rccl::RcclReduceOp rcclReduceOp_;
  ::rbln::Rccl::RcclDataType rcclDataType_;
};

class ScatterRBLNWork : public RBLNWork {
 public:
  ScatterRBLNWork(
      std::vector<at::Tensor>& outputs,
      std::vector<std::vector<at::Tensor>>& inputs,
      int root,
      uint32_t tag,
      uint64_t seq,
      std::shared_ptr<::rbln::Rccl> rccl,
      int device_id,
      int world_size)
      : RBLNWork(
            std::vector<std::vector<at::Tensor>>{outputs},
            OpType::SCATTER,
            seq,
            "rbln:scatter",
            std::optional<std::vector<at::Tensor>>(outputs)),
        outputs_(outputs),
        inputs_(inputs),
        root_(root),
        tag_(tag),
        rccl_(std::move(rccl)),
        device_id_(device_id),
        rank_(device_id),
        world_size_(world_size) {}

  ~ScatterRBLNWork() override = default;

  void run() override {
    RECORD_FUNCTION("rbln::scatter", std::vector<c10::IValue>(outputs_.begin(), outputs_.end()));
    RBLN_LOG_DEBUG("Executing ScatterRBLNWork::run(): rank={}, root={}, tensor_count={}", rank_, root_, inputs_.size());

    // Execute the scatter operation
    for (int i = 0; i < outputs_.size(); ++i) {
      void* send_buff = nullptr;
      at::Tensor send_tensor;
      const auto& inputs = inputs_[i];
      const uint64_t size = outputs_[i].numel() * outputs_[i].element_size();
      ensureRcclRawMemory(outputs_[i]);

      if (rank_ == root_) {
        RBLN_CHECK(inputs.size() == world_size_, "inputs.size() != world_size_");
        // Check if all inputs are contiguous in vaddr (can be non-contiguous in device memory)
        bool inputs_contiguous = true;
        for (int j = 0; j < inputs.size() - 1; ++j) {
          char* current_end = static_cast<char*>(inputs[j].data_ptr()) + inputs[j].numel() * inputs[j].element_size();
          char* next_begin = static_cast<char*>(inputs[j + 1].data_ptr());
          if (current_end != next_begin) {
            inputs_contiguous = false;
            break;
          }
        }

        inputs_contiguous = inputs_contiguous && isAlignedTo(size, rbln::RBLN_DEVICE_MEM_ALLOC_UNIT);

        if (inputs_contiguous) {
          ensureContiguousRcclRawMemory(inputs_[i]);
          std::vector<void*> input_ptrs;
          for (const auto& t : inputs_[i]) {
            input_ptrs.push_back(t.data_ptr());
          }
          bool send_device_addrs_contig =
              (rccl_->CheckChunksDeviceAddrsContiguous(
                   input_ptrs, static_cast<int>(size), ::rbln::Rccl::RcclDataType::RCCL_INT8_TYPE, "CheckSend") ==
               RBLNRetCode_SUCCESS);
          if (send_device_addrs_contig) {
            int ret = rccl_->Scatter(
                input_ptrs,
                outputs_[i].data_ptr(),
                static_cast<int>(size),
                ::rbln::Rccl::RcclDataType::RCCL_INT8_TYPE,
                root_,
                nullptr);
            TORCH_CHECK(ret == RBLNRetCode_SUCCESS, "RCCL Scatter failed with error code: ", ret);
            RBLN_LOG_DEBUG("RCCL Scatter completed successfully for contiguous inputs");
            continue;
          }
          RBLN_LOG_DEBUG("RCCL Scatter: send device addrs not contiguous, using concatenated buffer");
        }

        // Contiguous send tensor: single buffer (for non-contiguous vaddr or device addrs not contiguous)
        uint64_t send_numel = world_size_ * inputs[0].numel();
        std::vector<int64_t> send_sizes = {static_cast<int64_t>(send_numel)};
        send_tensor = torch::empty(send_sizes, inputs[0].options());
        int64_t offset = 0;
        for (const auto& input_tensor : inputs) {
          auto input_numel = input_tensor.numel();
          auto send_tensor_slice = send_tensor.slice(0, offset, offset + input_numel);
          auto input_tensor_flatten = input_tensor.flatten();
          send_tensor_slice.copy_(input_tensor_flatten);
          offset += input_numel;
        }
        ensureRcclRawMemory(send_tensor);
        send_buff = send_tensor.data_ptr();
      }

      int ret = rccl_->Scatter(
          std::vector<void*>{send_buff},
          outputs_[i].data_ptr(),
          static_cast<int>(size),
          ::rbln::Rccl::RcclDataType::RCCL_INT8_TYPE,
          root_,
          nullptr);
      RBLN_CHECK(ret == RBLNRetCode_SUCCESS, "RCCL Scatter failed with error code: {}", static_cast<int>(ret));
      RBLN_LOG_DEBUG("RCCL Scatter completed successfully");
    }
  }

 private:
  std::vector<at::Tensor> outputs_;
  std::vector<std::vector<at::Tensor>> inputs_;
  int root_;
  uint32_t tag_;
  std::shared_ptr<::rbln::Rccl> rccl_;
  int rank_;
  int device_id_;
  int world_size_;
};

class BarrierRBLNWork : public RBLNWork {
 public:
  BarrierRBLNWork(
      std::vector<c10::weak_intrusive_ptr<Work>> priorWork,
      uint32_t tag,
      uint64_t seq,
      std::shared_ptr<::rbln::Rccl> rccl,
      int rank,
      int device_id,
      int size)
      : RBLNWork(std::vector<std::vector<at::Tensor>>{}, OpType::BARRIER, seq, "rbln:barrier", std::nullopt),
        priorWork_(std::move(priorWork)),
        tag_(tag),
        rccl_(std::move(rccl)),
        rank_(rank),
        device_id_(device_id),
        size_(size) {}

  ~BarrierRBLNWork() override = default;

  void run() override {
    RECORD_FUNCTION("rbln::barrier", std::vector<c10::IValue>());
    RBLN_LOG_DEBUG("Executing BarrierRBLNWork::run(): tensor_count={}", priorWork_.size());
    // Wait on prior work to complete
    for (auto& weakWork : priorWork_) {
      auto work = weakWork.lock();
      if (work) {
        work->wait();
      }
    }

    // For single device, no actual barrier synchronization is needed
    if (size_ == 1) {
      RBLN_LOG_DEBUG("Single device barrier: no operation needed");
      return;
    }

    // For RBLN, we can implement barrier using a simple broadcast on a dummy tensor
    // to ensure all ranks participate in the barrier operation
    try {
      // Create a dummy tensor on RBLN device for all ranks
      // All ranks must have the same tensor shape and type for broadcast
      // RCCL requires 128-byte alignment, so use 64 float16 elements = 128 bytes
      constexpr int64_t BARRIER_ALIGNED_NUMEL = RCCL_ALLGATHER_ALIGNMENT / sizeof(at::Half);
      const auto device = c10::Device(c10::kPrivateUse1, static_cast<c10::DeviceIndex>(device_id_));
      const auto options = c10::TensorOptions().device(device).dtype(c10::kHalf);
      auto dummy_tensor = at::empty({BARRIER_ALIGNED_NUMEL}, options);

      // Only root rank (rank 0) sets the broadcast value
      // Non-root ranks keep the initial value (0) which will be overwritten by broadcast
      if (rank_ == 0) {
        dummy_tensor.fill_(42); // Root rank sets the value to broadcast
      }
      // Note: Non-root ranks keep dummy_tensor as 0, which will be overwritten by broadcast

      // Use broadcast as a barrier - all ranks participate and synchronize
      // Convert tensor data type for RCCL
      auto rccl_data_type = convertDataType(dummy_tensor.scalar_type());

      // Perform broadcast to synchronize all ranks
      int ret = 0;
      const auto numel = dummy_tensor.numel();
      const auto rcclCount = getRcclCount(numel, dummy_tensor.scalar_type());
      ensureRcclRawMemory(dummy_tensor);
      int rootRank = 0;
      if (rank_ == rootRank) {
        ret = rccl_->Broadcast(
            dummy_tensor.data_ptr(),
            dummy_tensor.data_ptr(),
            static_cast<int>(rcclCount),
            rccl_data_type,
            rootRank,
            nullptr); // NOLINT
      } else {
        ret = rccl_->Broadcast(
            nullptr, dummy_tensor.data_ptr(), static_cast<int>(rcclCount), rccl_data_type, rootRank, nullptr); // NOLINT
      }

      RBLN_CHECK(ret == 0, "RCCL Broadcast for barrier failed with error code: {}", static_cast<int>(ret));
    } catch (const std::exception& e) {
      RBLN_CHECK(false, "Barrier implementation failed: {}", e.what());
    }
  }

 private:
  std::vector<c10::weak_intrusive_ptr<Work>> priorWork_;
  uint32_t tag_;
  std::shared_ptr<::rbln::Rccl> rccl_;
  int rank_;
  int device_id_;
  int size_;
};

} // namespace

// ============================================================================
// ProcessGroupRBLN Implementation
// ============================================================================

ProcessGroupRBLN::ProcessGroupRBLN(
    const c10::intrusive_ptr<Store>& store,
    int rank,
    int size,
    int group_id,
    const std::vector<int>& global_ranks_in_group,
    const c10::intrusive_ptr<Options>& options,
    c10::intrusive_ptr<Backend> glooBackend)
    : Backend(rank, size),
      store_(store),
      group_id_(group_id),
      rccl_(std::make_shared<::rbln::Rccl>()),
      backendName_(RBLN_BACKEND_NAME),
      global_ranks_in_group_(global_ranks_in_group),
      glooBackend_(std::move(glooBackend)) {
  c10::rbln::get_device_count();

  // Check environment variable for sync/async mode
  // TORCH_RBLN_C10D_ASYNC=1 enables async mode
  const char* async_env = std::getenv("TORCH_RBLN_C10D_ASYNC");
  if (async_env != nullptr && std::string(async_env) == "1") {
    sync_mode_ = false;
  }

  if (sync_mode_ == false) {
    // Initialize work tracking - use single thread for RBLN
    // In the future, this could be made configurable via options
    const int numThreads = DEFAULT_NUM_WORKERS;
    workInProgress_.resize(numThreads);

    // Start worker threads
    threads_.resize(numThreads);
    for (const auto i : c10::irange(threads_.size())) {
      threads_[i] = std::thread(&ProcessGroupRBLN::runLoop, this, i);
    }
  }

  // Calculate global rank
  if (!global_ranks_in_group_.empty()) {
    RBLN_CHECK(rank_ < global_ranks_in_group_.size(), "rank_ should be less than global_ranks_in_group_'s length");
  }
  global_rank_ = global_ranks_in_group_.empty() ? rank_ : global_ranks_in_group_[rank_];

  // Static map to store global rank to device id mapping in default group
  static std::unordered_map<int, int> global_rank_to_device_id_in_default_group;
  static std::mutex map_mutex;
  static bool default_group_initialized = false;
  device_id_ = -1;
  // Initialize device_id_ based on default group or sub group
  {
    std::lock_guard<std::mutex> lock(map_mutex);
    if (global_ranks_in_group_.empty()) {
      RBLN_CHECK(!default_group_initialized, "Default group must be initialized here");
      // Default group: initialize the map and assign device_id_
      device_id_ = static_cast<int>(static_cast<unsigned char>(c10::rbln::get_device_index()));
      global_rank_to_device_id_in_default_group[global_rank_] = device_id_;
      default_group_initialized = true;
    } else {
      // Sub group: use the mapping from default group
      RBLN_CHECK(default_group_initialized, "Default group must be initialized before creating sub groups");
      RBLN_CHECK(
          global_rank_to_device_id_in_default_group.find(global_rank_) !=
              global_rank_to_device_id_in_default_group.end(),
          "Global rank {} not found in default group mapping",
          global_rank_);
      device_id_ = global_rank_to_device_id_in_default_group[global_rank_];
    }
  }
  RBLN_LOG_DEBUG("device_id: {}", device_id_);

  // Queue initialization work
  auto initSeq = nextSeq();
  auto initWork =
      c10::make_intrusive<InitRBLNWork>(this, rank, size, group_id, device_id_, global_ranks_in_group, initSeq, rccl_);
  enqueueOrExecute(initWork);

  // Wait for initialization to complete
  initWork->wait();
}

// ============================================================================
// Destructor
// ============================================================================

ProcessGroupRBLN::~ProcessGroupRBLN() {
  std::unique_lock<std::mutex> lock(workMutex_);
  workConsumeCV_.wait(lock, [&] { return workQueue_.empty(); });

  // Queue is empty, signal stop
  stop_ = true;

  // Release lock to allow threads to terminate
  lock.unlock();

  workProduceCV_.notify_all();

  // Wait for worker threads to terminate
  for (auto& thread : threads_) {
    thread.join();
  }
}

// ============================================================================
// Private Helper Methods
// ============================================================================

uint32_t ProcessGroupRBLN::nextTag() noexcept {
  return tag_++;
}

uint64_t ProcessGroupRBLN::nextSeq() noexcept {
  return seq_++;
}

namespace {

void logRcclUniqueId(const struct rccl_unique_id* id, const char* prefix, int rank) {
  if (id == nullptr)
    return;
  constexpr size_t ip_len = RCCL_IP_STR_LEN;
  auto safeIpStr = [](const char* p, size_t max_len) {
    size_t len = 0;
    while (len < max_len && p[len] != 0)
      ++len;
    return std::string(p, len);
  };
  RBLN_LOG_INFO(
      "rccl_unique_id {} (rank={}) root_ip={} self_ip={} self_rdma_ip={} root_port={} rdma_base_port={}",
      prefix,
      rank,
      safeIpStr(id->root_ip, ip_len),
      safeIpStr(id->self_ip, ip_len),
      safeIpStr(id->self_rdma_ip, ip_len),
      id->root_port,
      id->rdma_base_port);
}

} // namespace

void ProcessGroupRBLN::broadcastUniqueId(struct rccl_unique_id* rcclID) {
  RBLN_CHECK(rcclID != nullptr, "ProcessGroupRBLN: rcclID is null, cannot broadcast rccl_unique_id");
  RBLN_CHECK(store_ != nullptr, "ProcessGroupRBLN: store_ is null, cannot broadcast rccl_unique_id");
  RBLN_CHECK(rccl_ != nullptr, "ProcessGroupRBLN: rccl_ is null, cannot broadcast rccl_unique_id");

  const std::string storeKey = std::string(STORE_KEY_PREFIX_UNIQUE_ID) + std::to_string(group_id_);
  RBLN_LOG_INFO("broadcastUniqueId rank={} storeKey={}", rank_, storeKey);
  if (rank_ == 0) {
    RBLNRetCode ret = rccl_->GetUniqueIdForBroadcast(rcclID);
    RBLN_CHECK(
        ret == RBLNRetCode_SUCCESS,
        "ProcessGroupRBLN: GetUniqueIdForBroadcast failed (call after PrepareContextAndExportMem)");

    std::vector<uint8_t> vec(
        reinterpret_cast<uint8_t*>(rcclID), reinterpret_cast<uint8_t*>(rcclID) + sizeof(struct rccl_unique_id));
    store_->set(storeKey, vec);
    logRcclUniqueId(rcclID, "set to tcp_store", rank_);
  } else {
    std::vector<uint8_t> vec = store_->get(storeKey);
    RBLN_CHECK(
        vec.size() == sizeof(struct rccl_unique_id),
        "ProcessGroupRBLN: invalid rccl_unique_id size from tcp_store (got {}, expected {}), Rank 0 may have crashed or store/network issue.",
        vec.size(),
        sizeof(struct rccl_unique_id));
    std::memcpy(rcclID, vec.data(), vec.size());
    logRcclUniqueId(rcclID, "get from tcp_store", rank_);
  }
}

void ProcessGroupRBLN::enqueue(c10::intrusive_ptr<Work> work) {
  std::unique_lock<std::mutex> lock(workMutex_);
  workQueue_.push_back(std::move(work));
  lock.unlock();

  // Notify after releasing the lock so that the waiter
  // does not immediately block.
  workProduceCV_.notify_one();
}

void ProcessGroupRBLN::enqueueOrExecute(c10::intrusive_ptr<Work> work) {
  if (sync_mode_) {
    auto rblnWork = c10::intrusive_ptr<RBLNWork>::reclaim(static_cast<RBLNWork*>(work.get()));
    work.release(); // Release the original work pointer
    RBLNWork::execute(rblnWork);
  } else {
    enqueue(work);
  }
}

void ProcessGroupRBLN::runLoop(int workerIndex) {
  std::unique_lock<std::mutex> lock(workMutex_);

  while (!stop_) {
    if (workQueue_.empty()) {
      workProduceCV_.wait(lock);
      continue;
    }

    auto work = std::move(workQueue_.front());
    workQueue_.pop_front();
    workInProgress_[workerIndex] = work;
    lock.unlock();

    // Notify after releasing the lock so that the waiter
    // does not immediately block.
    workConsumeCV_.notify_one();

    // Execute the work using RBLNWork pattern
    auto rblnWork = c10::intrusive_ptr<RBLNWork>::reclaim(static_cast<RBLNWork*>(work.get()));
    work.release(); // Release the original work pointer
    RBLNWork::execute(rblnWork);

    lock.lock();
    workInProgress_[workerIndex].reset();
  }
}

// ============================================================================
// Collective Communication Operations
// ============================================================================

c10::intrusive_ptr<Work> ProcessGroupRBLN::broadcast(std::vector<at::Tensor>& inputs, const BroadcastOptions& opts) {
  static auto invalidArgument = [](const std::string& msg) {
    RBLN_CHECK(false, "ProcessGroupRBLN::broadcast: {}", msg);
  };

  // Limit input tensor count to 1
  RBLN_CHECK(inputs.size() == 1, "ProcessGroupRBLN::broadcast: Expecting one tensor only but got {}", inputs.size());

  assertRootRank(invalidArgument, opts.rootRank, size_);
  assertRootTensor(invalidArgument, opts.rootTensor, static_cast<int64_t>(inputs.size()));
  assertDense(invalidArgument, inputs);
  assertTypeAndSizesMatch(invalidArgument, inputs);

  const auto& device = inputs[0].device();
  if (device.is_cpu()) {
    RBLN_CHECK(glooBackend_, "ProcessGroupRBLN::broadcast: CPU tensors require gloo_backend");
    return glooBackend_->broadcast(inputs, opts);
  }
  if (device.type() != at::kPrivateUse1) {
    invalidArgument(c10::str("unsupported device type ", device.type()));
  }

  auto tag = nextTag();
  auto seq = nextSeq();

  auto work =
      c10::make_intrusive<BroadcastRBLNWork>(inputs, opts.rootRank, opts.rootTensor, tag, seq, rccl_, rank_, getSize());

  enqueueOrExecute(work);
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupRBLN::allgather(
    std::vector<std::vector<at::Tensor>>& outputs,
    std::vector<at::Tensor>& inputs,
    const AllgatherOptions& opts) {
  static auto invalidArgument = [](const std::string& msg) {
    RBLN_CHECK(false, "ProcessGroupRBLN::allgather: {}", msg);
  };

  if (inputs.empty()) {
    invalidArgument("requires non-empty input tensor list");
  }

  if (inputs.size() != outputs.size()) {
    invalidArgument("requires input/output tensor lists to have the same length");
  }

  for (const auto i : c10::irange(outputs.size())) {
    const auto expected = getSize();
    const auto actual = outputs[i].size();
    if (actual != expected) {
      invalidArgument(
          "invalid output tensor list at index " + std::to_string(i) + " (expected length " + std::to_string(expected) +
          ", got " + std::to_string(actual) + ")");
    }
  }

  assertDense(invalidArgument, inputs);

  // Expect all input/output tensors to have the same type and sizes
  const auto& options = inputs[0].options();
  const auto& sizes = inputs[0].sizes();
  assertTypeAndSizesMatch(invalidArgument, inputs, options, sizes);
  for (const auto& output : outputs) {
    assertTypeAndSizesMatch(invalidArgument, output, options, sizes);
  }

  const auto& device = inputs[0].device();
  if (device.is_cpu()) {
    RBLN_CHECK(glooBackend_, "ProcessGroupRBLN::allgather: CPU tensors require gloo_backend");
    return glooBackend_->allgather(outputs, inputs, opts);
  }
  if (device.type() != at::kPrivateUse1) {
    invalidArgument(c10::str("unsupported device type ", device.type()));
  }

  auto tag = nextTag();
  auto seq = nextSeq();
  const auto rank = getRank();
  const auto worldSize = getSize();
  auto work = c10::make_intrusive<AllgatherRBLNWork>(outputs, inputs, tag, seq, rccl_, rank, worldSize, device_id_);

  enqueueOrExecute(work);
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupRBLN::_allgather_base(
    at::Tensor& output_tensor,
    at::Tensor& input_tensor,
    const AllgatherOptions& opts) {
  auto tensor_list = at::chunk(output_tensor, getSize(), 0);
  std::vector<std::vector<at::Tensor>> outputs = {tensor_list};
  std::vector<at::Tensor> inputs = {input_tensor};
  return allgather(outputs, inputs, opts);
}

c10::intrusive_ptr<Work> ProcessGroupRBLN::allgather_into_tensor_coalesced(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllgatherOptions& opts) {
  static auto invalidArgument = [](const std::string& msg) {
    RBLN_CHECK(false, "ProcessGroupRBLN::allgather_into_tensor_coalesced: {}", msg);
  };

  RBLN_CHECK(outputTensors.size() == inputTensors.size());

  if (inputTensors.empty()) {
    invalidArgument("requires non-empty input tensor list");
  }

  const auto& device = inputTensors[0].device();
  if (device.is_cpu()) {
    RBLN_CHECK(glooBackend_, "ProcessGroupRBLN::allgather_into_tensor_coalesced: CPU tensors require gloo_backend");
    return glooBackend_->allgather_into_tensor_coalesced(outputTensors, inputTensors, opts);
  }
  if (device.type() != at::kPrivateUse1) {
    invalidArgument(c10::str("unsupported device type ", device.type()));
  }

  const auto worldSize = getSize();
  std::vector<std::vector<at::Tensor>> output_lists(outputTensors.size());
  for (size_t i = 0; i < outputTensors.size(); ++i) {
    output_lists[i] = outputTensors[i].chunk(worldSize);
  }
  return allgather(output_lists, inputTensors, opts);
}

c10::intrusive_ptr<Work> ProcessGroupRBLN::allreduce(std::vector<at::Tensor>& inputs, const AllreduceOptions& opts) {
  static auto invalidArgument = [](const std::string& msg) {
    RBLN_CHECK(false, "ProcessGroupRBLN::allreduce: {}", msg);
  };

  assertNonEmpty(invalidArgument, inputs);
  assertLayoutMatch(invalidArgument, inputs);
  assertTypeAndSizesMatch(invalidArgument, inputs);

  const auto& layout = inputs[0].layout();
  if (layout == c10::kSparse && opts.reduceOp != ReduceOp::SUM) {
    invalidArgument(
        "unsupported reduction operation "
        "(allreduce of sparse tensors only works with ReduceOp.SUM)");
  }

  const auto& device = inputs[0].device();
  if (device.is_cpu()) {
    RBLN_CHECK(glooBackend_, "ProcessGroupRBLN::allreduce: CPU tensors require gloo_backend");
    return glooBackend_->allreduce(inputs, opts);
  }
  if (device.type() != at::kPrivateUse1) {
    invalidArgument(c10::str("unsupported device type ", device.type()));
  }

  auto tag = nextTag();
  auto seq = nextSeq();
  auto work = c10::make_intrusive<AllreduceRBLNWork>(
      inputs, opts.reduceOp, tag, seq, rccl_, getSize(), device_id_, glooBackend_);

  enqueueOrExecute(work);
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupRBLN::reduce_scatter(
    std::vector<at::Tensor>& outputs,
    std::vector<std::vector<at::Tensor>>& inputs,
    const ReduceScatterOptions& opts) {
  static auto invalidArgument = [](const std::string& msg) {
    RBLN_CHECK(false, "ProcessGroupRBLN::reduce_scatter: {}", msg);
  };

  const auto rank = getRank();
  const auto worldSize = getSize();

  RBLN_CHECK(outputs.size() == 1, "reduce_scatter only supports 1 output");
  RBLN_CHECK(outputs.size() == inputs.size(), "requires input/output tensor lists to have the same length");
  RBLN_CHECK(static_cast<int>(inputs[0].size()) == worldSize, "invalid input tensor list size, must be world size");

  for (const auto i : c10::irange(inputs[0].size())) {
    RBLN_CHECK(outputs[0].dtype() == inputs[0][i].dtype());
    RBLN_CHECK(outputs[0].sizes().vec() == inputs[0][i].sizes().vec());
  }

  const auto& device = outputs[0].device();
  if (device.is_cpu()) {
    RBLN_CHECK(glooBackend_, "ProcessGroupRBLN::reduce_scatter: CPU tensors require gloo_backend");
    return glooBackend_->reduce_scatter(outputs, inputs, opts);
  }
  if (device.type() != at::kPrivateUse1) {
    invalidArgument(c10::str("unsupported device type ", device.type()));
  }

  auto tag = nextTag();
  auto seq = nextSeq();

  auto work = c10::make_intrusive<ReduceScatterRBLNWork>(
      outputs, inputs, opts.reduceOp, tag, seq, rccl_, rank, worldSize, device_id_, glooBackend_);

  enqueueOrExecute(work);
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupRBLN::_reduce_scatter_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    const ReduceScatterOptions& opts) {
  static auto invalidArgument = [](const std::string& msg) {
    RBLN_CHECK(false, "ProcessGroupRBLN::_reduce_scatter_base: {}", msg);
  };

  const auto worldSize = getSize();

  // Validate input and output tensors
  RBLN_CHECK(outputTensor.dtype() == inputTensor.dtype());
  RBLN_CHECK(inputTensor.numel() == (outputTensor.numel() * worldSize));

  // Split input tensor into chunks for each rank
  auto input_chunks = at::chunk(inputTensor, worldSize, 0);
  std::vector<std::vector<at::Tensor>> inputs = {input_chunks};
  std::vector<at::Tensor> outputs = {outputTensor};

  return reduce_scatter(outputs, inputs, opts);
}

c10::intrusive_ptr<Work> ProcessGroupRBLN::scatter(
    std::vector<at::Tensor>& outputs,
    std::vector<std::vector<at::Tensor>>& inputs,
    const ScatterOptions& opts) {
  static auto invalidArgument = [](const std::string& msg) { RBLN_CHECK(false, "ProcessGroupRBLN::scatter: {}", msg); };

  assertRootRank(invalidArgument, opts.rootRank, size_);
  assertNonEmpty(invalidArgument, outputs);

  // For scatter, inputs can be empty on non-root ranks
  if (getRank() == opts.rootRank) {
    RBLN_CHECK(!inputs.empty(), "inputs cannot be empty on root rank");
    RBLN_CHECK(!inputs[0].empty(), "inputs[0] cannot be empty on root rank");
    assertDense(invalidArgument, inputs[0]);
  }

  assertDense(invalidArgument, outputs);

  const auto& device = outputs[0].device();
  if (device.is_cpu()) {
    RBLN_CHECK(glooBackend_, "ProcessGroupRBLN::scatter: CPU tensors require gloo_backend");
    return glooBackend_->scatter(outputs, inputs, opts);
  }
  if (device.type() != at::kPrivateUse1) {
    invalidArgument(c10::str("unsupported device type ", device.type()));
  }

  auto tag = nextTag();
  auto seq = nextSeq();
  auto work = c10::make_intrusive<ScatterRBLNWork>(outputs, inputs, opts.rootRank, tag, seq, rccl_, device_id_, size_);

  enqueueOrExecute(work);
  return work;
}

// ============================================================================
// Point-to-Point Communication Operations
// ============================================================================

c10::intrusive_ptr<Work> ProcessGroupRBLN::send(std::vector<at::Tensor>& tensors, int dstRank, int tag) {
  static auto invalidArgument = [](const std::string& msg) { RBLN_CHECK(false, "ProcessGroupRBLN::send: {}", msg); };

  assertNonEmpty(invalidArgument, tensors);
  assertLayoutMatch(invalidArgument, tensors);
  assertTypeAndSizesMatch(invalidArgument, tensors);

  const auto& device = tensors[0].device();
  if (device.is_cpu()) {
    RBLN_CHECK(glooBackend_, "ProcessGroupRBLN::send: CPU tensors require gloo_backend");
    return glooBackend_->send(tensors, dstRank, tag);
  }
  if (device.type() != at::kPrivateUse1) {
    invalidArgument(c10::str("unsupported device type ", device.type()));
  }

  auto utag = nextTag();
  auto seq = nextSeq();
  auto work = c10::make_intrusive<SendRBLNWork>(tensors[0], dstRank, utag, seq, rccl_);

  enqueueOrExecute(work);
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupRBLN::recv(std::vector<at::Tensor>& tensors, int srcRank, int tag) {
  static auto invalidArgument = [](const std::string& msg) { RBLN_CHECK(false, "ProcessGroupRBLN::recv: {}", msg); };

  assertNonEmpty(invalidArgument, tensors);
  assertLayoutMatch(invalidArgument, tensors);
  assertTypeAndSizesMatch(invalidArgument, tensors);

  const auto& device = tensors[0].device();
  if (device.is_cpu()) {
    RBLN_CHECK(glooBackend_, "ProcessGroupRBLN::recv: CPU tensors require gloo_backend");
    return glooBackend_->recv(tensors, srcRank, tag);
  }
  if (device.type() != at::kPrivateUse1) {
    invalidArgument(c10::str("unsupported device type ", device.type()));
  }

  auto utag = nextTag();
  auto seq = nextSeq();
  auto work = c10::make_intrusive<RecvRBLNWork>(tensors[0], srcRank, utag, seq, rccl_);

  enqueueOrExecute(work);
  return work;
}

// ============================================================================
// Synchronization Operations
// ============================================================================

c10::intrusive_ptr<Work> ProcessGroupRBLN::barrier(const BarrierOptions& opts) {
  if (glooBackend_) {
    return glooBackend_->barrier(opts);
  }

  std::vector<c10::weak_intrusive_ptr<Work>> priorWork;

  // Snapshot all in progress and pending work as weak_ptr.
  // When executing a barrier, we need to ensure that all prior work
  // has completed before completing itself.
  {
    std::unique_lock<std::mutex> lock(workMutex_);
    // Reserve memory to avoid multiple reallocations during insert operations
    priorWork.reserve(priorWork.size() + workInProgress_.size() + workQueue_.size());
    priorWork.insert(priorWork.end(), workInProgress_.begin(), workInProgress_.end());
    priorWork.insert(priorWork.end(), workQueue_.begin(), workQueue_.end());
  }

  auto tag = nextTag();
  auto seq = nextSeq();
  auto work = c10::make_intrusive<BarrierRBLNWork>(std::move(priorWork), tag, seq, rccl_, rank_, device_id_, size_);
  enqueueOrExecute(work);
  return work;
}

// ============================================================================
// Sequence Number Management
// ============================================================================

void ProcessGroupRBLN::setSequenceNumberForGroup() {
  // RBLN backend starts sequence numbers at 0, similar to GLOO and NCCL
  // No additional synchronization needed as sequence numbers are managed
  // internally by the ProcessGroupRBLN instance
}

uint64_t ProcessGroupRBLN::getSequenceNumberForGroup() {
  return seq_;
}

} // namespace c10d
