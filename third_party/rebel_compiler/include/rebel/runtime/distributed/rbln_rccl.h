#ifndef REBEL_RUNTIME_DISTRIBUTED_RBLN_RCCL_H
#define REBEL_RUNTIME_DISTRIBUTED_RBLN_RCCL_H

#include <rebel/runtime/api/rbln_runtime_api.h>

#include <memory>
#include <vector>

struct rccl_unique_id;
struct rccl_comm;
struct rccl_extra_args;
namespace rbln {

/**
 * @brief The alignment unit for device memory allocation.
 *
 * This is the alignment unit for device memory allocation. It is used to ensure that the device
 * memory is allocated in a contiguous block.
 */
constexpr std::size_t RBLN_DEVICE_MEM_ALLOC_UNIT = 512;

class Context;

class Rccl {
 public:
  enum class RcclDataType : int {
    RCCL_INT8_TYPE = 0,
    RBLN_CUSTOM_FP16_TYPE = 1,
    RCCL_BF16_TYPE = 2,
    RCCL_MAX_DATA_TYPE = 3,
  };

  enum class RcclReduceOp : int {
    RCCL_REDUCE_OP_SUM = 0,
    RCCL_REDUCE_OP_MUL = 1,
    RCCL_REDUCE_OP_MAX = 2,
    RCCL_REDUCE_OP_MIN = 3,
    RCCL_REDUCE_OP_AVG = 4,
    RCCL_REDUCE_OP_NUM = 5,
  };

  Rccl();
  ~Rccl();

  RBLNRetCode Init(int rank, int size, int group_id, const std::vector<int>& global_ranks_in_group,
                   int torch_device_id = 0);
  /**
   * Prepares context and exports memory (GetGlobalContext + ExportMem). Call this before
   * broadcastUniqueRCCLID so that rank 0's rcclGetUniqueId runs after ExportMem. Then call
   * InitWithUniqueId(..., &rcclID); InitWithUniqueId will skip the preamble and only do
   * unique_id copy + CommInitRank + CommUserRank.
   */
  RBLNRetCode PrepareContextAndExportMem(int torch_device_id);

  /**
   * Fills *out with unique_id by calling rcclGetUniqueId on this instance's context.
   * Use when RCCL_PORT_GEN=1 (UMD): after PrepareContextAndExportMem, call this instead of
   * raw rcclGetUniqueId so the call uses the exported context (avoids crash in UMD).
   */
  RBLNRetCode GetUniqueIdForBroadcast(struct rccl_unique_id* out);

  /**
   * Same as Init() but for default group uses the provided rccl_unique_id instead of
   * calling GetUniqueId(). Used when ProcessGroupRBLN has broadcast the ID via store.
   * If PrepareContextAndExportMem was already called, skips GetGlobalContext/ExportMem.
   * @param id When non-null and default group, copy to unique_id_ and call CommInitRank only.
   */
  RBLNRetCode InitWithUniqueId(int rank, int size, int group_id,
                               const std::vector<int>& global_ranks_in_group, int torch_device_id,
                               const struct rccl_unique_id* id);
  RBLNRetCode Send(void* send_vaddr, int count, RcclDataType data_type, int peer,
                   rccl_extra_args* arg);
  RBLNRetCode Recv(void* recv_vaddr, int count, RcclDataType data_type, int peer,
                   rccl_extra_args* arg);
  RBLNRetCode Broadcast(void* send_vaddr, void* recv_vaddr, int count, RcclDataType data_type,
                        int root, rccl_extra_args* arg);
  RBLNRetCode AllGather(void* send_vaddr, const std::vector<void*>& recv_chunk_vaddrs, int count,
                        RcclDataType data_type, int root, rccl_extra_args* arg);
  RBLNRetCode AllReduce(void* send_vaddr, void* recv_vaddr, int count, RcclDataType data_type,
                        RcclReduceOp op, int root, rccl_extra_args* arg);
  RBLNRetCode Scatter(const std::vector<void*>& send_chunk_vaddrs, void* recv_vaddr, int count,
                      RcclDataType data_type, int root, rccl_extra_args* arg);
  RBLNRetCode ReduceScatter(const std::vector<void*>& send_chunk_vaddrs, void* recv_vaddr,
                            int recv_count, RcclDataType data_type, RcclReduceOp op, int root,
                            rccl_extra_args* arg);
  RBLNRetCode Wait(rccl_extra_args* arg);

  /**
   * Checks whether the given chunk pointers resolve to contiguous device addresses.
   * chunk_vaddrs[i] is the data_ptr() of the i-th chunk. Returns RBLNRetCode_SUCCESS only if all
   * chunks resolve and are contiguous. Used by AllGather, Scatter and ReduceScatter before
   * calling with num_chunks > 1.
   */
  RBLNRetCode CheckChunksDeviceAddrsContiguous(const std::vector<void*>& chunk_vaddrs,
                                               int elem_cnt_per_chunk, RcclDataType data_type,
                                               const char* op_name);

  // for torch-rbln backward compatibility
  RBLNRetCode AllGather(void* send_vaddr, void* recv_vaddr, int count, RcclDataType data_type,
                        int root, rccl_extra_args* arg);

  // for torch-rbln backward compatibility
  RBLNRetCode Scatter(void* send_vaddr, void* recv_vaddr, int count, RcclDataType data_type,
                      int root, rccl_extra_args* arg);

  // for torch-rbln backward compatibility
  RBLNRetCode ReduceScatter(void* send_vaddr, void* recv_vaddr, int recv_count,
                            RcclDataType data_type, RcclReduceOp op, int root,
                            rccl_extra_args* arg);

  static struct rccl_comm* default_comm_ptr_;
  static int global_rank_;

 private:
  inline bool IsForceRdma() const { return use_force_rdma_; }
  inline bool IsForceExportMem() const { return use_force_export_mem_; }
  inline void ExportMemIfForced() {
    if (IsForceRdma() || IsForceExportMem()) ExportMem();
  }
  /**
   * @brief Export memory only if device memory has changed since last export.
   *
   * This checks the MemoryChangeTracker flag set by CachingAllocator and
   * only calls ExportMem() when necessary. This avoids the ~3ms overhead
   * of calling ExportMem() on every CCL operation.
   */
  void ExportMemIfNeeded();
  bool VerifyInitialized() const;
  void ReadEnv();
  RBLNRetCode ExportMem();
  RBLNRetCode GetUniqueId();
  RBLNRetCode CommInitRank(int rank, int size);
  RBLNRetCode CommUserRank();
  RBLNRetCode InitDefaultGroup(int rank, int size, int group_id);
  RBLNRetCode InitSubGroup(int rank, int size, int group_id,
                           const std::vector<int>& global_ranks_in_group);

  /**
   * Resolves a single vaddr range to device address. Used by callers to check device addr
   * contiguity before using multi-chunk fast paths (AllGather recv, Scatter/ReduceScatter send).
   */
  RBLNRetCode GetSingleDeviceAddrFromVMem(void* vaddr, int count, RcclDataType data_type,
                                          uint64_t& device_addr_out);

  /** Resolves chunk pointers to device addrs and checks contiguity; used by AllGather,
   * Scatter and ReduceScatter. chunk_vaddrs[i] is the i-th chunk's virtual address.
   */
  RBLNRetCode ResolveBufferToDeviceAddrs(const std::vector<void*>& chunk_vaddrs,
                                         int elem_cnt_per_chunk, RcclDataType data_type,
                                         std::vector<uint64_t>& device_addrs, const char* op_name);

  bool is_initialized_ = false;
  /** True after PrepareContextAndExportMem(); InitWithUniqueId(id) then skips preamble. */
  bool context_and_export_ready_ = false;
  bool use_force_rdma_ = false;
  bool use_force_export_mem_ = false;
  std::shared_ptr<Context> ctx_ = nullptr;
  std::unique_ptr<struct rccl_unique_id> unique_id_;
  std::unique_ptr<struct rccl_comm> comm_;

  // These scalars are set during Init / PrepareContextAndExportMem / CommInitRank / CommUserRank,
  // but appear as format arguments in several RT_LOG(ERROR, ...) paths that can fire before the
  // setter has run (e.g. rcclCommInitRank failure on the first call leaves group_id_ / rccl_rank_
  // unset). Default them to -1 so failed-init logging is well-defined instead of UB.
  int rank_ = -1;  // group_rank
  int size_ = -1;
  int rccl_rank_ = -1;
  int group_id_ = -1;
  int torch_device_id_ = -1;
};

RBLNRetCode RcclInitExportMem(std::shared_ptr<Context> ctx);

}  // namespace rbln

#endif  // RBLN_RCCL_H
