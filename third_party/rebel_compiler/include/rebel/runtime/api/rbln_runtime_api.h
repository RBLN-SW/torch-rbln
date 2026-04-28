#ifndef REBEL_RUNTIME_API_RBLN_RUNTIME_API_H
#define REBEL_RUNTIME_API_RBLN_RUNTIME_API_H

#include <stdint.h>

#include <string>
#include <tuple>
#include <vector>

#ifdef __cplusplus

namespace rbln {
class MemoryStats;
}

extern "C" {
#endif

typedef enum {
  RBLNRetCode_SUCCESS = 0,
  RBLNRetCode_FAILURE,
  RBLNRetCode_INVALID,
} RBLNRetCode;

typedef enum {
  RBLNMemcpyType_H2D = 0,
  RBLNMemcpyType_D2H,
  RBLNMemcpyType_D2D,
} RBLNMemcpyType;

/**
 * @brief Checks if the device pool has been initialized.
 *
 * @param initialized [out] true if the device pool is initialized; false otherwise.
 *
 * @return 0 on success, or an error code on failure.
 */
RBLNRetCode rbln_check_device_pool_initialized(bool* initialized);

/**
 * @brief Initializes devices to be used for NPU executions at a specific device.
 *
 * @param device_id [in] Torch Device number of target device. i.e. 0 for rbln:0, 1 for rbln:1, etc.
 * @param device_ids [in] Array of device IDs to be used.
 * @param num_device [in] Number of device IDs in the array.
 *
 * @return 0 on success, or an error code on failure.
 */
RBLNRetCode rbln_register_device_id(int torch_device_id, int* device_ids, int num_device);

/**
 * @brief Returns the number of initialized NPU devices.
 *
 * @param count [out] The number of devices.
 *
 * @return 0 on success, or an error code on failure.
 */
RBLNRetCode rbln_get_device_count(int* count);

/**
 * @brief Allocates memory on the device, with caching allocator enabled.
 *
 * @details It is an alias to the `rbln_malloc_cache` function.
 *
 * @param device_id [in] Device number of target device.
 * @param size [in] Requested allocation size in bytes.
 * @param device_ptr_out [out] Pointer to allocated device memory.
 *
 * @return 0 on success, or an error code on failure.
 */
RBLNRetCode rbln_malloc(int device_id, uint64_t size, void** device_ptr_out);

/**
 * @brief Frees memory allocated with caching allocator.
 *
 * @details It is an alias to the `rbln_free_cache` function. It only can free the memory
 * allocated with `rbln_malloc` or `rbln_malloc_cache`.
 *
 * @param device_id [in] Device number of target device.
 * @param ptr [in] Pointer to device memory to free.
 *
 * @return 0 on success, or an error code on failure.
 */
RBLNRetCode rbln_free(int device_id, void* ptr);

/**
 * @brief Copies data between host and device.
 *
 * @param device_id [in] Device number of target device.
 * @param dst [in] Destination memory address.
 * @param src [in] Source memory address.
 * @param size [in] Size in bytes to copy.
 * @param type [in] Type of transfer.
 *
 * @return 0 on success, or an error code on failure.
 */
RBLNRetCode rbln_memcpy(int device_id, void* dst, const void* src, uint64_t size,
                        RBLNMemcpyType type);

/**
 * @brief Copies data between device and device using command stream.
 *
 * @param cs_addr [in] Command stream address.
 * @param shmVA [in] Shared memory address.
 * @param device_id [in] Device number of target device.
 * @param copy_ops [in] Copy operations. (src_addr, dst_addr, size)
 *
 * @return 0 on success, or an error code on failure.
 */
RBLNRetCode rbln_memcpy_d2d_by_cs(
    uint64_t cs_addr, uint64_t shmVA, int device_id,
    const std::vector<std::tuple<uint64_t, uint64_t, uint64_t>>& copy_ops);

/**
 * @brief Releases all unoccupied cached memory.
 *
 * @param device_id [in] Device number of target device.
 *
 * @return 0 on success, or an error code on failure.
 */
RBLNRetCode rbln_empty_cache(int device_id);

/**
 * @brief Resets the accumulated (historical) stats.
 *
 * @param device_id [in] Device number of target device.
 *
 * @return 0 on success, or an error code on failure.
 */
RBLNRetCode rbln_reset_accumulated_memory_stats(int device_id);

/**
 * @brief Resets the peak stats.
 *
 * @param device_id [in] Device number of target device.
 *
 * @return 0 on success, or an error code on failure.
 */
RBLNRetCode rbln_reset_peak_memory_stats(int device_id);

#ifdef __cplusplus
}

/**
 * @brief Gets memory statistics for a specific device (C++ only).
 *
 * This function returns the MemoryStats object for the specified device.
 * The returned object contains comprehensive memory statistics including
 * allocated, reserved, active block, and cached block memory information.
 *
 * @param device_id [in] Device number of target device.
 *
 * @return MemoryStats object on success, or throws exception on failure.
 */
rbln::MemoryStats rbln_get_memory_stats(int device_id);
#endif

/**
 * Allocate a virtual RBLN address of the size `size`. The virtual address will originally refer
 * to the host memory starting at `host_ptr`. The call to this function means the ownership of
 * the transfer of the memory area referenced by `host_ptr`, which means that the area may be
 * freed by the RBLN runtime and the memory must not be freed by the caller.
 *
 * @param size
 * @param host_ptr
 */
uint64_t rbln_v_malloc_with_host_ptr(size_t size, uintptr_t host_ptr);

namespace rbln {

enum class DataType {
  Undefined,
  Bool,           // 1 bytes (Bool)
  UInt8,          // 1 bytes (Byte)
  Int8,           // 1 bytes (Char)
  Int16,          // 2 bytes (Short)
  Int32,          // 4 bytes (Int)
  Int64,          // 8 bytes (Long)
  Float16,        // 2 bytes (Half)
  Float32,        // 4 bytes (Float)
  Float64,        // 8 bytes (Double)
  Float8_e4m3,    // 1 bytes
  Float8_e5m2,    // 1 bytes
  BFloat16,       // 2 bytes
  CustomFloat16,  // 2 bytes
  Complex32,      // 2 + 2 = 4 bytes (ComplexHalf)
  Complex64,      // 4 + 4 = 8 bytes (ComplexFloat)
  Complex128,     // 8 + 8 = 16 bytes (ComplexDouble)
};

struct MemoryInfo {
  uint32_t torch_device_id;
  uint64_t key_vaddr;
  DataType user_dtype;
  std::vector<int64_t> user_shape;
  DataType physical_dtype;
  std::vector<int64_t> physical_shape;
};

// Allocates a vmemory entry and eagerly allocates device memory.
RBLNRetCode rbln_malloc_eager(uint32_t torch_device_id, uint64_t size, uint64_t& vaddr);

// Allocates a vmemory entry. This function does not allocate host or device memory when called.
RBLNRetCode rbln_malloc_lazy(uint32_t torch_device_id, uint64_t size, uint64_t& vaddr);

// Releases the vmemory entry. This will free any associated host or device memory.
RBLNRetCode rbln_free(uint64_t vaddr);

// Marks the virtual memory at vaddr as logically zero-initialized without allocating any host
// memory or performing a device transfer. On the next device read, zeros are transferred via a
// temporary buffer; on the next device write, the transfer is skipped entirely. Use this for
// in-place zeroing of large tensors (e.g. KV-cache) to avoid host memory pressure.
RBLNRetCode rbln_mark_zeros(uint64_t vaddr);

RBLNRetCode rbln_set_memory_info(uint64_t vaddr, DataType user_dtype, DataType physical_dtype,
                                 const std::vector<int64_t>& shape);
RBLNRetCode rbln_set_raw_memory_alloc(uint64_t vaddr, uint64_t size);

// Retrieves detailed information for the vmemory entry.
RBLNRetCode rbln_get_memory_info(uint64_t vaddr, MemoryInfo& memory_info_out);

// Borrow a host pointer into the rbln virtual memory at `vaddr`. Triggers a
// device→host sync if the device view is currently authoritative; allocates
// host backing if none exists. After this call the host buffer is read-ready.
// The borrow MUST be released via `rbln_v_return_borrowed` with the returned
// `borrow_id_out`.
//
// Light counterpart of `rebel::torch::rbln_v_borrow_host_ptr` declared in
// `<rebel/torch/rbln_vmem_api.h>`. Distinct from the heavy variant only in
// return convention (RBLNRetCode vs Status); free of vmemory_manager.h
// dependencies so it is safe to call from torch-rbln without dragging in
// absl / model headers.
RBLNRetCode rbln_v_borrow_host_ptr(uint64_t vaddr, uint64_t size,
                                   uintptr_t& host_ptr_out, uint64_t& borrow_id_out);

// Acquire a host pointer for **overwrite-only** access. Same lifecycle as
// `rbln_v_borrow_host_ptr` (must be released via `rbln_v_return_borrowed`),
// but the device→host transfer is skipped even when the entry is
// physical-latest. Callers MUST overwrite the entire region before any
// consumer reads it; otherwise the host view will contain stale data.
// State transitions to USER_VIEW_IS_LATEST on return.
RBLNRetCode rbln_v_acquire_host_ptr_for_overwrite(uint64_t vaddr, uint64_t size,
                                                  uintptr_t& host_ptr_out, uint64_t& borrow_id_out);

// Release a previously borrowed host pointer. If `updated` is true, marks the
// host view as the latest source of truth; the next device consumer performs
// a lazy host→device copy.
RBLNRetCode rbln_v_return_borrowed(uint64_t borrow_id, bool updated);

// Copies the contents from host memory to the virtual memory area.
RBLNRetCode rbln_memcpy_h2v(uintptr_t src_host_ptr, uint64_t dst_vaddr, uint64_t size);

// Copies the contents from the virtual memory area to host memory.
RBLNRetCode rbln_memcpy_v2h(uint64_t src_vaddr, uintptr_t dst_host_ptr, uint64_t size);

// Copies the contents from a virtual memory area to another virtual memory area.
RBLNRetCode rbln_memcpy_v2v(uint64_t src_vaddr, uint64_t dst_vaddr, uint64_t size);

// Casts and copies the contents from host memory to the virtual memory area. The contents at
// the host memory are assumed to be in `from_dtype` and will be converted to `to_dtype`. The
// `size` should be the length in bytes of the host memory. It will fail if the `size` is not the
// multiple of the element size of `from_dtype`. The number of bytes written may differ from `size`
// if the element sizes of `from_dtype` and `to_dtype` differ.
//
// Implementation note: By default, the contents of the vmemory entry corresponding to dst_vaddr are
// synced down to host memory, and the contents of the source host memory(at `src_host_ptr`) are
// actually cast from from_dtype to to_dtype and copies to the host memory area of the vmemory.
// However, if the dtype of the host memory area (i.e., `from_dtype`) is the same as
// the device-side dtype (or physical dtype) of the vmemory entry for dst_vaddr, the data is copied
// directly to device memory without conversion. This can be useful when the user wants the raw
// dlfloat contents to avoid the precision loss.
RBLNRetCode rbln_memcpy_h2v_cast(uintptr_t src_host_ptr, uint64_t dst_vaddr, uint64_t size,
                                 DataType from_dtype, DataType to_dtype);

// Casts and copies the contents from the virtual memory area to host memory. The virtual memory
// contents are assumed to be in `from_dtype` and will be converted to `to_dtype`. The `size`
// should be the length in bytes in virtual memory. It will fail if the `size` is not the
// multiple of the element size of `from_dtype`. The number of bytes written may differ from
// `size` if the element sizes of `from_dtype` and `to_dtype` differ.
//
// Implementation note: By default, the contents of the vmemory entry corresponding to src_vaddr are
// synced down to host memory and the synced contents will actually be cast from `from_dtype` to
// `to_dtype` and copied to the `dst_host_ptr`. However, if the user dtype of the vmemory area
// corresponding to src_vaddr is `from_dtype` and the device dtype is `to_dtype`, the data is copied
// directly from device memory to dst_host_ptr without conversion. This characteristic can be used
// to resolve precision loss issues between dlfloat and float16. This can be useful when the user
// wants the raw dlfloat contents to avoid the precision loss.
RBLNRetCode rbln_memcpy_v2h_cast(uint64_t src_vaddr, uintptr_t dst_host_ptr, uint64_t size,
                                 DataType from_dtype, DataType to_dtype);

// Casts and copies the contents from one virtual memory area to another. The source virtual memory
// contents are assumed to be in `from_dtype` and will be converted to `to_dtype`. The `size`
// should be the length in bytes of the source virtual memory. It will fail if the `size` is not the
// multiple of the element size of `from_dtype`. The number of bytes written may
// differ from `size` if the element sizes of `from_dtype` and `to_dtype` differ.
RBLNRetCode rbln_memcpy_v2v_cast(uint64_t src_vaddr, uint64_t dst_vaddr, uint64_t size,
                                 DataType from_dtype, DataType to_dtype);

}  // namespace rbln
#endif  // RBLN_RUNTIME_API_H
