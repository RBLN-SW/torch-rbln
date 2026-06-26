#pragma once

#include <c10/core/CachingDeviceAllocator.h>
#include <c10/core/Device.h>
#include <c10/core/ScalarType.h>
#include <c10/rbln/RBLNMacros.h>
#include <rebel/runtime/api/rbln_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace c10::rbln {

/**
 * @brief Converts a PyTorch data type to the corresponding RBLN data type.
 *
 * @param dtype The PyTorch data type to convert.
 * @return The corresponding RBLN data type.
 */
C10_RBLN_API ::rbln::DataType to_rbln_dtype(c10::ScalarType dtype);

/**
 * @brief Converts memory information to a human-readable string.
 *
 * @param memory_info The memory information to convert.
 * @return A string representation of the memory information.
 */
C10_RBLN_API std::string to_string(const ::rbln::MemoryInfo& memory_info);

/**
 * @brief Returns the number of available RBLN devices in the system.
 *
 * This function queries the system to determine how many RBLN devices are
 * available for use. The returned count can be used to iterate through
 * available devices or validate device indices.
 *
 * @return The number of available RBLN devices (non-negative integer).
 */
C10_RBLN_API c10::DeviceIndex get_device_count();

/**
 * @brief Returns the number of physical NPUs visible to this process.
 *
 * Queries the runtime for how many physical NPUs are available, regardless of
 * RSD mode. Unlike get_device_count() (logical device count), this returns
 * the actual physical NPU count; get_device_count() may return 1 when RSD is active.
 *
 * @return The number of physical NPUs (non-negative integer).
 */
C10_RBLN_API c10::DeviceIndex get_physical_device_count();

/**
 * @brief Returns the currently active RBLN device.
 *
 * This function retrieves the device that is currently set as the active
 * device for RBLN operations. All subsequent device operations will use
 * this device unless explicitly changed.
 *
 * @return The currently active RBLN device.
 */
C10_RBLN_API c10::DeviceIndex get_device_index();

/**
 * @brief Sets the current active RBLN device.
 *
 * This function changes the active device to the specified device. All
 * subsequent device operations (memory allocation, kernel launches, etc.)
 * will use this device until changed again.
 *
 * @param device_index The RBLN device to set as the current active device.
 */
C10_RBLN_API void set_device_index(c10::DeviceIndex device_index);

/**
 * @brief Atomically sets the current device and returns the previous device.
 *
 * This function performs an atomic exchange operation: it sets the current
 * active device to the specified device and returns the device that was
 * previously active. This is useful for temporarily switching devices and
 * restoring the original device later.
 *
 * @param device_index The RBLN device to set as the current active device.
 * @return The device that was active before this call.
 */
C10_RBLN_API c10::DeviceIndex exchange_device_index(c10::DeviceIndex device_index);

/**
 * @brief Returns the torch device id backing a device pointer.
 *
 * Lightweight counterpart to get_memory_info() for the common case where only
 * the owning device is needed: it calls rbln_get_torch_device_id_from_vaddr()
 * directly and avoids the full VMemory JSON round-trip that get_memory_info()
 * performs. Prefer this on performance hot paths.
 *
 * @param data A pointer to device memory.
 * @return The torch device id (as a c10::DeviceIndex) backing the pointer.
 */
C10_RBLN_API c10::DeviceIndex get_torch_device_id(const void* data);

/**
 * @brief Retrieves memory information for a given data pointer.
 *
 * @note This performs a full VMemory JSON round-trip and is expensive — keep it
 * off performance hot paths. When only the owning device is needed, use
 * get_torch_device_id() instead.
 *
 * @param data A pointer to device memory.
 * @return Memory information associated with the given data pointer.
 */
C10_RBLN_API ::rbln::MemoryInfo get_memory_info(const void* data);

/**
 * @brief Checks if RBLN uses eager memory allocation.
 *
 * @return true if eager memory allocation is enabled, false otherwise.
 */
C10_RBLN_API bool is_eager_malloc();

/**
 * @brief Configure ``target``'s device allocation to match ``ref``'s layout and
 *        dtype, without copying data.
 *
 * A subsequent device-to-device copy between ``target`` and ``ref`` then stays
 * on the fast path. ``ref`` must already be device-resident. Used to make a
 * staging buffer match a KV cache's layout so the upload and the per-slot
 * device-to-device scatter are both fast.
 *
 * @param target_data Destination tensor's device pointer to configure.
 * @param ref_data    Reference tensor's device pointer whose layout is mirrored.
 */
C10_RBLN_API void set_device_layout_like(void* target_data, const void* ref_data);

/**
 * @brief Whether host-backed dummy device mode is active (RBLN_DUMMY_DEVICE).
 *
 * When true, the allocation and transfer functions below run on host memory
 * instead of the RBLN runtime, so device tensors can be built and compiled
 * without an NPU (execution still needs hardware). Cached after the first call.
 *
 * @return true if dummy device mode is enabled, false otherwise.
 */
C10_RBLN_API bool is_dummy_device();

/**
 * @brief Allocates memory on the specified RBLN device.
 *
 * This function allocates a contiguous block of memory on the given RBLN
 * device. The allocated memory is uninitialized and must be freed using
 * the corresponding free() function when no longer needed.
 *
 * @param device_index The RBLN device on which to allocate memory.
 * @param nbytes The number of bytes to allocate (must be positive).
 * @return A pointer to the allocated device memory, or nullptr on failure.
 */
C10_RBLN_API void* malloc(c10::DeviceIndex device_index, size_t nbytes);

/**
 * @brief Marks the virtual memory as logically zero-initialized.
 *
 * Sets the VMemory sync state to EMPTY_INIT_WITH_ZERO without allocating host memory or
 * performing any device transfer. On the next device read, zeros are transferred via a
 * temporary buffer; on the next device write, the transfer is skipped entirely.
 *
 * This is the preferred implementation of aten::zero_ for RBLN tensors. It avoids host
 * memory allocation, which is critical for large tensors such as KV-cache.
 *
 * @param rbln_data Pointer to the RBLN virtual memory (tensor data_ptr).
 */
C10_RBLN_API void mark_zeros(const void* rbln_data);

/**
 * @brief Frees memory allocated on an RBLN device.
 *
 * This function deallocates memory that was previously allocated using
 * malloc(). The device index is automatically determined from the pointer.
 *
 * @param data A pointer to device memory previously allocated by malloc().
 */
C10_RBLN_API void free(void* data);

/**
 * @brief Non-throwing free() for `noexcept` contexts (the c10 DataPtr deleter).
 *
 * The deleter runs in a noexcept destructor, so a throwing free() would
 * std::terminate; this logs on failure instead.
 */
C10_RBLN_API void free_nothrow(void* data) noexcept;

/**
 * @brief Copies data from host memory to device memory.
 *
 * This function performs a synchronous copy operation from host memory to
 * device memory.
 *
 * @param rbln_dst_data A pointer to the destination device memory.
 * @param cpu_src_data A pointer to the source host memory.
 * @param nbytes The number of bytes to copy (must be positive).
 */
C10_RBLN_API void memcpy_h2v(void* rbln_dst_data, const void* cpu_src_data, size_t nbytes);

/**
 * @brief Copies data from device memory to host memory.
 *
 * This function performs a synchronous copy operation from device memory
 * to host memory.
 *
 * @param cpu_dst_data A pointer to the destination host memory.
 * @param rbln_src_data A pointer to the source device memory.
 * @param nbytes The number of bytes to copy (must be positive).
 */
C10_RBLN_API void memcpy_v2h(void* cpu_dst_data, const void* rbln_src_data, size_t nbytes);

/**
 * @brief Copies data from device memory to device memory.
 *
 * This function performs a synchronous copy operation between two device
 * memory locations. The source and destination can be on the same or different
 * devices.
 *
 * @param rbln_dst_data A pointer to the destination device memory.
 * @param rbln_src_data A pointer to the source device memory.
 * @param nbytes The number of bytes to copy (must be positive).
 */
C10_RBLN_API void memcpy_v2v(void* rbln_dst_data, const void* rbln_src_data, size_t nbytes);

/**
 * @brief Asynchronously copies data from host memory to device memory.
 *
 * Falls back to synchronous copy when async is not possible (e.g., when
 * the vmem entry does not have a simple device layout).
 *
 * @param rbln_dst_data A pointer to the destination device memory.
 * @param cpu_src_data A pointer to the source host memory.
 * @param nbytes The number of bytes to copy (must be positive).
 */
C10_RBLN_API void memcpy_h2v_async(void* rbln_dst_data, const void* cpu_src_data, size_t nbytes);

/**
 * @brief Asynchronously copies data from device memory to host memory.
 *
 * Falls back to synchronous copy when async is not possible (e.g., when
 * the vmem entry does not have a simple device layout).
 *
 * @param cpu_dst_data A pointer to the destination host memory.
 * @param rbln_src_data A pointer to the source device memory.
 * @param nbytes The number of bytes to copy (must be positive).
 */
C10_RBLN_API void memcpy_v2h_async(void* cpu_dst_data, const void* rbln_src_data, size_t nbytes);

/**
 * @brief Asynchronously copies data between two device memory regions.
 *
 * Same-device copies use the async runtime entrypoint. Cross-device copies
 * fall back to synchronous memcpy_v2v (host-bounce), matching the sync
 * version's case split.
 *
 * @param rbln_dst_data A pointer to the destination device memory.
 * @param rbln_src_data A pointer to the source device memory.
 * @param nbytes The number of bytes to copy (must be positive).
 */
C10_RBLN_API void memcpy_v2v_async(void* rbln_dst_data, const void* rbln_src_data, size_t nbytes);

/**
 * @brief Waits for all pending async transfers on the given device to complete.
 *
 * @param device_index The RBLN device to synchronize.
 */
C10_RBLN_API void synchronize(c10::DeviceIndex device_index);

/**
 * @brief Descriptor for one device-to-device slab copy used by memcpy_v2v_multi.
 */
struct C10_RBLN_API V2VCopyOp {
  void* dst;
  const void* src;
  size_t nbytes;
};

/**
 * @brief Batched device-to-device copy through rbln_memcpy_v2v_multi.
 *
 * Empty input is a no-op. Each entry must have nbytes > 0 and non-null dst/src.
 *
 * Caller contract (NOT validated): every entry's src AND dst must reside on the
 * same RBLN device. The bulk runtime entrypoint targets one device per call and
 * does NOT host-bounce cross-device entries — mixing devices yields silent wrong
 * results. Callers with heterogeneous inputs should partition up front or use
 * memcpy_v2v per entry; V2VBatch handles this internally via fallback.
 *
 * The runtime may parallelise or reorder entries, so overlapping ranges across
 * entries yield undefined behaviour.
 */
C10_RBLN_API void memcpy_v2v_multi(const std::vector<V2VCopyOp>& copies);

/**
 * @brief Result of a borrow_host_ptr / acquire_host_ptr_for_overwrite call.
 *
 * The borrow id MUST be passed back to `return_borrowed` exactly once to
 * release the underlying virtual-memory entry. A successful borrow always
 * returns a non-zero `borrow_id`; the value `0` is reserved as a sentinel
 * meaning "no live borrow" so cleanup paths can pre-fill a zero in a vector
 * and call `return_borrowed` unconditionally for skipped entries.
 *
 * In RBLN_DUMMY_DEVICE mode the borrow is an identity host view: the id is
 * non-zero (honoring the contract) but `return_borrowed` is a no-op — there is
 * no ledger, so double-release is not detected and `updated=false` does NOT roll
 * back writes already made through the pointer.
 */
struct BorrowedHostPtr {
  uintptr_t host_ptr;
  uint64_t borrow_id;
};

/**
 * @brief Borrow a host pointer into the rbln virtual memory backing
 * `rbln_data`. Triggers a device→host sync if the device view is currently
 * authoritative; allocates host backing if none exists. After this call the
 * host buffer is read-ready.
 *
 * The borrow MUST be released via `return_borrowed(result.borrow_id, ...)`.
 *
 * @param rbln_data A pointer to rbln-device memory (typically tensor data_ptr).
 *        Must not be nullptr.
 * @param nbytes Number of bytes to borrow. Must be positive — callers with a
 *        legitimate zero-byte case must short-circuit before invoking.
 * @return Host pointer + non-zero borrow id; throws c10::Error via RBLN_CHECK
 *         on failure (invalid args, rebel-side error).
 */
C10_RBLN_API BorrowedHostPtr borrow_host_ptr(const void* rbln_data, size_t nbytes);

/**
 * @brief Non-throwing variant of `borrow_host_ptr`. Returns `std::nullopt` when
 * the runtime rejects the borrow (e.g. the backing entry is in a sub-state with
 * no host-mappable user view), instead of throwing. Use this where a borrow
 * failure is an expected, recoverable condition with a copy-based fallback —
 * it scopes failure handling to exactly the borrow call (other errors still
 * surface normally) rather than swallowing a broad c10::Error catch.
 *
 * @return Host pointer + non-zero borrow id on success; `std::nullopt` if the
 *         runtime could not provide an in-place host view. Invalid args
 *         (nullptr / zero nbytes) also return `std::nullopt`.
 */
C10_RBLN_API std::optional<BorrowedHostPtr> try_borrow_host_ptr(const void* rbln_data, size_t nbytes);

/**
 * @brief Acquire a host pointer for **overwrite-only** access into the rbln
 * virtual memory backing `rbln_data`. Same lifecycle as `borrow_host_ptr`,
 * but the device→host transfer is skipped even when the entry is
 * physical-latest.
 *
 * IMPORTANT: callers MUST overwrite **the entire borrowed region** before
 * `return_borrowed(..., updated=true)`; otherwise the region surfaces stale
 * bytes to subsequent device consumers. Use `borrow_host_ptr` instead if a
 * partial overwrite is intended.
 *
 * @param rbln_data A pointer to rbln-device memory. Must not be nullptr.
 * @param nbytes Number of bytes to acquire (must be positive).
 * @return Host pointer + non-zero borrow id; throws c10::Error via RBLN_CHECK
 *         on failure.
 */
C10_RBLN_API BorrowedHostPtr acquire_host_ptr_for_overwrite(void* rbln_data, size_t nbytes);

/**
 * @brief Non-throwing variant of `acquire_host_ptr_for_overwrite`. Returns
 * `std::nullopt` when the runtime rejects the acquire (same recoverable
 * sub-states as `try_borrow_host_ptr`) instead of throwing. Use this where the
 * caller has a copy-based fallback (e.g. a fresh `at::empty` + writeback).
 *
 * @return Host pointer + non-zero borrow id on success; `std::nullopt` if the
 *         runtime could not provide an in-place host view. Invalid args
 *         (nullptr / zero nbytes) also return `std::nullopt`.
 */
C10_RBLN_API std::optional<BorrowedHostPtr> try_acquire_host_ptr_for_overwrite(void* rbln_data, size_t nbytes);

/**
 * @brief Release a previously borrowed host pointer.
 *
 * @param borrow_id The id returned from `borrow_host_ptr` /
 *        `acquire_host_ptr_for_overwrite`. The value `0` is the
 *        "no live borrow" sentinel and is treated as a no-op so cleanup
 *        paths can release vectors of optional borrows uniformly.
 * @param updated If true, marks the host view as the latest source of truth;
 *        the next device consumer performs a lazy host→device copy. Must be
 *        true after a successful `acquire_host_ptr_for_overwrite` write
 *        sequence; otherwise the overwritten bytes are discarded.
 */
C10_RBLN_API void return_borrowed(uint64_t borrow_id, bool updated);

/**
 * @brief Returns comprehensive device memory statistics.
 *
 * Retrieves all memory metrics from the RBLN runtime in a single call and
 * returns a fully populated c10::CachingDeviceAllocator::DeviceStats.
 *
 * @param device The input device.
 * @return A populated DeviceStats snapshot for the device.
 */
C10_RBLN_API c10::CachingDeviceAllocator::DeviceStats get_device_stats(const c10::Device& device);

/**
 * @brief Releases all unoccupied cached memory currently held by the caching allocator.
 *
 * @param device The input device.
 */
C10_RBLN_API void empty_cache(const c10::Device& device);

/**
 * @brief Returns a dictionary of accelerator device memory allocator statistics.
 *
 * @param device The input device.
 * @return A map containing memory statistics.
 */
C10_RBLN_API std::map<std::string, uint64_t> memory_stats(const c10::Device& device);

/**
 * @brief Resets the "accumulated" (historical) stats tracked by the current accelerator memory allocator.
 *
 * @param device The input device.
 */
C10_RBLN_API void reset_accumulated_memory_stats(const c10::Device& device);

/**
 * @brief Resets the "peak" stats tracked by the current accelerator memory allocator.
 *
 * Peak memory statistics represent the maximum (highest) memory usage values that have
 * been reached since the last reset.
 *
 * This function resets all peak statistics (such as peak allocated memory and peak
 * reserved memory) to their current values, effectively starting a new tracking period
 * from the current memory state. This is useful for measuring memory usage during
 * specific phases of execution or after certain operations.
 *
 * @param device The input device.
 */
C10_RBLN_API void reset_peak_memory_stats(const c10::Device& device);

/**
 * @brief Enables or disables process-wide file offloading for RBLN virtual memory.
 *
 * When enabled, host-side regions backing RBLN tensors may be paged out to disk to reduce
 * host memory pressure. The setting applies to all RBLN devices initialized in the current
 * process and takes effect for subsequent vmemory operations; existing user views are not
 * migrated by the toggle itself.
 *
 * @param enabled If true, enable file offloading; if false, disable it.
 */
C10_RBLN_API void set_file_offloading_enabled(bool enabled);

} // namespace c10::rbln
