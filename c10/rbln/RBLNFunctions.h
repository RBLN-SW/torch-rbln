#pragma once

#include <c10/core/CachingDeviceAllocator.h>
#include <c10/core/Device.h>
#include <c10/core/ScalarType.h>
#include <c10/rbln/RBLNMacros.h>
#include <rebel/runtime/api/rbln_runtime_api.h>

#include <cstdint>
#include <map>
#include <string>

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
 * @brief Retrieves memory information for a given data pointer.
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
 * @brief Borrow a CPU-accessible host pointer backing an RBLN virtual memory region.
 *
 * Wraps rebel::torch::rbln_v_borrow_host_ptr. Between borrow and return, the
 * returned host pointer is safe for CPU reads/writes. Pair with return_borrowed().
 */
C10_RBLN_API uintptr_t borrow_host_ptr(const void* data, size_t nbytes, uint64_t& borrow_id_out);

/**
 * @brief Return a previously borrowed host pointer. updated=true marks host
 *        memory as latest truth for the next device access.
 */
C10_RBLN_API void return_borrowed(uint64_t borrow_id, bool updated);

/**
 * @brief Mark an rbln v-memory region as logically zero-initialised. Used
 *        before borrow_host_ptr to tell rebel the existing contents are
 *        irrelevant so the implicit d->h sync can be skipped.
 */
C10_RBLN_API void mark_zeros(const void* data);

} // namespace c10::rbln
