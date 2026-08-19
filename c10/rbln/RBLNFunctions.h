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
 * @brief Whether RBLN_DUMMY_DEVICE mode is active: a host-backed logical device
 * with no NPU, so tensors can be built and compiled without hardware (execution
 * still needs one). Cached after the first call.
 */
C10_RBLN_API bool is_dummy_device();

/**
 * @brief Nothrow view of get_device_count() (returns 0 on any failure), for the
 * liveness predicates that must never throw.
 */
C10_RBLN_API c10::DeviceIndex get_device_count_nothrow() noexcept;

/**
 * @brief Single source of truth: can an rbln_* call be serviced safely now?
 * Runtime loaded (librbln's rbln_runtime_available()), not shutting down, and a
 * device present -- dummy included (it host-backs via the runtime). Never throws.
 */
C10_RBLN_API bool runtime_available() noexcept;

/**
 * @brief Mark the runtime as shutting down so late frees / best-effort ops stop
 * dispatching into a possibly-unmapped runtime. Wired to a Python atexit hook.
 */
C10_RBLN_API void set_runtime_shutting_down(bool value) noexcept;

/**
 * @brief Per-process device-context tracking (CUDA parity with device_allocator).
 *
 * mark_device_context_initialized() records that THIS process has successfully
 * allocated device memory on a logical device; the query functions report whether
 * such allocator/context state exists. They gate the best-effort memory ops and back
 * initialized()/hasPrimaryContext(), so a process with the runtime + a device mapping
 * but no live context (e.g. a vLLM EngineCore parent) is correctly reported as
 * uninitialized. Set-once, monotonic, nothrow. RBLN device use after fork is
 * unsupported; bad-fork detection is not implemented yet.
 */
C10_RBLN_API void mark_device_context_initialized(c10::DeviceIndex device_index) noexcept;
C10_RBLN_API bool device_context_initialized(c10::DeviceIndex device_index) noexcept;
C10_RBLN_API bool any_device_context_initialized() noexcept;

/**
 * @brief Logical device indices this process has initialized (a live context on).
 *
 * The set of devices the device-less torch.accelerator.empty_cache() must flush —
 * every initialized device, not just the current one (CUDA/XPU parity). Extracted
 * as a seam so the "span all initialized devices" selection is unit-testable without
 * observing per-device runtime state (the runtime exposes memory stats for node 0
 * only). Empty when no context is initialized. Order is ascending by index.
 */
C10_RBLN_API std::vector<c10::DeviceIndex> initialized_device_indices();

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
 * @brief Descriptor for one host-to-device slab copy used by memcpy_h2v_multi.
 *
 * Layout matches V2VCopyOp / V2HCopyOp; the type is distinct on purpose. On LP64
 * the three runtime tuple signatures are indistinguishable, so a mixed-up list
 * would compile and DMA a host address as a device vaddr. Separate named types
 * make that a compile error.
 */
struct C10_RBLN_API H2VCopyOp {
  void* dst; // device (rbln virtual address)
  const void* src; // host
  size_t nbytes;
};

/**
 * @brief Descriptor for one device-to-host slab copy used by memcpy_v2h_multi.
 *
 * See H2VCopyOp for why this is a distinct type rather than a shared struct.
 */
struct C10_RBLN_API V2HCopyOp {
  void* dst; // host
  const void* src; // device (rbln virtual address)
  size_t nbytes;
};

/**
 * @brief Batched host-to-device copy through rbln_memcpy_h2v_multi.
 *
 * Empty input is a no-op. Each entry needs nbytes > 0 and non-null dst/src.
 *
 * Caller contract, none of it validated by the runtime:
 *   - every `dst` on the same RBLN device (H2VBatch partitions to hold this)
 *   - `dst` ranges mutually disjoint; `src` ranges may repeat or overlap
 *   - every `src` valid and unchanged until this call returns
 *
 * Entries are unordered and a failed call may have applied some of them (no
 * rollback). rbln_runtime_api.h documents no entry cap, but oversized calls do
 * time out in practice — see the cap in RBLNHostBatch.cpp.
 */
C10_RBLN_API void memcpy_h2v_multi(const std::vector<H2VCopyOp>& copies);

/**
 * @brief Batched device-to-host copy through rbln_memcpy_v2h_multi.
 *
 * Roles swapped: `src` (device) anchors homogeneity, `dst` host ranges must be
 * disjoint, `src` device ranges may repeat. Same unordered / no-rollback
 * semantics and the same lifetime requirement, here on `dst`.
 */
C10_RBLN_API void memcpy_v2h_multi(const std::vector<V2HCopyOp>& copies);

/**
 * @brief Result of a borrow_host_ptr / acquire_host_ptr_for_overwrite call.
 *
 * The borrow id MUST be passed back to `return_borrowed` exactly once to
 * release the underlying virtual-memory entry. A successful borrow always
 * returns a non-zero `borrow_id`; the value `0` is reserved as a sentinel
 * meaning "no live borrow" so cleanup paths can pre-fill a zero in a vector
 * and call `return_borrowed` unconditionally for skipped entries.
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
 * `rbln_data` may be an interior vaddr — a view into the middle of an allocation
 * (e.g. a tensor with a non-zero storage offset). The borrow is offset-correct:
 * the runtime resolves it against the enclosing allocation and returns
 * `host_ptr = allocation_base + (rbln_data - allocation_base)`. The range
 * `[rbln_data, rbln_data + nbytes)` must fit within that allocation, else the
 * borrow fails.
 *
 * The borrow MUST be released via `return_borrowed(result.borrow_id, ...)`.
 *
 * @param rbln_data A pointer to rbln-device memory (typically tensor data_ptr;
 *        an interior/offset vaddr is fine). Must not be nullptr.
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

/**
 * @brief Removes this process's file-offloading temp files and directories.
 *
 * Runtime teardown does this too, so call it only when shutdown may be killed first. Offloaded
 * RBLN tensors must not be used afterwards.
 *
 * @return The number of temp files removed; 0 when the runtime is unavailable.
 */
C10_RBLN_API uint64_t release_offload_temp_storage();

/**
 * @brief Diagnostic: time spent inside librbln boundary calls (borrow / v2v /
 * h2v / ...), so a profiler can split host overhead into "rebel runtime" vs
 * "torch-side dispatch". Gated: when disabled each boundary call pays only one
 * relaxed atomic load (no clock read), preserving ON==OFF latency; an explain
 * region flips it on for its duration. ``rt_timing_get`` fills ``2 * kRtTimingN``
 * uint64 slots as ``[ns, calls]`` per primitive (order matches the internal
 * RtIdx enum: v2v, v2v_multi, borrow, acquire, return, v2h, h2v, v2h_multi,
 * h2v_multi). New primitives are appended so existing slot indices stay put.
 */
constexpr std::size_t kRtTimingN = 9;
C10_RBLN_API void rt_timing_enable(bool on);
C10_RBLN_API void rt_timing_reset();
C10_RBLN_API void rt_timing_get(uint64_t* out);

// torch.rbln.explain() runtime-counter reads — thin pass-throughs to librbln's
// public C-API (see rebel/runtime/api/rbln_runtime_api.h). Process-global, lazy.
// These are LINKED (not dlsym'd): a librbln lacking them fails extension load.
// Build-time headers and library are resolved from the same source -- the
// rebel-compiler wheel or a REBEL_HOME tree -- so the linked ABI matches.
// The per-reason axes are positional; their meaning is interpreted Python-side, so
// no internal classification name crosses this boundary.
C10_RBLN_API uint32_t rt_prof_hidden_num();
C10_RBLN_API void rt_prof_hidden_get(uint64_t* counts, uint64_t* bytes, uint32_t n);
C10_RBLN_API uint32_t rt_prof_reject_num();
C10_RBLN_API void rt_prof_reject_get(uint64_t* counts, uint64_t* bytes, uint32_t n);
C10_RBLN_API void rt_prof_host_sync_d2h(uint64_t* count, uint64_t* bytes);
C10_RBLN_API void rt_prof_host_sync_h2d(uint64_t* count, uint64_t* bytes);
C10_RBLN_API void rt_prof_memory(uint64_t* current, uint64_t* peak);
C10_RBLN_API void rt_prof_reset();

} // namespace c10::rbln
