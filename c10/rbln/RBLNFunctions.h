#pragma once

#include <c10/core/CachingDeviceAllocator.h>
#include <c10/core/Device.h>
#include <c10/core/ScalarType.h>
#include <c10/core/Stream.h>
#include <c10/rbln/RBLNMacros.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace c10::rbln {

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
 * @brief What the NPUs behind one logical device are.
 *
 * total_memory sums the device's physical NPUs; num_chiplet and memory_per_chiplet stay per NPU.
 */
struct C10_RBLN_API DeviceProperties {
  std::string name;
  uint64_t total_memory = 0;
  uint64_t memory_per_chiplet = 0;
  uint32_t num_chiplet = 0;
  uint32_t npu_count = 0;
};

/**
 * @brief Reports what the logical device at device_index is, summed over its NPUs.
 *
 * Needs real hardware: raises in dummy mode, when no NPU is mapped to the index, when the runtime
 * cannot answer for one of them, or when the mapping aggregates unlike NPUs.
 */
C10_RBLN_API DeviceProperties get_device_properties(c10::DeviceIndex device_index);

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
 * @brief Returns the torch device id of the allocation a device pointer lies in.
 *
 * @param data A pointer to device memory, possibly inside an allocation.
 * @return The torch device id (as a c10::DeviceIndex) backing the pointer.
 */
C10_RBLN_API c10::DeviceIndex get_torch_device_id(const void* data);

/**
 * @brief Whether RBLN_DUMMY_DEVICE mode is active: a host-backed logical device
 * with no NPU, so tensors can be built and compiled without hardware (execution
 * still needs one). Cached after the first call.
 */
C10_RBLN_API bool is_dummy_device();

/**
 * @brief Nothrow view of get_device_count() (returns 0 on any failure). Warns with the
 * first line of the error only, on every failure.
 *
 * Backs torch.rbln.device_count(). ATen/DeviceAccelerator.h: deviceCount() "is *REQUIRED*
 * to not raise any exception".
 */
C10_RBLN_API c10::DeviceIndex get_device_count_nothrow() noexcept;

/**
 * @brief Throwing counterpart, named after c10::cuda::device_count_ensure_non_zero().
 *
 * Raises the detailed RBLN_* configuration error, or a clear "no devices" message.
 * Use at the point where a device is actually required; never on a probe path.
 */
C10_RBLN_API c10::DeviceIndex device_count_ensure_non_zero();

/**
 * @brief Claim the planned logical devices with the runtime (rbln_register_device_id).
 *
 * Idempotent, and normally unnecessary: to_device_id() already does it. Call it explicitly
 * only on a path that hands a device index to the runtime while bypassing to_device_id(),
 * such as RCCL init in ProcessGroupRBLN. Raises on an invalid mapping.
 */
C10_RBLN_API void commit_device_mapping();

/**
 * @brief Single source of truth: is RBLN usable as an accelerator right now?
 *
 * Runtime loaded, not shutting down, and at least one usable logical device -- dummy
 * included, since it host-backs through the runtime and still needs a valid mapping.
 * Never throws. Bound to both Python is_available() and RBLNHooksInterface::hasRBLN(),
 * so the two cannot diverge.
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
 * @brief Fills `nbytes` of device memory from `rbln_data` with zeros, in the order of the
 * current stream.
 *
 * @param rbln_data A pointer to device memory, possibly inside an allocation.
 * @param nbytes The number of bytes to fill.
 */
C10_RBLN_API void fill_zeros(void* rbln_data, size_t nbytes);

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
 * @brief Copies data from host memory to device memory in the order of the current stream.
 *
 * The host memory need not stay valid past the call, which returns once the copy is done.
 *
 * @param rbln_dst_data A pointer to the destination device memory.
 * @param cpu_src_data A pointer to the source host memory.
 * @param nbytes The number of bytes to copy (must be positive).
 */
C10_RBLN_API void memcpy_h2v_async(void* rbln_dst_data, const void* cpu_src_data, size_t nbytes);

/**
 * @brief Copies data from device memory to host memory in the order of the current stream.
 *
 * Returns once the host memory holds the data.
 *
 * @param cpu_dst_data A pointer to the destination host memory.
 * @param rbln_src_data A pointer to the source device memory.
 * @param nbytes The number of bytes to copy (must be positive).
 */
C10_RBLN_API void memcpy_v2h_async(void* cpu_dst_data, const void* rbln_src_data, size_t nbytes);

/**
 * @brief Copies data between two device memory regions in the order of the current stream.
 *
 * Devices of different contexts copy through the host, as memcpy_v2v does.
 *
 * @param rbln_dst_data A pointer to the destination device memory.
 * @param rbln_src_data A pointer to the source device memory.
 * @param nbytes The number of bytes to copy (must be positive).
 */
C10_RBLN_API void memcpy_v2v_async(void* rbln_dst_data, const void* rbln_src_data, size_t nbytes);

/**
 * @brief Blocks the host until the device is idle: waits the device's pending async
 * transfers, then drains every stream on it. Wider than synchronize_stream(), which
 * drains one stream and leaves work issued on another stream in flight.
 *
 * @param device_index The RBLN device to synchronize.
 */
C10_RBLN_API void synchronize(c10::DeviceIndex device_index);

// Streams / events (torch.Stream / torch.Event). A c10::Stream carries the per-device
// stream id as its StreamId (StreamId 0 is the default stream). An event surfaces as
// its opaque handle, which is non-zero for a valid event (so it never aliases nullptr).

/**
 * @brief Returns the current stream for the device (default stream if none set).
 */
C10_RBLN_API c10::Stream get_current_stream(c10::DeviceIndex device_index);

/**
 * @brief Returns the device's default stream (StreamId 0).
 */
C10_RBLN_API c10::Stream get_default_stream(c10::DeviceIndex device_index);

/**
 * @brief Returns a stream from a fixed per-device round-robin pool, filled one slot at
 * a time. Past the pool size, requests reuse earlier streams. Priorities do not exist
 * on RBLN and are not part of this API.
 */
C10_RBLN_API c10::Stream get_stream_from_pool(c10::DeviceIndex device_index);

/**
 * @brief Makes `stream` the current stream on its device (thread-local).
 */
C10_RBLN_API void set_current_stream(c10::Stream stream);

/**
 * @brief Non-blocking: true iff all work submitted to `stream` has completed.
 */
C10_RBLN_API bool query_stream(c10::Stream stream);

/**
 * @brief Blocks the host until all work on `stream` has completed.
 */
C10_RBLN_API void synchronize_stream(c10::Stream stream);

/**
 * @brief Creates an event on the device and returns its opaque uint64 handle.
 */
C10_RBLN_API uint64_t event_create(c10::DeviceIndex device_index);

/**
 * @brief Destroys an event. Never throws (called from destructors); no-op on 0.
 */
C10_RBLN_API void event_destroy(uint64_t event) noexcept;

/**
 * @brief Snapshots `stream`'s current position into the event (re-record overwrites).
 */
C10_RBLN_API void event_record(uint64_t event, c10::Stream stream);

/**
 * @brief Makes `stream` wait for `event`. Same-device: does not block the host.
 * Cross-device waits are not supported and degrade to a host-side wait on the event
 * (correct, but serializes the host).
 */
C10_RBLN_API void event_block(c10::Stream stream, uint64_t event);

/**
 * @brief Non-blocking: true iff the work recorded into the event has completed.
 */
C10_RBLN_API bool event_query(uint64_t event);

/**
 * @brief Blocks the host until the work recorded into the event has completed.
 */
C10_RBLN_API void event_synchronize(uint64_t event);

/**
 * @brief Descriptor for one device-to-device slab copy used by memcpy_v2v_multi.
 */
struct C10_RBLN_API V2VCopyOp {
  void* dst;
  const void* src;
  size_t nbytes;
};

/**
 * @brief Batched device-to-device copy, in the order of the current stream.
 *
 * Empty input is a no-op. Each entry must have nbytes > 0 and non-null dst/src.
 * Entries may run in any order, so overlapping ranges across entries yield undefined
 * behaviour.
 */
C10_RBLN_API void memcpy_v2v_multi(const std::vector<V2VCopyOp>& copies);

/**
 * @brief Descriptor for one host-to-device slab copy used by memcpy_h2v_multi.
 *
 * Layout matches V2VCopyOp / V2HCopyOp; the type is distinct on purpose, so a mixed-up list
 * that would copy a host address as a device one is a compile error.
 */
struct C10_RBLN_API H2VCopyOp {
  void* dst; // device
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
  const void* src; // device
  size_t nbytes;
};

/**
 * @brief Batched host-to-device copy, in the order of the current stream.
 *
 * Empty input is a no-op. Each entry needs nbytes > 0 and non-null dst/src. `dst` ranges
 * must be mutually disjoint; `src` ranges may repeat or overlap. Entries are unordered and
 * a failed call may have applied some of them (no rollback).
 */
C10_RBLN_API void memcpy_h2v_multi(const std::vector<H2VCopyOp>& copies);

/**
 * @brief Batched device-to-host copy, in the order of the current stream.
 *
 * Roles swapped: `dst` host ranges must be disjoint, `src` device ranges may repeat. Same
 * unordered / no-rollback semantics.
 */
C10_RBLN_API void memcpy_v2h_multi(const std::vector<V2HCopyOp>& copies);

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
 * Scope is the caching allocator of the context THIS process holds on `device`, the
 * same scope torch.cuda.memory_stats() reports. It counts every physical NPU the
 * logical device maps to, but not direct device allocations (weights), and not another
 * process using the same NPU. For a device-wide figure, use rbln-smi.
 *
 * @param device The input device.
 * @return A map containing memory statistics.
 */
C10_RBLN_API std::map<std::string, uint64_t> memory_stats(const c10::Device& device);

/**
 * @brief Returns memory allocator statistics broken down per chiplet.
 *
 * Same keys as memory_stats(), each prefixed with "npu.<n>.chiplet.<c>.". A device runs
 * out on its heaviest chiplet, which the aggregate memory_stats() hides. npu.<n> is the
 * n-th physical NPU of this logical device (see RBLN_NPUS_PER_DEVICE), so a 1:1 mapping
 * yields npu.0 only.
 *
 * Scope is the caching allocator of this process's context on `device`. Other direct
 * device allocations are not counted, and a second process on the same NPU is
 * invisible here -- see memory_stats().
 *
 * @param device The input device.
 * @return A map containing per-chiplet memory statistics.
 */
C10_RBLN_API std::map<std::string, uint64_t> memory_stats_per_chiplet(const c10::Device& device);

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
 * @brief Returns the free and total device DRAM of `device` in bytes, as (free, total).
 *
 * The kernel driver's figure for the NPU as a whole -- every process, not this process's
 * caching allocator (see memory_stats()) -- which is what torch.cuda.mem_get_info() reports
 * on CUDA. Summed over the physical NPUs of the logical device; one tensor still lives on
 * one NPU. A reading, not a reservation.
 *
 * Raises when the installed UMD/KMD does not provide the query or under RBLN_DUMMY_DEVICE:
 * there is no figure to report, and a guess here would size a KV cache wrong.
 *
 * @param device The input device.
 * @return (free bytes, total bytes).
 */
C10_RBLN_API std::pair<size_t, size_t> mem_get_info(const c10::Device& device);

/**
 * @brief Returns the driver's device-wide DRAM usage of `device` broken down per chiplet.
 *
 * Keys "npu.<n>.chiplet.<c>.{total,used,free}" plus "npu.<n>.{total,used,free}" for each
 * physical NPU of the logical device; npu.<n> is the NPU's position as in
 * memory_stats_per_chiplet(). Same scope and failure modes as mem_get_info(). A buffer lives
 * on one chiplet, which the device total hides.
 *
 * @param device The input device.
 * @return A map from key to bytes.
 */
C10_RBLN_API std::map<std::string, uint64_t> mem_get_info_per_chiplet(const c10::Device& device);

/**
 * @brief What the driver says of the DRAM of one chiplet of an NPU.
 */
struct C10_RBLN_API ChipletMemory {
  uint64_t total = 0;
  uint64_t free = 0;
};

/**
 * @brief Lays out per-NPU chiplet memory as the mem_get_info_per_chiplet() map.
 *
 * `npus[n]` becomes the "npu.<n>." entries: "npu.<n>.chiplet.<c>.{total,used,free}" and
 * "npu.<n>.{total,used,free}". Pure; this is the key mapping mem_get_info_per_chiplet() returns.
 *
 * @param npus The chiplets of each physical NPU of a logical device, in mapping order.
 * @return A map from key to bytes.
 */
C10_RBLN_API std::map<std::string, uint64_t> per_chiplet_memory_map(const std::vector<std::vector<ChipletMemory>>& npus);

/**
 * @brief Diagnostic: time spent inside runtime copy calls, so a profiler can split host
 * overhead into "runtime" vs "torch-side dispatch". Gated: when disabled each call pays only
 * one relaxed atomic load (no clock read), preserving ON==OFF latency; an explain region
 * flips it on for its duration. ``rt_timing_get`` fills ``2 * kRtTimingN`` uint64 slots as
 * ``[ns, calls]`` per primitive, in the order of the internal RtIdx enum: v2v, v2v_multi,
 * v2h, h2v, v2h_multi, h2v_multi.
 */
constexpr std::size_t kRtTimingN = 6;
C10_RBLN_API void rt_timing_enable(bool on);
C10_RBLN_API void rt_timing_reset();
C10_RBLN_API void rt_timing_get(uint64_t* out);

} // namespace c10::rbln
