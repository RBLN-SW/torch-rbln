#pragma once

#include <c10/core/CachingDeviceAllocator.h>
#include <c10/core/Device.h>
#include <c10/core/Stream.h>
#include <c10/rbln/RBLNMacros.h>
#include <c10/rbln/RBLNRuntime.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>

namespace c10::rbln::held {
struct Type;
} // namespace c10::rbln::held

namespace c10::rbln::caching {

/**
 * @brief Device memory for tensors, cached per device the way CUDA's caching allocator does.
 *
 * Blocks of up to kSmallSize bytes are carved out of kSmallSegment segments at kSmallRound
 * granularity; larger ones are whole multiples of kLargeRound, so they start on a page. A freed
 * block is reused on the stream it was allocated on; streams it was recorded on must reach the
 * point of the free first.
 */
constexpr size_t kSmallSize = size_t{1} << 20;
constexpr size_t kSmallRound = 512;
constexpr size_t kSmallSegment = size_t{2} << 20;
constexpr size_t kLargeRound = size_t{2} << 20;

C10_RBLN_API void* allocate(c10::DeviceIndex device_index, size_t nbytes);
C10_RBLN_API void release(void* ptr);
C10_RBLN_API void record_stream(void* ptr, c10::Stream stream);
/**
 * @brief Where `ptr` points within the live allocation it lies in, with `available` the bytes
 * from it to the allocation's end; throws, or is none, if it lies in no live allocation. What
 * locates bytes reads or writes them as torch holds a tensor, so an allocation a program holds
 * otherwise (see RBLNHeld.h) is put back as torch holds it first.
 */
C10_RBLN_API Location locate(const void* ptr);
C10_RBLN_API std::optional<Location> try_locate(const void* ptr);

/**
 * @brief Where `ptr` points, as `try_locate` says, leaving the allocation as it is held: the
 * allocation's first byte, and how a program holds it, if one does.
 */
struct Held {
  Location location;
  void* start = nullptr;
  std::shared_ptr<const held::Type> type;
};
C10_RBLN_API std::optional<Held> try_locate_held(const void* ptr);

/**
 * @brief Whether a program holds any live allocation.
 */
C10_RBLN_API bool any_held();

/**
 * @brief Makes this thread's locates leave held allocations as they are, or put them back as torch
 * holds them; returns the setting before.
 */
C10_RBLN_API bool locate_as_held(bool as_held);
C10_RBLN_API bool locating_as_held();

/**
 * @brief Runs `convert`, which puts the bytes of the allocation starting at `start` in `type`, or
 * back as torch holds them when `type` is none, and counts the allocation so until it is freed.
 * Other threads locating bytes wait while it runs; `convert` locates them as they are.
 */
C10_RBLN_API void convert_held(
    const void* start,
    const std::function<void()>& convert,
    std::shared_ptr<const held::Type> type);
C10_RBLN_API void empty_cache(c10::DeviceIndex device_index);
C10_RBLN_API c10::CachingDeviceAllocator::DeviceStats device_stats(c10::DeviceIndex device_index);
// torch.cuda.memory_stats()'s keys, for the aggregate pool.
C10_RBLN_API std::map<std::string, uint64_t> stats_map(c10::DeviceIndex device_index);
C10_RBLN_API void reset_peak(c10::DeviceIndex device_index);
C10_RBLN_API void reset_accumulated(c10::DeviceIndex device_index);

} // namespace c10::rbln::caching
