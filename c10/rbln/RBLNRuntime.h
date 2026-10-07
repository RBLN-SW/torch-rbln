#pragma once

#include <c10/core/Device.h>
#include <c10/core/Stream.h>
#include <c10/rbln/RBLNMacros.h>
#include <rebel/v2/runtime/device.h>
#include <rebel/v2/runtime/stream.h>

#include <cstdint>
#include <memory>
#include <optional>

namespace c10::rbln {

namespace rt = ::rebel::v2::runtime;

/**
 * @brief Where a device pointer points: the buffer of the segment it lies in and its offset.
 *
 * A device pointer is a handle, an address in a host range reserved without access for each
 * segment of device memory. Handles are unique within the process, so a pointer alone names
 * its segment and device, and `data_ptr() + offset` of a view is a handle too.
 */
struct Location {
  // The buffer of the segment the pointer lies in.
  std::shared_ptr<rt::DeviceBuffer> buffer;
  uint64_t offset = 0;
  // Bytes from `offset` to the end of what the pointer may reach: its segment, or its
  // allocation when the caching allocator locates it.
  uint64_t available = 0;
  c10::DeviceIndex device_index = -1;
};

/**
 * @brief The NPU logical device `device_index` runs on, opened on first use; under
 * RBLN_DUMMY_DEVICE, the dummy device of the NPU kind RBLN_FORCE_NPU_NAME names (RBLN-CA25).
 */
C10_RBLN_API std::shared_ptr<rt::Device> runtime_device(c10::DeviceIndex device_index);

/**
 * @brief A new segment of `nbytes` of device memory on `device_index`, as the handle of its
 * first byte, and its release. The caching allocator carves blocks out of segments; a block's
 * handle is its segment's plus its offset.
 */
C10_RBLN_API void* allocate_segment(c10::DeviceIndex device_index, size_t nbytes);
C10_RBLN_API void release_segment(void* handle);

/**
 * @brief The segment `ptr` lies in, or none. Tensors reach only their allocation within it,
 * which `caching::locate` bounds.
 */
C10_RBLN_API std::optional<Location> locate_segment(const void* ptr) noexcept;

/**
 * @brief The runtime stream a torch stream of an RBLN device runs on. StreamId 0 is the
 * device's default stream; the others come from its pool.
 */
C10_RBLN_API std::shared_ptr<rt::Stream> runtime_stream(c10::Stream stream);

/**
 * @brief Adds a stream to the pool of `device_index` and returns its StreamId.
 */
C10_RBLN_API c10::StreamId add_pool_stream(c10::DeviceIndex device_index);

/**
 * @brief Pool streams created so far on `device_index`, the default stream first.
 */
C10_RBLN_API std::vector<std::shared_ptr<rt::Stream>> runtime_streams(c10::DeviceIndex device_index);

} // namespace c10::rbln
