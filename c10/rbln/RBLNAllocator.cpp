#include <c10/core/CachingDeviceAllocator.h>
#include <c10/rbln/RBLNFunctions.h>
#include <c10/rbln/RBLNLogging.h>

namespace c10::rbln {

namespace {

void raw_delete(void* data) {
  if (data != nullptr) {
    RBLN_LOG_DEBUG("Freeing memory at {}", fmt::ptr(data));
    c10::rbln::free(data);
  }
}

} // namespace

struct RBLNAllocator final : public c10::DeviceAllocator {
  RBLNAllocator() = default;

  /**
   * @brief Allocates memory on the current device.
   *
   * @param nbytes The number of bytes to allocate.
   * @return A data pointer to the allocated memory.
   */
  c10::DataPtr allocate(size_t nbytes) override {
    const auto current_device_index = c10::rbln::get_device_index();
    const auto current_device = c10::Device(c10::kPrivateUse1, current_device_index);

    void* data = nullptr;
    if (nbytes > 0) {
      RBLN_LOG_DEBUG("Allocating {} bytes on {} device", nbytes, c10::str(current_device));
      data = c10::rbln::malloc(current_device_index, nbytes);
      RBLN_LOG_DEBUG("Allocated memory at {}", fmt::ptr(data));
    }

    return c10::DataPtr(data, data, &raw_delete, current_device);
  }

  /**
   * @brief Returns a function pointer to the raw deleter responsible for memory deallocation.
   *
   * @return A function pointer to the raw deleter.
   */
  c10::DeleterFnPtr raw_deleter() const override {
    const c10::DeleterFnPtr raw_delete_ptr = &raw_delete;
    RBLN_LOG_DEBUG("raw_delete_ptr={}", fmt::ptr(raw_delete_ptr));
    return raw_delete_ptr;
  }

  /**
   * @brief Copies data from a source pointer to a destination pointer.
   *
   * @param dst_data The destination pointer where data should be copied to.
   * @param src_data The source pointer from which data should be copied.
   * @param nbytes The number of bytes to copy.
   */
  void copy_data(void* dst_data, const void* src_data, size_t nbytes) const override {
    if (nbytes > 0) {
      RBLN_LOG_DEBUG("Copying {} bytes: {} -> {}", nbytes, fmt::ptr(src_data), fmt::ptr(dst_data));
      c10::rbln::memcpy_v2v(dst_data, src_data, nbytes);
    }
  }

  bool initialized() override {
    // RBLN runtime initializes lazily on first allocation; device availability
    // is a sufficient proxy for readiness.
    const bool is_initialized = (c10::rbln::get_device_count() > 0);
    RBLN_LOG_DEBUG("is_initialized={}", is_initialized);
    return is_initialized;
  }

  /**
   * @brief Empties the memory cache for the specified memory pool.
   *
   * RBLN does not support per-mempool scoping, so this function will flush the cache for the current device
   * regardless of the mempool_id provided.
   *
   * @param mempool_id The identifier for the memory pool to flush.
   */
  void emptyCache(c10::MempoolId_t /*mempool_id*/) override {
    const auto current_device_index = c10::rbln::get_device_index();
    const auto current_device = c10::Device(c10::kPrivateUse1, current_device_index);
    RBLN_LOG_DEBUG("Emptying cache for {}", c10::str(current_device));
    c10::rbln::empty_cache(current_device);
  }

  /**
   * @brief Records a stream association for a given data pointer.
   *
   * RBLN does not support stream-based asynchronous execution, so this function is a no-op.
   *
   * @param ptr The data pointer for which to record the stream association.
   * @param stream The stream to associate with the data pointer.
   */
  void recordStream(const c10::DataPtr& /*ptr*/, c10::Stream /*stream*/) override {
    RBLN_LOG_DEBUG("recordStream is no-op because RBLN does not support stream-based asynchronous execution");
  }

  /**
   * @brief Returns device memory statistics for the specified device index.
   *
   * @param device_index The index of the device for which to retrieve statistics.
   * @return A DeviceStats object containing memory statistics for the specified device.
   */
  c10::CachingDeviceAllocator::DeviceStats getDeviceStats(c10::DeviceIndex device_index) override {
    const auto device = c10::Device(c10::kPrivateUse1, device_index);
    RBLN_LOG_DEBUG("Getting device stats for {}", c10::str(device));
    const auto device_stats = c10::rbln::get_device_stats(device);
    return device_stats;
  }

  /**
   * @brief Resets accumulated memory statistics for the specified device index.
   *
   * @param device_index The index of the device for which to reset accumulated statistics.
   */
  void resetAccumulatedStats(c10::DeviceIndex device_index) override {
    const auto device = c10::Device(c10::kPrivateUse1, device_index);
    RBLN_LOG_DEBUG("Resetting accumulated stats for {}", c10::str(device));
    c10::rbln::reset_accumulated_memory_stats(device);
  }

  /**
   * @brief Resets peak memory statistics for the specified device index.
   *
   * @param device_index The index of the device for which to reset peak statistics.
   */
  void resetPeakStats(c10::DeviceIndex device_index) override {
    const auto device = c10::Device(c10::kPrivateUse1, device_index);
    RBLN_LOG_DEBUG("Resetting peak stats for {}", c10::str(device));
    c10::rbln::reset_peak_memory_stats(device);
  }
};

static RBLNAllocator allocator;
REGISTER_ALLOCATOR(c10::kPrivateUse1, &allocator);

} // namespace c10::rbln
