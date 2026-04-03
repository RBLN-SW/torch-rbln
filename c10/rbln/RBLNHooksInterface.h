#pragma once

#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <c10/rbln/RBLNMacros.h>

namespace c10::rbln {

struct C10_RBLN_API RBLNHooksInterface : public at::PrivateUse1HooksInterface {
  /**
   * @brief Check if RBLN backend was enabled at compilation time.
   * This function should NEVER throw.
   *
   * @return True if RBLN backend is built, False otherwise.
   */
  bool isBuilt() const override;

  /**
   * @brief Check if RBLN backend can be used at runtime.
   * This means it was built, runtime dependencies are available, and at least one device can be used.
   * This function should NEVER throw and should NOT initialize the context on any device.
   *
   * @return True if RBLN backend is available, False otherwise.
   */
  bool isAvailable() const override;

  /**
   * @brief Check if RBLN devices are available at runtime.
   * This function checks if RBLN runtime dependencies are available and at least one device can be used.
   *
   * @return True if RBLN devices are available, False otherwise.
   */
  bool hasRBLN() const;

  /**
   * @brief Returns the device of the input device pointer.
   *
   * @param data A pointer to the device memory.
   * @return The device of the device memory.
   */
  c10::Device getDeviceFromPtr(void* data) const override;

  /**
   * @brief Checks if the given device has a primary context.
   *
   * @param device_index The device to check.
   * @return true if the device has a primary context, false otherwise.
   */
  bool hasPrimaryContext(c10::DeviceIndex device_index) const override;

  /**
   * @brief Resizes the storage to the specified number of bytes.
   *
   * @param storage The storage to resize.
   * @param new_nbytes The number of bytes to resize the storage to.
   */
  void resizePrivateUse1Bytes(const c10::Storage& storage, size_t new_nbytes) const override;

  /**
   * @brief Creates a new generator for the specified device index.
   *
   * This function is responsible for creating and returning a new generator
   * instance associated with the given device index.
   *
   * @param device_index The device index for which the generator is to be created.
   * @return A new generator instance for the specified device index.
   */
  at::Generator getNewGenerator(c10::DeviceIndex device_index) const override;
};

struct C10_RBLN_API RBLNHooksArgs : public at::PrivateUse1HooksArgs {};

// register to PrivateUse1HooksInterface
C10_RBLN_API at::PrivateUse1HooksInterface* get_rbln_hooks();

} // namespace c10::rbln
