#pragma once

#include <c10/core/DeviceGuard.h>
#include <c10/rbln/RBLNFunctions.h>

namespace c10::rbln::impl {

struct RBLNGuardImpl final : public c10::impl::DeviceGuardImplInterface {
  static constexpr c10::DeviceType static_type = c10::kPrivateUse1;

  RBLNGuardImpl() = default;

  explicit RBLNGuardImpl(c10::DeviceType device_type);

  /**
   * @brief Returns the device type of the guard implementation.
   *
   * @return The device type of the guard implementation.
   */
  c10::DeviceType type() const override;

  /**
   * @brief Sets the current device to the input device, and returns the previous device.
   *
   * @param device The input device.
   * @return The previous device.
   */
  c10::Device exchangeDevice(c10::Device device) const override;

  /**
   * @brief Returns the current device.
   *
   * @return The current device.
   */
  c10::Device getDevice() const override;

  /**
   * @brief Sets the current device to the input device.
   *
   * @param device The input device.
   */
  void setDevice(c10::Device device) const override;

  /**
   * @brief Sets the current device to the input device without throwing exceptions.
   *
   * @param device The input device.
   */
  void uncheckedSetDevice(c10::Device device) const noexcept override;

  /**
   * @brief Returns the number of available devices.
   *
   * @return The number of available devices.
   */
  c10::DeviceIndex deviceCount() const noexcept override;

  /**
   * @brief Returns the current stream for the input device.
   *
   * @param device The input device.
   * @return The current stream.
   */
  c10::Stream getStream(c10::Device device) const override;

  /**
   * @brief Sets the current stream to the input stream, and returns the previous stream.
   *
   * @param stream The input stream.
   * @return The previous stream.
   */
  c10::Stream exchangeStream(c10::Stream stream) const override;
};

} // namespace c10::rbln::impl
