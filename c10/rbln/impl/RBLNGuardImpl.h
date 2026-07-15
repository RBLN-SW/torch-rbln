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
   * @brief Reports supported dtypes (backs torch.accelerator.get_device_capability).
   *
   * Advertises only the natively-dispatched dtypes (fp16/bf16); CPU-fallback dtypes are excluded.
   */
  c10::DeviceCapability getDeviceCapability(c10::Device device) const override;

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

  // Events (torch.Event). RBLN has a single in-order copy queue per device and
  // the UMD exposes only a whole-device drain, so an event records the device
  // it was captured on and every wait maps to synchronize(device). This
  // over-synchronizes relative to CUDA's per-event waits but is correct: the
  // queue is in-order, so draining it covers everything recorded before.

  /**
   * @brief Marks the event as recorded on the device of the input stream.
   * Allocates the event on first record.
   *
   * @param event In/out opaque event handle.
   * @param stream The stream to record on (only its device is used).
   * @param device_index The device index, or -1 for the stream's device.
   * @param flag Ignored (RBLN events carry no timing).
   */
  void record(void** event, const c10::Stream& stream, c10::DeviceIndex device_index, c10::EventFlag flag)
      const override;

  /**
   * @brief Makes the stream wait for the event. Host-blocks on the recorded
   * device's queue (stronger than a device-side wait, hence correct).
   *
   * @param event An opaque event handle. May be null (never recorded).
   * @param stream Unused.
   */
  void block(void* event, const c10::Stream& stream) const override;

  /**
   * @brief Returns true once the recorded work has completed. Drains the
   * recorded device's queue first, so unlike CUDA this may block; it never
   * returns false for a recorded event.
   *
   * @param event An opaque event handle. May be null (never recorded).
   * @return True.
   */
  bool queryEvent(void* event) const override;

  /**
   * @brief Blocks the host until the recorded work has completed.
   *
   * @param event An opaque event handle. May be null (never recorded).
   */
  void synchronizeEvent(void* event) const override;

  /**
   * @brief Frees the event.
   *
   * @param event An opaque event handle. May be null.
   * @param device_index Unused.
   */
  void destroyEvent(void* event, c10::DeviceIndex device_index) const noexcept override;

  /**
   * @brief Blocks the host until all pending work on the device has completed.
   *
   * @param device_index The device to synchronize.
   */
  void synchronizeDevice(c10::DeviceIndex device_index) const override;
};

} // namespace c10::rbln::impl
