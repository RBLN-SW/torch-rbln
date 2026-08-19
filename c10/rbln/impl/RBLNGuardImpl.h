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
   * supported_dtypes is the device's allocation/conversion capability, not native
   * op dispatch. RBLN advertises fp16/bf16 (kCapabilityDtypes) — the only dtypes
   * resident in device memory; other dtypes are CPU-backed even under device="rbln".
   */
  c10::DeviceCapability getDeviceCapability(c10::Device device) const override;

  /**
   * @brief Returns the current stream for the input device (default if none set).
   *
   * @param device The input device.
   * @return The current stream.
   */
  c10::Stream getStream(c10::Device device) const override;

  /**
   * @brief Returns the device's default stream (StreamId 0).
   *
   * @param device The input device.
   * @return The default stream.
   */
  c10::Stream getDefaultStream(c10::Device device) const override;

  /**
   * @brief Creates and returns a fresh stream on the input device.
   *
   * @param device The input device.
   * @param priority Ignored. RBLN has no stream priorities.
   * @return A new stream.
   */
  c10::Stream getNewStream(c10::Device device, int priority = 0) const override;

  /**
   * @brief Returns a stream from the device's fixed round-robin pool.
   *
   * @param device The input device.
   * @param isHighPriority Ignored. RBLN has no stream priorities.
   * @return A pooled stream.
   */
  c10::Stream getStreamFromGlobalPool(c10::Device device, bool isHighPriority = false) const override;

  /**
   * @brief Sets the current stream to the input stream, and returns the previous stream.
   *
   * @param stream The input stream.
   * @return The previous stream.
   */
  c10::Stream exchangeStream(c10::Stream stream) const override;

  /**
   * @brief Non-blocking: true iff all work submitted to the stream has completed.
   *
   * @param stream The stream to query.
   */
  bool queryStream(const c10::Stream& stream) const override;

  /**
   * @brief Blocks the host until all work on the stream has completed.
   *
   * @param stream The stream to synchronize.
   */
  void synchronizeStream(const c10::Stream& stream) const override;

  // Events (torch.Event). elapsedTime is intentionally left unimplemented (event
  // timing is not supported; the base throws "Backend doesn't support elapsedTime.").
  // Cross-device waits are not supported and degrade to a host-side wait (see
  // block()). The opaque void* handle is non-zero for a valid event, so it never
  // aliases nullptr.

  /**
   * @brief Records the event at the stream's current position (allocates on first
   * record; re-record overwrites the snapshot).
   *
   * @param event In/out opaque event handle.
   * @param stream The stream to record on.
   * @param device_index The device index, or -1 for the stream's device.
   * @param flag Ignored (event timing is not supported).
   */
  void record(void** event, const c10::Stream& stream, c10::DeviceIndex device_index, c10::EventFlag flag)
      const override;

  /**
   * @brief Makes the stream wait for the event. Same-device: does not block the
   * host. Cross-device waits are not supported and degrade to a host-side wait on
   * the event.
   *
   * @param event An opaque event handle. May be null (never recorded -> no-op).
   * @param stream The stream that must wait.
   */
  void block(void* event, const c10::Stream& stream) const override;

  /**
   * @brief Non-blocking: true iff the work recorded into the event has completed
   * (true for a null / never-recorded event).
   *
   * @param event An opaque event handle. May be null (never recorded).
   */
  bool queryEvent(void* event) const override;

  /**
   * @brief Blocks the host until the recorded work has completed.
   *
   * @param event An opaque event handle. May be null (never recorded -> no-op).
   */
  void synchronizeEvent(void* event) const override;

  /**
   * @brief Frees the event. Never throws.
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
