#include <c10/rbln/RBLNHooksInterface.h>
#include <c10/rbln/RBLNLogging.h>
#include <c10/rbln/RBLNSupportedDtypes.h>
#include <c10/rbln/impl/RBLNGuardImpl.h>

#include <c10/core/DeviceCapability.h>

#include <exception>

namespace c10::rbln::impl {

namespace {

C10_REGISTER_GUARD_IMPL(PrivateUse1, RBLNGuardImpl);

#define REGISTER_RBLN_HOOKS_INTERFACE()                                 \
  int register_rbln_hooks_interface() {                                 \
    at::RegisterPrivateUse1HooksInterface(c10::rbln::get_rbln_hooks()); \
    return 0;                                                           \
  }                                                                     \
  static const int _temp_rbln = register_rbln_hooks_interface();

REGISTER_RBLN_HOOKS_INTERFACE()

} // namespace

RBLNGuardImpl::RBLNGuardImpl(c10::DeviceType device_type) {
  RBLN_CHECK(
      device_type == c10::kPrivateUse1,
      "Only privateuseone device type is supported, but got {}",
      c10::str(device_type));
}

c10::DeviceType RBLNGuardImpl::type() const {
  const auto device_type = c10::kPrivateUse1;
  RBLN_LOG_DEBUG("device_type={}", c10::str(device_type));
  return device_type;
}

c10::Device RBLNGuardImpl::exchangeDevice(c10::Device device) const {
  const auto device_index = device.index();
  const auto original_device_index = c10::rbln::exchange_device_index(device_index);
  const auto original_device = c10::Device(c10::kPrivateUse1, original_device_index);
  RBLN_LOG_DEBUG("Setting current device: {} -> {}", c10::str(original_device), c10::str(device));
  return original_device;
}

c10::Device RBLNGuardImpl::getDevice() const {
  const auto current_device_index = c10::rbln::get_device_index();
  const auto current_device = c10::Device(c10::kPrivateUse1, current_device_index);
  RBLN_LOG_DEBUG("current_device={}", c10::str(current_device));
  return current_device;
}

void RBLNGuardImpl::setDevice(c10::Device device) const {
  RBLN_LOG_DEBUG("Setting device to {}", c10::str(device));
  const auto device_index = device.index();
  c10::rbln::set_device_index(device_index);
}

void RBLNGuardImpl::uncheckedSetDevice(c10::Device device) const noexcept {
  try {
    RBLN_LOG_DEBUG("Setting device to {}", c10::str(device));
    setDevice(device);
  } catch (const c10::Error& error) {
    RBLN_WARN_NOTHROW("Failed to set device: {}", error.msg());
  } catch (const std::exception& e) {
    RBLN_WARN_NOTHROW("Failed to set device (std::exception): {}", e.what());
  } catch (...) {
    RBLN_WARN_NOTHROW("Failed to set device: unknown exception");
  }
}

c10::DeviceIndex RBLNGuardImpl::deviceCount() const noexcept {
  try {
    const auto device_count = c10::rbln::get_device_count();
    RBLN_LOG_DEBUG("device_count={}", static_cast<int>(device_count));
    return device_count;
  } catch (const c10::Error& error) {
    RBLN_WARN_NOTHROW("Failed to get device count, returning 0: {}", error.msg());
    return 0;
  } catch (const std::exception& e) {
    // First call lazily parses RBLN_DEVICE_MAP / RBLN_NPUS_PER_DEVICE via
    // std::stoi, which throws std (not c10::Error); uncaught here -> terminate.
    RBLN_WARN_NOTHROW("Failed to get device count, returning 0 (std::exception): {}", e.what());
    return 0;
  } catch (...) {
    RBLN_WARN_NOTHROW("Failed to get device count, returning 0: unknown exception");
    return 0;
  }
}

c10::DeviceCapability RBLNGuardImpl::getDeviceCapability(c10::Device device) const {
  // Validate the requested index so a stale/out-of-range device can't silently
  // report a capability. A no-device host (count == 0) skips the range check.
  const auto count = deviceCount();
  if (count > 0) {
    const auto index = device.has_index() ? device.index() : c10::rbln::get_device_index();
    TORCH_CHECK(
        index >= 0 && index < count,
        "rbln device index ",
        static_cast<int>(index),
        " is out of range [0, ",
        static_cast<int>(count),
        ")");
  }

  // Capability = the dtypes resident in device memory (allocation/conversion
  // capability, not native-op dispatch). See kCapabilityDtypes.
  c10::DeviceCapability capability;
  capability.capability_data.capability_bits = 0;
  for (const auto scalar_type : c10::rbln::kCapabilityDtypes) {
    switch (scalar_type) {
      case c10::kHalf:
        capability.capability_data.supported_scalar_types.has_Half = 1;
        break;
      case c10::kBFloat16:
        capability.capability_data.supported_scalar_types.has_BFloat16 = 1;
        break;
      default:
        TORCH_CHECK(false, "unhandled capability dtype ", c10::toString(scalar_type));
    }
  }
  return capability;
}

c10::Stream RBLNGuardImpl::getStream(c10::Device device) const {
  // No context yet means there is no current stream to read, and nothing has been
  // selected: the default stream is the answer, without a runtime call. A failure
  // once the context exists is real -- swallowing it would silently move work off
  // the stream the caller selected.
  const auto index = device.has_index() ? device.index() : c10::rbln::get_device_index();
  if (!c10::rbln::device_context_initialized(index)) {
    return c10::rbln::get_default_stream(index);
  }
  return c10::rbln::get_current_stream(index);
}

c10::Stream RBLNGuardImpl::getDefaultStream(c10::Device device) const {
  return c10::rbln::get_default_stream(device.index());
}

c10::Stream RBLNGuardImpl::getNewStream(c10::Device device, int priority) const {
  (void)priority; // RBLN has no stream priorities.
  // Pooled, like CUDA's getNewStream: torch has no destroy hook for a stream, so
  // handing out a freshly created one per call would leak it.
  return c10::rbln::get_stream_from_pool(device.index());
}

c10::Stream RBLNGuardImpl::getStreamFromGlobalPool(c10::Device device, bool isHighPriority) const {
  (void)isHighPriority; // RBLN has no stream priorities.
  return c10::rbln::get_stream_from_pool(device.index());
}

c10::Stream RBLNGuardImpl::exchangeStream(c10::Stream stream) const {
  const auto original_stream = getStream(stream.device());
  c10::rbln::set_current_stream(stream);
  RBLN_LOG_DEBUG("Setting current stream: {} -> {}", c10::str(original_stream), c10::str(stream));
  return original_stream;
}

bool RBLNGuardImpl::queryStream(const c10::Stream& stream) const {
  return c10::rbln::query_stream(stream);
}

void RBLNGuardImpl::synchronizeStream(const c10::Stream& stream) const {
  c10::rbln::synchronize_stream(stream);
}

namespace {

// The void* event handle IS the opaque event handle (see RBLNGuardImpl.h). Route the
// int<->ptr casts through uintptr_t and keep them in one place.
void* to_event_ptr(uint64_t handle) {
  return reinterpret_cast<void*>(static_cast<uintptr_t>(handle)); // NOLINT(performance-no-int-to-ptr)
}

uint64_t to_event_handle(void* event) {
  return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(event));
}

} // namespace

void RBLNGuardImpl::record(void** event, const c10::Stream& stream, c10::DeviceIndex device_index, c10::EventFlag flag)
    const {
  (void)flag; // Event timing is not supported (elapsedTime is unimplemented).
  const auto event_device = device_index >= 0 ? device_index : stream.device_index();
  if (*event == nullptr) {
    // A valid event handle is non-zero, so it never aliases the "not yet created"
    // nullptr sentinel torch checks here.
    *event = to_event_ptr(c10::rbln::event_create(event_device));
  }
  c10::rbln::event_record(to_event_handle(*event), stream);
}

void RBLNGuardImpl::block(void* event, const c10::Stream& stream) const {
  if (event == nullptr) {
    return; // never recorded -> nothing to wait for
  }
  c10::rbln::event_block(stream, to_event_handle(event));
}

bool RBLNGuardImpl::queryEvent(void* event) const {
  if (event == nullptr) {
    return true; // never recorded -> complete
  }
  return c10::rbln::event_query(to_event_handle(event));
}

void RBLNGuardImpl::synchronizeEvent(void* event) const {
  if (event == nullptr) {
    return; // never recorded -> no-op
  }
  c10::rbln::event_synchronize(to_event_handle(event));
}

void RBLNGuardImpl::destroyEvent(void* event, c10::DeviceIndex device_index) const noexcept {
  (void)device_index;
  c10::rbln::event_destroy(to_event_handle(event)); // noexcept; no-op on null (handle 0)
}

void RBLNGuardImpl::synchronizeDevice(c10::DeviceIndex device_index) const {
  RBLN_LOG_DEBUG("Synchronizing device {}", static_cast<int>(device_index));
  c10::rbln::synchronize(device_index);
}

} // namespace c10::rbln::impl
