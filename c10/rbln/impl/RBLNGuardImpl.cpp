#include <c10/rbln/RBLNHooksInterface.h>
#include <c10/rbln/RBLNLogging.h>
#include <c10/rbln/impl/RBLNGuardImpl.h>

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

c10::Stream RBLNGuardImpl::getStream(c10::Device device) const {
  const auto current_stream = c10::Stream(c10::Stream::Default::DEFAULT, device);
  RBLN_LOG_DEBUG("current_stream={}", c10::str(current_stream));
  return current_stream;
}

c10::Stream RBLNGuardImpl::exchangeStream(c10::Stream stream) const {
  const auto current_device_index = c10::rbln::get_device_index();
  const auto current_device = c10::Device(c10::kPrivateUse1, current_device_index);
  const auto original_stream = c10::Stream(c10::Stream::Default::DEFAULT, current_device);
  RBLN_LOG_DEBUG("Setting current stream: {} -> {}", c10::str(original_stream), c10::str(stream));
  return original_stream;
}

namespace {

// Opaque payload behind the void* handles in the event API: the device the
// event was last recorded on, or -1 while never recorded.
struct RBLNEvent {
  c10::DeviceIndex device_index = -1;

  bool recorded() const {
    return device_index >= 0;
  }
};

void drain_if_recorded(void* event) {
  const auto* rbln_event = static_cast<const RBLNEvent*>(event);
  if (rbln_event != nullptr && rbln_event->recorded()) {
    c10::rbln::synchronize(rbln_event->device_index);
  }
}

} // namespace

void RBLNGuardImpl::record(void** event, const c10::Stream& stream, c10::DeviceIndex device_index, c10::EventFlag flag)
    const {
  (void)flag; // RBLN events carry no timing.
  if (*event == nullptr) {
    *event = new RBLNEvent();
  }
  auto* rbln_event = static_cast<RBLNEvent*>(*event);
  rbln_event->device_index = device_index >= 0 ? device_index : stream.device_index();
  RBLN_LOG_DEBUG("Recorded event on device {}", static_cast<int>(rbln_event->device_index));
}

void RBLNGuardImpl::block(void* event, const c10::Stream& stream) const {
  (void)stream; // Single in-order queue: a host-side drain orders everything.
  drain_if_recorded(event);
}

bool RBLNGuardImpl::queryEvent(void* event) const {
  drain_if_recorded(event);
  return true;
}

void RBLNGuardImpl::synchronizeEvent(void* event) const {
  drain_if_recorded(event);
}

void RBLNGuardImpl::destroyEvent(void* event, c10::DeviceIndex device_index) const noexcept {
  (void)device_index;
  delete static_cast<RBLNEvent*>(event);
}

void RBLNGuardImpl::synchronizeDevice(c10::DeviceIndex device_index) const {
  RBLN_LOG_DEBUG("Synchronizing device {}", static_cast<int>(device_index));
  c10::rbln::synchronize(device_index);
}

} // namespace c10::rbln::impl
