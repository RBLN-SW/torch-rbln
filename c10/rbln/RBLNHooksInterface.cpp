#include <c10/core/DeviceGuard.h>
#include <c10/rbln/RBLNFunctions.h>
#include <c10/rbln/RBLNGenerator.h>
#include <c10/rbln/RBLNHooksInterface.h>
#include <c10/rbln/RBLNLogging.h>
#include <c10/rbln/RBLNPinnedAllocator.h>
#include <c10/util/CallOnce.h>

namespace c10::rbln {

namespace {

TORCH_DECLARE_REGISTRY(PrivateUse1HooksRegistry, RBLNHooksInterface, RBLNHooksArgs);
C10_DEFINE_REGISTRY(PrivateUse1HooksRegistry, RBLNHooksInterface, RBLNHooksArgs)

#define REGISTER_PRIVATEUSE1_HOOKS(clsname) C10_REGISTER_CLASS(PrivateUse1HooksRegistry, clsname, clsname)

} // namespace

bool RBLNHooksInterface::isBuilt() const {
  // RBLN backend is built if this code is compiled and linked
  return true;
}

// The probe hooks -- isAvailable, hasRBLN, deviceCount, hasPrimaryContext -- answer
// questions torch treats as infallible, and Python answers the same ones from the same
// predicates. None of them logs: RBLN_LOG_DEBUG can throw when debug logging is enabled
// (fmt / sink, e.g. bad_alloc), which would make the two disagree on a path that must not
// fail. The hooks that select or allocate may throw by design and do log.

bool RBLNHooksInterface::isAvailable() const {
  return isBuilt() && hasRBLN();
}

bool RBLNHooksInterface::hasRBLN() const {
  // Same predicate the Python torch.rbln.is_available() is bound to. True in dummy mode as
  // well, provided the dummy mapping actually built.
  return c10::rbln::runtime_available();
}

c10::DeviceIndex RBLNHooksInterface::deviceCount() const {
  return c10::rbln::get_device_count_nothrow();
}

c10::DeviceIndex RBLNHooksInterface::getCurrentDevice() const {
  const auto device_index = c10::rbln::get_device_index();
  RBLN_LOG_DEBUG("device_index={}", static_cast<int>(device_index));
  return device_index;
}

void RBLNHooksInterface::setCurrentDevice(c10::DeviceIndex device) const {
  RBLN_LOG_DEBUG("device_index={}", static_cast<int>(device));
  c10::rbln::set_device_index(device);
}

c10::DeviceIndex RBLNHooksInterface::exchangeDevice(c10::DeviceIndex device) const {
  RBLN_LOG_DEBUG("device_index={}", static_cast<int>(device));
  return c10::rbln::exchange_device_index(device);
}

c10::DeviceIndex RBLNHooksInterface::maybeExchangeDevice(c10::DeviceIndex device) const {
  // Identical to exchangeDevice(): device selection never creates a context here.
  return exchangeDevice(device);
}

c10::Device RBLNHooksInterface::getDeviceFromPtr(void* data) const {
  RBLN_LOG_DEBUG("data={}", fmt::ptr(data));
  const auto device_index = c10::rbln::get_torch_device_id(data);
  const auto device = c10::Device(c10::kPrivateUse1, device_index);
  RBLN_LOG_DEBUG("device={}", c10::str(device));
  return device;
}

bool RBLNHooksInterface::hasPrimaryContext(c10::DeviceIndex device_index) const {
  // CUDA parity: a primary context exists for a device only once THIS process has used it
  // (a successful allocation), not merely because the device exists. Context flag first so
  // an unused device does not trigger device enumeration; runtime_available() then folds in
  // the shutdown/liveness check.
  return c10::rbln::device_context_initialized(device_index) && c10::rbln::runtime_available();
}

void RBLNHooksInterface::resizePrivateUse1Bytes(const c10::Storage& storage, size_t new_nbytes) const {
  RBLN_LOG_DEBUG("storage={}, new_nbytes={}", fmt::ptr(&storage), new_nbytes);
  // new_nbytes == 0 is a valid request (e.g. untyped_storage().resize_(0)); the
  // branch below already handles the zero-size case (null DataPtr, nbytes = 0).
  RBLN_CHECK(storage.resizable(), "Storage must be resizable");
  auto* allocator = storage.allocator();
  RBLN_CHECK(allocator != nullptr, "Cannot resize storage without allocator");

  const auto device = storage.device();
  const auto device_guard = c10::DeviceGuard(device);
  auto new_data_ptr = c10::DataPtr(nullptr, device);
  if (new_nbytes > 0) {
    RBLN_LOG_DEBUG("Allocating {} bytes on {} device", new_nbytes, c10::str(device));
    new_data_ptr = allocator->allocate(new_nbytes);
    auto* new_data = new_data_ptr.get();
    RBLN_LOG_DEBUG("Allocated memory at {}", fmt::ptr(new_data));

    const auto* old_data = storage.data();
    const auto old_nbytes = storage.nbytes();
    const auto copy_nbytes = std::min(new_nbytes, old_nbytes);
    RBLN_LOG_DEBUG("old_nbytes={}, copy_nbytes={}", old_nbytes, copy_nbytes);
    if ((old_data != nullptr) && (copy_nbytes > 0)) {
      RBLN_LOG_DEBUG("Copying {} bytes from old memory to new memory", copy_nbytes);
      c10::rbln::memcpy_v2v(new_data, old_data, copy_nbytes);
    }
  }
  RBLN_LOG_DEBUG("Updating storage with new data pointer and nbytes");
  storage.set_data_ptr_noswap(std::move(new_data_ptr));
  storage.set_nbytes(new_nbytes);
}

c10::Allocator* RBLNHooksInterface::getPinnedMemoryAllocator() const {
  return c10::rbln::get_pinned_memory_allocator();
}

bool RBLNHooksInterface::isPinnedPtr(const void* data) const {
  return c10::rbln::is_pinned_ptr(data);
}

at::Generator RBLNHooksInterface::getNewGenerator(c10::DeviceIndex device_index) const {
  RBLN_LOG_DEBUG("device_index={}", static_cast<int>(device_index));
  return at::make_generator<at::RBLNGeneratorImpl>(device_index);
}

at::PrivateUse1HooksInterface* get_rbln_hooks() {
  static const std::unique_ptr<at::PrivateUse1HooksInterface> rbln_hooks = []() {
    // Called from shared-library registration during dlopen(). Avoid logging here:
    // logger initialization may validate env vars, and any exception escaping
    // this path terminates the process before Python can surface a clean error.
    return std::make_unique<c10::rbln::RBLNHooksInterface>();
  }();
  return static_cast<at::PrivateUse1HooksInterface*>(rbln_hooks.get());
}

} // namespace c10::rbln
