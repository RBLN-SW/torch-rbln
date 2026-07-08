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

bool RBLNHooksInterface::isAvailable() const {
  const auto is_built = isBuilt();
  const auto has_rbln = hasRBLN();
  RBLN_LOG_DEBUG("is_built={}, has_rbln={}", is_built, has_rbln);
  const bool is_available = (is_built && has_rbln);
  RBLN_LOG_DEBUG("is_available={}", is_available);
  return is_available;
}

bool RBLNHooksInterface::hasRBLN() const {
  // isAvailable() must never throw (accelerator-hooks contract); runtime_available()
  // is nothrow and already true in dummy mode.
  const bool has_rbln = c10::rbln::runtime_available();
  RBLN_LOG_DEBUG("has_rbln={}", has_rbln);
  return has_rbln;
}

c10::Device RBLNHooksInterface::getDeviceFromPtr(void* data) const {
  RBLN_LOG_DEBUG("data={}", fmt::ptr(data));
  const auto device_index = c10::rbln::get_torch_device_id(data);
  const auto device = c10::Device(c10::kPrivateUse1, device_index);
  RBLN_LOG_DEBUG("device={}", c10::str(device));
  return device;
}

bool RBLNHooksInterface::hasPrimaryContext(c10::DeviceIndex device_index) const {
  RBLN_LOG_DEBUG("device_index={}", static_cast<int>(device_index));

  // thunk_loadable() first so a missing runtime (incl. dummy) is a clean false (no
  // segfault); throwing get_device_count() keeps a malformed config loud when present.
  const bool has_context =
      device_index >= 0 && c10::rbln::thunk_loadable() && device_index < c10::rbln::get_device_count();
  RBLN_LOG_DEBUG("has_context={}", has_context);
  return has_context;
}

void RBLNHooksInterface::resizePrivateUse1Bytes(const c10::Storage& storage, size_t new_nbytes) const {
  RBLN_LOG_DEBUG("storage={}, new_nbytes={}", fmt::ptr(&storage), new_nbytes);
  RBLN_CHECK(new_nbytes > 0, "New nbytes must be positive, but got {}", new_nbytes);
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
