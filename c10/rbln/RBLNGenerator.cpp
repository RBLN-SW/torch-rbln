#include <ATen/Utils.h>
#include <ATen/core/GeneratorForPrivateuseone.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <c10/rbln/RBLNGenerator.h>

namespace at {

RBLNGeneratorImpl::RBLNGeneratorImpl(DeviceIndex device_index)
    : GeneratorImpl(Device(DeviceType::PrivateUse1, device_index), DispatchKeySet(c10::DispatchKey::PrivateUse1)),
      seed_(0),
      offset_(0) {}

void RBLNGeneratorImpl::set_current_seed(uint64_t seed) {
  seed_ = seed;
}

void RBLNGeneratorImpl::set_offset(uint64_t offset) {
  offset_ = offset;
}

uint64_t RBLNGeneratorImpl::get_offset() const {
  return offset_;
}

uint64_t RBLNGeneratorImpl::current_seed() const {
  return seed_;
}

uint64_t RBLNGeneratorImpl::seed() {
  return seed_;
}

void RBLNGeneratorImpl::set_state(const c10::TensorImpl& new_state) {}

c10::intrusive_ptr<c10::TensorImpl> RBLNGeneratorImpl::get_state() const {
  static const size_t seed_size = sizeof(uint64_t);
  static const size_t offset_size = sizeof(uint64_t);
  static const size_t total_size = seed_size + offset_size;

  auto state_tensor = at::detail::empty_cpu(
      {static_cast<int64_t>(total_size)}, ScalarType::Byte, std::nullopt, std::nullopt, std::nullopt, std::nullopt);
  return state_tensor.getIntrusivePtr();
}

RBLNGeneratorImpl* RBLNGeneratorImpl::clone_impl() const {
  auto gen = new RBLNGeneratorImpl(device().index());
  gen->set_offset(offset_);
  gen->set_current_seed(seed_);
  gen->set_state(*get_state());
  return gen;
}

} // namespace at
