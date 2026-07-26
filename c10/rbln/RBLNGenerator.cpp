#include <ATen/ATen.h>
#include <ATen/Utils.h>
#include <ATen/core/GeneratorForPrivateuseone.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <c10/rbln/RBLNGenerator.h>

namespace at {

RBLNGeneratorImpl::RBLNGeneratorImpl(DeviceIndex device_index)
    : GeneratorImpl(Device(DeviceType::PrivateUse1, device_index), DispatchKeySet(c10::DispatchKey::PrivateUse1)),
      seed_(0),
      offset_(0),
      cpu_generator_(make_intrusive<CPUGeneratorImpl>()) {
  cpu_generator_->set_current_seed(seed_);
}

at::Generator RBLNGeneratorImpl::get_fallback_generator() const {
  return at::Generator(cpu_generator_);
}

void RBLNGeneratorImpl::set_current_seed(uint64_t seed) {
  seed_ = seed;
  cpu_generator_->set_current_seed(seed);
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

void RBLNGeneratorImpl::set_state(const c10::TensorImpl& new_state) {
  TORCH_CHECK(new_state.device().is_cpu(), "RBLN generator state must be a CPU tensor, but got ", new_state.device());

  TORCH_CHECK(new_state.dtype() == at::kByte, "RBLN generator state must be a ByteTensor, but got ", new_state.dtype());

  const auto* state_ptr = new_state.data();

  std::memcpy(&seed_, state_ptr, sizeof(seed_));

  std::memcpy(&offset_, state_ptr + sizeof(seed_), sizeof(offset_));

  auto fallback_state = at::empty(
      {static_cast<int64_t>(new_state.numel() - (sizeof(seed_) + sizeof(offset_)))},
      at::TensorOptions().dtype(at::kByte).device(at::kCPU));

  std::memcpy(fallback_state.data_ptr<uint8_t>(), state_ptr + sizeof(seed_) + sizeof(offset_), fallback_state.numel());

  cpu_generator_->set_state(*fallback_state.unsafeGetTensorImpl());
}

c10::intrusive_ptr<c10::TensorImpl> RBLNGeneratorImpl::get_state() const {
  auto fallback_state = cpu_generator_->get_state();

  TORCH_INTERNAL_ASSERT(fallback_state->device().is_cpu(), "CPU fallback generator state must be on CPU");

  TORCH_INTERNAL_ASSERT(fallback_state->dtype() == at::kByte, "CPU fallback generator state must be a ByteTensor");

  const auto fallback_state_numel = fallback_state->numel();

  auto state = at::empty(
      {static_cast<int64_t>(sizeof(seed_) + sizeof(offset_) + fallback_state_numel)},
      at::TensorOptions().dtype(at::kByte).device(at::kCPU));

  auto* state_ptr = state.data_ptr<uint8_t>();

  std::memcpy(state_ptr, &seed_, sizeof(seed_));

  std::memcpy(state_ptr + sizeof(seed_), &offset_, sizeof(offset_));

  std::memcpy(state_ptr + sizeof(seed_) + sizeof(offset_), fallback_state->data(), fallback_state_numel);

  return state.getIntrusivePtr();
}

RBLNGeneratorImpl* RBLNGeneratorImpl::clone_impl() const {
  auto gen = new RBLNGeneratorImpl(device().index());
  gen->set_state(*get_state());
  return gen;
}

} // namespace at
