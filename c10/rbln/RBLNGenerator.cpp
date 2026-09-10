#include <ATen/Utils.h>
#include <ATen/core/Generator.h>
#include <ATen/core/GeneratorForPrivateuseone.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <c10/rbln/RBLNGenerator.h>

#include <cstring>

namespace at {

namespace {
// get_state()/set_state() exchange the generator as raw bytes: the seed followed by the
// offset, host-endian -- the layout CUDA's and XPU's generators use.
constexpr size_t kSeedSize = sizeof(uint64_t);
constexpr size_t kOffsetSize = sizeof(uint64_t);
constexpr size_t kStateSize = kSeedSize + kOffsetSize;
} // namespace

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

void RBLNGeneratorImpl::set_state(const c10::TensorImpl& new_state) {
  at::detail::check_rng_state(new_state);
  const auto new_state_size = static_cast<size_t>(new_state.numel());
  TORCH_CHECK(
      new_state_size == kStateSize, "RNG state is wrong size: expected ", kStateSize, " bytes, got ", new_state_size);
  const auto* bytes = new_state.data_dtype_initialized<uint8_t>();
  std::memcpy(&seed_, bytes, kSeedSize);
  std::memcpy(&offset_, bytes + kSeedSize, kOffsetSize);
}

c10::intrusive_ptr<c10::TensorImpl> RBLNGeneratorImpl::get_state() const {
  auto state_tensor = at::detail::empty_cpu(
      {static_cast<int64_t>(kStateSize)}, ScalarType::Byte, std::nullopt, std::nullopt, std::nullopt, std::nullopt);
  auto* bytes = state_tensor.mutable_data_ptr<uint8_t>();
  std::memcpy(bytes, &seed_, kSeedSize);
  std::memcpy(bytes + kSeedSize, &offset_, kOffsetSize);
  return state_tensor.getIntrusivePtr();
}

RBLNGeneratorImpl* RBLNGeneratorImpl::clone_impl() const {
  auto gen = new RBLNGeneratorImpl(device().index());
  gen->set_current_seed(seed_);
  gen->set_offset(offset_);
  return gen;
}

} // namespace at
