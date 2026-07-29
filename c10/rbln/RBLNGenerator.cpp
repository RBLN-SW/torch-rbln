#include <ATen/ATen.h>
#include <ATen/Utils.h>
#include <ATen/core/GeneratorForPrivateuseone.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <c10/rbln/RBLNFunctions.h>
#include <c10/rbln/RBLNGenerator.h>
#include <c10/rbln/RBLNLogging.h>

namespace {

int64_t fallback_state_size() {
  static const int64_t size = at::CPUGeneratorImpl().get_state()->numel();
  return size;
}

c10::DeviceIndex resolve_device_index(c10::DeviceIndex device_index) {
  return device_index == -1 ? c10::rbln::get_device_index() : device_index;
}

} // namespace

namespace at {

RBLNGeneratorImpl::RBLNGeneratorImpl(DeviceIndex device_index)
    : GeneratorImpl(
          Device(DeviceType::PrivateUse1, resolve_device_index(device_index)),
          DispatchKeySet(c10::DispatchKey::PrivateUse1)),
      seed_(0),
      cpu_generator_(make_intrusive<CPUGeneratorImpl>()) {
  cpu_generator_->set_current_seed(seed_);
}

at::Generator RBLNGeneratorImpl::get_fallback_generator() const {
  return at::Generator(cpu_generator_);
}

void RBLNGeneratorImpl::set_current_seed(uint64_t seed) {
  std::lock_guard<std::mutex> lock(cpu_generator_->mutex_);
  seed_ = seed;
  cpu_generator_->set_current_seed(seed);
}

void RBLNGeneratorImpl::set_offset(uint64_t offset) {
  TORCH_CHECK(false, "RBLN Generator does not use offset");
}

uint64_t RBLNGeneratorImpl::get_offset() const {
  TORCH_CHECK(false, "RBLN Generator does not use offset");
}

uint64_t RBLNGeneratorImpl::current_seed() const {
  std::lock_guard<std::mutex> lock(cpu_generator_->mutex_);
  return seed_;
}

uint64_t RBLNGeneratorImpl::seed() {
  auto random_seed = c10::detail::getNonDeterministicRandom();
  this->set_current_seed(random_seed);
  return seed_;
}

void RBLNGeneratorImpl::set_state(const c10::TensorImpl& new_state) {
  const int64_t expected_size = fallback_state_size() + static_cast<int64_t>(sizeof(seed_));

  TORCH_CHECK(new_state.device().is_cpu(), "RBLN generator state must be a CPU tensor, but got ", new_state.device());

  TORCH_CHECK(new_state.dtype() == at::kByte, "RBLN generator state must be a ByteTensor, but got ", new_state.dtype());

  TORCH_CHECK(new_state.is_contiguous(), "RBLN generator state must be contiguous");

  TORCH_CHECK(
      new_state.numel() == expected_size,
      "RBLN generator state has invalid size: expected ",
      expected_size,
      " bytes, but got ",
      new_state.numel(),
      " bytes");

  std::lock_guard<std::mutex> lock(cpu_generator_->mutex_);

  const auto* state_ptr = static_cast<const uint8_t*>(new_state.data());

  std::memcpy(&seed_, state_ptr, sizeof(seed_));

  auto fallback_state = at::empty(
      {static_cast<int64_t>(new_state.numel() - sizeof(seed_))}, at::TensorOptions().dtype(at::kByte).device(at::kCPU));

  std::memcpy(fallback_state.data_ptr<uint8_t>(), state_ptr + sizeof(seed_), fallback_state.numel());

  cpu_generator_->set_state(*fallback_state.unsafeGetTensorImpl());
}

c10::intrusive_ptr<c10::TensorImpl> RBLNGeneratorImpl::get_state() const {
  std::lock_guard<std::mutex> lock(cpu_generator_->mutex_);

  auto fallback_state = cpu_generator_->get_state();

  TORCH_INTERNAL_ASSERT(fallback_state->device().is_cpu(), "CPU fallback generator state must be on CPU");

  TORCH_INTERNAL_ASSERT(fallback_state->dtype() == at::kByte, "CPU fallback generator state must be a ByteTensor");

  const auto fallback_state_numel = fallback_state->numel();

  auto state = at::empty(
      {static_cast<int64_t>(sizeof(seed_) + fallback_state_numel)},
      at::TensorOptions().dtype(at::kByte).device(at::kCPU));

  auto* state_ptr = state.data_ptr<uint8_t>();

  std::memcpy(state_ptr, &seed_, sizeof(seed_));

  std::memcpy(state_ptr + sizeof(seed_), fallback_state->data(), fallback_state_numel);

  return state.getIntrusivePtr();
}

RBLNGeneratorImpl* RBLNGeneratorImpl::clone_impl() const {
  auto gen = new RBLNGeneratorImpl(device().index());
  gen->set_state(*get_state());
  return gen;
}

DeviceType RBLNGeneratorImpl::device_type() {
  return DeviceType::PrivateUse1;
}

} // namespace at

namespace c10::rbln {

at::Generator make_rbln_generator(c10::DeviceIndex device_index) {
  RBLN_LOG_DEBUG("device_index={}", static_cast<int>(device_index));
  return at::make_generator<at::RBLNGeneratorImpl>(device_index);
}

const at::Generator& get_default_rbln_generator(c10::DeviceIndex device_index) {
  static const std::vector<at::Generator> generators = [] {
    const auto device_count = get_device_count();

    std::vector<at::Generator> result;
    result.reserve(device_count);

    for (c10::DeviceIndex i = 0; i < device_count; ++i) {
      auto generator = make_rbln_generator(i);
      generator.seed();
      result.emplace_back(std::move(generator));
    }

    return result;
  }();

  auto idx = device_index;
  if (idx == -1) {
    idx = get_device_index();
  }

  TORCH_CHECK(idx >= 0 && idx < static_cast<c10::DeviceIndex>(generators.size()), "Invalid RBLN device index: ", idx);

  return generators[idx];
}

} // namespace c10::rbln
