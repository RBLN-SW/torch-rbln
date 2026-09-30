#include <c10/rbln/RBLNFunctions.h>
#include <c10/rbln/RBLNGenerator.h>
#include <c10/rbln/RBLNLogging.h>

#include <mutex>
#include <unordered_map>

namespace at {

RBLNGeneratorImpl::RBLNGeneratorImpl(DeviceIndex device_index, uint64_t seed)
    : GeneratorImpl(
          Device(DeviceType::PrivateUse1, device_index == -1 ? c10::rbln::get_device_index() : device_index),
          DispatchKeySet(c10::DispatchKey::PrivateUse1)),
      cpu_generator_(make_intrusive<CPUGeneratorImpl>(seed)) {}

DeviceType RBLNGeneratorImpl::device_type() {
  return DeviceType::PrivateUse1;
}

Generator RBLNGeneratorImpl::fallback_generator() const {
  return Generator(cpu_generator_);
}

void RBLNGeneratorImpl::set_current_seed(uint64_t seed) {
  std::lock_guard<std::mutex> lock(cpu_generator_->mutex_);
  cpu_generator_->set_current_seed(seed);
}

// mt19937 has no offset (it is a Philox notion); CPUGeneratorImpl rejects these the same way.
void RBLNGeneratorImpl::set_offset(uint64_t /*offset*/) {
  TORCH_CHECK(false, "RBLN Generator does not use offset");
}

uint64_t RBLNGeneratorImpl::get_offset() const {
  TORCH_CHECK(false, "RBLN Generator does not use offset");
}

uint64_t RBLNGeneratorImpl::current_seed() const {
  std::lock_guard<std::mutex> lock(cpu_generator_->mutex_);
  return cpu_generator_->current_seed();
}

uint64_t RBLNGeneratorImpl::seed() {
  std::lock_guard<std::mutex> lock(cpu_generator_->mutex_);
  return cpu_generator_->seed();
}

// CPUGeneratorImpl::set_state validates the whole blob before it assigns anything, so a
// rejected state leaves the generator untouched.
void RBLNGeneratorImpl::set_state(const c10::TensorImpl& new_state) {
  std::lock_guard<std::mutex> lock(cpu_generator_->mutex_);
  cpu_generator_->set_state(new_state);
}

c10::intrusive_ptr<c10::TensorImpl> RBLNGeneratorImpl::get_state() const {
  std::lock_guard<std::mutex> lock(cpu_generator_->mutex_);
  return cpu_generator_->get_state();
}

RBLNGeneratorImpl* RBLNGeneratorImpl::clone_impl() const {
  auto* gen = new RBLNGeneratorImpl(device().index());
  gen->set_state(*get_state());
  return gen;
}

} // namespace at

namespace c10::rbln {

namespace {

struct DefaultGenerators {
  std::mutex mutex;
  // Node-based, so the references get_default_rbln_generator() hands out stay valid as
  // devices are added.
  std::unordered_map<c10::DeviceIndex, at::Generator> by_device;
  uint64_t seed = at::default_rng_seed_val;
};

DefaultGenerators& default_generators() {
  // Leaked: a generator can still be reached from a static destructor or an atexit hook.
  static auto* generators = new DefaultGenerators();
  return *generators;
}

} // namespace

const at::Generator& get_default_rbln_generator(c10::DeviceIndex device_index) {
  const auto index = device_index == -1 ? get_device_index() : device_index;
  const auto device_count = get_device_count();
  RBLN_CHECK(
      index >= 0 && index < device_count,
      "No default generator for rbln:{}: this process has {} logical device(s)",
      static_cast<int>(index),
      static_cast<int>(device_count));

  auto& generators = default_generators();
  std::lock_guard<std::mutex> lock(generators.mutex);
  auto it = generators.by_device.find(index);
  if (it == generators.by_device.end()) {
    it = generators.by_device.emplace(index, at::make_generator<at::RBLNGeneratorImpl>(index, generators.seed)).first;
  }
  return it->second;
}

void manual_seed_all(uint64_t seed) {
  auto& generators = default_generators();
  std::lock_guard<std::mutex> lock(generators.mutex);
  generators.seed = seed;
  for (auto& entry : generators.by_device) {
    std::lock_guard<std::mutex> generator_lock(entry.second.mutex());
    entry.second.set_current_seed(seed);
  }
}

} // namespace c10::rbln
