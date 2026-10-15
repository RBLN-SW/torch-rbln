#pragma once

#include <ATen/CPUGeneratorImpl.h>
#include <ATen/core/Generator.h>
#include <c10/rbln/RBLNMacros.h>

namespace at {

// No random op runs on the device yet: every one takes the CPU fallback, which samples from
// cpu_generator_. That CPU generator therefore holds the whole RNG state, and every override
// below delegates to it under its mutex -- the one the CPU kernels lock while they sample,
// which the Python Generator methods (locking this generator's own mutex_) would otherwise race.
struct C10_RBLN_API RBLNGeneratorImpl : public GeneratorImpl {
 public:
  explicit RBLNGeneratorImpl(DeviceIndex device_index = -1, uint64_t seed = default_rng_seed_val);
  ~RBLNGeneratorImpl() override = default;

  static DeviceType device_type();

  // The CPU generator to hand a CPU fallback kernel in place of this one.
  Generator fallback_generator() const;

 private:
  // Overridden from GeneratorImpl:
  void set_current_seed(uint64_t seed) override;
  void set_offset(uint64_t offset) override;
  uint64_t get_offset() const override;
  uint64_t current_seed() const override;
  uint64_t seed() override;
  void set_state(const c10::TensorImpl& new_state) override;
  c10::intrusive_ptr<c10::TensorImpl> get_state() const override;
  RBLNGeneratorImpl* clone_impl() const override;

  c10::intrusive_ptr<CPUGeneratorImpl> cpu_generator_;
};

} // namespace at

namespace c10::rbln {

/**
 * @brief Returns the default generator of an RBLN logical device, creating it on first use.
 *
 * Plan-only: it claims no device, so torch.manual_seed() before a launcher assigns RBLN_DEVICES
 * leaves the mapping open. A generator is created with the seed of the last manual_seed_all()
 * call, or default_rng_seed_val before any, so a device added by a later remap is seeded the
 * same as the rest.
 *
 * @param device_index The logical device index, or -1 for the current device.
 */
C10_RBLN_API const at::Generator& get_default_rbln_generator(c10::DeviceIndex device_index = -1);

/**
 * @brief Seeds every default generator created so far, and every one created later.
 *
 * Queries no device, so it cannot fail where torch.manual_seed() calls it: unconditionally,
 * including on a host with no NPU or a malformed RBLN_* configuration.
 */
C10_RBLN_API void manual_seed_all(uint64_t seed);

} // namespace c10::rbln
