#pragma once

#include <ATen/core/Generator.h>
#include <c10/rbln/RBLNMacros.h>

namespace at {

struct C10_RBLN_API RBLNGeneratorImpl : public GeneratorImpl {
 public:
  RBLNGeneratorImpl(DeviceIndex device_index = -1);
  ~RBLNGeneratorImpl() override = default;

  at::Generator get_fallback_generator() const;
  static c10::DeviceType device_type();

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

  uint64_t seed_;
  c10::intrusive_ptr<GeneratorImpl> cpu_generator_;
};

} // namespace at

namespace c10::rbln {

/**
 * @brief Creates a new generator for the specified RBLN device.
 *
 * The returned generator is independent of the device's default generator
 * and can be explicitly passed to random-number-generating operators.
 *
 * @param device_index The RBLN device index.
 * @return A newly created RBLN generator associated with `device_index`.
 */
C10_RBLN_API at::Generator make_rbln_generator(c10::DeviceIndex device_index);

/**
 * @brief Returns the persistent default generator for the specified RBLN
 * device.
 *
 * The same generator is returned for subsequent calls with the same device
 * index. If `device_index` is `-1`, the generator for the current RBLN device
 * is returned.
 *
 * @param device_index The RBLN device index, or `-1` to use the current device.
 * @return The default generator associated with the requested RBLN device.
 */
C10_RBLN_API const at::Generator& get_default_rbln_generator(c10::DeviceIndex device_index);

} // namespace c10::rbln