#pragma once

#include <ATen/core/Generator.h>
#include <c10/rbln/RBLNMacros.h>

namespace at {

struct C10_RBLN_API RBLNGeneratorImpl : public GeneratorImpl {
 public:
  RBLNGeneratorImpl(DeviceIndex device_index = -1);
  ~RBLNGeneratorImpl() override = default;

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
  uint64_t offset_;
};

} // namespace at
