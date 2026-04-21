// Option B fast path for aten::neg on RBLN PrivateUse1.
//
// Unary op — tests the template on a single-input arity to verify generality.

#include <ATen/Operators.h>
#include <ATen/core/Tensor.h>
#include "RBLNKernelTemplate.h"
#include <torch/library.h>

#include <array>

namespace c10::rbln::kcache {
namespace {

bool neg_guard(const at::Tensor& a) {
  return tensor_is_v1_safe(a);
}

at::Tensor neg_fallback(const at::Tensor& self) {
  auto out = at::empty(self.sizes(), self.options());
  at::_ops::neg_out::call(self, out);
  return out;
}

at::Tensor neg_rbln_b(const at::Tensor& self) {
  if (!g_b_enabled.load(std::memory_order_relaxed) || g_building_entry || !neg_guard(self)) {
    return neg_fallback(self);
  }

  CacheKey key{OpId::Neg, self.scalar_type(), {}, static_cast<int8_t>(self.device().index())};
  key.shape.assign(self.sizes().begin(), self.sizes().end());

  const CacheEntry* entry = KernelCache::instance().find(key);
  if (entry == nullptr) {
    entry = &KernelCache::instance().emplace(key, [&]() {
      BuildGuard guard;
      return build_entry_via_python("build_neg_runtime", self);
    });
  }

  std::array<const at::Tensor*, 1> ins{&self};
  return run_cached<1>(*entry, ins);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("neg", TORCH_FN(neg_rbln_b));
}

}  // namespace
}  // namespace c10::rbln::kcache
