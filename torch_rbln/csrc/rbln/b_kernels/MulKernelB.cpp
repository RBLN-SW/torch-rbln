// Option B fast path for aten::mul.Tensor on RBLN PrivateUse1.

#include <ATen/Operators.h>
#include <ATen/core/Tensor.h>
#include "RBLNKernelTemplate.h"
#include <torch/library.h>

#include <array>

namespace c10::rbln::kcache {
namespace {

bool mul_guard(const at::Tensor& a, const at::Tensor& b) {
  if (!tensor_is_v1_safe(a) || !tensor_is_v1_safe(b)) return false;
  if (a.sizes() != b.sizes()) return false;
  if (a.device() != b.device()) return false;
  return true;
}

at::Tensor mul_fallback(const at::Tensor& self, const at::Tensor& other) {
  auto out = at::empty(self.sizes(), self.options());
  at::_ops::mul_out::call(self, other, out);
  return out;
}

at::Tensor mul_tensor_rbln_b(const at::Tensor& self, const at::Tensor& other) {
  if (!g_b_enabled.load(std::memory_order_relaxed) || g_building_entry || !mul_guard(self, other)) {
    return mul_fallback(self, other);
  }

  CacheKey key{OpId::MulTensor, self.scalar_type(), {}, static_cast<int8_t>(self.device().index())};
  key.shape.assign(self.sizes().begin(), self.sizes().end());

  const CacheEntry* entry = KernelCache::instance().find(key);
  if (entry == nullptr) {
    entry = &KernelCache::instance().emplace(key, [&]() {
      BuildGuard guard;
      return build_entry_via_python("build_mul_runtime", self, other);
    });
  }

  std::array<const at::Tensor*, 2> ins{&self, &other};
  return run_cached<2>(*entry, ins);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("mul.Tensor", TORCH_FN(mul_tensor_rbln_b));
}

}  // namespace
}  // namespace c10::rbln::kcache
