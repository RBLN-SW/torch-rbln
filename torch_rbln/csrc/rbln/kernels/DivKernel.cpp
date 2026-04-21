// C++ fast path for aten::div.Tensor on RBLN PrivateUse1.

#include <ATen/Operators.h>
#include <ATen/core/Tensor.h>
#include "RBLNKernelTemplate.h"
#include <torch/library.h>

#include <array>

namespace c10::rbln::kernel {
namespace {

bool div_guard(const at::Tensor& a, const at::Tensor& b) {
  if (!tensor_is_v1_safe(a) || !tensor_is_v1_safe(b)) return false;
  if (a.sizes() != b.sizes()) return false;
  if (a.device() != b.device()) return false;
  return true;
}

at::Tensor div_fallback(const at::Tensor& self, const at::Tensor& other) {
  auto out = at::empty(self.sizes(), self.options());
  at::_ops::div_out::call(self, other, out);
  return out;
}

at::Tensor div_tensor_rbln(const at::Tensor& self, const at::Tensor& other) {
  if (!g_c_kernel_enabled.load(std::memory_order_relaxed) || g_building_entry || !div_guard(self, other)) {
    return div_fallback(self, other);
  }

  CacheKey key{OpId::DivTensor, self.scalar_type(), {}, static_cast<int8_t>(self.device().index())};
  key.shape.assign(self.sizes().begin(), self.sizes().end());

  const CacheEntry* entry = KernelCache::instance().find(key);
  if (entry == nullptr) {
    entry = &KernelCache::instance().emplace(key, [&]() {
      BuildGuard guard;
      return build_entry_via_python("build_div_runtime", self, other);
    });
  }

  std::array<const at::Tensor*, 2> ins{&self, &other};
  return run_cached<2>(*entry, ins);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("div.Tensor", TORCH_FN(div_tensor_rbln));
}

}  // namespace
}  // namespace c10::rbln::kernel
