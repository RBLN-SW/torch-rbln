// C++ fast path for aten::sub.Tensor on RBLN PrivateUse1.
// See AddKernelB.cpp for structural commentary.

#include <ATen/Operators.h>
#include <ATen/core/Tensor.h>
#include "RBLNKernelTemplate.h"
#include <torch/library.h>

#include <array>

namespace c10::rbln::kernel {
namespace {

bool sub_guard(const at::Tensor& a, const at::Tensor& b, const at::Scalar& alpha) {
  if (alpha.isFloatingPoint()) {
    if (alpha.toDouble() != 1.0) return false;
  } else if (alpha.isIntegral(false)) {
    if (alpha.toLong() != 1) return false;
  } else {
    return false;
  }
  if (!tensor_is_v1_safe(a) || !tensor_is_v1_safe(b)) return false;
  if (a.sizes() != b.sizes()) return false;
  if (a.device() != b.device()) return false;
  return true;
}

at::Tensor sub_fallback(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
  auto out = at::empty(self.sizes(), self.options());
  at::_ops::sub_out::call(self, other, alpha, out);
  return out;
}

at::Tensor sub_tensor_rbln(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
  if (!g_c_kernel_enabled.load(std::memory_order_relaxed) || g_building_entry || !sub_guard(self, other, alpha)) {
    return sub_fallback(self, other, alpha);
  }

  CacheKey key{OpId::SubTensor, self.scalar_type(), {}, static_cast<int8_t>(self.device().index())};
  key.shape.assign(self.sizes().begin(), self.sizes().end());

  const CacheEntry* entry = KernelCache::instance().find(key);
  if (entry == nullptr) {
    entry = &KernelCache::instance().emplace(key, [&]() {
      BuildGuard guard;
      return build_entry_via_python("build_sub_runtime", self, other);
    });
  }

  std::array<const at::Tensor*, 2> ins{&self, &other};
  return run_cached<2>(*entry, ins);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("sub.Tensor", TORCH_FN(sub_tensor_rbln));
}

}  // namespace
}  // namespace c10::rbln::kernel
