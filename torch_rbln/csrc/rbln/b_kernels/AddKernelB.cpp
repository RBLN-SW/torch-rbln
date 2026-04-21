// Option B fast path for aten::add.Tensor on RBLN PrivateUse1.
//
// V1 restrictions (guard fails -> Python slow path via aten::add_out):
//   - alpha must be 1
//   - both inputs fp16, contiguous, storage_offset==0, same shape, same device
//   - no requires_grad, numel > 0

#include <ATen/Operators.h>
#include <ATen/core/Tensor.h>
#include "RBLNKernelTemplate.h"
#include <torch/library.h>

#include <array>

namespace c10::rbln::kcache {
namespace {

bool add_guard(const at::Tensor& a, const at::Tensor& b, const at::Scalar& alpha) {
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

at::Tensor add_fallback(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
  auto out = at::empty(self.sizes(), self.options());
  at::_ops::add_out::call(self, other, alpha, out);
  return out;
}

at::Tensor add_tensor_rbln_b(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
  if (!g_b_enabled.load(std::memory_order_relaxed) || g_building_entry || !add_guard(self, other, alpha)) {
    return add_fallback(self, other, alpha);
  }

  CacheKey key{OpId::AddTensor, self.scalar_type(), {}, static_cast<int8_t>(self.device().index())};
  key.shape.assign(self.sizes().begin(), self.sizes().end());

  const CacheEntry* entry = KernelCache::instance().find(key);
  if (entry == nullptr) {
    entry = &KernelCache::instance().emplace(key, [&]() {
      BuildGuard guard;
      return build_entry_via_python("build_add_runtime", self, other);
    });
  }

  std::array<const at::Tensor*, 2> ins{&self, &other};
  return run_cached<2>(*entry, ins);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("add.Tensor", TORCH_FN(add_tensor_rbln_b));
}

}  // namespace
}  // namespace c10::rbln::kcache
