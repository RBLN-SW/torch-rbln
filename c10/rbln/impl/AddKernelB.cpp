// Option B fast path for aten::add.Tensor on RBLN PrivateUse1.
//
// V1 restrictions (guard fails -> Python slow path via at::add_outf):
//   - alpha must be 1
//   - both inputs fp16, contiguous, storage_offset==0, same shape, same device
//   - no requires_grad, numel > 0
//
// This kernel is instrumented with per-stage timing counters (g_hp) so the
// bench can break down where the ~700us steady-state cost goes.

#include <ATen/Operators.h>
#include <ATen/core/Tensor.h>
#include <c10/core/Device.h>
#include <c10/core/ScalarType.h>
#include <c10/rbln/impl/RBLNKernelTemplate.h>
#include <torch/library.h>

#include <array>
#include <chrono>

namespace c10::rbln::kcache {
namespace {

using clk = std::chrono::steady_clock;
static inline uint64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(clk::now().time_since_epoch()).count();
}

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
  const auto t0 = now_ns();

  if (!g_b_enabled.load(std::memory_order_relaxed) || g_building_entry || !add_guard(self, other, alpha)) {
    return add_fallback(self, other, alpha);
  }
  const auto t1 = now_ns();  // guard

  CacheKey key{OpId::AddTensor, self.scalar_type(), {}, static_cast<int8_t>(self.device().index())};
  key.shape.assign(self.sizes().begin(), self.sizes().end());

  const CacheEntry* entry = KernelCache::instance().find(key);
  if (entry == nullptr) {
    entry = &KernelCache::instance().emplace(key, [&]() {
      BuildGuard guard;
      return build_entry_via_python("build_add_runtime", self, other);
    });
  }
  const auto t2 = now_ns();  // find (+ miss if it happens — but bench runs post-warmup)

  // Inlined run_cached so we can time at::empty / map construction / prepare /
  // run separately. When the refactor stabilises we'll fold these probes back
  // into the template; for now keep them local to add so only one op is
  // instrumented.
  TORCH_CHECK(entry->num_outputs == 1);
  const OutProfile& op = entry->out_profiles[0];
  TORCH_CHECK(op.is_rbln_device);

  auto out = at::empty(op.shape, at::TensorOptions().dtype(op.dtype).device(self.device()));
  const auto t3 = now_ns();  // alloc

  std::map<uint32_t, uint64_t> dev_in;
  dev_in.emplace(0u, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(self.data_ptr())));
  dev_in.emplace(1u, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(other.data_ptr())));
  std::map<uint32_t, uintptr_t> cpu_in;
  std::map<uint32_t, uint64_t> dev_out;
  dev_out.emplace(0u, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(out.data_ptr())));
  std::map<uint32_t, uintptr_t> cpu_out;
  const auto t4 = now_ns();  // build_maps

  entry->runtime->PrepareInputs(dev_in, cpu_in);
  const auto t5 = now_ns();  // prepare_in

  entry->runtime->PrepareOutputs(dev_out, cpu_out);
  const auto t6 = now_ns();  // prepare_out

  entry->runtime->Run();
  const auto t7 = now_ns();  // run

  g_hp.n_calls.fetch_add(1, std::memory_order_relaxed);
  g_hp.guard_ns.fetch_add(t1 - t0, std::memory_order_relaxed);
  g_hp.find_ns.fetch_add(t2 - t1, std::memory_order_relaxed);
  g_hp.alloc_ns.fetch_add(t3 - t2, std::memory_order_relaxed);
  g_hp.build_maps_ns.fetch_add(t4 - t3, std::memory_order_relaxed);
  g_hp.prepare_in_ns.fetch_add(t5 - t4, std::memory_order_relaxed);
  g_hp.prepare_out_ns.fetch_add(t6 - t5, std::memory_order_relaxed);
  g_hp.run_ns.fetch_add(t7 - t6, std::memory_order_relaxed);
  g_hp.total_ns.fetch_add(t7 - t0, std::memory_order_relaxed);

  return out;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("add.Tensor", TORCH_FN(add_tensor_rbln_b));
}

}  // namespace
}  // namespace c10::rbln::kcache
