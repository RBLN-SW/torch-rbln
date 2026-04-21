#include <ATen/core/ScalarType.h>
#include "RBLNKernelCache.h"
#include <c10/rbln/RBLNMacros.h>
#include <c10/util/Exception.h>

#include <atomic>
#include <string>

namespace c10::rbln::kernel {

// Process-wide state for all C-kernel.
std::atomic<bool> g_c_kernel_enabled{true};
thread_local bool g_building_entry = false;
#if C10_RBLN_C_KERNEL_TIMING
HotPathCounters g_hp;
#endif

KernelCache& KernelCache::instance() {
  static KernelCache cache;
  return cache;
}

const CacheEntry* KernelCache::find(const CacheKey& key) {
  std::shared_lock<std::shared_mutex> rd(mu_);
  auto it = map_.find(key);
  return (it != map_.end()) ? &it->second : nullptr;
}

const CacheEntry& KernelCache::emplace(const CacheKey& key, std::function<CacheEntry()> miss_fn) {
  {
    std::shared_lock<std::shared_mutex> rd(mu_);
    auto it = map_.find(key);
    if (it != map_.end()) return it->second;
  }

  CacheEntry entry = miss_fn();

  std::unique_lock<std::shared_mutex> wr(mu_);
  auto [it, inserted] = map_.emplace(key, std::move(entry));
  return it->second;
}

at::ScalarType dtype_from_rbln_string(const std::string& s) {
  if (s == "float16") return at::kHalf;
  if (s == "float32") return at::kFloat;
  if (s == "bfloat16") return at::kBFloat16;
  if (s == "int32") return at::kInt;
  if (s == "int64") return at::kLong;
  if (s == "int16") return at::kShort;
  if (s == "int8") return at::kChar;
  if (s == "uint8") return at::kByte;
  if (s == "bool") return at::kBool;
  TORCH_CHECK(false, "rbln dtype string not mapped to torch dtype: ", s);
}

}  // namespace c10::rbln::kernel

// C ABI toggle — reachable from Python via ctypes, lets benchmarks flip the
// C-kernel path on/off within one process for all kernels.
extern "C" C10_RBLN_API void c10_rbln_c_kernel_set_enabled(int enabled) {
  c10::rbln::kernel::g_c_kernel_enabled.store(enabled != 0, std::memory_order_relaxed);
}

extern "C" C10_RBLN_API int c10_rbln_c_kernel_get_enabled() {
  return c10::rbln::kernel::g_c_kernel_enabled.load(std::memory_order_relaxed) ? 1 : 0;
}

// Hot-path breakdown counters — read-then-reset via single call. When
// C10_RBLN_C_KERNEL_TIMING is 0 the counters don't exist and this returns zeros.
// `out_ns` must point to 7 uint64_t slots, in the order:
//   [n_calls, alloc, build_maps, prepare_in, prepare_out, run, total]
extern "C" C10_RBLN_API void c10_rbln_c_kernel_read_timing(uint64_t* out_ns) {
#if C10_RBLN_C_KERNEL_TIMING
  using c10::rbln::kernel::g_hp;
  out_ns[0] = g_hp.n_calls.exchange(0, std::memory_order_relaxed);
  out_ns[1] = g_hp.alloc_ns.exchange(0, std::memory_order_relaxed);
  out_ns[2] = g_hp.build_maps_ns.exchange(0, std::memory_order_relaxed);
  out_ns[3] = g_hp.prepare_in_ns.exchange(0, std::memory_order_relaxed);
  out_ns[4] = g_hp.prepare_out_ns.exchange(0, std::memory_order_relaxed);
  out_ns[5] = g_hp.run_ns.exchange(0, std::memory_order_relaxed);
  out_ns[6] = g_hp.total_ns.exchange(0, std::memory_order_relaxed);
#else
  for (int i = 0; i < 7; ++i) out_ns[i] = 0;
#endif
}
