#pragma once

// Option B kernel cache.
//
// Each entry holds a rebel PyRblnSyncRuntime (via a strong Python reference to
// the owning DynamoRuntime) plus the cached output profiles needed to allocate
// the result tensor without re-entering Python. On cache hit the C++ kernel
// drives PrepareInputs / PrepareOutputs / Run directly via the rebel C++ API.

#include <ATen/core/ScalarType.h>
#include <c10/core/DeviceType.h>
#include <c10/macros/Export.h>
#include <c10/rbln/RBLNMacros.h>
#include <c10/util/SmallVector.h>
#include <pybind11/pybind11.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

// Forward declaration shim for rebel's PyRblnSyncRuntime. Including the real
// header (rebel/pyrbln_impl/compiled_model.h) drags in a transitive dependency
// on abseil + rebel-internal headers that aren't shipped. We don't instantiate
// the class here — only hold pointers to it and call non-virtual methods — so
// the linker binds to the mangled symbols exported by librbln.so regardless of
// how much of the body we expose. Signatures MUST match the upstream header:
//   rebel_compiler/rebel/include/rebel/pyrbln_impl/compiled_model.h (class
//   PyRblnSyncRuntime).
namespace rbln {
class PyRblnSyncRuntime {
 public:
  void Run();
  void PrepareInputs(const std::map<uint32_t, uint64_t>& device_inputs,
                     const std::map<uint32_t, uintptr_t>& cpu_inputs);
  void PrepareOutputs(const std::map<uint32_t, uint64_t>& device_outputs,
                      const std::map<uint32_t, uintptr_t>& cpu_outputs);
  // Lightweight variants: skip vmem manager re-sync and only patch the CS/GCE
  // with the new device addresses. Must match upstream header exactly:
  //   rebel/include/rebel/pyrbln_impl/compiled_model.h:162-163.
  void UpdateInputAddr(uint32_t input_idx, const std::vector<uint64_t>& device_addrs);
  void UpdateOutputAddr(uint32_t output_idx, const std::vector<uint64_t>& device_addrs);
};
}  // namespace rbln

namespace c10::rbln::kcache {

enum class OpId : uint8_t {
  AddTensor = 0,
  SubTensor = 1,
  MulTensor = 2,
  DivTensor = 3,
  Neg = 4,
};

struct CacheKey {
  OpId op;
  at::ScalarType dtype;
  c10::SmallVector<int64_t, 6> shape;
  int8_t device_index;

  bool operator==(const CacheKey& other) const noexcept {
    return op == other.op && dtype == other.dtype && device_index == other.device_index && shape == other.shape;
  }
};

struct CacheKeyHash {
  std::size_t operator()(const CacheKey& k) const noexcept {
    std::size_t h = std::hash<uint8_t>{}(static_cast<uint8_t>(k.op));
    h ^= std::hash<int>{}(static_cast<int>(k.dtype)) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    h ^= std::hash<int>{}(k.device_index) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    for (int64_t d : k.shape) {
      h ^= std::hash<int64_t>{}(d) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    }
    return h;
  }
};

struct OutProfile {
  std::vector<int64_t> shape;
  at::ScalarType dtype;
  bool is_rbln_device;  // false => cpu
};

struct CacheEntry {
  // Strong Python reference to the owning DynamoRuntime, keeps the rebel
  // compiled model and its PyRblnSyncRuntime alive for the lifetime of the
  // cache entry. Raw `runtime` is a non-owning observer into this object.
  pybind11::object py_dynamo_runtime;
  ::rbln::PyRblnSyncRuntime* runtime = nullptr;
  uint32_t num_inputs = 0;
  uint32_t num_outputs = 0;
  std::vector<OutProfile> out_profiles;
};

// Thread-safe process-global cache. Reads (hits) take a shared lock; writes
// (misses) take an exclusive lock. Callers must hold a stable reference to the
// returned entry for the duration of the kernel call; since entries are never
// evicted in V1, a raw pointer into the map is stable.
class C10_RBLN_API KernelCache {
 public:
  static KernelCache& instance();

  // Returns a pointer to the entry on hit, or nullptr on miss. The returned
  // pointer remains valid for the lifetime of the cache (no eviction in V1).
  const CacheEntry* find(const CacheKey& key);

  // Inserts an entry built by `miss_fn`. The miss function runs while holding
  // the exclusive lock; keep it short (just capture and move). If another
  // thread raced and already inserted, `miss_fn` is not called.
  const CacheEntry& emplace(const CacheKey& key, std::function<CacheEntry()> miss_fn);

 private:
  KernelCache() = default;
  std::shared_mutex mu_;
  std::unordered_map<CacheKey, CacheEntry, CacheKeyHash> map_;
};

// --------------------------------------------------------------------------
// Hot-path breakdown counters (cumulative). Bench code reads + resets these
// via the C ABI in RBLNKernelCache.cpp. Overhead per call is ~8 chrono::now
// probes (~150ns) which is small vs the >500us/op steady state.
struct HotPathCounters {
  std::atomic<uint64_t> n_calls{0};
  std::atomic<uint64_t> guard_ns{0};
  std::atomic<uint64_t> find_ns{0};
  std::atomic<uint64_t> alloc_ns{0};
  std::atomic<uint64_t> build_maps_ns{0};
  std::atomic<uint64_t> prepare_in_ns{0};
  std::atomic<uint64_t> prepare_out_ns{0};
  std::atomic<uint64_t> run_ns{0};
  std::atomic<uint64_t> total_ns{0};
};

extern HotPathCounters g_hp;

// Flag exposed to the bench: when true, AddKernelB uses UpdateInputAddr /
// UpdateOutputAddr instead of PrepareInputs / PrepareOutputs on the hot path.
extern std::atomic<bool> g_b_use_update_addr;

}  // namespace c10::rbln::kcache
