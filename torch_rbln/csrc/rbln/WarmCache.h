#pragma once

// Warm-runtime cache for the C++ dispatch shim.
//
// Goal: on warm shim calls (cache hit), bypass the Python wrapper entirely
// and drive the rebel runtime directly from C++ via PyRblnSyncRuntime. This
// eliminates the per-call pybind roundtrip + Python wrapper overhead
// (~100-200us, dominated by is_cpu_fallback_cases and compile_rbln_cached
// lookup on the Python side).
//
// Architecture:
//   - On first call of a shim op with a given input profile, the Python
//     wrapper compiles the op via torch.compile(backend="rbln") and harvests
//     the DynamoRuntime. It then installs an entry into this cache via the
//     pybind-exposed install(...) API.
//   - On subsequent calls with a matching input profile, the shim looks up
//     the entry and calls PyRblnSyncRuntime::{PrepareInputs, PrepareOutputs,
//     Run} directly.
//   - Entries are keyed by (schema-name, per-Tensor-input profile, per-Scalar
//     value). Shape/dtype/device changes produce a different key and trigger
//     a miss (fall back to Python, which in turn repopulates the cache for
//     the new profile).
//
// Lifetime / thread-safety:
//   - Process-global singleton cache.
//   - Reads take a shared lock (hot path); writes take an exclusive lock.
//   - Entries hold a strong py::object reference to the DynamoRuntime so the
//     underlying C++ rebel::PyRblnSyncRuntime is kept alive.
//   - No eviction in V1; a raw pointer into the map is stable for the
//     lifetime of the cache.

#include <ATen/core/ScalarType.h>
#include <c10/util/SmallVector.h>
#include <torch/csrc/utils/pybind.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <map>
#include <shared_mutex>
#include <string>
#include <unordered_map>

// Forward declaration of rebel's PyRblnSyncRuntime.
//
// Including the real header (rebel/pyrbln_impl/compiled_model.h) pulls in a
// transitive dependency on absl + other rebel-internal symbols that are not
// part of the shipped prod librbln.so. The linker resolves these method
// symbols against librbln.so at load time; signatures MUST match upstream.
// Reference: /home/chanheo/rebel_compiler/rebel/include/rebel/pyrbln_impl/
//            compiled_model.h.
namespace rbln {
class PyRblnSyncRuntime {
 public:
  void Run();
  void PrepareInputs(
      const std::map<uint32_t, uint64_t>& device_inputs,
      const std::map<uint32_t, uintptr_t>& cpu_inputs);
  void PrepareOutputs(
      const std::map<uint32_t, uint64_t>& device_outputs,
      const std::map<uint32_t, uintptr_t>& cpu_outputs);
};
} // namespace rbln

namespace torch_rbln::warmcache {

// Per-tensor input profile. Shape/dtype/device together guard against
// dispatching a cached runtime at a mismatched tensor.
struct TensorProfile {
  at::ScalarType dtype{at::ScalarType::Undefined};
  c10::SmallVector<int64_t, 6> shape;
  int8_t device_index{-1};

  bool operator==(const TensorProfile& o) const noexcept {
    return dtype == o.dtype && device_index == o.device_index && shape == o.shape;
  }
};

// Scalar values appearing as positional/keyword args. These are included
// because rebel backends commonly specialize the compiled graph on scalars
// (e.g. clamp's min/max, pow's exponent). Mismatched scalars must miss and
// rebuild.
struct ScalarValue {
  enum class Tag : uint8_t { Int, Float, Bool, Missing };
  Tag tag{Tag::Missing};
  int64_t i{0};
  double f{0.0};
  bool b{false};

  static ScalarValue fromInt(int64_t v) {
    return {Tag::Int, v, 0.0, false};
  }
  static ScalarValue fromFloat(double v) {
    return {Tag::Float, 0, v, false};
  }
  static ScalarValue fromBool(bool v) {
    return {Tag::Bool, 0, 0.0, v};
  }
  static ScalarValue missing() {
    return {};
  }

  bool operator==(const ScalarValue& o) const noexcept {
    if (tag != o.tag)
      return false;
    switch (tag) {
      case Tag::Int:
        return i == o.i;
      case Tag::Float:
        return f == o.f; // bit-identical compare ok for our use
      case Tag::Bool:
        return b == o.b;
      case Tag::Missing:
        return true;
    }
    return false;
  }
};

// Full cache key. `schema_name_intern` is an interned pointer (we compare by
// pointer equality, not string equality). Callers guarantee stability by
// using the op's fully-qualified name stored in the shim registry.
struct CacheKey {
  const char* schema_name_intern{nullptr};
  c10::SmallVector<TensorProfile, 4> inputs;
  c10::SmallVector<ScalarValue, 4> scalars;

  bool operator==(const CacheKey& o) const noexcept {
    return schema_name_intern == o.schema_name_intern && inputs == o.inputs && scalars == o.scalars;
  }
};

struct CacheKeyHash {
  std::size_t operator()(const CacheKey& k) const noexcept;
};

// Per-output descriptor. Shape/dtype are needed to allocate the output
// tensor on the hit path when the op does not receive an `out=` argument.
struct OutputProfile {
  c10::SmallVector<int64_t, 6> shape;
  at::ScalarType dtype{at::ScalarType::Undefined};
  bool is_rbln_device{true};
};

struct CacheEntry {
  // Strong reference to the DynamoRuntime Python object; keeps the underlying
  // rebel PyRblnSyncRuntime alive for the cache's lifetime.
  pybind11::object py_dyn_runtime;

  // Non-owning observer into the PyRblnSyncRuntime C++ instance. Raw pointer
  // lifetime is scoped to `py_dyn_runtime`.
  ::rbln::PyRblnSyncRuntime* runtime{nullptr};

  uint32_t num_inputs{0};
  uint32_t num_outputs{0};
  c10::SmallVector<OutputProfile, 2> out_profiles;
};

// Process-global cache. Entries are created via `install` on cache miss from
// the Python bootstrap path, then found via `find` on the hot path.
class WarmCache {
 public:
  static WarmCache& instance();

  // Hot path. Returns pointer to cached entry, or nullptr on miss. The
  // returned pointer is stable for the process lifetime (no eviction in V1).
  const CacheEntry* find(const CacheKey& key);

  // Drop a single entry — used when a hit attempt fails at runtime so the
  // next dispatch falls through to the pybind miss path (which exercises
  // the DynamoRuntime wrapper that handles edge-case v-memory routing).
  void erase(const CacheKey& key);

  // Miss path. Inserts entry under `key` if not already present. Called from
  // Python via pybind after a successful torch.compile. If a concurrent
  // inserter wins the race, this is a no-op (first writer wins).
  void install(CacheKey key, const CacheEntry& entry);

  // Enable/disable the warm-cache path globally. When disabled, find() always
  // returns nullptr. Disabled path leaves `install` a no-op too to avoid
  // cache bloat during bisection/bench.
  void set_enabled(bool v) {
    enabled_.store(v, std::memory_order_relaxed);
  }
  bool is_enabled() const {
    return enabled_.load(std::memory_order_relaxed);
  }

  size_t size();
  void clear();

  // Reentrancy guard used by the miss path: while Python is driving
  // torch.compile, any ATen dispatch that lands back on a shim op must take
  // the slow path (the cache entry does not exist yet; attempting to hit
  // would cause infinite recursion via a partially-built DynamoRuntime).
  // The guard is thread-local.
  static bool is_building_entry();
  static void enter_building();
  static void exit_building();

 private:
  WarmCache() = default;
  std::shared_mutex mu_;
  std::unordered_map<CacheKey, CacheEntry, CacheKeyHash> map_;
  std::atomic<bool> enabled_{true};
};

// ---------------------------------------------------------------------------
// Interned schema-name storage. `schema_name_intern` in CacheKey is a raw
// pointer; this helper returns a pointer to a string stored in a process-
// global pool that lives forever. Thread-safe; callers typically intern once
// per-shim-op at registration time and cache the result.
const char* intern_op_name(const std::string& name);

} // namespace torch_rbln::warmcache
