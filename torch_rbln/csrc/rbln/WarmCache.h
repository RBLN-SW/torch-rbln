#pragma once

// Warm-runtime cache for the C++ dispatch shim.
//
// Goal: on warm shim calls (cache hit), bypass the Python wrapper entirely
// and drive the rebel runtime directly from C++ through rbln_exec_api.h. This
// eliminates the per-call pybind roundtrip + Python wrapper overhead, which
// is dominated by is_cpu_fallback_cases and the compile_rbln_cached lookup on
// the Python side.
//
// Architecture:
//   - On first call of a shim op with a given input profile, the Python
//     wrapper compiles the op via torch.compile(backend="rbln") and harvests
//     the DynamoRuntime. It then installs an entry into this cache via the
//     pybind-exposed install(...) API.
//   - On subsequent calls with a matching input profile, the shim looks up
//     the entry and drives one execution through the C ABI.
//   - Entries are keyed by (schema-name, per-Tensor-input profile, per-Scalar
//     value). Shape/dtype/device changes produce a different key and trigger
//     a miss (fall back to Python, which in turn repopulates the cache for
//     the new profile).
//
// Lifetime / thread-safety:
//   - Process-global singleton cache.
//   - Reads take a shared lock (hot path); writes take an exclusive lock.
//   - Entries hold strong py::object references to the DynamoRuntime and to
//     the sync-runtime handle it owns, so the borrowed RblnSyncRuntime stays
//     valid for the cache's lifetime.
//   - No eviction; entries leave only through ``clear`` (all of them, or one
//     device's).

#include <ATen/core/ScalarType.h>
#include <c10/core/Device.h>
#include <c10/util/SmallVector.h>
#include <torch/csrc/utils/pybind.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <shared_mutex>
#include <string>
#include <unordered_map>

#include <rebel/runtime/api/rbln_exec_api.h>

namespace torch_rbln::warmcache {

// Per-tensor input profile. Shape/strides/storage_offset/dtype/device
// together pin down the exact layout the cached runtime was compiled for.
//
// Why strides + storage_offset: the warm-cache hit path passes the stack
// tensor's ``data_ptr()`` straight to the rebel runtime. The runtime was
// compiled assuming the input layout matches what the Python wrapper fed
// to ``torch.compile`` — that layout is always contig+offset=0 today
// (``compile_and_run_view_aware`` skips ``_install_warm_cache_pending``
// via the ``has_views`` gate whenever any view recipe was applied, so
// only contig+offset=0 inputs survive into install). If the key carried
// only shape+dtype, a later call with a non-contig or offset>0 input of
// the same shape would hit the contig-compiled entry and the runtime
// would read ``numel*itemsize`` contiguous bytes from a stride-aware
// pointer — silently wrong values for permute / transpose / expand and
// wrong-offset reads for narrow(dim, k>0, …). Note: this is the input-
// side counterpart of ``fix(warm-cache): bypass on non-contiguous out=
// dispatch`` (commit 6406446), which already handled the out= mirror.
struct TensorProfile {
  at::ScalarType dtype{at::ScalarType::Undefined};
  c10::SmallVector<int64_t, 6> shape;
  c10::SmallVector<int64_t, 6> strides;
  int64_t storage_offset{0};
  int8_t device_index{-1};

  bool operator==(const TensorProfile& o) const noexcept {
    return dtype == o.dtype && device_index == o.device_index && storage_offset == o.storage_offset &&
        shape == o.shape && strides == o.strides;
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
    return {.tag = Tag::Int, .i = v};
  }
  static ScalarValue fromFloat(double v) {
    return {.tag = Tag::Float, .f = v};
  }
  static ScalarValue fromBool(bool v) {
    return {.tag = Tag::Bool, .b = v};
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
  // The runtime is held as well as the DynamoRuntime that owns it: `runtime`
  // below is borrowed from the former, and an owner that rebinds its attribute
  // would otherwise drop it while an installed entry still points at it.
  pybind11::object py_dyn_runtime;
  pybind11::object py_runtime_handle;

  // Borrowed from `py_runtime_handle`'s native_handle(); valid only while it is.
  RblnSyncRuntime runtime{nullptr};

  c10::SmallVector<OutputProfile, 2> out_profiles;
};

// Shared pointer to a cache entry. Returned by ``find`` so the caller can
// safely use the entry even if another thread concurrently ``clear``s the
// key — the entry stays alive as long as any shared_ptr references it.
// Without this, a raw-pointer ``find`` could return a pointer that another
// thread invalidates before the caller reaches ``Run``, causing a
// use-after-free.
using CacheEntryPtr = std::shared_ptr<const CacheEntry>;

// Process-global cache. Entries are created via `install` on cache miss from
// the Python bootstrap path, then found via `find` on the hot path.
class WarmCache {
 public:
  static WarmCache& instance();

  // Hot path. Returns a shared_ptr to the cached entry, or empty on miss.
  // The shared_ptr keeps the entry alive across a concurrent ``clear`` from
  // a peer thread (use-after-free guard).
  CacheEntryPtr find(const CacheKey& key);

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
  // Drop the entries whose inputs live on ``device``, or all of them when no
  // device is given. An entry with no device input is dropped only by the
  // latter; no shim op has one today (each takes at least one input tensor).
  void clear(std::optional<c10::DeviceIndex> device = std::nullopt);

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
  std::unordered_map<CacheKey, std::shared_ptr<CacheEntry>, CacheKeyHash> map_;
  std::atomic<bool> enabled_{true};
};

// ---------------------------------------------------------------------------
// Interned schema-name storage. `schema_name_intern` in CacheKey is a raw
// pointer; this helper returns a pointer to a string stored in a process-
// global pool that lives forever. Thread-safe; callers typically intern once
// per-shim-op at registration time and cache the result.
const char* intern_op_name(const std::string& name);

} // namespace torch_rbln::warmcache
