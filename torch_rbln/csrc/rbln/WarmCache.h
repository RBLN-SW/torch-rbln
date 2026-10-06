#pragma once

// Warm-op cache for the C++ dispatch shim.
//
// On warm shim calls (cache hit) the op runs from C++ without entering Python:
//   - The first call of a shim op with a given input profile goes to the Python
//     wrapper, which compiles the op into an OpFunction, runs it, and installs
//     it here with the positions of the call's tensors it took.
//   - Later calls with a matching input profile find the entry and run its
//     OpFunction over the stack's tensors.
//   - Entries are keyed by (schema-name, per-Tensor-input profile, per-Scalar
//     value, float32 precision). Shape/dtype/device changes produce a different
//     key and miss.
//
// Process-global; reads take a shared lock (hot path), writes an exclusive
// one. Entries leave through ``clear`` (all of them, or one device's) and
// ``forget`` (those of a function the compiled op cache lets go of).

#include <ATen/core/ScalarType.h>
#include <c10/core/Device.h>
#include <c10/util/SmallVector.h>
#include <rbln/runtime/precision.h>
#include <torch_rbln/csrc/rbln/OpFunction.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <shared_mutex>
#include <string>
#include <unordered_map>

namespace torch_rbln::warmcache {

// Per-tensor input profile. The hit path binds the stack tensor's storage as
// is, so strides and storage_offset are part of the key: a view of the same
// shape as the contiguous tensor an entry was installed for must miss.
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
// because an op is compiled with its scalars as constants
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
  // An op that makes a float32 value compiles to another program at each.
  ::rbln::runtime::Float32Precision float32_precision{};

  bool operator==(const CacheKey& o) const noexcept {
    return schema_name_intern == o.schema_name_intern && inputs == o.inputs && scalars == o.scalars &&
        float32_precision == o.float32_precision;
  }
};

struct CacheKeyHash {
  std::size_t operator()(const CacheKey& k) const noexcept;
};

struct CacheEntry {
  std::shared_ptr<OpFunction> function;
  // For each input of `function`, the position of the tensor it takes among the
  // call's tensor arguments, as ``build_cache_key`` walks them.
  c10::SmallVector<uint32_t, 4> inputs;
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
  // Python once the op compiled and ran. If a concurrent inserter wins the
  // race, this is a no-op (first writer wins).
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
  // Drop the entries that run ``function``.
  void forget(const OpFunction* function);

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
