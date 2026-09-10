#include <torch_rbln/csrc/rbln/WarmCache.h>

#include <fcntl.h>
#include <sys/types.h>
#include <unistd.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <vector>

namespace torch_rbln::warmcache {

// ---- Hash -------------------------------------------------------------------

namespace {
inline void hash_combine_size_t(std::size_t& seed, std::size_t v) {
  seed ^= v + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
}
} // namespace

std::size_t CacheKeyHash::operator()(const CacheKey& k) const noexcept {
  std::size_t h = std::hash<const void*>{}(static_cast<const void*>(k.schema_name_intern));
  for (const auto& in : k.inputs) {
    hash_combine_size_t(h, std::hash<int>{}(static_cast<int>(in.dtype)));
    hash_combine_size_t(h, std::hash<int>{}(in.device_index));
    hash_combine_size_t(h, std::hash<int64_t>{}(in.storage_offset));
    for (int64_t d : in.shape) {
      hash_combine_size_t(h, std::hash<int64_t>{}(d));
    }
    // shape/strides separator so e.g. shape=(2,3) strides=(3,1) differs
    // from shape=(2,3,1) strides=(3,1).
    hash_combine_size_t(h, 0x5a5a5a5aULL);
    for (int64_t s : in.strides) {
      hash_combine_size_t(h, std::hash<int64_t>{}(s));
    }
    hash_combine_size_t(h, 0xa5a5a5a5ULL);
  }
  for (const auto& s : k.scalars) {
    hash_combine_size_t(h, std::hash<uint8_t>{}(static_cast<uint8_t>(s.tag)));
    switch (s.tag) {
      case ScalarValue::Tag::Int:
        hash_combine_size_t(h, std::hash<int64_t>{}(s.i));
        break;
      case ScalarValue::Tag::Float:
        hash_combine_size_t(h, std::hash<double>{}(s.f));
        break;
      case ScalarValue::Tag::Bool:
        hash_combine_size_t(h, std::hash<bool>{}(s.b));
        break;
      case ScalarValue::Tag::Missing:
        break;
    }
  }
  return h;
}

// ---- Intern pool ------------------------------------------------------------

namespace {
std::mutex& intern_mutex() {
  static std::mutex m;
  return m;
}
std::unordered_map<std::string, const char*>& intern_pool() {
  static std::unordered_map<std::string, const char*> p;
  return p;
}
} // namespace

const char* intern_op_name(const std::string& name) {
  std::lock_guard<std::mutex> lk(intern_mutex());
  auto it = intern_pool().find(name);
  if (it != intern_pool().end())
    return it->second;
  // The storage lives forever (owned by the map's key string).
  auto [ins, _] = intern_pool().emplace(name, nullptr);
  ins->second = ins->first.c_str();
  return ins->second;
}

// ---- WarmCache singleton ----------------------------------------------------

WarmCache& WarmCache::instance() {
  // Leaky singleton: CacheEntry holds a strong pybind11::object reference to
  // the DynamoRuntime. Running ~WarmCache after Py_Finalize() decrefs that
  // py::object on a finalized interpreter and aborts inside libpython.
  // Allocate with `new` so the entries outlive Python finalize; the OS
  // reclaims the process memory at exit.
  //
  // Default ON: hot path drives rebel runtime directly from C++ on warm hits;
  // miss-path entries that fail v-memory lookup are retired by try_warmcache_hit
  // so cold-path correctness is preserved.
  static auto* c = [] {
    auto* p = new WarmCache();
    p->set_enabled(true);
    return p;
  }();
  return *c;
}

namespace {
// Custom deleter for CacheEntry. ``py_dyn_runtime`` holds a strong
// pybind11::object reference; its destructor decrefs a Python object and
// must therefore run under the GIL. The shared_ptr's last refcount drop
// can happen on any thread (e.g. when a hit-path's local shared_ptr goes
// out of scope after the calling Python thread released the GIL), so we
// route every destruction through this deleter to guarantee GIL hold.
//
// Because it takes the GIL, the last reference to an entry must never drop
// while ``mu_`` is held: a thread that already holds the GIL may be waiting
// for ``mu_`` (``clear`` and ``size`` are called from Python), and the two
// would wait on each other. ``disable`` and ``clear`` therefore move the
// references they drop out of the map under the lock and release them after
// unlocking.
void cache_entry_deleter(CacheEntry* p) {
  if (p == nullptr) {
    return;
  }
  pybind11::gil_scoped_acquire gil;
  delete p;
}
} // namespace

CacheEntryPtr WarmCache::find(const CacheKey& key) {
  if (!enabled_.load(std::memory_order_relaxed))
    return nullptr;
  std::shared_lock<std::shared_mutex> rd(mu_);
  auto it = map_.find(key);
  return (it != map_.end()) ? it->second : nullptr;
}

void WarmCache::install(CacheKey key, const CacheEntry& entry) {
  if (!enabled_.load(std::memory_order_relaxed))
    return;
  std::unique_lock<std::shared_mutex> wr(mu_);
  std::shared_ptr<CacheEntry> entry_ptr(new CacheEntry(entry), &cache_entry_deleter);
  // First-writer-wins: if another thread beat us to it, keep the earlier one.
  map_.try_emplace(std::move(key), std::move(entry_ptr));
}

void WarmCache::disable(const CacheKey& key) {
  std::shared_ptr<CacheEntry> tombstone(new CacheEntry(), &cache_entry_deleter);
  std::shared_ptr<CacheEntry> replaced;
  {
    std::unique_lock<std::shared_mutex> wr(mu_);
    auto it = map_.find(key);
    if (it == map_.end()) {
      map_.emplace(key, std::move(tombstone));
    } else {
      replaced = std::move(it->second);
      it->second = std::move(tombstone);
    }
  }
  // ``replaced`` dies here, after the lock is released (see the deleter). The
  // hit path that called us still holds the failed entry, so what dies here
  // is at most a tombstone left by a concurrent failure on the same key; an
  // in-flight find() borrower keeps its entry alive until it releases.
}

size_t WarmCache::size() {
  std::shared_lock<std::shared_mutex> rd(mu_);
  return map_.size();
}

void WarmCache::clear(std::optional<c10::DeviceIndex> device) {
  // The dropped entries die after the lock is released (see the deleter).
  std::vector<std::shared_ptr<CacheEntry>> dropped;
  {
    std::unique_lock<std::shared_mutex> wr(mu_);
    dropped.reserve(map_.size());
    for (auto it = map_.begin(); it != map_.end();) {
      const auto& inputs = it->first.inputs;
      const bool drop = !device.has_value() || std::any_of(inputs.begin(), inputs.end(), [&](const TensorProfile& tp) {
        return tp.device_index == *device;
      });
      if (drop) {
        dropped.push_back(std::move(it->second));
        it = map_.erase(it);
      } else {
        ++it;
      }
    }
  }
}

namespace {
thread_local bool t_building_entry = false;
thread_local bool t_inject_hit_failure = false;
} // namespace

bool WarmCache::is_building_entry() {
  return t_building_entry;
}
void WarmCache::enter_building() {
  t_building_entry = true;
}
void WarmCache::exit_building() {
  t_building_entry = false;
}

void WarmCache::inject_hit_failure() {
  t_inject_hit_failure = true;
}

bool WarmCache::consume_injected_hit_failure() {
  const bool v = t_inject_hit_failure;
  t_inject_hit_failure = false;
  return v;
}

} // namespace torch_rbln::warmcache
