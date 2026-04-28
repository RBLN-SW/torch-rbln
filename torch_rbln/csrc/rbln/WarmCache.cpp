#include <torch_rbln/csrc/rbln/WarmCache.h>

#include <fcntl.h>
#include <sys/types.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>

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
    for (int64_t d : in.shape) {
      hash_combine_size_t(h, std::hash<int64_t>{}(d));
    }
    // size-separator so (a,b) vs (a,b,0) differ
    hash_combine_size_t(h, 0x5a5a5a5aULL);
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

namespace {
// Default OFF in V1 — enable explicitly via TORCH_RBLN_WARMCACHE=1 (or "on"/"true").
//
// Rationale: the warm-cache hot path calls into rebel runtime methods with
// raw v-memory addresses. Some tensor allocation paths (e.g. composite
// decompositions that allocate output via empty()) produce data_ptr values
// whose v-memory keys are not yet registered with the runtime, tripping
// rebel's `Invalid key_vaddr=0` guard. The pybind miss path goes through
// DynamoRuntime which performs the v-memory bookkeeping that handles those
// cases. Until we land a clean integration with rebel's allocator metadata
// (FINE-565), we keep the cache off by default and let users opt in for
// profiling / micro-bench scenarios where the savings are measurable.
bool env_default_enabled() {
  const char* env = std::getenv("TORCH_RBLN_WARMCACHE");
  if (env == nullptr || env[0] == '\0')
    return false;
  if (env[0] == '1')
    return true;
  std::string v(env);
  if (v == "on" || v == "ON" || v == "true" || v == "TRUE")
    return true;
  return false;
}
} // namespace

WarmCache& WarmCache::instance() {
  // Leaky singleton: CacheEntry holds a strong pybind11::object reference to
  // the DynamoRuntime. Running ~WarmCache after Py_Finalize() decrefs that
  // py::object on a finalized interpreter and aborts inside libpython.
  // Allocate with `new` so the entries outlive Python finalize; the OS
  // reclaims the process memory at exit.
  static auto* c = [] {
    auto* p = new WarmCache();
    p->set_enabled(env_default_enabled());
    return p;
  }();
  return *c;
}

const CacheEntry* WarmCache::find(const CacheKey& key) {
  if (!enabled_.load(std::memory_order_relaxed))
    return nullptr;
  std::shared_lock<std::shared_mutex> rd(mu_);
  auto it = map_.find(key);
  return (it != map_.end()) ? &it->second : nullptr;
}

void WarmCache::install(CacheKey key, const CacheEntry& entry) {
  if (!enabled_.load(std::memory_order_relaxed))
    return;
  std::unique_lock<std::shared_mutex> wr(mu_);
  // First-writer-wins: if another thread beat us to it, keep the earlier one.
  map_.try_emplace(std::move(key), entry);
}

void WarmCache::erase(const CacheKey& key) {
  std::unique_lock<std::shared_mutex> wr(mu_);
  map_.erase(key);
}

size_t WarmCache::size() {
  std::shared_lock<std::shared_mutex> rd(mu_);
  return map_.size();
}

void WarmCache::clear() {
  std::unique_lock<std::shared_mutex> wr(mu_);
  map_.clear();
}

namespace {
thread_local bool t_building_entry = false;
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

} // namespace torch_rbln::warmcache
