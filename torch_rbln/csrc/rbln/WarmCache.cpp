#include <torch_rbln/csrc/rbln/WarmCache.h>

#include <algorithm>
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
  // Leaky: entries hold runtime objects, which must not outlive the runtime's
  // own statics at exit.
  static auto* c = new WarmCache();
  return *c;
}

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
  // First-writer-wins: if another thread beat us to it, keep the earlier one.
  map_.try_emplace(std::move(key), std::make_shared<CacheEntry>(entry));
}

size_t WarmCache::size() {
  std::shared_lock<std::shared_mutex> rd(mu_);
  return map_.size();
}

void WarmCache::clear(std::optional<c10::DeviceIndex> device) {
  // The dropped entries die at the end of this function, outside the lock:
  // destroying an executor waits for its runs.
  std::vector<std::shared_ptr<CacheEntry>> dropped;
  {
    std::unique_lock<std::shared_mutex> wr(mu_);
    for (auto it = map_.begin(); it != map_.end();) {
      const auto& inputs = it->first.inputs;
      const bool drop = !device.has_value() ||
          std::ranges::any_of(inputs, [&](const TensorProfile& tp) { return tp.device_index == *device; });
      if (drop) {
        dropped.push_back(std::move(it->second));
        it = map_.erase(it);
      } else {
        ++it;
      }
    }
  }
}

} // namespace torch_rbln::warmcache
