#include <c10/rbln/RBLNCachingAllocator.h>
#include <c10/rbln/RBLNFunctions.h>
#include <c10/rbln/RBLNHeld.h>
#include <c10/rbln/RBLNLogging.h>
#include <c10/rbln/RBLNRuntime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

namespace c10::rbln::caching {

namespace {

using c10::CachingAllocator::Stat;
using c10::CachingAllocator::StatType;
using c10::CachingDeviceAllocator::DeviceStats;

struct Block {
  uintptr_t segment = 0;
  uintptr_t ptr = 0;
  size_t size = 0;
  size_t requested = 0;
  bool small = false;
  bool allocated = false;
  c10::StreamId stream = 0;
  // Streams other than `stream` that used the block, and the points of them the block waits
  // for once freed.
  std::vector<c10::Stream> uses;
  std::vector<rt::Event> pending;
  Block* prev = nullptr;
  Block* next = nullptr;
  // How a program holds the allocation, if one does.
  std::shared_ptr<const held::Type> held;
};

// The live allocations programs hold, so that locating bytes looks for one only while there is.
std::atomic<int64_t> held_allocations{0};

// Taken while an allocation is put in a held type or back, so that no other thread locates the
// bytes of one halfway; the thread converting locates them as they are.
std::recursive_mutex& held_mutex() {
  static std::recursive_mutex mutex;
  return mutex;
}
thread_local bool converting = false;
thread_local bool as_held = false;

struct ByStreamSize {
  bool operator()(const Block* a, const Block* b) const {
    return std::tie(a->stream, a->size, a->ptr) < std::tie(b->stream, b->size, b->ptr);
  }
};

using Pool = std::set<Block*, ByStreamSize>;

void increase(c10::CachingAllocator::StatArray& stats, bool small, size_t amount) {
  stats[static_cast<size_t>(StatType::AGGREGATE)].increase(amount);
  stats[static_cast<size_t>(small ? StatType::SMALL_POOL : StatType::LARGE_POOL)].increase(amount);
}

void decrease(c10::CachingAllocator::StatArray& stats, bool small, size_t amount) {
  stats[static_cast<size_t>(StatType::AGGREGATE)].decrease(amount);
  stats[static_cast<size_t>(small ? StatType::SMALL_POOL : StatType::LARGE_POOL)].decrease(amount);
}

size_t rounded(size_t nbytes) {
  const size_t unit = nbytes <= kSmallSize ? kSmallRound : kLargeRound;
  return (std::max<size_t>(nbytes, 1) + unit - 1) / unit * unit;
}

class DeviceCache {
 public:
  explicit DeviceCache(c10::DeviceIndex device_index) : device_index_(device_index) {}

  void* allocate(size_t nbytes, c10::StreamId stream) {
    std::lock_guard<std::mutex> lock(mutex_);
    retire_pending();
    const size_t size = rounded(nbytes);
    const bool small = size <= kSmallSize;
    auto& pool = small ? small_ : large_;
    Block* block = take(pool, size, stream);
    if (block == nullptr) {
      block = new_segment(small ? kSmallSegment : size, small, stream);
    }
    split(pool, block, size);
    block->allocated = true;
    block->requested = nbytes;
    active_.emplace(block->ptr, block);
    increase(stats_.allocation, small, 1);
    increase(stats_.allocated_bytes, small, block->size);
    increase(stats_.active, small, 1);
    increase(stats_.active_bytes, small, block->size);
    increase(stats_.requested_bytes, small, nbytes);
    return reinterpret_cast<void*>(block->ptr);
  }

  struct Found {
    uintptr_t start = 0;
    uint64_t available = 0;
    std::shared_ptr<const held::Type> held;
  };

  // The live allocation `ptr` lies in: its first byte, the bytes from `ptr` to its end, and how a
  // program holds it. One past the end is the end of an empty view there.
  std::optional<Found> find(uintptr_t ptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = active_.upper_bound(ptr);
    if (it == active_.begin()) {
      return std::nullopt;
    }
    --it;
    const Block* block = it->second;
    if (ptr - block->ptr > block->requested) {
      return std::nullopt;
    }
    return Found{block->ptr, block->ptr + block->requested - ptr, block->held};
  }

  void set_held(uintptr_t start, std::shared_ptr<const held::Type> type) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = active_.find(start);
    RBLN_CHECK(it != active_.end(), "{} starts no live RBLN allocation", fmt::ptr(reinterpret_cast<void*>(start)));
    auto& held = it->second->held;
    held_allocations += static_cast<int64_t>(type != nullptr) - static_cast<int64_t>(held != nullptr);
    held = std::move(type);
  }

  void release(uintptr_t ptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = active_.find(ptr);
    RBLN_CHECK(it != active_.end(), "{} is no live RBLN allocation", fmt::ptr(reinterpret_cast<void*>(ptr)));
    Block* block = it->second;
    active_.erase(it);
    block->allocated = false;
    if (block->held) {
      block->held.reset();
      --held_allocations;
    }
    decrease(stats_.allocation, block->small, 1);
    decrease(stats_.allocated_bytes, block->small, block->size);
    decrease(stats_.requested_bytes, block->small, block->requested);
    if (!block->uses.empty()) {
      for (const auto& stream : block->uses) {
        block->pending.push_back(runtime_stream(stream)->record());
      }
      block->uses.clear();
      pending_.push_back(block);
      return;
    }
    free_block(block);
  }

  void record_stream(uintptr_t ptr, c10::Stream stream) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = active_.find(ptr);
    if (it == active_.end() || stream.id() == it->second->stream) {
      return;
    }
    auto& uses = it->second->uses;
    if (std::find(uses.begin(), uses.end(), stream) == uses.end()) {
      uses.push_back(stream);
    }
  }

  void empty_cache() {
    std::lock_guard<std::mutex> lock(mutex_);
    retire_pending();
    release_free_segments(small_);
    release_free_segments(large_);
  }

  DeviceStats stats() {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
  }

  void reset_peak() {
    std::lock_guard<std::mutex> lock(mutex_);
    for_each_stat([](Stat& stat) { stat.reset_peak(); });
  }

  void reset_accumulated() {
    std::lock_guard<std::mutex> lock(mutex_);
    for_each_stat([](Stat& stat) { stat.reset_accumulated(); });
    stats_.num_alloc_retries = 0;
    stats_.num_ooms = 0;
    stats_.num_device_alloc = 0;
    stats_.num_device_free = 0;
  }

 private:
  template <typename F>
  void for_each_stat(F f) {
    for (auto* stats : {&stats_.allocation, &stats_.segment, &stats_.active, &stats_.inactive_split,
                        &stats_.allocated_bytes, &stats_.reserved_bytes, &stats_.active_bytes,
                        &stats_.inactive_split_bytes, &stats_.requested_bytes}) {
      for (auto& stat : *stats) {
        f(stat);
      }
    }
  }

  // The smallest free block of `stream` that holds `size` bytes; `split` returns what the
  // request leaves of it to the pool.
  Block* take(Pool& pool, size_t size, c10::StreamId stream) {
    Block key;
    key.stream = stream;
    key.size = size;
    auto it = pool.lower_bound(&key);
    if (it == pool.end() || (*it)->stream != stream) {
      return nullptr;
    }
    Block* block = *it;
    pool.erase(it);
    if (block->prev != nullptr || block->next != nullptr) {
      decrease(stats_.inactive_split, block->small, 1);
      decrease(stats_.inactive_split_bytes, block->small, block->size);
    }
    return block;
  }

  Block* new_segment(size_t size, bool small, c10::StreamId stream) {
    void* handle = nullptr;
    try {
      handle = allocate_segment(device_index_, size);
    } catch (const std::exception&) {
      // Out of device memory: give back what is cached and try once more.
      stats_.num_alloc_retries++;
      retire_pending();
      release_free_segments(small_);
      release_free_segments(large_);
      try {
        handle = allocate_segment(device_index_, size);
      } catch (const std::exception& e) {
        stats_.num_ooms++;
        RBLN_CHECK(
            false,
            "rbln:{} is out of memory for {} bytes ({} bytes allocated, {} reserved): {}",
            static_cast<int>(device_index_),
            size,
            stats_.allocated_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
            stats_.reserved_bytes[static_cast<size_t>(StatType::AGGREGATE)].current,
            e.what());
      }
    }
    auto* block = new Block;
    block->segment = reinterpret_cast<uintptr_t>(handle);
    block->ptr = block->segment;
    block->size = size;
    block->small = small;
    block->stream = stream;
    increase(stats_.segment, small, 1);
    increase(stats_.reserved_bytes, small, size);
    stats_.num_device_alloc++;
    return block;
  }

  // Leaves `block` with `size` bytes and returns the rest to `pool`.
  void split(Pool& pool, Block* block, size_t size) {
    const size_t rest = block->size - size;
    if (rest < (block->small ? kSmallRound : kLargeRound)) {
      return;
    }
    auto* remainder = new Block;
    remainder->segment = block->segment;
    remainder->ptr = block->ptr + size;
    remainder->size = rest;
    remainder->small = block->small;
    remainder->stream = block->stream;
    remainder->prev = block;
    remainder->next = block->next;
    if (block->next != nullptr) {
      block->next->prev = remainder;
    }
    block->next = remainder;
    block->size = size;
    pool.insert(remainder);
    increase(stats_.inactive_split, remainder->small, 1);
    increase(stats_.inactive_split_bytes, remainder->small, rest);
  }

  bool mergeable(const Block* block) const {
    return block != nullptr && !block->allocated && block->pending.empty();
  }

  // Returns a block no stream uses any more to its pool, merged with free neighbors.
  void free_block(Block* block) {
    decrease(stats_.active, block->small, 1);
    decrease(stats_.active_bytes, block->small, block->size);
    auto& pool = block->small ? small_ : large_;
    for (Block* neighbor : {block->prev, block->next}) {
      if (!mergeable(neighbor)) {
        continue;
      }
      pool.erase(neighbor);
      if (neighbor->prev != nullptr || neighbor->next != nullptr) {
        decrease(stats_.inactive_split, neighbor->small, 1);
        decrease(stats_.inactive_split_bytes, neighbor->small, neighbor->size);
      }
      if (neighbor == block->prev) {
        block->ptr = neighbor->ptr;
        block->prev = neighbor->prev;
        if (block->prev != nullptr) {
          block->prev->next = block;
        }
      } else {
        block->next = neighbor->next;
        if (block->next != nullptr) {
          block->next->prev = block;
        }
      }
      block->size += neighbor->size;
      delete neighbor;
    }
    pool.insert(block);
    if (block->prev != nullptr || block->next != nullptr) {
      increase(stats_.inactive_split, block->small, 1);
      increase(stats_.inactive_split_bytes, block->small, block->size);
    }
  }

  void retire_pending() {
    auto done = [](Block* block) {
      return std::all_of(
          block->pending.begin(), block->pending.end(), [](const rt::Event& event) { return event.query(); });
    };
    std::vector<Block*> still;
    for (Block* block : pending_) {
      if (done(block)) {
        block->pending.clear();
        free_block(block);
      } else {
        still.push_back(block);
      }
    }
    pending_ = std::move(still);
  }

  void release_free_segments(Pool& pool) {
    for (auto it = pool.begin(); it != pool.end();) {
      Block* block = *it;
      if (block->prev != nullptr || block->next != nullptr) {
        ++it;
        continue;
      }
      it = pool.erase(it);
      release_segment(reinterpret_cast<void*>(block->segment));
      decrease(stats_.segment, block->small, 1);
      decrease(stats_.reserved_bytes, block->small, block->size);
      stats_.num_device_free++;
      delete block;
    }
  }

  c10::DeviceIndex device_index_;
  std::mutex mutex_;
  Pool small_;
  Pool large_;
  std::map<uintptr_t, Block*> active_;
  std::vector<Block*> pending_;
  DeviceStats stats_;
};

std::mutex caches_mutex;
std::map<c10::DeviceIndex, std::unique_ptr<DeviceCache>> caches;

DeviceCache& cache(c10::DeviceIndex device_index) {
  std::lock_guard<std::mutex> lock(caches_mutex);
  auto& entry = caches[device_index];
  if (!entry) {
    entry = std::make_unique<DeviceCache>(device_index);
  }
  return *entry;
}

DeviceCache* cache_of(const void* ptr) {
  const auto location = locate_segment(ptr);
  return location ? &cache(location->device_index) : nullptr;
}

} // namespace

void* allocate(c10::DeviceIndex device_index, size_t nbytes) {
  return cache(device_index).allocate(nbytes, get_current_stream(device_index).id());
}

void release(void* ptr) {
  DeviceCache* owner = cache_of(ptr);
  RBLN_CHECK(owner != nullptr, "{} is not RBLN device memory, or it was freed", fmt::ptr(ptr));
  owner->release(reinterpret_cast<uintptr_t>(ptr));
}

void record_stream(void* ptr, c10::Stream stream) {
  if (DeviceCache* owner = cache_of(ptr)) {
    owner->record_stream(reinterpret_cast<uintptr_t>(ptr), stream);
  }
}

std::optional<Held> try_locate_held(const void* ptr) {
  auto location = locate_segment(ptr);
  if (!location) {
    return std::nullopt;
  }
  auto found = cache(location->device_index).find(reinterpret_cast<uintptr_t>(ptr));
  if (!found) {
    return std::nullopt;
  }
  location->available = found->available;
  return Held{std::move(*location), reinterpret_cast<void*>(found->start), std::move(found->held)};
}

void convert_held(const void* start, const std::function<void()>& convert, std::shared_ptr<const held::Type> type) {
  std::lock_guard<std::recursive_mutex> lock(held_mutex());
  auto location = locate_segment(start);
  RBLN_CHECK(location.has_value(), "{} is not RBLN device memory", fmt::ptr(start));
  converting = true;
  try {
    convert();
  } catch (...) {
    converting = false;
    throw;
  }
  converting = false;
  cache(location->device_index).set_held(reinterpret_cast<uintptr_t>(start), std::move(type));
}

bool any_held() {
  return held_allocations.load() != 0;
}

bool locate_as_held(bool value) {
  return std::exchange(as_held, value);
}

bool locating_as_held() {
  return as_held;
}

std::optional<Location> try_locate(const void* ptr) {
  if (held_allocations.load() == 0 || converting || as_held) {
    auto found = try_locate_held(ptr);
    return found ? std::optional<Location>(std::move(found->location)) : std::nullopt;
  }
  std::lock_guard<std::recursive_mutex> lock(held_mutex());
  auto found = try_locate_held(ptr);
  if (!found) {
    return std::nullopt;
  }
  if (found->type) {
    convert_held(found->start, [&] { held::release(found->start, *found->type); }, nullptr);
  }
  return std::move(found->location);
}

Location locate(const void* ptr) {
  auto location = try_locate(ptr);
  RBLN_CHECK(location.has_value(), "{} is not in live RBLN device memory", fmt::ptr(ptr));
  return *location;
}

void empty_cache(c10::DeviceIndex device_index) {
  cache(device_index).empty_cache();
}

DeviceStats device_stats(c10::DeviceIndex device_index) {
  return cache(device_index).stats();
}

std::map<std::string, uint64_t> stats_map(c10::DeviceIndex device_index) {
  const auto stats = device_stats(device_index);
  std::map<std::string, uint64_t> out;
  auto put = [&](const std::string& name, const c10::CachingAllocator::StatArray& stat) {
    const std::pair<const char*, StatType> pools[] = {
        {"all", StatType::AGGREGATE}, {"small_pool", StatType::SMALL_POOL}, {"large_pool", StatType::LARGE_POOL}};
    for (const auto& [pool, type] : pools) {
      const auto& s = stat[static_cast<size_t>(type)];
      const auto prefix = name + "." + pool + ".";
      out[prefix + "current"] = static_cast<uint64_t>(s.current);
      out[prefix + "peak"] = static_cast<uint64_t>(s.peak);
      out[prefix + "allocated"] = static_cast<uint64_t>(s.allocated);
      out[prefix + "freed"] = static_cast<uint64_t>(s.freed);
    }
  };
  put("allocation", stats.allocation);
  put("segment", stats.segment);
  put("active", stats.active);
  put("inactive_split", stats.inactive_split);
  put("allocated_bytes", stats.allocated_bytes);
  put("reserved_bytes", stats.reserved_bytes);
  put("active_bytes", stats.active_bytes);
  put("inactive_split_bytes", stats.inactive_split_bytes);
  put("requested_bytes", stats.requested_bytes);
  out["num_alloc_retries"] = static_cast<uint64_t>(stats.num_alloc_retries);
  out["num_ooms"] = static_cast<uint64_t>(stats.num_ooms);
  out["num_device_alloc"] = static_cast<uint64_t>(stats.num_device_alloc);
  out["num_device_free"] = static_cast<uint64_t>(stats.num_device_free);
  return out;
}

void reset_peak(c10::DeviceIndex device_index) {
  cache(device_index).reset_peak();
}

void reset_accumulated(c10::DeviceIndex device_index) {
  cache(device_index).reset_accumulated();
}

} // namespace c10::rbln::caching
