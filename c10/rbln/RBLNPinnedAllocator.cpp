#include <c10/rbln/RBLNLogging.h>
#include <c10/rbln/RBLNPinnedAllocator.h>

#include <sys/mman.h>
#include <unistd.h>

#include <cstdlib>
#include <map>
#include <mutex>

namespace c10::rbln {

namespace {

// Allocation base -> size; ordered so interior pointers match via upper_bound.
std::mutex pinned_registry_mutex;
std::map<uintptr_t, size_t> pinned_registry;

void raw_pinned_delete(void* data) {
  if (data == nullptr) {
    return;
  }
  RBLN_LOG_DEBUG("Freeing pinned host memory at {}", fmt::ptr(data));
  size_t nbytes = 0;
  {
    const std::lock_guard<std::mutex> guard(pinned_registry_mutex);
    auto it = pinned_registry.find(reinterpret_cast<uintptr_t>(data));
    if (it != pinned_registry.end()) {
      nbytes = it->second;
      pinned_registry.erase(it);
    }
  }
  if (nbytes > 0) {
    munlock(data, nbytes); // best-effort, mirrors the mlock in allocate
  }
  std::free(data);
}

struct RBLNPinnedAllocator final : public c10::Allocator {
  /**
   * @brief Allocates page-aligned host memory and page-locks it (best effort).
   *
   * @param nbytes The number of bytes to allocate.
   * @return A data pointer to the pinned host memory (device is CPU).
   */
  c10::DataPtr allocate(size_t nbytes) override {
    void* data = nullptr;
    if (nbytes > 0) {
      static const size_t page_size = static_cast<size_t>(sysconf(_SC_PAGESIZE));
      if (posix_memalign(&data, page_size, nbytes) != 0) {
        TORCH_CHECK(false, "Failed to allocate ", nbytes, " bytes of pinned host memory");
      }
      // RLIMIT_MEMLOCK failure is non-fatal: stays semantically pinned.
      if (mlock(data, nbytes) != 0) {
        RBLN_LOG_DEBUG("mlock of {} bytes failed (errno={}); continuing unlocked", nbytes, errno);
      }
      RBLN_LOG_DEBUG("Allocated {} bytes of pinned host memory at {}", nbytes, fmt::ptr(data));
      const std::lock_guard<std::mutex> guard(pinned_registry_mutex);
      pinned_registry.emplace(reinterpret_cast<uintptr_t>(data), nbytes);
    }
    return c10::DataPtr(data, data, &raw_pinned_delete, c10::Device(c10::kCPU));
  }

  c10::DeleterFnPtr raw_deleter() const override {
    return &raw_pinned_delete;
  }

  void copy_data(void* dst_data, const void* src_data, size_t nbytes) const override {
    if (nbytes > 0) {
      std::memcpy(dst_data, src_data, nbytes);
    }
  }
};

} // namespace

c10::Allocator* get_pinned_memory_allocator() {
  static RBLNPinnedAllocator allocator;
  return &allocator;
}

bool is_pinned_ptr(const void* data) {
  if (data == nullptr) {
    return false;
  }
  const auto addr = reinterpret_cast<uintptr_t>(data);
  const std::lock_guard<std::mutex> guard(pinned_registry_mutex);
  auto it = pinned_registry.upper_bound(addr);
  if (it == pinned_registry.begin()) {
    return false;
  }
  --it;
  return addr < it->first + it->second;
}

} // namespace c10::rbln
