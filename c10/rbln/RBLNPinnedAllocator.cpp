#include <c10/rbln/RBLNFunctions.h>
#include <c10/rbln/RBLNLogging.h>
#include <c10/rbln/RBLNPinnedAllocator.h>

#include <rebel/runtime/api/rbln_runtime_api.h>

#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cstdlib>
#include <iterator>
#include <map>
#include <mutex>
#include <vector>

namespace c10::rbln {

namespace {

constexpr size_t kHugePage = size_t{2} << 20;

struct PinnedRange {
  size_t nbytes = 0;
  // Runtime (torch) device ids the range is registered with, ascending.
  std::vector<int> registered_devices;
  // Caller-owned memory registered through register_host_memory(): never freed here.
  bool external = false;
};

// Allocation base -> range; ordered so interior pointers match via upper_bound.
std::mutex pinned_registry_mutex;
std::map<uintptr_t, PinnedRange> pinned_registry;

// Locked lookup of the allocation containing `addr`; end() when there is none.
std::map<uintptr_t, PinnedRange>::iterator find_containing_locked(uintptr_t addr) {
  auto it = pinned_registry.upper_bound(addr);
  if (it == pinned_registry.begin()) {
    return pinned_registry.end();
  }
  --it;
  return addr < it->first + it->second.nbytes ? it : pinned_registry.end();
}

// Whether the runtime can register host memory for `torch_device_id` (cached per device).
bool host_register_supported(int torch_device_id) noexcept {
  static std::mutex mutex;
  static std::map<int, bool> cache;
  const std::lock_guard<std::mutex> guard(mutex);
  auto it = cache.find(torch_device_id);
  if (it != cache.end()) {
    return it->second;
  }
  bool supported = false;
  if (::rbln::rbln_host_register_supported(static_cast<uint32_t>(torch_device_id), supported) != RBLNRetCode_SUCCESS) {
    supported = false;
  }
  if (!supported) {
    RBLN_LOG_DEBUG(
        "Host-memory registration unavailable on torch_device_id={}; pinned buffers stay unregistered",
        torch_device_id);
  }
  cache.emplace(torch_device_id, supported);
  return supported;
}

// Register [base, base + nbytes) on `torch_device_id`; the range entry is updated only on success.
void register_range_on(uintptr_t base, size_t nbytes, int torch_device_id) noexcept {
  if (!host_register_supported(torch_device_id)) {
    return;
  }
  if (::rbln::rbln_host_register(static_cast<uint32_t>(torch_device_id), base, static_cast<uint64_t>(nbytes)) !=
      RBLNRetCode_SUCCESS) {
    RBLN_LOG_DEBUG(
        "rbln_host_register({:#x}, {} bytes) failed on torch_device_id={}; copies keep the per-command-buffer pin",
        base,
        nbytes,
        torch_device_id);
    return;
  }
  const std::lock_guard<std::mutex> guard(pinned_registry_mutex);
  auto it = pinned_registry.find(base);
  if (it == pinned_registry.end()) {
    // Freed while we were registering: undo, the deleter already ran without this device.
    ::rbln::rbln_host_unregister(static_cast<uint32_t>(torch_device_id), base);
    return;
  }
  auto& devices = it->second.registered_devices;
  devices.insert(std::upper_bound(devices.begin(), devices.end(), torch_device_id), torch_device_id);
  RBLN_LOG_DEBUG("Registered pinned host memory {:#x} ({} bytes) on torch_device_id={}", base, nbytes, torch_device_id);
}

// Unregister `range` from every device it was registered with (skipped once the runtime is gone).
void unregister_range_everywhere(void* data, const PinnedRange& range) noexcept {
  if (range.registered_devices.empty() || !rbln_runtime_available()) {
    return;
  }
  for (int torch_device_id : range.registered_devices) {
    if (::rbln::rbln_host_unregister(static_cast<uint32_t>(torch_device_id), reinterpret_cast<uintptr_t>(data)) !=
        RBLNRetCode_SUCCESS) {
      RBLN_LOG_DEBUG("rbln_host_unregister({}) failed on torch_device_id={}", fmt::ptr(data), torch_device_id);
    }
  }
}

// Register [base, base + nbytes) on every device this process has initialized.
void register_range_on_initialized_devices(uintptr_t base, size_t nbytes) noexcept {
  for (const auto device_index : initialized_device_indices()) {
    register_range_on(base, nbytes, static_cast<int>(device_index));
  }
}

void raw_pinned_delete(void* data) {
  if (data == nullptr) {
    return;
  }
  RBLN_LOG_DEBUG("Freeing pinned host memory at {}", fmt::ptr(data));
  PinnedRange range;
  {
    const std::lock_guard<std::mutex> guard(pinned_registry_mutex);
    auto it = pinned_registry.find(reinterpret_cast<uintptr_t>(data));
    if (it != pinned_registry.end()) {
      range = std::move(it->second);
      pinned_registry.erase(it);
    }
  }
  unregister_range_everywhere(data, range);
  if (range.nbytes > 0) {
    munlock(data, range.nbytes); // best-effort, mirrors the mlock in allocate
    if (range.nbytes >= kHugePage) {
      madvise(data, range.nbytes, MADV_NOHUGEPAGE);
    }
  }
  // Allocator deleters work on raw pointers from posix_memalign; RAII does not apply.
  std::free(data); // NOLINT(cppcoreguidelines-no-malloc)
}

struct RBLNPinnedAllocator final : public c10::Allocator {
  /**
   * @brief Allocates page-aligned host memory, page-locks it (best effort) and registers
   * it with the runtime on every initialized RBLN device.
   *
   * @param nbytes The number of bytes to allocate.
   * @return A data pointer to the pinned host memory (device is CPU).
   */
  c10::DataPtr allocate(size_t nbytes) override {
    void* data = nullptr;
    if (nbytes > 0) {
      static const size_t page_size = static_cast<size_t>(sysconf(_SC_PAGESIZE));
      // From 2 MiB up, align to and advise huge pages (fewer, larger pins).
      const bool huge = nbytes >= kHugePage;
      const size_t alignment = huge ? kHugePage : page_size;
      const size_t alloc_bytes = huge ? (nbytes + kHugePage - 1) / kHugePage * kHugePage : nbytes;
      if (posix_memalign(&data, alignment, alloc_bytes) != 0) {
        TORCH_CHECK(false, "Failed to allocate ", nbytes, " bytes of pinned host memory");
      }
      if (huge && madvise(data, alloc_bytes, MADV_HUGEPAGE) != 0) {
        RBLN_LOG_DEBUG("madvise(MADV_HUGEPAGE) of {} bytes failed (errno={}); continuing", alloc_bytes, errno);
      }
      // RLIMIT_MEMLOCK failure is non-fatal: stays semantically pinned.
      if (mlock(data, nbytes) != 0) {
        RBLN_LOG_DEBUG("mlock of {} bytes failed (errno={}); continuing unlocked", nbytes, errno);
      }
      RBLN_LOG_DEBUG("Allocated {} bytes of pinned host memory at {}", nbytes, fmt::ptr(data));
      {
        const std::lock_guard<std::mutex> guard(pinned_registry_mutex);
        pinned_registry.emplace(reinterpret_cast<uintptr_t>(data), PinnedRange{alloc_bytes, {}});
      }
      register_range_on_initialized_devices(reinterpret_cast<uintptr_t>(data), alloc_bytes);
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
  const std::lock_guard<std::mutex> guard(pinned_registry_mutex);
  return find_containing_locked(reinterpret_cast<uintptr_t>(data)) != pinned_registry.end();
}

void ensure_pinned_registered(const void* data, int torch_device_id) noexcept {
  if (data == nullptr) {
    return;
  }
  uintptr_t base = 0;
  size_t nbytes = 0;
  {
    const std::lock_guard<std::mutex> guard(pinned_registry_mutex);
    if (pinned_registry.empty()) {
      return;
    }
    auto it = find_containing_locked(reinterpret_cast<uintptr_t>(data));
    if (it == pinned_registry.end()) {
      return;
    }
    const auto& devices = it->second.registered_devices;
    if (std::binary_search(devices.begin(), devices.end(), torch_device_id)) {
      return;
    }
    base = it->first;
    nbytes = it->second.nbytes;
  }
  register_range_on(base, nbytes, torch_device_id);
}

void register_host_memory(void* data, size_t nbytes) {
  TORCH_CHECK(data != nullptr, "register_host_memory: data cannot be nullptr");
  TORCH_CHECK(nbytes > 0, "register_host_memory: nbytes must be positive");
  const auto base = reinterpret_cast<uintptr_t>(data);
  {
    const std::lock_guard<std::mutex> guard(pinned_registry_mutex);
    // Overlap with a live range is refused.
    auto next = pinned_registry.lower_bound(base);
    TORCH_CHECK(
        next == pinned_registry.end() || next->first >= base + nbytes,
        "register_host_memory: [",
        fmt::ptr(data),
        ", +",
        nbytes,
        ") overlaps a registered range");
    if (next != pinned_registry.begin()) {
      auto prev = std::prev(next);
      TORCH_CHECK(
          prev->first + prev->second.nbytes <= base,
          "register_host_memory: [",
          fmt::ptr(data),
          ", +",
          nbytes,
          ") overlaps a registered range");
    }
    pinned_registry.emplace(base, PinnedRange{nbytes, {}, /*external=*/true});
  }
  RBLN_LOG_DEBUG("Registered external host memory {} ({} bytes)", fmt::ptr(data), nbytes);
  register_range_on_initialized_devices(base, nbytes);
}

void unregister_host_memory(void* data) {
  TORCH_CHECK(data != nullptr, "unregister_host_memory: data cannot be nullptr");
  PinnedRange range;
  {
    const std::lock_guard<std::mutex> guard(pinned_registry_mutex);
    auto it = pinned_registry.find(reinterpret_cast<uintptr_t>(data));
    TORCH_CHECK(
        it != pinned_registry.end() && it->second.external,
        "unregister_host_memory: ",
        fmt::ptr(data),
        " is not the start of a range registered with register_host_memory");
    range = std::move(it->second);
    pinned_registry.erase(it);
  }
  unregister_range_everywhere(data, range);
  RBLN_LOG_DEBUG("Unregistered external host memory {}", fmt::ptr(data));
}

bool pinned_ptr_registered_on(const void* data, int torch_device_id) noexcept {
  if (data == nullptr) {
    return false;
  }
  const std::lock_guard<std::mutex> guard(pinned_registry_mutex);
  auto it = find_containing_locked(reinterpret_cast<uintptr_t>(data));
  if (it == pinned_registry.end()) {
    return false;
  }
  const auto& devices = it->second.registered_devices;
  return std::binary_search(devices.begin(), devices.end(), torch_device_id);
}

} // namespace c10::rbln
