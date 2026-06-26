#include <ATen/ATen.h>
#include <c10/rbln/DeviceMappingManager.h>
#include <c10/rbln/RBLNFunctions.h>
#include <c10/rbln/RBLNLogging.h>
#include <c10/util/CallOnce.h>
#include <rebel/runtime/memory_stats.h>

#include <atomic>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <vector>

namespace c10::rbln {

namespace {

// Default current logical device is 0
thread_local c10::DeviceIndex current_device_index_ = 0;

// Dummy mode only: tracks each host-backed allocation [base, base+nbytes) and
// its logical device, so get_torch_device_id() can resolve an interior/view
// pointer to its owning device (matching the real vaddr lookup) and free() can
// reject stale/double frees. Guarded by is_dummy_device(); never touched on the
// real-device path. Ordered map enables the range lookup.
struct DummyAlloc {
  size_t nbytes;
  c10::DeviceIndex device;
};
std::mutex dummy_alloc_mutex_;
std::map<uintptr_t, DummyAlloc> dummy_allocs_;

void dummy_register_alloc(const void* data, size_t nbytes, c10::DeviceIndex device_index) {
  const std::lock_guard<std::mutex> lock(dummy_alloc_mutex_);
  dummy_allocs_[reinterpret_cast<uintptr_t>(data)] = DummyAlloc{nbytes, device_index};
}

// Erase by base pointer; returns false if it was not a live allocation
// (unknown / already freed).
bool dummy_take_alloc(const void* data) {
  const std::lock_guard<std::mutex> lock(dummy_alloc_mutex_);
  return dummy_allocs_.erase(reinterpret_cast<uintptr_t>(data)) > 0;
}

// Resolve any address within a live allocation to its owning logical device;
// throws on a stale / non-device pointer (parity with the real vaddr lookup).
c10::DeviceIndex dummy_lookup_device(const void* data) {
  const auto addr = reinterpret_cast<uintptr_t>(data);
  const std::lock_guard<std::mutex> lock(dummy_alloc_mutex_);
  auto it = dummy_allocs_.upper_bound(addr); // first base > addr
  if (it != dummy_allocs_.begin()) {
    --it; // greatest base <= addr
    if (addr < it->first + it->second.nbytes) {
      return it->second.device;
    }
  }
  RBLN_CHECK(
      false,
      "get_torch_device_id: {:#x} is not within any RBLN_DUMMY_DEVICE allocation (stale pointer or not device memory)",
      addr);
}

// True iff [data, data+nbytes) lies fully within a single live allocation.
bool dummy_range_ok(const void* data, size_t nbytes) {
  const auto addr = reinterpret_cast<uintptr_t>(data);
  const std::lock_guard<std::mutex> lock(dummy_alloc_mutex_);
  auto it = dummy_allocs_.upper_bound(addr);
  if (it == dummy_allocs_.begin()) {
    return false;
  }
  --it; // greatest base <= addr
  const auto end = it->first + it->second.nbytes;
  return addr < end && nbytes <= end - addr;
}

// Throwing device-pointer bounds check for dummy copy/borrow paths — keeps the
// host backing from doing OOB reads/writes that the real runtime would reject.
void dummy_check_range(const void* data, size_t nbytes) {
  RBLN_CHECK(
      dummy_range_ok(data, nbytes),
      "RBLN_DUMMY_DEVICE: device pointer {} + {} bytes is not within a live allocation (stale or out of bounds)",
      fmt::ptr(data),
      nbytes);
}

// Synthetic non-zero borrow ids for dummy mode: satisfies the BorrowedHostPtr
// contract (a successful borrow returns a non-zero id). return_borrowed() no-ops
// in dummy mode, so the value only needs to be non-zero.
std::atomic<uint64_t> dummy_borrow_counter_{0};
uint64_t next_dummy_borrow_id() {
  return dummy_borrow_counter_.fetch_add(1, std::memory_order_relaxed) + 1;
}

void check_device_index(c10::DeviceIndex device_index) {
  // Dropped a dead `<= max()` check (always true for an int8 DeviceIndex); an
  // invalid/negative index is caught by the OOB-safe isDeviceAssigned() lookup.
  auto& manager = DeviceMappingManager::getInstance();
  // No logical devices: selecting one is pure bookkeeping (nothing to validate);
  // actual device use still fails at the point of use.
  if (manager.getLogicalDeviceCount() == 0) {
    return;
  }
  if (!manager.isDeviceAssigned(device_index)) {
    const auto env_display = getRblnNpuMappingEnvDisplay();
    RBLN_CHECK(
        false,
        "Logical device rbln: {} is not assigned (this process has {} logical device(s)). Env RBLN_DEVICE_MAP={}, RBLN_NPUS_PER_DEVICE={}.",
        static_cast<int>(device_index),
        static_cast<int>(manager.getLogicalDeviceCount()),
        env_display.device_map,
        env_display.npus_per_device);
  }
}

std::string to_string(::rbln::DataType rbln_dtype) {
  switch (rbln_dtype) {
    case ::rbln::DataType::Undefined:
      return "Undefined";
    case ::rbln::DataType::UInt8:
      return "UInt8";
    case ::rbln::DataType::Int8:
      return "Int8";
    case ::rbln::DataType::Int16:
      return "Int16";
    case ::rbln::DataType::Int32:
      return "Int32";
    case ::rbln::DataType::Int64:
      return "Int64";
    case ::rbln::DataType::Float16:
      return "Float16";
    case ::rbln::DataType::Float32:
      return "Float32";
    case ::rbln::DataType::Float64:
      return "Float64";
    case ::rbln::DataType::Complex32:
      return "Complex32";
    case ::rbln::DataType::Complex64:
      return "Complex64";
    case ::rbln::DataType::Complex128:
      return "Complex128";
    case ::rbln::DataType::Bool:
      return "Bool";
    case ::rbln::DataType::BFloat16:
      return "BFloat16";
    case ::rbln::DataType::Float8_e5m2:
      return "Float8_e5m2";
    case ::rbln::DataType::Float8_e4m3:
      return "Float8_e4m3";
    case ::rbln::DataType::CustomFloat16:
      return "CustomFloat16";
    default:
      RBLN_CHECK(false, "Unsupported RBLN dtype: {}", static_cast<int>(rbln_dtype));
  }
}

int to_device_id(c10::DeviceIndex device_index) {
  // Shared precursor to every device-touching runtime call (alloc, synchronize,
  // memory stats, ...). With no NPU, fail here with one clear message before
  // reaching the runtime, which may not handle an unregistered device.
  RBLN_CHECK(
      DeviceMappingManager::getInstance().getLogicalDeviceCount() > 0,
      "Cannot use rbln:{}: no logical device available (this process sees 0 RBLN device(s)). "
      "If this host has no NPU, set RBLN_DUMMY_DEVICE=1 for host-backed tensors/compilation "
      "(execution still needs hardware); otherwise check RBLN_DEVICES and that the NPU driver is available.",
      static_cast<int>(device_index));
  // Cast directly — do NOT round through `unsigned char`, which would alias a
  // stray negative index to a real id (e.g. -1 -> 255).
  RBLN_CHECK(
      device_index >= 0,
      "Internal error: negative logical device index ({}) reached to_device_id",
      static_cast<int>(device_index));
  return static_cast<int>(device_index);
}

} // namespace

::rbln::DataType to_rbln_dtype(c10::ScalarType dtype) {
  switch (dtype) {
    case c10::kByte:
      return ::rbln::DataType::UInt8;
    case c10::kChar:
      return ::rbln::DataType::Int8;
    case c10::kShort:
      return ::rbln::DataType::Int16;
    case c10::kInt:
      return ::rbln::DataType::Int32;
    case c10::kLong:
      return ::rbln::DataType::Int64;
    case c10::kHalf:
      return ::rbln::DataType::Float16;
    case c10::kFloat:
      return ::rbln::DataType::Float32;
    case c10::kDouble:
      return ::rbln::DataType::Float64;
    case c10::kComplexHalf:
      return ::rbln::DataType::Complex32;
    case c10::kComplexFloat:
      return ::rbln::DataType::Complex64;
    case c10::kComplexDouble:
      return ::rbln::DataType::Complex128;
    case c10::kBool:
      return ::rbln::DataType::Bool;
    case c10::kBFloat16:
      return ::rbln::DataType::BFloat16;
    case c10::kFloat8_e5m2:
      return ::rbln::DataType::Float8_e5m2;
    case c10::kFloat8_e4m3fn:
      return ::rbln::DataType::Float8_e4m3;
    default:
      RBLN_CHECK(false, "Unsupported dtype: {}", c10::str(dtype));
  }
}

std::string to_string(const ::rbln::MemoryInfo& memory_info) {
  const auto memory_info_string = fmt::format(
      "MemoryInfo(torch_device_id={}, key_vaddr={:#x}, user_dtype={}, user_shape={}, physical_dtype={}, physical_shape={})",
      memory_info.torch_device_id,
      memory_info.key_vaddr,
      to_string(memory_info.user_dtype),
      fmt::join(memory_info.user_shape, ","),
      to_string(memory_info.physical_dtype),
      fmt::join(memory_info.physical_shape, ","));
  return memory_info_string;
}

c10::DeviceIndex get_device_count() {
  auto& manager = DeviceMappingManager::getInstance();
  const auto device_count = manager.getLogicalDeviceCount();
  RBLN_LOG_DEBUG("logical_device_count={}", static_cast<int>(device_count));
  return device_count;
}

c10::DeviceIndex get_physical_device_count() {
  if (is_dummy_device()) {
    // Dummy mode must not touch the runtime (the host may have no SDK/driver);
    // there is no physical NPU, so report 0.
    return 0;
  }
  // Directly query the runtime API for physical device count
  // This bypasses the RSD mode logic and always returns the actual physical count
  int device_count = 0;
  RBLN_CHECK(
      !rbln_get_device_count(&device_count),
      "rbln_get_device_count failed; no NPU is visible — check that the device is present, the rbln kernel "
      "driver is loaded, and the device is not held by another process");

  const auto physical_device_count = static_cast<c10::DeviceIndex>(device_count);
  RBLN_LOG_DEBUG("physical_NPU_count={}", static_cast<int>(physical_device_count));
  return physical_device_count;
}

c10::DeviceIndex get_device_index() {
  RBLN_LOG_DEBUG("current logical device=rbln:{}", static_cast<int>(current_device_index_));
  return current_device_index_;
}

void set_device_index(c10::DeviceIndex device_index) {
  RBLN_LOG_DEBUG("logical device=rbln:{}", static_cast<int>(device_index));
  // A negative index is the "keep current device" sentinel (CUDA convention;
  // Python maps device=None to it): intentional no-op. Only >= 0 is validated.
  if (device_index >= 0) {
    RBLN_LOG_DEBUG(
        "Setting current logical device: rbln:{} -> rbln:{}",
        static_cast<int>(current_device_index_),
        static_cast<int>(device_index));
    check_device_index(device_index);
    current_device_index_ = device_index;
  }
}

c10::DeviceIndex exchange_device_index(c10::DeviceIndex device_index) {
  const auto original_device_index = get_device_index();
  RBLN_LOG_DEBUG(
      "Setting current logical device: rbln:{} -> rbln:{}",
      static_cast<int>(original_device_index),
      static_cast<int>(device_index));

  if (device_index != original_device_index) {
    set_device_index(device_index);
  } else if (device_index >= 0) {
    // Same as set_device_index: validate mapping when the index is unchanged (see DeviceGuard).
    check_device_index(device_index);
  }

  return original_device_index;
}

c10::DeviceIndex get_torch_device_id(const void* data) {
  RBLN_LOG_DEBUG("data={}", fmt::ptr(data));
  RBLN_CHECK(data != nullptr, "data cannot be nullptr");

  if (is_dummy_device()) {
    // Range lookup so interior/view pointers resolve to the owning device.
    return dummy_lookup_device(data);
  }

  const auto vaddr = reinterpret_cast<uint64_t>(data);
  uint32_t torch_device_id = 0;
  RBLN_CHECK(
      !::rbln::rbln_get_torch_device_id_from_vaddr(vaddr, torch_device_id),
      "rbln_get_torch_device_id_from_vaddr failed for vaddr={:#x}; the pointer may not be RBLN device memory or may "
      "be stale",
      vaddr);
  return static_cast<c10::DeviceIndex>(torch_device_id);
}

::rbln::MemoryInfo get_memory_info(const void* data) {
  RBLN_LOG_DEBUG("data={}", fmt::ptr(data));
  RBLN_CHECK(data != nullptr, "data cannot be nullptr");
  // Backed by runtime v-memory only; the dummy host backing has no such entry.
  RBLN_CHECK(
      !is_dummy_device(), "get_memory_info is not available in RBLN_DUMMY_DEVICE mode (no runtime-backed v-memory)");
  // get_memory_info() does a full VMemory JSON serialize+parse round-trip on
  // every call. Warn so unexpected hot-path callers are easy to spot in logs;
  // when only the device id is needed, get_torch_device_id() is the cheap path.
  RBLN_LOG_WARN(
      "get_memory_info({}) performs a full VMemory JSON round-trip (slow) — avoid on performance hot paths; "
      "use get_torch_device_id() when only the device id is needed",
      fmt::ptr(data));

  const auto vaddr = reinterpret_cast<uint64_t>(data);
  ::rbln::MemoryInfo memory_info;
  RBLN_LOG_DEBUG("Calling rbln_get_memory_info: vaddr={:#x}", vaddr);
  RBLN_CHECK(
      !::rbln::rbln_get_memory_info(vaddr, memory_info),
      "rbln_get_memory_info failed for vaddr={:#x}; the pointer may not be RBLN device memory or may be stale",
      vaddr);
  RBLN_LOG_DEBUG("memory_info={}", to_string(memory_info));
  return memory_info;
}

bool is_eager_malloc() {
  static const bool eager_malloc = []() {
    const auto* env = std::getenv("TORCH_RBLN_EAGER_MALLOC");
    return (env != nullptr) ? (std::string(env) == "1") : false;
  }();
  RBLN_LOG_DEBUG("eager_malloc={}", eager_malloc);
  return eager_malloc;
}

bool is_dummy_device() {
  // Cheap, total, runtime-free: dummy mode is requested iff RBLN_DUMMY_DEVICE is
  // a positive integer. Deliberately does NOT go through DeviceMappingManager —
  // its init queries the runtime on a non-dummy host and would throw without a
  // driver, which must not happen for a plain predicate. Cached on first call.
  static const bool dummy = []() {
    const char* env = std::getenv("RBLN_DUMMY_DEVICE");
    if (env == nullptr || env[0] == '\0') {
      return false;
    }
    char* end = nullptr;
    const long value = std::strtol(env, &end, 10);
    if (end == env) {
      return false; // no leading integer
    }
    while (std::isspace(static_cast<unsigned char>(*end)) != 0) {
      ++end; // tolerate trailing whitespace (matches parseEnvInt)
    }
    return *end == '\0' && value > 0;
  }();
  return dummy;
}

void* malloc(c10::DeviceIndex device_index, size_t nbytes) {
  RBLN_LOG_DEBUG("logical device=rbln:{}, nbytes={}", static_cast<int>(device_index), nbytes);
  RBLN_CHECK(nbytes > 0, "nbytes must be positive, but got {}", nbytes);
  check_device_index(device_index);

  if (is_dummy_device()) {
    // No NPU: back the device tensor with host memory so the copies below are
    // plain memmove and tensors can be built/compiled without hardware. Record
    // the owning device so get_torch_device_id() can resolve it later.
    void* data = std::malloc(nbytes); // NOLINT(cppcoreguidelines-no-malloc)
    RBLN_CHECK(data != nullptr, "dummy host allocation failed ({} bytes)", nbytes);
    dummy_register_alloc(data, nbytes, device_index);
    return data;
  }

  // to_device_id() enforces that a device is actually available (device_count >
  // 0), so allocation fails cleanly here on a host with no NPU.
  const auto torch_device_id = static_cast<uint32_t>(to_device_id(device_index));
  const auto size = static_cast<uint64_t>(nbytes);
  uint64_t vaddr = 0;
  const auto eager_malloc = is_eager_malloc();
  if (eager_malloc) {
    RBLN_LOG_DEBUG(
        "Calling rbln_malloc_eager: rbln:{}, torch_device_id={}, size={}",
        static_cast<int>(device_index),
        torch_device_id,
        size);
    RBLN_CHECK(
        !::rbln::rbln_malloc_eager(torch_device_id, size, vaddr),
        "rbln_malloc_eager failed (rbln:{}, {} bytes); the device may be out of memory or hold stale allocations",
        static_cast<int>(device_index),
        size);
  } else {
    RBLN_LOG_DEBUG(
        "Calling rbln_malloc_lazy: rbln:{}, torch_device_id={}, size={}",
        static_cast<int>(device_index),
        torch_device_id,
        size);
    RBLN_CHECK(
        !::rbln::rbln_malloc_lazy(torch_device_id, size, vaddr),
        "rbln_malloc_lazy failed (rbln:{}, {} bytes); the device may be out of memory or hold stale allocations",
        static_cast<int>(device_index),
        size);
  }

  auto* data = reinterpret_cast<void*>(vaddr); // NOLINT(performance-no-int-to-ptr)
  RBLN_LOG_DEBUG("data={}", fmt::ptr(data));
  RBLN_CHECK(data != nullptr, "data cannot be nullptr");
  return data;
}

void mark_zeros(const void* rbln_data) {
  RBLN_LOG_DEBUG("rbln_data={}", fmt::ptr(rbln_data));
  RBLN_CHECK(rbln_data != nullptr, "rbln_data cannot be nullptr");

  if (is_dummy_device()) {
    // No size here to memset; the factory layer (zero_/_efficientzerotensor)
    // does the host memset in dummy mode.
    return;
  }

  const auto vaddr = reinterpret_cast<uint64_t>(rbln_data);
  RBLN_CHECK(!::rbln::rbln_mark_zeros(vaddr), "rbln_mark_zeros failed for vaddr={:#x}", vaddr);
  RBLN_LOG_DEBUG("vaddr={:#x} marked as zero-initialized", vaddr);
}

void free(void* data) {
  RBLN_LOG_DEBUG("data={}", fmt::ptr(data));
  RBLN_CHECK(data != nullptr, "data cannot be nullptr");

  if (is_dummy_device()) {
    // Reject stale/double frees before std::free (which would abort) — parity
    // with rbln_free's bad-address error.
    RBLN_CHECK(
        dummy_take_alloc(data),
        "dummy free: {} is not a live allocation (stale pointer or double free)",
        fmt::ptr(data));
    std::free(data); // NOLINT(cppcoreguidelines-no-malloc)
    return;
  }

  const auto vaddr = reinterpret_cast<uint64_t>(data);
  RBLN_LOG_DEBUG("Calling rbln_free: vaddr={:#x}", vaddr);
  RBLN_CHECK(
      !::rbln::rbln_free(vaddr),
      "rbln_free failed for vaddr={:#x} (device may have been reset/lost, or the address was already freed)",
      vaddr);
}

void free_nothrow(void* data) noexcept {
  // Non-throwing free for the noexcept DataPtr deleter: rbln_free is extern "C"
  // and RBLN_WARN_NOTHROW is itself nothrow, so no try/catch is needed.
  if (data == nullptr) {
    return;
  }
  if (is_dummy_device()) {
    if (dummy_take_alloc(data)) {
      std::free(data); // NOLINT(cppcoreguidelines-no-malloc)
    } else {
      RBLN_WARN_NOTHROW("dummy free: {} is not a live allocation; leaking rather than aborting", fmt::ptr(data));
    }
    return;
  }
  const auto vaddr = reinterpret_cast<uint64_t>(data);
  if (::rbln::rbln_free(vaddr) != 0) {
    RBLN_WARN_NOTHROW("rbln_free failed for vaddr={:#x}; leaking rather than aborting", vaddr);
  }
}

void set_device_layout_like(void* target_data, const void* ref_data) {
  RBLN_CHECK(target_data != nullptr, "set_device_layout_like: target is nullptr");
  RBLN_CHECK(ref_data != nullptr, "set_device_layout_like: ref is nullptr");
  const auto target_vaddr = reinterpret_cast<uint64_t>(target_data);
  const auto ref_vaddr = reinterpret_cast<uint64_t>(ref_data);
  RBLN_LOG_DEBUG("set_device_layout_like: target={:#x} ref={:#x}", target_vaddr, ref_vaddr);
  RBLN_CHECK(
      !::rbln::rbln_set_device_alloc_layout_like(target_vaddr, ref_vaddr), "rbln_set_device_alloc_layout_like failed");
}

void memcpy_h2v(void* rbln_dst_data, const void* cpu_src_data, size_t nbytes) {
  RBLN_LOG_DEBUG(
      "dst_rbln_data={}, src_cpu_data={}, nbytes={}", fmt::ptr(rbln_dst_data), fmt::ptr(cpu_src_data), nbytes);
  RBLN_CHECK(nbytes > 0, "nbytes must be positive, but got {}", nbytes);
  RBLN_CHECK(cpu_src_data != nullptr, "cpu_src_data cannot be nullptr");
  RBLN_CHECK(rbln_dst_data != nullptr, "rbln_dst_data cannot be nullptr");

  if (is_dummy_device()) {
    dummy_check_range(rbln_dst_data, nbytes);
    std::memmove(rbln_dst_data, cpu_src_data, nbytes);
    return;
  }

  const auto src_host_ptr = reinterpret_cast<uintptr_t>(cpu_src_data);
  const auto dst_vaddr = reinterpret_cast<uint64_t>(rbln_dst_data);
  const auto size = static_cast<uint64_t>(nbytes);
  RBLN_LOG_DEBUG(
      "Calling rbln_memcpy_h2v: src_host_ptr={:#x}, dst_vaddr={:#x}, size={}", src_host_ptr, dst_vaddr, size);
  RBLN_CHECK(
      !::rbln::rbln_memcpy_h2v(src_host_ptr, dst_vaddr, size),
      "rbln_memcpy_h2v failed ({} bytes, dst vaddr={:#x}); the device may be busy or faulted",
      size,
      dst_vaddr);
}

void memcpy_v2h(void* cpu_dst_data, const void* rbln_src_data, size_t nbytes) {
  RBLN_LOG_DEBUG(
      "dst_cpu_data={}, src_rbln_data={}, nbytes={}", fmt::ptr(cpu_dst_data), fmt::ptr(rbln_src_data), nbytes);
  RBLN_CHECK(nbytes > 0, "nbytes must be positive, but got {}", nbytes);
  RBLN_CHECK(rbln_src_data != nullptr, "rbln_src_data cannot be nullptr");
  RBLN_CHECK(cpu_dst_data != nullptr, "cpu_dst_data cannot be nullptr");

  if (is_dummy_device()) {
    dummy_check_range(rbln_src_data, nbytes);
    std::memmove(cpu_dst_data, rbln_src_data, nbytes);
    return;
  }

  const auto src_vaddr = reinterpret_cast<uint64_t>(rbln_src_data);
  const auto dst_host_ptr = reinterpret_cast<uintptr_t>(cpu_dst_data);
  const auto size = static_cast<uint64_t>(nbytes);
  RBLN_LOG_DEBUG(
      "Calling rbln_memcpy_v2h: src_vaddr={:#x}, dst_host_ptr={:#x}, size={}", src_vaddr, dst_host_ptr, size);
  RBLN_CHECK(
      !::rbln::rbln_memcpy_v2h(src_vaddr, dst_host_ptr, size),
      "rbln_memcpy_v2h failed ({} bytes, src vaddr={:#x}); the device may be busy or faulted",
      size,
      src_vaddr);
}

void memcpy_v2v(void* rbln_dst_data, const void* rbln_src_data, size_t nbytes) {
  RBLN_LOG_DEBUG(
      "dst_rbln_data={}, src_rbln_data={}, nbytes={}", fmt::ptr(rbln_dst_data), fmt::ptr(rbln_src_data), nbytes);
  RBLN_CHECK(nbytes > 0, "nbytes must be positive, but got {}", nbytes);
  RBLN_CHECK(rbln_src_data != nullptr, "rbln_src_data cannot be nullptr");
  RBLN_CHECK(rbln_dst_data != nullptr, "rbln_dst_data cannot be nullptr");

  if (is_dummy_device()) {
    dummy_check_range(rbln_dst_data, nbytes);
    dummy_check_range(rbln_src_data, nbytes);
    std::memmove(rbln_dst_data, rbln_src_data, nbytes);
    return;
  }

  const auto src_vaddr = reinterpret_cast<uint64_t>(rbln_src_data);
  const auto dst_vaddr = reinterpret_cast<uint64_t>(rbln_dst_data);
  const auto size = static_cast<uint64_t>(nbytes);

  const auto src_torch_device_id = get_torch_device_id(rbln_src_data);
  const auto dst_torch_device_id = get_torch_device_id(rbln_dst_data);

  RBLN_LOG_DEBUG(
      "src=rbln:{}, dst=rbln:{}", static_cast<int>(src_torch_device_id), static_cast<int>(dst_torch_device_id));

  if (src_torch_device_id == dst_torch_device_id) {
    RBLN_LOG_DEBUG("Performing same-device copy");

    RBLN_LOG_DEBUG("Calling rbln_memcpy_v2v: src_vaddr={:#x}, dst_vaddr={:#x}, size={}", src_vaddr, dst_vaddr, size);
    // Tag the failure so `at::native::rbln::submit_or_fallback` can route a
    // rejected same-device copy to its CPU fallback (the batched path already
    // emits "rbln_memcpy_v2v_multi failed"; the per-entry path was message-less
    // and escaped the gate as a hard crash).
    RBLN_CHECK(!::rbln::rbln_memcpy_v2v(src_vaddr, dst_vaddr, size), "rbln_memcpy_v2v failed");
  } else {
    RBLN_LOG_DEBUG("Performing cross-device copy");

    std::vector<uint8_t> host_buffer(nbytes);
    const auto host_buffer_data = host_buffer.data();
    RBLN_LOG_DEBUG("Allocated {} bytes of temporary host buffer at {}", nbytes, fmt::ptr(host_buffer_data));
    const auto host_ptr = reinterpret_cast<uintptr_t>(host_buffer_data);

    RBLN_LOG_DEBUG("Calling rbln_memcpy_v2h: src_vaddr={:#x}, dst_host_ptr={:#x}, size={}", src_vaddr, host_ptr, size);
    RBLN_CHECK(
        !::rbln::rbln_memcpy_v2h(src_vaddr, host_ptr, size),
        "rbln_memcpy_v2h failed during cross-device transfer ({} bytes, src vaddr={:#x})",
        size,
        src_vaddr);
    RBLN_LOG_DEBUG("Calling rbln_memcpy_h2v: src_host_ptr={:#x}, dst_vaddr={:#x}, size={}", host_ptr, dst_vaddr, size);
    RBLN_CHECK(
        !::rbln::rbln_memcpy_h2v(host_ptr, dst_vaddr, size),
        "rbln_memcpy_h2v failed during cross-device transfer ({} bytes, dst vaddr={:#x})",
        size,
        dst_vaddr);
  }
}

void memcpy_h2v_async(void* rbln_dst_data, const void* cpu_src_data, size_t nbytes) {
  RBLN_LOG_DEBUG(
      "dst_rbln_data={}, src_cpu_data={}, nbytes={}", fmt::ptr(rbln_dst_data), fmt::ptr(cpu_src_data), nbytes);
  RBLN_CHECK(nbytes > 0, "nbytes must be positive, but got {}", nbytes);
  RBLN_CHECK(cpu_src_data != nullptr, "cpu_src_data cannot be nullptr");
  RBLN_CHECK(rbln_dst_data != nullptr, "rbln_dst_data cannot be nullptr");

  if (is_dummy_device()) {
    dummy_check_range(rbln_dst_data, nbytes);
    std::memmove(rbln_dst_data, cpu_src_data, nbytes);
    return;
  }

  const auto src_host_ptr = reinterpret_cast<uintptr_t>(cpu_src_data);
  const auto dst_vaddr = reinterpret_cast<uint64_t>(rbln_dst_data);
  const auto size = static_cast<uint64_t>(nbytes);
  uint64_t handle = 0;
  RBLN_LOG_DEBUG(
      "Calling rbln_memcpy_h2v_async: src_host_ptr={:#x}, dst_vaddr={:#x}, size={}", src_host_ptr, dst_vaddr, size);
  RBLN_CHECK(
      !::rbln::rbln_memcpy_h2v_async(src_host_ptr, dst_vaddr, size, &handle),
      "rbln_memcpy_h2v_async failed ({} bytes, dst vaddr={:#x}); the device may be busy or faulted",
      size,
      dst_vaddr);
  RBLN_LOG_DEBUG("H2V async dispatched (handle={}, 0=sync fallback)", handle);
}

void memcpy_v2h_async(void* cpu_dst_data, const void* rbln_src_data, size_t nbytes) {
  RBLN_LOG_DEBUG(
      "dst_cpu_data={}, src_rbln_data={}, nbytes={}", fmt::ptr(cpu_dst_data), fmt::ptr(rbln_src_data), nbytes);
  RBLN_CHECK(nbytes > 0, "nbytes must be positive, but got {}", nbytes);
  RBLN_CHECK(rbln_src_data != nullptr, "rbln_src_data cannot be nullptr");
  RBLN_CHECK(cpu_dst_data != nullptr, "cpu_dst_data cannot be nullptr");

  if (is_dummy_device()) {
    dummy_check_range(rbln_src_data, nbytes);
    std::memmove(cpu_dst_data, rbln_src_data, nbytes);
    return;
  }

  const auto src_vaddr = reinterpret_cast<uint64_t>(rbln_src_data);
  const auto dst_host_ptr = reinterpret_cast<uintptr_t>(cpu_dst_data);
  const auto size = static_cast<uint64_t>(nbytes);
  uint64_t handle = 0;
  RBLN_LOG_DEBUG(
      "Calling rbln_memcpy_v2h_async: src_vaddr={:#x}, dst_host_ptr={:#x}, size={}", src_vaddr, dst_host_ptr, size);
  RBLN_CHECK(
      !::rbln::rbln_memcpy_v2h_async(src_vaddr, dst_host_ptr, size, &handle),
      "rbln_memcpy_v2h_async failed ({} bytes, src vaddr={:#x}); the device may be busy or faulted",
      size,
      src_vaddr);
  RBLN_LOG_DEBUG("V2H async dispatched (handle={}, 0=sync fallback)", handle);
}

void memcpy_v2v_async(void* rbln_dst_data, const void* rbln_src_data, size_t nbytes) {
  RBLN_LOG_DEBUG(
      "dst_rbln_data={}, src_rbln_data={}, nbytes={}", fmt::ptr(rbln_dst_data), fmt::ptr(rbln_src_data), nbytes);
  RBLN_CHECK(nbytes > 0, "nbytes must be positive, but got {}", nbytes);
  RBLN_CHECK(rbln_src_data != nullptr, "rbln_src_data cannot be nullptr");
  RBLN_CHECK(rbln_dst_data != nullptr, "rbln_dst_data cannot be nullptr");

  if (is_dummy_device()) {
    dummy_check_range(rbln_dst_data, nbytes);
    dummy_check_range(rbln_src_data, nbytes);
    std::memmove(rbln_dst_data, rbln_src_data, nbytes);
    return;
  }

  const auto src_vaddr = reinterpret_cast<uint64_t>(rbln_src_data);
  const auto dst_vaddr = reinterpret_cast<uint64_t>(rbln_dst_data);
  const auto size = static_cast<uint64_t>(nbytes);

  const auto src_torch_device_id = get_torch_device_id(rbln_src_data);
  const auto dst_torch_device_id = get_torch_device_id(rbln_dst_data);

  if (src_torch_device_id != dst_torch_device_id) {
    // rbln_memcpy_v2v_async only handles same-device copies; cross-device needs
    // a host bounce, which we cannot do async without owning a buffer past return.
    RBLN_LOG_DEBUG("Cross-device v2v, falling back to sync memcpy_v2v");
    memcpy_v2v(rbln_dst_data, rbln_src_data, nbytes);
    return;
  }

  uint64_t handle = 0;
  RBLN_LOG_DEBUG(
      "Calling rbln_memcpy_v2v_async: src_vaddr={:#x}, dst_vaddr={:#x}, size={}", src_vaddr, dst_vaddr, size);
  RBLN_CHECK(
      !::rbln::rbln_memcpy_v2v_async(src_vaddr, dst_vaddr, size, &handle),
      "rbln_memcpy_v2v_async failed ({} bytes, src vaddr={:#x}, dst vaddr={:#x}); the device may be busy or faulted",
      size,
      src_vaddr,
      dst_vaddr);
  RBLN_LOG_DEBUG("V2V async dispatched (handle={}, 0=sync fallback)", handle);
}

void synchronize(c10::DeviceIndex device_index) {
  RBLN_LOG_DEBUG("Synchronizing device {}", static_cast<int>(device_index));
  check_device_index(device_index);
  if (is_dummy_device()) {
    // Host-backed transfers are synchronous; nothing to drain.
    return;
  }
  const auto torch_device_id = static_cast<uint32_t>(to_device_id(device_index));
  RBLN_CHECK(
      !::rbln::rbln_device_synchronize(torch_device_id),
      "rbln_device_synchronize failed for rbln:{} (device may be busy or in a faulted state)",
      static_cast<int>(device_index));
}

void memcpy_v2v_multi(const std::vector<V2VCopyOp>& copies) {
  if (copies.empty()) {
    return;
  }
  if (is_dummy_device()) {
    for (const auto& c : copies) {
      RBLN_CHECK(c.nbytes > 0, "memcpy_v2v_multi: nbytes must be positive");
      RBLN_CHECK(c.src != nullptr, "memcpy_v2v_multi: src cannot be nullptr");
      RBLN_CHECK(c.dst != nullptr, "memcpy_v2v_multi: dst cannot be nullptr");
      dummy_check_range(c.dst, c.nbytes);
      dummy_check_range(c.src, c.nbytes);
      std::memmove(c.dst, c.src, c.nbytes);
    }
    return;
  }
  std::vector<std::tuple<uint64_t, uint64_t, uint64_t>> rbln_copies;
  rbln_copies.reserve(copies.size());
  for (const auto& c : copies) {
    RBLN_CHECK(c.nbytes > 0, "memcpy_v2v_multi: nbytes must be positive");
    RBLN_CHECK(c.src != nullptr, "memcpy_v2v_multi: src cannot be nullptr");
    RBLN_CHECK(c.dst != nullptr, "memcpy_v2v_multi: dst cannot be nullptr");
    rbln_copies.emplace_back(
        reinterpret_cast<uint64_t>(c.src), reinterpret_cast<uint64_t>(c.dst), static_cast<uint64_t>(c.nbytes));
  }
  RBLN_LOG_DEBUG("Calling rbln_memcpy_v2v_multi: n_copies={}", copies.size());
  // Error message matched by `at::native::rbln::submit_or_fallback` to gate CPU fallback — keep stable.
  RBLN_CHECK(!::rbln::rbln_memcpy_v2v_multi(rbln_copies), "rbln_memcpy_v2v_multi failed");
}

// In dummy mode the pointer already IS host memory, so a borrow is the identity
// (host_ptr == the pointer) with a synthetic non-zero id; return_borrowed no-ops.
BorrowedHostPtr borrow_host_ptr(const void* rbln_data, size_t nbytes) {
  RBLN_LOG_DEBUG("rbln_data={}, nbytes={}", fmt::ptr(rbln_data), nbytes);
  RBLN_CHECK(rbln_data != nullptr, "rbln_data cannot be nullptr");
  RBLN_CHECK(nbytes > 0, "nbytes must be positive, but got {}", nbytes);

  if (is_dummy_device()) {
    dummy_check_range(rbln_data, nbytes);
    return BorrowedHostPtr{reinterpret_cast<uintptr_t>(rbln_data), next_dummy_borrow_id()};
  }

  const auto vaddr = reinterpret_cast<uint64_t>(rbln_data);
  const auto size = static_cast<uint64_t>(nbytes);
  uintptr_t host_ptr = 0;
  uint64_t borrow_id = 0;
  RBLN_LOG_DEBUG("Calling rbln_v_borrow_host_ptr: vaddr={:#x}, size={}", vaddr, size);
  RBLN_CHECK(
      !::rbln::rbln_v_borrow_host_ptr(vaddr, size, host_ptr, borrow_id),
      "rbln_v_borrow_host_ptr failed (vaddr={:#x}, size={}); see rebel runtime logs for details",
      vaddr,
      size);
  return BorrowedHostPtr{host_ptr, borrow_id};
}

std::optional<BorrowedHostPtr> try_borrow_host_ptr(const void* rbln_data, size_t nbytes) {
  if (rbln_data == nullptr || nbytes == 0) {
    return std::nullopt;
  }
  if (is_dummy_device()) {
    if (!dummy_range_ok(rbln_data, nbytes)) {
      return std::nullopt;
    }
    return BorrowedHostPtr{reinterpret_cast<uintptr_t>(rbln_data), next_dummy_borrow_id()};
  }
  const auto vaddr = reinterpret_cast<uint64_t>(rbln_data);
  const auto size = static_cast<uint64_t>(nbytes);
  uintptr_t host_ptr = 0;
  uint64_t borrow_id = 0;
  // Non-zero return == failure (mirrors borrow_host_ptr's RBLN_CHECK(!...)).
  // A failure here is an expected, recoverable condition for callers with a
  // copy-based fallback, so report it as nullopt rather than throwing.
  if (::rbln::rbln_v_borrow_host_ptr(vaddr, size, host_ptr, borrow_id)) {
    return std::nullopt;
  }
  return BorrowedHostPtr{host_ptr, borrow_id};
}

BorrowedHostPtr acquire_host_ptr_for_overwrite(void* rbln_data, size_t nbytes) {
  RBLN_LOG_DEBUG("rbln_data={}, nbytes={}", fmt::ptr(rbln_data), nbytes);
  RBLN_CHECK(rbln_data != nullptr, "rbln_data cannot be nullptr");
  RBLN_CHECK(nbytes > 0, "nbytes must be positive, but got {}", nbytes);

  if (is_dummy_device()) {
    dummy_check_range(rbln_data, nbytes);
    return BorrowedHostPtr{reinterpret_cast<uintptr_t>(rbln_data), next_dummy_borrow_id()};
  }

  const auto vaddr = reinterpret_cast<uint64_t>(rbln_data);
  const auto size = static_cast<uint64_t>(nbytes);
  uintptr_t host_ptr = 0;
  uint64_t borrow_id = 0;
  RBLN_LOG_DEBUG("Calling rbln_v_acquire_host_ptr_for_overwrite: vaddr={:#x}, size={}", vaddr, size);
  RBLN_CHECK(
      !::rbln::rbln_v_acquire_host_ptr_for_overwrite(vaddr, size, host_ptr, borrow_id),
      "rbln_v_acquire_host_ptr_for_overwrite failed (vaddr={:#x}, size={}); see rebel runtime logs for details",
      vaddr,
      size);
  return BorrowedHostPtr{host_ptr, borrow_id};
}

std::optional<BorrowedHostPtr> try_acquire_host_ptr_for_overwrite(void* rbln_data, size_t nbytes) {
  if (rbln_data == nullptr || nbytes == 0) {
    return std::nullopt;
  }
  if (is_dummy_device()) {
    if (!dummy_range_ok(rbln_data, nbytes)) {
      return std::nullopt;
    }
    return BorrowedHostPtr{reinterpret_cast<uintptr_t>(rbln_data), next_dummy_borrow_id()};
  }
  const auto vaddr = reinterpret_cast<uint64_t>(rbln_data);
  const auto size = static_cast<uint64_t>(nbytes);
  uintptr_t host_ptr = 0;
  uint64_t borrow_id = 0;
  // Non-zero return == failure (mirrors acquire_host_ptr_for_overwrite's
  // RBLN_CHECK(!...)). A failure here is an expected, recoverable condition for
  // callers with a copy-based fallback, so report it as nullopt rather than
  // throwing.
  if (::rbln::rbln_v_acquire_host_ptr_for_overwrite(vaddr, size, host_ptr, borrow_id)) {
    return std::nullopt;
  }
  return BorrowedHostPtr{host_ptr, borrow_id};
}

void return_borrowed(uint64_t borrow_id, bool updated) {
  // borrow_id == 0 is a "no live borrow" sentinel — see header. Cleanup
  // paths may call this unconditionally over entries that were skipped.
  if (borrow_id == 0) {
    return;
  }
  if (is_dummy_device()) {
    // Dummy borrows are identity host views; nothing to release.
    return;
  }
  RBLN_LOG_DEBUG("borrow_id={}, updated={}", borrow_id, updated);
  RBLN_CHECK(
      !::rbln::rbln_v_return_borrowed(borrow_id, updated),
      "rbln_v_return_borrowed failed (borrow_id={}, updated={}); see rebel runtime logs for details",
      borrow_id,
      updated);
}

c10::CachingDeviceAllocator::DeviceStats get_device_stats(const c10::Device& device) {
  RBLN_LOG_DEBUG("logical device={}", c10::str(device));
  const auto device_index = device.index();
  check_device_index(device_index);

  if (is_dummy_device()) {
    // No runtime allocator in dummy mode; report empty stats.
    return c10::CachingDeviceAllocator::DeviceStats{};
  }

  const auto device_id = to_device_id(device_index);
  RBLN_LOG_DEBUG("Calling rbln_get_memory_stats: device_id={}", device_id);
  const auto memory_stats = rbln_get_memory_stats(device_id);

  c10::CachingDeviceAllocator::DeviceStats stats{};
  constexpr auto kAggregate = static_cast<size_t>(c10::CachingAllocator::StatType::AGGREGATE);

  // allocated_bytes
  stats.allocated_bytes[kAggregate].current = static_cast<int64_t>(memory_stats.GetAllocatedCurrent());
  stats.allocated_bytes[kAggregate].peak = static_cast<int64_t>(memory_stats.GetAllocatedPeak());
  stats.allocated_bytes[kAggregate].allocated = static_cast<int64_t>(memory_stats.GetAllocatedTotalAllocated());
  stats.allocated_bytes[kAggregate].freed = static_cast<int64_t>(memory_stats.GetAllocatedTotalFreed());

  // reserved_bytes
  stats.reserved_bytes[kAggregate].current = static_cast<int64_t>(memory_stats.GetReservedCurrent());
  stats.reserved_bytes[kAggregate].peak = static_cast<int64_t>(memory_stats.GetReservedPeak());
  stats.reserved_bytes[kAggregate].allocated = static_cast<int64_t>(memory_stats.GetReservedTotalAllocated());
  stats.reserved_bytes[kAggregate].freed = static_cast<int64_t>(memory_stats.GetReservedTotalFreed());

  // active_bytes
  stats.active_bytes[kAggregate].current = static_cast<int64_t>(memory_stats.GetActiveCurrent());
  stats.active_bytes[kAggregate].peak = static_cast<int64_t>(memory_stats.GetActivePeak());

  // inactive_split_bytes — mapped from memory_stats's "cached" (reusable fragmented blocks).
  stats.inactive_split_bytes[kAggregate].current = static_cast<int64_t>(memory_stats.GetCachedCurrent());
  stats.inactive_split_bytes[kAggregate].peak = static_cast<int64_t>(memory_stats.GetCachedPeak());

  // scalar counters
  stats.num_alloc_retries = static_cast<int64_t>(memory_stats.GetNumAllocRetries());
  stats.num_ooms = static_cast<int64_t>(memory_stats.GetNumOoms());
  stats.num_device_alloc = static_cast<int64_t>(memory_stats.GetNumDeviceAlloc());
  stats.num_device_free = static_cast<int64_t>(memory_stats.GetNumDeviceFree());

  RBLN_LOG_DEBUG(
      "allocated(current={}, peak={}, allocated={}, freed={}), reserved(current={}, peak={}, allocated={}, freed={}), "
      "active(current={}, peak={})",
      stats.allocated_bytes[kAggregate].current,
      stats.allocated_bytes[kAggregate].peak,
      stats.allocated_bytes[kAggregate].allocated,
      stats.allocated_bytes[kAggregate].freed,
      stats.reserved_bytes[kAggregate].current,
      stats.reserved_bytes[kAggregate].peak,
      stats.reserved_bytes[kAggregate].allocated,
      stats.reserved_bytes[kAggregate].freed,
      stats.active_bytes[kAggregate].current,
      stats.active_bytes[kAggregate].peak);
  return stats;
}

void empty_cache(const c10::Device& device) {
  RBLN_LOG_DEBUG("logical device={}", c10::str(device));
  const auto device_index = device.index();
  check_device_index(device_index);

  if (is_dummy_device()) {
    return; // host malloc/free are immediate; no runtime cache to flush
  }

  const auto device_id = to_device_id(device_index);
  RBLN_LOG_DEBUG("Calling rbln_empty_cache: device_id={}", device_id);
  RBLN_CHECK(
      !rbln_empty_cache(device_id),
      "rbln_empty_cache failed for rbln:{} (device may be busy or in a faulted state)",
      static_cast<int>(device_index));
}

std::map<std::string, uint64_t> memory_stats(const c10::Device& device) {
  RBLN_LOG_DEBUG("logical device={}", c10::str(device));
  const auto device_index = device.index();
  check_device_index(device_index);

  if (is_dummy_device()) {
    return {}; // no runtime allocator stats in dummy mode
  }

  const auto device_id = to_device_id(device_index);
  RBLN_LOG_DEBUG("Calling rbln_get_memory_stats: rbln:{}, device_id={}", static_cast<int>(device_index), device_id);
  const auto stats = rbln_get_memory_stats(device_id);

  const auto memory_stats = stats.GetMemoryStats();
  RBLN_LOG_DEBUG("memory_stats={}", memory_stats);
  return memory_stats;
}

void reset_accumulated_memory_stats(const c10::Device& device) {
  RBLN_LOG_DEBUG("logical device={}", c10::str(device));
  const auto device_index = device.index();
  check_device_index(device_index);

  if (is_dummy_device()) {
    return; // no runtime allocator stats in dummy mode
  }

  const auto device_id = to_device_id(device_index);
  RBLN_LOG_DEBUG("Calling rbln_reset_accumulated_memory_stats: device_id={}", device_id);
  RBLN_CHECK(
      !rbln_reset_accumulated_memory_stats(device_id),
      "rbln_reset_accumulated_memory_stats failed for rbln:{}",
      static_cast<int>(device_index));
}

void reset_peak_memory_stats(const c10::Device& device) {
  RBLN_LOG_DEBUG("logical device={}", c10::str(device));
  const auto device_index = device.index();
  check_device_index(device_index);

  if (is_dummy_device()) {
    return; // no runtime allocator stats in dummy mode
  }

  const auto device_id = to_device_id(device_index);
  RBLN_LOG_DEBUG("Calling rbln_reset_peak_memory_stats: device_id={}", device_id);
  RBLN_CHECK(
      !rbln_reset_peak_memory_stats(device_id),
      "rbln_reset_peak_memory_stats failed for rbln:{}",
      static_cast<int>(device_index));
}

void set_file_offloading_enabled(bool enabled) {
  RBLN_LOG_DEBUG("Calling rbln_set_file_offloading_enabled: enabled={}", enabled);
  RBLN_CHECK(
      !::rbln::rbln_set_file_offloading_enabled(enabled),
      "rbln_set_file_offloading_enabled failed (enabled={})",
      enabled);
}

} // namespace c10::rbln
