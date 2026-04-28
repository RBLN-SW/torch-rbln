#include <ATen/ATen.h>
#include <c10/rbln/DeviceMappingManager.h>
#include <c10/rbln/RBLNFunctions.h>
#include <c10/rbln/RBLNLogging.h>
#include <c10/util/CallOnce.h>
#include <rebel/runtime/memory_stats.h>

#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace c10::rbln {

namespace {

// Default current logical device is 0
thread_local c10::DeviceIndex current_device_index_ = 0;

void check_device_index(c10::DeviceIndex device_index) {
  constexpr auto max_device_index = std::numeric_limits<c10::DeviceIndex>::max();
  RBLN_CHECK(
      device_index <= max_device_index, "Overflowed logical device index (rbln:", static_cast<int>(device_index), ")");
  auto& manager = DeviceMappingManager::getInstance();
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
  const auto device_id = static_cast<int>(static_cast<unsigned char>(device_index));
  return device_id;
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
  // Directly query the runtime API for physical device count
  // This bypasses the RSD mode logic and always returns the actual physical count
  int device_count = 0;
  RBLN_CHECK(!rbln_get_device_count(&device_count));

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

::rbln::MemoryInfo get_memory_info(const void* data) {
  RBLN_LOG_DEBUG("data={}", fmt::ptr(data));
  RBLN_CHECK(data != nullptr, "data cannot be nullptr");

  const auto vaddr = reinterpret_cast<uint64_t>(data);
  ::rbln::MemoryInfo memory_info;
  RBLN_LOG_DEBUG("Calling rbln_get_memory_info: vaddr={:#x}", vaddr);
  RBLN_CHECK(!::rbln::rbln_get_memory_info(vaddr, memory_info));
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

void* malloc(c10::DeviceIndex device_index, size_t nbytes) {
  RBLN_LOG_DEBUG("logical device=rbln:{}, nbytes={}", static_cast<int>(device_index), nbytes);
  RBLN_CHECK(nbytes > 0, "nbytes must be positive, but got {}", nbytes);
  check_device_index(device_index);

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
    RBLN_CHECK(!::rbln::rbln_malloc_eager(torch_device_id, size, vaddr));
  } else {
    RBLN_LOG_DEBUG(
        "Calling rbln_malloc_lazy: rbln:{}, torch_device_id={}, size={}",
        static_cast<int>(device_index),
        torch_device_id,
        size);
    RBLN_CHECK(!::rbln::rbln_malloc_lazy(torch_device_id, size, vaddr));
  }

  auto* data = reinterpret_cast<void*>(vaddr); // NOLINT(performance-no-int-to-ptr)
  RBLN_LOG_DEBUG("data={}", fmt::ptr(data));
  RBLN_CHECK(data != nullptr, "data cannot be nullptr");
  return data;
}

void mark_zeros(const void* rbln_data) {
  RBLN_LOG_DEBUG("rbln_data={}", fmt::ptr(rbln_data));
  RBLN_CHECK(rbln_data != nullptr, "rbln_data cannot be nullptr");

  const auto vaddr = reinterpret_cast<uint64_t>(rbln_data);
  RBLN_CHECK(!::rbln::rbln_mark_zeros(vaddr), "rbln_mark_zeros failed for vaddr={:#x}", vaddr);
  RBLN_LOG_DEBUG("vaddr={:#x} marked as zero-initialized", vaddr);
}

void free(void* data) {
  RBLN_LOG_DEBUG("data={}", fmt::ptr(data));
  RBLN_CHECK(data != nullptr, "data cannot be nullptr");

  const auto vaddr = reinterpret_cast<uint64_t>(data);
  RBLN_LOG_DEBUG("Calling rbln_free: vaddr={:#x}", vaddr);
  RBLN_CHECK(!::rbln::rbln_free(vaddr));
}

void memcpy_h2v(void* rbln_dst_data, const void* cpu_src_data, size_t nbytes) {
  RBLN_LOG_DEBUG(
      "dst_rbln_data={}, src_cpu_data={}, nbytes={}", fmt::ptr(rbln_dst_data), fmt::ptr(cpu_src_data), nbytes);
  RBLN_CHECK(nbytes > 0, "nbytes must be positive, but got {}", nbytes);
  RBLN_CHECK(cpu_src_data != nullptr, "cpu_src_data cannot be nullptr");
  RBLN_CHECK(rbln_dst_data != nullptr, "rbln_dst_data cannot be nullptr");

  const auto src_host_ptr = reinterpret_cast<uintptr_t>(cpu_src_data);
  const auto dst_vaddr = reinterpret_cast<uint64_t>(rbln_dst_data);
  const auto size = static_cast<uint64_t>(nbytes);
  RBLN_LOG_DEBUG(
      "Calling rbln_memcpy_h2v: src_host_ptr={:#x}, dst_vaddr={:#x}, size={}", src_host_ptr, dst_vaddr, size);
  RBLN_CHECK(!::rbln::rbln_memcpy_h2v(src_host_ptr, dst_vaddr, size));
}

void memcpy_v2h(void* cpu_dst_data, const void* rbln_src_data, size_t nbytes) {
  RBLN_LOG_DEBUG(
      "dst_cpu_data={}, src_rbln_data={}, nbytes={}", fmt::ptr(cpu_dst_data), fmt::ptr(rbln_src_data), nbytes);
  RBLN_CHECK(nbytes > 0, "nbytes must be positive, but got {}", nbytes);
  RBLN_CHECK(rbln_src_data != nullptr, "rbln_src_data cannot be nullptr");
  RBLN_CHECK(cpu_dst_data != nullptr, "cpu_dst_data cannot be nullptr");

  const auto src_vaddr = reinterpret_cast<uint64_t>(rbln_src_data);
  const auto dst_host_ptr = reinterpret_cast<uintptr_t>(cpu_dst_data);
  const auto size = static_cast<uint64_t>(nbytes);
  RBLN_LOG_DEBUG(
      "Calling rbln_memcpy_v2h: src_vaddr={:#x}, dst_host_ptr={:#x}, size={}", src_vaddr, dst_host_ptr, size);
  RBLN_CHECK(!::rbln::rbln_memcpy_v2h(src_vaddr, dst_host_ptr, size));
}

void memcpy_v2v(void* rbln_dst_data, const void* rbln_src_data, size_t nbytes) {
  RBLN_LOG_DEBUG(
      "dst_rbln_data={}, src_rbln_data={}, nbytes={}", fmt::ptr(rbln_dst_data), fmt::ptr(rbln_src_data), nbytes);
  RBLN_CHECK(nbytes > 0, "nbytes must be positive, but got {}", nbytes);
  RBLN_CHECK(rbln_src_data != nullptr, "rbln_src_data cannot be nullptr");
  RBLN_CHECK(rbln_dst_data != nullptr, "rbln_dst_data cannot be nullptr");

  const auto src_memory_info = get_memory_info(rbln_src_data);
  const auto dst_memory_info = get_memory_info(rbln_dst_data);
  const auto src_device_index = static_cast<c10::DeviceIndex>(src_memory_info.torch_device_id);
  const auto dst_device_index = static_cast<c10::DeviceIndex>(dst_memory_info.torch_device_id);
  RBLN_LOG_DEBUG("src=rbln:{}, dst=rbln:{}", static_cast<int>(src_device_index), static_cast<int>(dst_device_index));

  const auto src_vaddr = reinterpret_cast<uint64_t>(rbln_src_data);
  const auto dst_vaddr = reinterpret_cast<uint64_t>(rbln_dst_data);
  const auto size = static_cast<uint64_t>(nbytes);
  if (src_device_index == dst_device_index) {
    RBLN_LOG_DEBUG("Performing same-device copy");

    RBLN_LOG_DEBUG("Calling rbln_memcpy_v2v: src_vaddr={:#x}, dst_vaddr={:#x}, size={}", src_vaddr, dst_vaddr, size);
    RBLN_CHECK(!::rbln::rbln_memcpy_v2v(src_vaddr, dst_vaddr, size));
  } else {
    RBLN_LOG_DEBUG("Performing cross-device copy");

    std::vector<uint8_t> host_buffer(nbytes);
    const auto host_buffer_data = host_buffer.data();
    RBLN_LOG_DEBUG("Allocated {} bytes of temporary host buffer at {}", nbytes, fmt::ptr(host_buffer_data));
    const auto host_ptr = reinterpret_cast<uintptr_t>(host_buffer_data);

    RBLN_LOG_DEBUG("Calling rbln_memcpy_v2h: src_vaddr={:#x}, dst_host_ptr={:#x}, size={}", src_vaddr, host_ptr, size);
    RBLN_CHECK(!::rbln::rbln_memcpy_v2h(src_vaddr, host_ptr, size));
    RBLN_LOG_DEBUG("Calling rbln_memcpy_h2v: src_host_ptr={:#x}, dst_vaddr={:#x}, size={}", host_ptr, dst_vaddr, size);
    RBLN_CHECK(!::rbln::rbln_memcpy_h2v(host_ptr, dst_vaddr, size));
  }
}

BorrowedHostPtr borrow_host_ptr(const void* rbln_data, size_t nbytes) {
  RBLN_LOG_DEBUG("rbln_data={}, nbytes={}", fmt::ptr(rbln_data), nbytes);
  RBLN_CHECK(rbln_data != nullptr, "rbln_data cannot be nullptr");
  RBLN_CHECK(nbytes > 0, "nbytes must be positive, but got {}", nbytes);

  const auto vaddr = reinterpret_cast<uint64_t>(rbln_data);
  const auto size = static_cast<uint64_t>(nbytes);
  uintptr_t host_ptr = 0;
  uint64_t borrow_id = 0;
  RBLN_LOG_DEBUG("Calling rbln_v_borrow_host_ptr: vaddr={:#x}, size={}", vaddr, size);
  RBLN_CHECK(!::rbln::rbln_v_borrow_host_ptr(vaddr, size, host_ptr, borrow_id));
  return BorrowedHostPtr{host_ptr, borrow_id};
}

BorrowedHostPtr acquire_host_ptr_for_overwrite(void* rbln_data, size_t nbytes) {
  RBLN_LOG_DEBUG("rbln_data={}, nbytes={}", fmt::ptr(rbln_data), nbytes);
  RBLN_CHECK(rbln_data != nullptr, "rbln_data cannot be nullptr");
  RBLN_CHECK(nbytes > 0, "nbytes must be positive, but got {}", nbytes);

  const auto vaddr = reinterpret_cast<uint64_t>(rbln_data);
  const auto size = static_cast<uint64_t>(nbytes);
  uintptr_t host_ptr = 0;
  uint64_t borrow_id = 0;
  RBLN_LOG_DEBUG("Calling rbln_v_acquire_host_ptr_for_overwrite: vaddr={:#x}, size={}", vaddr, size);
  RBLN_CHECK(!::rbln::rbln_v_acquire_host_ptr_for_overwrite(vaddr, size, host_ptr, borrow_id));
  return BorrowedHostPtr{host_ptr, borrow_id};
}

void return_borrowed(uint64_t borrow_id, bool updated) {
  if (borrow_id == 0) {
    return;
  }
  RBLN_LOG_DEBUG("borrow_id={}, updated={}", borrow_id, updated);
  RBLN_CHECK(!::rbln::rbln_v_return_borrowed(borrow_id, updated));
}

c10::CachingDeviceAllocator::DeviceStats get_device_stats(const c10::Device& device) {
  RBLN_LOG_DEBUG("logical device={}", c10::str(device));
  const auto device_index = device.index();
  check_device_index(device_index);

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

  const auto device_id = to_device_id(device_index);
  RBLN_LOG_DEBUG("Calling rbln_empty_cache: device_id={}", device_id);
  RBLN_CHECK(!rbln_empty_cache(device_id));
}

std::map<std::string, uint64_t> memory_stats(const c10::Device& device) {
  RBLN_LOG_DEBUG("logical device={}", c10::str(device));
  const auto device_index = device.index();
  check_device_index(device_index);

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

  const auto device_id = to_device_id(device_index);
  RBLN_LOG_DEBUG("Calling rbln_reset_accumulated_memory_stats: device_id={}", device_id);
  RBLN_CHECK(!rbln_reset_accumulated_memory_stats(device_id));
}

void reset_peak_memory_stats(const c10::Device& device) {
  RBLN_LOG_DEBUG("logical device={}", c10::str(device));
  const auto device_index = device.index();
  check_device_index(device_index);

  const auto device_id = to_device_id(device_index);
  RBLN_LOG_DEBUG("Calling rbln_reset_peak_memory_stats: device_id={}", device_id);
  RBLN_CHECK(!rbln_reset_peak_memory_stats(device_id));
}

} // namespace c10::rbln
