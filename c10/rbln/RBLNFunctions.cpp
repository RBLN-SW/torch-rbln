#include <ATen/ATen.h>
#include <c10/rbln/DeviceMappingManager.h>
#include <c10/rbln/RBLNFunctions.h>
#include <c10/rbln/RBLNLogging.h>
#include <c10/util/CallOnce.h>
#include <rebel/runtime/memory_stats.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <vector>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>

namespace c10::rbln {

// rt-timing: time spent inside librbln boundary calls, for the explain profiler's
// "rebel runtime vs torch dispatch" split. Gated: when disabled each boundary call
// pays one relaxed atomic load (no clock read), so ON==OFF latency holds; an explain
// region flips it on for its duration. Index order MUST match kRtTimingN / the
// Python _RT_PRIMS tuple.
namespace {
enum RtIdx : std::uint8_t { RT_V2V = 0, RT_V2V_MULTI, RT_BORROW, RT_ACQUIRE, RT_RETURN, RT_V2H, RT_H2V, RT_N };
static_assert(static_cast<std::size_t>(RT_N) == kRtTimingN, "RtIdx count must match kRtTimingN in the header");
std::atomic<bool> g_rt_enabled{false};
struct RtAcc {
  std::atomic<uint64_t> ns{0};
  std::atomic<uint64_t> cnt{0};
};
RtAcc* rt_accs() {
  static std::array<RtAcc, RT_N> accs;
  return accs.data();
}
struct RtTimer {
  int idx;
  bool on;
  std::chrono::steady_clock::time_point t0;
  explicit RtTimer(int i) : idx(i), on(g_rt_enabled.load(std::memory_order_relaxed)) {
    if (on) {
      t0 = std::chrono::steady_clock::now();
    }
  }
  ~RtTimer() {
    if (!on) {
      return;
    }
    const auto dt = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
    rt_accs()[idx].ns.fetch_add(static_cast<uint64_t>(dt), std::memory_order_relaxed);
    rt_accs()[idx].cnt.fetch_add(1, std::memory_order_relaxed);
  }
};
} // namespace

void rt_timing_enable(bool on) {
  g_rt_enabled.store(on, std::memory_order_relaxed);
}
void rt_timing_reset() {
  for (std::size_t i = 0; i < kRtTimingN; ++i) {
    rt_accs()[i].ns.store(0, std::memory_order_relaxed);
    rt_accs()[i].cnt.store(0, std::memory_order_relaxed);
  }
}
void rt_timing_get(uint64_t* out) {
  for (std::size_t i = 0; i < kRtTimingN; ++i) {
    out[2 * i] = rt_accs()[i].ns.load(std::memory_order_relaxed);
    out[2 * i + 1] = rt_accs()[i].cnt.load(std::memory_order_relaxed);
  }
}

// torch.rbln.explain() runtime-counter reads. Thin pass-throughs to librbln's
// public C-API (rbln_prof_*, declared in rbln_runtime_api.h via RBLNFunctions.h).
uint32_t rt_prof_hidden_num() {
  return rbln_prof_v2v_hidden_num_reasons();
}
void rt_prof_hidden_get(uint64_t* counts, uint64_t* bytes, uint32_t n) {
  rbln_prof_get_v2v_hidden_d2h(counts, bytes, n);
}
uint32_t rt_prof_reject_num() {
  return rbln_prof_v2v_reject_num_reasons();
}
void rt_prof_reject_get(uint64_t* counts, uint64_t* bytes, uint32_t n) {
  rbln_prof_get_v2v_reject(counts, bytes, n);
}
void rt_prof_host_sync_d2h(uint64_t* count, uint64_t* bytes) {
  rbln_prof_get_host_sync_d2h(count, bytes);
}
void rt_prof_host_sync_h2d(uint64_t* count, uint64_t* bytes) {
  rbln_prof_get_host_sync_h2d(count, bytes);
}
void rt_prof_memory(uint64_t* current, uint64_t* peak) {
  rbln_prof_get_memory(current, peak);
}
void rt_prof_reset() {
  rbln_prof_reset_v2v_hidden_d2h();
  rbln_prof_reset_v2v_reject();
  rbln_prof_reset_host_sync_d2h();
  rbln_prof_reset_host_sync_h2d();
}

namespace {

// Default current logical device is 0
thread_local c10::DeviceIndex current_device_index_ = 0;

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
  //
  // Also the commit point: reaching here means the process has decided to use a device, so
  // this is where the plan is claimed with the runtime and the mapping freezes. Idempotent.
  // check_device_index() stays plan-only -- selecting a device is bookkeeping.
  DeviceMappingManager::getInstance().commit();
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
  // Read live per call (not a process-lifetime static) so the value reflects the current
  // environment. getenv is not thread-safe against a concurrent setenv/putenv, but this is
  // safe here: only malloc() consults it, free is mode-agnostic (so an env change between an
  // allocation and its free can't cause an alloc/free mismatch), and the env is not mutated
  // concurrently with allocation (prod fixes it at startup; tests toggle it only at quiescent
  // points, single-threaded per worker).
  const auto* env = std::getenv("TORCH_RBLN_EAGER_MALLOC");
  const bool eager_malloc = (env != nullptr) && (std::strcmp(env, "1") == 0);
  RBLN_LOG_DEBUG("eager_malloc={}", eager_malloc);
  return eager_malloc;
}

bool is_dummy_device() {
  // Runtime-free: reads the RBLN_DUMMY_DEVICE flag directly, not via
  // DeviceMappingManager (whose init would query the runtime). Cached.
  static const bool dummy = dummyDeviceEnabled();
  return dummy;
}

// --- Device-runtime liveness ------------------------------------------------
// The device runtime (driver) is loaded lazily and is absent on compile/CPU-only/CI
// hosts or unmapped at shutdown, when a raw rbln_* call SEGFAULTs. Runtime-touching
// leaves gate on runtime_available(), built on librbln's own rbln_runtime_available().

namespace {

std::atomic<bool> runtime_shutting_down_{false}; // set at teardown via a Python atexit hook

// Torn down (shutting down or driver absent) vs "no device present" (reported by
// to_device_id()): teardown-safe ops no-op only on the former.
bool runtime_torn_down() noexcept {
  return runtime_shutting_down_.load(std::memory_order_relaxed) || !rbln_runtime_available();
}

// Mandatory-op guard: clean throw, never a SEGFAULT. Required even in dummy mode.
void require_runtime(const char* op) {
  RBLN_CHECK(
      !runtime_shutting_down_.load(std::memory_order_relaxed), "Cannot {}: the RBLN runtime is shutting down.", op);
  RBLN_CHECK(
      rbln_runtime_available(),
      "Cannot {}: the RBLN device runtime is not loaded; install the RBLN SDK/runtime "
      "(required even in RBLN_DUMMY_DEVICE mode).",
      op);
}

// c10::Error::what() appends "Exception raised from ..." plus a full C++ backtrace.
// Keep only the human-readable first line for warnings on nothrow paths.
std::string first_line(std::string_view text) {
  return std::string(text.substr(0, text.find('\n')));
}

} // namespace

c10::DeviceIndex get_device_count_nothrow() noexcept {
  // Nothrow view of get_device_count(); failures map to 0. First line only, because
  // e.what() carries the C++ stack trace and every co-tenant walks this path. The full
  // text is still raised by device_count_ensure_non_zero() and by the allocation path.
  try {
    return get_device_count();
  } catch (const std::exception& e) {
    RBLN_WARN_NOTHROW("get_device_count failed, treating as 0 device(s): {}", first_line(e.what()));
    return 0;
  } catch (...) {
    RBLN_WARN_NOTHROW("get_device_count failed, treating as 0 device(s): unknown exception");
    return 0;
  }
}

c10::DeviceIndex device_count_ensure_non_zero() {
  // Throwing counterpart of the noexcept query, named after c10::cuda's
  // device_count_ensure_non_zero(). This is where a malformed RBLN_* config becomes a
  // loud, detailed error: the availability path stays quiet, the point of use does not.
  const auto device_count = get_device_count();
  RBLN_CHECK(
      device_count > 0,
      "No RBLN devices are available (0 logical device(s)). Check that an NPU is present, the rbln kernel driver "
      "is loaded, and RBLN_DEVICES / RBLN_DEVICE_MAP / RBLN_NPUS_PER_DEVICE select at least one device.");
  return device_count;
}

void commit_device_mapping() {
  DeviceMappingManager::getInstance().commit();
}

void set_runtime_shutting_down(bool value) noexcept {
  runtime_shutting_down_.store(value, std::memory_order_relaxed);
}

bool runtime_available() noexcept {
  // Driver loaded, not shutting down, at least one usable logical device. Bound to Python
  // is_available() and to RBLNHooksInterface::hasRBLN(), so the two cannot disagree.
  //
  // Dummy mode is NOT short-circuited: doing so reported True for a dummy device whose
  // mapping had failed to build -- available yet unusable.
  return !runtime_shutting_down_.load(std::memory_order_relaxed) && rbln_runtime_available() &&
      get_device_count_nothrow() > 0;
}

// --- Per-process device-context tracking ------------------------------------
// A per-logical-device bit set on the first successful device malloc, mirroring CUDA's
// device_allocator populated on first use. Backs initialized()/hasPrimaryContext() and
// gates the best-effort memory ops, so a process with the runtime + a mapping but no live
// context (e.g. a vLLM EngineCore parent) reports uninitialized. Set-once, lock-free.
// RBLN device use after fork is unsupported; bad-fork detection is not implemented yet
// (the mask is inherited stale in a fork child).
namespace {
// Two 64-bit words cover the full valid DeviceIndex range. DeviceMappingManager caps
// logical devices at numeric_limits<DeviceIndex>::max(), so valid indices are
// [0, max) = [0, 126] and index 127 is never a device — no valid device is silently
// untracked (the earlier single-word tracker dropped indices 64+).
constexpr c10::DeviceIndex kMaxTrackedDevices = std::numeric_limits<c10::DeviceIndex>::max(); // 127
std::array<std::atomic<uint64_t>, 2> g_context_init_mask{}; // 128 bits
} // namespace

void mark_device_context_initialized(c10::DeviceIndex device_index) noexcept {
  if (device_index >= 0 && device_index < kMaxTrackedDevices) {
    g_context_init_mask[device_index >> 6].fetch_or(uint64_t{1} << (device_index & 63), std::memory_order_relaxed);
  }
}

bool device_context_initialized(c10::DeviceIndex device_index) noexcept {
  return device_index >= 0 && device_index < kMaxTrackedDevices &&
      ((g_context_init_mask[device_index >> 6].load(std::memory_order_relaxed) >> (device_index & 63)) & 1U) != 0;
}

bool any_device_context_initialized() noexcept {
  return (g_context_init_mask[0].load(std::memory_order_relaxed) |
          g_context_init_mask[1].load(std::memory_order_relaxed)) != 0;
}

std::vector<c10::DeviceIndex> initialized_device_indices() {
  std::vector<c10::DeviceIndex> indices;
  // Context flag first: nothing initialized anywhere -> empty, without asking the runtime
  // for a count this process has nothing to report against.
  if (!any_device_context_initialized()) {
    return indices;
  }
  const auto device_count = get_device_count();
  for (c10::DeviceIndex idx = 0; idx < device_count; ++idx) {
    if (device_context_initialized(idx)) {
      indices.push_back(idx);
    }
  }
  return indices;
}

void* malloc(c10::DeviceIndex device_index, size_t nbytes) {
  RBLN_LOG_DEBUG("logical device=rbln:{}, nbytes={}", static_cast<int>(device_index), nbytes);
  RBLN_CHECK(nbytes > 0, "nbytes must be positive, but got {}", nbytes);
  check_device_index(device_index);

  // Allocation is the gateway: clean throw (not SEGFAULT) if the runtime is gone;
  // to_device_id() then throws on a host with no device.
  require_runtime("allocate device memory");
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
  mark_device_context_initialized(device_index); // this process now owns context on this device
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

  require_runtime("free device memory");
  const auto vaddr = reinterpret_cast<uint64_t>(data);
  RBLN_LOG_DEBUG("Calling rbln_free: vaddr={:#x}", vaddr);
  RBLN_CHECK(
      !::rbln::rbln_free(vaddr),
      "rbln_free failed for vaddr={:#x} (device may have been reset/lost, or the address was already freed)",
      vaddr);
}

void free_nothrow(void* data) noexcept {
  // Noexcept deleter: rbln_free and RBLN_WARN_NOTHROW are both nothrow.
  if (data == nullptr) {
    return;
  }
  // Torn-down runtime: rbln_free would deref a dead runtime -> SEGFAULT; leak instead.
  // Uses runtime_torn_down() (cheap), not runtime_available(), to avoid
  // get_device_count()'s GIL/Python side effects in the deleter at teardown.
  if (runtime_torn_down()) {
    RBLN_WARN_NOTHROW(
        "rbln_free skipped for {}: RBLN runtime unavailable; leaking rather than crashing", fmt::ptr(data));
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
  RtTimer _rt(RT_H2V);
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
  RBLN_CHECK(
      !::rbln::rbln_memcpy_h2v(src_host_ptr, dst_vaddr, size),
      "rbln_memcpy_h2v failed ({} bytes, dst vaddr={:#x}); the device may be busy or faulted",
      size,
      dst_vaddr);
}

void memcpy_v2h(void* cpu_dst_data, const void* rbln_src_data, size_t nbytes) {
  RtTimer _rt(RT_V2H);
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
  RBLN_CHECK(
      !::rbln::rbln_memcpy_v2h(src_vaddr, dst_host_ptr, size),
      "rbln_memcpy_v2h failed ({} bytes, src vaddr={:#x}); the device may be busy or faulted",
      size,
      src_vaddr);
}

void memcpy_v2v(void* rbln_dst_data, const void* rbln_src_data, size_t nbytes) {
  RtTimer _rt(RT_V2V);
  RBLN_LOG_DEBUG(
      "dst_rbln_data={}, src_rbln_data={}, nbytes={}", fmt::ptr(rbln_dst_data), fmt::ptr(rbln_src_data), nbytes);
  RBLN_CHECK(nbytes > 0, "nbytes must be positive, but got {}", nbytes);
  RBLN_CHECK(rbln_src_data != nullptr, "rbln_src_data cannot be nullptr");
  RBLN_CHECK(rbln_dst_data != nullptr, "rbln_dst_data cannot be nullptr");

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
  // No-op only during teardown (runtime may be unmapped); otherwise a missing device
  // -- no driver or no NPU -- throws via to_device_id() (torch.cuda.synchronize()
  // parity; see RBLNNoDeviceTest). No raw call is reached before that throw.
  if (runtime_shutting_down_.load(std::memory_order_relaxed)) {
    return;
  }
  check_device_index(device_index);
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
  RtTimer _rt(RT_V2V_MULTI);
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

BorrowedHostPtr borrow_host_ptr(const void* rbln_data, size_t nbytes) {
  RtTimer _rt(RT_BORROW);
  RBLN_LOG_DEBUG("rbln_data={}, nbytes={}", fmt::ptr(rbln_data), nbytes);
  RBLN_CHECK(rbln_data != nullptr, "rbln_data cannot be nullptr");
  RBLN_CHECK(nbytes > 0, "nbytes must be positive, but got {}", nbytes);

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
  RtTimer _rt(RT_BORROW);
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
  RtTimer _rt(RT_ACQUIRE);
  RBLN_LOG_DEBUG("rbln_data={}, nbytes={}", fmt::ptr(rbln_data), nbytes);
  RBLN_CHECK(rbln_data != nullptr, "rbln_data cannot be nullptr");
  RBLN_CHECK(nbytes > 0, "nbytes must be positive, but got {}", nbytes);

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
  RtTimer _rt(RT_ACQUIRE);
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
  RtTimer _rt(RT_RETURN);
  RBLN_LOG_DEBUG("borrow_id={}, updated={}", borrow_id, updated);
  RBLN_CHECK(
      !::rbln::rbln_v_return_borrowed(borrow_id, updated),
      "rbln_v_return_borrowed failed (borrow_id={}, updated={}); see rebel runtime logs for details",
      borrow_id,
      updated);
}

c10::CachingDeviceAllocator::DeviceStats get_device_stats(const c10::Device& device) {
  RBLN_LOG_DEBUG("logical device={}", c10::str(device));
  // Best-effort query: empty stats when the runtime is unavailable.
  if (!runtime_available()) {
    return c10::CachingDeviceAllocator::DeviceStats{};
  }
  const auto device_index = device.index();
  check_device_index(device_index);
  // CUDA parity (see memory_stats): report zero stats for a valid device this process
  // has not allocated on, matching PyTorch's generic path (empty only for an
  // uninitialized allocator). An initialized device still queries the runtime, so
  // genuine failures surface rather than being masked as zero.
  if (!device_context_initialized(device_index)) {
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
  // Two-level context gate (CUDA parity). Context flag FIRST: no allocator state anywhere
  // → no-op (a no-context parent or malformed config; nothing to free). Otherwise validate
  // the index (invalid throws) and skip a device never used here.
  if (!any_device_context_initialized() || !runtime_available()) {
    return;
  }
  const auto device_index = device.index();
  check_device_index(device_index);
  if (!device_context_initialized(device_index)) {
    return;
  }
  const auto device_id = to_device_id(device_index);
  RBLN_LOG_DEBUG("Calling rbln_empty_cache: device_id={}", device_id);
  // Live context: surface a genuine failure (CUDA parity — an initialized allocator
  // propagates real errors); rc included for diagnosis.
  const auto rc = rbln_empty_cache(device_id);
  RBLN_CHECK(
      !rc,
      "rbln_empty_cache failed for rbln:{} (rc={}); the device may be busy or in a faulted state",
      static_cast<int>(device_index),
      static_cast<int>(rc));
}

std::map<std::string, uint64_t> memory_stats(const c10::Device& device) {
  RBLN_LOG_DEBUG("logical device={}", c10::str(device));
  // Best-effort query: empty when the runtime is unavailable.
  if (!runtime_available()) {
    return {};
  }
  const auto device_index = device.index();
  check_device_index(device_index);
  // CUDA parity: report empty stats for a valid device this process has not allocated
  // on (memory_allocated()/memory_reserved() then read 0), matching PyTorch's generic
  // path, which returns empty only for an uninitialized allocator. An initialized
  // device still queries the runtime, so genuine failures surface -- device 0 works;
  // the runtime rejects per-node stats for a device index > 0 (INIT_INVALID_ARGUMENT,
  // Invalid node_id), and fixing that is a runtime-side change, not a swallow here.
  // (check_device_index above still rejects an invalid index -- a bad index throws.)
  if (!device_context_initialized(device_index)) {
    return {};
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
  // Two-level context gate (see empty_cache); context flag first.
  if (!any_device_context_initialized() || !runtime_available()) {
    return;
  }
  const auto device_index = device.index();
  check_device_index(device_index);
  if (!device_context_initialized(device_index)) {
    return;
  }
  const auto device_id = to_device_id(device_index);
  RBLN_LOG_DEBUG("Calling rbln_reset_accumulated_memory_stats: device_id={}", device_id);
  // Live context: surface a genuine failure (a silent success would mislead the caller).
  const auto rc = rbln_reset_accumulated_memory_stats(device_id);
  RBLN_CHECK(
      !rc,
      "rbln_reset_accumulated_memory_stats failed for rbln:{} (rc={})",
      static_cast<int>(device_index),
      static_cast<int>(rc));
}

void reset_peak_memory_stats(const c10::Device& device) {
  RBLN_LOG_DEBUG("logical device={}", c10::str(device));
  // Two-level context gate (see empty_cache); context flag first. This is also the only
  // guard for the generic torch.accelerator.reset_peak path (no Python init-guard upstream).
  if (!any_device_context_initialized() || !runtime_available()) {
    return;
  }
  const auto device_index = device.index();
  check_device_index(device_index);
  if (!device_context_initialized(device_index)) {
    return;
  }
  const auto device_id = to_device_id(device_index);
  RBLN_LOG_DEBUG("Calling rbln_reset_peak_memory_stats: device_id={}", device_id);
  // Live context: surface a genuine failure.
  const auto rc = rbln_reset_peak_memory_stats(device_id);
  RBLN_CHECK(
      !rc,
      "rbln_reset_peak_memory_stats failed for rbln:{} (rc={})",
      static_cast<int>(device_index),
      static_cast<int>(rc));
}

void set_file_offloading_enabled(bool enabled) {
  RBLN_LOG_DEBUG("Calling rbln_set_file_offloading_enabled: enabled={}", enabled);
  // Reachable without an allocation (torch.rbln.offload()), so gated directly:
  // best-effort no-op when the runtime is unavailable.
  if (!runtime_available()) {
    return;
  }
  RBLN_CHECK(
      !::rbln::rbln_set_file_offloading_enabled(enabled),
      "rbln_set_file_offloading_enabled failed (enabled={})",
      enabled);
}

uint64_t release_offload_temp_storage() {
  RBLN_LOG_DEBUG("Calling rbln_release_offload_temp_storage");
  // Shutdown-path call, so gated the same way as the offload toggle above.
  if (!runtime_available()) {
    return 0;
  }
  uint64_t num_files_removed = 0;
  RBLN_CHECK(
      !::rbln::rbln_release_offload_temp_storage(&num_files_removed), "rbln_release_offload_temp_storage failed");
  return num_files_removed;
}

} // namespace c10::rbln
