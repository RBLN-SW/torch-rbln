#include <ATen/ATen.h>
#include <c10/rbln/DeviceMappingManager.h>
#include <c10/rbln/RBLNFunctions.h>
#include <c10/rbln/RBLNCachingAllocator.h>
#include <c10/rbln/RBLNLogging.h>
#include <c10/rbln/RBLNProfiler.h>
#include <c10/rbln/RBLNRuntime.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <functional>
#include <map>
#include <set>
#include <mutex>
#include <vector>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>

namespace c10::rbln {

// rt-timing: time spent inside runtime copy calls, for the explain profiler's
// "runtime vs torch dispatch" split. Gated: when disabled each boundary call
// pays one relaxed atomic load (no clock read), so ON==OFF latency holds; an explain
// region flips it on for its duration. Index order MUST match kRtTimingN / the
// Python _RT_PRIMS tuple.
namespace {
enum RtIdx : std::uint8_t {
  RT_V2V = 0,
  RT_V2V_MULTI,
  RT_V2H,
  RT_H2V,
  RT_V2H_MULTI,
  RT_H2V_MULTI,
  RT_N
};
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

// --- Device-runtime liveness ------------------------------------------------
// The driver is loaded lazily and is absent on compile/CPU-only/CI hosts, and a call into a
// torn-down runtime at shutdown must not crash. Runtime-touching leaves gate on
// runtime_available().

namespace {

std::atomic<bool> runtime_shutting_down_{false}; // set at teardown via a Python atexit hook

bool driver_available() noexcept {
  return rt::Device::available();
}

// Torn down (shutting down or driver absent) vs "no device present" (reported by
// to_device_id()): teardown-safe ops no-op only on the former.
bool runtime_torn_down() noexcept {
  return runtime_shutting_down_.load(std::memory_order_relaxed) || !driver_available();
}

// Mandatory-op guard: clean throw, never a SEGFAULT.
void require_runtime(const char* op) {
  RBLN_CHECK(
      !runtime_shutting_down_.load(std::memory_order_relaxed), "Cannot {}: the RBLN runtime is shutting down.", op);
  RBLN_CHECK(
      is_dummy_device() || driver_available(),
      "Cannot {}: the RBLN driver is not loaded; install the RBLN driver and runtime.",
      op);
}

// c10::Error::what() appends "Exception raised from ..." plus a full C++ backtrace.
// Keep only the human-readable first line for warnings on nothrow paths.
std::string first_line(std::string_view text) {
  return std::string(text.substr(0, text.find('\n')));
}

} // namespace

c10::DeviceIndex get_device_count() {
  auto& manager = DeviceMappingManager::getInstance();
  const auto device_count = manager.getLogicalDeviceCount();
  RBLN_LOG_DEBUG("logical_device_count={}", static_cast<int>(device_count));
  return device_count;
}

DeviceProperties get_device_properties(c10::DeviceIndex device_index) {
  RBLN_CHECK(
      !is_dummy_device(),
      "get_device_properties() reports what the hardware says, and RBLN_DUMMY_DEVICE has no NPU behind it");

  const auto physical_device_ids = DeviceMappingManager::getInstance().getPhysicalDeviceIds(device_index);
  RBLN_CHECK(!physical_device_ids.empty(), "no NPU is mapped to rbln:{}", static_cast<int>(device_index));

  DeviceProperties properties;
  for (const int physical_device_id : physical_device_ids) {
    const auto npu = rt::Device::open(DeviceMappingManager::getInstance().systemNpu(physical_device_id))->properties();
    const uint64_t per_chiplet = npu.chiplets == 0 ? 0 : npu.memory / npu.chiplets;
    if (properties.npu_count == 0) {
      properties.name = npu.npu;
      properties.memory_per_chiplet = per_chiplet;
      properties.num_chiplet = npu.chiplets;
    } else {
      RBLN_CHECK(
          properties.name == npu.npu && properties.num_chiplet == npu.chiplets,
          "rbln:{} aggregates unlike NPUs: {} with {} chiplet(s) and {} with {}",
          static_cast<int>(device_index),
          properties.name,
          properties.num_chiplet,
          npu.npu,
          npu.chiplets);
    }
    properties.total_memory += npu.memory;
    properties.npu_count++;
  }
  RBLN_LOG_DEBUG(
      "rbln:{} name={} total_memory={} npu_count={}",
      static_cast<int>(device_index),
      properties.name,
      properties.total_memory,
      properties.npu_count);
  return properties;
}

c10::DeviceIndex get_physical_device_count() {
  if (is_dummy_device()) {
    // Dummy mode must not touch the driver (the host may have none); there is no physical
    // NPU, so report 0.
    return 0;
  }
  const auto physical_device_count =
      static_cast<c10::DeviceIndex>(DeviceMappingManager::getInstance().seenNpus().size());
  RBLN_LOG_DEBUG("physical_NPU_count={}", static_cast<int>(physical_device_count));
  return physical_device_count;
}

c10::DeviceIndex get_device_index() {
  // The selection is thread_local while the plan is process-wide, so an RBLN_* change before
  // the mapping commits can leave this thread pointing past the end of a rebuilt plan.
  // Report a device that exists; the selection is bookkeeping, so nothing has to unwind.
  // Checked on read because other threads' selections are unreachable from here.
  //
  // Nothrow enumeration, not getLogicalDeviceCount(): this backs
  // torch._C._accelerator_getDeviceIndex(), which torch calls from total predicates such as
  // reset_peak_memory_stats(), so a malformed RBLN_* config must not raise here. It maps a
  // failed plan to 0, and with no plan there is nothing to validate against.
  const auto device_count = get_device_count_nothrow();
  if (device_count > 0 && current_device_index_ >= device_count) {
    RBLN_LOG_DEBUG(
        "current logical device rbln:{} is outside the {} planned device(s); reporting rbln:0",
        static_cast<int>(current_device_index_),
        static_cast<int>(device_count));
    current_device_index_ = 0;
  }
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
  RBLN_CHECK(data != nullptr, "data cannot be nullptr");
  auto found = caching::try_locate_held(data);
  RBLN_CHECK(found.has_value(), "{} is not in live RBLN device memory", fmt::ptr(data));
  return found->location.device_index;
}

bool is_dummy_device() {
  // Runtime-free: reads the RBLN_DUMMY_DEVICE flag directly, not via
  // DeviceMappingManager (whose init would query the runtime). Cached.
  static const bool dummy = dummyDeviceEnabled();
  return dummy;
}
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
  //
  // A part-way commit failure is the same shape: the plan keeps its device count, but the
  // devices it did not claim can never be claimed, so every later device use rethrows.
  // hasFailedCommit() last: getInstance() runs the registered mapping-ready callback -- which
  // may reach back into Python -- without a catch, so the first touch of the singleton has to
  // happen inside get_device_count_nothrow()'s catch-all, not on this noexcept boundary.
  return !runtime_shutting_down_.load(std::memory_order_relaxed) && (is_dummy_device() || driver_available()) &&
      get_device_count_nothrow() > 0 && !DeviceMappingManager::getInstance().hasFailedCommit();
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
  to_device_id(device_index);
  static std::once_flag eager_malloc_warned;
  std::call_once(eager_malloc_warned, [] {
    if (std::getenv("TORCH_RBLN_EAGER_MALLOC") != nullptr) {
      RBLN_LOG_WARN("TORCH_RBLN_EAGER_MALLOC has no effect: device memory is allocated with the tensor");
    }
  });
  void* data = caching::allocate(device_index, nbytes);
  RBLN_LOG_DEBUG("data={}", fmt::ptr(data));
  mark_device_context_initialized(device_index); // this process now owns context on this device
  return data;
}

namespace {

// Runs `op` on the current stream of `device_index`, after what is queued on it.
void on_current_stream(c10::DeviceIndex device_index, const std::function<void()>& op) {
  runtime_stream(get_current_stream(device_index))->run(op);
}

Location located(const void* data, size_t nbytes, const char* what) {
  RBLN_CHECK(data != nullptr, "{} cannot be nullptr", what);
  auto location = caching::locate(data);
  RBLN_CHECK(
      nbytes <= location.available,
      "{} bytes from {} run past the end of its RBLN allocation ({} bytes left)",
      nbytes,
      fmt::ptr(data),
      location.available);
  return location;
}

} // namespace

void fill_zeros(void* rbln_data, size_t nbytes) {
  if (nbytes == 0) {
    return;
  }
  const auto dst = located(rbln_data, nbytes, "rbln_data");
  on_current_stream(dst.device_index, [&] { dst.buffer->device()->fill(*dst.buffer, dst.offset, nbytes, 0); });
}

void free(void* data) {
  RBLN_LOG_DEBUG("data={}", fmt::ptr(data));
  RBLN_CHECK(data != nullptr, "data cannot be nullptr");
  require_runtime("free device memory");
  caching::release(data);
}

void free_nothrow(void* data) noexcept {
  if (data == nullptr) {
    return;
  }
  // Torn-down runtime: releasing would reach a dead driver; leak instead.
  if (runtime_shutting_down_.load(std::memory_order_relaxed)) {
    RBLN_WARN_NOTHROW("free skipped for {}: the RBLN runtime is shutting down; leaking", fmt::ptr(data));
    return;
  }
  try {
    caching::release(data);
  } catch (const std::exception& e) {
    RBLN_WARN_NOTHROW("free failed for {}; leaking rather than aborting: {}", fmt::ptr(data), first_line(e.what()));
  }
}

void memcpy_h2v(void* rbln_dst_data, const void* cpu_src_data, size_t nbytes) {
  RtTimer _rt(RT_H2V);
  RBLN_LOG_DEBUG(
      "dst_rbln_data={}, src_cpu_data={}, nbytes={}", fmt::ptr(rbln_dst_data), fmt::ptr(cpu_src_data), nbytes);
  RBLN_CHECK(nbytes > 0, "nbytes must be positive, but got {}", nbytes);
  RBLN_CHECK(cpu_src_data != nullptr, "cpu_src_data cannot be nullptr");
  const auto dst = located(rbln_dst_data, nbytes, "rbln_dst_data");
  on_current_stream(
      dst.device_index, [&] { dst.buffer->device()->write(*dst.buffer, dst.offset, cpu_src_data, nbytes); });
}

void memcpy_v2h(void* cpu_dst_data, const void* rbln_src_data, size_t nbytes) {
  RtTimer _rt(RT_V2H);
  RBLN_LOG_DEBUG(
      "dst_cpu_data={}, src_rbln_data={}, nbytes={}", fmt::ptr(cpu_dst_data), fmt::ptr(rbln_src_data), nbytes);
  RBLN_CHECK(nbytes > 0, "nbytes must be positive, but got {}", nbytes);
  RBLN_CHECK(cpu_dst_data != nullptr, "cpu_dst_data cannot be nullptr");
  const auto src = located(rbln_src_data, nbytes, "rbln_src_data");
  on_current_stream(
      src.device_index, [&] { src.buffer->device()->read(*src.buffer, src.offset, cpu_dst_data, nbytes); });
}

namespace {

void copy_v2v(const Location& dst, const Location& src, size_t nbytes) {
  const auto& src_device = src.buffer->device();
  const auto& dst_device = dst.buffer->device();
  if (dst_device->sharesContext(*src_device)) {
    if (src.device_index != dst.device_index) {
      on_current_stream(src.device_index, [] {});
    }
    on_current_stream(dst.device_index, [&] {
      dst_device->copy(*dst.buffer, dst.offset, *src.buffer, src.offset, nbytes);
    });
    return;
  }
  // Devices of different contexts reach each other through the host.
  prof::record_bounce(prof::BounceSite::kRbln2RblnIndirect, nbytes);
  std::vector<uint8_t> host(nbytes);
  on_current_stream(src.device_index, [&] { src_device->read(*src.buffer, src.offset, host.data(), nbytes); });
  on_current_stream(dst.device_index, [&] { dst_device->write(*dst.buffer, dst.offset, host.data(), nbytes); });
}

} // namespace

void memcpy_v2v(void* rbln_dst_data, const void* rbln_src_data, size_t nbytes) {
  RtTimer _rt(RT_V2V);
  RBLN_LOG_DEBUG(
      "dst_rbln_data={}, src_rbln_data={}, nbytes={}", fmt::ptr(rbln_dst_data), fmt::ptr(rbln_src_data), nbytes);
  RBLN_CHECK(nbytes > 0, "nbytes must be positive, but got {}", nbytes);
  copy_v2v(located(rbln_dst_data, nbytes, "rbln_dst_data"), located(rbln_src_data, nbytes, "rbln_src_data"), nbytes);
}

// Asynchronous copies may run at once: each is ordered on the current stream and done on
// return, which every caller's contract allows.
void memcpy_h2v_async(void* rbln_dst_data, const void* cpu_src_data, size_t nbytes) {
  memcpy_h2v(rbln_dst_data, cpu_src_data, nbytes);
}

void memcpy_v2h_async(void* cpu_dst_data, const void* rbln_src_data, size_t nbytes) {
  memcpy_v2h(cpu_dst_data, rbln_src_data, nbytes);
}

void memcpy_v2v_async(void* rbln_dst_data, const void* rbln_src_data, size_t nbytes) {
  memcpy_v2v(rbln_dst_data, rbln_src_data, nbytes);
}

void synchronize(c10::DeviceIndex device_index) {
  RBLN_LOG_DEBUG("Synchronizing device {}", static_cast<int>(device_index));
  // No-op only during teardown; otherwise a missing device -- no driver or no NPU -- throws
  // via to_device_id() (torch.cuda.synchronize() parity; see RBLNNoDeviceTest).
  if (runtime_shutting_down_.load(std::memory_order_relaxed)) {
    return;
  }
  check_device_index(device_index);
  to_device_id(device_index);
  runtime_stream(get_default_stream(device_index));
  for (const auto& stream : runtime_streams(device_index)) {
    stream->synchronize();
  }
}

namespace {

// A device index of -1 means "current device" (torch passes it for an index-less
// device like `torch.Stream(device="rbln")`); resolve it up front.
c10::DeviceIndex resolve_device_index(c10::DeviceIndex device_index) {
  return device_index < 0 ? get_device_index() : device_index;
}

// Every stream torch hands out comes from this per-device pool: there is no destroy hook,
// so an unbounded create would leak and lengthen every device synchronize.
constexpr size_t kStreamPoolSize = 32;

thread_local std::map<c10::DeviceIndex, c10::StreamId> current_streams_;

struct EventEntry {
  c10::DeviceIndex device_index = -1;
  std::optional<rt::Event> event;
};

std::mutex events_mutex_;
std::map<uint64_t, EventEntry> events_;
uint64_t next_event_ = 1;

EventEntry event_entry(uint64_t event) {
  std::lock_guard<std::mutex> lock(events_mutex_);
  auto it = events_.find(event);
  RBLN_CHECK(it != events_.end(), "no RBLN event {:#x}", event);
  return it->second;
}

} // namespace

c10::Stream get_current_stream(c10::DeviceIndex device_index) {
  device_index = resolve_device_index(device_index);
  check_device_index(device_index);
  const auto it = current_streams_.find(device_index);
  const c10::StreamId id = it == current_streams_.end() ? 0 : it->second;
  return c10::Stream(c10::Stream::UNSAFE, c10::Device(c10::kPrivateUse1, device_index), id);
}

c10::Stream get_default_stream(c10::DeviceIndex device_index) {
  device_index = resolve_device_index(device_index);
  check_device_index(device_index);
  // StreamId 0 is the default stream.
  return c10::Stream(c10::Stream::DEFAULT, c10::Device(c10::kPrivateUse1, device_index));
}

c10::Stream get_stream_from_pool(c10::DeviceIndex device_index) {
  device_index = resolve_device_index(device_index);
  check_device_index(device_index);
  to_device_id(device_index);
  struct Pool {
    std::vector<c10::StreamId> ids;
    size_t next = 0;
  };
  static std::mutex pool_mutex;
  static std::map<c10::DeviceIndex, Pool> pools;
  std::lock_guard<std::mutex> lock(pool_mutex);
  auto& pool = pools[device_index];
  // next <= size, so next == size means this slot is still empty.
  if (pool.next == pool.ids.size()) {
    pool.ids.push_back(add_pool_stream(device_index));
    mark_device_context_initialized(device_index); // a stream implies a live context here
  }
  const auto id = pool.ids[pool.next];
  pool.next = (pool.next + 1) % kStreamPoolSize;
  return c10::Stream(c10::Stream::UNSAFE, c10::Device(c10::kPrivateUse1, device_index), id);
}

void set_current_stream(c10::Stream stream) {
  const auto device_index = stream.device_index();
  check_device_index(device_index);
  current_streams_[device_index] = stream.id();
}

bool query_stream(c10::Stream stream) {
  if (runtime_shutting_down_.load(std::memory_order_relaxed)) {
    return true; // nothing left to wait for
  }
  return runtime_stream(stream)->query();
}

void synchronize_stream(c10::Stream stream) {
  if (runtime_shutting_down_.load(std::memory_order_relaxed)) {
    return;
  }
  runtime_stream(stream)->synchronize();
}

uint64_t event_create(c10::DeviceIndex device_index) {
  device_index = resolve_device_index(device_index);
  check_device_index(device_index);
  std::lock_guard<std::mutex> lock(events_mutex_);
  // A null handle is torch's "not created yet" sentinel, so 0 is never a handle.
  const uint64_t event = next_event_++;
  events_[event] = EventEntry{device_index, std::nullopt};
  return event;
}

void event_destroy(uint64_t event) noexcept {
  // Called from ~Event, possibly during interpreter teardown, so it must never throw.
  if (event == 0) {
    return;
  }
  std::lock_guard<std::mutex> lock(events_mutex_);
  events_.erase(event);
}

void event_record(uint64_t event, c10::Stream stream) {
  if (runtime_shutting_down_.load(std::memory_order_relaxed)) {
    return;
  }
  auto recorded = runtime_stream(stream)->record();
  std::lock_guard<std::mutex> lock(events_mutex_);
  auto it = events_.find(event);
  RBLN_CHECK(it != events_.end(), "no RBLN event {:#x}", event);
  it->second = EventEntry{stream.device_index(), std::move(recorded)};
}

void event_block(c10::Stream stream, uint64_t event) {
  if (runtime_shutting_down_.load(std::memory_order_relaxed)) {
    return;
  }
  const auto entry = event_entry(event);
  if (!entry.event) {
    return; // never recorded: nothing to wait for
  }
  if (entry.device_index == stream.device_index()) {
    // Same device: does not block the host.
    runtime_stream(stream)->wait(*entry.event);
  } else {
    // A stream waits only on events of its own device; wait on the host instead -- correct
    // ordering, at the cost of serializing the host.
    entry.event->synchronize();
  }
}

bool event_query(uint64_t event) {
  if (runtime_shutting_down_.load(std::memory_order_relaxed)) {
    return true; // nothing left to wait for
  }
  const auto entry = event_entry(event);
  return !entry.event || entry.event->query();
}

void event_synchronize(uint64_t event) {
  if (runtime_shutting_down_.load(std::memory_order_relaxed)) {
    return;
  }
  const auto entry = event_entry(event);
  if (entry.event) {
    entry.event->synchronize();
  }
}

// A failure keeps the "<name> failed" message at::native::rbln::submit_or_fallback matches to
// route a rejected batch to its CPU fallback.
#define RBLN_BATCH(name, body)                                     \
  try {                                                            \
    body;                                                          \
  } catch (const std::exception& e) {                              \
    RBLN_CHECK(false, name " failed: {}", first_line(e.what()));   \
  }

namespace {

constexpr size_t kHostPage = 4096;

size_t page_rounded(size_t nbytes) {
  return (nbytes + kHostPage - 1) / kHostPage * kHostPage;
}

// Runs the copies `add` gathers as one job on the current stream of `device_index`, after
// what is queued there and on the current streams of the devices of `after`; returns once
// it is done.
void run_copies(
    c10::DeviceIndex device_index,
    const std::set<c10::DeviceIndex>& after,
    const std::function<void(rt::Device::Copies&)>& add) {
  for (const auto other : after) {
    if (other != device_index) {
      on_current_stream(other, [] {});
    }
  }
  auto device = runtime_device(device_index);
  rt::Device::Copies copies;
  add(copies);
  on_current_stream(device_index, [&] { device->copy(copies); });
}

void v2v_batch(const std::vector<V2VCopyOp>& copies) {
  struct Copy {
    Location dst;
    Location src;
    size_t nbytes;
  };
  std::map<c10::DeviceIndex, std::vector<Copy>> by_device;
  for (const auto& c : copies) {
    RBLN_CHECK(c.nbytes > 0, "nbytes must be positive, but got {}", c.nbytes);
    auto src = located(c.src, c.nbytes, "src");
    auto dst = located(c.dst, c.nbytes, "dst");
    if (!dst.buffer->device()->sharesContext(*src.buffer->device())) {
      copy_v2v(dst, src, c.nbytes);
      continue;
    }
    by_device[dst.device_index].push_back({std::move(dst), std::move(src), c.nbytes});
  }
  for (const auto& [device_index, group] : by_device) {
    std::set<c10::DeviceIndex> sources;
    for (const auto& c : group) {
      sources.insert(c.src.device_index);
    }
    run_copies(device_index, sources, [&](rt::Device::Copies& batch) {
      for (const auto& c : group) {
        batch.onDevice(*c.dst.buffer, c.dst.offset, *c.src.buffer, c.src.offset, c.nbytes);
      }
    });
  }
}

// Host sides go through one staging buffer, a page per entry at least, which the device
// copies from and to directly.
void h2v_batch(const std::vector<H2VCopyOp>& copies) {
  std::map<c10::DeviceIndex, std::vector<std::pair<Location, const H2VCopyOp*>>> by_device;
  for (const auto& c : copies) {
    RBLN_CHECK(c.nbytes > 0, "nbytes must be positive, but got {}", c.nbytes);
    RBLN_CHECK(c.src != nullptr, "src cannot be nullptr");
    auto dst = located(c.dst, c.nbytes, "dst");
    by_device[dst.device_index].emplace_back(std::move(dst), &c);
  }
  for (const auto& [device_index, group] : by_device) {
    size_t total = 0;
    for (const auto& [dst, c] : group) {
      total += page_rounded(c->nbytes);
    }
    auto staging = rt::HostBuffer::allocate(total);
    run_copies(device_index, {}, [&](rt::Device::Copies& batch) {
      size_t offset = 0;
      for (const auto& [dst, c] : group) {
        std::memcpy(staging->data() + offset, c->src, c->nbytes);
        batch.toDevice(*dst.buffer, dst.offset, staging->data() + offset, c->nbytes);
        offset += page_rounded(c->nbytes);
      }
    });
  }
}

void v2h_batch(const std::vector<V2HCopyOp>& copies) {
  std::map<c10::DeviceIndex, std::vector<std::pair<Location, const V2HCopyOp*>>> by_device;
  for (const auto& c : copies) {
    RBLN_CHECK(c.nbytes > 0, "nbytes must be positive, but got {}", c.nbytes);
    RBLN_CHECK(c.dst != nullptr, "dst cannot be nullptr");
    auto src = located(c.src, c.nbytes, "src");
    by_device[src.device_index].emplace_back(std::move(src), &c);
  }
  for (const auto& [device_index, group] : by_device) {
    size_t total = 0;
    for (const auto& [src, c] : group) {
      total += page_rounded(c->nbytes);
    }
    auto staging = rt::HostBuffer::allocate(total);
    run_copies(device_index, {}, [&](rt::Device::Copies& batch) {
      size_t offset = 0;
      for (const auto& [src, c] : group) {
        batch.toHost(staging->data() + offset, *src.buffer, src.offset, c->nbytes);
        offset += page_rounded(c->nbytes);
      }
    });
    size_t offset = 0;
    for (const auto& [src, c] : group) {
      std::memcpy(c->dst, staging->data() + offset, c->nbytes);
      offset += page_rounded(c->nbytes);
    }
  }
}

} // namespace

void memcpy_v2v_multi(const std::vector<V2VCopyOp>& copies) {
  RtTimer _rt(RT_V2V_MULTI);
  RBLN_BATCH("rbln_memcpy_v2v_multi", v2v_batch(copies))
}

void memcpy_h2v_multi(const std::vector<H2VCopyOp>& copies) {
  RtTimer _rt(RT_H2V_MULTI);
  RBLN_BATCH("rbln_memcpy_h2v_multi", h2v_batch(copies))
}

void memcpy_v2h_multi(const std::vector<V2HCopyOp>& copies) {
  RtTimer _rt(RT_V2H_MULTI);
  RBLN_BATCH("rbln_memcpy_v2h_multi", v2h_batch(copies))
}

#undef RBLN_BATCH

namespace {

// Whether an allocator query of `device` has anything to report (CUDA parity): nothing for a
// device this process never allocated on, or when the runtime is unavailable. An invalid
// index still throws.
bool has_allocator_state(const c10::Device& device) {
  if (!any_device_context_initialized() || !runtime_available()) {
    return false;
  }
  check_device_index(device.index());
  return device_context_initialized(device.index());
}

} // namespace

c10::CachingDeviceAllocator::DeviceStats get_device_stats(const c10::Device& device) {
  RBLN_LOG_DEBUG("logical device={}", c10::str(device));
  if (!has_allocator_state(device)) {
    return c10::CachingDeviceAllocator::DeviceStats{};
  }
  return caching::device_stats(device.index());
}

void empty_cache(const c10::Device& device) {
  RBLN_LOG_DEBUG("logical device={}", c10::str(device));
  if (has_allocator_state(device)) {
    caching::empty_cache(device.index());
  }
}

std::map<std::string, uint64_t> memory_stats(const c10::Device& device) {
  RBLN_LOG_DEBUG("logical device={}", c10::str(device));
  if (!has_allocator_state(device)) {
    return {};
  }
  return caching::stats_map(device.index());
}

std::map<std::string, uint64_t> memory_stats_per_chiplet(const c10::Device& device) {
  RBLN_LOG_DEBUG("logical device={}", c10::str(device));
  if (!has_allocator_state(device)) {
    return {};
  }
  // The allocator places every block on chiplet 0 of the device's one NPU.
  std::map<std::string, uint64_t> out;
  for (const auto& [key, value] : caching::stats_map(device.index())) {
    out["npu.0.chiplet.0." + key] = value;
  }
  return out;
}

namespace {

// The chiplets of each physical NPU of the logical device, in mapping order. Commits the
// mapping (a device use).
std::vector<std::vector<ChipletMemory>> device_memory_per_npu(const c10::Device& device) {
  RBLN_CHECK(
      runtime_available(), "Cannot query device memory for {}: no RBLN runtime or device available", c10::str(device));
  RBLN_CHECK(
      !is_dummy_device(),
      "Device memory info is not available for {}: RBLN_DUMMY_DEVICE has no NPU behind it",
      c10::str(device));
  const auto device_index = device.index();
  check_device_index(device_index);
  to_device_id(device_index);

  std::vector<std::vector<ChipletMemory>> npus;
  for (const int physical_id : DeviceMappingManager::getInstance().getPhysicalDeviceIds(device_index)) {
    const auto npu = rt::Device::open(DeviceMappingManager::getInstance().systemNpu(physical_id));
    std::vector<ChipletMemory> chiplets;
    for (uint32_t chiplet = 0; chiplet < npu->chiplets(); ++chiplet) {
      const auto info = npu->memoryInfo(chiplet);
      chiplets.push_back(ChipletMemory{info.total, info.free});
    }
    npus.push_back(std::move(chiplets));
  }
  return npus;
}

} // namespace

std::pair<size_t, size_t> mem_get_info(const c10::Device& device) {
  RBLN_LOG_DEBUG("logical device={}", c10::str(device));
  size_t free = 0;
  size_t total = 0;
  for (const auto& chiplets : device_memory_per_npu(device)) {
    for (const auto& chiplet : chiplets) {
      free += chiplet.free;
      total += chiplet.total;
    }
  }
  RBLN_LOG_DEBUG("mem_get_info: free={}, total={}", free, total);
  return {free, total};
}

std::map<std::string, uint64_t> mem_get_info_per_chiplet(const c10::Device& device) {
  RBLN_LOG_DEBUG("logical device={}", c10::str(device));
  return per_chiplet_memory_map(device_memory_per_npu(device));
}

std::map<std::string, uint64_t> per_chiplet_memory_map(const std::vector<std::vector<ChipletMemory>>& npus) {
  std::map<std::string, uint64_t> out;
  for (size_t npu = 0; npu < npus.size(); ++npu) {
    const auto npu_prefix = "npu." + std::to_string(npu) + ".";
    uint64_t total = 0;
    uint64_t free = 0;
    for (size_t chiplet = 0; chiplet < npus[npu].size(); ++chiplet) {
      const auto& c = npus[npu][chiplet];
      const auto prefix = npu_prefix + "chiplet." + std::to_string(chiplet) + ".";
      out[prefix + "total"] = c.total;
      out[prefix + "used"] = c.total - c.free;
      out[prefix + "free"] = c.free;
      total += c.total;
      free += c.free;
    }
    out[npu_prefix + "total"] = total;
    out[npu_prefix + "used"] = total - free;
    out[npu_prefix + "free"] = free;
  }
  return out;
}

void reset_accumulated_memory_stats(const c10::Device& device) {
  RBLN_LOG_DEBUG("logical device={}", c10::str(device));
  if (has_allocator_state(device)) {
    caching::reset_accumulated(device.index());
  }
}

void reset_peak_memory_stats(const c10::Device& device) {
  RBLN_LOG_DEBUG("logical device={}", c10::str(device));
  if (has_allocator_state(device)) {
    caching::reset_peak(device.index());
  }
}

} // namespace c10::rbln
