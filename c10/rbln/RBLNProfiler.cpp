#include <c10/rbln/RBLNProfiler.h>

#include <array>
#include <atomic>

namespace c10::rbln::prof {
namespace {
// Relaxed atomics, written ONLY on cold branches (see header). No fences, no
// false-sharing concern: contention requires concurrent host bounces, which
// are themselves the slow thing being measured.
std::array<std::atomic<uint64_t>, kNumBounceSites> g_count{};
std::array<std::atomic<uint64_t>, kNumBounceSites> g_bytes{};
// (A) WHERE hook. null unless the higher layer installed it (= trace enabled).
std::atomic<BounceCaptureFn> g_bounce_capture_fn{nullptr};
} // namespace

void record_bounce(BounceSite site, uint64_t nbytes) noexcept {
  const int i = static_cast<int>(site);
  g_count[i].fetch_add(1, std::memory_order_relaxed);
  if (nbytes != 0) {
    g_bytes[i].fetch_add(nbytes, std::memory_order_relaxed);
  }
  // opt-in call-site capture: only fires when trace installed the hook.
  if (auto fn = g_bounce_capture_fn.load(std::memory_order_relaxed)) {
    fn(static_cast<uint8_t>(site));
  }
}

void set_bounce_capture_fn(BounceCaptureFn fn) noexcept {
  g_bounce_capture_fn.store(fn, std::memory_order_relaxed);
}

BounceSnapshot dump_bounces() noexcept {
  BounceSnapshot s{};
  for (int i = 0; i < kNumBounceSites; ++i) {
    s.count[i] = g_count[i].load(std::memory_order_relaxed);
    s.bytes[i] = g_bytes[i].load(std::memory_order_relaxed);
  }
  return s;
}

void reset_bounces() noexcept {
  for (int i = 0; i < kNumBounceSites; ++i) {
    g_count[i].store(0, std::memory_order_relaxed);
    g_bytes[i].store(0, std::memory_order_relaxed);
  }
}

} // namespace c10::rbln::prof
