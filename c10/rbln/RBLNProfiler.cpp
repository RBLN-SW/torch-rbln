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
} // namespace

void record_bounce(BounceSite site, uint64_t nbytes) noexcept {
  const int i = static_cast<int>(site);
  g_count[i].fetch_add(1, std::memory_order_relaxed);
  if (nbytes != 0) {
    g_bytes[i].fetch_add(nbytes, std::memory_order_relaxed);
  }
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
