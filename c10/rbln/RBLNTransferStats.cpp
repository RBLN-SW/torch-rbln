#include <c10/rbln/RBLNTransferStats.h>

#include <array>
#include <atomic>

namespace c10::rbln {

namespace {

// Single global counter set. Seven slots:
//  0: v2v calls  1: v2v bytes
//  2: v2h calls  3: v2h bytes
//  4: h2v calls  5: h2v bytes
//  6: fallback_dispatches
std::array<std::atomic<uint64_t>, 7>& counters() {
  static std::array<std::atomic<uint64_t>, 7> c{};
  return c;
}

inline void inc(int idx, uint64_t v = 1) {
  counters()[idx].fetch_add(v, std::memory_order_relaxed);
}

} // namespace

void bump_v2v(size_t nbytes) { inc(0); inc(1, nbytes); }
void bump_v2h(size_t nbytes) { inc(2); inc(3, nbytes); }
void bump_h2v(size_t nbytes) { inc(4); inc(5, nbytes); }

void bump_fallback() { inc(6); }

TransferStatsSnapshot transfer_stats_snapshot() {
  auto& c = counters();
  return {
      c[0].load(std::memory_order_relaxed), c[1].load(std::memory_order_relaxed),
      c[2].load(std::memory_order_relaxed), c[3].load(std::memory_order_relaxed),
      c[4].load(std::memory_order_relaxed), c[5].load(std::memory_order_relaxed),
      c[6].load(std::memory_order_relaxed),
  };
}

void transfer_stats_reset() {
  for (auto& v : counters()) v.store(0, std::memory_order_relaxed);
}

} // namespace c10::rbln
