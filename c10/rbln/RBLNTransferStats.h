#pragma once

#include <c10/rbln/RBLNMacros.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>

namespace c10::rbln {

/**
 * @brief Lightweight always-on counters for RBLN host/device transfers.
 *
 * These counters track per-call counts and per-call byte totals for v2v / v2h
 * / h2v / borrow_host_ptr (read & write). They are intended for diagnostic
 * use (perf debugging, fallback-coverage checks); the overhead is one atomic
 * increment per call.
 *
 * The counters live in a single thread-safe singleton — call ::snapshot() to
 * read all counters as a struct, ::reset() to zero them, and use the
 * ::bump_*() helpers from the transfer-functions to increment.
 *
 * Counters are also bumped for cpu_fallback dispatches so we can verify which
 * ops still hit the fallback path.
 */
struct TransferStatsSnapshot {
  uint64_t v2v_calls = 0;
  uint64_t v2v_bytes = 0;
  uint64_t v2h_calls = 0;
  uint64_t v2h_bytes = 0;
  uint64_t h2v_calls = 0;
  uint64_t h2v_bytes = 0;
  uint64_t borrow_r_calls = 0;
  uint64_t borrow_r_bytes = 0;
  uint64_t borrow_w_calls = 0;
  uint64_t borrow_w_bytes = 0;
  uint64_t fallback_dispatches = 0;
};

C10_RBLN_API TransferStatsSnapshot transfer_stats_snapshot();
C10_RBLN_API void transfer_stats_reset();

// Internal bump helpers — called from RBLNFunctions.cpp / RBLNLogging.cpp.
void bump_v2v(size_t nbytes);
void bump_v2h(size_t nbytes);
void bump_h2v(size_t nbytes);
void bump_borrow(size_t nbytes, bool for_overwrite);
void bump_fallback();

} // namespace c10::rbln
