#pragma once
// RBLNProfiler — always-on, near-zero-cost counters for HIDDEN torch-rbln
// overhead (host round-trips and fallbacks a user did not ask for and cannot
// see from their PyTorch code).
//
// SCOPE DECISION (deliberate — see profiler design discussion):
// A signal lives here, in the always-on verdict, ONLY if it is BOTH
//   (1) recorded strictly inside an ALREADY-slow branch (host bounce / staging
//       alloc / exception catch), so a single relaxed fetch_add is unmeasurable
//       against the µs–ms transfer that branch already paid — i.e. ON == OFF; and
//   (2) ACTIONABLE toward "am I using torch-rbln well / did my optimization
//       land" — a number the user can actually change by editing their code.
//
// Intentionally NOT here:
//   * cpu_fallback / recompile / warm-hit counts — already tracked by the
//     g_diag_* dispatch counters in DispatchShim.cpp; read those, don't dup.
//   * command-stream submission count, structural alignment/dtype padding —
//     measurable but NOT user-actionable, so they are NOISE for the verdict.
//     If ever exposed they belong in a runtime-owned RAW tier, off the verdict.
//   * device-residency (hidden d2h), idle-time, memory gauges — these are the
//     runtime's truth and are added later from rebel-compiler, never polled
//     from here.
// "Measurable" does not mean "worth showing": the verdict carries only what
// changes the user's behavior; everything uncertain stays raw until real
// workloads prove it matters.

#include <array>
#include <cstdint>

#include <c10/rbln/RBLNMacros.h>

namespace c10::rbln::prof {

// One class of hidden host-bounce / fallback incident. Each is a cold branch
// confirmed (file:line) to already pay a host round-trip or staging alloc.
enum class BounceSite : uint8_t {
  kRbln2RblnIndirect = 0, // copy_: non-direct v2v -> v2h + h2v host bounce   (RBLNCopy.cpp)
  kCpu2RblnStaging, // copy_: cpu src staged via at::empty + cpu copy   (RBLNCopy.cpp)
  kCpu2RblnNoncontigDst, // copy_: non-contig rbln dst pulled to host + h2v   (RBLNCopy.cpp)
  kStridedV2VFallback, // strided_v2v_copy -> dst.copy_(src.cpu()) bounce   (RBLNStridedV2V.cpp)
  kV2VBatchToPerEntry, // batched memcpy_v2v_multi rejected -> per-entry     (RBLNV2VBatch.cpp)
  kHostBatchToPerEntry, // batched h2v/v2h_multi rejected -> per-entry       (RBLNHostBatch.cpp)
  kNumBounceSites,
};

// Record one incident. Call ONLY after the slow branch is entered. nbytes may
// be 0 when the incident is not a byte transfer (e.g. batch->per-entry: the
// fallback itself is the signal, bytes are attributed by the per-entry copies).
C10_RBLN_API void record_bounce(BounceSite site, uint64_t nbytes) noexcept;

// (A) WHERE for bounces — opt-in Python call-site capture. c10 has no Python, so
// the higher layer (torch_rbln DispatchShim, which holds pybind) installs a capture
// hook; record_bounce invokes it ONLY when installed (= trace enabled). OFF by
// default -> one relaxed null-load on the already-slow bounce branch, ON == OFF.
// The hook receives the BounceSite value (as uint8_t to keep this header Python-free).
using BounceCaptureFn = void (*)(uint8_t site) noexcept;
C10_RBLN_API void set_bounce_capture_fn(BounceCaptureFn fn) noexcept;

inline constexpr int kNumBounceSites = static_cast<int>(BounceSite::kNumBounceSites);

// Per-site snapshot for readout (torch_rbln.profiler.dump()). Lazy: only read
// at report() time, never during the run.
struct BounceSnapshot {
  std::array<uint64_t, kNumBounceSites> count;
  std::array<uint64_t, kNumBounceSites> bytes;
};
C10_RBLN_API BounceSnapshot dump_bounces() noexcept;
C10_RBLN_API void reset_bounces() noexcept;

} // namespace c10::rbln::prof
