#ifndef REBEL_VMEMORY_PROFILER_STATS_H
#define REBEL_VMEMORY_PROFILER_STATS_H

#include <cstdint>

// Process-global profiler counters for the rebel runtime. Every counter is
// recorded ONLY at a point that is already slow (a device alloc/free, a v2v
// host-sync slow path) — never on the fast device path — so recording is
// unmeasurable against the work that point already does, and reads are lazy (at
// report() time).
//
// Scope: only signals a user cannot otherwise see AND can act on. Generic
// dispatch/utilization counters (command-stream count, host-wait / device-idle,
// total host-traffic bytes) are intentionally NOT here — they are a different
// profiler's job (torch.profiler / nsys) and, on the executed path, live in the
// TVM runtime where they read ~0, so surfacing them would mislead.
//
// Kept in a dedicated header so wiring counters does not recompile every TU that
// includes vmemory_manager.h.
namespace rbln {

// --- v2v slow-path CAUSE breakdown (COUNT only). A device v2v copy the caller
// issued that the runtime had to service via a host round-trip — the hidden
// overhead torch-rbln's own counters are structurally blind to. Attributed by
// src state: reason 0 = src not device-resident; reason 1 = src device-only
// (a real device->host transfer); reason 2 = src synced (host memcpy, no
// transfer). bytes[] is retained for ABI but is not accumulated (always 0).
void prof_get_v2v_hidden_d2h(uint64_t* counts, uint64_t* bytes, uint32_t n);
uint32_t prof_v2v_hidden_num_reasons();
void prof_reset_v2v_hidden_d2h();

// --- hidden device->host transfers in the host-sync (v2h) path. COUNT of REAL
// physical->user transfers the manager actually performed (every
// TransferDataFromPhysicalToUser). Lets a profiler separate a host bounce that
// truly crossed the device from one served entirely on the host (host-resident
// src -> memcpy, NO transfer): when the torch-side copy-path bounce count is > 0
// but this counter's delta is 0, no device crossing happened. The manager EMITS
// this (it alone knows real-vs-served); the per-tensor sync_state stays hidden.
// Recorded only on the already-slow real-transfer path. COUNT only.
void prof_record_host_sync_d2h(uint64_t bytes);
void prof_get_host_sync_d2h(uint64_t* count, uint64_t* bytes);
void prof_reset_host_sync_d2h();

// --- hidden host->device transfers (symmetric to d2h above). COUNT of REAL
// host->device transfers the manager actually performed: the lazy push at the
// device-consume boundary (EnsureSyncedOnPhysicalView -> DoRblnHostToDeviceCopy),
// the fast host->virtual plan's main copy, and the direct DoMemcopy(kHost2Device).
// A host-served write (host-resident dst -> memcpy, NO transfer) reaches none of
// these. The manager EMITS this (it alone knows real-vs-served); the per-tensor
// sync_state stays hidden. Recorded only on the real-transfer path. COUNT only.
void prof_record_host_sync_h2d(uint64_t bytes);
void prof_get_host_sync_h2d(uint64_t* count, uint64_t* bytes);
void prof_reset_host_sync_h2d();

// --- (E) device memory gauge: current live device bytes + high-water peak.
// Recorded at every BufferAllocator alloc/free (the single device-allocation
// chokepoint), so the gauge is complete. This is a RESOURCE gauge, not a
// hidden-overhead signal; it is kept because it is accurate and actionable
// (OOM / high-water), and read as a level (no reset).
void prof_record_device_alloc(uint64_t bytes);
void prof_record_device_free(uint64_t bytes);
void prof_get_memory(uint64_t* current_bytes, uint64_t* peak_bytes);

}  // namespace rbln

#endif  // REBEL_VMEMORY_PROFILER_STATS_H
