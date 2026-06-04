#ifndef REBEL_VMEMORY_PROFILER_STATS_H
#define REBEL_VMEMORY_PROFILER_STATS_H

#include <cstdint>

// Process-global profiler counters for the rebel runtime. All counters are
// recorded ONLY at points that are already slow (a physical DMA, a job submit,
// a host-blocking wait, a device alloc/free) — never on the fast device-to-
// device path — so reading them costs nothing and recording is unmeasurable
// against the work that point already does. Reads are lazy (at report() time).
// Kept in a dedicated header so wiring counters does not recompile every TU
// that includes vmemory_manager.h.
namespace rbln {

// --- v2v slow-path CAUSE breakdown (COUNT only). The d2h *bytes* of the v2v
// slow path are owned by the d2h leaf below (prof_record_d2h); this counter
// only attributes the cause by src state so there is no byte double-count.
// reason 0 = src not device-resident; reason 1 = src device-only (real d2h);
// reason 2 = src synced (host memcpy, no transfer). bytes[] is retained for ABI
// but is no longer accumulated (always 0).
void prof_get_v2v_hidden_d2h(uint64_t* counts, uint64_t* bytes, uint32_t n);
uint32_t prof_v2v_hidden_num_reasons();
void prof_reset_v2v_hidden_d2h();

// --- Leaf-level host<->device byte/count totals. Recorded EXACTLY ONCE at the
// physical DMA leaf (DoRblnHostToDeviceCopy / DoRblnDeviceToHostCopy), so each
// transfer is counted once regardless of which op/caller triggered it.
void prof_record_h2v(uint64_t bytes);
void prof_record_d2h(uint64_t bytes);

// --- (D) command-stream submissions; (C) cumulative host time blocked on the
// device (steady_clock around the already-blocking WaitJob). region_wall minus
// host_wait_ns approximates host-side / device-idle time in the sync model.
void prof_record_cs_dispatch();
void prof_record_host_wait_ns(uint64_t ns);

// --- (E) device memory gauge: current live device bytes + high-water peak.
void prof_record_device_alloc(uint64_t bytes);
void prof_record_device_free(uint64_t bytes);

// Lazy readout. Fills out[0..n) in this fixed order (n must be >= 8):
//   0 h2v_bytes   1 h2v_count   2 d2h_bytes   3 d2h_count
//   4 cs_count    5 host_wait_ns 6 mem_current_bytes  7 mem_peak_bytes
void prof_get_scalars(uint64_t* out, uint32_t n);
void prof_reset_scalars();

}  // namespace rbln

#endif  // REBEL_VMEMORY_PROFILER_STATS_H
