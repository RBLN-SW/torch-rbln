#ifndef REBEL_RUNTIME_API_RBLN_KINETO_API_H
#define REBEL_RUNTIME_API_RBLN_KINETO_API_H

// C-ABI for delivering one rbln profiler session to an external consumer. It maps
// onto the Perfetto / chrome-trace model: a Device is a "process" row, a Lane a
// "thread" sub-row, and Slices are the timed bars laid on a lane. Returned as
// flat device/lane/slice arrays linked by id.

#include <rebel/runtime/api/rbln_retcode.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// One device row in the trace.
typedef struct {
  int32_t pid;       // unique id of this device row
  const char* name;  // valid only during the sink call
} RblnKinetoDevice;

// One lane within a device row.
typedef struct {
  int32_t device_pid;    // pid of the owning device row
  int32_t resource_tid;  // unique id of this lane within the device row
  const char* name;      // valid only during the sink call
} RblnKinetoLane;

// A key/value annotation attached to a slice.
typedef struct {
  const char* name;         // valid only during the sink call
  const char* value;        // valid only during the sink call
  int32_t value_is_quoted;  // 1 if `value` is a string literal, 0 if raw (number/json)
} RblnKinetoAnnotation;

// Coarse activity class, set by the producer (which owns the category vocabulary).
// The consumer maps these to its own activity types and needs no knowledge of the
// rbln category strings; `categories` (below) are display-only.
typedef enum {
  RBLN_KINETO_KIND_RUNTIME = 0,  // default; host-runtime-like
  RBLN_KINETO_KIND_COMPUTE,      // neural-engine / cluster compute
  RBLN_KINETO_KIND_DMA,          // DMA copy engines
  RBLN_KINETO_KIND_SYNC,         // device sync / nop
} RblnKinetoActivityKind;

// One slice (a single timed activity on a lane).
//
// start_steady_ns/end_steady_ns are absolute steady_clock (monotonic) nanoseconds,
// not wall-clock/system time.
typedef struct {
  int32_t device_pid;             // pid of the owning device row
  int32_t resource_tid;           // tid of the owning lane
  const char* name;               // valid only during the sink call
  const char* const* categories;  // display-only strings; classify via `kind`
  uint32_t categories_count;
  int64_t start_steady_ns;
  int64_t end_steady_ns;
  const RblnKinetoAnnotation* annotations;  // array of `annotations_count`
  uint32_t annotations_count;
  int64_t corr_id;              // correlation id, or 0 if unset
  RblnKinetoActivityKind kind;  // activity class (categories are display-only)
} RblnKinetoSlice;

// The full result of one profiling session, delivered to the sink in one call.
// Every pointer below (and every string it reaches) is valid ONLY for the
// duration of the sink call; copy anything that must outlive it.
typedef struct {
  const RblnKinetoDevice* devices;  // sorted by pid
  uint32_t devices_count;
  const RblnKinetoLane* lanes;
  uint32_t lanes_count;
  const RblnKinetoSlice* slices;
  uint32_t slices_count;
} RblnKinetoExport;

// Callback invoked once per session with the fully materialized export. Every
// pointer it receives is valid only until it returns.
typedef void (*RblnKinetoExportSink)(const RblnKinetoExport* exp, void* user_data);

/**
 * @brief Reports whether the rbln profiler runtime is currently active.
 *
 * @param active_out [out] Set to 1 if profiling is active, 0 otherwise.
 *
 * @return 0 on success, or an error code on failure.
 */
RBLNRetCode rbln_kineto_is_active(int32_t* active_out);

/**
 * @brief Begins one profiling session.
 *
 * @details Discards any data captured before this call and starts a fresh
 * capture. Trace output is written under the directory named by the
 * RBLN_PROFILER_DIR environment variable.
 *
 * @return 0 on success, or an error code on failure.
 */
RBLNRetCode rbln_kineto_begin_session(void);

/**
 * @brief Ends one profiling session and delivers its result.
 *
 * @details Invokes `sink` exactly once, synchronously, with the materialized
 * export before returning; all pointers passed to `sink` are valid only during
 * that call. If the session captured nothing, `sink` is not called and
 * `*exported_out` is set to 0; otherwise it is set to 1. The session is reset for
 * reuse afterward.
 *
 * @param sink [in] Callback that receives the export.
 * @param user_data [in] Opaque pointer passed through to `sink`.
 * @param exported_out [out] Set to 1 if `sink` was invoked, 0 otherwise.
 *
 * @return 0 on success, or an error code on failure.
 */
RBLNRetCode rbln_kineto_end_session_and_export(RblnKinetoExportSink sink, void* user_data,
                                               int32_t* exported_out);

#ifdef __cplusplus
}
#endif

#endif  // REBEL_RUNTIME_API_RBLN_KINETO_API_H
