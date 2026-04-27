#pragma once

// Minimal forward declarations for the subset of rebel-compiler's v-mem API we
// use from cpu_fallback_rbln. The full header
// `<rebel/torch/rbln_vmem_api.h>` transitively pulls in VMemoryManager and
// vmemmodel enums, which drag in absl — headers we don't want to depend on
// from torch-rbln. Live symbols are in librbln_runtime.so / librbln.so.
//
// Source of truth:
//   /home/chanheo/rebel_compiler/rebel/include/rebel/torch/rbln_vmem_api.h

#include <rebel/common/status.h>

#include <cstdint>

namespace rebel {
namespace torch {

// Borrow a host pointer into the rbln virtual memory at `vaddr`. Triggers a
// device→host sync if the device view is currently authoritative; allocates
// host backing if none exists. The borrow MUST be released via
// `rbln_v_return_borrowed` with the returned `borrow_id_out`.
//
// `write_only=true` skips the device→host transfer even when the entry is
// physical-latest. Callers using this MUST overwrite the entire borrowed
// region — any unwritten bytes will contain stale host data. State transitions
// to USER_VIEW_IS_LATEST on return.
::rbln::Status rbln_v_borrow_host_ptr(uint64_t vaddr, uint64_t size,
                                      uintptr_t& host_ptr_out, uint64_t& borrow_id_out,
                                      bool write_only = false);

// Release a previously borrowed host pointer. If `updated` is true, marks the
// host view as the latest source of truth; the next device consumer performs
// a lazy host→device copy.
::rbln::Status rbln_v_return_borrowed(uint64_t borrow_id, bool updated);

} // namespace torch
} // namespace rebel
