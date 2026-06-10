#pragma once

#include <c10/core/Allocator.h>
#include <c10/rbln/RBLNMacros.h>

namespace c10::rbln {

/**
 * @brief Returns the host allocator backing CPU tensors created with
 * `pin_memory=True` while the RBLN backend is the active accelerator.
 *
 * The RBLN UMD has no DMA-mapped host allocation API yet, so allocations are
 * page-aligned host memory page-locked via best-effort `mlock`. This keeps
 * the full `pin_memory` UX working with standard semantics; when the runtime
 * grows a pinned-host API, only this allocator changes.
 *
 * @return The pinned host allocator. Never null.
 */
C10_RBLN_API c10::Allocator* get_pinned_memory_allocator();

/**
 * @brief Checks whether `data` lies inside an allocation made by the pinned
 * host allocator. Never throws; null and foreign pointers return false.
 *
 * @param data A host pointer (any offset within an allocation is accepted).
 * @return True if the pointer is inside a pinned allocation.
 */
C10_RBLN_API bool is_pinned_ptr(const void* data);

} // namespace c10::rbln
