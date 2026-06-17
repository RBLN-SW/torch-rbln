#pragma once

#include <c10/core/Allocator.h>
#include <c10/rbln/RBLNMacros.h>

namespace c10::rbln {

/**
 * @brief Returns the host allocator backing CPU tensors created with
 * `pin_memory=True` while the RBLN backend is the active accelerator.
 *
 * The UMD has no DMA-mapped host allocation API yet, so this is page-aligned
 * host memory with best-effort `mlock`; swap in the UMD API here when it lands.
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
