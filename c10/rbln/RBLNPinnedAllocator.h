#pragma once

#include <c10/core/Allocator.h>
#include <c10/rbln/RBLNMacros.h>

#include <cstdint>

namespace c10::rbln {

/**
 * @brief Returns the host allocator backing CPU tensors created with
 * `pin_memory=True` while the RBLN backend is the active accelerator.
 *
 * Page-aligned host memory (2 MiB-aligned and THP-advised from 2 MiB up), page-locked
 * best effort, and -- the part that makes it "pinned" to the device -- registered with
 * the runtime through `rbln_host_register` on every RBLN device this process has
 * initialized. A registered buffer is addressed by its device VA in every later
 * host<->device copy, so the kernel reuses one pin instead of pinning the pages on each
 * command buffer (the RBLN counterpart of cudaHostRegister). Registration is best
 * effort: on a runtime without it (UMD < 3.5, RBLN_HOST_MEMORY_REGISTER=0) the tensor is
 * still pinned in the torch sense and copies take their usual path.
 *
 * A device initialized after the allocation picks the buffer up lazily: the copy
 * entry points call ensure_pinned_registered() before handing the pointer to the
 * runtime.
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

/**
 * @brief Registers the pinned allocation containing `data` with the runtime for
 * `torch_device_id`, if it is pinned and not registered there yet. No-op for a
 * pageable pointer, a device already covered, or a runtime without registration.
 * Never throws: registration is an optimization, the copy that follows is correct
 * either way.
 *
 * Called by the host<->device copy entry points with the device the copy targets,
 * so an allocation made before that device was initialized still gets its pin.
 *
 * @param data A host pointer (any offset within the allocation).
 * @param torch_device_id The runtime device id the copy targets.
 */
C10_RBLN_API void ensure_pinned_registered(const void* data, int torch_device_id) noexcept;

/**
 * @brief Whether the pinned allocation containing `data` is registered with the runtime
 * for `torch_device_id`. Diagnostic; false for anything that is not pinned.
 */
C10_RBLN_API bool pinned_ptr_registered_on(const void* data, int torch_device_id) noexcept;

} // namespace c10::rbln
