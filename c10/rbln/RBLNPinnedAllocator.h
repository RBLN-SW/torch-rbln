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
 * @brief Registers a caller-owned host buffer as pinned: the RBLN counterpart of
 * cudaHostRegister for memory this allocator did not hand out (a shared-memory pool, a
 * cache's slab). The range enters the same registry as allocator memory, so is_pinned_ptr()
 * reports it (non_blocking copies go async) and it is registered with every initialized
 * device now and with any later device on its first copy. Overlapping a live range is an
 * error; the caller keeps ownership of the memory and must unregister before freeing it.
 *
 * Runtime registration is best effort as everywhere else: without it (UMD < 3.5) the range
 * is still "pinned" for torch's purposes and copies take their usual path.
 *
 * @param data Start of the buffer (page alignment is not required for the pin itself, but
 *   only page-aligned copy operands take the device-VA path).
 * @param nbytes Length in bytes; must be positive.
 * @throws c10::Error on a null pointer, zero length, or overlap with a registered range.
 */
C10_RBLN_API void register_host_memory(void* data, size_t nbytes);

/**
 * @brief Reverses register_host_memory(). `data` must be the exact start passed to it.
 * Unregisters from every device the range was registered with (the runtime drains the
 * device's pending transfers first).
 *
 * @throws c10::Error if `data` is not a live external registration.
 */
C10_RBLN_API void unregister_host_memory(void* data);

/**
 * @brief Whether the pinned allocation containing `data` is registered with the runtime
 * for `torch_device_id`. Diagnostic; false for anything that is not pinned.
 */
C10_RBLN_API bool pinned_ptr_registered_on(const void* data, int torch_device_id) noexcept;

} // namespace c10::rbln
