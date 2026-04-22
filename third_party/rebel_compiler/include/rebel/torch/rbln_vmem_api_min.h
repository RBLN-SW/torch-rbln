#ifndef REBEL_TORCH_RBLN_VMEM_API_MIN_H
#define REBEL_TORCH_RBLN_VMEM_API_MIN_H

// Minimal shim for the subset of rebel's internal vmem API we need to drive
// host-backed virtual memory. Full header
// (/home/chanheo/rebel_compiler/rebel/include/rebel/torch/rbln_vmem_api.h)
// pulls in absl and nlohmann via TensorDtype; we dodge those by declaring only
// the entry points we call. Symbols are provided by librbln.so.

#include <cstdint>

namespace rbln {

// ABI-compatible stand-in for the real rbln::Status (which holds a
// std::unique_ptr<State>). Because we cannot see the full State definition
// here, we store the pointer as opaque and skip destruction — the real Status
// owns the allocation, so we leak an error state on the rare non-OK return.
// For the borrow/return callsites we only inspect IsOK() on success paths.
class State;

class [[nodiscard]] Status final {
 public:
  Status() noexcept : state_(nullptr) {}
  Status(Status&& other) noexcept : state_(other.state_) { other.state_ = nullptr; }
  Status& operator=(Status&& other) noexcept {
    if (this != &other) {
      state_ = other.state_;
      other.state_ = nullptr;
    }
    return *this;
  }
  // Non-trivial dtor keeps the Itanium ABI equivalent to the real Status
  // (which is non-trivially destructible via unique_ptr). We deliberately do
  // not delete state_ — the memory ownership is internal to librbln; the
  // wrapper only ever looks at IsOK() after a borrow/return call.
  ~Status() {}
  Status(const Status&) = delete;
  Status& operator=(const Status&) = delete;

  bool IsOK() const { return state_ == nullptr; }

 private:
  State* state_;
};

} // namespace rbln

namespace rebel::torch {

// Borrow a host pointer backing the given virtual memory region. The returned
// host_ptr is valid until rbln_v_return_borrowed is called with borrow_id_out.
rbln::Status rbln_v_borrow_host_ptr(uint64_t vaddr, uint64_t size,
                                    uintptr_t& host_ptr_out,
                                    uint64_t& borrow_id_out);

// Return a borrowed host pointer. updated=true marks the host side as the
// latest truth so rebel will propagate to device on next read.
rbln::Status rbln_v_return_borrowed(uint64_t borrow_id, bool updated);

// Mark the v-memory at key_vaddr as logically zero-initialised. No host
// memory is allocated and no device transfer occurs. On the next device
// read, zeros are transferred via a temporary buffer. On the next device
// write, the transfer is skipped entirely (PHYSICAL_VIEW_IS_LATEST).
// Used as the "about to be fully overwritten" hint so a subsequent
// rbln_v_borrow_host_ptr doesn't trigger a useless d->h sync of the
// previous (and irrelevant) contents.
rbln::Status rbln_v_mark_zeros(uint64_t key_vaddr);

} // namespace rebel::torch

#endif // REBEL_TORCH_RBLN_VMEM_API_MIN_H
