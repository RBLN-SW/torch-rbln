#pragma once

#include <c10/core/ScalarType.h>
#include <c10/rbln/RBLNMacros.h>

#include <array>
#include <vector>

namespace c10::rbln {

// Single source of truth for the RBLN supported dtype catalog.
// Separate per-path arrays so extensions to `dispatch` (e.g., int dtypes)
// don't leak into `sdpa` or `amp`, which must stay float-only.
// float32: the device may cast fp32 ops to a custom float format, so an fp32
// op dispatched here can run in reduced precision.
inline constexpr std::array<c10::ScalarType, 3> kDispatchDtypes = {
    c10::kHalf,
    c10::kBFloat16,
    c10::kFloat};
inline constexpr std::array<c10::ScalarType, 2> kSdpaDtypes = {
    c10::kHalf,
    c10::kBFloat16};
// AMP autocast is not implemented yet: no AutocastPrivateUse1 cast policy is
// registered, so dispatching an op under torch.autocast("rbln") would hit a
// missing kernel (NotImplementedError). Advertise an empty set so PyTorch
// disables autocast with a warning instead of crashing. Restore the float
// dtypes once AutocastPrivateUse1 is registered.
inline constexpr std::array<c10::ScalarType, 0> kAmpDtypes = {};

// torch.accelerator device-capability dtypes (backs get_device_capability):
// dtypes actually resident in RBLN device memory. Other dtypes accept
// device="rbln" but are CPU-backed (0 device bytes), so they are not a device
// capability. Empirically fp16/bf16, and fp32 since it is in kDispatchDtypes
// (verified in test_accelerator_contract.py).
inline constexpr std::array<c10::ScalarType, 3> kCapabilityDtypes = {
    c10::kHalf,
    c10::kBFloat16,
    c10::kFloat};

constexpr bool is_dispatch_dtype(c10::ScalarType s) noexcept {
  for (const auto d : kDispatchDtypes) {
    if (s == d) {
      return true;
    }
  }
  return false;
}

// Runtime extension of the eager-dispatch catalog.
//
// `TORCH_RBLN_DISPATCH_DTYPES` is a comma-separated list of extra dtypes (torch
// names or the usual aliases: float64, int32, int64, int16, int8, uint8, bool) whose
// tensors the eager shim dispatches to the RBLN compile path instead of the CPU
// fallback. Unset (default) leaves the catalog as is.
//
// This is opt-in on purpose: the device may compute these dtypes in a reduced or
// saturating form (see docs/EAGER_DISPATCH.md), so it is worth it only when the caller
// does not rely on the host semantics of that dtype. The value is read live
// from the environment (re-parsed only when the string changes), like the other
// TORCH_RBLN_* runtime gates in DispatchShim.cpp. An unknown dtype name throws, so
// these are not noexcept: the error has to reach the caller as a Python exception.
C10_RBLN_API bool is_dispatch_dtype_rt(c10::ScalarType s);
// Catalog + environment extension, in catalog order then env order (deduplicated).
C10_RBLN_API std::vector<c10::ScalarType> dispatch_dtypes_rt();

} // namespace c10::rbln
