#pragma once

#include <c10/core/ScalarType.h>

#include <array>

namespace c10::rbln {

// Single source of truth for the RBLN supported dtype catalog.
// Separate per-path arrays so extensions to `dispatch` (e.g., int dtypes)
// don't leak into `sdpa` or `amp`, which must stay float-only.
inline constexpr std::array<c10::ScalarType, 2> kDispatchDtypes = {c10::kHalf, c10::kBFloat16};
inline constexpr std::array<c10::ScalarType, 2> kSdpaDtypes = {c10::kHalf, c10::kBFloat16};
// AMP autocast is not implemented yet: no AutocastPrivateUse1 cast policy is
// registered, so dispatching an op under torch.autocast("rbln") would hit a
// missing kernel (NotImplementedError). Advertise an empty set so torch
// disables autocast with a warning instead of crashing. Restore the float
// dtypes once AutocastPrivateUse1 is registered.
inline constexpr std::array<c10::ScalarType, 0> kAmpDtypes = {};

// torch.accelerator device-capability dtypes: the set RBLN can *allocate* and
// *type-convert* on device. Per the upstream contract this is independent of
// which ops dispatch natively, so it is broader than kDispatchDtypes and
// includes fp32/int32/int64. Must match the v2v/copy engine set
// (ENGINE_DTYPES in test/utils_v2v.py).
inline constexpr std::array<c10::ScalarType, 5> kCapabilityDtypes = {
    c10::kHalf, c10::kBFloat16, c10::kFloat, c10::kInt, c10::kLong};

constexpr bool is_dispatch_dtype(c10::ScalarType s) noexcept {
  for (const auto d : kDispatchDtypes) {
    if (s == d) {
      return true;
    }
  }
  return false;
}

} // namespace c10::rbln
