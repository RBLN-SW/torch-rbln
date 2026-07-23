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

// torch.accelerator device-capability dtypes (backs get_device_capability):
// dtypes actually resident in RBLN device memory. Other dtypes accept
// device="rbln" but are CPU-backed (0 device bytes), so they are not a device
// capability. Empirically only fp16/bf16 (verified in test_accelerator_contract.py).
inline constexpr std::array<c10::ScalarType, 2> kCapabilityDtypes = {c10::kHalf, c10::kBFloat16};

constexpr bool is_dispatch_dtype(c10::ScalarType s) noexcept {
  for (const auto d : kDispatchDtypes) {
    if (s == d) {
      return true;
    }
  }
  return false;
}

} // namespace c10::rbln
