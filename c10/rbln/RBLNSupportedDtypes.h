#pragma once

#include <c10/core/ScalarType.h>

#include <array>

namespace c10::rbln {

// Single source of truth for the RBLN supported dtype catalog.
// Separate per-path arrays so extensions to `dispatch` (e.g., int dtypes)
// don't leak into `sdpa` or `amp`, which must stay float-only.
inline constexpr std::array<c10::ScalarType, 2> kDispatchDtypes = {c10::kHalf, c10::kBFloat16};
inline constexpr std::array<c10::ScalarType, 2> kSdpaDtypes = {c10::kHalf, c10::kBFloat16};
inline constexpr std::array<c10::ScalarType, 2> kAmpDtypes = {c10::kHalf, c10::kBFloat16};

constexpr bool is_dispatch_dtype(c10::ScalarType s) noexcept {
  for (const auto d : kDispatchDtypes) {
    if (s == d) {
      return true;
    }
  }
  return false;
}

} // namespace c10::rbln
