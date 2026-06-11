#pragma once

#include <c10/core/ScalarType.h>

#include <array>

namespace c10::rbln {

// Single source of truth for the RBLN supported dtype catalog.
// Separate per-path arrays so extensions to `dispatch` (e.g., int dtypes)
// don't leak into `sdpa` or `amp`, which must stay float-only.
inline constexpr std::array<c10::ScalarType, 1> kDispatchDtypes = {c10::kHalf};
inline constexpr std::array<c10::ScalarType, 1> kSdpaDtypes = {c10::kHalf};
inline constexpr std::array<c10::ScalarType, 1> kAmpDtypes = {c10::kHalf};

constexpr bool is_dispatch_dtype(c10::ScalarType s) noexcept {
  for (const auto d : kDispatchDtypes) {
    if (s == d) {
      return true;
    }
  }
  return false;
}

} // namespace c10::rbln
