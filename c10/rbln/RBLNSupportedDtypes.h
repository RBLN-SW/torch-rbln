#pragma once

#include <c10/core/ScalarType.h>
#include <c10/rbln/RBLNMacros.h>

#include <array>
#include <vector>

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

// Runtime extension of the eager-dispatch catalog.
//
// `TORCH_RBLN_DISPATCH_DTYPES` is a comma-separated list of extra dtypes (torch
// names or the usual aliases: float32|fp32, float64, int32, int64, int16, int8,
// uint8, bool) whose tensors the eager shim dispatches to the RBLN compile path
// instead of the CPU fallback. Unset (default) leaves the catalog as is.
//
// This is opt-in on purpose: the device holds every float as DLFloat16
// (1-6-9), so a float32 op dispatched to the device computes in reduced
// precision. It is worth it when the surrounding graph already produces its
// float32 tensors on the device (a device function returns float32 as dlf16
// physical bytes) and the alternative is a host round trip per op; it is not
// worth it when the caller relies on float32 precision. The value is read live
// from the environment (re-parsed only when the string changes), like the other
// TORCH_RBLN_* runtime gates in DispatchShim.cpp.
C10_RBLN_API bool is_dispatch_dtype_rt(c10::ScalarType s) noexcept;
// Catalog + environment extension, in catalog order then env order (deduplicated).
C10_RBLN_API std::vector<c10::ScalarType> dispatch_dtypes_rt();

// Strict dispatch: for the dtypes named in `TORCH_RBLN_DISPATCH_STRICT` (comma-separated
// dtype names, or "all" = every dtype in the catalog + extension) the eager shim does NOT
// take the *performance* fallbacks — the align-penalty routing of unaligned last dims to
// the CPU fallback here, and the 64-alignment fallback in the Python compile path. The op
// goes through the compile path as a first-class device op even when the compiler has to
// wrap it in host pad/depad, so the true device-path cost of that dtype/shape is what the
// caller measures. Safety fallbacks (tracer, dispatch mode, mixed devices, dtype mismatch,
// NaN/Inf scan, reentrancy) are unaffected. Ops without a Python wrapper (the explicit
// CPU-fallback registrations) are unaffected too; they show up in the fallback log.
C10_RBLN_API bool is_strict_dispatch_dtype_rt(c10::ScalarType s) noexcept;
C10_RBLN_API std::vector<c10::ScalarType> strict_dispatch_dtypes_rt();

} // namespace c10::rbln
