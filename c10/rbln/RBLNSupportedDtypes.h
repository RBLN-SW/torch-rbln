#pragma once

#include <c10/core/ScalarType.h>
#include <c10/rbln/RBLNMacros.h>
#include <c10/util/ArrayRef.h>

#include <array>

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
// capability. A dtype becomes resident when a device op produces it, and eager
// dispatch sends exactly the kDispatchDtypes to the device, so this is that set
// (measured in test_accelerator_contract.py).
inline constexpr auto kCapabilityDtypes = kDispatchDtypes;

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
// `TORCH_RBLN_DISPATCH_DTYPES` names extra dtypes whose tensors the eager shim
// dispatches to the RBLN compile path instead of the CPU fallback. Python
// parses it once at `import torch_rbln` and passes the result here through
// `set_dispatch_extension`, so the Python gate and the C++ shim always see the
// same set. Empty (default) leaves the catalog as is.
//
// This is opt-in on purpose: the device may compute these dtypes in a reduced
// or saturating form (see docs/EAGER_DISPATCH.md), so it is worth it only when
// the caller does not rely on the host semantics of that dtype.
//
// The dtypes the extension may admit. Every floating one needs a NaN/Inf
// scanner in DispatchShim.cpp (scanner_for).
inline constexpr std::array<c10::ScalarType, 7> kDispatchExtensionDtypes = {
    c10::kDouble,
    c10::kInt,
    c10::kLong,
    c10::kShort,
    c10::kChar,
    c10::kByte,
    c10::kBool};

// Replace the extension. Each dtype must be in kDispatchExtensionDtypes;
// otherwise this throws and the previous extension stays in place.
C10_RBLN_API void set_dispatch_extension(c10::ArrayRef<c10::ScalarType> dtypes);
// Catalog or extension.
C10_RBLN_API bool is_dispatch_dtype_rt(c10::ScalarType s) noexcept;

} // namespace c10::rbln
