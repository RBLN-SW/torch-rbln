#include <c10/rbln/RBLNSupportedDtypes.h>
#include <rebel/v2/runtime/precision.h>

namespace c10::rbln {

bool dispatches(c10::ScalarType s) {
  return is_dispatch_dtype(s) ||
      (s == c10::kFloat && ::rebel::v2::runtime::float32Precision() == ::rebel::v2::runtime::Float32Precision::kDevice);
}

} // namespace c10::rbln
