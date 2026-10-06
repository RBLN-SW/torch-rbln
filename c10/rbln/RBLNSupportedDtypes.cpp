#include <c10/rbln/RBLNSupportedDtypes.h>
#include <rbln/runtime/precision.h>

namespace c10::rbln {

bool dispatches(c10::ScalarType s) {
  return is_dispatch_dtype(s) ||
      (s == c10::kFloat && ::rbln::runtime::float32Precision() == ::rbln::runtime::Float32Precision::kDevice);
}

} // namespace c10::rbln
