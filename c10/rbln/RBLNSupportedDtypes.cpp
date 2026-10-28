#include <c10/rbln/RBLNSupportedDtypes.h>
#include <c10/util/Exception.h>

#include <algorithm>
#include <atomic>
#include <cstdint>

namespace c10::rbln {

namespace {

// Bit i set: c10::ScalarType i is admitted by the extension. Written once at
// import (set_dispatch_extension), read on every shim dispatch.
std::atomic<uint64_t> g_dispatch_extension_mask{0};

static_assert(
    static_cast<unsigned>(c10::ScalarType::NumOptions) <= 64,
    "the dispatch extension mask holds one bit per ScalarType");

} // namespace

void set_dispatch_extension(c10::ArrayRef<c10::ScalarType> dtypes) {
  uint64_t mask = 0;
  for (const auto s : dtypes) {
    TORCH_CHECK(
        std::ranges::find(kDispatchExtensionDtypes, s) !=
            kDispatchExtensionDtypes.end(),
        "the dispatch extension cannot admit ",
        c10::toString(s));
    mask |= uint64_t{1} << static_cast<unsigned>(s);
  }
  g_dispatch_extension_mask.store(mask, std::memory_order_relaxed);
}

bool is_dispatch_dtype_rt(c10::ScalarType s) noexcept {
  if (is_dispatch_dtype(s)) {
    return true;
  }
  const auto idx = static_cast<unsigned>(s);
  return idx < 64 &&
      ((g_dispatch_extension_mask.load(std::memory_order_relaxed) >> idx) &
       uint64_t{1});
}

} // namespace c10::rbln
