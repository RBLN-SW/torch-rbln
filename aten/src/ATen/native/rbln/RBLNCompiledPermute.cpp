#include <ATen/native/rbln/RBLNCompiledPermute.h>
#include <c10/util/Exception.h>

#include <algorithm>
#include <map>
#include <numeric>

namespace at::native::rbln {

namespace {

CompiledPermuteImpl g_impl = nullptr;

// A program binds its operands once and rebinding costs ~2 ms for a 117 MB buffer, so a
// caller that alternates buffers (a double-buffered pipeline) needs one program per buffer.
// Slots are handed out per source address, capped so an unbounded caller cannot grow the
// cache without limit.
constexpr int64_t kMaxSlots = 8;

int64_t slot_for(const void* src) {
  static thread_local std::map<const void*, int64_t> slots;
  auto it = slots.find(src);
  if (it == slots.end()) {
    it = slots.emplace(src, static_cast<int64_t>(slots.size()) % kMaxSlots).first;
  }
  return it->second;
}

// Below this a strided copy is cheap enough that a compiled program (and its one-time
// compilation) is not worth it.
constexpr int64_t kMinBytes = int64_t{16} << 20;

// `src` is a pure permutation of a contiguous tensor exactly when ordering its dimensions by
// descending stride yields a contiguous view. Returns that ordering's inverse -- the dims a
// permute of the contiguous base needs to reproduce `src` -- or an empty vector.
std::vector<int64_t> permutation_of_contiguous(const at::Tensor& src) {
  const int64_t dim = src.dim();
  std::vector<int64_t> order(dim);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) { return src.stride(a) > src.stride(b); });
  if (!src.permute(order).is_contiguous())
    return {};
  std::vector<int64_t> dims(dim);
  for (int64_t i = 0; i < dim; ++i)
    dims[order[i]] = i;
  return dims;
}

} // namespace

void set_compiled_permute_impl(CompiledPermuteImpl impl) {
  g_impl = impl;
}

bool try_compiled_permute_copy(const at::Tensor& dst, const at::Tensor& src) {
  if (g_impl == nullptr)
    return false;
  if (!src.device().is_privateuseone() || dst.device() != src.device())
    return false;
  if (dst.scalar_type() != src.scalar_type() || !dst.is_contiguous())
    return false;
  if (src.is_contiguous() || src.storage_offset() != 0)
    return false;
  if (src.numel() * src.element_size() < kMinBytes)
    return false;
  const std::vector<int64_t> dims = permutation_of_contiguous(src);
  if (dims.empty())
    return false;
  std::vector<int64_t> order(dims.size());
  for (size_t i = 0; i < dims.size(); ++i)
    order[dims[i]] = static_cast<int64_t>(i);
  try {
    // A program the compiler miscomputes throws; the strided walk is always right.
    g_impl(src.permute(order).contiguous(), dims, slot_for(src.data_ptr()), dst);
  } catch (const std::exception&) {
    return false;
  }
  return true;
}

} // namespace at::native::rbln
