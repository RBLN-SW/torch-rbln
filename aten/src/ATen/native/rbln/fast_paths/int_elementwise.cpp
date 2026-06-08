// Fast paths for the integer metadata arithmetic that the fp16-only RBLN
// device cannot run natively and must fall back to CPU: sub.out / mul.out /
// clamp.out on small int16/int32/int64 tensors.
//
// These show up every decode step in the attention metadata builder
// (partition-size math: ``cs - pidx * partition_len``, ``clamp(.., 0, len)``)
// and in ``logits_indices = query_start_loc[1:] - 1``. The generic
// ``redispatchBoxed(CPU)`` path pays boxing + TensorIterator setup that
// dominates the cost for these tiny tensors; a direct host loop skips it.
//
// Correctness: we compute in int64 and store in the output dtype, so mixed
// int dtypes from scalar type-promotion (e.g. int32 * <int64 scalar>) are
// handled. Any shape/dtype we do not explicitly support bails to the boxed
// fallback (returns false), so results are never wrong -- only un-accelerated.
#include <ATen/core/Tensor.h>
#include <ATen/native/rbln/RBLNCPUFastPaths.h>
#include <c10/util/ArrayRef.h>
#include <torch/csrc/jit/runtime/operator.h>

#include <cstddef>
#include <cstdint>

namespace at::native::rbln {
namespace {

inline bool is_int_contig(const at::Tensor& t) {
  if (!t.defined() || !t.is_contiguous()) {
    return false;
  }
  const auto st = t.scalar_type();
  return st == at::kShort || st == at::kInt || st == at::kLong;
}

inline int64_t load_i64(const void* base, at::ScalarType st, int64_t i) {
  switch (st) {
    case at::kShort:
      return static_cast<const int16_t*>(base)[i];
    case at::kInt:
      return static_cast<const int32_t*>(base)[i];
    case at::kLong:
      return static_cast<const int64_t*>(base)[i];
    default:
      return 0; // unreachable: guarded by is_int_contig
  }
}

inline void store_i64(void* base, at::ScalarType st, int64_t i, int64_t v) {
  switch (st) {
    case at::kShort:
      static_cast<int16_t*>(base)[i] = static_cast<int16_t>(v);
      break;
    case at::kInt:
      static_cast<int32_t*>(base)[i] = static_cast<int32_t>(v);
      break;
    case at::kLong:
      static_cast<int64_t*>(base)[i] = static_cast<int64_t>(v);
      break;
    default:
      break; // unreachable
  }
}

// Broadcast mode of ``other`` against ``self`` for an element-wise binary op.
//   0 = same shape, 1 = scalar (other.numel()==1),
//   2 = last-dim broadcast (other is 1-D of size self.size(-1)),
//  -1 = unsupported -> bail.
inline int broadcast_mode(const at::Tensor& self, const at::Tensor& other) {
  if (other.numel() == 1) {
    return 1;
  }
  if (other.sizes() == self.sizes()) {
    return 0;
  }
  if (self.dim() >= 1 && other.dim() == 1 && other.numel() == self.size(self.dim() - 1)) {
    return 2;
  }
  return -1;
}

// other-index for element i under the given broadcast mode.
inline int64_t other_index(int mode, int64_t i, int64_t inner) {
  switch (mode) {
    case 1:
      return 0;
    case 2:
      return i % inner;
    default:
      return i;
  }
}

template <typename BinOp>
bool run_binary(
    const at::Tensor& self,
    const at::Tensor& other,
    const at::Tensor& out,
    torch::jit::Stack* stack,
    size_t arguments_begin,
    BinOp op) {
  if (!is_int_contig(self) || !is_int_contig(other) || !is_int_contig(out)) {
    return false;
  }
  if (out.sizes() != self.sizes() || out.numel() != self.numel()) {
    return false;
  }
  const int mode = broadcast_mode(self, other);
  if (mode < 0) {
    return false;
  }

  const auto sdt = self.scalar_type();
  const auto odt = other.scalar_type();
  const auto rdt = out.scalar_type();
  const void* sp = self.data_ptr();
  const void* op_ = other.data_ptr();
  void* rp = out.data_ptr();
  const int64_t n = self.numel();
  const int64_t inner = self.dim() >= 1 ? self.size(self.dim() - 1) : 1;

  for (int64_t i = 0; i < n; ++i) {
    const int64_t a = load_i64(sp, sdt, i);
    const int64_t b = load_i64(op_, odt, other_index(mode, i, inner));
    store_i64(rp, rdt, i, op(a, b));
  }

  stack->resize(arguments_begin);
  stack->emplace_back(c10::IValue(out));
  return true;
}

// schema: sub.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out)
bool sub_int_handler(c10::ArrayRef<at::Tensor> cpu_tensors, torch::jit::Stack* stack, size_t arguments_begin) {
  if (cpu_tensors.size() < 3) {
    return false;
  }
  // alpha lives on the stack right after self, other.
  const auto& alpha_iv = (*stack)[arguments_begin + 2];
  if (!alpha_iv.isScalar()) {
    return false;
  }
  const auto alpha = alpha_iv.toScalar();
  if (!(alpha.isIntegral(/*includeBool=*/false) && alpha.toLong() == 1)) {
    return false;
  }
  return run_binary(
      cpu_tensors[0], cpu_tensors[1], cpu_tensors.back(), stack, arguments_begin, [](int64_t a, int64_t b) {
        return a - b;
      });
}

// schema: mul.out(Tensor self, Tensor other, *, Tensor(a!) out)
bool mul_int_handler(c10::ArrayRef<at::Tensor> cpu_tensors, torch::jit::Stack* stack, size_t arguments_begin) {
  if (cpu_tensors.size() < 3) {
    return false;
  }
  return run_binary(
      cpu_tensors[0], cpu_tensors[1], cpu_tensors.back(), stack, arguments_begin, [](int64_t a, int64_t b) {
        return a * b;
      });
}

// schema: clamp.out(Tensor self, Scalar? min=None, Scalar? max=None, *, Tensor(a!) out)
bool clamp_int_handler(c10::ArrayRef<at::Tensor> cpu_tensors, torch::jit::Stack* stack, size_t arguments_begin) {
  if (cpu_tensors.size() < 2) {
    return false;
  }
  const auto& self = cpu_tensors[0];
  const auto& out = cpu_tensors.back();
  if (!is_int_contig(self) || !is_int_contig(out)) {
    return false;
  }
  if (out.sizes() != self.sizes() || out.numel() != self.numel()) {
    return false;
  }

  const auto& min_iv = (*stack)[arguments_begin + 1];
  const auto& max_iv = (*stack)[arguments_begin + 2];
  bool has_min = false;
  bool has_max = false;
  int64_t lo = 0;
  int64_t hi = 0;
  if (!min_iv.isNone()) {
    if (!min_iv.isScalar() || !min_iv.toScalar().isIntegral(/*includeBool=*/false)) {
      return false;
    }
    has_min = true;
    lo = min_iv.toScalar().toLong();
  }
  if (!max_iv.isNone()) {
    if (!max_iv.isScalar() || !max_iv.toScalar().isIntegral(/*includeBool=*/false)) {
      return false;
    }
    has_max = true;
    hi = max_iv.toScalar().toLong();
  }
  if (!has_min && !has_max) {
    return false;
  }

  const auto sdt = self.scalar_type();
  const auto rdt = out.scalar_type();
  const void* sp = self.data_ptr();
  void* rp = out.data_ptr();
  const int64_t n = self.numel();
  for (int64_t i = 0; i < n; ++i) {
    int64_t v = load_i64(sp, sdt, i);
    if (has_min && v < lo) {
      v = lo;
    }
    if (has_max && v > hi) {
      v = hi;
    }
    store_i64(rp, rdt, i, v);
  }

  stack->resize(arguments_begin);
  stack->emplace_back(c10::IValue(out));
  return true;
}

REGISTER_RBLN_CPU_FAST_PATH("aten::sub.out", sub_int_handler);
REGISTER_RBLN_CPU_FAST_PATH("aten::mul.out", mul_int_handler);
REGISTER_RBLN_CPU_FAST_PATH("aten::clamp.out", clamp_int_handler);

} // namespace
} // namespace at::native::rbln
