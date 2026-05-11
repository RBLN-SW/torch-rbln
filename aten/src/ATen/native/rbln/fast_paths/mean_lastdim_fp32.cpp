// fast path: aten::mean.out reducing the last dim, fp32 contiguous,
// keepdim=True, dtype=None.
//
// RMSNorm sums-of-squares along the hidden dim before rsqrt; that
// signature is fixed (``mean(x², dim=-1, keepdim=True)``). General
// ``mean.out`` runs a TensorIterator + reduction kernel; the
// last-dim-only contiguous case is a tight nested loop with a single
// scaling division. We accumulate in double to avoid an additional
// round of fp32 rounding for sums longer than ~1k elements (LLaMA hidden
// dims start at 2048 for 1B).
#include <ATen/core/Tensor.h>
#include <ATen/native/rbln/RBLNCPUFastPaths.h>
#include <c10/util/ArrayRef.h>
#include <torch/csrc/jit/runtime/operator.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace at::native::rbln {
namespace {

// out[r] = sum(in[r, :inner]) / inner, accumulated in double.
void micro_mean_lastdim_fp32_contig(
    const float* __restrict__ in,
    float* __restrict__ out,
    size_t outer,
    size_t inner) {
  const float inv = 1.0f / static_cast<float>(inner);
  for (size_t r = 0; r < outer; ++r) {
    const float* p = in + r * inner;
    double sum = 0.0;
    for (size_t i = 0; i < inner; ++i) {
      sum += p[i];
    }
    out[r] = static_cast<float>(sum * inv);
  }
}

bool mean_lastdim_handler(
    c10::ArrayRef<at::Tensor> cpu_tensors,
    torch::jit::Stack* stack,
    size_t arguments_begin) {
  // schema: mean.out(Tensor self, int[]? dim, bool keepdim=False,
  //                  ScalarType? dtype=None, *, Tensor(a!) out)
  // stack at arguments_begin: [self, dim, keepdim, dtype, out]
  if (cpu_tensors.size() < 2) {
    return false;
  }
  const auto& self = cpu_tensors[0];
  const auto& out = cpu_tensors.back();
  if (!self.defined() || !self.is_contiguous() || self.scalar_type() != at::kFloat) {
    return false;
  }
  if (!out.defined() || !out.is_contiguous() || out.scalar_type() != at::kFloat) {
    return false;
  }

  const auto& dim_iv = (*stack)[arguments_begin + 1];
  const auto& keepdim_iv = (*stack)[arguments_begin + 2];
  const auto& dtype_iv = (*stack)[arguments_begin + 3];
  if (!dim_iv.isIntList() || !keepdim_iv.isBool() || !keepdim_iv.toBool() || !dtype_iv.isNone()) {
    return false;
  }
  const auto dim_list = dim_iv.toIntVector();
  if (dim_list.size() != 1) {
    return false;
  }
  const int64_t self_dim = self.dim();
  if (self_dim < 1) {
    return false;
  }
  int64_t dim = dim_list[0];
  if (dim < 0) {
    dim += self_dim;
  }
  if (dim != self_dim - 1) {
    return false;
  }
  const int64_t inner = self.size(self_dim - 1);
  if (inner <= 0) {
    return false;
  }
  const int64_t outer = self.numel() / inner;
  if (out.numel() != outer) {
    return false;
  }

  micro_mean_lastdim_fp32_contig(
      self.data_ptr<float>(),
      out.data_ptr<float>(),
      static_cast<size_t>(outer),
      static_cast<size_t>(inner));
  stack->resize(arguments_begin);
  stack->emplace_back(c10::IValue(out));
  return true;
}

REGISTER_RBLN_CPU_FAST_PATH("aten::mean.out", mean_lastdim_handler);

} // namespace
} // namespace at::native::rbln
