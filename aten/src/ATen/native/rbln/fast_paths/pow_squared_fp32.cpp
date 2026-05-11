// fast path: aten::pow.Tensor_Scalar_out for exponent == 2 on fp32
// contiguous inputs.
//
// RMSNorm's first step (``x.pow(2).mean(...)``) hits this hot path on the
// fp32-promoted accumulator. The general ``pow`` kernel runs through
// std::pow plus type-promotion checks; for ``exp == 2`` a single
// multiplication is roughly an order of magnitude faster. We only commit
// to the fast path when the exponent value compares bit-equal to 2, which
// covers all observed RMSNorm call sites.
#include <ATen/core/Tensor.h>
#include <ATen/native/rbln/RBLNCPUFastPaths.h>
#include <c10/util/ArrayRef.h>
#include <torch/csrc/jit/runtime/operator.h>

#include <cstddef>

namespace at::native::rbln {
namespace {

void micro_square_fp32_contig(const float* __restrict__ in, float* __restrict__ out, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    out[i] = in[i] * in[i];
  }
}

bool pow_squared_handler(
    c10::ArrayRef<at::Tensor> cpu_tensors,
    torch::jit::Stack* stack,
    size_t arguments_begin) {
  // schema: pow.Tensor_Scalar_out(Tensor self, Scalar exponent, *, Tensor(a!) out)
  // stack args at arguments_begin: [self, exponent, out]
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
  if (out.numel() != self.numel() || out.sizes() != self.sizes()) {
    return false;
  }

  const auto& exp_iv = (*stack)[arguments_begin + 1];
  if (!exp_iv.isScalar()) {
    return false;
  }
  const auto exp_s = exp_iv.toScalar();
  const bool is_two = (exp_s.isFloatingPoint() && exp_s.toDouble() == 2.0) ||
                      (exp_s.isIntegral(false) && exp_s.toLong() == 2);
  if (!is_two) {
    return false;
  }

  micro_square_fp32_contig(self.data_ptr<float>(), out.data_ptr<float>(), self.numel());
  stack->resize(arguments_begin);
  stack->emplace_back(c10::IValue(out));
  return true;
}

REGISTER_RBLN_CPU_FAST_PATH("aten::pow.Tensor_Scalar_out", pow_squared_handler);

} // namespace
} // namespace at::native::rbln
