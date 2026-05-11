// fast path: aten::rsqrt.out for fp32 contiguous inputs.
//
// LLaMA RMSNorm calls ``rsqrt`` on the post-mean accumulator (fp32 due to
// the higher-precision compDtype used by the mean micro-kernel). Routing
// the op through TensorIterator + the boxed CPU dispatcher costs
// ~30-50 µs/call from framework overhead alone, dwarfing the actual
// rsqrt math on the buffer sizes seen by RMSNorm (~1k elements). This
// handler skips both and emits the rsqrt loop directly into the borrowed
// host buffer of the out tensor.
#include <ATen/core/Tensor.h>
#include <ATen/native/rbln/RBLNCPUFastPaths.h>
#include <c10/util/ArrayRef.h>
#include <torch/csrc/jit/runtime/operator.h>

#include <cmath>
#include <cstddef>

namespace at::native::rbln {
namespace {

void micro_rsqrt_fp32_contig(const float* __restrict__ in, float* __restrict__ out, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    out[i] = 1.0f / std::sqrt(in[i]);
  }
}

bool rsqrt_handler(
    c10::ArrayRef<at::Tensor> cpu_tensors,
    torch::jit::Stack* stack,
    size_t arguments_begin) {
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
  micro_rsqrt_fp32_contig(self.data_ptr<float>(), out.data_ptr<float>(), self.numel());
  stack->resize(arguments_begin);
  stack->emplace_back(c10::IValue(out));
  return true;
}

REGISTER_RBLN_CPU_FAST_PATH("aten::rsqrt.out", rsqrt_handler);

} // namespace
} // namespace at::native::rbln
