#include <ATen/native/rbln/RBLNRepeat.h>

#include <ATen/core/Tensor.h>
#include <ATen/ops/empty.h>
#include <c10/core/ScalarType.h>
#include <c10/rbln/RBLNLogging.h>

#include <cstdint>
#include <cstring>
#include <vector>

namespace at::native::rbln {

at::Tensor repeat_interleave_Tensor_rbln(const at::Tensor& repeats, std::optional<c10::SymInt> output_size) {
  RBLN_SCOPE_GUARD();

  RBLN_CHECK(repeats.dim() == 1, "repeat_interleave: repeats must be 1-D, got {}-D", repeats.dim());
  RBLN_CHECK(
      repeats.scalar_type() == at::kLong || repeats.scalar_type() == at::kInt,
      "repeat_interleave: repeats dtype must be int32 or int64, got {}",
      c10::str(repeats.scalar_type()));

  // Pull repeats to host as int64.
  at::Tensor host = repeats;
  if (!host.device().is_cpu())
    host = host.cpu();
  if (!host.is_contiguous())
    host = host.contiguous();

  const int64_t n = host.numel();
  std::vector<int64_t> rep(n);
  if (host.scalar_type() == at::kLong) {
    if (n > 0) {
      std::memcpy(rep.data(), host.data_ptr<int64_t>(), n * sizeof(int64_t));
    }
  } else {
    const auto* src = host.data_ptr<int32_t>();
    for (int64_t i = 0; i < n; ++i)
      rep[i] = static_cast<int64_t>(src[i]);
  }

  // Compute total length. Reject negatives.
  int64_t total = 0;
  for (int64_t r : rep) {
    RBLN_CHECK(r >= 0, "repeat_interleave: repeats must be non-negative, got {}", r);
    total += r;
  }

  // Validate `output_size` hint when provided.
  if (output_size.has_value()) {
    const int64_t expected = output_size->expect_int();
    RBLN_CHECK(expected == total, "repeat_interleave: output_size={} doesn't match sum(repeats)={}", expected, total);
  }

  // Build the output index list on host. Upstream preserves the input
  // dtype, so a int32 `repeats` produces a int32 output.
  const auto out_dtype = repeats.scalar_type();
  auto cpu_out = at::empty(
      {total}, at::TensorOptions().dtype(out_dtype).device(c10::kCPU).memory_format(c10::MemoryFormat::Contiguous));
  if (total > 0) {
    if (out_dtype == at::kLong) {
      auto* out_ptr = cpu_out.data_ptr<int64_t>();
      int64_t pos = 0;
      for (int64_t i = 0; i < n; ++i) {
        const int64_t r = rep[i];
        for (int64_t k = 0; k < r; ++k) {
          out_ptr[pos++] = i;
        }
      }
    } else {
      auto* out_ptr = cpu_out.data_ptr<int32_t>();
      int64_t pos = 0;
      for (int64_t i = 0; i < n; ++i) {
        const int64_t r = rep[i];
        for (int64_t k = 0; k < r; ++k) {
          out_ptr[pos++] = static_cast<int32_t>(i);
        }
      }
    }
  }

  // Land the result on the same device as `repeats`. For an RBLN-resident
  // `repeats` input this is one h2v of a (typically small) index list; the
  // composite `repeat_interleave.self_*` then feeds the result into our
  // native v2v `index_select` without any host round-trip of `self`.
  return cpu_out.to(repeats.device());
}

} // namespace at::native::rbln
