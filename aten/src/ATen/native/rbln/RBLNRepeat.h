#pragma once

#include <ATen/core/Tensor.h>
#include <c10/core/SymInt.h>

#include <optional>

// Mirrors aten/src/ATen/native/Repeat.cpp upstream — host of
// `aten::repeat_interleave` and friends. New native kernels for these go
// here.

namespace at::native::rbln {

/**
 * @brief Native RBLN registration for `repeat_interleave.Tensor`.
 *
 * Schema: repeat_interleave.Tensor(Tensor repeats, *, SymInt? output_size=None) -> Tensor
 *
 * Returns a 1-D int64 index tensor where each value `i` appears `repeats[i]`
 * times consecutively. With `repeats == [3, 1, 2]` the result is
 * `[0, 0, 0, 1, 2, 2]`.
 *
 * The other two variants — `repeat_interleave.self_Tensor` and
 * `repeat_interleave.self_int` — are CompositeImplicitAutograd ops upstream;
 * they decompose into a call to this `.Tensor` form (to build the gather
 * indices) followed by `index_select`. With both registrations on the RBLN
 * backend, the composite chain runs entirely on-device without a host
 * round-trip of `self`.
 *
 * No real device-side compute happens here: the index pattern is computed
 * on host from the (small) `repeats` vector, then h2v'd to a fresh device
 * buffer. The win over the default CPU fallback is structural — it lets
 * the larger gather downstream stay on v2v.
 */
at::Tensor repeat_interleave_Tensor_rbln(
    const at::Tensor& repeats,
    std::optional<c10::SymInt> output_size);

} // namespace at::native::rbln
