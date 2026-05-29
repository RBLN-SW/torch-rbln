#pragma once

#include <ATen/core/List.h>
#include <ATen/core/Tensor.h>

// Mirrors aten/src/ATen/native/TensorShape.cpp upstream — the file that hosts
// cat / stack / chunk / unbind / split / narrow_copy / transpose_copy and
// friends. New native-v2v kernels for any of those should land here rather
// than in op-specific files.

namespace at::native::rbln {

// ---------------------------------------------------------------------------
// cat
//
// Bypasses rebel-compiler entirely; cat is reduced to a sequence of
// contiguous v2v copies derived from each input's stride structure.
// Non-contiguous inputs are not materialised — instead we slice `out` per
// input via `narrow()` and hand both views to the strided v2v engine.
//
// Restrictions vs PyTorch upstream cat:
//   - All inputs and `out` must live on the same RBLN device.
//   - `out` is staged through a contig buffer when non-contiguous so the
//     per-input narrow math stays in canonical row-major.
//   - No input may overlap `out` (checked via `at::assert_no_overlap`).
//
// `stack` decomposes to `cat` upstream via CompositeImplicitAutograd, so
// it inherits this kernel without a separate registration.
// ---------------------------------------------------------------------------

at::Tensor& cat_out_rbln(const at::ITensorListRef& tensors, int64_t dim, at::Tensor& out);

} // namespace at::native::rbln
