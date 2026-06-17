#pragma once

#include <ATen/core/Tensor.h>

// Mirrors aten/src/ATen/native/TensorAdvancedIndexing.cpp upstream — the file
// that hosts index_select / index_copy / index_add / index_put / gather /
// scatter and friends. New native-v2v kernels for any of those should land
// here rather than in op-specific files.

namespace at::native::rbln {

// ---------------------------------------------------------------------------
// index_select
//
// Decomposes the gather into a sequence of contiguous slice copies. The
// `index` tensor is read on host (if it lives on the device, it is fetched
// via v2h before the loop). Consecutive index runs are coalesced into a
// single v2v call to reduce per-call overhead.
// ---------------------------------------------------------------------------

/// Pre-allocated `out` overload.
at::Tensor& index_select_out_rbln(
    const at::Tensor& self, int64_t dim, const at::Tensor& index, at::Tensor& out);

/// Non-out overload: allocates a fresh contiguous output and delegates.
at::Tensor index_select_rbln(const at::Tensor& self, int64_t dim, const at::Tensor& index);

// ---------------------------------------------------------------------------
// index_copy
//
// Schema: index_copy.out(Tensor self, int dim, Tensor index, Tensor source,
//                        *, Tensor(a!) out) -> Tensor(a!)
//
// `out` is shape-equal to `self`; `out[..., index[i], ...] = source[..., i, ...]`
// along `dim`; all other positions equal `self`. Two phases:
//   1. Initialise `out` with `self`'s contents — skipped when `out` aliases
//      `self` (the in-place `index_copy_` case dispatches with `out == self`).
//   2. For each run of consecutive index values, slice `source` and `out`
//      along `dim` and hand the views to the strided v2v engine.
//
// Routing via TORCH_LIBRARY_IMPL(aten, PrivateUse1) makes both `index_copy_`
// and the functional `index_copy` inherit this path because they delegate to
// `index_copy.out` upstream (structured_delegate: index_copy.out).
// ---------------------------------------------------------------------------

at::Tensor& index_copy_out_rbln(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Tensor& source,
    at::Tensor& out);

} // namespace at::native::rbln
