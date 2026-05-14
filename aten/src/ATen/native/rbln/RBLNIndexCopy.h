#pragma once

#include <ATen/core/Tensor.h>

namespace at::native::rbln {

/**
 * @brief Native v2v implementation of aten::index_copy.out.
 *
 * Schema: index_copy.out(Tensor self, int dim, Tensor index, Tensor source,
 *                        *, Tensor(a!) out) -> Tensor(a!)
 *
 * Semantics (PyTorch): out is shape-equal to self; out[..., index[i], ...] =
 * source[..., i, ...] along `dim`; all other positions equal self.
 *
 * The implementation has two phases:
 *   1. Initialise `out` with `self`'s contents — skipped when `out` aliases
 *      `self` (the in-place `index_copy_` case dispatches with out == self).
 *   2. For each run of consecutive index values, slice `source` and `out`
 *      along `dim` and hand the views to the strided v2v engine.
 *
 * Routing via TORCH_LIBRARY_IMPL(aten, PrivateUse1) makes both `index_copy_`
 * and the functional `index_copy` inherit this path because they delegate to
 * `index_copy.out` upstream (`structured_delegate: index_copy.out`).
 */
at::Tensor& index_copy_out_rbln(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Tensor& source,
    at::Tensor& out);

} // namespace at::native::rbln
