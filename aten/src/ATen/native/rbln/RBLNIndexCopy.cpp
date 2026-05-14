#include <ATen/native/rbln/RBLNIndexCopy.h>

#include <ATen/core/Tensor.h>
#include <ATen/native/Resize.h>
#include <ATen/native/rbln/RBLNIndexUtils.h>
#include <ATen/native/rbln/RBLNStridedV2V.h>
#include <ATen/native/rbln/RBLNTensorUtils.h>
#include <ATen/ops/empty.h>
#include <c10/rbln/RBLNLogging.h>
#include <c10/rbln/RBLNV2VBatch.h>

#include <cstdint>
#include <vector>

namespace at::native::rbln {

at::Tensor& index_copy_out_rbln(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Tensor& source,
    at::Tensor& out) {
  RBLN_SCOPE_GUARD();

  RBLN_CHECK(
      self.device().is_privateuseone(),
      "index_copy_out_rbln: self must be on RBLN device, got {}",
      c10::str(self.device()));
  RBLN_CHECK(
      out.device() == self.device(),
      "index_copy: out device {} doesn't match self device {}",
      c10::str(out.device()),
      c10::str(self.device()));
  RBLN_CHECK(
      source.device() == self.device(),
      "index_copy: source device {} doesn't match self device {}",
      c10::str(source.device()),
      c10::str(self.device()));
  RBLN_CHECK(
      out.scalar_type() == self.scalar_type(),
      "index_copy: out dtype {} doesn't match self dtype {}",
      c10::str(out.scalar_type()),
      c10::str(self.scalar_type()));
  RBLN_CHECK(
      source.scalar_type() == self.scalar_type(),
      "index_copy: source dtype {} doesn't match self dtype {}",
      c10::str(source.scalar_type()),
      c10::str(self.scalar_type()));

  const int64_t rank = self.dim();
  const auto idx_host = read_index_to_host(index, "index_copy");
  const int64_t n_idx = static_cast<int64_t>(idx_host.size());

  // 0-D self: dim must be 0 or -1; index must contain exactly one value (=0);
  // source must also be 0-D. The whole op reduces to out := source.
  if (rank == 0) {
    RBLN_CHECK(dim == 0 || dim == -1, "index_copy: dim {} out of range for 0-D self", dim);
    RBLN_CHECK(n_idx == 1, "index_copy: index for 0-D self must have 1 value, got {}", n_idx);
    RBLN_CHECK(idx_host[0] == 0, "index_copy: index value {} out of range [0, 1) for 0-D self", idx_host[0]);
    RBLN_CHECK(source.dim() == 0, "index_copy: source must be 0-D when self is 0-D, got {}-D", source.dim());
    at::native::resize_output(out, {});
    strided_v2v_copy(out, source);
    return out;
  }

  const int64_t axis = dim < 0 ? dim + rank : dim;
  RBLN_CHECK(axis >= 0 && axis < rank, "index_copy: dim {} out of range for rank-{} self", dim, rank);

  // Shape validations: source matches self on every non-axis dim, and source
  // has size n_idx along axis.
  RBLN_CHECK(source.dim() == rank, "index_copy: source rank {} doesn't match self rank {}", source.dim(), rank);
  for (int64_t i = 0; i < rank; ++i) {
    if (i == axis) {
      RBLN_CHECK(
          source.size(i) == n_idx,
          "index_copy: source.size(dim={}) = {} must equal index.numel() = {}",
          axis,
          source.size(i),
          n_idx);
    } else {
      RBLN_CHECK(
          source.size(i) == self.size(i),
          "index_copy: source.size({}) = {} doesn't match self.size({}) = {}",
          i,
          source.size(i),
          i,
          self.size(i));
    }
  }

  // Resize `out` to self's shape (no-op when out is the in-place target and
  // already shaped correctly, or when out.sizes() == self.sizes()).
  at::native::resize_output(out, self.sizes());

  // Non-contig `out`: stage through a contig buffer and copy_ back at the end.
  // The engine assumes a contig output for the "out aliasing self" detection
  // to be unambiguous, and most upstream callers pass a contig (or freshly
  // empty) `out` anyway.
  if (!out.is_contiguous()) {
    auto staging = at::empty(self.sizes(), out.options().memory_format(c10::MemoryFormat::Contiguous));
    index_copy_out_rbln(self, dim, index, source, staging);
    out.copy_(staging);
    return out;
  }

  // Empty index: out := self (or no-op if they already alias).
  if (n_idx == 0 || self.numel() == 0) {
    if (self.numel() > 0 && !is_same_view(out, self)) {
      strided_v2v_copy(out, self);
    }
    return out;
  }

  // Bounds-check index values once on host.
  const int64_t axis_extent = self.size(axis);
  for (int64_t v : idx_host) {
    RBLN_CHECK(v >= 0 && v < axis_extent, "index_copy: index value {} out of range [0, {})", v, axis_extent);
  }

  // Phase 1: initialise `out` with `self`. Skipped when `out` aliases `self`
  // (in-place dispatch) — saves a redundant full-tensor v2v.
  if (!is_same_view(out, self)) {
    strided_v2v_copy(out, self);
  }

  // Phase 2: overwrite indexed positions. `run.value` is the axis position the
  // run starts at in `out`/`self`; `run.pos` is the corresponding start in
  // `source`. One V2VBatch spans all runs so a future batched v2v API can fuse
  // them into a single backend call.
  c10::rbln::V2VBatch batch;
  const auto runs = coalesce_runs(idx_host);
  RBLN_LOG_DEBUG(
      "index_copy_out_rbln: axis={} n_idx={} runs={} self_sizes={} source_strides={}",
      axis,
      n_idx,
      runs.size(),
      c10::str(self.sizes()),
      c10::str(source.strides()));
  for (const auto& run : runs) {
    auto src_view = source.narrow(axis, run.pos, run.length);
    auto dst_view = out.narrow(axis, run.value, run.length);
    strided_v2v_copy(dst_view, src_view, batch);
  }
  // batch.submit() runs on destructor.

  return out;
}

} // namespace at::native::rbln
