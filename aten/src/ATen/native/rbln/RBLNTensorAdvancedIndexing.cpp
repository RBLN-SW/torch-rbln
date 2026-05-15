#include <ATen/native/rbln/RBLNTensorAdvancedIndexing.h>

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

// ===========================================================================
// index_select
// ===========================================================================

at::Tensor& index_select_out_rbln(const at::Tensor& self, int64_t dim, const at::Tensor& index, at::Tensor& out) {
  RBLN_SCOPE_GUARD();

  RBLN_CHECK(
      self.device().is_privateuseone(),
      "index_select_out_rbln: self must be on RBLN device, got {}",
      c10::str(self.device()));
  RBLN_CHECK(
      out.device() == self.device(),
      "index_select: out device {} doesn't match self device {}",
      c10::str(out.device()),
      c10::str(self.device()));
  RBLN_CHECK(
      out.scalar_type() == self.scalar_type(),
      "index_select: out dtype {} doesn't match self dtype {}",
      c10::str(out.scalar_type()),
      c10::str(self.scalar_type()));

  const int64_t rank = self.dim();
  const auto idx_host = read_index_to_host(index, "index_select");
  const int64_t n_out = static_cast<int64_t>(idx_host.size());

  // Determine the output shape so we can resize `out` uniformly.
  std::vector<int64_t> out_shape;
  if (rank == 0) {
    // 0-D self: index_select returns a 0-D output. dim must be 0 or -1; index
    // must contain exactly one value, equal to 0.
    RBLN_CHECK(dim == 0 || dim == -1, "index_select: dim {} out of range for 0-D self", dim);
    RBLN_CHECK(n_out == 1, "index_select: index to scalar can have only 1 value, got {}", n_out);
    RBLN_CHECK(idx_host[0] == 0, "index_select: index value {} out of range [0, 1) for 0-D self", idx_host[0]);
  } else {
    const int64_t a = dim < 0 ? dim + rank : dim;
    RBLN_CHECK(a >= 0 && a < rank, "index_select: dim {} out of range for rank-{} self", dim, rank);
    out_shape.assign(self.sizes().begin(), self.sizes().end());
    out_shape[a] = n_out;
  }

  at::native::resize_output(out, out_shape);

  // Non-contig out: stage through a contig buffer and copy_ back at the end.
  if (!out.is_contiguous()) {
    auto staging = at::empty(out_shape, out.options().memory_format(c10::MemoryFormat::Contiguous));
    index_select_out_rbln(self, dim, index, staging);
    out.copy_(staging);
    return out;
  }

  // 0-D fast path: single element v2v via the engine.
  if (rank == 0) {
    strided_v2v_copy(out, self);
    return out;
  }

  const int64_t axis = dim < 0 ? dim + rank : dim;

  if (n_out == 0 || self.numel() == 0) {
    return out;
  }

  // Bounds-check the index once on host.
  const int64_t axis_extent = self.size(axis);
  for (int64_t v : idx_host) {
    RBLN_CHECK(v >= 0 && v < axis_extent, "index_select: index value {} out of range [0, {})", v, axis_extent);
  }

  // Per-run slab copy: self.narrow(axis, run.value, run.length) is a view with
  // the same strides as self (so the engine sees the correct stride pattern
  // for the chosen axis), and out.narrow(axis, run.pos, run.length) is a slice
  // of the contig out. The engine handles every contig combination — when
  // self is contig along axis, runs collapse into one v2v each; when not, the
  // outer loop iterates axis positions inside the engine.
  c10::rbln::V2VBatch batch;
  const auto runs = coalesce_runs(idx_host);
  RBLN_LOG_DEBUG(
      "index_select_out_rbln: axis={} n_out={} runs={} self_sizes={} self_strides={}",
      axis,
      n_out,
      runs.size(),
      c10::str(self.sizes()),
      c10::str(self.strides()));
  for (const auto& run : runs) {
    auto src_view = self.narrow(axis, run.value, run.length);
    auto dst_view = out.narrow(axis, run.pos, run.length);
    strided_v2v_copy(dst_view, src_view, batch);
  }
  batch.submit();

  return out;
}

at::Tensor index_select_rbln(const at::Tensor& self, int64_t dim, const at::Tensor& index) {
  const int64_t rank = self.dim();

  std::vector<int64_t> out_shape;
  if (rank == 0) {
    // 0-D self → output is 0-D (rank mirrors self). index_select_out_rbln
    // validates dim ∈ {0, -1} and index has exactly one value (= 0).
  } else {
    const int64_t axis = dim < 0 ? dim + rank : dim;
    RBLN_CHECK(axis >= 0 && axis < rank, "index_select: dim {} out of range for rank-{} self", dim, rank);
    out_shape.assign(self.sizes().begin(), self.sizes().end());
    out_shape[axis] = index.numel();
  }

  at::Tensor out = at::empty(out_shape, self.options().memory_format(c10::MemoryFormat::Contiguous));
  return index_select_out_rbln(self, dim, index, out);
}

// ===========================================================================
// index_copy
// ===========================================================================

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

  // 0-D self: idx values must all be 0. PyTorch's index_copy meta admits
  // mismatched ranks when either side is 0-D, so source.dim() can be 0
  // (scalar copy) or 1 (every write targets self[0]; last-write-wins picks
  // source[n_idx - 1]).
  if (rank == 0) {
    RBLN_CHECK(dim == 0 || dim == -1, "index_copy: dim {} out of range for 0-D self", dim);
    RBLN_CHECK(n_idx >= 1, "index_copy: index for 0-D self must have at least 1 value, got {}", n_idx);
    for (int64_t i = 0; i < n_idx; ++i) {
      RBLN_CHECK(idx_host[i] == 0, "index_copy: index value {} out of range [0, 1) for 0-D self", idx_host[i]);
    }
    at::native::resize_output(out, {});
    if (source.dim() == 0) {
      strided_v2v_copy(out, source);
    } else if (source.dim() == 1) {
      RBLN_CHECK(
          source.size(0) == n_idx,
          "index_copy: source.size(0) = {} must equal index.numel() = {} when self is 0-D",
          source.size(0),
          n_idx);
      // Last write wins: copy source[n_idx - 1] (a 0-D view) into out.
      strided_v2v_copy(out, source.select(0, n_idx - 1));
    } else {
      RBLN_CHECK(false, "index_copy: source rank {} not supported when self is 0-D (must be 0 or 1)", source.dim());
    }
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
  batch.submit();

  return out;
}

} // namespace at::native::rbln
