#include <ATen/native/rbln/RBLNTensorAdvancedIndexing.h>

#include <ATen/MemoryOverlap.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/stack.h> // torch::jit::drop / push
#include <ATen/native/Resize.h>
#include <ATen/native/rbln/RBLNCPUFallback.h>
#include <ATen/native/rbln/RBLNIndexUtils.h>
#include <ATen/native/rbln/RBLNStridedV2V.h>
#include <ATen/native/rbln/RBLNTensorUtils.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/index_copy.h>
#include <ATen/ops/index_select.h>
#include <ATen/ops/nonzero.h>
#include <ATen/ops/where.h>
#include <c10/rbln/RBLNFallbackConfig.h>
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

  // Mirror upstream PyTorch's `aten::index_select.out` overlap checks.
  at::assert_no_internal_overlap(out);
  at::assert_no_overlap(out, self);
  at::assert_no_overlap(out, index);

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
  submit_or_fallback(batch, "index_select_out_rbln", [&] {
    const auto cpu_self = self.cpu();
    const auto cpu_index = index.cpu();
    const auto cpu_out = at::index_select(cpu_self, dim, cpu_index);
    out.copy_(cpu_out);
  });

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
// index.Tensor_out (advanced indexing) — native white-list fast-path
// ===========================================================================
//
// White-list = a SINGLE index on one axis (all other slots `None`), which is a
// plain on-device gather: each output element is a contiguous slice of `self`,
// so the whole thing reduces to `self.index_select(axis, flat_idx)` (native
// v2v) and writes straight into the pre-allocated `out`:
//   - integer index of rank >= 1 (1-D / N-D): `flat_idx = idx.reshape(-1)`;
//     `out` is `self`'s shape with `axis` replaced by `idx`'s shape, so viewing
//     it with `axis` flattened is exactly the index_select output. (A 0-D index
//     does not reach here — `x[scalar]` decomposes to select upstream.)
//   - 1-D boolean mask over `axis`: `flat_idx` = its True positions.
//
// The fast path is taken only when `out` already has the exact expected shape
// (else the resize the .out contract may require is left to the fallback).
//
// Everything else (multiple index tensors, N-D / non-leading masks, broadcasting,
// non-contiguous `self`) stays on the CPU fallback. Boxed so that path can reuse
// `cpu_fallback_rbln`. The functional `index.Tensor` (`x[idx]`) reaches this via
// the CompositeExplicitAutogradNonFunctional out wrapper.
void index_out_rbln(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  // aten::index.Tensor_out(Tensor self, Tensor?[] indices, *, Tensor(a!) out)
  const int64_t num_args = static_cast<int64_t>(op.schema().arguments().size());
  const int64_t base = static_cast<int64_t>(stack->size()) - num_args;
  const at::Tensor self = (*stack)[base].toTensor();
  const c10::List<c10::optional<at::Tensor>> indices = (*stack)[base + 1].toOptionalTensorList();

  // Find the single non-None index and its axis (its position in `indices` is
  // the indexed dim; leading `None`s shift it, e.g. `x[:, idx]` -> axis 1).
  int64_t axis = -1;
  int64_t num_present = 0;
  for (size_t i = 0; i < indices.size(); ++i) {
    const c10::optional<at::Tensor> e = indices.get(i);
    if (e.has_value() && e->defined()) {
      ++num_present;
      axis = static_cast<int64_t>(i);
    }
  }

  // `self.numel() > 0`: an empty `self` (some dim is 0) makes advanced indexing a
  //   no-op that skips bounds checks (e.g. empty(3,0)[[3]] -> (1,0), no IndexError);
  //   leave that to the fallback rather than replicate the special case.
  // `indices.size() <= self.dim()`: more index slots than dims is "too many
  //   indices" (IndexError); the fallback raises it correctly.
  const bool eligible = num_present == 1 && self.defined() && self.numel() > 0 && self.device().is_privateuseone() &&
      self.is_contiguous() && static_cast<int64_t>(indices.size()) <= self.dim() && axis >= 0 && axis < self.dim();

  if (eligible) {
    const at::Tensor idx = indices.get(axis).value();
    at::Tensor out = (*stack)[base + 2].toTensor();
    const auto st = idx.scalar_type();

    // Build the 1-D integer index along `axis` (host-side: the index is tiny and
    // index_select copies it to host anyway, so no net v2h added) together with
    // the exact output shape aten::index.Tensor must produce for this form.
    at::Tensor flat_idx;
    std::vector<int64_t> expected;
    bool fast = false;
    if ((st == at::kLong || st == at::kInt) && idx.dim() >= 1) {
      // integer index of rank >= 1: out = self[:axis] + idx.shape + self[axis+1:]
      // (a 0-D index does not reach here — it decomposes to select upstream).
      flat_idx = idx.reshape({-1});
      // Host-side and widened to int64: the negative-wrap add and the bounds
      // compare below must not overflow int32 when an axis exceeds INT32_MAX.
      flat_idx = (flat_idx.is_cpu() ? flat_idx.contiguous() : flat_idx.cpu().contiguous()).to(at::kLong);
      // advanced indexing wraps negatives; index_select does not.
      if (flat_idx.numel() > 0 && flat_idx.lt(0).any().item<bool>())
        flat_idx = at::where(flat_idx.lt(0), flat_idx + self.size(axis), flat_idx);
      for (int64_t d = 0; d < axis; ++d)
        expected.push_back(self.size(d));
      for (int64_t d = 0; d < idx.dim(); ++d)
        expected.push_back(idx.size(d));
      for (int64_t d = axis + 1; d < self.dim(); ++d)
        expected.push_back(self.size(d));
      fast = true;
    } else if (st == at::kBool && idx.dim() == 1 && idx.size(0) == self.size(axis)) {
      // 1-D boolean mask over `axis`: out = self[:axis] + (K,) + self[axis+1:]
      const at::Tensor mask_cpu = idx.is_cpu() ? idx.contiguous() : idx.cpu().contiguous();
      flat_idx = mask_cpu.nonzero().squeeze(-1); // True positions -> (K,)
      for (int64_t d = 0; d < axis; ++d)
        expected.push_back(self.size(d));
      expected.push_back(flat_idx.numel());
      for (int64_t d = axis + 1; d < self.dim(); ++d)
        expected.push_back(self.size(d));
      fast = true;
    }

    // Take the fast path only when `out` already has the exact shape aten::index
    // would produce — the structured .out contract may otherwise require a resize
    // that only the CPU fallback performs. `out` is contiguous, so a view with
    // `axis` flattened aliases it and index_select writes into its storage.
    if (fast && out.defined() && out.is_contiguous() && out.sizes().equals(expected)) {
      // Advanced indexing raises IndexError (not a plain RuntimeError) when an
      // index is out of range; match that before delegating to index_select.
      TORCH_CHECK_INDEX(
          flat_idx.numel() == 0 ||
              (flat_idx.ge(0).all().item<bool>() && flat_idx.lt(self.size(axis)).all().item<bool>()),
          "index out of bounds for dimension ",
          axis,
          " with size ",
          self.size(axis));
      std::vector<int64_t> vshape = self.sizes().vec();
      vshape[axis] = flat_idx.numel();
      at::Tensor out_view = out.view(vshape);
      index_select_out_rbln(self, axis, flat_idx, out_view);
      torch::jit::drop(*stack, static_cast<size_t>(num_args));
      torch::jit::push(*stack, std::move(out));
      return;
    }
  }

  // Not white-listed -> identical to the previous (pre-kernel) behaviour.
  c10::rbln::log_cpu_fallback(op.schema().name());
  at::native::rbln::cpu_fallback_rbln(op, stack);
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

  // Mirror upstream PyTorch's `aten::index_copy.out` overlap checks
  // (self ↔ out aliasing excluded — in-place dispatch pattern).
  at::assert_no_internal_overlap(out);
  at::assert_no_overlap(out, index);
  at::assert_no_overlap(out, source);

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

  // Bounds-check index values once on host, and detect duplicates in the same
  // pass. Duplicates make Phase 2's per-run dst slices overlap on the axis,
  // which the batched v2v API rejects (it may reorder/parallelise entries, so
  // overlapping ranges are undefined behaviour by contract). When duplicates
  // exist we fall off the batched path and submit each run as its own v2v —
  // sequential per-run calls preserve PyTorch's last-write-wins semantics.
  const int64_t axis_extent = self.size(axis);
  std::vector<bool> seen(static_cast<size_t>(axis_extent), false);
  bool has_duplicate_index = false;
  for (int64_t v : idx_host) {
    RBLN_CHECK(v >= 0 && v < axis_extent, "index_copy: index value {} out of range [0, {})", v, axis_extent);
    if (seen[static_cast<size_t>(v)]) {
      has_duplicate_index = true;
    } else {
      seen[static_cast<size_t>(v)] = true;
    }
  }

  // Phase 1: initialise `out` with `self`. Skipped when `out` aliases `self`
  // (in-place dispatch) — saves a redundant full-tensor v2v.
  if (!is_same_view(out, self)) {
    strided_v2v_copy(out, self);
  }

  // Phase 2: overwrite indexed positions. `run.value` is the axis position the
  // run starts at in `out`/`self`; `run.pos` is the corresponding start in
  // `source`. With unique indices, one V2VBatch spans all runs so a future
  // batched v2v API can fuse them into a single backend call. With duplicate
  // indices, the per-run dst slices overlap; submit each run separately so
  // the bulk API sees one (non-overlapping) entry at a time.
  const auto runs = coalesce_runs(idx_host);
  RBLN_LOG_DEBUG(
      "index_copy_out_rbln: axis={} n_idx={} runs={} self_sizes={} source_strides={} has_duplicate_index={}",
      axis,
      n_idx,
      runs.size(),
      c10::str(self.sizes()),
      c10::str(source.strides()),
      has_duplicate_index);
  if (has_duplicate_index) {
    for (const auto& run : runs) {
      auto src_view = source.narrow(axis, run.pos, run.length);
      auto dst_view = out.narrow(axis, run.value, run.length);
      strided_v2v_copy(dst_view, src_view);
    }
  } else {
    c10::rbln::V2VBatch batch;
    for (const auto& run : runs) {
      auto src_view = source.narrow(axis, run.pos, run.length);
      auto dst_view = out.narrow(axis, run.value, run.length);
      strided_v2v_copy(dst_view, src_view, batch);
    }
    submit_or_fallback(batch, "index_copy_out_rbln", [&] {
      const auto cpu_self = self.cpu();
      const auto cpu_index = index.cpu();
      const auto cpu_source = source.cpu();
      const auto cpu_out = at::index_copy(cpu_self, dim, cpu_index, cpu_source);
      out.copy_(cpu_out);
    });
  }

  return out;
}

} // namespace at::native::rbln
