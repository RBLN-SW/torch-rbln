#include <ATen/native/rbln/RBLNIndexSelect.h>

#include <ATen/core/Tensor.h>
#include <ATen/native/Resize.h>
#include <ATen/native/rbln/RBLNStridedV2V.h>
#include <ATen/ops/empty.h>
#include <c10/rbln/RBLNLogging.h>
#include <c10/rbln/RBLNV2VBatch.h>

#include <cstdint>
#include <cstring>
#include <vector>

namespace at::native::rbln {

namespace {

// Read `index` into a host-side int64 buffer. Validates dtype is integral.
std::vector<int64_t> read_index_to_host(const at::Tensor& index) {
  RBLN_CHECK(index.dim() <= 1, "index_select: index must be 0- or 1-D, got {}-D", index.dim());
  RBLN_CHECK(
      index.scalar_type() == at::kLong || index.scalar_type() == at::kInt,
      "index_select: index dtype must be int32 or int64, got {}",
      c10::str(index.scalar_type()));

  at::Tensor host = index;
  if (!host.device().is_cpu())
    host = host.cpu();
  if (!host.is_contiguous())
    host = host.contiguous();

  const int64_t n = host.numel();
  std::vector<int64_t> values(n);
  if (host.scalar_type() == at::kLong) {
    std::memcpy(values.data(), host.data_ptr<int64_t>(), n * sizeof(int64_t));
  } else {
    const auto* src = host.data_ptr<int32_t>();
    for (int64_t i = 0; i < n; ++i)
      values[i] = static_cast<int64_t>(src[i]);
  }
  return values;
}

// Group consecutive integer runs (e.g. [3,4,5, 9, 12,13]) into (src_start,
// out_start, length) so we can emit one strided_v2v_copy per run instead of
// per element.
struct IndexRun {
  int64_t src_start;
  int64_t out_start;
  int64_t length;
};

std::vector<IndexRun> coalesce_runs(const std::vector<int64_t>& idx) {
  std::vector<IndexRun> runs;
  if (idx.empty())
    return runs;
  IndexRun cur{idx[0], 0, 1};
  for (int64_t i = 1; i < static_cast<int64_t>(idx.size()); ++i) {
    if (idx[i] == cur.src_start + cur.length) {
      cur.length += 1;
    } else {
      runs.push_back(cur);
      cur = IndexRun{idx[i], i, 1};
    }
  }
  runs.push_back(cur);
  return runs;
}

} // namespace

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
  const auto idx_host = read_index_to_host(index);
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

  // Per-run slab copy: src.narrow(axis, run.src_start, run.length) is a view
  // with the same strides as self (so the engine sees the correct stride
  // pattern for the chosen axis), and out.narrow(axis, run.out_start, run.length)
  // is a slice of the contig out. The engine handles every contig combination
  // — when self is contig along axis, runs collapse into one v2v each; when
  // not, the outer loop iterates axis positions inside the engine.
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
    auto src_view = self.narrow(axis, run.src_start, run.length);
    auto dst_view = out.narrow(axis, run.out_start, run.length);
    strided_v2v_copy(dst_view, src_view, batch);
  }
  // batch.submit() runs on destructor.

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

} // namespace at::native::rbln
