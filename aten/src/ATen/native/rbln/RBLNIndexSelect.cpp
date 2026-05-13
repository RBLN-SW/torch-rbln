#include <ATen/native/rbln/RBLNIndexSelect.h>

#include <ATen/core/Tensor.h>
#include <ATen/native/Resize.h>
#include <ATen/native/rbln/RBLNStrideUtils.h>
#include <ATen/ops/empty.h>
#include <c10/rbln/RBLNFunctions.h>
#include <c10/rbln/RBLNLogging.h>

#include <algorithm>
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

  // Bring to CPU as a contiguous int64 vector. We go via .to(cpu).contiguous()
  // — for already-CPU contig tensors this is a no-op alias.
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

// Group consecutive integer runs (e.g. [3,4,5, 9, 12,13]) into (start, length)
// pairs so we can emit a single v2v per run instead of per element.
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
  // Pull the index values onto the host once (also validates dtype / shape).
  const auto idx_host = read_index_to_host(index);
  const int64_t n_out = static_cast<int64_t>(idx_host.size());

  // Determine the output shape so we can resize `out` uniformly.
  std::vector<int64_t> out_shape;
  if (rank == 0) {
    // 0-D self: PyTorch's index_select returns a 0-D output (rank mirrors self).
    // dim must be 0 or -1; index must contain exactly one value, equal to 0.
    RBLN_CHECK(dim == 0 || dim == -1, "index_select: dim {} out of range for 0-D self", dim);
    RBLN_CHECK(n_out == 1, "index_select: index to scalar can have only 1 value, got {}", n_out);
    RBLN_CHECK(idx_host[0] == 0, "index_select: index value {} out of range [0, 1) for 0-D self", idx_host[0]);
  } else {
    const int64_t a = dim < 0 ? dim + rank : dim;
    RBLN_CHECK(a >= 0 && a < rank, "index_select: dim {} out of range for rank-{} self", dim, rank);
    out_shape.assign(self.sizes().begin(), self.sizes().end());
    out_shape[a] = n_out;
  }

  // Resize via the upstream helper so a wrong-shape non-empty `out` triggers
  // the canonical UserWarning ("An output with one or more elements was
  // resized...").
  at::native::resize_output(out, out_shape);

  // If the caller's `out` is non-contiguous, stage through a contig buffer and
  // copy_ at the end. The main v2v kernel below assumes canonical row-major
  // strides on `out`.
  if (!out.is_contiguous()) {
    auto staging = at::empty(out_shape, out.options().memory_format(c10::MemoryFormat::Contiguous));
    index_select_out_rbln(self, dim, index, staging);
    out.copy_(staging);
    return out;
  }

  // 0-D fast path: just v2v the single element.
  if (rank == 0) {
    const size_t elm_size = static_cast<size_t>(self.element_size());
    c10::rbln::memcpy_v2v(out.data_ptr(), self.data_ptr(), elm_size);
    return out;
  }

  const int64_t axis = dim < 0 ? dim + rank : dim;

  if (n_out == 0 || self.numel() == 0) {
    return out;
  }

  const int64_t axis_extent = self.size(axis);
  for (int64_t v : idx_host) {
    RBLN_CHECK(v >= 0 && v < axis_extent, "index_select: index value {} out of range [0, {})", v, axis_extent);
  }

  const int64_t elm_size = static_cast<int64_t>(self.element_size());

  // Cap the outer iteration at axis + 1 so the index axis is itself an outer
  // dim (we drive it explicitly via the index run loop, not row-major).
  const auto self_sizes = self.sizes();
  const auto self_strides = self.strides();
  const int64_t self_contig_start = contig_suffix_start(self_sizes, self_strides);
  const int64_t outer_end = std::max<int64_t>(self_contig_start, axis + 1);

  // Inner block size = product of dims [outer_end, rank).
  int64_t inner_block_elems = 1;
  for (int64_t i = outer_end; i < rank; ++i)
    inner_block_elems *= self_sizes[i];
  const int64_t inner_block_bytes = inner_block_elems * elm_size;

  // Output byte strides (canonical row-major).
  std::vector<int64_t> out_byte_stride(rank);
  {
    int64_t s = elm_size;
    for (int64_t i = rank - 1; i >= 0; --i) {
      out_byte_stride[i] = s;
      s *= out_shape[i];
    }
  }

  // Outer iteration covers dims [0, outer_end) split into pre-axis and
  // between-axis ranges. Dim `axis` itself is driven by the index run loop,
  // not row-major iteration, so it is excluded from both ranges.
  std::vector<int64_t> pre_axis_sizes(self_sizes.begin(), self_sizes.begin() + axis);
  std::vector<int64_t> btw_axis_sizes(self_sizes.begin() + axis + 1, self_sizes.begin() + outer_end);

  int64_t pre_count = 1;
  for (int64_t d : pre_axis_sizes)
    pre_count *= d;
  int64_t btw_count = 1;
  for (int64_t d : btw_axis_sizes)
    btw_count *= d;

  // Run coalescing (a length-L run = L consecutive indices [k, k+1, ..., k+L-1]
  // emitted as one v2v of L*inner_block_bytes) is only valid when consecutive
  // axis positions are adjacent in source memory, i.e. self_strides[axis] ==
  // inner_block_elems. For non-contig self where the axis lives in the outer
  // iteration (outer_end > axis + 1), stride[axis] is larger and consecutive
  // index values map to memory positions separated by padding bytes — coalescing
  // would read those padding bytes into the output. Detect this and emit one
  // v2v per axis-position in such runs.
  const bool axis_runs_are_contig_in_memory = (self_strides[axis] == inner_block_elems);
  const auto runs = coalesce_runs(idx_host);

  const uint8_t* self_base = static_cast<const uint8_t*>(self.data_ptr());
  uint8_t* out_base = static_cast<uint8_t*>(out.data_ptr());

  RBLN_LOG_DEBUG(
      "index_select_out_rbln: self shape={} strides={} axis={} outer_end={} inner_block_bytes={} n_out={} runs={} coalesce={}",
      c10::str(self_sizes),
      c10::str(self_strides),
      axis,
      outer_end,
      inner_block_bytes,
      n_out,
      runs.size(),
      axis_runs_are_contig_in_memory);

  std::vector<int64_t> pre_idx(pre_axis_sizes.size(), 0);
  std::vector<int64_t> btw_idx(btw_axis_sizes.size(), 0);
  for (int64_t p = 0; p < pre_count; ++p) {
    std::fill(btw_idx.begin(), btw_idx.end(), 0);
    for (int64_t b = 0; b < btw_count; ++b) {
      // Compute the part of src/dst offsets that don't depend on the run.
      int64_t src_off_elems_base = 0;
      int64_t dst_off_bytes_base = 0;
      for (int64_t d = 0; d < axis; ++d) {
        src_off_elems_base += pre_idx[d] * self_strides[d];
        dst_off_bytes_base += pre_idx[d] * out_byte_stride[d];
      }
      for (int64_t d = 0; d < static_cast<int64_t>(btw_axis_sizes.size()); ++d) {
        const int64_t out_d = axis + 1 + d;
        src_off_elems_base += btw_idx[d] * self_strides[out_d];
        dst_off_bytes_base += btw_idx[d] * out_byte_stride[out_d];
      }

      for (const auto& run : runs) {
        if (axis_runs_are_contig_in_memory) {
          // Fast path: one v2v for the whole run.
          const int64_t src_off_elems = src_off_elems_base + run.src_start * self_strides[axis];
          const int64_t dst_off_bytes = dst_off_bytes_base + run.out_start * out_byte_stride[axis];
          const size_t bytes = static_cast<size_t>(run.length * inner_block_bytes);
          c10::rbln::memcpy_v2v(out_base + dst_off_bytes, self_base + src_off_elems * elm_size, bytes);
        } else {
          // Slow path: one v2v per axis position (self is non-contig at axis).
          for (int64_t k = 0; k < run.length; ++k) {
            const int64_t src_off_elems = src_off_elems_base + (run.src_start + k) * self_strides[axis];
            const int64_t dst_off_bytes = dst_off_bytes_base + (run.out_start + k) * out_byte_stride[axis];
            c10::rbln::memcpy_v2v(
                out_base + dst_off_bytes, self_base + src_off_elems * elm_size, static_cast<size_t>(inner_block_bytes));
          }
        }
      }

      advance_multi_index(btw_idx, btw_axis_sizes);
    }
    advance_multi_index(pre_idx, pre_axis_sizes);
  }

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
