#include <ATen/native/rbln/RBLNIndexSelect.h>

#include <ATen/core/Tensor.h>
#include <ATen/ops/empty.h>
#include <c10/rbln/RBLNFunctions.h>
#include <c10/rbln/RBLNLogging.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace at::native::rbln {

namespace {

// Same coalescing rule as RBLNCat: return the smallest dim j such that the
// tensor's suffix [j, rank) is contiguous in memory (size-1 dims are free).
int64_t contig_suffix_start(c10::IntArrayRef sizes, c10::IntArrayRef strides) {
  const int64_t rank = static_cast<int64_t>(sizes.size());
  if (rank == 0)
    return 0;
  int64_t expected_stride = 1;
  int64_t j = rank;
  for (int64_t i = rank - 1; i >= 0; --i) {
    if (sizes[i] == 1) { j = i; continue; }
    if (strides[i] == expected_stride) {
      expected_stride *= sizes[i];
      j = i;
    } else {
      break;
    }
  }
  return j;
}

bool advance_multi_index(std::vector<int64_t>& idx, c10::IntArrayRef outer_sizes) {
  for (int64_t d = static_cast<int64_t>(outer_sizes.size()) - 1; d >= 0; --d) {
    if (++idx[d] < outer_sizes[d]) return true;
    idx[d] = 0;
  }
  return false;
}

// Read `index` into a host-side int64 buffer. Validates dtype is integral.
std::vector<int64_t> read_index_to_host(const at::Tensor& index) {
  RBLN_CHECK(index.dim() <= 1,
             "index_select: index must be 0- or 1-D, got {}-D", index.dim());
  RBLN_CHECK(index.scalar_type() == at::kLong || index.scalar_type() == at::kInt,
             "index_select: index dtype must be int32 or int64, got {}",
             c10::str(index.scalar_type()));

  // Bring to CPU as a contiguous int64 vector. We go via .to(cpu).contiguous()
  // — for already-CPU contig tensors this is a no-op alias.
  at::Tensor host = index;
  if (!host.device().is_cpu()) host = host.cpu();
  if (!host.is_contiguous()) host = host.contiguous();

  const int64_t n = host.numel();
  std::vector<int64_t> values(n);
  if (host.scalar_type() == at::kLong) {
    std::memcpy(values.data(), host.data_ptr<int64_t>(), n * sizeof(int64_t));
  } else {
    const auto* src = host.data_ptr<int32_t>();
    for (int64_t i = 0; i < n; ++i) values[i] = static_cast<int64_t>(src[i]);
  }
  return values;
}

// Group consecutive integer runs (e.g. [3,4,5, 9, 12,13]) into (start, length)
// pairs so we can emit a single v2v per run instead of per element.
struct IndexRun { int64_t src_start; int64_t out_start; int64_t length; };

std::vector<IndexRun> coalesce_runs(const std::vector<int64_t>& idx) {
  std::vector<IndexRun> runs;
  if (idx.empty()) return runs;
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

at::Tensor& index_select_out_rbln(
    const at::Tensor& self, int64_t dim, const at::Tensor& index, at::Tensor& out) {
  RBLN_SCOPE_GUARD();

  RBLN_CHECK(self.device().is_privateuseone(),
             "index_select_out_rbln: self must be on RBLN device, got {}",
             c10::str(self.device()));
  RBLN_CHECK(out.device() == self.device(),
             "index_select: out device {} doesn't match self device {}",
             c10::str(out.device()), c10::str(self.device()));
  RBLN_CHECK(out.scalar_type() == self.scalar_type(),
             "index_select: out dtype {} doesn't match self dtype {}",
             c10::str(out.scalar_type()), c10::str(self.scalar_type()));

  const int64_t rank = self.dim();
  RBLN_CHECK(rank > 0,
             "index_select: self must have at least 1 dim, got 0-D tensor");
  const int64_t axis = dim < 0 ? dim + rank : dim;
  RBLN_CHECK(axis >= 0 && axis < rank,
             "index_select: dim {} out of range for rank-{} self", dim, rank);

  // Pull the index values onto the host once.
  const auto idx_host = read_index_to_host(index);
  const int64_t n_out = static_cast<int64_t>(idx_host.size());

  // Determine and validate output shape.
  std::vector<int64_t> out_shape(self.sizes().begin(), self.sizes().end());
  out_shape[axis] = n_out;
  if (!out.defined() || out.sizes() != at::IntArrayRef(out_shape)) {
    out.resize_(out_shape);
  }
  RBLN_CHECK(out.is_contiguous(),
             "index_select_out_rbln: out tensor must be contiguous");

  // Handle the trivial cases.
  if (n_out == 0 || self.numel() == 0) {
    return out;
  }

  const int64_t axis_extent = self.size(axis);
  for (int64_t v : idx_host) {
    RBLN_CHECK(v >= 0 && v < axis_extent,
               "index_select: index value {} out of range [0, {})", v, axis_extent);
  }

  const int64_t elm_size = static_cast<int64_t>(self.element_size());

  // Self contig analysis: cap the outer iteration at axis + 1 so the cat axis
  // is itself an outer dim (per-axis-position v2v).
  const auto self_sizes = self.sizes();
  const auto self_strides = self.strides();
  const int64_t self_contig_start = contig_suffix_start(self_sizes, self_strides);
  const int64_t outer_end = std::max<int64_t>(self_contig_start, axis + 1);

  // Inner block size = product of dims strictly after outer_end-1.
  int64_t inner_block_elems = 1;
  for (int64_t i = outer_end; i < rank; ++i) inner_block_elems *= self_sizes[i];
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

  // For dims between (axis, outer_end) — only relevant when self is non-contig
  // and outer_end > axis+1 — we additionally need to iterate them. Outer dims
  // are [0, outer_end) with the constraint axis ∈ outer dims.
  std::vector<int64_t> outer_sizes(self_sizes.begin(), self_sizes.begin() + outer_end);

  // Strip dim `axis` from the outer iteration: we instead drive it via the
  // index run loop. The remaining outer dims iterate "pre-axis" (0..axis) and
  // "between-axis-and-contig" (axis+1..outer_end) independently of the index.
  // Concretely: full iteration order = (pre-axis multi-index, run, in-run
  // offset, between-axis multi-index) which we collapse into a single outer
  // loop in row-major over (pre + between) dims.

  std::vector<int64_t> pre_axis_sizes(outer_sizes.begin(), outer_sizes.begin() + axis);
  std::vector<int64_t> btw_axis_sizes(outer_sizes.begin() + axis + 1, outer_sizes.end());

  int64_t pre_count = 1;
  for (int64_t d : pre_axis_sizes) pre_count *= d;
  int64_t btw_count = 1;
  for (int64_t d : btw_axis_sizes) btw_count *= d;

  const auto runs = coalesce_runs(idx_host);

  const uint8_t* self_base = static_cast<const uint8_t*>(self.data_ptr());
  uint8_t* out_base = static_cast<uint8_t*>(out.data_ptr());

  RBLN_LOG_DEBUG(
      "index_select_out_rbln: self shape={} strides={} axis={} outer_end={} inner_block_bytes={} n_out={} runs={}",
      c10::str(self_sizes), c10::str(self_strides), axis, outer_end,
      inner_block_bytes, n_out, runs.size());

  std::vector<int64_t> pre_idx(pre_axis_sizes.size(), 0);
  for (int64_t p = 0; p < pre_count; ++p) {
    std::vector<int64_t> btw_idx(btw_axis_sizes.size(), 0);
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
        const int64_t src_axis = run.src_start;
        const int64_t out_axis = run.out_start;
        const int64_t bytes = run.length * inner_block_bytes;

        const int64_t src_off_elems = src_off_elems_base + src_axis * self_strides[axis];
        const int64_t dst_off_bytes = dst_off_bytes_base + out_axis * out_byte_stride[axis];

        const uint8_t* src = self_base + src_off_elems * elm_size;
        uint8_t* dst = out_base + dst_off_bytes;
        c10::rbln::memcpy_v2v(dst, src, static_cast<size_t>(bytes));
      }

      if (b + 1 < btw_count) advance_multi_index(btw_idx, btw_axis_sizes);
    }
    if (p + 1 < pre_count) advance_multi_index(pre_idx, pre_axis_sizes);
  }

  return out;
}

at::Tensor index_select_rbln(const at::Tensor& self, int64_t dim, const at::Tensor& index) {
  const int64_t rank = self.dim();
  const int64_t axis = dim < 0 ? dim + rank : dim;
  RBLN_CHECK(axis >= 0 && axis < rank,
             "index_select: dim {} out of range for rank-{} self", dim, rank);

  std::vector<int64_t> out_shape(self.sizes().begin(), self.sizes().end());
  out_shape[axis] = index.numel();

  at::Tensor out = at::empty(out_shape,
                             self.options().memory_format(c10::MemoryFormat::Contiguous));
  return index_select_out_rbln(self, dim, index, out);
}

} // namespace at::native::rbln
