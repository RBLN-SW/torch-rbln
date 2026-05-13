#include <ATen/native/rbln/RBLNCat.h>

#include <ATen/MemoryOverlap.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/Resize.h>
#include <ATen/native/rbln/RBLNStrideUtils.h>
#include <ATen/ops/empty.h>
#include <c10/core/ScalarType.h>
#include <c10/rbln/RBLNFunctions.h>
#include <c10/rbln/RBLNLogging.h>
#include <c10/util/Exception.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace at::native::rbln {

at::Tensor& cat_out_rbln(const at::ITensorListRef& tensors, int64_t dim, at::Tensor& out) {
  RBLN_SCOPE_GUARD();

  // Materialise the input list once so we can iterate it multiple times.
  auto materialised = tensors.materialize();
  RBLN_CHECK(materialised.size() > 0, "cat: tensors list must be non-empty");

  // Snapshot all inputs once (the IListRef is single-pass-ish).
  std::vector<at::Tensor> raw_inputs;
  raw_inputs.reserve(materialised.size());
  for (const at::Tensor& t : materialised) {
    raw_inputs.push_back(t);
  }

  // PyTorch's `at::native::cat` skips 1-D empty tensors (shape == (0,)) from
  // shape/rank validation entirely — they're a legacy "placeholder" pattern
  // that callers sprinkle in to seed an empty accumulator. Match that.
  auto is_legacy_empty_1d = [](const at::Tensor& t) { return t.dim() == 1 && t.numel() == 0; };

  // Pick the first non-legacy-empty input as the canonical reference for rank
  // / non-axis sizes / device. Result dtype still considers ALL inputs (including
  // legacy-empty ones) per PyTorch's promotion rules.
  const at::Tensor* first_ptr = nullptr;
  for (const at::Tensor& t : raw_inputs) {
    if (!is_legacy_empty_1d(t)) {
      first_ptr = &t;
      break;
    }
  }

  auto common_dtype = raw_inputs.front().scalar_type();
  for (size_t i = 1; i < raw_inputs.size(); ++i) {
    common_dtype = c10::promoteTypes(common_dtype, raw_inputs[i].scalar_type());
  }

  // All-legacy-empty case: result is the canonical 1-D empty, regardless of dim.
  if (first_ptr == nullptr) {
    const auto device = raw_inputs.front().device();
    RBLN_CHECK(device.is_privateuseone(), "cat_out_rbln: inputs must be on RBLN device, got {}", c10::str(device));
    TORCH_CHECK_TYPE(
        out.device() == device, "cat: out must be on the same device as inputs, got ", out.device(), " vs ", device);
    RBLN_CHECK(
        out.scalar_type() == common_dtype,
        "cat: out dtype mismatch ({} vs {})",
        c10::str(out.scalar_type()),
        c10::str(common_dtype));
    at::native::resize_output(out, {0});
    return out;
  }

  const at::Tensor& first = *first_ptr;
  const int64_t rank = first.dim();
  RBLN_CHECK(rank > 0, "cat requires non-zero-rank inputs, got {}-dim tensor", rank);

  const int64_t axis = dim < 0 ? dim + rank : dim;
  RBLN_CHECK(axis >= 0 && axis < rank, "cat: dim {} out of range for rank-{} inputs", dim, rank);

  const auto device = first.device();
  RBLN_CHECK(device.is_privateuseone(), "cat_out_rbln: inputs must be on RBLN device, got {}", c10::str(device));

  // Validate every non-legacy-empty input + accumulate the output axis extent.
  int64_t total_axis = 0;
  for (const at::Tensor& t : raw_inputs) {
    if (is_legacy_empty_1d(t)) {
      continue;
    }
    RBLN_CHECK(
        t.device() == device,
        "cat: all inputs must be on the same device, got {} and {}",
        c10::str(device),
        c10::str(t.device()));
    RBLN_CHECK(t.dim() == rank, "cat: rank mismatch ({} vs {})", rank, t.dim());
    for (int64_t i = 0; i < rank; ++i) {
      if (i == axis)
        continue;
      RBLN_CHECK(
          t.size(i) == first.size(i),
          "cat: size mismatch at non-concat dim {} ({} vs {})",
          i,
          t.size(i),
          first.size(i));
    }
    total_axis += t.size(axis);
  }

  // Promote inputs (excluding legacy-empty placeholders, which we drop now)
  // to the common dtype. `.to(common_dtype)` is a no-op when dtype matches.
  std::vector<at::Tensor> inputs;
  inputs.reserve(raw_inputs.size());
  for (const at::Tensor& t : raw_inputs) {
    if (is_legacy_empty_1d(t)) {
      continue;
    }
    inputs.push_back(t.scalar_type() == common_dtype ? t : t.to(common_dtype));
  }

  // cat/stack family must raise TypeError (not RuntimeError) for cross-device
  // `out` — see test/ops/test_ops.py:test_out Case 3 / TypeError list.
  TORCH_CHECK_TYPE(
      out.device() == device, "cat: out must be on the same device as inputs, got ", out.device(), " vs ", device);
  RBLN_CHECK(
      out.scalar_type() == common_dtype,
      "cat: out dtype mismatch ({} vs {})",
      c10::str(out.scalar_type()),
      c10::str(common_dtype));

  // Resize via the upstream helper so a wrong-shape non-empty `out` triggers
  // the canonical UserWarning ("An output with one or more elements was
  // resized...").
  std::vector<int64_t> out_shape(first.sizes().begin(), first.sizes().end());
  out_shape[axis] = total_axis;
  at::native::resize_output(out, out_shape);

  // Empty result (zero on cat axis or any other dim): nothing to copy.
  if (out.numel() == 0) {
    return out;
  }

  // If the caller's `out` is non-contiguous, stage through a contig buffer and
  // copy_ at the end. The main v2v kernel below assumes canonical row-major
  // strides on `out`.
  if (!out.is_contiguous()) {
    auto staging = at::empty(out_shape, out.options().memory_format(c10::MemoryFormat::Contiguous));
    cat_out_rbln(tensors, dim, staging);
    out.copy_(staging);
    return out;
  }

  // Reject any input that overlaps the output storage. In-place cat is not
  // supported; we rely on upstream `at::assert_no_overlap` so the overlap
  // detection matches PyTorch's semantics (full / partial / too-hard).
  for (const at::Tensor& t : inputs) {
    at::assert_no_overlap(out, t);
  }

  const int64_t elm_size = static_cast<int64_t>(out.element_size());

  // Output byte-strides under canonical row-major layout.
  std::vector<int64_t> out_byte_stride(rank);
  {
    int64_t s = elm_size;
    for (int64_t i = rank - 1; i >= 0; --i) {
      out_byte_stride[i] = s;
      s *= out_shape[i];
    }
  }

  uint8_t* out_base = static_cast<uint8_t*>(out.data_ptr());

  int64_t axis_offset = 0; // running offset along the cat axis (in elements)

  for (const at::Tensor& t : inputs) {
    // Empty inputs contribute no bytes but still advance axis_offset by their
    // axis extent (which is 0 when the cat dim is the empty dim; otherwise the
    // tensor would have numel>0). Skip before any v2v emission so we never call
    // memcpy_v2v with nbytes==0 (which the runtime rejects).
    if (t.numel() == 0) {
      axis_offset += t.size(axis);
      continue;
    }

    const auto in_sizes = t.sizes();
    const auto in_strides = t.strides();

    // Find the input's innermost contiguous suffix. We can absorb the cat
    // axis itself into the v2v block: the output's byte stride at `axis` is
    // exactly the inner-block size (output is canonical contiguous), so the
    // input slab [axis_offset, axis_offset + input.shape[axis]) along the
    // output's axis dim is contiguous in output too. Concretely:
    //   outer_end = max(contig_start_of_input, axis)
    //   block dims = [outer_end, rank)        (block_elems incl. axis if it's contig)
    //   outer dims = [0, outer_end)           (axis IS in outer iter iff axis < outer_end,
    //                                          which only happens when contig_start > axis)
    const int64_t in_contig_start = contig_suffix_start(in_sizes, in_strides);
    const int64_t outer_end = std::max<int64_t>(in_contig_start, axis);

    // Block size = product of inner (post-outer_end) dims.
    int64_t block_elems = 1;
    for (int64_t i = outer_end; i < rank; ++i)
      block_elems *= in_sizes[i];
    const int64_t block_bytes = block_elems * elm_size;

    std::vector<int64_t> outer_sizes(in_sizes.begin(), in_sizes.begin() + outer_end);
    std::vector<int64_t> idx(outer_end, 0);

    const uint8_t* in_base = static_cast<const uint8_t*>(t.data_ptr());

    int64_t outer_count = 1;
    for (int64_t d : outer_sizes)
      outer_count *= d;

    RBLN_LOG_DEBUG(
        "cat_out_rbln: input shape={} strides={} contig_start={} outer_end={} block_bytes={} outer_count={}",
        c10::str(in_sizes),
        c10::str(in_strides),
        in_contig_start,
        outer_end,
        block_bytes,
        outer_count);

    for (int64_t o = 0; o < outer_count; ++o) {
      // Source offset (in elements) from input strides.
      int64_t src_off_elems = 0;
      for (int64_t d = 0; d < outer_end; ++d)
        src_off_elems += idx[d] * in_strides[d];
      const uint8_t* src = in_base + src_off_elems * elm_size;

      // Destination offset (in bytes) — output is contiguous, so we use its
      // canonical byte strides. If axis is part of the outer iteration (true
      // only when input contig_start > axis), its coord is shifted by the
      // running axis_offset; otherwise axis is absorbed into the v2v block
      // and we add axis_offset * out_byte_stride[axis] once.
      int64_t dst_off_bytes = 0;
      for (int64_t d = 0; d < outer_end; ++d) {
        const int64_t coord = (d == axis) ? (axis_offset + idx[d]) : idx[d];
        dst_off_bytes += coord * out_byte_stride[d];
      }
      if (axis >= outer_end) {
        // axis was absorbed into the block — its contribution is the static offset.
        dst_off_bytes += axis_offset * out_byte_stride[axis];
      }
      uint8_t* dst = out_base + dst_off_bytes;

      c10::rbln::memcpy_v2v(dst, src, static_cast<size_t>(block_bytes));

      advance_multi_index(idx, outer_sizes);
    }

    axis_offset += in_sizes[axis];
  }

  return out;
}

} // namespace at::native::rbln
