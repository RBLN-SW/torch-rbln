#include <ATen/native/rbln/RBLNTensorShape.h>

#include <ATen/MemoryOverlap.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/Resize.h>
#include <ATen/native/rbln/RBLNStridedV2V.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/empty.h>
#include <c10/core/ScalarType.h>
#include <c10/rbln/RBLNFallbackConfig.h>
#include <c10/rbln/RBLNLogging.h>
#include <c10/rbln/RBLNV2VBatch.h>
#include <c10/util/Exception.h>

#include <cstdint>
#include <vector>

namespace at::native::rbln {

namespace {

// PyTorch's `at::native::cat` skips 1-D empty tensors (shape == (0,)) from
// shape/rank validation entirely — they're a legacy "placeholder" pattern
// that callers sprinkle in to seed an empty accumulator.
bool is_legacy_empty_1d(const at::Tensor& t) {
  return t.dim() == 1 && t.numel() == 0;
}

} // namespace

at::Tensor& cat_out_rbln(const at::ITensorListRef& tensors, int64_t dim, at::Tensor& out) {
  RBLN_SCOPE_GUARD();

  // Materialise the input list once so we can iterate it multiple times.
  auto materialised = tensors.materialize();
  RBLN_CHECK(materialised.size() > 0, "cat: tensors list must be non-empty");

  std::vector<at::Tensor> raw_inputs;
  raw_inputs.reserve(materialised.size());
  for (const at::Tensor& t : materialised) {
    raw_inputs.push_back(t);
  }

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

  // Mirror upstream PyTorch's `aten::cat.out` overlap checks.
  at::assert_no_internal_overlap(out);
  for (const auto& t : inputs) {
    at::assert_no_overlap(out, t);
  }

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
  // copy_ at the end. We don't strictly need this — the engine handles
  // non-contig dst views — but staging keeps the per-input slicing math (which
  // assumes canonical row-major `out`) trivially correct, and the final v2v
  // back is one strided call.
  if (!out.is_contiguous()) {
    auto staging = at::empty(out_shape, out.options().memory_format(c10::MemoryFormat::Contiguous));
    cat_out_rbln(tensors, dim, staging);
    out.copy_(staging);
    return out;
  }

  // Per-input slab copy: `out.narrow(axis, axis_offset, n)` carves out the
  // destination view for one input; the engine handles whatever stride
  // pattern the input has. All sub-copies share a single V2VBatch so any
  // future batched v2v API can fuse them into one backend call.
  c10::rbln::V2VBatch batch;
  int64_t axis_offset = 0;
  for (const at::Tensor& t : inputs) {
    const int64_t extent = t.size(axis);
    if (t.numel() == 0) {
      // Empty inputs contribute no bytes but still advance the axis offset by
      // their (possibly zero) axis extent.
      axis_offset += extent;
      continue;
    }
    auto dst_view = out.narrow(axis, axis_offset, extent);
    strided_v2v_copy(dst_view, t, batch);
    axis_offset += extent;
  }
  submit_or_fallback(batch, "cat_out_rbln", [&] {
    std::vector<at::Tensor> cpu_tensors;
    cpu_tensors.reserve(inputs.size());
    for (const auto& t : inputs) {
      cpu_tensors.push_back(t.cpu());
    }
    const auto cpu_out = at::cat(cpu_tensors, axis);
    out.copy_(cpu_out);
  });

  return out;
}

} // namespace at::native::rbln
