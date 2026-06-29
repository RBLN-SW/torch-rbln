#include <ATen/native/rbln/RBLNStridedV2V.h>

#include <ATen/native/rbln/RBLNStrideUtils.h>
#include <ATen/native/rbln/RBLNTensorUtils.h>
#include <c10/rbln/RBLNFallbackConfig.h>
#include <c10/rbln/RBLNLogging.h>
#include <c10/rbln/RBLNProfiler.h>
#include <c10/util/Exception.h>

#include <cstdint>
#include <functional>
#include <string_view>
#include <vector>

namespace at::native::rbln {

void strided_v2v_copy(const at::Tensor& dst, const at::Tensor& src, c10::rbln::V2VBatch& batch) {
  RBLN_SCOPE_GUARD();

  RBLN_CHECK(
      dst.device().is_privateuseone() && src.device().is_privateuseone(),
      "strided_v2v_copy: both tensors must be on an RBLN (PrivateUse1) device, got dst={} src={}",
      c10::str(dst.device()),
      c10::str(src.device()));
  RBLN_CHECK(
      dst.device() == src.device(),
      "strided_v2v_copy: dst and src must be on the same RBLN device, got dst={} src={}",
      c10::str(dst.device()),
      c10::str(src.device()));
  RBLN_CHECK(
      dst.scalar_type() == src.scalar_type(),
      "strided_v2v_copy: dtype mismatch (dst={} src={})",
      c10::str(dst.scalar_type()),
      c10::str(src.scalar_type()));
  RBLN_CHECK(
      dst.sizes() == src.sizes(),
      "strided_v2v_copy: shape mismatch (dst={} src={})",
      c10::str(dst.sizes()),
      c10::str(src.sizes()));
  RBLN_CHECK(dst.numel() > 0, "strided_v2v_copy: numel must be > 0 (caller should short-circuit)");

  // Self-copy on the same view is a no-op (and would issue an aliased v2v).
  // Sizes already validated above, so is_same_view fully characterises identity.
  if (is_same_view(dst, src)) {
    RBLN_LOG_DEBUG("strided_v2v_copy: identity copy, no-op");
    return;
  }

  const auto rank = dst.dim();
  const auto elm = static_cast<size_t>(dst.element_size());

  // Fast path: both fully contiguous → single v2v.
  if (dst.is_contiguous() && src.is_contiguous()) {
    batch.enqueue(dst.data_ptr(), src.data_ptr(), static_cast<size_t>(dst.numel()) * elm);
    return;
  }

  // 0-D fast path. is_contiguous() above already covers this for both sides,
  // but defending here in case future contig semantics change.
  if (rank == 0) {
    batch.enqueue(dst.data_ptr(), src.data_ptr(), elm);
    return;
  }

  const auto sizes = dst.sizes();
  const auto src_strides = src.strides();
  const auto dst_strides = dst.strides();

  const int64_t inner_start = common_inner_start(sizes, src_strides, dst_strides);

  // Inner block size = product of dims [inner_start, rank). May be 1 if no
  // joint contig suffix exists (e.g. both sides non-contig at the innermost
  // non-size-1 dim) — that is correct, just slow.
  int64_t inner_block_elems = 1;
  for (int64_t i = inner_start; i < rank; ++i) {
    inner_block_elems *= sizes[i];
  }
  const size_t inner_block_bytes = static_cast<size_t>(inner_block_elems) * elm;

  // Outer description (dims [0, inner_start)). Byte-strides; stride 0
  // (broadcast) is preserved verbatim so the same source memory replicates
  // across writes.
  std::vector<int64_t> outer_sizes_vec(sizes.begin(), sizes.begin() + inner_start);
  std::vector<int64_t> src_byte_strides(inner_start);
  std::vector<int64_t> dst_byte_strides(inner_start);
  const int64_t elm_signed = static_cast<int64_t>(elm);
  int64_t outer_count = 1;
  for (int64_t i = 0; i < inner_start; ++i) {
    src_byte_strides[i] = src_strides[i] * elm_signed;
    dst_byte_strides[i] = dst_strides[i] * elm_signed;
    outer_count *= sizes[i];
  }

  RBLN_LOG_DEBUG(
      "strided_v2v_copy: sizes={} src_strides={} dst_strides={} inner_start={} inner_block_bytes={} outer_count={}",
      c10::str(sizes),
      c10::str(src_strides),
      c10::str(dst_strides),
      inner_start,
      inner_block_bytes,
      outer_count);

  batch.enqueue_strided(
      dst.data_ptr(),
      src.data_ptr(),
      inner_block_bytes,
      c10::IntArrayRef(outer_sizes_vec),
      c10::IntArrayRef(src_byte_strides),
      c10::IntArrayRef(dst_byte_strides));
}

void strided_v2v_copy(const at::Tensor& dst, const at::Tensor& src) {
  c10::rbln::V2VBatch batch;
  strided_v2v_copy(dst, src, batch);
  submit_or_fallback(batch, "strided_v2v_copy", [&] { dst.copy_(src.cpu()); });
}

void submit_or_fallback(c10::rbln::V2VBatch& batch, const char* op_name, std::function<void()> cpu_fallback) {
  try {
    batch.submit();
  } catch (const c10::Error& e) {
    const std::string_view error_message = e.what();
    // TODO: Replace substring match with a typed exception when the wrapper API allows.
    // Route both v2v backend rejections to the CPU fallback: the batched path
    // throws "rbln_memcpy_v2v_multi failed", and when it drains per-entry the
    // same-device path throws "rbln_memcpy_v2v failed". (The runtime rejects v2v
    // into interior offsets of large untyped pool allocations — e.g. vLLM's KV
    // cache; the host-bounce CPU fallback writes those correctly.)
    if (error_message.find("rbln_memcpy_v2v_multi failed") == std::string_view::npos &&
        error_message.find("rbln_memcpy_v2v failed") == std::string_view::npos) {
      throw; // validation error — propagate
    }
    if (c10::rbln::is_fallback_disabled("strided_copy_error")) {
      throw;
    }
    RBLN_LOG_WARN(
        "{}: batched strided copy failed — falling back to CPU op. "
        "Underlying error: {}",
        op_name,
        error_message);
    // PROFILER (cold branch): a strided v2v (cat / index_select / index_copy /
    // copy_) was rejected on device and fell back to a host CPU op.
    c10::rbln::prof::record_bounce(c10::rbln::prof::BounceSite::kStridedV2VFallback, 0);
    cpu_fallback();
  }
}

} // namespace at::native::rbln
