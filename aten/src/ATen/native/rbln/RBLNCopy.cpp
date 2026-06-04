// TODO: The previous copy optimizations based on physical shape and physical dtype were removed during
// v-memory integration. Revisit these optimizations in a future pass.
#include <ATen/native/rbln/RBLNCopy.h>
#include <ATen/native/rbln/RBLNStrideUtils.h>
#include <ATen/native/rbln/RBLNStridedV2V.h>
#include <ATen/native/rbln/RBLNTensorUtils.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_strided.h>
#include <c10/rbln/RBLNFallbackConfig.h>
#include <c10/rbln/RBLNFunctions.h>
#include <c10/rbln/RBLNLogging.h>
#include <rebel/runtime/api/rbln_runtime_api.h>
#include <c10/rbln/RBLNProfiler.h>

#include <cstddef>

namespace at::native::rbln {

namespace {

// Recursion guard: a non-direct rbln->rbln copy_ services itself by a v2h
// (get_cpu_copy_of_rbln_tensor) followed by a cpu->rbln copy. We count that
// round-trip ONCE (kRbln2RblnIndirect); this flag suppresses the inner cpu->rbln
// staging / noncontig counters so a single copy_ is not double-counted.
thread_local int g_indirect_d2d_depth = 0;
struct IndirectD2DGuard {
  IndirectD2DGuard() {
    ++g_indirect_d2d_depth;
  }
  ~IndirectD2DGuard() {
    --g_indirect_d2d_depth;
  }
};

bool is_direct_copy(const at::Tensor& src, const at::Tensor& dst) {
  const bool same_sizes = (src.sizes() == dst.sizes());
  const bool same_dtype = (src.scalar_type() == dst.scalar_type());
  const bool both_contiguous = (src.is_contiguous() && dst.is_contiguous());
  const bool direct_copy = (same_sizes && same_dtype && both_contiguous);
  return direct_copy;
}

void tensor_copy_from_cpu_to_rbln(const at::Tensor& cpu_src, const at::Tensor& rbln_dst) {
  RBLN_SCOPE_GUARD();
  RBLN_LOG_DEBUG("src_data={}, dst_data={}", fmt::ptr(cpu_src.data_ptr()), fmt::ptr(rbln_dst.data_ptr()));
  RBLN_CHECK(cpu_src.device().is_cpu() && rbln_dst.device().is_privateuseone());

  const auto direct_copy = is_direct_copy(cpu_src, rbln_dst);
  if (direct_copy) {
    RBLN_LOG_DEBUG("Directly copying CPU src to RBLN dst");
    auto* dst_data = rbln_dst.data_ptr();
    const auto* src_data = cpu_src.data_ptr();
    const auto nbytes = at::detail::computeStorageNbytes(cpu_src.sizes(), cpu_src.strides(), cpu_src.element_size());
    c10::rbln::memcpy_h2v(dst_data, src_data, nbytes);
  } else {
    if (rbln_dst.is_contiguous()) {
      const auto dst_sizes = rbln_dst.sizes();
      const auto dst_dtype = rbln_dst.scalar_type();
      const bool same_sizes = (cpu_src.sizes() == dst_sizes);
      const bool same_dtype = (cpu_src.scalar_type() == dst_dtype);

      RBLN_LOG_DEBUG("Preparing contiguous CPU src matching dst sizes/dtype");
      auto prepared_cpu_src = cpu_src;
      if (!same_sizes || !same_dtype || !cpu_src.is_contiguous()) {
        // PROFILER (cold branch): hidden staging alloc + CPU copy before h2v,
        // forced by a broadcast / dtype / contiguity mismatch on the cpu src.
        // Suppressed when nested inside an rbln->rbln indirect copy (counted once there).
        if (g_indirect_d2d_depth == 0) {
          c10::rbln::prof::record_bounce(
              c10::rbln::prof::BounceSite::kCpu2RblnStaging,
              static_cast<uint64_t>(rbln_dst.numel()) * rbln_dst.element_size());
        }
        prepared_cpu_src =
            at::empty(dst_sizes, dst_dtype, std::nullopt, c10::Device(c10::kCPU), false, c10::MemoryFormat::Contiguous);

        // Upstream at::native::copy_() handles broadcasting, dtype conversion, and non-contiguous tensors.
        prepared_cpu_src.copy_(cpu_src);
      }
      RBLN_CHECK(prepared_cpu_src.sizes() == dst_sizes);
      RBLN_CHECK(prepared_cpu_src.scalar_type() == dst_dtype);
      RBLN_CHECK(prepared_cpu_src.is_contiguous());

      RBLN_LOG_DEBUG("Copying prepared CPU src to RBLN dst");
      auto* dst_data = rbln_dst.data_ptr();
      const auto* src_data = prepared_cpu_src.data_ptr();
      const auto nbytes = at::detail::computeStorageNbytes(
          prepared_cpu_src.sizes(), prepared_cpu_src.strides(), prepared_cpu_src.element_size());
      c10::rbln::memcpy_h2v(dst_data, src_data, nbytes);
    } else {
      // PROFILER (cold branch): a non-contiguous rbln dst is pulled to host
      // (hidden v2h), written on CPU, then copied back (h2v). Suppressed when
      // nested inside an rbln->rbln indirect copy (counted once there).
      if (g_indirect_d2d_depth == 0) {
        c10::rbln::prof::record_bounce(
            c10::rbln::prof::BounceSite::kCpu2RblnNoncontigDst,
            static_cast<uint64_t>(rbln_dst.numel()) * rbln_dst.element_size());
      }
      RBLN_LOG_DEBUG("Creating CPU copy of non-contiguous RBLN dst");
      auto cpu_dst = at::native::rbln::get_cpu_copy_of_rbln_tensor(rbln_dst);

      RBLN_LOG_DEBUG("Copying CPU src to CPU copy");
      // Upstream at::native::copy_() handles broadcasting, dtype conversion, and non-contiguous tensors.
      cpu_dst.copy_(cpu_src);

      RBLN_LOG_DEBUG("Copying CPU copy back to RBLN dst");
      auto* dst_data = rbln_dst.data_ptr();
      const auto* src_data = cpu_dst.data_ptr();
      const auto nbytes = at::detail::computeStorageNbytes(cpu_dst.sizes(), cpu_dst.strides(), cpu_dst.element_size());
      c10::rbln::memcpy_h2v(dst_data, src_data, nbytes);
    }
  }
}

void tensor_copy_from_rbln_to_cpu(const at::Tensor& rbln_src, const at::Tensor& cpu_dst) {
  RBLN_SCOPE_GUARD();
  RBLN_LOG_DEBUG("src_data={}, dst_data={}", fmt::ptr(rbln_src.data_ptr()), fmt::ptr(cpu_dst.data_ptr()));
  RBLN_CHECK(rbln_src.device().is_privateuseone() && cpu_dst.device().is_cpu());

  const auto direct_copy = is_direct_copy(rbln_src, cpu_dst);
  if (direct_copy) {
    RBLN_LOG_DEBUG("Directly copying RBLN src to CPU dst");

    auto* dst_data = cpu_dst.data_ptr();
    const auto* src_data = rbln_src.data_ptr();
    const auto nbytes = at::detail::computeStorageNbytes(rbln_src.sizes(), rbln_src.strides(), rbln_src.element_size());
    c10::rbln::memcpy_v2h(dst_data, src_data, nbytes);
  } else {
    RBLN_LOG_DEBUG("Creating CPU copy of RBLN src");
    const auto cpu_src = at::native::rbln::get_cpu_copy_of_rbln_tensor(rbln_src);

    RBLN_LOG_DEBUG("Copying CPU copy to CPU dst");
    // Upstream at::native::copy_() handles broadcasting, dtype conversion, and non-contiguous tensors.
    cpu_dst.copy_(cpu_src);
  }
}

void tensor_copy_from_rbln_to_rbln(const at::Tensor& rbln_src, const at::Tensor& rbln_dst) {
  RBLN_SCOPE_GUARD();
  RBLN_LOG_DEBUG("src_data={}, dst_data={}", fmt::ptr(rbln_src.data_ptr()), fmt::ptr(rbln_dst.data_ptr()));
  RBLN_CHECK(rbln_src.device().is_privateuseone() && rbln_dst.device().is_privateuseone());

  if (is_direct_copy(rbln_src, rbln_dst)) {
    RBLN_LOG_DEBUG("Directly copying RBLN src to RBLN dst");

    auto* dst_data = rbln_dst.data_ptr();
    const auto* src_data = rbln_src.data_ptr();
    const auto nbytes = at::detail::computeStorageNbytes(rbln_src.sizes(), rbln_src.strides(), rbln_src.element_size());
    c10::rbln::memcpy_v2v(dst_data, src_data, nbytes);
    return;
  }

  // Strided copy: route to the on-device v2v engine while the outer iteration
  // count stays within the runtime per-dst v2v cap (::rbln::kMaxV2VMultiCopies,
  // the single source shared with the runtime); above it the engine fans out to
  // a host fallback anyway, so bounce via host here.
  if (rbln_src.sizes() == rbln_dst.sizes() && rbln_src.scalar_type() == rbln_dst.scalar_type() &&
      rbln_src.device() == rbln_dst.device()) {
    const auto inner_start = common_inner_start(rbln_src.sizes(), rbln_src.strides(), rbln_dst.strides());
    int64_t outer_count = 1;
    for (int64_t i = 0; i < inner_start; ++i) {
      outer_count *= rbln_src.size(i);
    }
    if (outer_count <= static_cast<int64_t>(::rbln::kMaxV2VMultiCopies)) {
      strided_v2v_copy(rbln_dst, rbln_src);
      return;
    }
  }

  // PROFILER (cold branch): reaching here is a non-direct device->device copy_
  // that round-trips host — v2h (get_cpu_copy_of_rbln_tensor) then h2v (the
  // headline "hidden host bounce", e.g. #94 KV partial-block copy). The direct
  // and on-device strided-v2v paths above both return first, so this fall-through
  // always means a real host bounce. Count incident + bytes; IndirectD2DGuard
  // suppresses the inner cpu->rbln staging counters so it is not double-counted.
  c10::rbln::prof::record_bounce(
      c10::rbln::prof::BounceSite::kRbln2RblnIndirect,
      static_cast<uint64_t>(rbln_src.numel()) * rbln_src.element_size());
  RBLN_LOG_DEBUG("Creating CPU copy of RBLN src");
  const auto cpu_src = at::native::rbln::get_cpu_copy_of_rbln_tensor(rbln_src);

  RBLN_LOG_DEBUG("Copying CPU copy to RBLN dst");
  const IndirectD2DGuard guard;
  tensor_copy_from_cpu_to_rbln(cpu_src, rbln_dst);
}

void copy_impl_rbln(const at::Tensor& src, const at::Tensor& dst) {
  RBLN_SCOPE_GUARD();
  RBLN_LOG_DEBUG("src_metadata={}", at::native::rbln::get_tensor_metadata_string(src));
  RBLN_LOG_DEBUG("dst_metadata={}", at::native::rbln::get_tensor_metadata_string(dst));

  RBLN_LOG_DEBUG("Validating tensors with TensorIterator");
  const auto iter = at::TensorIteratorConfig()
                        .add_output(dst)
                        .add_const_input(src)
                        .resize_outputs(false)
                        .check_all_same_dtype(false)
                        .check_all_same_device(false)
                        .build();

  if (iter.numel() == 0) {
    RBLN_LOG_DEBUG("No elements to copy");
    return;
  }
  const auto src_numel = src.numel();
  const auto dst_numel = dst.numel();
  RBLN_CHECK(src_numel > 0, "Source tensor must have positive numel, got {}", src_numel);
  RBLN_CHECK(dst_numel > 0, "Destination tensor must have positive numel, got {}", dst_numel);

  const auto src_device = src.device();
  const auto dst_device = dst.device();
  if (src_device.is_cpu() && dst_device.is_privateuseone()) {
    tensor_copy_from_cpu_to_rbln(src, dst);
  } else if (src_device.is_privateuseone() && dst_device.is_cpu()) {
    tensor_copy_from_rbln_to_cpu(src, dst);
  } else if (src_device.is_privateuseone() && dst_device.is_privateuseone()) {
    tensor_copy_from_rbln_to_rbln(src, dst);
  } else {
    RBLN_CHECK(
        false, "Tensor copy from {} device to {} device is not supported", c10::str(src_device), c10::str(dst_device));
  }
}

} // namespace

at::Tensor _copy_from_rbln(const at::Tensor& src, const at::Tensor& dst, bool non_blocking) {
  RBLN_SCOPE_GUARD();
  RBLN_LOG_DEBUG("src_data={}, dst_data={}", fmt::ptr(src.data_ptr()), fmt::ptr(dst.data_ptr()));

  if (non_blocking) {
    if (c10::rbln::is_fallback_disabled("non_blocking_copy")) {
      RBLN_CHECK(
          false,
          "Non-blocking copy is not supported on RBLN devices. "
          "To enable fallback to blocking copy, remove 'non_blocking_copy' from `TORCH_RBLN_DISABLE_FALLBACK`.");
    } else {
      RBLN_WARN_ONCE("Non-blocking copy is not supported, falling back to blocking copy");
    }
  }

  copy_impl_rbln(src, dst);

  return dst;
}

at::Tensor _copy_from_and_resize_rbln(const at::Tensor& src, const at::Tensor& dst) {
  RBLN_SCOPE_GUARD();
  RBLN_LOG_DEBUG("src_data={}, dst_data={}", fmt::ptr(src.data_ptr()), fmt::ptr(dst.data_ptr()));

  const auto src_sizes = src.sizes();
  const auto dst_sizes = dst.sizes();
  RBLN_LOG_DEBUG("src_sizes={}, dst_sizes={}", c10::str(src_sizes), c10::str(dst_sizes));
  if (dst_sizes != src_sizes) {
    RBLN_LOG_DEBUG("Resizing dst to match src");
    dst.resize_(src_sizes);
  }

  copy_impl_rbln(src, dst);

  return dst;
}

at::Tensor clone_rbln(const at::Tensor& self, std::optional<c10::MemoryFormat> memory_format) {
  RBLN_SCOPE_GUARD();
  // aten::clone's default is MemoryFormat::Preserve, which is a *directive*
  // (allocate the output with whatever layout best preserves the source's
  // observable strides), not a concrete storage layout you can hand to
  // ``at::empty``. Mirror PyTorch's reference resolution before allocating:
  //   * Preserve + dense contiguous source → Contiguous
  //   * Preserve + channels_last source → ChannelsLast (2D/3D)
  //   * Preserve + arbitrary strided view → empty_strided(self.sizes(),
  //     self.strides()) so the output observes identical strides
  // For anything else the user explicitly named (Contiguous, ChannelsLast,
  // ChannelsLast3d) we pass it through to ``at::empty`` as before.
  const auto mf_req = memory_format.value_or(c10::MemoryFormat::Preserve);
  const size_t nbytes = static_cast<size_t>(self.numel()) * self.element_size();

  // Direct-d2d eligibility: contiguous storage layout, zero storage offset,
  // and the user view spans the entire storage. When all three hold the
  // source elements are contiguous in storage, so a single
  // c10::rbln::memcpy_v2v moves them to a freshly allocated output without
  // going through aten::copy_'s dispatch + TensorIterator path.
  const bool direct_d2d_eligible = self.is_contiguous() && self.storage_offset() == 0 &&
      static_cast<int64_t>(self.storage().nbytes()) == self.numel() * self.element_size();

  at::Tensor out;
  if (mf_req == c10::MemoryFormat::Preserve) {
    if (self.is_non_overlapping_and_dense()) {
      // Strided dense source: replay the strides on a fresh buffer so the
      // clone observes identical layout.
      out = at::empty_strided(self.sizes(), self.strides(), self.options());
    } else if (self.is_contiguous(at::MemoryFormat::ChannelsLast)) {
      out = at::empty(self.sizes(), self.options().memory_format(at::MemoryFormat::ChannelsLast));
    } else if (self.is_contiguous(at::MemoryFormat::ChannelsLast3d)) {
      out = at::empty(self.sizes(), self.options().memory_format(at::MemoryFormat::ChannelsLast3d));
    } else {
      out = at::empty(self.sizes(), self.options().memory_format(at::MemoryFormat::Contiguous));
    }
  } else {
    out = at::empty(self.sizes(), self.options().memory_format(mf_req));
  }

  if (direct_d2d_eligible && nbytes > 0 && out.is_contiguous() && out.strides() == self.strides()) {
    // Bypass aten::copy_ dispatch — write directly into the fresh output
    // buffer. Both self and out live on the same RBLN device (Tensor::options
    // preserves device), so this is always a same-device v2v. Require
    // matching strides so the byte order is identical; if the requested
    // memory format reshuffles strides (e.g. Preserve on a strided view
    // gives empty_strided with non-contig strides), fall through to the
    // copy_ path which honours strides.
    c10::rbln::memcpy_v2v(out.data_ptr(), self.data_ptr(), nbytes);
  } else {
    // Non-contig / partial-storage view: keep the default composite path
    // so aten::copy_'s TensorIterator handles the strided gather.
    out.copy_(self);
  }
  return out;
}

} // namespace at::native::rbln
