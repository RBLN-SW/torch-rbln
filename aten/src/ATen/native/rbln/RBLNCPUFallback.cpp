// #define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/stack.h>
#include <ATen/native/CPUFallback.h>
#include <ATen/native/rbln/RBLNCPUFallback.h>
#include <ATen/native/rbln/RBLNCopy.h>
#include <ATen/native/rbln/RBLNTensorUtils.h>

#include <c10/rbln/RBLNFunctions.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/_copy_from_and_resize.h>
#include <ATen/ops/_to_cpu.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/from_blob.h>
#endif

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

namespace at::native::rbln {

// DIAG: per-stage cumulative ns + per-stage call count
std::atomic<uint64_t> g_diag_calls{0};
std::atomic<uint64_t> g_diag_ns_setup{0};
std::atomic<uint64_t> g_diag_ns_dispatch{0};
std::atomic<uint64_t> g_diag_ns_writeback{0};
std::atomic<uint64_t> g_diag_ns_release{0};

std::tuple<uint64_t, uint64_t, uint64_t, uint64_t, uint64_t> diag_dump_cpu_fallback_stages() {
  return std::make_tuple(g_diag_calls.load(std::memory_order_relaxed),
                         g_diag_ns_setup.load(std::memory_order_relaxed),
                         g_diag_ns_dispatch.load(std::memory_order_relaxed),
                         g_diag_ns_writeback.load(std::memory_order_relaxed),
                         g_diag_ns_release.load(std::memory_order_relaxed));
}

void diag_reset_cpu_fallback_stages() {
  g_diag_calls.store(0, std::memory_order_relaxed);
  g_diag_ns_setup.store(0, std::memory_order_relaxed);
  g_diag_ns_dispatch.store(0, std::memory_order_relaxed);
  g_diag_ns_writeback.store(0, std::memory_order_relaxed);
  g_diag_ns_release.store(0, std::memory_order_relaxed);
}

namespace {

inline uint64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// convenience helper for converting tensors to cpu
template <
    typename T,
    std::enable_if_t<std::is_same_v<T, at::Tensor> || std::is_same_v<T, std::optional<at::Tensor>>, int> = 1>
std::vector<T> to_cpu(const std::vector<T>& tensors) {
  // We can't just call at::to_cpu() on the entire list of Tensors
  // Because it will break on undefined tensors. Separate out undefined tensors first.
  const int num = tensors.size();
  std::vector<T> cpu_tensors(num);
  std::vector<at::Tensor> valid_tensors;
  std::vector<bool> to_translate(num);
  for (const auto i : c10::irange(num)) {
    to_translate[i] = false;
    // Explicitly handling undefined tensors here instead of letting `at::_to_cpu` handle it.
    // Otherwise, we'd need to require all backends with their own implementation of _to_cpu
    // to properly handle undefined tensors.
    if constexpr (std::is_same_v<T, std::optional<at::Tensor>>) {
      if (tensors[i].has_value() && tensors[i].value().defined()) {
        const at::Tensor& tensor_ref = tensors[i].value();
        to_translate[i] = true;
        valid_tensors.push_back(tensors[i].value());
      } else {
        cpu_tensors[i] = tensors[i];
      }
    } else {
      if (tensors[i].defined()) {
        const at::Tensor& tensor_ref = tensors[i];
        to_translate[i] = true;
        valid_tensors.push_back(tensors[i]);
      } else {
        cpu_tensors[i] = tensors[i];
      }
    }
  }

  // copy device to cpu
  auto cpu_valid_tensors = at::_to_cpu(valid_tensors);
  for (int i = 0, defined_pos = 0; i < num; ++i) {
    if (to_translate[i]) {
      cpu_tensors[i] = std::move(cpu_valid_tensors[defined_pos++]);
    }
  }
  return cpu_tensors;
}

std::optional<c10::Device> compute_target_device(
    std::vector<at::Tensor>& t_args,
    const std::vector<c10::List<at::Tensor>>& tlist_args) {
  // Decide what device to move the output tensor(s) to.
  // The current convention is that we use the first tensor arg to pick the device
  // Barring that, we take the first tensor from a TensorList arg.
  if (!t_args.empty()) {
    return t_args[0].device();
  } else {
    // We need to loop through all of the (potentially multiple) TensorList arguments
    // In case, e.g. the first one is empty but the second is not.
    for (auto& tens_list : tlist_args) {
      for (const auto i : c10::irange(tens_list.size())) {
        return tens_list.get(i).device();
      }
    }
  }
  return std::nullopt;
}

bool validate_tensor_list(const c10::List<at::Tensor>& tensorlist) {
  bool flag = false;

  for (const auto& i : c10::irange(tensorlist.size())) {
    if (tensorlist[i].defined())
      flag = true;
  }

  return flag;
}

// Borrow a host pointer from the rbln virtual memory backing `t` and wrap it
// as a CPU tensor with the same sizes/strides. Writes the resulting borrow id
// into `borrow_id_out` (0 means "nothing to return"). Returns an undefined
// tensor if `t` isn't an rbln tensor — the caller falls back to the existing
// copy-based path for that slot.
at::Tensor borrow_rbln_as_cpu(const at::Tensor& t, uint64_t& borrow_id_out) {
  borrow_id_out = 0;
  if (!t.defined()) {
    return {};
  }
  if (t.device().type() != c10::DeviceType::PrivateUse1) {
    // CPU / other-device tensor: nothing to borrow.
    return {};
  }
  if (!t.is_contiguous()) {
    // Non-contiguous strided views may reach beyond `t.nbytes()` via stride
    // arithmetic; borrowing only `numel*itemsize` would alias an undersized
    // host region and corrupt reads. Caller falls back to the copy path.
    return {};
  }

  const uint64_t nbytes = t.nbytes();
  if (nbytes == 0) {
    // Zero-element / empty tensor: no host region to borrow. Use an empty CPU
    // tensor with matching dtype/shape instead.
    return at::empty(t.sizes(), t.options().device(at::kCPU));
  }

  auto borrowed = c10::rbln::borrow_host_ptr(t.data_ptr(), nbytes);
  borrow_id_out = borrowed.borrow_id;

  auto options = at::TensorOptions().dtype(t.dtype()).device(at::kCPU);
  return at::from_blob(reinterpret_cast<void*>(borrowed.host_ptr), t.sizes(), t.strides(), options);
}

// TensorList variant of borrow_rbln_as_cpu. Walks each element; rbln entries
// are borrowed (no D2H copy when host-latest), others fall through to the
// legacy batched at::_to_cpu. Borrow ids are appended to `borrow_ids_out` so
// the caller can release them after the op runs.
std::vector<at::Tensor> borrow_rbln_list_as_cpu(
    const std::vector<at::Tensor>& tensors,
    std::vector<uint64_t>& borrow_ids_out) {
  std::vector<at::Tensor> cpu_tensors(tensors.size());
  std::vector<at::Tensor> leftover; // non-rbln / empty — to_cpu later
  std::vector<size_t> leftover_indices;
  for (size_t i = 0; i < tensors.size(); ++i) {
    uint64_t bid = 0;
    cpu_tensors[i] = borrow_rbln_as_cpu(tensors[i], bid);
    borrow_ids_out.push_back(bid);
    if (bid == 0 && !cpu_tensors[i].defined()) {
      leftover.push_back(tensors[i]);
      leftover_indices.push_back(i);
    }
  }
  if (!leftover.empty()) {
    auto filled = to_cpu(leftover);
    for (size_t k = 0; k < filled.size(); ++k) {
      cpu_tensors[leftover_indices[k]] = std::move(filled[k]);
    }
  }
  return cpu_tensors;
}

} // namespace

void cpu_fallback_rbln(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack,
    bool error_on_views,
    c10::DispatchKey cpu_dispatch_key) {
  const uint64_t _diag_t0 = now_ns();
  g_diag_calls.fetch_add(1, std::memory_order_relaxed);
  TORCH_CHECK(
      c10::BackendComponent::CPUBit == c10::toBackendComponent(cpu_dispatch_key),
      "Expected CPU backend DispatchKey but got ",
      c10::toString(cpu_dispatch_key));
  auto& schema_args = op.schema().arguments();
  const auto num_arguments = schema_args.size();
  auto arguments = torch::jit::last(stack, num_arguments);
  const auto arguments_begin = stack->size() - num_arguments;

  std::vector<at::Tensor> tensor_args;
  std::vector<int> tensor_args_indices;

  std::vector<c10::List<at::Tensor>> tensorlist_args;
  std::vector<int> tensorlist_args_indices;

  std::vector<c10::List<std::optional<at::Tensor>>> optional_tensorlist_args;
  std::vector<int> optional_tensorlist_args_indices;

  std::optional<c10::Device> tgt_device = std::nullopt;
  // save converted cpu tensor for TensorList and optional TensorList
  std::vector<c10::IValue> tensorlist_cpu_args;
  std::vector<c10::IValue> optional_tensorlist_cpu_args;
  // Per-TensorList-arg borrow-id vectors. Entries with non-zero ids are
  // wrapped via the v-mem borrow path; zeros indicate the legacy `to_cpu`
  // fallback (used for non-rbln tensors, contiguity-guard skips, etc.).
  std::vector<std::vector<uint64_t>> tensorlist_borrow_ids;

  // Step 1: Convert all non-CPU tensor inputs into CPU tensors
  // and put them on the stack at the correct indices.
  for (const auto idx : c10::irange(arguments.size())) {
    const auto& ivalue = arguments[idx];
    if (ivalue.isTensor()) {
      tensor_args.push_back(ivalue.toTensor());
      tensor_args_indices.push_back(idx);
    } else if (ivalue.isTensorList()) {
      // Note: we copy each TensorList argument to CPU individually out of convenience,
      // but XLA would benefit from materializing all tensor and TensorList args onto the CPU at the same time.
      // We can improve this if we need better perf for XLA's CPU fallbacks.
      tensorlist_args.push_back(ivalue.toTensorList());
      tensorlist_args_indices.push_back(idx);
      auto rbln_list = ivalue.toTensorVector();
      std::vector<uint64_t> bids;
      auto cpu_list = borrow_rbln_list_as_cpu(rbln_list, bids);
      tensorlist_borrow_ids.push_back(std::move(bids));
      auto cpu_ivalue = c10::IValue(c10::List<at::Tensor>(cpu_list));
      tensorlist_cpu_args.push_back(cpu_ivalue);
      (*stack)[arguments_begin + idx] = std::move(cpu_ivalue);
    } else if (ivalue.isOptionalTensorList()) {
      optional_tensorlist_args.push_back(ivalue.toOptionalTensorList());
      optional_tensorlist_args_indices.push_back(idx);
      auto cpu_ivalue = c10::IValue(c10::List<std::optional<at::Tensor>>(to_cpu(ivalue.toOptionalTensorVector())));
      optional_tensorlist_cpu_args.push_back(cpu_ivalue);
      (*stack)[arguments_begin + idx] = c10::IValue(cpu_ivalue);
    } else if (ivalue.isDevice()) {
      tgt_device = ivalue.toDevice();
      (*stack)[arguments_begin + idx] = c10::IValue(c10::Device(kCPU));
    }
  }
  // Stage tensor args onto the stack as CPU views. We obtain a host pointer
  // into the existing vmem region (no D2H copy) and wrap it as a CPU tensor
  // via at::from_blob; borrow ids are tracked so we can release them after
  // the op runs. Slots that can't be borrowed (write-alias outputs, undefined
  // tensors, non-rbln tensors, contiguity-guard skips) fall through to the
  // batched `at::_to_cpu` copy path.
  std::vector<uint64_t> borrow_ids(tensor_args.size(), 0);
  std::vector<at::Tensor> cpu_tensors(tensor_args.size());

  for (size_t i = 0; i < tensor_args.size(); ++i) {
    // Skip the borrow fast path for write-alias outputs (`out=` tensors).
    // Borrowed tensors are wrapped via at::from_blob and have a fixed-size
    // host-mapped storage, so the CPU op's resize path (which fires for
    // wrong-shape `out=`, broadcasting binary ops, etc.) silently no-ops
    // and the op then writes broadcasted values into the unresized buffer.
    // Routing write-alias slots through the legacy `to_cpu` path gives
    // them fresh CPU storage that PyTorch core can resize freely.
    const auto schema_idx = tensor_args_indices[i];
    const auto* alias_info = schema_args[schema_idx].alias_info();
    const bool is_write_alias = (alias_info != nullptr && alias_info->isWrite());
    if (is_write_alias) {
      continue;
    }
    cpu_tensors[i] = borrow_rbln_as_cpu(tensor_args[i], borrow_ids[i]);
  }
  // Fill any slots that weren't borrowed (write-alias, undefined, non-rbln,
  // or contiguity guard) by routing them through the legacy batched copy.
  std::vector<at::Tensor> non_borrowed;
  std::vector<size_t> non_borrowed_indices;
  for (size_t i = 0; i < tensor_args.size(); ++i) {
    if (borrow_ids[i] == 0 && !cpu_tensors[i].defined()) {
      non_borrowed.push_back(tensor_args[i]);
      non_borrowed_indices.push_back(i);
    }
  }
  if (!non_borrowed.empty()) {
    auto filled = to_cpu(non_borrowed);
    for (size_t k = 0; k < filled.size(); ++k) {
      cpu_tensors[non_borrowed_indices[k]] = std::move(filled[k]);
    }
  }

  for (const auto i : c10::irange(tensor_args_indices.size())) {
    auto idx = tensor_args_indices[i];
    (*stack)[arguments_begin + idx] = c10::IValue(cpu_tensors[i]);
  }

  const uint64_t _diag_t1 = now_ns();
  g_diag_ns_setup.fetch_add(_diag_t1 - _diag_t0, std::memory_order_relaxed);

  // Step 2: Call the underlying CPU implementation of the operator
  op.redispatchBoxed(c10::DispatchKeySet(cpu_dispatch_key), stack);

  const uint64_t _diag_t2 = now_ns();
  g_diag_ns_dispatch.fetch_add(_diag_t2 - _diag_t1, std::memory_order_relaxed);

  // Step 3: Mutable alias write-back.
  // - Legacy path: the CPU op wrote into the fresh CPU tensor at cpu_tensors[i];
  //   we copy that result back into the original rbln tensor.
  // - Borrow path, in-place case (borrow_ids[i] != 0): the CPU op wrote
  //   directly into the borrowed host pointer, which already aliases the rbln
  //   tensor's host-backed vmem; we only need to mark the borrow as updated so
  //   the next device consumer lazily syncs.
  // - Borrow path, resized-empty case: the composite's functional wrapper
  //   allocates an empty out (numel=0) and lets the CPU kernel resize+fill it.
  //   borrow_rbln_as_cpu couldn't borrow a zero-size vaddr, so borrow_id is 0
  //   and cpu_tensors[i] holds fresh CPU storage. We resize the original rbln
  //   tensor to match, borrow its now-sized vmem, memcpy the CPU content, and
  //   return the borrow as updated — replacing the eager H2D that
  //   at::_copy_from_and_resize would do.
  std::vector<bool> borrow_write(tensor_args.size(), false);
  for (const auto i : c10::irange(tensor_args_indices.size())) {
    auto tensor_idx = tensor_args_indices[i];
    const AliasInfo* alias_info = schema_args[tensor_idx].alias_info();
    if (alias_info == nullptr || !alias_info->isWrite()) {
      continue;
    }
    if (!tensor_args[i].defined()) {
      continue;
    }
    if (borrow_ids[i] != 0) {
      borrow_write[i] = true;
      continue;
    }

    // Borrow-based write-back is only safe when BOTH the cpu_out and the rbln
    // out are contiguous. The cpu->host memcpy below copies `nbytes` of
    // contiguous bytes; if `rbln_out` were strided/noncontiguous, the borrow
    // would alias only the [vaddr, vaddr+nbytes) span — leaving the rest of
    // the strided storage stale. Caller's noncontiguous out= must go through
    // the legacy `_copy_from_and_resize` path which honors strides.
    const bool borrow_resize_case = tensor_args[i].device().type() == c10::DeviceType::PrivateUse1 &&
        cpu_tensors[i].defined() && cpu_tensors[i].is_contiguous() && cpu_tensors[i].nbytes() > 0 &&
        tensor_args[i].is_contiguous();
    if (borrow_resize_case) {
      auto& rbln_out = tensor_args[i];
      if (rbln_out.sizes() != cpu_tensors[i].sizes()) {
        rbln_out.resize_(cpu_tensors[i].sizes());
      }
      const uint64_t nbytes = rbln_out.nbytes();
      if (nbytes > 0) {
        // Acquire-for-overwrite: we immediately memcpy over the whole region, so any
        // D2H from a stale PHYSICAL_VIEW_IS_LATEST state would be thrown away.
        auto borrowed = c10::rbln::acquire_host_ptr_for_overwrite(rbln_out.data_ptr(), nbytes);
        std::memcpy(reinterpret_cast<void*>(borrowed.host_ptr), cpu_tensors[i].data_ptr(), nbytes);
        c10::rbln::return_borrowed(borrowed.borrow_id, /*updated=*/true);
      }
    } else {
      at::_copy_from_and_resize(cpu_tensors[i], tensor_args[i]);
    }
  }

  // We also need to explicit reapply input mutations to inputs that are lists
  // of tensors. On the borrow path any element with a non-zero borrow id will
  // be committed via `rbln_v_return_borrowed(updated=true)` at the release
  // step; for those we just mark instead of doing the eager H2D copy.
  std::vector<std::vector<bool>> tensorlist_borrow_write(tensorlist_args_indices.size());
  for (const auto i : c10::irange(tensorlist_args_indices.size())) {
    auto tensorlist_idx = tensorlist_args_indices[i];
    const AliasInfo* alias_info = schema_args[tensorlist_idx].alias_info();
    const auto& cpu_list = tensorlist_cpu_args[i].toTensorVector();
    tensorlist_borrow_write[i].assign(tensorlist_args[i].size(), false);
    if (alias_info != nullptr && alias_info->isWrite()) {
      for (const auto idx : c10::irange(tensorlist_args[i].size())) {
        if (!cpu_list[idx].defined())
          continue;
        const bool is_borrowed = i < tensorlist_borrow_ids.size() && idx < tensorlist_borrow_ids[i].size() &&
            tensorlist_borrow_ids[i][idx] != 0;
        if (is_borrowed) {
          tensorlist_borrow_write[i][idx] = true;
        } else {
          at::_copy_from_and_resize(cpu_list[idx], tensorlist_args[i][idx]);
        }
      }
    }
  }

  // We also need to explicit reapply input mutations to inputs that are lists
  // of optional tensors
  for (const auto i : c10::irange(optional_tensorlist_args_indices.size())) {
    auto tensorlist_idx = optional_tensorlist_args_indices[i];
    const AliasInfo* alias_info = schema_args[tensorlist_idx].alias_info();
    if (alias_info != nullptr && alias_info->isWrite()) {
      const auto& cpu_tensors = optional_tensorlist_cpu_args[i].toOptionalTensorList();
      for (const auto idx : c10::irange(optional_tensorlist_args[i].size())) {
        if (cpu_tensors[idx].has_value() && cpu_tensors[idx].value().defined()) {
          const std::optional<at::Tensor>& optional_tensor = optional_tensorlist_args[i][idx];
          at::_copy_from_and_resize(cpu_tensors[idx].value(), optional_tensor.value());
        }
      }
    }
  }

  const uint64_t _diag_t3 = now_ns();
  g_diag_ns_writeback.fetch_add(_diag_t3 - _diag_t2, std::memory_order_relaxed);

  // Release any vmem borrows issued on the input path. Write-alias inputs use
  // `updated=true` so the rbln tensor's host view becomes the latest source of
  // truth and the next device consumer triggers a lazy host→device sync.
  for (size_t i = 0; i < borrow_ids.size(); ++i) {
    c10::rbln::return_borrowed(borrow_ids[i], borrow_write[i]);
  }
  for (size_t i = 0; i < tensorlist_borrow_ids.size(); ++i) {
    for (size_t k = 0; k < tensorlist_borrow_ids[i].size(); ++k) {
      const bool upd = (i < tensorlist_borrow_write.size() && k < tensorlist_borrow_write[i].size())
          ? tensorlist_borrow_write[i][k]
          : false;
      c10::rbln::return_borrowed(tensorlist_borrow_ids[i][k], upd);
    }
  }

  // Step 4: Convert any CPU output tensors back to the original input device.
  // For mutable alias'd outputs, we also need to take special care
  // to move the ORIGINAL input tensor back onto the stack, in place of
  // the temporary CPU output tensor that we created.
  //
  // Note [CPU Fallback Does Not Handle View Operators]
  // Also note that we are incapable of handling immutable aliases properly.
  // Why?
  // Schemas with an immutable alias'd tensor outputs correspond to view operators.
  // For example, the `view_as` schema from native_functions.yaml:
  // `view_as(Tensor(a) self, Tensor other) -> Tensor(a)`
  // We can't handle these ops properly, because view ops are supposed to return
  // a NEW tensor that shares the SAME storage as the original tensor.
  // However, the new tensor that we created cannot share the same storage,
  // since it lives on CPU and the original tensor lives on a different device.
  // Because of that, we warn if someone attempts to call the
  // CPU fallback on a view operator (this is to maintain BC for view ops for XLA
  // that fall back to CPU).
  const auto& schema_returns = op.schema().returns();
  const auto& num_returns = schema_returns.size();
  auto returns = torch::jit::last(stack, num_returns);
  const auto returns_begin = stack->size() - num_returns;

  if (tgt_device == std::nullopt) {
    tgt_device = compute_target_device(tensor_args, tensorlist_args);
  }

  for (const auto idx : c10::irange(returns.size())) {
    const AliasInfo* alias_info = schema_returns[idx].alias_info();
    if (alias_info != nullptr && alias_info->isWrite()) {
      // Case (1): mutable alias case.
      // Move the input ivalue directly onto the stack in place of
      // the existing cpu output tensor.
      bool found_alias = false;
      if (returns[idx].isTensor() && returns[idx].toTensor().defined()) {
        // We could store some extra metadata on the function schema to avoid
        // the loop here if we need to improve perf.
        for (const auto i : c10::irange(tensor_args_indices.size())) {
          auto input_tensor_idx = tensor_args_indices[i];
          const auto& input_tensor = cpu_tensors[i];
          const AliasInfo* input_alias_info = schema_args[input_tensor_idx].alias_info();
          // Checked above; adding assert to guard against breakage of the below
          // condition due to changing the above if test.
          TORCH_INTERNAL_ASSERT_DEBUG_ONLY(alias_info != nullptr);
          if (input_tensor.defined() &&
              (alias_info == input_alias_info || (input_alias_info != nullptr && *alias_info == *input_alias_info))) {
            // We've found the original input tensor that aliases with the
            // current output. Wrap it in an IValue and put it directly on the
            // stack.
            (*stack)[returns_begin + idx] = c10::IValue(tensor_args[i]);
            found_alias = true;
            break;
          }
        }
      } else if (returns[idx].isTensorList() && validate_tensor_list(returns[idx].toTensorList())) {
        for (const auto i : c10::irange(tensorlist_args_indices.size())) {
          auto input_tensor_idx = tensorlist_args_indices[i];
          const AliasInfo* input_alias_info = schema_args[input_tensor_idx].alias_info();
          // Checked above; adding assert to guard against breakage of the below
          // condition due to changing the above if test.
          TORCH_INTERNAL_ASSERT_DEBUG_ONLY(alias_info != nullptr);
          if (validate_tensor_list(tensorlist_args[i]) &&
              (alias_info == input_alias_info || (input_alias_info != nullptr && *alias_info == *input_alias_info))) {
            // We've found the original input tensor that aliases with the
            // current output. Wrap it in an IValue and put it directly on the
            // stack.
            (*stack)[returns_begin + idx] = c10::IValue(tensorlist_args[i]);
            found_alias = true;
            break;
          }
        }
      }
      TORCH_CHECK(
          found_alias,
          "The operator ",
          op.schema().operator_name(),
          " appears to have invalid alias information. ",
          "Found a return tensor argument with a mismatched mutable alias: ",
          schema_returns[idx]);
    } else {
      if (alias_info != nullptr && !alias_info->isWrite()) {
        // Case (3): immutable alias (view) case.
        // Warn here, since we're copying and not creating a view.
        // If this operator is needed, the backend should provide a kernel for
        // it. See Note [CPU Fallback Does Not Handle View Operators]
        std::stringstream dev_str;
        if (tgt_device) {
          dev_str << *tgt_device;
        } else {
          dev_str << "<none>";
        }
        if (error_on_views) {
          TORCH_CHECK(
              false,
              "The operator ",
              op.schema().operator_name(),
              " appears to be a view operator, ",
              "but it has no implementation for the backend \"",
              dev_str.str(),
              "\". View operators don't support ",
              "since the tensor's storage cannot be shared across devices.");
        } else {
          TORCH_WARN(
              false,
              "The operator ",
              op.schema().operator_name(),
              " appears to be a view operator, ",
              "but it has no implementation for the backend \"",
              dev_str.str(),
              "\". View operators don't support falling back to run on the CPU, ",
              "since the tensor's storage cannot be shared across devices.");
        }
      }
      // Case (2): copy case.
      // Copy the cpu output tensor to the original device. On the v-mem
      // borrow path we allocate a fresh rbln tensor, borrow its host-backed
      // view, memcpy the CPU result into it, and return the borrow with
      // updated=true — that way the next device consumer triggers a lazy
      // host→device sync instead of an eager H2D copy here.

      // We technically  might not have a target device, e.g. if you call
      // torch.cat() with an empty list In that case, we shouldn't have any
      // tensors to schlep across devices anyway.
      if (tgt_device) {
        const bool use_borrow_out = tgt_device->type() == c10::DeviceType::PrivateUse1;
        if (returns[idx].isTensor() && returns[idx].toTensor().defined()) {
          const auto& cpu_out = returns[idx].toTensor();
          if (use_borrow_out && cpu_out.is_contiguous()) {
            auto rbln_out = at::empty(cpu_out.sizes(), cpu_out.options().device(*tgt_device));
            const uint64_t nbytes = rbln_out.nbytes();
            if (nbytes > 0) {
              // Acquire-for-overwrite: the tensor is a freshly-allocated at::empty;
              // any pre-existing data (in practice there is none, but if the
              // allocator ever caches a warm buffer the old device data is
              // irrelevant) will be overwritten by the memcpy below.
              auto borrowed = c10::rbln::acquire_host_ptr_for_overwrite(rbln_out.data_ptr(), nbytes);
              std::memcpy(reinterpret_cast<void*>(borrowed.host_ptr), cpu_out.data_ptr(), nbytes);
              c10::rbln::return_borrowed(borrowed.borrow_id, /*updated=*/true);
            }
            (*stack)[returns_begin + idx] = c10::IValue(rbln_out);
          } else {
            (*stack)[returns_begin + idx] = c10::IValue(cpu_out.to(*tgt_device));
          }
        } else if (returns[idx].isTensorList() && validate_tensor_list(returns[idx].toTensorList())) {
          // TensorList output: keep the legacy .to(device) path for now; the
          // borrow-based write-back can be extended here if profiling shows
          // meaningful cost.
          const auto& cpu_tensors = returns[idx].toTensorList().vec();
          std::vector<at::Tensor> tensors;
          tensors.reserve(cpu_tensors.size());

          for (const auto& tensor : cpu_tensors) {
            tensors.push_back(tensor.to(*tgt_device));
          }
          (*stack)[returns_begin + idx] = c10::IValue(c10::List<at::Tensor>(tensors));
        }
      }
    }
  }
  g_diag_ns_release.fetch_add(now_ns() - _diag_t3, std::memory_order_relaxed);
}

} // namespace at::native::rbln
