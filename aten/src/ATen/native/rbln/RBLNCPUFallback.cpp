// #define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/stack.h>
#include <ATen/native/CPUFallback.h>
#include <ATen/native/rbln/RBLNCopy.h>
#include <ATen/native/rbln/RBLNTensorUtils.h>
#include <c10/rbln/RBLNFunctions.h>
#include <c10/rbln/RBLNLogging.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/_copy_from_and_resize.h>
#include <ATen/ops/_to_cpu.h>
#include <ATen/ops/from_blob.h>
#endif

#include <cstdlib>
#include <sstream>
#include <string_view>
#include <vector>

namespace at::native::rbln {

namespace {

// Gate the zero-copy CPU-fallback alias path. When enabled, integer rbln
// tensors that are read-only inputs of an op are exposed to the CPU op as
// at::from_blob aliases over rebel's borrowed host pointer instead of being
// materialised through memcpy_v2h. See project_session_2026_04_22_end.md.
bool is_borrow_alias_enabled() {
  static const bool enabled = []() {
    const auto* env = std::getenv("TORCH_RBLN_BORROW_ALIAS");
    return env != nullptr && std::string_view(env) == "1";
  }();
  return enabled;
}

// Opt-in extension: also alias write inputs (mutable alias args). Measured to
// regress Llama 1B step time by ~40% when enabled unconditionally (likely
// because rebel has to sync the device side on return_borrowed(updated=true)
// for large tensors). Keep this off by default until we understand the cost,
// and expose the knob so we can iterate. Requires TORCH_RBLN_BORROW_ALIAS=1.
bool is_borrow_alias_write_enabled() {
  static const bool enabled = []() {
    const auto* env = std::getenv("TORCH_RBLN_BORROW_ALIAS_WRITE");
    return env != nullptr && std::string_view(env) == "1";
  }();
  return enabled;
}

bool is_borrow_aliasable_dtype(c10::ScalarType dtype) {
  // Only integer/boolean dtypes where raw host bytes round-trip cleanly; rebel
  // never reinterprets these the way it may for float16. fp dtypes stay on the
  // existing device-backed path and continue to flow through NPU compile.
  switch (dtype) {
    case c10::kBool:
    case c10::kByte:
    case c10::kChar:
    case c10::kShort:
    case c10::kInt:
    case c10::kLong:
      return true;
    default:
      return false;
  }
}

bool can_borrow_alias(const at::Tensor& t) {
  if (!t.defined()) return false;
  if (!t.device().is_privateuseone()) return false;
  if (!is_borrow_aliasable_dtype(t.scalar_type())) return false;
  // Contiguous required so at::from_blob sees a linear memory region. Non-
  // contiguous views have interior strides that rebel's borrow cannot expose
  // as a single host-backed range; let those fall back through memcpy_v2h.
  if (!t.is_contiguous()) return false;
  if (t.numel() == 0) return false;
  // View-safe: we borrow the whole storage and offset into it. Sanity-check
  // that the view lies inside the storage (it always should).
  const auto element_nbytes = static_cast<size_t>(t.numel()) * t.element_size();
  const auto offset_nbytes = static_cast<size_t>(t.storage_offset()) * t.element_size();
  if (offset_nbytes + element_nbytes > t.storage().nbytes()) return false;
  return true;
}

// Borrow the backing host pointer of an rbln tensor and wrap it as a CPU
// tensor via at::from_blob. We borrow the full storage (not just the tensor
// slice) so views with storage_offset > 0 or sub-storage extent still work:
// rebel only knows about storage-level v-memory entries, and re-borrowing the
// same storage twice in the same fallback call is safe because each borrow
// gets its own borrow_id. ``writable`` threads the schema's isWrite() bit
// through to return_borrowed so rebel marks the host side as latest only for
// mutating ops.
at::Tensor borrow_alias_as_cpu(const at::Tensor& rbln_t, bool writable) {
  uint64_t borrow_id = 0;
  const auto storage_nbytes = rbln_t.storage().nbytes();
  const auto element_size = rbln_t.element_size();
  const auto offset_nbytes = static_cast<size_t>(rbln_t.storage_offset()) * element_size;
  const uintptr_t storage_host_ptr = c10::rbln::borrow_host_ptr(
      rbln_t.storage().data_ptr().get(), storage_nbytes, borrow_id);
  const uintptr_t tensor_host_ptr = storage_host_ptr + offset_nbytes;
  auto deleter = [borrow_id, writable](void*) {
    c10::rbln::return_borrowed(borrow_id, /*updated=*/writable);
  };
  return at::from_blob(
      reinterpret_cast<void*>(tensor_host_ptr),
      rbln_t.sizes(),
      rbln_t.strides(),
      std::move(deleter),
      at::TensorOptions().dtype(rbln_t.scalar_type()).device(at::kCPU));
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

} // namespace

void cpu_fallback_rbln(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack,
    bool error_on_views,
    c10::DispatchKey cpu_dispatch_key) {
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
      auto cpu_ivalue = c10::IValue(c10::List<at::Tensor>(to_cpu(ivalue.toTensorVector())));
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
  // Build the CPU tensor view for each tensor argument. When the borrow-alias
  // gate is on, integer rbln tensors (read-only AND write mutable) are exposed
  // as at::from_blob wrappers over the rebel-borrowed host pointer (zero copy);
  // everything else still flows through the batched at::_to_cpu path.
  //
  // aliased_mask[i] records whether cpu_tensors[i] is a borrow alias; used in
  // Step 3 to skip the redundant _copy_from_and_resize on write inputs (the
  // alias deleter already marks the host side as latest via updated=true).
  std::vector<at::Tensor> cpu_tensors(tensor_args.size());
  std::vector<bool> aliased_mask(tensor_args.size(), false);
  std::vector<at::Tensor> fallback_inputs;
  std::vector<size_t> fallback_positions;
  fallback_inputs.reserve(tensor_args.size());
  fallback_positions.reserve(tensor_args.size());
  const bool alias_enabled = is_borrow_alias_enabled();
  const bool alias_write_enabled = is_borrow_alias_write_enabled();
  for (const auto i : c10::irange(tensor_args_indices.size())) {
    const auto idx = tensor_args_indices[i];
    const AliasInfo* alias_info = schema_args[idx].alias_info();
    const bool schema_is_write = (alias_info != nullptr && alias_info->isWrite());
    const bool aliasable = alias_enabled
                           && can_borrow_alias(tensor_args[i])
                           && (!schema_is_write || alias_write_enabled);
    if (aliasable) {
      cpu_tensors[i] = borrow_alias_as_cpu(tensor_args[i], /*writable=*/schema_is_write);
      aliased_mask[i] = true;
    } else {
      fallback_positions.push_back(i);
      fallback_inputs.push_back(tensor_args[i]);
    }
  }
  if (!fallback_inputs.empty()) {
    auto fallback_cpu = to_cpu(fallback_inputs);
    for (size_t j = 0; j < fallback_positions.size(); ++j) {
      cpu_tensors[fallback_positions[j]] = std::move(fallback_cpu[j]);
    }
  }

  for (const auto i : c10::irange(tensor_args_indices.size())) {
    auto idx = tensor_args_indices[i];
    (*stack)[arguments_begin + idx] = c10::IValue(cpu_tensors[i]);
  }

  // Step 2: Call the underlying CPU implementation of the operator
  op.redispatchBoxed(c10::DispatchKeySet(cpu_dispatch_key), stack);

  // Step 3: We need to take special care to handle mutable aliases properly:
  // If any input tensors are mutable aliases, we need to
  // directly copy the updated data on the CPU tensors back to the original inputs.
  for (const auto i : c10::irange(tensor_args_indices.size())) {
    auto tensor_idx = tensor_args_indices[i];
    const AliasInfo* alias_info = schema_args[tensor_idx].alias_info();
    if (alias_info != nullptr && alias_info->isWrite()) {
      if (!tensor_args[i].defined())
        continue;
      // Alias path already writes directly into the rbln-backed host pointer;
      // the deleter tagged the borrow with updated=true so rebel will sync on
      // the next device read. No explicit device-side copy needed.
      if (aliased_mask[i])
        continue;
      at::_copy_from_and_resize(cpu_tensors[i], tensor_args[i]);
    }
  }

  // We also need to explicit reapply input mutations to inputs that are lists
  // of tensors
  for (const auto i : c10::irange(tensorlist_args_indices.size())) {
    auto tensorlist_idx = tensorlist_args_indices[i];
    const AliasInfo* alias_info = schema_args[tensorlist_idx].alias_info();
    if (alias_info != nullptr && alias_info->isWrite()) {
      const auto& cpu_tensors = tensorlist_cpu_args[i].toTensorVector();
      for (const auto idx : c10::irange(tensorlist_args[i].size())) {
        if (!cpu_tensors[idx].defined())
          continue;
        at::_copy_from_and_resize(cpu_tensors[idx], tensorlist_args[i][idx]);
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
      // Copy the cpu output tensor to the original device.

      // We technically  might not have a target device, e.g. if you call
      // torch.cat() with an empty list In that case, we shouldn't have any
      // tensors to schlep across devices anyway.
      if (tgt_device) {
        if (returns[idx].isTensor() && returns[idx].toTensor().defined()) {
          (*stack)[returns_begin + idx] = c10::IValue(returns[idx].toTensor().to(*tgt_device));
        } else if (returns[idx].isTensorList() && validate_tensor_list(returns[idx].toTensorList())) {
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
}

} // namespace at::native::rbln
