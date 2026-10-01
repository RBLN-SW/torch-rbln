// #define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/stack.h>
#include <ATen/native/CPUFallback.h>
#include <ATen/native/rbln/RBLNCPUFallback.h>
#include <ATen/native/rbln/RBLNCPUFastPaths.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/_copy_from_and_resize.h>
#include <ATen/ops/_to_cpu.h>
#include <ATen/ops/empty.h>
#endif

#include <mutex>
#include <shared_mutex>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace at::native::rbln {

namespace {
// Per-op schema cache: same op handle reuses one FunctionSchema pointer for the
// process lifetime, so caching per-arg kinds + alias-write flags avoids walking
// schema_args every dispatch (LLaMA-1B eager hits this 10066x across <10 ops).
// Populate under unique_lock; readers take a shared_lock. OptionalTensor (`Tensor?`,
// e.g. linear's bias) must be classified explicitly — else the arg stays on the
// stack as an RBLN tensor under CPU dispatch and blows up at the next hop.
enum class CpuFbArgKind : uint8_t {
  Other = 0,
  Tensor,
  OptionalTensor,
  TensorList,
  OptionalTensorList,
  Device,
};

struct CpuFbSchemaInfo {
  std::vector<CpuFbArgKind> arg_kind; // per-positional arg
  std::vector<bool> is_write_alias; // alias_info != null && isWrite
  std::vector<bool> is_pure_out; // kwarg_only && name=="out" && is_write_alias
  bool populated = false;
};

struct CpuFbSchemaCache {
  std::shared_mutex mu;
  std::unordered_map<const c10::FunctionSchema*, CpuFbSchemaInfo> by_schema;
};

CpuFbSchemaCache& schema_cache() {
  static auto* c = new CpuFbSchemaCache();
  return *c;
}

const CpuFbSchemaInfo& get_or_populate_schema_info(const c10::FunctionSchema& schema) {
  auto& cache = schema_cache();
  const auto* key = &schema;
  {
    std::shared_lock<std::shared_mutex> rd(cache.mu);
    auto it = cache.by_schema.find(key);
    if (it != cache.by_schema.end() && it->second.populated) {
      return it->second;
    }
  }
  std::unique_lock<std::shared_mutex> wr(cache.mu);
  auto& info = cache.by_schema[key];
  if (info.populated) {
    return info;
  }
  const auto& args = schema.arguments();
  const auto n = args.size();
  info.arg_kind.assign(n, CpuFbArgKind::Other);
  info.is_write_alias.assign(n, false);
  info.is_pure_out.assign(n, false);
  for (size_t i = 0; i < n; ++i) {
    const auto& a = args[i];
    const auto& type = a.type();
    using namespace c10;
    if (type->isSubtypeOf(*TensorType::get())) {
      info.arg_kind[i] = CpuFbArgKind::Tensor;
    } else if (type->isSubtypeOf(*ListType::ofTensors())) {
      info.arg_kind[i] = CpuFbArgKind::TensorList;
    } else if (type->isSubtypeOf(*ListType::ofOptionalTensors())) {
      info.arg_kind[i] = CpuFbArgKind::OptionalTensorList;
    } else if (auto opt = type->cast<OptionalType>(); opt && opt->getElementType()->isSubtypeOf(*TensorType::get())) {
      info.arg_kind[i] = CpuFbArgKind::OptionalTensor;
    } else if (type->kind() == TypeKind::DeviceObjType) {
      info.arg_kind[i] = CpuFbArgKind::Device;
    }
    const auto* alias = a.alias_info();
    const bool is_w = (alias != nullptr && alias->isWrite());
    info.is_write_alias[i] = is_w;
    info.is_pure_out[i] = is_w && a.kwarg_only() && a.name() == "out";
  }
  info.populated = true;
  return info;
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

  // Step 1: copy all non-CPU tensor inputs into CPU tensors and stage them on the stack at
  // the correct indices, switching on the cached per-arg schema kind.
  const auto& schema_info = get_or_populate_schema_info(op.schema());
  const auto n_args = arguments.size();
  for (size_t idx = 0; idx < n_args; ++idx) {
    const auto kind = schema_info.arg_kind[idx];
    if (kind == CpuFbArgKind::Other) {
      continue;
    }
    const auto& ivalue = arguments[idx];
    switch (kind) {
      case CpuFbArgKind::Tensor:
        tensor_args.push_back(ivalue.toTensor());
        tensor_args_indices.push_back(idx);
        break;
      case CpuFbArgKind::OptionalTensor:
        // `Tensor?` runtime IValue is Tensor when the optional has a value,
        // else None. Only stage the present case onto tensor_args; absent
        // (None) needs no transformation — the CPU kernel sees None too.
        if (ivalue.isTensor()) {
          tensor_args.push_back(ivalue.toTensor());
          tensor_args_indices.push_back(idx);
        }
        break;
      case CpuFbArgKind::TensorList: {
        tensorlist_args.push_back(ivalue.toTensorList());
        tensorlist_args_indices.push_back(idx);
        auto cpu_ivalue = c10::IValue(c10::List<at::Tensor>(to_cpu(ivalue.toTensorVector())));
        tensorlist_cpu_args.push_back(cpu_ivalue);
        (*stack)[arguments_begin + idx] = std::move(cpu_ivalue);
        break;
      }
      case CpuFbArgKind::OptionalTensorList: {
        optional_tensorlist_args.push_back(ivalue.toOptionalTensorList());
        optional_tensorlist_args_indices.push_back(idx);
        auto cpu_ivalue = c10::IValue(c10::List<std::optional<at::Tensor>>(to_cpu(ivalue.toOptionalTensorVector())));
        optional_tensorlist_cpu_args.push_back(cpu_ivalue);
        (*stack)[arguments_begin + idx] = c10::IValue(cpu_ivalue);
        break;
      }
      case CpuFbArgKind::Device:
        tgt_device = ivalue.toDevice();
        (*stack)[arguments_begin + idx] = c10::IValue(c10::Device(kCPU));
        break;
      case CpuFbArgKind::Other:
        break; // unreachable — gated above
    }
  }

  // A kwarg-only `out=` is only written, so it gets fresh CPU storage instead of a copy. An
  // in-place `self` is read too, and other output names (max.dim_max) are not audited.
  std::vector<at::Tensor> cpu_tensors(tensor_args.size());
  std::vector<at::Tensor> copied;
  std::vector<size_t> copied_indices;
  for (size_t i = 0; i < tensor_args.size(); ++i) {
    if (schema_info.is_pure_out[tensor_args_indices[i]] && tensor_args[i].defined()) {
      cpu_tensors[i] = at::empty(tensor_args[i].sizes(), tensor_args[i].options().device(at::kCPU));
    } else {
      copied.push_back(tensor_args[i]);
      copied_indices.push_back(i);
    }
  }
  if (!copied.empty()) {
    auto filled = to_cpu(copied);
    for (size_t k = 0; k < filled.size(); ++k) {
      cpu_tensors[copied_indices[k]] = std::move(filled[k]);
    }
  }

  for (const auto i : c10::irange(tensor_args_indices.size())) {
    auto idx = tensor_args_indices[i];
    (*stack)[arguments_begin + idx] = c10::IValue(cpu_tensors[i]);
  }

  // Step 2: call the underlying CPU implementation. A CPUFastPathRegistry micro-kernel that
  // accepts the op replaces the stack itself; otherwise the boxed CPU dispatcher runs it.
  auto fast_path_fn = CPUFastPathRegistry::instance().try_get(op.schema());
  const bool fast_path_taken = fast_path_fn != nullptr && fast_path_fn(cpu_tensors, stack, arguments_begin);
  if (!fast_path_taken) {
    op.redispatchBoxed(c10::DispatchKeySet(cpu_dispatch_key), stack);
  }

  // Step 3: copy the mutated inputs back to their devices, resizing any the kernel resized.
  for (const auto i : c10::irange(tensor_args_indices.size())) {
    if (schema_info.is_write_alias[tensor_args_indices[i]] && tensor_args[i].defined()) {
      at::_copy_from_and_resize(cpu_tensors[i], tensor_args[i]);
    }
  }

  for (const auto i : c10::irange(tensorlist_args_indices.size())) {
    auto tensorlist_idx = tensorlist_args_indices[i];
    const AliasInfo* alias_info = schema_args[tensorlist_idx].alias_info();
    if (alias_info != nullptr && alias_info->isWrite()) {
      const auto& cpu_list = tensorlist_cpu_args[i].toTensorVector();
      for (const auto idx : c10::irange(tensorlist_args[i].size())) {
        if (cpu_list[idx].defined()) {
          at::_copy_from_and_resize(cpu_list[idx], tensorlist_args[i][idx]);
        }
      }
    }
  }

  for (const auto i : c10::irange(optional_tensorlist_args_indices.size())) {
    auto tensorlist_idx = optional_tensorlist_args_indices[i];
    const AliasInfo* alias_info = schema_args[tensorlist_idx].alias_info();
    if (alias_info != nullptr && alias_info->isWrite()) {
      const auto& cpu_list = optional_tensorlist_cpu_args[i].toOptionalTensorList();
      for (const auto idx : c10::irange(optional_tensorlist_args[i].size())) {
        if (cpu_list[idx].has_value() && cpu_list[idx].value().defined()) {
          const std::optional<at::Tensor>& optional_tensor = optional_tensorlist_args[i][idx];
          at::_copy_from_and_resize(cpu_list[idx].value(), optional_tensor.value());
        }
      }
    }
  }

  // Step 4: convert CPU output tensors back to the original input device. For
  // mutable-alias outputs, move the ORIGINAL input tensor back onto the stack in
  // place of the temporary CPU output.
  //
  // Note [CPU Fallback Does Not Handle View Operators]
  // Immutable-alias outputs (view ops, e.g. `view_as(Tensor(a) self, ...) ->
  // Tensor(a)`) can't be handled: a view must return a tensor sharing the input's
  // storage, but our CPU temporary lives on a different device. We warn instead
  // (BC for XLA-style view ops that fall back to CPU).
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
          const auto& cpu_list = returns[idx].toTensorList().vec();
          std::vector<at::Tensor> tensors;
          tensors.reserve(cpu_list.size());

          for (const auto& tensor : cpu_list) {
            tensors.push_back(tensor.to(*tgt_device));
          }
          (*stack)[returns_begin + idx] = c10::IValue(c10::List<at::Tensor>(tensors));
        }
      }
    }
  }
}

} // namespace at::native::rbln
