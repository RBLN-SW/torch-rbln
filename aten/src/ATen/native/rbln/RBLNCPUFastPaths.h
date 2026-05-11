#pragma once

#include <c10/rbln/RBLNMacros.h>

#include <ATen/core/Tensor.h>
#include <ATen/core/function_schema.h>
#include <c10/util/ArrayRef.h>
#include <torch/csrc/jit/runtime/operator.h>

#include <cstddef>
#include <shared_mutex>
#include <string>
#include <unordered_map>

namespace at::native::rbln {

// Specialized host micro-kernel that bypasses ``redispatchBoxed(CPU)``'s
// TensorIterator + boxed dispatcher for a single aten op. Each handler
// validates its own dtype / contig / shape preconditions, runs a plain
// host loop, and replaces ``stack``'s arguments with the result tensor.
//
// Return value:
//   true  — fast path taken, ``stack`` already contains the single return.
//   false — preconditions not met or any guard tripped; caller falls back
//           to ``op.redispatchBoxed(CPU)``.
//
// Registration is per-operator-name via :func:`REGISTER_RBLN_CPU_FAST_PATH`;
// each handler lives in its own translation unit under
// ``aten/src/ATen/native/rbln/fast_paths/`` and self-registers via static
// initializer. Adding a fast path = add one .cpp file, no central edit.
struct CPUFastPathHandler {
  using Fn = bool (*)(
      c10::ArrayRef<at::Tensor> cpu_tensors,
      torch::jit::Stack* stack,
      size_t arguments_begin);
};

class CPUFastPathRegistry {
 public:
  static C10_RBLN_API CPUFastPathRegistry& instance();

  // Register at module-init. ``op_name`` is the schema-qualified operator
  // name (e.g. ``"aten::rsqrt.out"``).
  C10_RBLN_API void register_handler(const std::string& op_name, CPUFastPathHandler::Fn fn);

  // Per-call lookup. First call for a given FunctionSchema* warms the
  // schema cache; subsequent calls are shared_lock + hashmap O(1).
  // Returns ``nullptr`` when no handler is registered for the op.
  C10_RBLN_API CPUFastPathHandler::Fn try_get(const c10::FunctionSchema& schema) const;

  // Introspection: does a handler exist for ``op_name``? Used by tests to
  // verify the static-init registration fired without going through the
  // FunctionSchema lookup path.
  C10_RBLN_API bool has_handler_for_op(const std::string& op_name) const;

 private:
  CPUFastPathRegistry() = default;
  CPUFastPathRegistry(const CPUFastPathRegistry&) = delete;
  CPUFastPathRegistry& operator=(const CPUFastPathRegistry&) = delete;

  mutable std::shared_mutex mu_;
  // Source of truth, keyed by op-name string. Populated at static init.
  std::unordered_map<std::string, CPUFastPathHandler::Fn> by_name_;
  // Per-call cache, keyed by FunctionSchema pointer (stable for process
  // lifetime). Populated lazily on first lookup; serves shared_lock reads
  // thereafter.
  mutable std::unordered_map<const c10::FunctionSchema*, CPUFastPathHandler::Fn> by_schema_;
};

// Helper macro for handler-file self-registration. The static initializer
// runs at .so load time. Counter suffix keeps the variable name unique per
// translation unit; cross-TU collisions are impossible because each handler
// lives in its own .cpp.
#define REGISTER_RBLN_CPU_FAST_PATH_IMPL(op_name, fn, line)                  \
  static int _rbln_cpu_fast_path_reg_##line = []() {                         \
    ::at::native::rbln::CPUFastPathRegistry::instance().register_handler(    \
        op_name, fn);                                                        \
    return 0;                                                                \
  }()

#define REGISTER_RBLN_CPU_FAST_PATH_PROXY(op_name, fn, line) \
  REGISTER_RBLN_CPU_FAST_PATH_IMPL(op_name, fn, line)

#define REGISTER_RBLN_CPU_FAST_PATH(op_name, fn) \
  REGISTER_RBLN_CPU_FAST_PATH_PROXY(op_name, fn, __LINE__)

} // namespace at::native::rbln
