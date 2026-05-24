#include <ATen/native/rbln/RBLNCPUFastPaths.h>

namespace at::native::rbln {

CPUFastPathRegistry& CPUFastPathRegistry::instance() {
  // Leak-on-exit pattern (same convention as DispatchShim singletons in
  // torch_rbln/csrc/rbln/DispatchShim.cpp): function-local statics run
  // their destructors after Py_Finalize() at process teardown, which
  // would tear down the handler map while a background thread could
  // still be inside cpu_fallback_rbln. Allocating with ``new`` and
  // never deleting keeps the storage alive until the OS reclaims the
  // process.
  static auto* p = new CPUFastPathRegistry();
  return *p;
}

void CPUFastPathRegistry::register_handler(const std::string& op_name, CPUFastPathHandler::Fn fn) {
  std::unique_lock<std::shared_mutex> wr(mu_);
  by_name_.emplace(op_name, fn);
}

bool CPUFastPathRegistry::has_handler_for_op(const std::string& op_name) const {
  std::shared_lock<std::shared_mutex> rd(mu_);
  return by_name_.find(op_name) != by_name_.end();
}

CPUFastPathHandler::Fn CPUFastPathRegistry::try_get(const c10::FunctionSchema& schema) const {
  // Fast path: shared lock, look up cached handler by schema pointer.
  {
    std::shared_lock<std::shared_mutex> rd(mu_);
    auto it = by_schema_.find(&schema);
    if (it != by_schema_.end()) {
      return it->second;
    }
  }
  // Slow path: schema seen for the first time. Resolve op name → handler
  // and cache the result (including the negative cache for ops without a
  // registered fast path) under the unique lock.
  std::unique_lock<std::shared_mutex> wr(mu_);
  // Re-check under the write lock in case another thread populated the
  // entry between the shared unlock and our acquisition.
  auto it = by_schema_.find(&schema);
  if (it != by_schema_.end()) {
    return it->second;
  }
  std::string name = schema.operator_name().name;
  const auto& overload = schema.overload_name();
  if (!overload.empty()) {
    name += "." + overload;
  }
  CPUFastPathHandler::Fn fn = nullptr;
  auto by_name_it = by_name_.find(name);
  if (by_name_it != by_name_.end()) {
    fn = by_name_it->second;
  }
  by_schema_.emplace(&schema, fn);
  return fn;
}

} // namespace at::native::rbln
