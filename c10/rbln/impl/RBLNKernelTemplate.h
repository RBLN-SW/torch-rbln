#pragma once

// Generic Option B kernel harness. Each ATen op maps onto this by providing:
//
//   - a unique ``OpId`` for cache keying
//   - a Python bootstrap function name (returns dict; see _b_bootstrap.py)
//   - a guard predicate (V1 restrictions)
//   - a fallback callable that routes to the legacy Python path (typically via
//     the corresponding ``aten::op.out`` overload)
//
// The hot-path shape (cache lookup -> at::empty -> Prepare{Inputs,Outputs} ->
// Run) is shared across every op and lives in ``run_kernel`` below.

#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/empty.h>
#include <c10/rbln/impl/RBLNKernelCache.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/utils/pybind.h>

#include <array>
#include <cstdint>
#include <map>
#include <string>

namespace c10::rbln::kcache {

// Reentrancy guard: while a cache miss is inside Python driving torch.compile,
// any ATen dispatch that lands back on a B kernel must take the slow path.
// Process-wide thread_local (single definition in RBLNKernelCache.cpp).
extern thread_local bool g_building_entry;

// Process-wide enable flag. Single definition in RBLNKernelCache.cpp.
extern std::atomic<bool> g_b_enabled;

struct BuildGuard {
  BuildGuard() { g_building_entry = true; }
  ~BuildGuard() { g_building_entry = false; }
};

at::ScalarType dtype_from_rbln_string(const std::string& s);

// Build a cache entry by driving torch.compile via _b_bootstrap.py. The
// bootstrap function at ``builder_attr`` must accept the sample input tensors
// and return a dict shaped like _b_bootstrap._runtime_info().
//
// The variadic ``sample_inputs`` are forwarded to Python as positional args.
template <typename... SampleTs>
CacheEntry build_entry_via_python(const char* builder_attr, const SampleTs&... sample_inputs) {
  namespace py = pybind11;
  py::gil_scoped_acquire gil;
  py::module_ mod = py::module_::import("torch_rbln._internal._b_bootstrap");
  py::object result = mod.attr(builder_attr)(sample_inputs...);
  py::dict info = py::cast<py::dict>(result);

  CacheEntry entry;
  entry.py_dynamo_runtime = py::object(info["dyn_runtime"]);
  auto raw_ptr = py::cast<uintptr_t>(info["runtime_raw_ptr"]);
  entry.runtime = reinterpret_cast<::rbln::PyRblnSyncRuntime*>(raw_ptr);
  entry.num_inputs = py::cast<uint32_t>(info["num_inputs"]);
  entry.num_outputs = py::cast<uint32_t>(info["num_outputs"]);

  py::list profiles = info["out_profiles"].cast<py::list>();
  entry.out_profiles.reserve(entry.num_outputs);
  for (auto handle_p : profiles) {
    py::dict p = py::reinterpret_borrow<py::dict>(handle_p);
    OutProfile op;
    op.shape = py::cast<std::vector<int64_t>>(p["shape"]);
    op.dtype = dtype_from_rbln_string(py::cast<std::string>(p["dtype"]));
    op.is_rbln_device = (py::cast<std::string>(p["device"]) == "rbln");
    entry.out_profiles.push_back(std::move(op));
  }
  return entry;
}

// Execute a single-output kernel given N input tensors already validated.
// Returns the freshly allocated output tensor (rbln device).
template <size_t N>
at::Tensor run_cached(const CacheEntry& entry, const std::array<const at::Tensor*, N>& ins) {
  TORCH_CHECK(entry.num_outputs == 1, "Option B template: expected 1 output, got ", entry.num_outputs);
  const OutProfile& op = entry.out_profiles[0];
  TORCH_CHECK(op.is_rbln_device, "Option B template: expected rbln-device output");

  auto out = at::empty(
      op.shape,
      at::TensorOptions().dtype(op.dtype).device(ins[0]->device()));

  std::map<uint32_t, uint64_t> dev_in;
  for (uint32_t i = 0; i < N; ++i) {
    dev_in.emplace(i, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ins[i]->data_ptr())));
  }
  std::map<uint32_t, uintptr_t> cpu_in;
  std::map<uint32_t, uint64_t> dev_out;
  dev_out.emplace(0u, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(out.data_ptr())));
  std::map<uint32_t, uintptr_t> cpu_out;

  entry.runtime->PrepareInputs(dev_in, cpu_in);
  entry.runtime->PrepareOutputs(dev_out, cpu_out);
  entry.runtime->Run();
  return out;
}

// V1 tensor predicate shared by every kernel. Extra per-op checks (alpha,
// broadcasting) are left to the caller's guard.
inline bool tensor_is_v1_safe(const at::Tensor& t) {
  if (t.scalar_type() != at::kHalf) return false;
  if (!t.is_contiguous()) return false;
  if (t.storage_offset() != 0) return false;
  if (t.device().type() != c10::DeviceType::PrivateUse1) return false;
  if (t.requires_grad()) return false;
  if (t.numel() == 0) return false;
  return true;
}

}  // namespace c10::rbln::kcache
