#include <ATen/native/rbln/RBLNCopy.h>
#include <ATen/native/rbln/RBLNTensorUtils.h>
#include <c10/rbln/DeviceMappingManager.h>
#include <c10/rbln/RBLNFallbackConfig.h>
#include <c10/rbln/RBLNFunctions.h>
#include <c10/rbln/RBLNLogging.h>
#include <torch/csrc/utils/pybind.h>
#include <torch_rbln/csrc/distributed/c10d/rbln/ProcessGroupRBLNModule.hpp>
#include <ATen/native/rbln/RBLNCPUFallback.h>
#include <torch_rbln/csrc/rbln/DispatchShim.h>
#include <torch_rbln/csrc/rbln/WarmCache.h>
#include <exception>
#include <vector>

namespace {

/**
 * @brief Add PyMethodDef array to a vector, removing null terminators
 *
 * This helper function concatenates PyMethodDef arrays by removing
 * the null terminator from the existing vector and appending new methods.
 * This is used to combine distributed method definitions with the main module.
 *
 * @param method_vector The vector to append methods to
 * @param method_definitions The PyMethodDef array to append
 */
void add_py_method_definitions(std::vector<PyMethodDef>& method_vector, PyMethodDef* method_definitions) {
  if (!method_vector.empty()) {
    // Remove nullptr terminator from existing vector
    method_vector.pop_back();
  }
  while (true) {
    method_vector.push_back(*method_definitions);
    if (!method_definitions->ml_name) {
      break;
    }
    method_definitions++;
  }
}

/**
 * @brief Register public device management API with Python
 *
 * This function registers the public device management functions
 * that are exposed to Python users for device operations.
 *
 * @param module The Python module to register the functions with
 */
void register_public_device_api(py::module_& module) {
  module.def("current_device", &c10::rbln::get_device_index, "Get the current device.");
  module.def("device_count", &c10::rbln::get_device_count, "Get the number of devices.");
  module.def("set_device", &c10::rbln::set_device_index, "Set the current device.");
  module.def(
      "physical_device_count",
      &c10::rbln::get_physical_device_count,
      "Get the number of physical devices (ignores RSD mode).");
  module.def(
      "_exchange_device",
      &c10::rbln::exchange_device_index,
      "Exchange the current device and return the original device.");

  // Memory management functions
  module.def("empty_cache", &c10::rbln::empty_cache, "Release all unoccupied cached memory.");
  module.def("memory_stats", &c10::rbln::memory_stats, "Get memory allocator statistics.");
  module.def(
      "reset_accumulated_memory_stats", &c10::rbln::reset_accumulated_memory_stats, "Reset accumulated memory stats.");
  module.def("reset_peak_memory_stats", &c10::rbln::reset_peak_memory_stats, "Reset peak memory stats.");

  // Register DeviceTopology structures
  py::class_<c10::rbln::DeviceTopologyEntry>(module, "DeviceTopologyEntry")
      .def_property_readonly("logical_device_index", &c10::rbln::DeviceTopologyEntry::getLogicalDeviceIndex)
      .def_property_readonly("physical_device_ids", &c10::rbln::DeviceTopologyEntry::getPhysicalDeviceIds)
      .def_property_readonly("is_aggregated", &c10::rbln::DeviceTopologyEntry::isAggregated);

  py::class_<c10::rbln::DeviceTopology>(module, "DeviceTopology")
      .def_property_readonly("entries", &c10::rbln::DeviceTopology::getEntries)
      .def_property_readonly("unused_physical_device_ids", &c10::rbln::DeviceTopology::getUnusedPhysicalDeviceIds);

  // Direct binding to DeviceMappingManager for device topology
  module.def(
      "_get_device_topology",
      []() -> c10::rbln::DeviceTopology { return c10::rbln::DeviceMappingManager::getInstance().getDeviceTopology(); },
      "Get the complete device topology.");
}

/**
 * @brief Register internal API functions with Python
 *
 * This function registers internal API functions that are used
 * by the RBLN backend but not directly exposed to end users.
 * These functions are prefixed with underscore to indicate they are internal.
 *
 * @param module The Python module to register the functions with
 */
void register_internal_api(py::module_& module) {
  // Tensor creation and manipulation functions
  module.def(
      "_create_tensor_from_ptr", &at::native::rbln::create_tensor_from_ptr, "Internal: create tensor from device ptr");

  // Mark the virtual memory as logically zero-initialized without allocating host memory.
  // Preferred implementation of aten::zero_ for large RBLN tensors (e.g. KV-cache).
  module.def(
      "_mark_zeros",
      [](uint64_t vaddr) {
        c10::rbln::mark_zeros(reinterpret_cast<const void*>(vaddr)); // NOLINT(performance-no-int-to-ptr)
      },
      "Internal: mark RBLN virtual memory as zero-initialized (no host alloc)");

  // Logging utilities
  module.def("_log_cpu_fallback", &c10::rbln::log_cpu_fallback, "Internal: log CPU fallback");

  // C++ dispatch shim: install a boxed C++ handler on PrivateUse1 for the given
  // op, with pre-check + cpu_fallback_rbln on fail and Python callback on pass.
  // `skip_dtype_args` names positional arg indices whose dtype must not be
  // compared to float16 (e.g. where.self_out's cond is bool).
  module.def(
      "_register_cpp_shim",
      &torch_rbln::shim::register_cpp_shim,
      "Internal: install a C++ dispatch shim for an op and register its Python impl",
      pybind11::arg("op_name"),
      pybind11::arg("py_fn"),
      pybind11::arg("skip_dtype_args") = std::vector<size_t>{});

  // Warm-cache API. The C++ shim populates a thread-local `pending` entry on
  // every miss-path dispatch; the generated Python wrapper calls
  // `_warmcache_install_pending` after a successful first compile + run so the
  // runtime is cached for subsequent invocations with the same input profile.
  module.def(
      "_warmcache_install_pending",
      &torch_rbln::shim::install_warmcache_from_pending,
      "Internal: install a warm-cache entry from the thread-local pending key "
      "set by the shim on the way into the miss path",
      pybind11::arg("dyn_runtime"),
      pybind11::arg("runtime_raw_ptr"),
      pybind11::arg("num_inputs"),
      pybind11::arg("num_outputs"),
      pybind11::arg("out_profiles"));

  module.def(
      "_warmcache_set_enabled",
      [](bool v) { torch_rbln::warmcache::WarmCache::instance().set_enabled(v); },
      "Internal: enable/disable the warm-runtime cache path globally");
  module.def(
      "_warmcache_is_enabled",
      []() { return torch_rbln::warmcache::WarmCache::instance().is_enabled(); },
      "Internal: query whether warm-cache is currently enabled");
  module.def(
      "_warmcache_size",
      []() { return torch_rbln::warmcache::WarmCache::instance().size(); },
      "Internal: number of entries in the warm-cache (debug/bench only)");
  module.def(
      "_warmcache_clear",
      []() { torch_rbln::warmcache::WarmCache::instance().clear(); },
      "Internal: drop all warm-cache entries (tests / benchmarks)");
  module.def(
      "_warmcache_is_building",
      []() { return torch_rbln::warmcache::WarmCache::is_building_entry(); },
      "Internal: true iff the current thread is inside the miss-path compile");
  module.def(
      "_warmcache_enter_building",
      []() { torch_rbln::warmcache::WarmCache::enter_building(); },
      "Internal: mark the current thread as inside the miss-path compile "
      "(reentrancy guard; pairs with _warmcache_exit_building)");
  module.def(
      "_warmcache_exit_building",
      []() { torch_rbln::warmcache::WarmCache::exit_building(); },
      "Internal: clear the miss-path reentrancy flag set by _warmcache_enter_building");

  // DIAG: cpu_fallback_rbln per-stage timing
  module.def("_cpu_fallback_diag_dump", &at::native::rbln::diag_dump_cpu_fallback_stages,
             "DIAG: dump (calls, ns_setup, ns_dispatch, ns_writeback, ns_release) for cpu_fallback_rbln");
  module.def("_cpu_fallback_diag_reset", &at::native::rbln::diag_reset_cpu_fallback_stages,
             "DIAG: reset cpu_fallback_rbln stage timers");

  // Fallback configuration
  module.def(
      "_is_fallback_disabled",
      &c10::rbln::is_fallback_disabled,
      "Internal: check if specified fallback category is disabled");
}

/**
 * @brief Register distributed method definitions with the module
 *
 * This function adds distributed method definitions to the global method vector.
 * These methods are used for distributed training functionality such as
 * ProcessGroupRBLN initialization.
 *
 * @param method_vector The global method vector to add distributed methods to
 */
void register_distributed_method(std::vector<PyMethodDef>& method_vector) {
  add_py_method_definitions(method_vector, torch_rbln::distributed::get_distributed_method_definitions());
}

} // anonymous namespace

// Global vector to store all method definitions
static std::vector<PyMethodDef> global_method_definitions;

/**
 * @brief Initialize the torch_rbln._C module
 *
 * This function creates the main torch_rbln._C module and registers:
 * 1. Distributed functions from ProcessGroupRBLNModule
 * 2. RBLN-specific bindings
 *
 * REFACTORING NOTE - Circular Dependency Resolution:
 * ================================================
 * This implementation was refactored to resolve circular dependency issues
 * that occurred when trying to import torch._C._distributed_c10d during module
 * initialization. The original approach caused build errors related to _C.pyi
 * generation and import conflicts.
 *
 * Key changes made:
 * 1. Separated ProcessGroupRBLN bindings into ProcessGroupRBLNModule.cpp/hpp
 * 2. Used lazy loading pattern - distributed bindings are initialized on-demand
 * 3. Structured registration into logical functions for better maintainability:
 *    - register_distributed_method(): Adds distributed method definitions
 *    - register_public_device_api(): Registers public device functions
 *    - register_internal_api(): Registers internal backend functions
 *
 * This approach follows the ascend-torch pattern and ensures clean separation
 * of concerns while avoiding circular import issues.
 *
 * The initialization is organized into logical sections for better maintainability:
 * - Method definitions setup
 * - Module creation
 * - RBLN-specific bindings registration
 *
 * @return PyObject* The created module
 */
extern "C" PyObject* initModule() {
  // Step 1: Register distributed method definitions
  register_distributed_method(global_method_definitions);

  // Step 2: Create the module definition
  static struct PyModuleDef torch_rbln_module_definition = {
      PyModuleDef_HEAD_INIT, "torch_rbln._C", nullptr, -1, global_method_definitions.data()};
  PyObject* created_module = PyModule_Create(&torch_rbln_module_definition);

  // Step 3: Initialize RBLN-specific bindings
  py::gil_scoped_acquire gil_acquire;
  py::module_ python_module = py::reinterpret_borrow<py::module_>(created_module);

  // Set module documentation
  python_module.doc() = "Torch RBLN low‑level bindings.";

  // Step 4: Register all RBLN components
  register_public_device_api(python_module);
  register_internal_api(python_module);

  c10::rbln::register_rbln_device_mapping_initialized_callback([]() {
    py::gil_scoped_acquire gil;
    try {
      py::module_ m = py::module_::import("torch_rbln.device.device");
      m.attr("_on_device_mapping_ready_from_cpp")();
    } catch (const py::error_already_set&) {
      PyErr_Clear();
    } catch (const std::exception& e) {
      RBLN_LOG_DEBUG("device_mapping_ready_callback: {}", e.what());
    } catch (...) {
      RBLN_LOG_DEBUG("device_mapping_ready_callback: unknown exception");
    }
  });

  return created_module;
}

/**
 * @brief Python module initialization entry point
 *
 * This is the standard Python C extension entry point that gets called
 * when the module is imported. It delegates to initModule() for the
 * actual initialization work.
 *
 * @return PyObject* The initialized module
 */
PyMODINIT_FUNC PyInit__C(void) {
  return initModule();
}
