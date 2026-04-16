#include <ATen/native/rbln/RBLNCopy.h>
#include <ATen/native/rbln/RBLNTensorUtils.h>
#include <c10/rbln/DeviceMappingManager.h>
#include <c10/rbln/RBLNFallbackConfig.h>
#include <c10/rbln/RBLNFunctions.h>
#include <c10/rbln/RBLNLogging.h>
#include <torch/csrc/utils/pybind.h>
#include <torch_rbln/csrc/distributed/c10d/rbln/ProcessGroupRBLNModule.hpp>
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

  // Synchronization
  module.def("synchronize", &c10::rbln::synchronize, "Wait for all pending async transfers on a device.");

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

  // Logging utilities
  module.def("_log_cpu_fallback", &c10::rbln::log_cpu_fallback, "Internal: log CPU fallback");

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
