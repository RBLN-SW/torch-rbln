#pragma once

#include <Python.h>

namespace torch_rbln {
namespace distributed {

/**
 * @brief Initialize ProcessGroupRBLN Python bindings
 *
 * This function creates the _distributed_c10d submodule under torch_rbln._C
 * and binds the ProcessGroupRBLN class to it, inheriting from
 * torch._C._distributed_c10d.Backend.
 *
 * @return PyObject* Py_True on success, Py_False on failure
 */
PyObject* initialize_process_group_rbln_bindings(PyObject* _unused, PyObject* noargs);

/**
 * @brief Get Python method definitions for distributed functions
 *
 * Returns an array of PyMethodDef structures that define the Python
 * functions exposed by the distributed module.
 *
 * @return PyMethodDef* Array of method definitions
 */
PyMethodDef* get_distributed_method_definitions();

} // namespace distributed
} // namespace torch_rbln
