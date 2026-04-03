# ==============================================================================
# FindPybind.cmake - PyTorch-compatible pybind11 configuration
# ==============================================================================
#
# PROBLEM:
# When torch-rbln uses a different pybind11 version than PyTorch, cross-module
# type inheritance fails with "Assertion 'parent_tinfo != nullptr' failed" error.
# This occurs because pybind11 maintains an internal type registry that is
# version-specific and not shared across modules built with different versions.
#
# ROOT CAUSE:
# - PyTorch (pip-installed) bundles pybind11 3.x (e.g., 3.0.1)
# - System pybind11 (pip install pybind11) is typically 2.x (e.g., 2.13.6)
# - pybind11 2.x and 3.x have incompatible ABIs and type registries
# - When ProcessGroupRBLN tries to inherit from torch._C._distributed_c10d.Backend,
#   pybind11 cannot find the parent type info because it was registered with
#   a different pybind11 version
#
# SOLUTION:
# Use PyTorch's bundled pybind11 to ensure ABI compatibility and shared type
# registry. This allows proper cross-module type inheritance for classes like
# ProcessGroupRBLN that inherit from PyTorch's Backend class.
#
# ==============================================================================

find_package(Python3
  COMPONENTS Interpreter Development
  REQUIRED
)

# Get Python site-packages path (used for both PyTorch and fallback pybind11)
if(NOT DEFINED PYTHON_SITE_PACKAGES)
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(NOT PYTHON_SITE_PACKAGES)
    message(FATAL_ERROR "Failed to get Python site packages")
  endif()
endif()

# Use TORCH_INCLUDE_DIRS from FindTorch.cmake if already defined
# Otherwise, derive from PYTHON_SITE_PACKAGES (same approach as FindTorch.cmake)
if(DEFINED TORCH_INCLUDE_DIRS)
  # TORCH_INCLUDE_DIRS is a list, get the first element
  list(GET TORCH_INCLUDE_DIRS 0 TORCH_INCLUDE_DIR)
else()
  set(TORCH_INCLUDE_DIR "${PYTHON_SITE_PACKAGES}/torch/include")
endif()

# Check if PyTorch's pybind11 exists (pybind11 3.x bundled with PyTorch 2.9+)
set(TORCH_PYBIND11_DIR "${TORCH_INCLUDE_DIR}/pybind11")
if(EXISTS "${TORCH_PYBIND11_DIR}")
  message(STATUS "Using PyTorch's bundled pybind11: ${TORCH_PYBIND11_DIR}")
  # Create imported target for PyTorch's pybind11
  add_library(pybind11::headers INTERFACE IMPORTED)
  set_target_properties(pybind11::headers PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${TORCH_INCLUDE_DIR};${Python3_INCLUDE_DIRS}"
  )
  add_library(pybind11::pybind11 INTERFACE IMPORTED)
  # On Linux, Python extension modules don't need to link libpython.
  # The Python interpreter already provides all required symbols.
  # Linking libpython.a (static) to a shared library causes linker errors.
  set_target_properties(pybind11::pybind11 PROPERTIES
    INTERFACE_LINK_LIBRARIES "pybind11::headers"
  )
  add_library(pybind11::module INTERFACE IMPORTED)
  set_target_properties(pybind11::module PROPERTIES
    INTERFACE_LINK_LIBRARIES "pybind11::pybind11"
  )

  # Get Python module extension (e.g., .cpython-312-x86_64-linux-gnu.so)
  if(NOT DEFINED PYTHON_MODULE_EXTENSION)
    execute_process(
      COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"
      OUTPUT_VARIABLE PYTHON_MODULE_EXTENSION
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
  endif()

  # Define pybind11_add_module function for compatibility
  function(pybind11_add_module target_name)
    add_library(${target_name} MODULE ${ARGN})
    target_link_libraries(${target_name} PRIVATE pybind11::module)
    set_target_properties(${target_name} PROPERTIES
      CXX_VISIBILITY_PRESET hidden
      VISIBILITY_INLINES_HIDDEN ON
      PREFIX "${PYTHON_MODULE_PREFIX}"
      SUFFIX "${PYTHON_MODULE_EXTENSION}"
    )
  endfunction()
else()
  # Fallback to system pybind11 if PyTorch's is not found
  message(WARNING "PyTorch's pybind11 not found, falling back to system pybind11")
  find_package(pybind11 REQUIRED
    PATHS ${PYTHON_SITE_PACKAGES}/pybind11/share/cmake
    NO_DEFAULT_PATH
  )
endif()
