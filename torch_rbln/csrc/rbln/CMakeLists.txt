set(target torch_rbln_csrc_rbln)

find_package(Torch REQUIRED)

# C++ kernels for the eager hot path. They use pybind11 (for the miss-path
# bootstrap into Python) so they belong in the Python extension module, not
# in libc10_rbln.so.
set(c_kernel_sources
  kernels/RBLNKernelCache.cpp
  kernels/AddKernel.cpp
  kernels/SubKernel.cpp
  kernels/MulKernel.cpp
  kernels/DivKernel.cpp
  kernels/NegKernel.cpp
)

pybind11_add_module(${target}
  Module.cpp
  ../distributed/c10d/rbln/ProcessGroupRBLNModule.cpp
  ${c_kernel_sources}
)


find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")

target_link_libraries(${target} PUBLIC torch_rbln ${TORCH_PYTHON_LIBRARY})

# Enable hot-path breakdown counters (for bench). Off by default in release
# builds; turn on via -DTORCH_RBLN_C_KERNEL_TIMING=ON to the top-level cmake.
if(TORCH_RBLN_C_KERNEL_TIMING)
  target_compile_definitions(${target} PRIVATE C10_RBLN_C_KERNEL_TIMING=1)
endif()
set_target_properties(${target} PROPERTIES
  OUTPUT_NAME _C
  PREFIX ""
  INSTALL_RPATH "$ORIGIN/lib:$ORIGIN/../tvm:$ORIGIN/../torch/lib"
)
install(TARGETS ${target}
  DESTINATION .
)


# ---------------------------------------------------------------------------
# Type stub generation (_C.pyi)
#
# stubgen imports the built _C.so to inspect its API, which requires:
#   PYTHONPATH      → directory containing _C.*.so
#   LD_LIBRARY_PATH → libtorch_rbln.so, libtorch.so, etc.
# ---------------------------------------------------------------------------
set(target torch_rbln_csrc_rbln_stubgen)

find_program(STUBGEN stubgen REQUIRED)
add_custom_target(${target} ALL
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/_C.pyi
)
add_custom_command(
  DEPENDS $<TARGET_FILE:torch_rbln_csrc_rbln>
  COMMAND ${CMAKE_COMMAND} -E env
    "PYTHONPATH=${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
    "LD_LIBRARY_PATH=${CMAKE_LIBRARY_OUTPUT_DIRECTORY}:${TORCH_INSTALL_PREFIX}/lib:$ENV{LD_LIBRARY_PATH}"
    ${STUBGEN} -m _C -o ${CMAKE_CURRENT_BINARY_DIR}
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/_C.pyi
  WORKING_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
  COMMENT "Generating stubs for ${PROJECT_NAME}"
)
add_dependencies(${target} torch_rbln_csrc_rbln)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/_C.pyi
  DESTINATION _C
  RENAME __init__.pyi
)

# Distributed functions are now integrated into the main _C module
