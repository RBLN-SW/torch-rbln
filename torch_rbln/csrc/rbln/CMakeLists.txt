set(target torch_rbln_csrc_rbln)

find_package(Torch REQUIRED)

pybind11_add_module(${target} Module.cpp ../distributed/c10d/rbln/ProcessGroupRBLNModule.cpp)


find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")

target_link_libraries(${target} PUBLIC torch_rbln ${TORCH_PYTHON_LIBRARY})
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
