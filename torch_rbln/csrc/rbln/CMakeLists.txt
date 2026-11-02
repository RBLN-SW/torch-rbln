set(target torch_rbln_csrc_rbln)

pybind11_add_module(${target} Module.cpp DispatchShim.cpp WarmCache.cpp
  ../distributed/c10d/rbln/ProcessGroupRBLNModule.cpp
  profiler/kineto/rbln_kineto_emitter.cc
  profiler/kineto/rbln_kineto_adapter.cc)


find_library(TORCH_PYTHON_LIBRARY torch_python
  PATH "${TORCH_INSTALL_PREFIX}/lib")

target_link_libraries(${target} PUBLIC torch_rbln ${TORCH_PYTHON_LIBRARY})
set_target_properties(${target} PROPERTIES
  OUTPUT_NAME _C
  PREFIX ""
  INSTALL_RPATH
    "$ORIGIN/lib:$ORIGIN/../${RBLN_RUNTIME_RELDIR}:$ORIGIN/../torch/lib"
)
install(TARGETS ${target}
  DESTINATION .
)
