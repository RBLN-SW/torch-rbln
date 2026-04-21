set(target c10_rbln)
set(sources
  impl/RBLNGuardImpl.cpp
  impl/RBLNKernelCache.cpp
  impl/AddKernelB.cpp
  impl/SubKernelB.cpp
  impl/MulKernelB.cpp
  impl/DivKernelB.cpp
  impl/NegKernelB.cpp
  DeviceMappingManager.cpp
  RBLNAllocator.cpp
  RBLNFallbackConfig.cpp
  RBLNFunctions.cpp
  RBLNGenerator.cpp
  RBLNHooksInterface.cpp
  RBLNLogging.cpp
)
set(headers
  impl/RBLNGuardImpl.h
  impl/RBLNKernelCache.h
  impl/RBLNKernelTemplate.h
  DeviceMappingManager.h
  RBLNFallbackConfig.h
  RBLNFunctions.h
  RBLNGenerator.h
  RBLNHooksInterface.h
  RBLNLogging.h
  RBLNMacros.h
)
add_library(${target}
  ${sources}
  ${headers}
)

if(NOT DEFINED TORCH_FOUND OR NOT DEFINED REBEL_FOUND)
  message(FATAL_ERROR "Torch and Rebel are required.")
endif()

# libtorch_python provides the pybind11 type casters for at::Tensor, c10::Device,
# c10::Scalar, etc. Needed by the Option B kernel cache's miss path.
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib" NO_DEFAULT_PATH)

target_link_libraries(${target} PUBLIC
  torch_rbln_interface
  ${TORCH_LIBRARIES}
  ${REBEL_LIBRARIES}
)
target_link_libraries(${target} PRIVATE
  spdlog::spdlog_header_only
  pybind11::headers
  Python3::Module
  ${TORCH_PYTHON_LIBRARY}
)
set_target_properties(${target} PROPERTIES
  DEFINE_SYMBOL C10_RBLN_BUILD_MAIN_LIB
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
)
target_compile_options(${target} PRIVATE ${TORCH_CXX_FLAGS})
target_include_directories(${target} SYSTEM PUBLIC
  ${TORCH_INCLUDE_DIRS}
  ${REBEL_INCLUDE_DIRS}
)

install(TARGETS ${target}
  EXPORT Caffe2Targets
  DESTINATION lib
)
