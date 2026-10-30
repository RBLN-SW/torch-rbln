set(target torch_rbln)
set(sources
  native/rbln/RBLNCPUFallback.cpp
  native/rbln/RBLNCPUFastPaths.cpp
  native/rbln/RBLNCopy.cpp
  native/rbln/RBLNRepeat.cpp
  native/rbln/RBLNResize.cpp
  native/rbln/RBLNStridedV2V.cpp
  native/rbln/RBLNTensorAdvancedIndexing.cpp
  native/rbln/RBLNTensorFactories.cpp
  native/rbln/RBLNTensorShape.cpp
  native/rbln/RBLNTensorUtils.cpp
  native/RBLNRegisterOps.cpp
  native/rbln/fast_paths/int_elementwise.cpp
)
set(headers
  native/rbln/RBLNCPUFallback.h
  native/rbln/RBLNCPUFastPaths.h
  native/rbln/RBLNCopy.h
  native/rbln/RBLNIndexUtils.h
  native/rbln/RBLNRepeat.h
  native/rbln/RBLNResize.h
  native/rbln/RBLNStridedV2V.h
  native/rbln/RBLNStrideUtils.h
  native/rbln/RBLNTensorAdvancedIndexing.h
  native/rbln/RBLNTensorFactories.h
  native/rbln/RBLNTensorShape.h
  native/rbln/RBLNTensorUtils.h
)
add_library(${target} SHARED
  ${sources}
  ${headers}
)
add_library(TorchRBLN::${target} ALIAS ${target})

target_link_libraries(${target} PUBLIC
  c10_rbln
  c10d_rbln
)
torch_rbln_always_link(${target})
set_target_properties(${target} PROPERTIES
  DEFINE_SYMBOL TORCH_RBLN_BUILD_MAIN_LIB
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
)
target_include_directories(${target} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
  $<INSTALL_INTERFACE:include>
)

install(TARGETS ${target}
  EXPORT TorchRBLNTargets
  DESTINATION lib
)

install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
  DESTINATION include
  FILES_MATCHING PATTERN "*.h"
  PATTERN "impl" EXCLUDE
  PATTERN "detail" EXCLUDE
)
