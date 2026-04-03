set(target torch_rbln)
set(sources
  native/rbln/RBLNCPUFallback.cpp
  native/rbln/RBLNCopy.cpp
  native/rbln/RBLNResize.cpp
  native/rbln/RBLNTensorFactories.cpp
  native/rbln/RBLNTensorUtils.cpp
  native/RBLNRegisterOps.cpp
)
set(headers
  native/rbln/RBLNCPUFallback.h
  native/rbln/RBLNCopy.h
  native/rbln/RBLNResize.h
  native/rbln/RBLNTensorFactories.h
  native/rbln/RBLNTensorUtils.h
)
add_library(${target}
  ${sources}
  ${headers}
)

target_link_libraries(${target} PUBLIC
  c10_rbln
  c10d_rbln
)
set_target_properties(${target} PROPERTIES
  DEFINE_SYMBOL TORCH_RBLN_BUILD_MAIN_LIB
)
target_compile_options(${target} PRIVATE
  -fvisibility=hidden
)
target_include_directories(${target} SYSTEM PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
  $<INSTALL_INTERFACE:include>
)

install(TARGETS ${target}
  EXPORT Caffe2Targets
  DESTINATION lib
)

install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
  DESTINATION include
  FILES_MATCHING PATTERN "*.h"
  PATTERN "impl" EXCLUDE
  PATTERN "detail" EXCLUDE
)
