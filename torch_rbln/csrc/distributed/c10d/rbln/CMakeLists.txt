set(target c10d_rbln)
set(sources
  ProcessGroupRBLN.cpp
)

set(headers
  ProcessGroupRBLN.hpp
)
add_library(${target}
  ${sources}
  ${headers}
)

target_link_libraries(${target} PUBLIC
  torch_rbln_interface
  ${TORCH_LIBRARIES}
  ${REBEL_LIBRARIES}
)
target_link_libraries(${target} PRIVATE
  c10_rbln
)
set_target_properties(${target} PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
)

target_include_directories(${target} SYSTEM PUBLIC
  ${TORCH_INCLUDE_DIRS}
  ${REBEL_INCLUDE_DIRS}
)

install(TARGETS ${target}
  EXPORT Caffe2Targets
  DESTINATION lib
)
