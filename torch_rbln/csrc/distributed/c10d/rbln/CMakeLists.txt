set(target c10d_rbln)
set(sources
  ProcessGroupRBLN.cpp
  # TEMPORARY: remove together with RdmaIpAutoDiscovery.{hpp,cpp} once
  # librbln-ccl performs RoCE GID auto-discovery internally.
  RdmaIpAutoDiscovery.cpp
)

set(headers
  ProcessGroupRBLN.hpp
)
add_library(${target} SHARED
  ${sources}
  ${headers}
)
add_library(TorchRBLN::${target} ALIAS ${target})

target_link_libraries(${target} PUBLIC
  torch_rbln_interface
)
target_link_libraries(${target} PRIVATE
  c10_rbln
)
set_target_properties(${target} PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
)

install(TARGETS ${target}
  EXPORT TorchRBLNTargets
  DESTINATION lib
)

install(FILES ${headers}
  DESTINATION include/torch_rbln/csrc/distributed/c10d/rbln
)
