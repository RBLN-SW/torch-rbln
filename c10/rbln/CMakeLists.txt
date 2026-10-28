set(target c10_rbln)
set(sources
  impl/RBLNGuardImpl.cpp
  DeviceMappingManager.cpp
  RBLNAllocator.cpp
  RBLNFallbackConfig.cpp
  RBLNFunctions.cpp
  RBLNHostBatch.cpp
  RBLNGenerator.cpp
  RBLNHooksInterface.cpp
  RBLNLogging.cpp
  RBLNPinnedAllocator.cpp
  RBLNProfiler.cpp
  RBLNV2VBatch.cpp
)
set(headers
  impl/RBLNGuardImpl.h
  DeviceMappingManager.h
  RBLNFallbackConfig.h
  RBLNFunctions.h
  RBLNHostBatch.h
  RBLNGenerator.h
  RBLNHooksInterface.h
  RBLNLogging.h
  RBLNMacros.h
  RBLNPinnedAllocator.h
  RBLNProfiler.h
  RBLNSupportedDtypes.h
  RBLNV2VBatch.h
  detail/RBLNCopyBatchImpl.h
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
  spdlog::spdlog_header_only
)
torch_rbln_always_link(${target})
set_target_properties(${target} PROPERTIES
  DEFINE_SYMBOL C10_RBLN_BUILD_MAIN_LIB
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
)

install(TARGETS ${target}
  EXPORT TorchRBLNTargets
  DESTINATION lib
)
