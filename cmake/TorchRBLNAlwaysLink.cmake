include_guard(GLOBAL)

function(torch_rbln_always_link target)
  # The library registers the RBLN backend in static initializers, so it must
  # stay linked into consumers that reference none of its symbols.
  string(CONCAT link_flag
    "$<$<CXX_COMPILER_ID:GNU,Clang>:-Wl,--no-as-needed,"
    "\"$<TARGET_FILE:TorchRBLN::${target}>\",--as-needed>")
  target_link_libraries(${target} INTERFACE "${link_flag}")
endfunction()
