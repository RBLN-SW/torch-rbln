include_guard(GLOBAL)

function(torch_rbln_add_runtime_target)
  cmake_parse_arguments(PARSE_ARGV 0 arg "" "LIBRARY;INCLUDE_DIR" "")
  if(NOT TARGET TorchRBLN::rbln)
    add_library(TorchRBLN::rbln SHARED IMPORTED)
    set_target_properties(TorchRBLN::rbln PROPERTIES
      IMPORTED_LOCATION "${arg_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${arg_INCLUDE_DIR}"
    )
  endif()
endfunction()
