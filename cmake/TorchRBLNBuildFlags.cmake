include_guard(DIRECTORY)

block()
  get_filename_component(repository_dir "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
  get_target_property(runtime_include_dir TorchRBLN::rbln
    INTERFACE_INCLUDE_DIRECTORIES)
  # GCC applies the last matching map, so nested paths go after their parents.
  set(prefix_maps
    -ffile-prefix-map=${repository_dir}/=
    -ffile-prefix-map=${CMAKE_BINARY_DIR}/=
    -ffile-prefix-map=${TORCH_INSTALL_PREFIX}/=torch/
    -ffile-prefix-map=${runtime_include_dir}/=rebel/include/
  )
  if(TARGET TorchRBLN::torch_rbln_interface)
    get_target_property(torch_rbln_include_dir TorchRBLN::torch_rbln_interface
      INTERFACE_INCLUDE_DIRECTORIES)
    list(APPEND prefix_maps
      -ffile-prefix-map=${torch_rbln_include_dir}/=torch_rbln/include/)
  endif()
  add_compile_options(${prefix_maps})

  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(
      $<$<NOT:$<CONFIG:Debug>>:-D_FORTIFY_SOURCE=2>
      -fstack-protector-strong
      -fstack-clash-protection
    )
    add_link_options(
      -Wl,-z,relro
      -Wl,-z,now
      -Wl,-z,noexecstack
      -Wl,--as-needed
    )
  endif()
endblock()
