# Finds the RBLN Runtime with the resolver `torch-rbln` uses at import time.
#
# Imported target:
#   `TorchRBLN::rbln`      - `librbln.so` and its headers
#
# Result variables:
#   `RBLN_FOUND`           - True if `librbln.so` and its headers are found
#   `RBLN_LIBRARY`         - Path to `librbln.so`
#   `RBLN_INCLUDE_DIR`     - Directory that contains the RBLN Runtime headers
#   `RBLN_RUNTIME_RELDIR`  - Directory, relative to the parent of the install
#                            prefix, where install RPATHs look for `librbln.so`
#   `RBLN_ABI_CURRENT`     - `RBLN_ABI_CURRENT` from `rbln_abi.h`, or `""` when
#                            `rebel-compiler` predates the ABI handshake
#
# Hints:
#   Set `Python3_EXECUTABLE` to choose the interpreter whose `rebel-compiler`
#   this module finds.
#   Set both `REBEL_HOME` and `RBLN_USE_EXTERNAL_REBEL_COMPILER` in the
#   environment to build against an external `rebel-compiler` tree.

include(FindPackageHandleStandardArgs)

set(_rbln_use_external_tree "$ENV{RBLN_USE_EXTERNAL_REBEL_COMPILER}")
set(_rbln_external_tree_dir "$ENV{REBEL_HOME}")
if(_rbln_use_external_tree OR _rbln_external_tree_dir)
  if(NOT _rbln_use_external_tree OR NOT _rbln_external_tree_dir)
    message(FATAL_ERROR
      "RBLN_USE_EXTERNAL_REBEL_COMPILER and REBEL_HOME must be set together: "
      "both to build against an external rebel-compiler tree, or neither to "
      "use the installed rebel-compiler. Got "
      "RBLN_USE_EXTERNAL_REBEL_COMPILER='${_rbln_use_external_tree}' and "
      "REBEL_HOME='${_rbln_external_tree_dir}'.")
  endif()
endif()

set(RBLN_INCLUDE_DIR "")
set(RBLN_LIBRARY "")
set(RBLN_RUNTIME_RELDIR "")
set(_rbln_failure_reason "")

if(NOT Python3_Interpreter_FOUND)
  find_package(Python3 COMPONENTS Interpreter QUIET)
endif()
if(NOT Python3_Interpreter_FOUND)
  string(CONCAT _rbln_failure_reason
    "Locating the RBLN Runtime needs a Python interpreter. Point "
    "Python3_EXECUTABLE at one.")
else()
  set(_rbln_resolver_args "")
  if(_rbln_external_tree_dir)
    set(_rbln_resolver_args --library-dir "${_rbln_external_tree_dir}/build")
  endif()
  execute_process(
    COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_LIST_DIR}/../../tools/find_rebel_runtime.py"
      ${_rbln_resolver_args}
    OUTPUT_VARIABLE _rbln_resolver_output
    ERROR_VARIABLE _rbln_resolver_error
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _rbln_resolver_result
  )
  if(_rbln_resolver_result EQUAL 0)
    if(_rbln_resolver_error AND NOT RBLN_FIND_QUIETLY)
      message(STATUS "${_rbln_resolver_error}")
    endif()
    string(REPLACE "\n" ";" _rbln_resolver_lines "${_rbln_resolver_output}")
    foreach(_rbln_resolver_line IN LISTS _rbln_resolver_lines)
      if(_rbln_resolver_line MATCHES "^LIBRARY=(.+)$")
        set(RBLN_LIBRARY "${CMAKE_MATCH_1}")
      elseif(_rbln_resolver_line MATCHES "^LIBRARY_RELDIR=(.*)$")
        set(RBLN_RUNTIME_RELDIR "${CMAKE_MATCH_1}")
      elseif(_rbln_resolver_line MATCHES "^INCLUDE_DIR=(.+)$")
        set(RBLN_INCLUDE_DIR "${CMAKE_MATCH_1}")
      endif()
    endforeach()
  elseif(_rbln_resolver_output MATCHES "^ERROR=")
    string(REGEX REPLACE "^ERROR=" "" _rbln_failure_reason
      "${_rbln_resolver_output}")
  else()
    if(NOT _rbln_resolver_error)
      set(_rbln_resolver_error "${_rbln_resolver_result}")
    endif()
    string(CONCAT _rbln_failure_reason
      "Locating the RBLN Runtime with '${Python3_EXECUTABLE}' failed: "
      "${_rbln_resolver_error}")
  endif()
  unset(_rbln_resolver_args)
  unset(_rbln_resolver_output)
  unset(_rbln_resolver_error)
  unset(_rbln_resolver_result)
  unset(_rbln_resolver_lines)
endif()

if(RBLN_LIBRARY AND _rbln_external_tree_dir)
  set(RBLN_INCLUDE_DIR "${_rbln_external_tree_dir}/rebel/include")
  if(NOT EXISTS "${RBLN_LIBRARY}")
    set(RBLN_LIBRARY "")
  endif()
  if(NOT EXISTS "${RBLN_INCLUDE_DIR}/rebel/runtime/api/rbln_runtime_api.h")
    set(RBLN_INCLUDE_DIR "")
  endif()
  if(NOT RBLN_LIBRARY OR NOT RBLN_INCLUDE_DIR)
    string(CONCAT _rbln_failure_reason
      "REBEL_HOME='${_rbln_external_tree_dir}' is not a built rebel-compiler "
      "tree: it needs build/librbln.so and the headers under rebel/include.")
  endif()
elseif(RBLN_LIBRARY)
  set(_rbln_missing_headers "")
  foreach(_rbln_header IN ITEMS
    rebel/runtime/api/rbln_runtime_api.h
    rebel/runtime/api/rbln_exec_api.h
    rebel/runtime/api/rbln_kineto_api.h
    rebel/runtime/api/rbln_retcode.h
    rebel/runtime/memory_stats.h
    rebel/runtime/distributed/rbln_rccl.h
  )
    if(NOT EXISTS "${RBLN_INCLUDE_DIR}/${_rbln_header}")
      list(APPEND _rbln_missing_headers "${_rbln_header}")
    endif()
  endforeach()
  if(_rbln_missing_headers)
    list(JOIN _rbln_missing_headers ", " _rbln_missing_headers)
    string(CONCAT _rbln_failure_reason
      "Found the RBLN Runtime library '${RBLN_LIBRARY}' but not "
      "${_rbln_missing_headers}")
    if(RBLN_INCLUDE_DIR)
      string(APPEND _rbln_failure_reason " in '${RBLN_INCLUDE_DIR}'")
    endif()
    string(APPEND _rbln_failure_reason
      ". Reinstall the rebel-compiler version that this project pins, or set "
      "both RBLN_USE_EXTERNAL_REBEL_COMPILER and REBEL_HOME to build against "
      "an external rebel-compiler tree.")
    set(RBLN_INCLUDE_DIR "")
  endif()
  unset(_rbln_missing_headers)
endif()
unset(_rbln_use_external_tree)
unset(_rbln_external_tree_dir)

# A `;` in the failure reason makes `find_package_handle_standard_args()` fail
# with a fatal error, even in a `QUIET` find.
string(REPLACE ";" "," _rbln_failure_reason "${_rbln_failure_reason}")
find_package_handle_standard_args(RBLN
  REQUIRED_VARS RBLN_LIBRARY RBLN_INCLUDE_DIR
  REASON_FAILURE_MESSAGE "${_rbln_failure_reason}"
)
unset(_rbln_failure_reason)
if(NOT RBLN_FOUND)
  return()
endif()
if(NOT RBLN_FIND_QUIETLY)
  message(STATUS "RBLN Runtime headers: ${RBLN_INCLUDE_DIR}")
endif()

include(TorchRBLNAddRuntimeTarget)
torch_rbln_add_runtime_target(
  LIBRARY "${RBLN_LIBRARY}"
  INCLUDE_DIR "${RBLN_INCLUDE_DIR}"
)

set(RBLN_ABI_CURRENT "")
set(_rbln_abi_header "${RBLN_INCLUDE_DIR}/rebel/runtime/api/rbln_abi.h")
if(EXISTS "${_rbln_abi_header}")
  # Reinstalling `rebel-compiler` can update this header in place.
  set_property(DIRECTORY APPEND
    PROPERTY CMAKE_CONFIGURE_DEPENDS "${_rbln_abi_header}")

  file(STRINGS "${_rbln_abi_header}" _rbln_abi_lines
    REGEX "^[ \t]*#[ \t]*define[ \t]+RBLN_ABI_CURRENT[ \t]+[0-9]+[ \t]*$")
  if(NOT _rbln_abi_lines)
    message(FATAL_ERROR
      "'${_rbln_abi_header}' does not define RBLN_ABI_CURRENT as an integer, "
      "so the build cannot record the RBLN Runtime ABI version. Reinstall the "
      "rebel-compiler version that this project pins.")
  endif()
  list(GET _rbln_abi_lines 0 _rbln_abi_line)
  string(REGEX REPLACE
    "^[ \t]*#[ \t]*define[ \t]+RBLN_ABI_CURRENT[ \t]+([0-9]+)[ \t]*$" "\\1"
    RBLN_ABI_CURRENT "${_rbln_abi_line}")
  if(RBLN_ABI_CURRENT STREQUAL "0")
    message(FATAL_ERROR
      "'${_rbln_abi_header}' defines RBLN_ABI_CURRENT as 0, which torch-rbln "
      "reads as no recorded ABI version, so the import-time ABI check would be "
      "skipped. Reinstall the rebel-compiler version that this project pins.")
  endif()
  if(NOT RBLN_FIND_QUIETLY)
    message(STATUS "RBLN_ABI_CURRENT: ${RBLN_ABI_CURRENT}")
  endif()
else()
  message(WARNING
    "This rebel-compiler predates the ABI version handshake and has no "
    "'${_rbln_abi_header}'. The build records no ABI version, so torch-rbln "
    "cannot detect an incompatible librbln.so at import time.")
endif()
unset(_rbln_abi_header)
unset(_rbln_abi_lines)
unset(_rbln_abi_line)

# A library in an external tree is outside `site-packages`, so the install
# RPATH points to an `_rbln_runtime` symlink.
if(NOT RBLN_RUNTIME_RELDIR)
  set(RBLN_RUNTIME_RELDIR "_rbln_runtime")
endif()

get_filename_component(_rbln_library_dir "${RBLN_LIBRARY}" DIRECTORY)
list(APPEND CMAKE_BUILD_RPATH ${_rbln_library_dir})
list(APPEND CMAKE_INSTALL_RPATH "$ORIGIN/../../${RBLN_RUNTIME_RELDIR}")

# In a source checkout, this symlink lets the install RPATH find `librbln.so`.
install(CODE "
  set(_link \"${CMAKE_INSTALL_PREFIX}/../${RBLN_RUNTIME_RELDIR}\")
  get_filename_component(_link_parent \"\${_link}\" DIRECTORY)
  if(EXISTS \"\${_link}\" AND NOT IS_SYMLINK \"\${_link}\")
    # In `site-packages`, this directory belongs to `rebel-compiler`.
    message(STATUS
      \"Not replacing \${_link} with a symlink: it is a real directory\")
  else()
    # An existing link can point to a different virtual environment.
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E make_directory \"\${_link_parent}\")
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E create_symlink
        \"${_rbln_library_dir}\" \"\${_link}\")
  endif()
")
unset(_rbln_library_dir)
