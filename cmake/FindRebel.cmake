# - Try to find the rbln runtime
# Once done, this will define
#   REBEL_FOUND            - True if the runtime headers and libraries are found
#   REBEL_INCLUDE_DIRS     - The directory holding rbln/runtime/*.h
#   REBEL_LIBRARIES        - librbln_rt.so and librbln_artifact.so
#   REBEL_RUNTIME_RELDIR   - Where the runtime library sits relative to the directory the install
#                            prefix is in, used to build install RPATHs (see below)
#   REBEL_ABI_SCRIPT       - rbln/cmake/RblnAbi.cmake, which computes the ABI id of the headers
#
# REBEL_HOME names a rebel-compiler tree built with rbln: headers in rbln/include and libraries
# in build/rbln. Without it, the rbln package a rebel-compiler wheel installed for Python3 serves,
# through its CMake package.

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

include(FindPackageHandleStandardArgs)

set(_REBEL_HOME "$ENV{REBEL_HOME}")
if(NOT _REBEL_HOME)
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -m rbln.cmake
    OUTPUT_VARIABLE _rbln_config_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_VARIABLE _rbln_config_error
    RESULT_VARIABLE _rbln_config_missing)
  if(_rbln_config_missing)
    message(FATAL_ERROR
      "FindRebel: set REBEL_HOME to a rebel-compiler tree built with rbln (headers in "
      "rbln/include, libraries in build/rbln), or install the rebel-compiler wheel: "
      "${_rbln_config_error}")
  endif()
  find_package(rbln CONFIG REQUIRED PATHS "${_rbln_config_dir}" NO_DEFAULT_PATH)
  set(REBEL_FOUND TRUE)
  set(REBEL_INCLUDE_DIRS ${rbln_INCLUDE_DIRS})
  set(REBEL_LIBRARIES rbln::rt rbln::artifact)
  set(REBEL_ABI_SCRIPT ${rbln_ABI_SCRIPT})
  message(STATUS "FindRebel: the installed rbln in ${rbln_LIBRARY_DIR}")
  # torch_rbln/lib and rbln/lib sit side by side in site-packages.
  set(REBEL_RUNTIME_RELDIR "rbln/lib")
  list(APPEND CMAKE_BUILD_RPATH ${rbln_LIBRARY_DIR})
  list(APPEND CMAKE_INSTALL_RPATH "$ORIGIN/../../${REBEL_RUNTIME_RELDIR}")
  unset(_rbln_config_dir)
  unset(_rbln_config_error)
  unset(_rbln_config_missing)
  unset(_REBEL_HOME)
  return()
endif()
set(rebel_library_path "${_REBEL_HOME}/build/rbln")

# find_path/find_library skip the search when their cache entry is already set, so drop paths
# cached from an earlier configure against another tree.
unset(REBEL_INCLUDE_DIR CACHE)
unset(REBEL_RUNTIME_LIBRARY CACHE)
unset(REBEL_ARTIFACT_LIBRARY CACHE)
unset(REBEL_ABI_SCRIPT CACHE)

find_path(REBEL_INCLUDE_DIR
  NAMES rbln/runtime/device.h
  PATHS "${_REBEL_HOME}/rbln/include"
  NO_DEFAULT_PATH
)
find_library(REBEL_RUNTIME_LIBRARY
  NAMES rbln_rt
  PATHS ${rebel_library_path}
  NO_DEFAULT_PATH
)
find_library(REBEL_ARTIFACT_LIBRARY
  NAMES rbln_artifact
  PATHS ${rebel_library_path}
  NO_DEFAULT_PATH
)
find_file(REBEL_ABI_SCRIPT
  NAMES RblnAbi.cmake
  PATHS "${_REBEL_HOME}/rbln/cmake"
  NO_DEFAULT_PATH
)

find_package_handle_standard_args(REBEL
  REQUIRED_VARS REBEL_INCLUDE_DIR REBEL_RUNTIME_LIBRARY REBEL_ARTIFACT_LIBRARY REBEL_ABI_SCRIPT
)
if(NOT REBEL_FOUND)
  message(FATAL_ERROR "FindRebel: the rbln runtime was not found under ${_REBEL_HOME}.")
endif()

set(REBEL_INCLUDE_DIRS ${REBEL_INCLUDE_DIR})
set(REBEL_LIBRARIES ${REBEL_RUNTIME_LIBRARY} ${REBEL_ARTIFACT_LIBRARY})
message(STATUS "FindRebel: include ${REBEL_INCLUDE_DIRS}, libraries ${REBEL_LIBRARIES}")

mark_as_advanced(
  REBEL_INCLUDE_DIR
  REBEL_RUNTIME_LIBRARY
  REBEL_ARTIFACT_LIBRARY
  REBEL_ABI_SCRIPT
)

# Install RPATHs are relative to the directory the install prefix sits in. The runtime lives
# outside it, so a neutrally named symlink next to the install prefix stands in.
set(REBEL_RUNTIME_RELDIR "_rbln_runtime")

list(APPEND CMAKE_BUILD_RPATH ${rebel_library_path})
list(APPEND CMAKE_INSTALL_RPATH "$ORIGIN/../../${REBEL_RUNTIME_RELDIR}")

install(CODE "
  set(_link \"${CMAKE_INSTALL_PREFIX}/../${REBEL_RUNTIME_RELDIR}\")
  get_filename_component(_link_parent \"\${_link}\" DIRECTORY)
  # Recreated every install: a link left by an earlier one can name another tree.
  execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory \"\${_link_parent}\")
  execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink \"${rebel_library_path}\" \"\${_link}\")
")

unset(_REBEL_HOME)
unset(rebel_library_path)
