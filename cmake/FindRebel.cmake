# - Try to find the rebel.v2 runtime
# Once done, this will define
#   REBEL_FOUND            - True if the runtime headers and libraries are found
#   REBEL_INCLUDE_DIRS     - The directory holding rebel/v2/runtime/*.h
#   REBEL_LIBRARIES        - librebel_v2_rt.so and librebel_v2_artifact.so
#   REBEL_RUNTIME_RELDIR   - Where the runtime library sits relative to the directory the install
#                            prefix is in, used to build install RPATHs (see below)
#   REBEL_ABI_SCRIPT       - rebel/v2/cmake/RebelV2Abi.cmake, which computes the ABI id of the headers
#
# REBEL_HOME names a rebel-compiler tree built with rebel.v2: headers in rebel/v2/include and
# libraries in build/rebel/v2. Without it, the rebel.v2 package a rebel-compiler wheel installed for
# Python3 serves, through its CMake package.

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

include(FindPackageHandleStandardArgs)

set(_REBEL_HOME "$ENV{REBEL_HOME}")
if(NOT _REBEL_HOME)
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -m rebel.v2.cmake
    OUTPUT_VARIABLE _rebel_v2_config_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_VARIABLE _rebel_v2_config_error
    RESULT_VARIABLE _rebel_v2_config_missing)
  if(_rebel_v2_config_missing)
    message(FATAL_ERROR
      "FindRebel: set REBEL_HOME to a rebel-compiler tree built with rebel.v2 (headers in "
      "rebel/v2/include, libraries in build/rebel/v2), or install the rebel-compiler wheel: "
      "${_rebel_v2_config_error}")
  endif()
  find_package(rebel_v2 CONFIG REQUIRED PATHS "${_rebel_v2_config_dir}" NO_DEFAULT_PATH)
  set(REBEL_FOUND TRUE)
  set(REBEL_INCLUDE_DIRS ${rebel_v2_INCLUDE_DIRS})
  set(REBEL_LIBRARIES rebel_v2::rt rebel_v2::artifact)
  set(REBEL_ABI_SCRIPT ${rebel_v2_ABI_SCRIPT})
  message(STATUS "FindRebel: the installed rebel.v2 in ${rebel_v2_LIBRARY_DIR}")
  # torch_rbln/lib and rebel/v2/lib both sit in site-packages.
  set(REBEL_RUNTIME_RELDIR "rebel/v2/lib")
  list(APPEND CMAKE_BUILD_RPATH ${rebel_v2_LIBRARY_DIR})
  list(APPEND CMAKE_INSTALL_RPATH "$ORIGIN/../../${REBEL_RUNTIME_RELDIR}")
  unset(_rebel_v2_config_dir)
  unset(_rebel_v2_config_error)
  unset(_rebel_v2_config_missing)
  unset(_REBEL_HOME)
  return()
endif()
set(rebel_library_path "${_REBEL_HOME}/build/rebel/v2")

# find_path/find_library skip the search when their cache entry is already set, so drop paths
# cached from an earlier configure against another tree.
unset(REBEL_INCLUDE_DIR CACHE)
unset(REBEL_RUNTIME_LIBRARY CACHE)
unset(REBEL_ARTIFACT_LIBRARY CACHE)
unset(REBEL_ABI_SCRIPT CACHE)

find_path(REBEL_INCLUDE_DIR
  NAMES rebel/v2/runtime/device.h
  PATHS "${_REBEL_HOME}/rebel/v2/include"
  NO_DEFAULT_PATH
)
find_library(REBEL_RUNTIME_LIBRARY
  NAMES rebel_v2_rt
  PATHS ${rebel_library_path}
  NO_DEFAULT_PATH
)
find_library(REBEL_ARTIFACT_LIBRARY
  NAMES rebel_v2_artifact
  PATHS ${rebel_library_path}
  NO_DEFAULT_PATH
)
find_file(REBEL_ABI_SCRIPT
  NAMES RebelV2Abi.cmake
  PATHS "${_REBEL_HOME}/rebel/v2/cmake"
  NO_DEFAULT_PATH
)

find_package_handle_standard_args(REBEL
  REQUIRED_VARS REBEL_INCLUDE_DIR REBEL_RUNTIME_LIBRARY REBEL_ARTIFACT_LIBRARY REBEL_ABI_SCRIPT
)
if(NOT REBEL_FOUND)
  message(FATAL_ERROR "FindRebel: the rebel.v2 runtime was not found under ${_REBEL_HOME}.")
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
set(REBEL_RUNTIME_RELDIR "_rebel_v2_runtime")

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
