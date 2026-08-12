# - Try to find Rebel
# Once done, this will define
#   REBEL_FOUND            - True if Rebel headers and librbln.so are found
#   REBEL_INCLUDE_DIRS     - Include directories for Rebel (wheel or external)
#   REBEL_LIBRARIES        - Libraries to link with Rebel (librbln.so)
#   REBEL_RUNTIME_RELDIR   - Where the runtime library sits relative to site-packages, used to
#                            build install RPATHs (see below)
#
# Headers and librbln.so are both required at build time. Where they live is not hard-coded:
# tools/find_rebel_runtime.py runs the same resolver torch-rbln uses at import time, so a
# rebel-compiler that relocates the library needs no change here. REBEL_HOME switches to an
# external Rebel tree (rebel/include and build/).

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

set(PACKAGE_NAME REBEL)
include(FindPackageHandleStandardArgs)

# Capture env vars once to avoid fragile if() parsing with empty values
set(_REBEL_USE_EXTERNAL "$ENV{RBLN_USE_EXTERNAL_REBEL_COMPILER}")
set(_REBEL_HOME "$ENV{REBEL_HOME}")

set(rebel_include_dir "")
set(rebel_library_path "")
set(rebel_library_reldir "")

# Use the interpreter the build was given: a freshly found one may belong to a different
# environment and would then report that environment's rebel-compiler.
if(DEFINED Python_EXECUTABLE)
  set(_rebel_python "${Python_EXECUTABLE}")
elseif(DEFINED Python3_EXECUTABLE)
  set(_rebel_python "${Python3_EXECUTABLE}")
else()
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  set(_rebel_python "${Python3_EXECUTABLE}")
endif()

# Ask the resolver where the library is; --library-dir pins it to an external tree.
set(_rebel_finder "${CMAKE_CURRENT_LIST_DIR}/../tools/find_rebel_runtime.py")
set(_rebel_finder_args "")

# 1) External: both RBLN_USE_EXTERNAL_REBEL_COMPILER and REBEL_HOME must be set together
if(_REBEL_USE_EXTERNAL OR _REBEL_HOME)
  if(NOT _REBEL_USE_EXTERNAL OR NOT _REBEL_HOME)
    message(FATAL_ERROR
      "FindRebel: RBLN_USE_EXTERNAL_REBEL_COMPILER and REBEL_HOME must be set together. "
      "Either set both (for an external rebel tree) or neither (for the installed wheel). "
      "Current: RBLN_USE_EXTERNAL_REBEL_COMPILER=${_REBEL_USE_EXTERNAL}, REBEL_HOME=${_REBEL_HOME}")
  endif()
  set(_rebel_finder_args --library-dir "${_REBEL_HOME}/build")
endif()

execute_process(
  COMMAND ${_rebel_python} "${_rebel_finder}" ${_rebel_finder_args}
  OUTPUT_VARIABLE _rebel_finder_output
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _rebel_finder_result
)
if(NOT _rebel_finder_result EQUAL 0)
  message(FATAL_ERROR
    "FindRebel: could not locate the rebel runtime library. ${_rebel_finder_output}")
endif()
string(REPLACE "\n" ";" _rebel_finder_lines "${_rebel_finder_output}")
foreach(_line IN LISTS _rebel_finder_lines)
  if(_line MATCHES "^LIBRARY_DIR=(.+)$")
    set(rebel_library_path "${CMAKE_MATCH_1}")
  elseif(_line MATCHES "^LIBRARY_RELDIR=(.*)$")
    set(rebel_library_reldir "${CMAKE_MATCH_1}")
  elseif(_line MATCHES "^INCLUDE_DIR=(.+)$")
    set(rebel_include_dir "${CMAKE_MATCH_1}")
  endif()
endforeach()


if(_REBEL_HOME)
  # Headers come from the external tree, matching the library built there.
  set(rebel_include_dir "${_REBEL_HOME}/rebel/include")
  message(STATUS "FindRebel: EXTERNAL (REBEL_HOME) -- include: ${rebel_include_dir}, library: ${rebel_library_path}")
else()
  # Installed wheel: headers (rebel/include) and the library ship in the same rebel-compiler
  # wheel, from the same build.
  # Check the full runtime header set the extension pulls in (directly or
  # transitively), so a partially-packaged wheel is diagnosed here rather than
  # as a compile error later.
  if(NOT EXISTS "${rebel_include_dir}/rebel/runtime/api/rbln_runtime_api.h"
     OR NOT EXISTS "${rebel_include_dir}/rebel/runtime/api/rbln_kineto_api.h"
     OR NOT EXISTS "${rebel_include_dir}/rebel/runtime/api/rbln_retcode.h"
     OR NOT EXISTS "${rebel_include_dir}/rebel/runtime/memory_stats.h"
     OR NOT EXISTS "${rebel_include_dir}/rebel/runtime/distributed/rbln_rccl.h")
    message(FATAL_ERROR
      "FindRebel: Rebel runtime headers not found under ${rebel_include_dir}. "
      "Install a rebel-compiler wheel that ships headers under rebel/include "
      "(>=0.11.1.dev322), or build against an external Rebel tree by setting "
      "both RBLN_USE_EXTERNAL_REBEL_COMPILER and REBEL_HOME.")
  endif()
  message(STATUS "FindRebel: WHEEL (site-packages) -- include: ${rebel_include_dir}, library: ${rebel_library_path}")
endif()

# find_path/find_library skip the search when their cache entry is already set,
# so drop paths cached from an earlier vendored, wheel, or external config.
unset(${PACKAGE_NAME}_INCLUDE_DIR CACHE)
unset(${PACKAGE_NAME}_LIBRARY CACHE)

find_path(${PACKAGE_NAME}_INCLUDE_DIR
  NAMES rebel/runtime/api/rbln_runtime_api.h
  PATHS ${rebel_include_dir}
  NO_DEFAULT_PATH
)

# Library is required at build time for ABI/linking of C code
find_library(${PACKAGE_NAME}_LIBRARY
  NAMES rbln
  PATHS ${rebel_library_path}
  NO_DEFAULT_PATH
)

find_package_handle_standard_args(REBEL
  REQUIRED_VARS ${PACKAGE_NAME}_INCLUDE_DIR ${PACKAGE_NAME}_LIBRARY
  VERSION_VAR ${PACKAGE_NAME}_VERSION
)

if(NOT REBEL_FOUND)
  message(FATAL_ERROR "FindRebel: Rebel was not found. RBLN requires Rebel to build.")
endif()

if(REBEL_FOUND)
  set(REBEL_INCLUDE_DIRS ${${PACKAGE_NAME}_INCLUDE_DIR})
  set(REBEL_LIBRARIES ${${PACKAGE_NAME}_LIBRARY})
endif()

mark_as_advanced(
  ${PACKAGE_NAME}_INCLUDE_DIR
  ${PACKAGE_NAME}_LIBRARY
)

# Install RPATHs are relative to the directory the install prefix sits in, i.e. site-packages
# for an installed wheel. A library inside site-packages keeps that relationship at runtime, so
# the RPATH follows it when rebel-compiler moves the library; outside it (external tree, source
# checkout) a neutrally named symlink next to the install prefix stands in.
if(rebel_library_reldir)
  set(REBEL_RUNTIME_RELDIR "${rebel_library_reldir}")
else()
  set(REBEL_RUNTIME_RELDIR "_rbln_runtime")
endif()

list(APPEND CMAKE_BUILD_RPATH ${rebel_library_path})
list(APPEND CMAKE_INSTALL_RPATH "$ORIGIN/../../${REBEL_RUNTIME_RELDIR}")

# A source checkout resolves $ORIGIN/../../<reldir> inside the checkout, so link it there.
install(CODE "
  set(_link \"${CMAKE_INSTALL_PREFIX}/../${REBEL_RUNTIME_RELDIR}\")
  get_filename_component(_link_parent \"\${_link}\" DIRECTORY)
  if(EXISTS \"\${_link}\" AND NOT IS_SYMLINK \"\${_link}\")
    # Installing straight into site-packages: rebel-compiler's own directory, which the RPATH
    # already resolves to.
    message(STATUS \"FindRebel: \${_link} is a real directory, leaving it in place\")
  else()
    # Recreated every install, not only when missing: a link left by an earlier one can still
    # name a virtualenv this build has nothing to do with.
    execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory \"\${_link_parent}\")
    execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink \"${rebel_library_path}\" \"\${_link}\")
  endif()
")

unset(PACKAGE_NAME)
unset(_REBEL_USE_EXTERNAL)
unset(_REBEL_HOME)
unset(rebel_include_dir)
unset(rebel_library_path)
