# - Try to find Rebel
# Once done, this will define
#   REBEL_FOUND         - True if Rebel headers and librbln.so are found
#   REBEL_INCLUDE_DIRS  - Include directories for Rebel (wheel or external)
#   REBEL_LIBRARIES     - Libraries to link with Rebel (librbln.so)
#   REBEL_ABI_CURRENT   - RBLN_ABI_CURRENT read out of rbln_abi.h, or "" when the
#                         Rebel being built against predates the ABI handshake
#
# Both headers and librbln.so are required at build time so that C code
# linking against the Rebel ABI builds correctly. By default headers and the
# .so both come from the installed rebel-compiler wheel in site-packages
# (rebel/include and tvm/, shipped from the same build). REBEL_HOME switches to
# an external Rebel tree (rebel/include and build/).

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

set(PACKAGE_NAME REBEL)
include(FindPackageHandleStandardArgs)

# Capture env vars once to avoid fragile if() parsing with empty values
set(_REBEL_USE_EXTERNAL "$ENV{RBLN_USE_EXTERNAL_REBEL_COMPILER}")
set(_REBEL_HOME "$ENV{REBEL_HOME}")

set(rebel_include_dir "")
set(rebel_library_path "")

# 1) External: both RBLN_USE_EXTERNAL_REBEL_COMPILER and REBEL_HOME must be set together
if(_REBEL_USE_EXTERNAL OR _REBEL_HOME)
  if(NOT _REBEL_USE_EXTERNAL OR NOT _REBEL_HOME)
    message(FATAL_ERROR
      "FindRebel: RBLN_USE_EXTERNAL_REBEL_COMPILER and REBEL_HOME must be set together. "
      "Either set both (for an external rebel tree) or neither (for the installed wheel). "
      "Current: RBLN_USE_EXTERNAL_REBEL_COMPILER=${_REBEL_USE_EXTERNAL}, REBEL_HOME=${_REBEL_HOME}")
  endif()
  set(rebel_include_dir "${_REBEL_HOME}/rebel/include")
  set(rebel_library_path "${_REBEL_HOME}/build")
  message(STATUS "FindRebel: EXTERNAL (REBEL_HOME) -- include: ${rebel_include_dir}, library: ${rebel_library_path}")
# 2) Installed wheel: both headers (rebel/include) and librbln.so (tvm/) ship
#    inside the rebel-compiler wheel in site-packages, from the same build.
else()
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  set(rebel_include_dir "${PYTHON_SITE_PACKAGES}/rebel/include")
  set(rebel_library_path "${PYTHON_SITE_PACKAGES}/tvm")
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

# ABI snapshot #################################################################
# Record RBLN_ABI_CURRENT of the Rebel we build against; torch_rbln compares that
# frozen copy against the librbln.so it actually loads (torch_rbln/_internal/abi_check.py).
# MIN_SUPPORTED is not read here: it describes what a future runtime still accepts,
# so it can only be queried from the .so at import time. A missing header means the
# Rebel predates the handshake (rebel_compiler #12426) -- not a build error, just no
# snapshot to check against.
set(REBEL_ABI_CURRENT "")
set(_rebel_abi_header "${REBEL_INCLUDE_DIRS}/rebel/runtime/api/rbln_abi.h")
if(EXISTS "${_rebel_abi_header}")
  # Swapping a wheel in place keeps the same header path, so make its contents a
  # configure dependency or the snapshot silently goes stale.
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_rebel_abi_header}")

  file(STRINGS "${_rebel_abi_header}" _rebel_abi_lines
    REGEX "^[ \t]*#[ \t]*define[ \t]+RBLN_ABI_CURRENT[ \t]+[0-9]+[ \t]*$")
  if(NOT _rebel_abi_lines)
    message(FATAL_ERROR
      "FindRebel: ${_rebel_abi_header} exists but defines no integer RBLN_ABI_CURRENT. "
      "Refusing to build a wheel whose recorded ABI would be wrong.")
  endif()
  list(GET _rebel_abi_lines 0 _rebel_abi_line)
  string(REGEX REPLACE "^[ \t]*#[ \t]*define[ \t]+RBLN_ABI_CURRENT[ \t]+([0-9]+)[ \t]*$" "\\1"
    REBEL_ABI_CURRENT "${_rebel_abi_line}")
  if(REBEL_ABI_CURRENT STREQUAL "0")
    message(FATAL_ERROR
      "FindRebel: ${_rebel_abi_header} declares RBLN_ABI_CURRENT 0. Zero is not a valid "
      "contract number (torch-rbln reads it as 'no snapshot recorded').")
  endif()
  message(STATUS "FindRebel: rebel ABI snapshot -- RBLN_ABI_CURRENT=${REBEL_ABI_CURRENT}")
else()
  message(WARNING
    "FindRebel: no ${_rebel_abi_header} -- this Rebel predates the ABI version handshake. "
    "The resulting torch-rbln records no ABI snapshot, so it cannot detect at import time "
    "that it was loaded against an incompatible librbln.so.")
endif()
unset(_rebel_abi_header)
unset(_rebel_abi_lines)
unset(_rebel_abi_line)

mark_as_advanced(
  ${PACKAGE_NAME}_INCLUDE_DIR
  ${PACKAGE_NAME}_LIBRARY
)

list(APPEND CMAKE_BUILD_RPATH ${rebel_library_path})
list(APPEND CMAKE_INSTALL_RPATH "$ORIGIN/../../tvm")

install(CODE "execute_process(
  COMMAND ${CMAKE_COMMAND} -E create_symlink ${rebel_library_path} tvm
  WORKING_DIRECTORY ${CMAKE_INSTALL_PREFIX}/..
)")

unset(PACKAGE_NAME)
unset(_REBEL_USE_EXTERNAL)
unset(_REBEL_HOME)
unset(rebel_include_dir)
unset(rebel_library_path)
