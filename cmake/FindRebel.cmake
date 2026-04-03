# - Try to find Rebel
# Once done, this will define
#   REBEL_FOUND         - True if Rebel headers and librbln.so are found
#   REBEL_INCLUDE_DIRS  - Include directories for Rebel (vendored or external)
#   REBEL_LIBRARIES     - Libraries to link with Rebel (librbln.so)
#
# Both headers and librbln.so are required at build time so that C code
# linking against the Rebel ABI builds correctly. Headers come from
# third_party/rebel_compiler/include (vendored) or REBEL_HOME (external).
# The .so comes from the same locations (e.g. REBEL_HOME/build or tvm site-packages).

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

set(PACKAGE_NAME REBEL)
include(FindPackageHandleStandardArgs)

# Capture env vars once to avoid fragile if() parsing with empty values
set(_REBEL_USE_EXTERNAL "$ENV{RBLN_USE_EXTERNAL_REBEL_COMPILER}")
set(_REBEL_HOME "$ENV{REBEL_HOME}")

# Vendored: minimal Rebel runtime headers in third_party (same layout: rebel/runtime/api/rbln_runtime_api.h)
set(REBEL_INCLUDE_DIR_VENDORED ${CMAKE_SOURCE_DIR}/third_party/rebel_compiler/include)

set(rebel_include_dir "")
set(rebel_library_path "")

# 1) External: both RBLN_USE_EXTERNAL_REBEL_COMPILER and REBEL_HOME must be set together
if(_REBEL_USE_EXTERNAL OR _REBEL_HOME)
  if(NOT _REBEL_USE_EXTERNAL OR NOT _REBEL_HOME)
    message(FATAL_ERROR
      "FindRebel: RBLN_USE_EXTERNAL_REBEL_COMPILER and REBEL_HOME must be set together. "
      "Either set both (for external rebel) or neither (for vendored). "
      "Current: RBLN_USE_EXTERNAL_REBEL_COMPILER=${_REBEL_USE_EXTERNAL}, REBEL_HOME=${_REBEL_HOME}")
  endif()
  set(rebel_include_dir "${_REBEL_HOME}/rebel/include")
  set(rebel_library_path "${_REBEL_HOME}/build")
  message(STATUS "FindRebel: EXTERNAL (REBEL_HOME) -- include: ${rebel_include_dir}, library: ${rebel_library_path}")
# 2) Vendored: use third_party/rebel_compiler/include (external dependency vendored in-tree)
else()
  if(NOT EXISTS "${REBEL_INCLUDE_DIR_VENDORED}/rebel/runtime/api/rbln_runtime_api.h")
    message(FATAL_ERROR
      "FindRebel: Vendored Rebel headers not found at ${REBEL_INCLUDE_DIR_VENDORED}. "
      "Ensure third_party/rebel_compiler/include/rebel/runtime/ is present.")
  endif()
  set(rebel_include_dir "${REBEL_INCLUDE_DIR_VENDORED}")
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  set(rebel_library_path "${PYTHON_SITE_PACKAGES}/tvm")
  message(STATUS "FindRebel: VENDORED (c10/rbln/rebel/include) -- include: ${rebel_include_dir}, library: ${rebel_library_path}")
endif()

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
