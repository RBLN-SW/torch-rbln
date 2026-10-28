cmake_minimum_required(VERSION 3.31...4.4)

# Options ######################################################################
option(RBLN_USE_CCACHE "Use ccache if it is installed" ON)
option(BUILD_TESTING "Build C++ test binaries" ON)

# CMake configurations #########################################################
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "In-source builds are not supported")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH OFF)

set(CMAKE_BUILD_RPATH "$ORIGIN")
set(CMAKE_INSTALL_RPATH "$ORIGIN")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_MODULE_PATH
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules
)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(RBLN_USE_CCACHE)
  find_program(CCACHE ccache)
  if(CCACHE)
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE})
    set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE})
    set(CMAKE_CUDA_COMPILER_LAUNCHER ${CCACHE})
  endif()
endif()

project(torch-rbln
  LANGUAGES CXX C
)

include(TorchRBLNCheckCompiler)

# dependencies #################################################################
include(External/torch)

find_package(RBLN MODULE REQUIRED)

include(TorchRBLNBuildFlags)
include(TorchRBLNAlwaysLink)

if(RBLN_ABI_CURRENT STREQUAL "")
  set(RBLN_ABI_CURRENT_PY "None")
else()
  set(RBLN_ABI_CURRENT_PY "${RBLN_ABI_CURRENT}")
endif()
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/abi_snapshot.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/generated/_abi_snapshot.py
  @ONLY
)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/generated/_abi_snapshot.py
  DESTINATION _internal
)

set(aten_native_dir ${CMAKE_CURRENT_SOURCE_DIR}/aten/src/ATen/native)
set(register_ops_file ${CMAKE_CURRENT_BINARY_DIR}/generated/register_ops.py)
file(GLOB_RECURSE codegen_sources CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/tools/codegen/*.py
)
# Codegen depends on a copy of torch's `version.py` because an installed file
# can keep an older mtime.
configure_file(
  ${TORCH_INSTALL_PREFIX}/version.py
  ${CMAKE_CURRENT_BINARY_DIR}/torch_version.py
  COPYONLY
)
add_custom_command(
  OUTPUT ${register_ops_file}
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/run_codegen.py
    ${aten_native_dir}/native_functions.yaml
    ${aten_native_dir}/tags.yaml
    ${register_ops_file}
  DEPENDS
    ${aten_native_dir}/native_functions.yaml
    ${aten_native_dir}/tags.yaml
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/run_codegen.py
    ${codegen_sources}
    ${CMAKE_CURRENT_BINARY_DIR}/torch_version.py
  COMMENT "Generating register_ops.py"
  VERBATIM
)
add_custom_target(torch_rbln_register_ops ALL
  DEPENDS ${register_ops_file}
)
install(FILES ${register_ops_file}
  DESTINATION _internal
)

include(External/spdlog)

include(External/pybind11)

# `torch-rbln` target ##########################################################
set(target torch_rbln_interface)
add_library(${target} INTERFACE)
target_compile_features(${target} INTERFACE cxx_std_20)
target_compile_definitions(${target} INTERFACE
  FMT_HEADER_ONLY
)
target_include_directories(${target} INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(${target} INTERFACE
  torch_library
  TorchRBLN::rbln
)
install(TARGETS ${target}
  EXPORT TorchRBLNTargets
)

# subdirectories ###############################################################
add_subdirectory(aten/src/ATen)
add_subdirectory(c10)
add_subdirectory(torch_rbln/csrc/rbln)
add_subdirectory(torch_rbln/csrc/distributed/c10d/rbln)
if(BUILD_TESTING)
  enable_testing()
  include(External/googletest)
  add_subdirectory(test/cpp)
else()
  message(STATUS "Skipping C++ tests (BUILD_TESTING=OFF)")
endif()

# Package for C++ consumers ####################################################
include(CMakePackageConfigHelpers)
set(config_install_dir share/cmake/TorchRBLN)
install(EXPORT TorchRBLNTargets
  NAMESPACE TorchRBLN::
  DESTINATION ${config_install_dir}
)
# Both directories come from resolved file paths because some installers
# symlink each file without symlinking its directory.
get_filename_component(rbln_library_dir "${RBLN_LIBRARY}" REALPATH)
get_filename_component(rbln_library_dir "${rbln_library_dir}" DIRECTORY)
get_filename_component(rbln_include_dir
  "${RBLN_INCLUDE_DIR}/rebel/runtime/api/rbln_runtime_api.h" REALPATH)
get_filename_component(rbln_include_dir
  "${rbln_include_dir}/../../../.." ABSOLUTE)
file(RELATIVE_PATH rbln_include_offset
  "${rbln_library_dir}" "${rbln_include_dir}")
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/TorchRBLNConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/TorchRBLNConfig.cmake
  INSTALL_DESTINATION ${config_install_dir}
)
set(config_files
  ${CMAKE_CURRENT_BINARY_DIR}/TorchRBLNConfig.cmake
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/TorchRBLNAddRuntimeTarget.cmake
)
if(DEFINED TORCH_RBLN_VERSION)
  write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/TorchRBLNConfigVersion.cmake
    VERSION ${TORCH_RBLN_VERSION}
    COMPATIBILITY AnyNewerVersion
  )
  list(APPEND config_files
    ${CMAKE_CURRENT_BINARY_DIR}/TorchRBLNConfigVersion.cmake)
endif()
install(FILES ${config_files}
  DESTINATION ${config_install_dir}
)
