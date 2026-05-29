cmake_minimum_required(VERSION 3.18
  FATAL_ERROR
)

# Options ######################################################################
option(RBLN_USE_CCACHE "Use ccache if it is installed" ON)

# CMake configurations #########################################################
# Prohibit in-source builds
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "In-source build are not supported")
endif()

# Set C++ and C standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Generate compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set not to use link path for RPATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH OFF)

# Set RPATH for the build and install
set(CMAKE_BUILD_RPATH "$ORIGIN")
set(CMAKE_INSTALL_RPATH "$ORIGIN")

# Set position independent code for binaries
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Set CMake module path
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Set output directory for artifacts of the build result
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Enable CTest
enable_testing()

# If ccache installed, use it as a compiler launcher.
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

# dependencies #################################################################
# Set `TORCH_FOUND`, `TORCH_INCLUDE_DIRS`, `TORCH_LIBRARIES`, `TORCH_CXX_FLAGS`
include(FindTorch)

# Set `REBEL_FOUND`, `REBEL_INCLUDE_DIRS`, and `REBEL_LIBRARIES`
include(FindRebel)

# Add GoogleTest features
include(FindGTest)

# Add spdlog for logging
include(Findspdlog)

# Set `pybind11::module`, `pybind11::embed`, `pybind11::headers`, and
# `pybind11::pybind11`
include(FindPybind)

# torch-rbln target ############################################################
# Define target interface for all libaraies and executables
set(target torch_rbln_interface)
add_library(${target} INTERFACE)
target_compile_definitions(${target} INTERFACE
  FMT_HEADER_ONLY
)
target_compile_options(${target} INTERFACE
  ${TORCH_CXX_FLAGS}
  -ffile-prefix-map=${CMAKE_CURRENT_SOURCE_DIR}/=
  -ffile-prefix-map=${CMAKE_CURRENT_BINARY_DIR}/=
  -ffile-prefix-map=${CMAKE_INSTALL_PREFIX}/=
  -ffile-prefix-map=${TORCH_INSTALL_PREFIX}/=torch/
)
# Hardening flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(${target} INTERFACE
    # _FORTIFY_SOURCE is a no-op without -O; gate on non-Debug to silence the warning.
    $<$<NOT:$<CONFIG:Debug>>:-D_FORTIFY_SOURCE=2>
    -fstack-protector-strong
    -fstack-clash-protection
  )
  target_link_options(${target} INTERFACE
    -Wl,-z,relro
    -Wl,-z,now
    -Wl,-z,noexecstack
    # Scope to non-executables: tests need libtorch_rbln in NEEDED for static-init op registration.
    $<$<NOT:$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>>:-Wl,--as-needed>
  )
endif()
target_include_directories(${target} SYSTEM INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(${target} INTERFACE)

# subdirectories ###############################################################
add_subdirectory(aten/src/ATen)
add_subdirectory(c10)
add_subdirectory(torch_rbln/csrc/rbln)
add_subdirectory(torch_rbln/csrc/distributed/c10d/rbln)
add_subdirectory(test/cpp)
