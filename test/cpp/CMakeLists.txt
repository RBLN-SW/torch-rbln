cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

if(NOT TARGET torch_rbln)
  project(torch_rbln_cpp_tests LANGUAGES CXX)

  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)

  set(BUILD_SHARED_LIBS ON)

  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

  enable_testing()

  get_filename_component(TORCH_RBLN_SOURCE_DIR
    "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
  set(CMAKE_MODULE_PATH ${TORCH_RBLN_SOURCE_DIR}/cmake)

  if(NOT DEFINED TORCH_RBLN_ROOT)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    string(CONCAT torch_rbln_probe_code
      "import importlib.util; print(importlib.util.find_spec('torch_rbln')"
      ".submodule_search_locations[0])")
    execute_process(
      COMMAND ${Python3_EXECUTABLE} -I -c "${torch_rbln_probe_code}"
      OUTPUT_VARIABLE TORCH_RBLN_ROOT
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE torch_rbln_probe
    )
    if(NOT torch_rbln_probe EQUAL 0)
      message(FATAL_ERROR
        "Could not find an installed torch_rbln; "
        "set -DTORCH_RBLN_ROOT=<site-packages>/torch_rbln explicitly.")
    endif()
  endif()
  find_library(TORCH_RBLN_LIBRARY
    NAMES torch_rbln
    PATHS "${TORCH_RBLN_ROOT}/lib"
    NO_DEFAULT_PATH
    REQUIRED
  )
  find_library(C10_RBLN_LIBRARY
    NAMES c10_rbln
    PATHS "${TORCH_RBLN_ROOT}/lib"
    NO_DEFAULT_PATH
    REQUIRED
  )
  find_library(C10D_RBLN_LIBRARY
    NAMES c10d_rbln
    PATHS "${TORCH_RBLN_ROOT}/lib"
    NO_DEFAULT_PATH
    REQUIRED
  )

  include(TorchSetup)
  include(FindREBEL)
  include(GTestSetup)

  add_library(torch_rbln SHARED IMPORTED)
  set_target_properties(torch_rbln PROPERTIES
    IMPORTED_LOCATION "${TORCH_RBLN_LIBRARY}"
  )
  target_include_directories(torch_rbln INTERFACE
    ${TORCH_RBLN_ROOT}/include
    ${TORCH_INCLUDE_DIRS}
    ${REBEL_INCLUDE_DIRS}
  )
  target_link_libraries(torch_rbln INTERFACE
    ${C10_RBLN_LIBRARY}
    ${C10D_RBLN_LIBRARY}
    ${TORCH_LIBRARIES}
    ${REBEL_LIBRARIES}
  )
  target_compile_definitions(torch_rbln INTERFACE
    FMT_HEADER_ONLY
  )
  target_compile_options(torch_rbln INTERFACE
    ${TORCH_CXX_FLAGS}
  )
endif()

set(sources
  c10d/ProcessGroupRBLNTest.cpp
  core/RBLNAllocatorTest.cpp
  core/RBLNCopyOpTest.cpp
  core/RBLNDeviceGuardTest.cpp
  core/RBLNNoDeviceTest.cpp
  core/RBLNDummyDeviceTest.cpp
  core/RBLNEventTest.cpp
  core/RBLNStreamTest.cpp
  core/RBLNFunctionsTest.cpp
  core/RBLNHostBatchTest.cpp
  core/RBLNLoggingTest.cpp
  core/RBLNPinnedAllocatorTest.cpp
  core/RBLNSupportedDtypesTest.cpp
  core/RBLNTensorUtilsTest.cpp
  core/RBLNV2VBatchTest.cpp
)

foreach(source ${sources})
  get_filename_component(test_file_name ${source} NAME_WE)
  set(target test_cpp_${test_file_name})
  add_executable(${target} ${source})
  target_link_libraries(${target} PUBLIC torch_rbln GTest::gtest_main)
  gtest_discover_tests(${target}
    DISCOVERY_MODE PRE_TEST
    DISCOVERY_TIMEOUT 600
  )
endforeach()
