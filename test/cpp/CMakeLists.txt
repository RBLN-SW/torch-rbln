cmake_minimum_required(VERSION 3.31...4.4)

if(NOT TARGET TorchRBLN::torch_rbln)
  project(torch_rbln_cpp_tests LANGUAGES CXX)

  set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)

  set(BUILD_SHARED_LIBS ON)

  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

  enable_testing()

  set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/../../cmake)
  include(TorchRBLNCheckCompiler)

  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  string(CONCAT torch_rbln_probe_code
    "import importlib.util; print(importlib.util.find_spec('torch_rbln')"
    ".submodule_search_locations[0])")
  execute_process(
    # Without `-I`, a `torch_rbln` in the working directory shadows the
    # installed one.
    COMMAND ${Python3_EXECUTABLE} -I -c "${torch_rbln_probe_code}"
    OUTPUT_VARIABLE torch_rbln_package_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  find_package(TorchRBLN CONFIG REQUIRED
    HINTS "${torch_rbln_package_dir}/share/cmake")
  include(TorchRBLNBuildFlags)
  include(External/googletest)
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
  target_link_libraries(${target} PRIVATE
    TorchRBLN::torch_rbln
    GTest::gtest_main
  )
  gtest_discover_tests(${target}
    DISCOVERY_MODE PRE_TEST
    DISCOVERY_TIMEOUT 600
  )
endforeach()
