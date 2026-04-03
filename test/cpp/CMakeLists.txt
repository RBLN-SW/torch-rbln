set(sources
  c10d/ProcessGroupRBLNTest.cpp
  core/RBLNAllocatorTest.cpp
  core/RBLNCopyOpTest.cpp
  core/RBLNDeviceGuardTest.cpp
  core/RBLNFunctionsTest.cpp
  core/RBLNLoggingTest.cpp
  core/RBLNTensorUtilsTest.cpp
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

  list(APPEND brpath
    $ORIGIN
    $ORIGIN/../lib
    ${CMAKE_BUILD_RPATH}
  )
  list(APPEND irpath
    $ORIGIN
    $ORIGIN/../lib
    ${CMAKE_INSTALL_RPATH}
  )
  set_target_properties(${target} PROPERTIES
    BUILD_RPATH "${brpath}"
    INSTALL_RPATH "${irpath}"
  )

endforeach()
