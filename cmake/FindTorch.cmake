find_package(Python3
  COMPONENTS Interpreter Development
  REQUIRED
)
if(NOT DEFINED PYTHON_SITE_PACKAGES)
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(NOT PYTHON_SITE_PACKAGES)
    message(FATAL_ERROR "Failed to get Python site packages")
  endif()
endif()

find_package(Torch REQUIRED
  PATHS
    ${PYTHON_SITE_PACKAGES}/torch/share/cmake/Torch
    ${PYTHON_SITE_PACKAGES}/torch/share/cmake
  NO_DEFAULT_PATH
)

# Check if Torch version is 2.8.0 or higher and enforce GCC/G++ 13+
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.__version__)"
  OUTPUT_VARIABLE TORCH_VERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE TORCH_RESULT
)
if(TORCH_VERSION VERSION_GREATER_EQUAL "2.8.0")
  message(STATUS "Torch version ${TORCH_VERSION} detected (>= 2.8.0), checking compiler version...")

  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "13.0")
      message(FATAL_ERROR "Torch ${TORCH_VERSION} requires GCC/G++ version 13.0 or higher. "
                          "Current version: ${CMAKE_CXX_COMPILER_VERSION}. "
                          "Please use a newer compiler.")
    endif()
    message(STATUS "GCC/G++ version ${CMAKE_CXX_COMPILER_VERSION} meets requirement (>= 13.0)")
  else()
    message(FATAL_ERROR "Torch ${TORCH_VERSION} requires GCC/G++ compiler (version 13.0 or higher). "
                        "Current compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}. "
                        "Please use GCC/G++ 13.0 or higher.")
  endif()
endif()

# Set runtime paths for torch
list(APPEND CMAKE_BUILD_RPATH ${PYTHON_SITE_PACKAGES}/torch/lib)
list(APPEND CMAKE_INSTALL_RPATH "$ORIGIN/../../torch/lib")

# Create a symlink to the torch directory in the install prefix
install(CODE "execute_process(
  COMMAND ${CMAKE_COMMAND} -E create_symlink ${PYTHON_SITE_PACKAGES}/torch torch
  WORKING_DIRECTORY ${CMAKE_INSTALL_PREFIX}/..
)")
