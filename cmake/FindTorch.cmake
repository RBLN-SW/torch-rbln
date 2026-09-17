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

# torch-rbln and rebel-compiler both require GCC/G++ 13+.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "13.0")
    message(FATAL_ERROR "torch-rbln requires GCC/G++ version 13.0 or higher. "
                        "Current version: ${CMAKE_CXX_COMPILER_VERSION}. "
                        "Please use CC=gcc-13 CXX=g++-13 (Debian/Ubuntu) or gcc-toolset-13 (RHEL).")
  endif()
  message(STATUS "GCC/G++ version ${CMAKE_CXX_COMPILER_VERSION} meets requirement (>= 13.0)")
else()
  message(FATAL_ERROR "torch-rbln requires the GCC/G++ compiler (version 13.0 or higher). "
                      "Current compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}.")
endif()

# Set runtime paths for torch
list(APPEND CMAKE_BUILD_RPATH ${PYTHON_SITE_PACKAGES}/torch/lib)
list(APPEND CMAKE_INSTALL_RPATH "$ORIGIN/../../torch/lib")

# Create a symlink to the torch directory in the install prefix
install(CODE "execute_process(
  COMMAND ${CMAKE_COMMAND} -E create_symlink ${PYTHON_SITE_PACKAGES}/torch torch
  WORKING_DIRECTORY ${CMAKE_INSTALL_PREFIX}/..
)")
