cmake_minimum_required(VERSION 3.20.0)

set(LLVM_SUBPROJECT_TITLE "Merkurius")

# Merkurius builds either inside an LLVM build, as an external project
# (-DLLVM_EXTERNAL_PROJECTS=merkurius), or standalone against an LLVM and MLIR
# build or installation (-DMLIR_DIR=...).
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(Merkurius LANGUAGES C CXX)
  set(MERKURIUS_STANDALONE_BUILD ON)
else()
  set(MERKURIUS_STANDALONE_BUILD OFF)
endif()

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(FATAL_ERROR "In-source builds are not allowed. "
    "Create a build directory and run cmake from there.")
endif()

set(MERKURIUS_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(MERKURIUS_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

set(MERKURIUS_VERSION_MAJOR 0)
set(MERKURIUS_VERSION_MINOR 1)
set(MERKURIUS_VERSION_PATCHLEVEL 0)
set(MERKURIUS_VERSION
  "${MERKURIUS_VERSION_MAJOR}.${MERKURIUS_VERSION_MINOR}.${MERKURIUS_VERSION_PATCHLEVEL}")

if(MERKURIUS_STANDALONE_BUILD)
  find_package(MLIR REQUIRED CONFIG)
  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")

  find_program(LLVM_TABLEGEN_EXE "llvm-tblgen" ${LLVM_TOOLS_BINARY_DIR}
    NO_DEFAULT_PATH)
  if(NOT LLVM_TABLEGEN_EXE)
    message(FATAL_ERROR "llvm-tblgen not found in ${LLVM_TOOLS_BINARY_DIR}; "
      "build LLVM first, or set LLVM_TABLEGEN_EXE.")
  endif()

  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
  set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

  # LLVM_DEFINITIONS is a space-separated string; HandleLLVMOptions overwrites
  # it, so split and apply it first.
  separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
  add_definitions(${LLVM_DEFINITIONS_LIST})

  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)
  include(HandleLLVMOptions)

  set(LLVM_LIT_ARGS "-sv" CACHE STRING "Default options for lit")
  if(NOT LLVM_EXTERNAL_LIT AND EXISTS "${LLVM_TOOLS_BINARY_DIR}/llvm-lit")
    set(LLVM_EXTERNAL_LIT "${LLVM_TOOLS_BINARY_DIR}/llvm-lit")
  endif()

  include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
  link_directories(${LLVM_BUILD_LIBRARY_DIR})
else()
  if(NOT "mlir" IN_LIST LLVM_ENABLE_PROJECTS)
    message(FATAL_ERROR "Merkurius needs MLIR: add mlir to LLVM_ENABLE_PROJECTS.")
  endif()
  set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir)
  set(MLIR_INCLUDE_DIRS
    ${MLIR_MAIN_SRC_DIR}/include ${LLVM_BINARY_DIR}/tools/mlir/include)
  list(APPEND CMAKE_MODULE_PATH "${MLIR_MAIN_SRC_DIR}/cmake/modules")
  include(AddMLIR)
  include_directories(${MLIR_INCLUDE_DIRS})
endif()

list(APPEND CMAKE_MODULE_PATH "${MERKURIUS_SOURCE_DIR}/cmake/modules")
include(AddMerkurius)

include_directories(BEFORE
  ${MERKURIUS_BINARY_DIR}/include
  ${MERKURIUS_SOURCE_DIR}/include)

# LLVMConfig.cmake does not export LLVM_INCLUDE_TESTS, so standalone builds
# include the tests by default.
if(DEFINED LLVM_INCLUDE_TESTS)
  set(MERKURIUS_INCLUDE_TESTS_DEFAULT ${LLVM_INCLUDE_TESTS})
else()
  set(MERKURIUS_INCLUDE_TESTS_DEFAULT ON)
endif()
option(MERKURIUS_INCLUDE_TESTS "Generate build targets for the Merkurius tests."
  ${MERKURIUS_INCLUDE_TESTS_DEFAULT})

set(MERKURIUS_GTEST_AVAIL OFF)
if(MERKURIUS_INCLUDE_TESTS)
  if(TARGET llvm_gtest)
    set(MERKURIUS_GTEST_AVAIL ON)
  elseif(EXISTS ${LLVM_THIRD_PARTY_DIR}/unittest/googletest/include/gtest/gtest.h)
    add_subdirectory(${LLVM_THIRD_PARTY_DIR}/unittest third-party/unittest)
    set(MERKURIUS_GTEST_AVAIL ON)
  else()
    message(WARNING "Skipping Merkurius unit tests: googletest not found.")
  endif()
endif()

add_custom_target(merkurius-headers)
set_target_properties(merkurius-headers PROPERTIES FOLDER "Merkurius/Misc")

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)

if(MERKURIUS_INCLUDE_TESTS)
  if(MERKURIUS_GTEST_AVAIL)
    add_subdirectory(unittests)
  endif()
  add_subdirectory(test)
endif()
