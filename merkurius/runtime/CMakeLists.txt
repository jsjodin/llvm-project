# libmerkrt, the runtime library linked into every program (design
# document, "Runtime system"). It is built as a static library in the build
# directory's lib, where the driver looks for it beside bin/merkc, and
# installed there too.
add_library(merkrt STATIC
  array.cpp
  encode.cpp
  exception.cpp
  heap.cpp
  mailbox.cpp
  process.cpp
  main.cpp
  node.cpp
  offload.cpp
  read.cpp
  sha256.cpp
  string.cpp
  trace.cpp
  type-info.cpp
  write.cpp
  )
set_target_properties(merkrt PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY ${LLVM_LIBRARY_OUTPUT_INTDIR}
  POSITION_INDEPENDENT_CODE ON
  FOLDER "Merkurius/Runtime")
install(TARGETS merkrt
  ARCHIVE DESTINATION lib${LLVM_LIBDIR_SUFFIX}
  COMPONENT merkrt)

# The host's side of target regions (offload.cpp) uses the offload plugin's
# RPC protocol, LLVM libc's header-only one.
target_include_directories(merkrt PRIVATE ${LLVM_MAIN_SRC_DIR}/../libc)

# libmerkdevice-amdgcn.a, the device's runtime (device/device.cpp), as
# bitcode for any AMD GPU, which the driver links into the device image
# beside the OpenMP device runtime. It is compiled by this build's Clang, as
# the OpenMP device runtime is.
if (TARGET clang AND TARGET llvm-ar AND "AMDGPU" IN_LIST LLVM_TARGETS_TO_BUILD)
  set(merk_device_bc ${CMAKE_CURRENT_BINARY_DIR}/device-amdgcn.bc)
  set(merk_device_array_bc ${CMAKE_CURRENT_BINARY_DIR}/device-array-amdgcn.bc)
  set(merk_device_lib ${LLVM_LIBRARY_OUTPUT_INTDIR}/libmerkdevice-amdgcn.a)
  add_custom_command(OUTPUT ${merk_device_bc}
    COMMAND $<TARGET_FILE:clang> --target=amdgcn-amd-amdhsa -nogpulib -O2
      -flto -fvisibility=hidden -fno-exceptions -fno-rtti -std=c++17
      -I${LLVM_MAIN_SRC_DIR}/../libc
      -c ${CMAKE_CURRENT_SOURCE_DIR}/device/device.cpp -o ${merk_device_bc}
    DEPENDS clang ${CMAKE_CURRENT_SOURCE_DIR}/device/device.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/rpc-opcodes.h
      ${CMAKE_CURRENT_SOURCE_DIR}/device-ring.h
    COMMENT "Building the Merkurius device runtime for AMD GPUs")
  add_custom_command(OUTPUT ${merk_device_array_bc}
    COMMAND $<TARGET_FILE:clang> --target=amdgcn-amd-amdhsa -nogpulib -O2
      -flto -fvisibility=hidden -fno-exceptions -fno-rtti -std=c++17
      -c ${CMAKE_CURRENT_SOURCE_DIR}/array.cpp -o ${merk_device_array_bc}
    DEPENDS clang ${CMAKE_CURRENT_SOURCE_DIR}/array.cpp
    COMMENT "Building the Merkurius device runtime's arrays for AMD GPUs")
  add_custom_command(OUTPUT ${merk_device_lib}
    COMMAND ${CMAKE_COMMAND} -E remove -f ${merk_device_lib}
    COMMAND $<TARGET_FILE:llvm-ar> rcs ${merk_device_lib} ${merk_device_bc}
      ${merk_device_array_bc}
    DEPENDS llvm-ar ${merk_device_bc} ${merk_device_array_bc})
  add_custom_target(merkdevice ALL DEPENDS ${merk_device_lib})
  set_target_properties(merkdevice PROPERTIES FOLDER "Merkurius/Runtime")
  install(FILES ${merk_device_lib}
    DESTINATION lib${LLVM_LIBDIR_SUFFIX}
    COMPONENT merkrt)
endif()
