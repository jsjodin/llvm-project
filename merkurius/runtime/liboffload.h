//===-- runtime/liboffload.h - What the runtime uses of liboffload -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The functions, types and values of LLVM's offload library, liboffload
// (offload/liboffload), that the runtime calls to load and launch GPU
// processes' kernels (docs/INPROGRESS.md, GPU processes without omp target,
// step 4c). liboffload's own header, OffloadAPI.h, is generated in the
// runtimes' build, which merkrt is not built after, so what is used is
// declared here, as offload/liboffload/API/*.td defines it.
//
// The functions are weak references: a program links libLLVMOffload only
// if it has GPU process kinds, and without it the node has no GPUs for
// processes.
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_RUNTIME_LIBOFFLOAD_H_
#define MERKURIUS_RUNTIME_LIBOFFLOAD_H_

#include <cstddef>
#include <cstdint>

extern "C" {

struct ol_error_struct_t {
  std::uint32_t code; // ol_errc_t, OL_ERRC_SUCCESS = 0
  const char *details;
};
// Null for success.
using ol_result_t = const ol_error_struct_t *;

using ol_platform_handle_t = struct ol_platform_impl_t *;
using ol_device_handle_t = struct ol_device_impl_t *;
using ol_context_handle_t = struct ol_context_impl_t *;
using ol_queue_handle_t = struct ol_queue_impl_t *;
using ol_program_handle_t = struct ol_program_impl_t *;
using ol_symbol_handle_t = struct ol_symbol_impl_t *;

// ol_device_info_t
constexpr std::uint32_t kOlDeviceInfoPlatform{1};
constexpr std::uint32_t kOlDeviceInfoName{2};
constexpr std::uint32_t kOlDeviceInfoUid{4};
constexpr std::uint32_t kOlDeviceInfoNumComputeUnits{12};
constexpr std::uint32_t kOlDeviceInfoNumLanes{19};
// ol_platform_info_t
constexpr std::uint32_t kOlPlatformInfoBackend{3};
// ol_platform_backend_t
constexpr std::uint32_t kOlPlatformBackendHost{4};
// ol_symbol_kind_t
constexpr std::uint32_t kOlSymbolKindKernel{0};
constexpr std::uint32_t kOlSymbolKindGlobalVariable{1};
// ol_symbol_info_t
constexpr std::uint32_t kOlSymbolInfoGlobalVariableAddress{1};
// ol_alloc_type_t
constexpr std::uint32_t kOlAllocTypeDevice{1};
// ol_platform_backend_t
constexpr std::uint32_t kOlPlatformBackendAmdgpu{2};
// ol_kernel_launch_prop_type_t
constexpr std::uint32_t kOlKernelLaunchPropTypeNone{0};
constexpr std::uint32_t kOlKernelLaunchPropTypeIsCooperative{1};
constexpr std::uint32_t kOlKernelLaunchPropTypeStackSize{2};

// A launch's properties, ended by one of type kOlKernelLaunchPropTypeNone.
struct ol_kernel_launch_prop_t {
  std::uint32_t type;
  void *data;
};

struct ol_dimensions_t {
  std::uint32_t x, y, z;
};
struct ol_kernel_launch_size_args_t {
  std::size_t dimensions;
  ol_dimensions_t numGroups;
  ol_dimensions_t groupSize;
  std::size_t dynSharedMemory;
};

ol_result_t olInit(const void *initArgs) __attribute__((weak));
ol_result_t olIterateDevices(bool (*callback)(ol_device_handle_t, void *),
    void *data) __attribute__((weak));
ol_result_t olGetDeviceInfo(ol_device_handle_t device, std::uint32_t property,
    std::size_t size, void *value) __attribute__((weak));
ol_result_t olGetPlatformInfo(ol_platform_handle_t platform,
    std::uint32_t property, std::size_t size, void *value)
    __attribute__((weak));
ol_result_t olPlatformRegisterRPCCallback(ol_platform_handle_t platform,
    unsigned (*callback)(void *port, unsigned lanes)) __attribute__((weak));
ol_result_t olCreateContext(std::size_t count, ol_device_handle_t *devices,
    ol_context_handle_t *context) __attribute__((weak));
ol_result_t olCreateProgram(ol_context_handle_t context,
    ol_device_handle_t device, const void *image, std::size_t size,
    ol_program_handle_t *program) __attribute__((weak));
ol_result_t olGetSymbol(ol_program_handle_t program, const char *name,
    std::uint32_t kind, ol_symbol_handle_t *symbol) __attribute__((weak));
ol_result_t olGetSymbolInfo(ol_symbol_handle_t symbol, std::uint32_t property,
    std::size_t size, void *value) __attribute__((weak));
ol_result_t olCreateQueue(ol_context_handle_t context,
    ol_device_handle_t device, ol_queue_handle_t *queue) __attribute__((weak));
ol_result_t olDestroyQueue(ol_queue_handle_t queue) __attribute__((weak));
ol_result_t olLaunchKernel(ol_queue_handle_t queue, ol_device_handle_t device,
    ol_symbol_handle_t kernel, const ol_kernel_launch_size_args_t *size,
    const void *properties, std::size_t count, void **arguments,
    const std::size_t *sizes) __attribute__((weak));
ol_result_t olGetKernelMaxCooperativeGroupCount(ol_device_handle_t device,
    ol_symbol_handle_t kernel, const ol_kernel_launch_size_args_t *size,
    std::uint32_t *count) __attribute__((weak));
ol_result_t olSyncQueue(ol_queue_handle_t queue) __attribute__((weak));
// Queues limited to some compute units, for GPU processes (compute-units.h).
ol_result_t olGetDeviceComputeUnitMask(ol_device_handle_t device,
    std::size_t words, std::uint32_t *mask) __attribute__((weak));
ol_result_t olCreateQueueWithComputeUnitMask(ol_context_handle_t context,
    ol_device_handle_t device, const std::uint32_t *mask, std::size_t words,
    ol_queue_handle_t *queue) __attribute__((weak));
ol_result_t olGetKernelMaxCooperativeGroupCountOnQueue(ol_queue_handle_t queue,
    ol_symbol_handle_t kernel, const ol_kernel_launch_size_args_t *size,
    std::uint32_t *count) __attribute__((weak));
ol_result_t olLaunchHostFunction(ol_queue_handle_t queue,
    void (*callback)(void *data), void *data) __attribute__((weak));
ol_result_t olMemAlloc(ol_context_handle_t context, ol_device_handle_t device,
    std::uint32_t type, std::size_t size, void **memory) __attribute__((weak));
ol_result_t olMemAllocHost(ol_context_handle_t context,
    ol_device_handle_t device, std::size_t size, void **memory)
    __attribute__((weak));
ol_result_t olMemFree(ol_context_handle_t context, void *memory)
    __attribute__((weak));
ol_result_t olMemcpy(ol_queue_handle_t queue, void *to, ol_device_handle_t toDevice,
    const void *from, ol_device_handle_t fromDevice, std::size_t size)
    __attribute__((weak));

} // extern "C"

#endif // MERKURIUS_RUNTIME_LIBOFFLOAD_H_
