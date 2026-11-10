//===-- runtime/devices.cpp - The host's calls of the devices -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// devices.h over libomptarget. Its functions are weak references: a program
// without offloading has no device, and a region then runs on the host,
// which calls none of these.
//
//===----------------------------------------------------------------------===//

#include "devices.h"

extern "C" void *omp_target_alloc(std::size_t, int) __attribute__((weak));
extern "C" void omp_target_free(void *, int) __attribute__((weak));
extern "C" void *llvm_omp_target_alloc_host(std::size_t, int)
    __attribute__((weak));
extern "C" void llvm_omp_target_free_host(void *, int) __attribute__((weak));
extern "C" int omp_target_memcpy(void *, const void *, std::size_t,
    std::size_t, std::size_t, int, int) __attribute__((weak));
extern "C" int omp_get_initial_device(void) __attribute__((weak));
extern "C" void __tgt_register_rpc_callback(unsigned (*)(void *, unsigned))
    __attribute__((weak));

namespace merkurius::runtime {

void *DeviceAllocate(int device, std::size_t bytes) {
  return omp_target_alloc ? omp_target_alloc(bytes, device) : nullptr;
}

void DeviceFree(int device, void *memory) {
  if (omp_target_free) {
    omp_target_free(memory, device);
  }
}

void *DeviceAllocateHost(int device, std::size_t bytes) {
  return llvm_omp_target_alloc_host ? llvm_omp_target_alloc_host(bytes, device)
                                    : nullptr;
}

void DeviceFreeHost(int device, void *memory) {
  if (llvm_omp_target_free_host) {
    llvm_omp_target_free_host(memory, device);
  }
}

bool CopyToDevice(int device, void *to, const void *from, std::size_t bytes) {
  return omp_target_memcpy && omp_get_initial_device &&
      omp_target_memcpy(to, from, bytes, 0, 0, device,
          omp_get_initial_device()) == 0;
}

bool CopyFromDevice(int device, void *to, const void *from, std::size_t bytes) {
  return omp_target_memcpy && omp_get_initial_device &&
      omp_target_memcpy(to, from, bytes, 0, 0, omp_get_initial_device(),
          device) == 0;
}

void RegisterRpcCallback(unsigned (*callback)(void *port, unsigned lanes)) {
  if (__tgt_register_rpc_callback) {
    __tgt_register_rpc_callback(callback);
  }
}

} // namespace merkurius::runtime
