//===-- runtime/devices.cpp - The host's calls of the devices -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// devices.h over libomptarget, or over liboffload in a program without it.
// Both are weak references: a program without offloading has no device, and
// a region then runs on the host, which calls none of these.
//
//===----------------------------------------------------------------------===//

#include "devices.h"
#include "process.h"
#include <cstdio>
#include <cstring>
#include <pthread.h>

extern "C" void *omp_target_alloc(std::size_t, int) __attribute__((weak));
extern "C" void omp_target_free(void *, int) __attribute__((weak));
extern "C" void *llvm_omp_target_alloc_host(std::size_t, int)
    __attribute__((weak));
extern "C" void llvm_omp_target_free_host(void *, int) __attribute__((weak));
extern "C" int omp_target_memcpy(void *, const void *, std::size_t,
    std::size_t, std::size_t, int, int) __attribute__((weak));
extern "C" int omp_get_initial_device(void) __attribute__((weak));
extern "C" int omp_get_num_devices(void) __attribute__((weak));
extern "C" int omp_get_device_from_uid(const char *) __attribute__((weak));
extern "C" void __tgt_register_rpc_callback(unsigned (*)(void *, unsigned))
    __attribute__((weak));

namespace merkurius::runtime {

namespace {

Gpu gpus[kMostGpus];
int gpuCount{0};
pthread_once_t gpusOnce = PTHREAD_ONCE_INIT;
pthread_mutex_t callbackLock = PTHREAD_MUTEX_INITIALIZER;
unsigned (*rpcCallback)(void *, unsigned){nullptr};
bool gpusFound{false};

[[noreturn]] void Fail(const char *what, ol_result_t result) {
  std::fflush(stdout);
  std::fprintf(stderr, "Mercury runtime: %s: %s\n", what,
      result && result->details ? result->details : "failed");
  ExitProgram(1);
}

int OpenMPDevices();

// Whether the device is one of OpenMP's, which libomptarget's calls take.
bool OpenMPDevice(int device) {
  return omp_target_alloc && device < OpenMPDevices();
}

bool AddDevice(ol_device_handle_t device, void *) {
  Gpu gpu{};
  gpu.device = device;
  if (ol_result_t result{olGetDeviceInfo(
          device, kOlDeviceInfoPlatform, sizeof gpu.platform, &gpu.platform)}) {
    Fail("could not ask liboffload for a device's platform", result);
  }
  std::uint32_t backend{0};
  if (ol_result_t result{olGetPlatformInfo(
          gpu.platform, kOlPlatformInfoBackend, sizeof backend, &backend)}) {
    Fail("could not ask liboffload for a platform's backend", result);
  }
  if (backend == kOlPlatformBackendHost || gpuCount == kMostGpus) {
    return true;
  }
  olGetDeviceInfo(device, kOlDeviceInfoName, sizeof gpu.name - 1, gpu.name);
  olGetDeviceInfo(device, kOlDeviceInfoUid, sizeof gpu.uid - 1, gpu.uid);
  gpus[gpuCount++] = gpu;
  return true;
}

// How many devices OpenMP numbers, or 0 without OpenMP's offloading.
int OpenMPDevices() {
  return omp_target_alloc && omp_get_num_devices ? omp_get_num_devices() : 0;
}

// Each GPU's number: OpenMP's for its UID, or else the next after OpenMP's
// devices.
void NumberGpus() {
  int next{OpenMPDevices()};
  for (int i{0}; i < gpuCount; ++i) {
    int number{next > 0 && omp_get_device_from_uid
            ? omp_get_device_from_uid(gpus[i].uid)
            : -1};
    gpus[i].number = number >= 0 && number < OpenMPDevices() ? number : next++;
  }
}

void RegisterWithPlatforms(unsigned (*callback)(void *, unsigned)) {
  for (int i{0}; i < gpuCount; ++i) {
    bool done{false};
    for (int j{0}; j < i; ++j) {
      done = done || gpus[j].platform == gpus[i].platform;
    }
    if (done) {
      continue;
    }
    if (ol_result_t result{
            olPlatformRegisterRPCCallback(gpus[i].platform, callback)}) {
      Fail("could not register the runtime's RPC callback with liboffload",
          result);
    }
  }
}

void FindGpus() {
  if (!olInit) {
    return;
  }
  if (ol_result_t result{olInit(nullptr)}) {
    Fail("could not initialize liboffload", result);
  }
  if (ol_result_t result{olIterateDevices(AddDevice, nullptr)}) {
    Fail("could not list liboffload's devices", result);
  }
  NumberGpus();
  for (int i{0}; i < gpuCount; ++i) {
    if (ol_result_t result{
            olCreateContext(1, &gpus[i].device, &gpus[i].context)}) {
      Fail("could not make a liboffload context for a GPU", result);
    }
  }
  // After liboffload's teardown, which its start registered.
  EndExitWhileRegionsRun();
  pthread_mutex_lock(&callbackLock);
  gpusFound = true;
  if (rpcCallback) {
    RegisterWithPlatforms(rpcCallback);
  }
  pthread_mutex_unlock(&callbackLock);
}

} // namespace

int GpuCount() {
  pthread_once(&gpusOnce, FindGpus);
  return gpuCount;
}

const Gpu *GpuOf(int number) {
  int count{GpuCount()};
  for (int i{0}; i < count; ++i) {
    if (gpus[i].number == number) {
      return &gpus[i];
    }
  }
  return nullptr;
}

void *DeviceAllocate(int device, std::size_t bytes) {
  if (OpenMPDevice(device)) {
    return omp_target_alloc(bytes, device);
  }
  const Gpu *gpu{GpuOf(device)};
  void *memory{nullptr};
  if (!gpu ||
      olMemAlloc(gpu->context, gpu->device, kOlAllocTypeDevice, bytes,
          &memory) != nullptr) {
    return nullptr;
  }
  return memory;
}

void DeviceFree(int device, void *memory) {
  if (OpenMPDevice(device)) {
    omp_target_free(memory, device);
  } else if (const Gpu *gpu{GpuOf(device)}) {
    olMemFree(gpu->context, memory);
  }
}

void *DeviceAllocateHost(int device, std::size_t bytes) {
  if (OpenMPDevice(device)) {
    return llvm_omp_target_alloc_host
        ? llvm_omp_target_alloc_host(bytes, device)
        : nullptr;
  }
  const Gpu *gpu{GpuOf(device)};
  void *memory{nullptr};
  if (!gpu ||
      olMemAllocHost(gpu->context, gpu->device, bytes, &memory) != nullptr) {
    return nullptr;
  }
  return memory;
}

void DeviceFreeHost(int device, void *memory) {
  if (OpenMPDevice(device)) {
    if (llvm_omp_target_free_host) {
      llvm_omp_target_free_host(memory, device);
    }
  } else if (const Gpu *gpu{GpuOf(device)}) {
    olMemFree(gpu->context, memory);
  }
}

bool CopyToDevice(int device, void *to, const void *from, std::size_t bytes) {
  if (OpenMPDevice(device)) {
    return omp_target_memcpy && omp_get_initial_device &&
        omp_target_memcpy(
            to, from, bytes, 0, 0, device, omp_get_initial_device()) == 0;
  }
  const Gpu *gpu{GpuOf(device)};
  return gpu &&
      olMemcpy(nullptr, to, gpu->device, from, nullptr, bytes) == nullptr;
}

bool CopyFromDevice(int device, void *to, const void *from, std::size_t bytes) {
  if (OpenMPDevice(device)) {
    return omp_target_memcpy && omp_get_initial_device &&
        omp_target_memcpy(
            to, from, bytes, 0, 0, omp_get_initial_device(), device) == 0;
  }
  const Gpu *gpu{GpuOf(device)};
  return gpu &&
      olMemcpy(nullptr, to, nullptr, from, gpu->device, bytes) == nullptr;
}

void RegisterRpcCallback(unsigned (*callback)(void *port, unsigned lanes)) {
  if (__tgt_register_rpc_callback) {
    __tgt_register_rpc_callback(callback);
  }
  pthread_mutex_lock(&callbackLock);
  rpcCallback = callback;
  if (gpusFound) {
    RegisterWithPlatforms(callback);
  }
  pthread_mutex_unlock(&callbackLock);
}

} // namespace merkurius::runtime
