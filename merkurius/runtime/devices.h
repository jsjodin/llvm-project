//===-- runtime/devices.h - The host's calls of the devices -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// What the host's runtime asks of the devices: memory on a device, host
// memory that a device reads directly, copies between them, and the
// callback that serves the devices' RPC opcodes (docs/INPROGRESS.md, GPU
// processes without omp target, step 3). A device is OpenMP's number of it,
// which the device's runtime sends with its opcodes.
//
// Behind it is libomptarget, through weak references, so that a program
// without offloading links without it. A program with GPU processes but no
// target regions has no libomptarget, and these calls then use liboffload,
// the runtime's own copy, through which it loads the processes' images
// (step 4c).
//
// The GPUs for processes are the devices that liboffload finds, but for
// the host. liboffload has no numbers for them, only handles, in the order
// of its plugins and their devices. OpenMP numbers a device when an image
// for it is first registered, so its numbers depend on the program's
// images (offload/libompaccsupport/PluginManager.cpp). A GPU has OpenMP's
// number for the device with the same UID, so that a GPU process's device
// number and a region's agree; a GPU that OpenMP does not number, or every
// GPU in a program without OpenMP's offloading, is numbered after OpenMP's
// devices, in liboffload's order. The calls below use libomptarget for
// OpenMP's devices and liboffload for the others.
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_RUNTIME_DEVICES_H_
#define MERKURIUS_RUNTIME_DEVICES_H_

#include "liboffload.h"
#include <cstddef>

namespace merkurius::runtime {

// `bytes` of the device's memory, or null.
void *DeviceAllocate(int device, std::size_t bytes);
void DeviceFree(int device, void *memory);

// `bytes` of host memory that the device reads and writes directly, or null.
void *DeviceAllocateHost(int device, std::size_t bytes);
void DeviceFreeHost(int device, void *memory);

// Copies `bytes` from the host to the device, or from the device to the
// host; whether it did.
bool CopyToDevice(int device, void *to, const void *from, std::size_t bytes);
bool CopyFromDevice(int device, void *to, const void *from, std::size_t bytes);

// Registers the callback that serves our RPC opcodes on every device, before
// libc's: it answers RPC_UNHANDLED_OPCODE for any other. It serves
// libomptarget's devices, and liboffload's once it has found them.
void RegisterRpcCallback(unsigned (*callback)(void *port, unsigned lanes));

// A GPU for processes, as liboffload has it.
struct Gpu {
  int number;
  ol_device_handle_t device;
  ol_platform_handle_t platform;
  ol_context_handle_t context;
  char name[64]; // as --offload-arch names it: gfx1030
  char uid[128];
};

// The most GPUs the runtime uses for processes.
constexpr int kMostGpus{64};

// How many GPUs there are for processes; liboffload is initialized the
// first time it is asked.
int GpuCount();
// The GPU with the number, or null. Without OpenMP's offloading, they are
// numbered from 0.
const Gpu *GpuOf(int number);

} // namespace merkurius::runtime

#endif // MERKURIUS_RUNTIME_DEVICES_H_
