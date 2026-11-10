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
// without offloading links without it. A GPU process's image will be
// loaded through liboffload, the runtime's own copy, which these calls will
// then use for it (step 4).
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_RUNTIME_DEVICES_H_
#define MERKURIUS_RUNTIME_DEVICES_H_

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
// libc's: it answers RPC_UNHANDLED_OPCODE for any other.
void RegisterRpcCallback(unsigned (*callback)(void *port, unsigned lanes));

} // namespace merkurius::runtime

#endif // MERKURIUS_RUNTIME_DEVICES_H_
