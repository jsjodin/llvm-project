//===-- runtime/device/device.cpp - The device's runtime -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The runtime that target regions call on a GPU (design document, "The
// device runtime"): send and receive over the offload plugin's RPC server,
// with our opcodes (rpc-opcodes.h), which the host's callback (offload.cpp)
// handles. It is compiled to bitcode, which the driver links into the device
// image beside the OpenMP device runtime, whose RPC client it uses. The
// host's merkrt has the same functions for a region that runs on the host.
//
//===----------------------------------------------------------------------===//

#include "../rpc-opcodes.h"
#include "shared/rpc.h"

// The OpenMP device runtime's client, which the plugin points at the host's
// buffer when it loads the image.
extern "C" rpc::Client __llvm_rpc_client;

extern "C" void *malloc(unsigned long);
// The OpenMP device runtime's, by the device's steady clock, whose
// frequency the plugin gives it.
extern "C" double omp_get_wtime(void);

using merkurius::runtime::rpcPoll;
using merkurius::runtime::rpcSend;

// A cell of the heap, which the device does not have: the OpenMP device
// runtime's malloc, which is LLVM libc's GPU allocator when the OpenMP
// device runtime is built with libc, and otherwise a bump allocator of
// 1 MiB. Nothing is freed yet (docs/INPROGRESS.md, The device's side of
// the first slice, step 6, the device's arena).
extern "C" void *merk_device_alloc(uint64_t bytes) { return malloc(bytes); }

// Sends a one-word message to the mailbox whose handle is the cell of five
// words at `handle`. The host queues it, or sends it on to another node,
// encoding it by `typeInfo`, which is the host's. It returns once the host
// has, as a send on the host does: a send alone returns when the buffer is
// handed over, so a region could end before its messages were queued, and
// the next send could take another port, which the server may serve first.
extern "C" void merk_device_send(
    const uint64_t *handle, uint64_t typeInfo, uint64_t message) {
  rpc::Client::Port port{__llvm_rpc_client.open<rpcSend>()};
  port.send([&](rpc::Buffer *buffer, uint32_t) {
    for (int i{0}; i < 5; ++i) {
      buffer->data[i] = handle[i];
    }
    buffer->data[5] = typeInfo;
    buffer->data[6] = message;
  });
  port.recv([](rpc::Buffer *, uint32_t) {});
}

namespace {

// One poll of the mailbox: whether it had a message, which is then in
// `message`.
bool Poll(const uint64_t *handle, uint64_t *message) {
  rpc::Client::Port port{__llvm_rpc_client.open<rpcPoll>()};
  port.send([&](rpc::Buffer *buffer, uint32_t) {
    for (int i{0}; i < 5; ++i) {
      buffer->data[i] = handle[i];
    }
  });
  bool found{false};
  port.recv([&](rpc::Buffer *buffer, uint32_t) {
    found = buffer->data[0] != 0;
    *message = buffer->data[1];
  });
  return found;
}


} // namespace

// Takes the oldest message of the mailbox whose handle is the cell of five
// words at `handle`, polling the host, which has the mailbox
// (docs/INPROGRESS.md, The device's side of the first slice, Decision 2),
// until it has one or `milliseconds` pass, or forever if it is negative.
// It returns 1 with the message in `message`, or 0.
extern "C" int merk_device_receive(
    const uint64_t *handle, int64_t milliseconds, uint64_t *message) {
  double deadline{omp_get_wtime() + static_cast<double>(milliseconds) / 1000};
  for (;;) {
    if (Poll(handle, message)) {
      return 1;
    }
    if (milliseconds >= 0 && omp_get_wtime() >= deadline) {
      return 0;
    }
    // About 8 microseconds (127 * 64 clocks at about 1 GHz) between polls,
    // so that the device does not keep the host's server thread busy.
    __builtin_amdgcn_s_sleep(127);
  }
}
