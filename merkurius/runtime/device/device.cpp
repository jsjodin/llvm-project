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

using merkurius::runtime::rpcSend;

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
