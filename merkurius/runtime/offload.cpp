//===-- runtime/offload.cpp - The host's side of target regions ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// What a target region's sends and receives need of the host (design
// document, "Send and receive in a target region"). A region that runs on a
// GPU calls the device's runtime (device/device.cpp), which sends our
// opcodes (rpc-opcodes.h) over the offload plugin's RPC server; the callback
// here handles them on the plugin's server thread. A region that runs on the
// host calls the same functions here.
//
// The callback is registered with libomptarget's __tgt_register_rpc_callback,
// which is weak, so that a program without offloading links without
// libomptarget. One server thread serves every device, so the callback never
// waits (docs/INPROGRESS.md, The device's side of the first slice).
//
//===----------------------------------------------------------------------===//

#include "rpc-opcodes.h"
#include "shared/rpc.h"
#include <cstdint>

extern "C" void merk_mailbox_send(std::int64_t nodeHigh, std::int64_t nodeLow,
    std::int64_t placeAndSlot, std::int64_t generation, std::int64_t secret,
    std::uint64_t typeInfo, std::uint64_t value);

extern "C" void __tgt_register_rpc_callback(unsigned (*)(void *, unsigned))
    __attribute__((weak));

namespace {

using namespace merkurius::runtime;

// A handle's cell of five words, as library/process.m has it, sent.
void SendTo(const std::uint64_t *handle, std::uint64_t typeInfo,
    std::uint64_t message) {
  merk_mailbox_send(static_cast<std::int64_t>(handle[0]),
      static_cast<std::int64_t>(handle[1]),
      static_cast<std::int64_t>(handle[2]),
      static_cast<std::int64_t>(handle[3]),
      static_cast<std::int64_t>(handle[4]), typeInfo, message);
}

unsigned HandleOpcode(void *raw, unsigned) {
  auto &port{*static_cast<rpc::Server::Port *>(raw)};
  switch (port.get_opcode()) {
  case rpcSend:
    // The answer tells the device that the message is queued.
    port.recv_and_send([](rpc::Buffer *buffer, std::uint32_t) {
      SendTo(buffer->data, buffer->data[5], buffer->data[6]);
    });
    return rpc::RPC_SUCCESS;
  default:
    return rpc::RPC_UNHANDLED_OPCODE;
  }
}

} // namespace

// A region that runs on the host sends as the device does.
extern "C" void merk_device_send(const std::uint64_t *handle,
    std::uint64_t typeInfo, std::uint64_t message) {
  SendTo(handle, typeInfo, message);
}

// At start-up, before any region runs.
extern "C" void merk_offload_start(void) {
  if (__tgt_register_rpc_callback) {
    __tgt_register_rpc_callback(HandleOpcode);
  }
}
