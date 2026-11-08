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

extern "C" std::int64_t merk_mailbox_poll(std::int64_t nodeHigh,
    std::int64_t nodeLow, std::int64_t placeAndSlot, std::int64_t generation,
    std::int64_t secret, std::int64_t position);
extern "C" std::uint64_t merk_mailbox_message(std::int64_t nodeHigh,
    std::int64_t nodeLow, std::int64_t placeAndSlot, std::int64_t generation,
    std::int64_t secret, std::int64_t position);
extern "C" void merk_mailbox_remove(std::int64_t nodeHigh,
    std::int64_t nodeLow, std::int64_t placeAndSlot, std::int64_t generation,
    std::int64_t secret, std::int64_t position);
extern "C" std::int64_t merk_mailbox_deadline(std::int64_t milliseconds);
extern "C" std::int64_t merk_mailbox_wait(std::int64_t nodeHigh,
    std::int64_t nodeLow, std::int64_t placeAndSlot, std::int64_t generation,
    std::int64_t secret, std::int64_t position, std::int64_t deadline);

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

// The mailbox's oldest message, removed, if it has one. Only the region
// that has the handle receives from the mailbox, as only the process that
// made a mailbox receives from it, so the message stays the oldest between
// the poll and the removal.
bool TakeOldest(const std::uint64_t *handle, std::uint64_t &message) {
  const std::int64_t *h{reinterpret_cast<const std::int64_t *>(handle)};
  if (merk_mailbox_poll(h[0], h[1], h[2], h[3], h[4], 0) <= 0) {
    return false;
  }
  message = merk_mailbox_message(h[0], h[1], h[2], h[3], h[4], 0);
  merk_mailbox_remove(h[0], h[1], h[2], h[3], h[4], 0);
  return true;
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
  case rpcPoll:
    // Never waits: the server thread serves every device.
    port.recv_and_send([](rpc::Buffer *buffer, std::uint32_t) {
      std::uint64_t message{0};
      bool found{TakeOldest(buffer->data, message)};
      buffer->data[0] = found;
      buffer->data[1] = message;
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

// A region that runs on the host receives as receive/5 does with any:
// it waits on the mailbox, then takes its oldest message.
extern "C" int merk_device_receive(const std::uint64_t *handle,
    std::int64_t milliseconds, std::uint64_t *message) {
  const std::int64_t *h{reinterpret_cast<const std::int64_t *>(handle)};
  if (!merk_mailbox_wait(h[0], h[1], h[2], h[3], h[4], 0,
          merk_mailbox_deadline(milliseconds))) {
    return 0;
  }
  return TakeOldest(handle, *message);
}
