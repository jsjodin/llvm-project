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
#include <cstdio>
#include <cstdlib>
#include <pthread.h>

extern "C" std::int32_t merk_compare_type_infos(
    std::uint64_t x, std::uint64_t y);

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

[[noreturn]] void Fail(const char *message, std::uint64_t key) {
  std::fflush(stdout);
  std::fprintf(stderr, "Mercury runtime: %s%016llx\n", message,
      static_cast<unsigned long long>(key));
  std::exit(1);
}

// The message types that device code sends, by their keys, which the
// modules with target regions register at start-up: a table of open
// addressing, whose keys are hashes already, and 0 an empty entry.
struct MessageType {
  std::uint64_t key;
  std::uint64_t typeInfo;
};
pthread_mutex_t typesLock = PTHREAD_MUTEX_INITIALIZER;
MessageType *types{nullptr};
std::uint64_t typeCapacity{0}, typeCount{0};

MessageType &Slot(MessageType *table, std::uint64_t capacity,
    std::uint64_t key) {
  std::uint64_t i{key & (capacity - 1)};
  while (table[i].key != 0 && table[i].key != key) {
    i = (i + 1) & (capacity - 1);
  }
  return table[i];
}

void GrowTypes() {
  std::uint64_t capacity{typeCapacity ? 2 * typeCapacity : 64};
  auto *table{static_cast<MessageType *>(
      std::calloc(capacity, sizeof(MessageType)))};
  if (!table) {
    Fail("out of memory for message types: ", 0);
  }
  for (std::uint64_t i{0}; i < typeCapacity; ++i) {
    if (types[i].key != 0) {
      Slot(table, capacity, types[i].key) = types[i];
    }
  }
  std::free(types);
  types = table;
  typeCapacity = capacity;
}

// The host's type_info of the message type with the key.
std::uint64_t TypeInfoOfKey(std::uint64_t key) {
  pthread_mutex_lock(&typesLock);
  std::uint64_t typeInfo{
      typeCapacity ? Slot(types, typeCapacity, key).typeInfo : 0};
  pthread_mutex_unlock(&typesLock);
  if (!typeInfo) {
    Fail("a message was sent of a type that no module registered: ", key);
  }
  return typeInfo;
}

// A handle's cell of five words, as library/process.m has it, sent a
// message of the type with the key.
void SendTo(const std::uint64_t *handle, std::uint64_t key,
    std::uint64_t message) {
  merk_mailbox_send(static_cast<std::int64_t>(handle[0]),
      static_cast<std::int64_t>(handle[1]),
      static_cast<std::int64_t>(handle[2]),
      static_cast<std::int64_t>(handle[3]),
      static_cast<std::int64_t>(handle[4]), TypeInfoOfKey(key), message);
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
    std::uint64_t key, std::uint64_t message) {
  SendTo(handle, key, message);
}

// Registers a message type that device code may send, by its key, a hash
// of its name that code generation computes. Two modules may register the
// same type; two types with the same key stop the program.
extern "C" void merk_register_message_type(
    std::uint64_t key, std::uint64_t typeInfo) {
  if (key == 0) {
    Fail("a message type's key is the empty entry's: ", key);
  }
  pthread_mutex_lock(&typesLock);
  if (2 * (typeCount + 1) > typeCapacity) {
    GrowTypes();
  }
  MessageType &slot{Slot(types, typeCapacity, key)};
  if (slot.key == 0) {
    slot = {key, typeInfo};
    ++typeCount;
  } else if (merk_compare_type_infos(slot.typeInfo, typeInfo) != 0) {
    Fail("two message types have the same key: ", key);
  }
  pthread_mutex_unlock(&typesLock);
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
