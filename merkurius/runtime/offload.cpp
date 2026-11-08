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

#include "process.h"
#include "device-ring.h"
#include "fragment.h"
#include "rpc-opcodes.h"
#include "shared/rpc.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

extern "C" merkurius::runtime::DeviceRing *merk_mailbox_attach(
    std::int64_t nodeHigh, std::int64_t nodeLow, std::int64_t placeAndSlot,
    std::int64_t generation, std::int64_t secret, bool arrays, int device,
    merkurius::runtime::DeviceRing *(*allocate)(int), std::uint64_t *head);
extern "C" void merk_mailbox_drain(std::int64_t nodeHigh, std::int64_t nodeLow,
    std::int64_t placeAndSlot, std::int64_t generation, std::int64_t secret);
extern "C" void merk_mailbox_trim(std::int64_t nodeHigh, std::int64_t nodeLow,
    std::int64_t placeAndSlot, std::int64_t generation, std::int64_t secret);
extern "C" void merk_mailbox_detach_rings(void);

extern "C" void *llvm_omp_target_alloc_host(std::size_t, int)
    __attribute__((weak));
extern "C" void llvm_omp_target_free_host(void *, int) __attribute__((weak));
extern "C" void __tgt_register_rpc_callback(unsigned (*)(void *, unsigned))
    __attribute__((weak));
extern "C" void *omp_target_alloc(std::size_t, int) __attribute__((weak));
extern "C" void omp_target_free(void *, int) __attribute__((weak));
extern "C" int omp_target_memcpy(void *, const void *, std::size_t,
    std::size_t, std::size_t, int, int) __attribute__((weak));
extern "C" int omp_get_initial_device(void) __attribute__((weak));

namespace {

using namespace merkurius::runtime;

// Not exit: on the plugin's server thread, its handlers would wait for
// a kernel that is itself waiting for the server.
[[noreturn]] void Fail(const char *message, std::uint64_t key) {
  std::fflush(stdout);
  std::fprintf(stderr, "Mercury runtime: %s%016llx\n", message,
      static_cast<unsigned long long>(key));
  std::_Exit(1);
}

[[noreturn]] void Fail(const char *message) {
  std::fflush(stdout);
  std::fprintf(stderr, "Mercury runtime: %s\n", message);
  std::_Exit(1);
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

// The devices' arenas (docs/INPROGRESS.md, The device's side of the first
// slice, step 6): the chunks the host allocated on each device for its
// cells and arrays, which it frees when no region is running, and where
// the device keeps its current chunk, which it then clears. Regions on
// any device are counted together.
struct DeviceArena {
  int device;
  std::uint64_t address;
  void **chunks;
  std::uint64_t count, capacity;
};
pthread_mutex_t arenaLock = PTHREAD_MUTEX_INITIALIZER;
DeviceArena *arenas{nullptr};
std::uint64_t arenaCount{0}, arenaCapacity{0};
std::uint64_t runningRegions{0};

// The mailboxes' rings, to free when no region is running. Its lock is taken
// with the mailboxes' lock held, and so comes after it, and after arenaLock.
pthread_mutex_t ringsLock = PTHREAD_MUTEX_INITIALIZER;
struct RingOfDevice {
  DeviceRing *ring;
  int device;
};
RingOfDevice *rings{nullptr};
std::uint64_t ringCount{0}, ringCapacity{0};

// How many bytes of chunks the devices may hold at once, all told: a
// program that allocates without end would otherwise fill the device's
// memory, and on a GPU that also drives a display, starve its graphics
// until the driver resets the GPU. MERK_DEVICE_ARENA_BYTES replaces it.
std::uint64_t arenaBytes{0}, arenaLimit{std::uint64_t{8} << 30};

// Grows an array of `T` to hold one more than `count`.
template <typename T>
void Reserve(T *&items, std::uint64_t count, std::uint64_t &capacity) {
  if (count < capacity) {
    return;
  }
  capacity = capacity ? 2 * capacity : 8;
  items = static_cast<T *>(std::realloc(items, capacity * sizeof(T)));
  if (!items) {
    Fail("out of memory for the devices' arenas");
  }
}

// The device's arena, made when it first asks for a chunk.
DeviceArena &ArenaOf(int device, std::uint64_t address) {
  for (std::uint64_t i{0}; i < arenaCount; ++i) {
    if (arenas[i].device == device) {
      return arenas[i];
    }
  }
  Reserve(arenas, arenaCount, arenaCapacity);
  arenas[arenaCount] = {device, address, nullptr, 0, 0};
  return arenas[arenaCount++];
}

// A chunk of `bytes` on the device, for its arena at `address`.
void *NewChunk(int device, std::uint64_t address, std::uint64_t bytes) {
  pthread_mutex_lock(&arenaLock);
  arenaBytes += bytes;
  bool tooMany{arenaBytes > arenaLimit};
  pthread_mutex_unlock(&arenaLock);
  if (tooMany) {
    Fail("the devices' arenas are full: see MERK_DEVICE_ARENA_BYTES");
  }
  void *chunk{omp_target_alloc(bytes, device)};
  if (!chunk) {
    Fail("out of memory on a device, for its arena");
  }
  pthread_mutex_lock(&arenaLock);
  DeviceArena &arena{ArenaOf(device, address)};
  Reserve(arena.chunks, arena.count, arena.capacity);
  arena.chunks[arena.count++] = chunk;
  pthread_mutex_unlock(&arenaLock);
  return chunk;
}

// With arenaLock held and no region running: every chunk freed, and each
// device's current chunk cleared.
void FreeArenas() {
  arenaBytes = 0;
  for (std::uint64_t i{0}; i < arenaCount; ++i) {
    DeviceArena &arena{arenas[i]};
    if (arena.count == 0) {
      continue;
    }
    for (std::uint64_t j{0}; j < arena.count; ++j) {
      omp_target_free(arena.chunks[j], arena.device);
    }
    arena.count = 0;
    const std::uint64_t none[kArenaClearWords]{};
    if (omp_target_memcpy(reinterpret_cast<void *>(arena.address), none,
            sizeof none, 0, 0, arena.device, omp_get_initial_device()) != 0) {
      Fail("could not clear a device's arena");
    }
  }
}

std::size_t ArrayBytes(std::uint64_t size) {
  return sizeof(std::uint64_t) * (1 + size);
}

// Copies the device's array of `size` elements to `copy` on the host.
void CopyFromDevice(int device, const void *array, std::uint64_t size,
    std::uint64_t *copy) {
  if (omp_target_memcpy(copy, array, ArrayBytes(size), 0, 0,
          omp_get_initial_device(), device) != 0) {
    Fail("could not copy an array from a device");
  }
}

// A message that is the device's array: copied to the host for the send,
// which makes its own copy, or encodes it for another node.
void SendArray(const std::uint64_t *handle, std::uint64_t key, int device,
    const void *array, std::uint64_t size) {
  auto *copy{static_cast<std::uint64_t *>(Malloc(ArrayBytes(size)))};
  CopyFromDevice(device, array, size, copy);
  SendTo(handle, key, reinterpret_cast<std::uint64_t>(copy));
  std::free(copy);
}

// Copies the mailbox's oldest message that the device has not taken, an
// array, to the device's array of its size. The device then moves the
// ring's head, after which the message goes.
void TakeArray(const std::uint64_t *handle, int device, void *array) {
  const std::int64_t *h{reinterpret_cast<const std::int64_t *>(handle)};
  merk_mailbox_trim(h[0], h[1], h[2], h[3], h[4]);
  auto *message{reinterpret_cast<const std::uint64_t *>(
      merk_mailbox_message(h[0], h[1], h[2], h[3], h[4], 0))};
  if (omp_target_memcpy(array, message, ArrayBytes(message[0]), 0, 0, device,
          omp_get_initial_device()) != 0) {
    Fail("could not copy an array to a device");
  }
}

// A mailbox's ring, in memory that the device reads directly. It is freed
// when no region is running.
DeviceRing *NewRing(int device) {
  auto *ring{static_cast<DeviceRing *>(
      llvm_omp_target_alloc_host(sizeof(DeviceRing), device))};
  if (!ring) {
    Fail("out of memory for a mailbox's ring");
  }
  std::memset(static_cast<void *>(ring), 0, sizeof(DeviceRing));
  pthread_mutex_lock(&ringsLock);
  Reserve(rings, ringCount, ringCapacity);
  rings[ringCount++] = {ring, device};
  pthread_mutex_unlock(&ringsLock);
  return ring;
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
  case rpcAttach:
    port.recv_and_send([](rpc::Buffer *buffer, std::uint32_t) {
      const std::int64_t *h{reinterpret_cast<const std::int64_t *>(buffer->data)};
      std::uint64_t head{0};
      DeviceRing *ring{merk_mailbox_attach(h[0], h[1], h[2], h[3], h[4],
          buffer->data[6] != 0, static_cast<int>(buffer->data[5]), NewRing,
          &head)};
      buffer->data[0] = reinterpret_cast<std::uint64_t>(ring);
      buffer->data[1] = head;
    });
    return rpc::RPC_SUCCESS;
  case rpcDrain:
    port.recv_and_send([](rpc::Buffer *buffer, std::uint32_t) {
      const std::int64_t *h{reinterpret_cast<const std::int64_t *>(buffer->data)};
      merk_mailbox_drain(h[0], h[1], h[2], h[3], h[4]);
    });
    return rpc::RPC_SUCCESS;
  case rpcTakeArray:
    port.recv_and_send([](rpc::Buffer *buffer, std::uint32_t) {
      TakeArray(buffer->data, static_cast<int>(buffer->data[5]),
          reinterpret_cast<void *>(buffer->data[6]));
    });
    return rpc::RPC_SUCCESS;
  case rpcChunk:
    port.recv_and_send([](rpc::Buffer *buffer, std::uint32_t) {
      std::uint64_t bytes{buffer->data[2]};
      buffer->data[0] = reinterpret_cast<std::uint64_t>(NewChunk(
          static_cast<int>(buffer->data[0]), buffer->data[1], bytes));
      buffer->data[1] = bytes;
    });
    return rpc::RPC_SUCCESS;
  case rpcGiveArray:
    port.recv_and_send([](rpc::Buffer *buffer, std::uint32_t) {
      // In a fragment, as a message is.
      std::uint64_t size{buffer->data[2]};
      Fragment fragment;
      auto *copy{static_cast<std::uint64_t *>(
          fragment.Allocate(ArrayBytes(size)))};
      CopyFromDevice(static_cast<int>(buffer->data[0]),
          reinterpret_cast<const void *>(buffer->data[1]), size, copy);
      buffer->data[0] = reinterpret_cast<std::uint64_t>(copy);
    });
    return rpc::RPC_SUCCESS;
  case rpcSendArray:
    port.recv_and_send([](rpc::Buffer *buffer, std::uint32_t) {
      SendArray(buffer->data, buffer->data[5],
          static_cast<int>(buffer->data[7] & 0xff),
          reinterpret_cast<const void *>(buffer->data[6]),
          buffer->data[7] >> 8);
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

extern "C" void merk_device_send_array(const std::uint64_t *handle,
    std::uint64_t key, const std::uint64_t *array) {
  SendTo(handle, key, reinterpret_cast<std::uint64_t>(array));
}

// A region that runs on the host gives the array itself.
extern "C" std::uint64_t *merk_device_give_array(
    std::uint64_t *array, std::int64_t) {
  return array;
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

// Before and after each target region, on the host, wherever the region
// runs: the last region to end frees the devices' arenas. What a region
// gives back is scalars, arrays mapped from the host and copies on the
// host of arrays it made, so nothing in the arenas outlives it.
extern "C" void merk_region_enter(void) {
  merkurius::runtime::Pin();
  pthread_mutex_lock(&arenaLock);
  ++runningRegions;
  pthread_mutex_unlock(&arenaLock);
}

extern "C" void merk_region_exit(void) {
  merkurius::runtime::Unpin();
  pthread_mutex_lock(&arenaLock);
  if (--runningRegions == 0) {
    merk_mailbox_detach_rings();
    pthread_mutex_lock(&ringsLock);
    for (std::uint64_t i{0}; i < ringCount; ++i) {
      llvm_omp_target_free_host(rings[i].ring, rings[i].device);
    }
    ringCount = 0;
    pthread_mutex_unlock(&ringsLock);
    FreeArenas();
  }
  pthread_mutex_unlock(&arenaLock);
}

// At start-up, before any region runs.
extern "C" void merk_offload_start(void) {
  if (const char *limit{std::getenv("MERK_DEVICE_ARENA_BYTES")}) {
    arenaLimit = std::strtoull(limit, nullptr, 10);
  }
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

// A region that runs on the host receives an array as any other message.
extern "C" int merk_device_receive_array(const std::uint64_t *handle,
    std::int64_t milliseconds, std::uint64_t *message) {
  return merk_device_receive(handle, milliseconds, message);
}
