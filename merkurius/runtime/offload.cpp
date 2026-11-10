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
// The device calls, and the callback's registration, are devices.h's. One
// server thread serves every device, so the callback never waits
// (docs/INPROGRESS.md, The device's side of the first slice).
//
//===----------------------------------------------------------------------===//

#include "copy.h"
#include "device-ring.h"
#include "devices.h"
#include "fragment.h"
#include "process.h"
#include "rpc-opcodes.h"
#include "shared/rpc.h"
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <pthread.h>
#include <unistd.h>

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
    std::int64_t generation, std::int64_t secret, std::uint64_t kind,
    int device, merkurius::runtime::DeviceRing *(*allocate)(int),
    std::uint64_t *head);
extern "C" bool merk_mailbox_copy_for_ring(std::int64_t nodeHigh,
    std::int64_t nodeLow, std::int64_t placeAndSlot, std::int64_t generation,
    std::int64_t secret,
    std::uint64_t (*copy)(
        int device, std::uint64_t typeInfo, std::uint64_t value, void **held),
    void (*release)(int device, void *held), std::uint64_t *result);
extern "C" bool merk_mailbox_copy_array_for_ring(std::int64_t nodeHigh,
    std::int64_t nodeLow, std::int64_t placeAndSlot, std::int64_t generation,
    std::int64_t secret,
    void (*copy)(int device, const std::uint64_t *array, void *to), void *to);
extern "C" void merk_mailbox_drain(std::int64_t nodeHigh, std::int64_t nodeLow,
    std::int64_t placeAndSlot, std::int64_t generation, std::int64_t secret);
extern "C" void merk_mailbox_detach_rings(void);
extern "C" [[noreturn]] void merk_uncaught_exception(const char *exception);


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
// slice, step 6): the chunks the host allocated for each arena of a device
// for its cells and arrays, how many bytes they have, and where the device
// keeps its current chunk. A region's image has an arena, whose chunks the
// host frees when no region is running, and whose current chunk it then
// clears. Regions on any device are counted together. Each GPU process has
// an arena of its own, which goes when the process ends.
struct DeviceArena {
  int device;
  std::uint64_t address;
  void **chunks;
  std::uint64_t count, capacity, bytes;
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

// The device's arena at the address, made when it first asks for a chunk.
DeviceArena &ArenaOf(int device, std::uint64_t address) {
  for (std::uint64_t i{0}; i < arenaCount; ++i) {
    if (arenas[i].device == device && arenas[i].address == address) {
      return arenas[i];
    }
  }
  Reserve(arenas, arenaCount, arenaCapacity);
  arenas[arenaCount] = {device, address, nullptr, 0, 0, 0};
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
  void *chunk{DeviceAllocate(device, bytes)};
  if (!chunk) {
    Fail("out of memory on a device, for its arena");
  }
  pthread_mutex_lock(&arenaLock);
  DeviceArena &arena{ArenaOf(device, address)};
  Reserve(arena.chunks, arena.count, arena.capacity);
  arena.chunks[arena.count++] = chunk;
  arena.bytes += bytes;
  pthread_mutex_unlock(&arenaLock);
  return chunk;
}

void FreeChunks(DeviceArena &arena) {
  for (std::uint64_t j{0}; j < arena.count; ++j) {
    DeviceFree(arena.device, arena.chunks[j]);
  }
  arena.count = 0;
  arenaBytes -= arena.bytes;
  arena.bytes = 0;
}

// With arenaLock held and no region or GPU process running, so that only
// regions' images have arenas: every chunk freed, and each device's current
// chunk cleared.
void FreeArenas() {
  for (std::uint64_t i{0}; i < arenaCount; ++i) {
    DeviceArena &arena{arenas[i]};
    if (arena.count == 0) {
      continue;
    }
    FreeChunks(arena);
    const std::uint64_t none[kArenaClearWords]{};
    if (!CopyToDevice(arena.device, reinterpret_cast<void *>(arena.address),
            none, sizeof none)) {
      Fail("could not clear a device's arena");
    }
  }
}

std::size_t ArrayBytes(std::uint64_t size) {
  return sizeof(std::uint64_t) * (1 + size);
}

// Copies the device's array of `size` elements to `copy` on the host.
void CopyArrayFromDevice(int device, const void *array, std::uint64_t size,
    std::uint64_t *copy) {
  if (!CopyFromDevice(device, copy, array, ArrayBytes(size))) {
    Fail("could not copy an array from a device");
  }
}

// A message that is the device's array: copied to the host for the send,
// which makes its own copy, or encodes it for another node.
void SendArray(const std::uint64_t *handle, std::uint64_t key, int device,
    const void *array, std::uint64_t size) {
  auto *copy{static_cast<std::uint64_t *>(Malloc(ArrayBytes(size)))};
  CopyArrayFromDevice(device, array, size, copy);
  SendTo(handle, key, reinterpret_cast<std::uint64_t>(copy));
  std::free(copy);
}

void CopyArrayToDevice(int device, const std::uint64_t *message, void *array) {
  if (!CopyToDevice(device, array, message, ArrayBytes(message[0]))) {
    Fail("could not copy an array to a device");
  }
}

// Copies the mailbox's oldest message that the device has not taken, an
// array, to the device's array of its size, with the mailbox's lock held,
// so that the message stays while it is copied; whether the mailbox was
// open. The device then moves the ring's head, after which the message
// goes.
bool TakeArray(const std::uint64_t *handle, void *array) {
  const std::int64_t *h{reinterpret_cast<const std::int64_t *>(handle)};
  return merk_mailbox_copy_array_for_ring(
      h[0], h[1], h[2], h[3], h[4], CopyArrayToDevice, array);
}

// Chunks of host memory that a device reads and writes directly, for a
// copy of a message (copy.h), as rpc-opcodes.h's rpcPinnedChunk describes
// them: the next chunk of the copy, the device's number and the chunk's
// size, and then its cells. The device writes a copy into chunks that it
// asks for, and the host one that it makes for the device.
struct PinnedChunk {
  PinnedChunk *next;
  std::uint64_t device;
  std::uint64_t bytes;
};

constexpr std::uint64_t kPinnedChunkBytes{4096};

// How many such chunks are allocated, for MERK_CHECK_PINNED.
std::atomic<std::int64_t> pinnedChunks{0};

PinnedChunk *NewPinnedChunk(int device, std::uint64_t bytes) {
  std::uint64_t size{sizeof(PinnedChunk) + bytes};
  size = size < kPinnedChunkBytes ? kPinnedChunkBytes : size;
  auto *chunk{static_cast<PinnedChunk *>(DeviceAllocateHost(device, size))};
  if (!chunk) {
    Fail("out of host memory for a device's message");
  }
  *chunk = {nullptr, static_cast<std::uint64_t>(device), size};
  pinnedChunks.fetch_add(1);
  return chunk;
}

void FreePinnedChunks(PinnedChunk *chunk) {
  while (chunk) {
    pinnedChunks.fetch_sub(1);
    PinnedChunk *next{chunk->next};
    DeviceFreeHost(static_cast<int>(chunk->device), chunk);
    chunk = next;
  }
}

// Where the host's copy of a message for a device goes (copy.h).
class PinnedMemory {
public:
  explicit PinnedMemory(int device) : device_{device} {}
  static constexpr bool kOtherImage{true};
  PinnedChunk *First() const { return first_; }

  void *Allocate(std::uint64_t bytes) {
    bytes = (bytes + 7) & ~std::uint64_t{7};
    if (static_cast<std::uint64_t>(limit_ - next_) < bytes) {
      PinnedChunk *chunk{NewPinnedChunk(device_, bytes)};
      if (last_) {
        last_->next = chunk;
      } else {
        first_ = chunk;
      }
      last_ = chunk;
      next_ = reinterpret_cast<char *>(chunk + 1);
      limit_ = reinterpret_cast<char *>(chunk) + chunk->bytes;
    }
    void *cell{next_};
    next_ += bytes;
    return cell;
  }

  // On the plugin's server thread, where exit would hang.
  [[noreturn]] void Fail(const char *message, const char *detail) {
    std::fflush(stdout);
    std::fprintf(stderr, "Mercury runtime: %s%s\n", message, detail);
    std::_Exit(1);
  }

private:
  int device_;
  PinnedChunk *first_{nullptr};
  PinnedChunk *last_{nullptr};
  char *next_{nullptr};
  char *limit_{nullptr};
};

// The mailbox's message, copied by its type_info into host memory that the
// device reads, for merk_mailbox_copy_for_ring.
std::uint64_t CopyForDevice(
    int device, std::uint64_t typeInfo, std::uint64_t value, void **held) {
  PinnedMemory memory{device};
  std::uint64_t copy{CopyTyped(value, Resolve(typeInfo, nullptr), memory)};
  *held = memory.First();
  return copy;
}

void ReleaseForDevice(int, void *held) {
  FreePinnedChunks(static_cast<PinnedChunk *>(held));
}

// A mailbox's ring, in memory that the device reads directly. It is freed
// when no region or GPU process is running, or when the GPU process whose
// mailbox it is ends (FreeRing).
DeviceRing *NewRing(int device) {
  auto *ring{static_cast<DeviceRing *>(
      DeviceAllocateHost(device, sizeof(DeviceRing)))};
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

// An error on a device, reported for its first lane that has one, as the
// host reports the same. A region is running, so the program ends at once
// (ExitProgram), with the kernel, and the device gets no answer.
[[noreturn]] void ReportDeviceFailure(rpc::Server::Port &port) {
  std::uint64_t kinds[64]{};
  std::uint64_t sizes[64]{};
  void *texts[64]{};
  port.recv([&](rpc::Buffer *buffer, std::uint32_t lane) {
    kinds[lane] = buffer->data[0];
  });
  port.recv_n(texts, sizes, [](std::uint64_t size) { return Malloc(size + 1); });
  for (int lane{0}; lane < 64; ++lane) {
    if (!texts[lane]) {
      continue;
    }
    auto *text{static_cast<char *>(texts[lane])};
    text[sizes[lane]] = '\0';
    if (kinds[lane] == 0) {
      merk_uncaught_exception(text);
    }
    if (kinds[lane] == 2) {
      Fail(text);
    }
    std::fflush(stdout);
    std::fprintf(stderr, "Not yet implemented: %s\n", text);
    ExitProgram(1);
  }
  Fail("a device reported an error with no text");
}

unsigned HandleOpcode(void *raw, unsigned) {
  auto &port{*static_cast<rpc::Server::Port *>(raw)};
  switch (port.get_opcode()) {
  case rpcFail:
    ReportDeviceFailure(port);
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
          buffer->data[6], static_cast<int>(buffer->data[5]), NewRing, &head)};
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
      buffer->data[0] =
          TakeArray(buffer->data, reinterpret_cast<void *>(buffer->data[6]));
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
      CopyArrayFromDevice(static_cast<int>(buffer->data[0]),
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
  case rpcPinnedChunk:
    port.recv_and_send([](rpc::Buffer *buffer, std::uint32_t) {
      buffer->data[0] = reinterpret_cast<std::uint64_t>(
          NewPinnedChunk(static_cast<int>(buffer->data[0]), buffer->data[1]));
    });
    return rpc::RPC_SUCCESS;
  case rpcSendTerm:
    // The copy is in the chunks, which the send copies again.
    port.recv_and_send([](rpc::Buffer *buffer, std::uint32_t) {
      SendTo(buffer->data, buffer->data[5], buffer->data[6]);
      FreePinnedChunks(reinterpret_cast<PinnedChunk *>(buffer->data[7]));
    });
    return rpc::RPC_SUCCESS;
  case rpcTakeTerm:
    port.recv_and_send([](rpc::Buffer *buffer, std::uint32_t) {
      const std::int64_t *h{
          reinterpret_cast<const std::int64_t *>(buffer->data)};
      std::uint64_t copy{0};
      bool found{merk_mailbox_copy_for_ring(h[0], h[1], h[2], h[3], h[4],
          CopyForDevice, ReleaseForDevice, &copy)};
      buffer->data[0] = found;
      buffer->data[1] = copy;
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

// A region that runs on the host sends a message of any other type as the
// device does, by its key.
extern "C" void merk_device_send_term(const std::uint64_t *handle,
    std::uint64_t key, std::uint64_t message, const void *) {
  SendTo(handle, key, message);
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
// While the program is ending, a region neither starts nor cleans up. Its
// thread waits for the end.
static void StopIfProgramEnding() {
  if (merkurius::runtime::ProgramEnding()) {
    for (;;) {
      pause();
    }
  }
}

namespace merkurius::runtime {
bool RegionsRunning() {
  pthread_mutex_lock(&arenaLock);
  bool running{runningRegions != 0};
  pthread_mutex_unlock(&arenaLock);
  return running;
}
} // namespace merkurius::runtime

// Checks that messages were freed, for tests. When the program ends, it
// waits up to ten seconds for the regions and GPU processes still running,
// so that what their mailboxes held has gone. With MERK_CHECK_PINNED set,
// it reports how many chunks of copies of messages between the host and
// the devices are still allocated (docs/INPROGRESS.md, GPU processes
// without omp target, step 4e); with MERK_CHECK_MESSAGES, how many messages
// that were queued neither a process read nor a mailbox freed, leaving out
// those still in open mailboxes, as their processes may still run; and with
// MERK_CHECK_DEVICE_MEMORY, how many chunks of the devices' arenas and how
// many mailboxes' rings are still allocated.
namespace merkurius::runtime {
static bool Set(const char *name) {
  const char *value{std::getenv(name)};
  return value && *value;
}

void CheckFreedAtEnd() {
  bool pinned{Set("MERK_CHECK_PINNED")}, messages{Set("MERK_CHECK_MESSAGES")};
  bool device{Set("MERK_CHECK_DEVICE_MEMORY")};
  if (!pinned && !messages && !device) {
    return;
  }
  for (int i{0}; i < 10000 && RegionsRunning(); ++i) {
    timespec millisecond{0, 1000000};
    nanosleep(&millisecond, nullptr);
  }
  std::fflush(stdout);
  if (pinned) {
    std::fprintf(stderr,
        "Mercury runtime: %lld chunks of host memory for devices' messages "
        "are allocated\n",
        static_cast<long long>(pinnedChunks.load()));
  }
  if (messages) {
    std::fprintf(stderr,
        "Mercury runtime: %lld messages are neither read nor freed\n",
        static_cast<long long>(HeldMessages()));
  }
  if (device) {
    pthread_mutex_lock(&arenaLock);
    std::uint64_t chunks{0};
    for (std::uint64_t i{0}; i < arenaCount; ++i) {
      chunks += arenas[i].count;
    }
    pthread_mutex_unlock(&arenaLock);
    pthread_mutex_lock(&ringsLock);
    std::uint64_t ringsLeft{ringCount};
    pthread_mutex_unlock(&ringsLock);
    std::fprintf(stderr,
        "Mercury runtime: %llu chunks of the devices' arenas and %llu rings "
        "are allocated\n",
        static_cast<unsigned long long>(chunks),
        static_cast<unsigned long long>(ringsLeft));
  }
}
} // namespace merkurius::runtime

// With arenaLock held, once no region or GPU process runs: the rings, the
// copies their devices may have been reading, and the regions' arenas go.
static void FreeWhenIdle() {
  merk_mailbox_detach_rings();
  pthread_mutex_lock(&ringsLock);
  for (std::uint64_t i{0}; i < ringCount; ++i) {
    DeviceFreeHost(rings[i].device, rings[i].ring);
  }
  ringCount = 0;
  pthread_mutex_unlock(&ringsLock);
  FreeArenas();
}

// A process that runs on a GPU (gpu-processes.cpp) counts as a running
// region while its kernel runs, so that a ring that it may read stays, and
// the program ends at once if main ends first. What is its own, its arena
// and its mailbox's ring, goes when it ends, before it stops counting, on a
// thread of the runtime's: its kernel's end runs on liboffload's thread,
// which must not call liboffload.
namespace merkurius::runtime {
void GpuProcessStarted() {
  pthread_mutex_lock(&arenaLock);
  ++runningRegions;
  pthread_mutex_unlock(&arenaLock);
}

void GpuProcessEnded() {
  pthread_mutex_lock(&arenaLock);
  // As the program ends, its end frees everything.
  if (--runningRegions == 0 && !ProgramEnding()) {
    FreeWhenIdle();
  }
  pthread_mutex_unlock(&arenaLock);
}

// A GPU process's arena: the words on the device where it keeps its current
// chunk and its tables, cleared, or null.
void *NewProcessArena(int device) {
  constexpr std::size_t bytes{sizeof(std::uint64_t) * (kArenaClearWords + 1)};
  void *arena{DeviceAllocate(device, bytes)};
  const std::uint64_t none[kArenaClearWords + 1]{};
  if (arena && !CopyToDevice(device, arena, none, bytes)) {
    DeviceFree(device, arena);
    arena = nullptr;
  }
  return arena;
}

// Frees a GPU process's arena, and the chunks it took, once its kernel has
// ended.
void FreeProcessArena(int device, void *address) {
  pthread_mutex_lock(&arenaLock);
  for (std::uint64_t i{0}; i < arenaCount; ++i) {
    DeviceArena &arena{arenas[i]};
    if (arena.device == device &&
        arena.address == reinterpret_cast<std::uint64_t>(address)) {
      FreeChunks(arena);
      std::free(arena.chunks);
      arena = arenas[--arenaCount];
      break;
    }
  }
  pthread_mutex_unlock(&arenaLock);
  DeviceFree(device, address);
}

// Frees the ring of a GPU process's mailbox once its kernel has ended.
void FreeRing(int device, DeviceRing *ring) {
  pthread_mutex_lock(&ringsLock);
  for (std::uint64_t i{0}; i < ringCount; ++i) {
    if (rings[i].ring == ring) {
      rings[i] = rings[--ringCount];
      break;
    }
  }
  pthread_mutex_unlock(&ringsLock);
  DeviceFreeHost(device, ring);
}
} // namespace merkurius::runtime

extern "C" void merk_region_enter(void) {
  merkurius::runtime::Pin();
  pthread_mutex_lock(&arenaLock);
  ++runningRegions;
  pthread_mutex_unlock(&arenaLock);
  // After the count, so that ExitProgram sees the region or the region sees
  // that the program is ending.
  StopIfProgramEnding();
}

extern "C" void merk_region_exit(void) {
  StopIfProgramEnding();
  pthread_mutex_lock(&arenaLock);
  if (--runningRegions == 0) {
    FreeWhenIdle();
  }
  pthread_mutex_unlock(&arenaLock);
  merkurius::runtime::Unpin();
}

// A target teams region's num_teams, before it starts: OpenMP's is 32 bits,
// and libomptarget takes one that is not positive for its default.
extern "C" void merk_check_teams(std::int64_t teams) {
  if (teams >= 1 && teams <= INT32_MAX) {
    return;
  }
  char text[128];
  std::snprintf(text, sizeof text,
      "Software Error: target teams: the number of teams, %lld, is not from 1 "
      "to %d",
      static_cast<long long>(teams), INT32_MAX);
  merk_uncaught_exception(text);
}

// At start-up, before any region runs.
extern "C" void merk_offload_start(void) {
  if (const char *limit{std::getenv("MERK_DEVICE_ARENA_BYTES")}) {
    arenaLimit = std::strtoull(limit, nullptr, 10);
  }
  RegisterRpcCallback(HandleOpcode);
}

// A region that runs on the host receives as receive/5 does with any:
// it waits on the mailbox, then takes its oldest message.
extern "C" int merk_device_receive(const std::uint64_t *handle,
    std::int64_t milliseconds, std::uint64_t *message) {
  const std::int64_t *h{reinterpret_cast<const std::int64_t *>(handle)};
  if (merk_mailbox_wait(h[0], h[1], h[2], h[3], h[4], 0,
          merk_mailbox_deadline(milliseconds)) <= 0) {
    return 0;
  }
  return TakeOldest(handle, *message);
}

// A region that runs on the host receives a message of any other type as
// it receives a word.
extern "C" int merk_device_receive_term(const std::uint64_t *handle,
    std::int64_t milliseconds, const void *, std::uint64_t *message) {
  return merk_device_receive(handle, milliseconds, message);
}

// A region that runs on the host receives an array as any other message.
extern "C" int merk_device_receive_array(const std::uint64_t *handle,
    std::int64_t milliseconds, std::uint64_t *message) {
  return merk_device_receive(handle, milliseconds, message);
}
