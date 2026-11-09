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

#include "../device-ring.h"
#include "../rpc-opcodes.h"
#include "shared/rpc.h"

// The OpenMP device runtime's client, which the plugin points at the host's
// buffer when it loads the image.
extern "C" rpc::Client __llvm_rpc_client;

// The OpenMP device runtime's, by the device's steady clock, whose
// frequency the plugin gives it.
extern "C" double omp_get_wtime(void);
extern "C" int omp_get_device_num(void);

using merkurius::runtime::rpcChunk;
using merkurius::runtime::rpcGiveArray;
using merkurius::runtime::DeviceRing;
using merkurius::runtime::kRingCacheSlots;
using merkurius::runtime::kRingSlots;
using merkurius::runtime::rpcAttach;
using merkurius::runtime::rpcDrain;
using merkurius::runtime::rpcSend;
using merkurius::runtime::rpcWriteInt;
using merkurius::runtime::rpcWriteText;
using merkurius::runtime::rpcSendArray;
using merkurius::runtime::rpcTakeArray;

namespace {

// A chunk of the arena, which the host allocated on the device: its size
// in bytes, and how many of them are taken, this header's included. The
// count may pass the size, by the allocations that did not fit.
struct Chunk {
  uint64_t size;
  uint64_t used;
};

constexpr uint64_t kChunkBytes{uint64_t{1} << 20};

// The arena's current chunk, or none, which the host clears when it frees
// the chunks, once no region is running; and whether a wave is asking the
// host for a new chunk. One asks at a time: a chunk that lost a race to
// become the current one would be wasted but for one cell, and a GPU has
// thousands of waves.
//
// The arena also has the table of the rings this device has attached, by
// the mailbox's slot, each in the arena, which the host clears with it.
struct RingEntry {
  uint64_t slot, generation, secret;
  DeviceRing *ring;
  // The next message to take: only this device's receiver moves it.
  uint64_t head;
  // The next ring whose mailbox's slot falls in the same place of the table.
  RingEntry *next;
};

struct Arena {
  Chunk *current;
  RingEntry *rings[kRingCacheSlots];
  uint32_t refilling;
};
Arena arena{nullptr, {}, 0};

// A new chunk from the host, the current one from now on, of at least
// `bytes` after its header, of which the first `bytes` are taken.
void *NewChunk(uint64_t bytes) {
  uint64_t wanted{sizeof(Chunk) + bytes};
  Chunk *chunk{nullptr};
  uint64_t size{0};
  {
    rpc::Client::Port port{__llvm_rpc_client.open<rpcChunk>()};
    port.send([&](rpc::Buffer *buffer, uint32_t) {
      buffer->data[0] = static_cast<uint64_t>(omp_get_device_num());
      buffer->data[1] = reinterpret_cast<uint64_t>(&arena);
      buffer->data[2] = wanted > kChunkBytes ? wanted : kChunkBytes;
    });
    port.recv([&](rpc::Buffer *buffer, uint32_t) {
      chunk = reinterpret_cast<Chunk *>(buffer->data[0]);
      size = buffer->data[1];
    });
  }
  chunk->size = size;
  chunk->used = wanted;
  __atomic_store_n(&arena.current, chunk, __ATOMIC_RELEASE);
  return chunk + 1;
}

} // namespace

// Memory that the threads of a parallel region on the device can all reach,
// unlike a thread's stack. Freed in the reverse order.
extern "C" void *__kmpc_alloc_shared(uint64_t bytes);
extern "C" void __kmpc_free_shared(void *pointer, uint64_t bytes);

extern "C" void *merk_device_shared_alloc(uint64_t bytes) {
  return __kmpc_alloc_shared(bytes);
}

extern "C" void merk_device_shared_free(void *pointer, uint64_t bytes) {
  __kmpc_free_shared(pointer, bytes);
}

// A cell, or an array, in the device's arena (docs/INPROGRESS.md, The
// device's side of the first slice, step 6): bumped from the current
// chunk with an atomic add. When it does not fit, one lane on the device
// asks the host for a new chunk and takes its cell from it; the others
// sleep and try again in it.
extern "C" void *merk_device_alloc(uint64_t bytes) {
  bytes = (bytes + 7) & ~uint64_t{7};
  for (;;) {
    Chunk *chunk{__atomic_load_n(&arena.current, __ATOMIC_ACQUIRE)};
    if (chunk) {
      uint64_t at{__atomic_fetch_add(&chunk->used, bytes, __ATOMIC_RELAXED)};
      if (at + bytes <= chunk->size) {
        return reinterpret_cast<char *>(chunk) + at;
      }
    }
    uint32_t idle{0};
    if (rpc::is_first_lane(rpc::get_lane_mask()) &&
        __atomic_compare_exchange_n(&arena.refilling, &idle, 1, false,
            __ATOMIC_ACQUIRE, __ATOMIC_RELAXED)) {
      // Another wave may have made one since this one looked.
      void *cell{nullptr};
      if (__atomic_load_n(&arena.current, __ATOMIC_ACQUIRE) == chunk) {
        cell = NewChunk(bytes);
      }
      __atomic_store_n(&arena.refilling, 0, __ATOMIC_RELEASE);
      if (cell) {
        return cell;
      }
    } else {
      __builtin_amdgcn_s_sleep(127);
    }
  }
}

// What a region gives for an output array: the array, if it is the given
// array numbered `number`, which is mapped back; or else, for -1, a copy on
// the host of one the device made, since the arena does not outlive the
// region.
extern "C" uint64_t *merk_device_give_array(uint64_t *array, int64_t number) {
  if (number >= 0) {
    return array;
  }
  uint64_t *copy{nullptr};
  rpc::Client::Port port{__llvm_rpc_client.open<rpcGiveArray>()};
  port.send([&](rpc::Buffer *buffer, uint32_t) {
    buffer->data[0] = static_cast<uint64_t>(omp_get_device_num());
    buffer->data[1] = reinterpret_cast<uint64_t>(array);
    buffer->data[2] = array[0];
  });
  port.recv([&](rpc::Buffer *buffer, uint32_t) {
    copy = reinterpret_cast<uint64_t *>(buffer->data[0]);
  });
  return copy;
}

// Sends a one-word message to the mailbox whose handle is the cell of five
// words at `handle`. The host queues it, or sends it on to another node,
// encoding it by the type_info that a module registered under `key`, a
// hash of the message type's name. It returns once the host
// has, as a send on the host does: a send alone returns when the buffer is
// handed over, so a region could end before its messages were queued, and
// the next send could take another port, which the server may serve first.
extern "C" void merk_device_send(
    const uint64_t *handle, uint64_t key, uint64_t message) {
  rpc::Client::Port port{__llvm_rpc_client.open<rpcSend>()};
  port.send([&](rpc::Buffer *buffer, uint32_t) {
    for (int i{0}; i < 5; ++i) {
      buffer->data[i] = handle[i];
    }
    buffer->data[5] = key;
    buffer->data[6] = message;
  });
  port.recv([](rpc::Buffer *, uint32_t) {});
}

// Sends an array of one-word elements, as merk_device_send sends a word:
// the host copies the array from the device, given its address, and
// returns once it has queued it.
extern "C" void merk_device_send_array(
    const uint64_t *handle, uint64_t key, const uint64_t *array) {
  rpc::Client::Port port{__llvm_rpc_client.open<rpcSendArray>()};
  port.send([&](rpc::Buffer *buffer, uint32_t) {
    for (int i{0}; i < 5; ++i) {
      buffer->data[i] = handle[i];
    }
    buffer->data[5] = key;
    buffer->data[6] = reinterpret_cast<uint64_t>(array);
    buffer->data[7] =
        array[0] << 8 | static_cast<uint64_t>(omp_get_device_num());
  });
  port.recv([](rpc::Buffer *, uint32_t) {});
}

// Output from the device: the host prints it, and these return once it has,
// so that a region's output comes before what follows the region.
extern "C" void merk_device_write_string(const char *text) {
  uint64_t size{0};
  while (text[size]) {
    ++size;
  }
  rpc::Client::Port port{__llvm_rpc_client.open<rpcWriteText>()};
  port.send_n(text, size);
  port.recv([](rpc::Buffer *, uint32_t) {});
}

extern "C" void merk_device_write_int(int64_t value) {
  rpc::Client::Port port{__llvm_rpc_client.open<rpcWriteInt>()};
  port.send([&](rpc::Buffer *buffer, uint32_t) {
    buffer->data[0] = static_cast<uint64_t>(value);
  });
  port.recv([](rpc::Buffer *, uint32_t) {});
}

extern "C" void merk_device_nl(void) { merk_device_write_string("\n"); }

namespace {

// Calls `poll` until it finds a message or `milliseconds` pass, or forever
// if it is negative; whether it found one.
template <typename Poll> bool Wait(int64_t milliseconds, Poll poll) {
  double deadline{omp_get_wtime() + static_cast<double>(milliseconds) / 1000};
  for (;;) {
    if (poll()) {
      return true;
    }
    if (milliseconds >= 0 && omp_get_wtime() >= deadline) {
      return false;
    }
    // About 8 microseconds (127 * 64 clocks at about 1 GHz) between polls,
    // so that the device does not keep the host's server thread busy.
    __builtin_amdgcn_s_sleep(127);
  }
}

// The entry of the mailbox's ring, attaching it the first time, or null if
// the mailbox is closed.
RingEntry *RingOf(const uint64_t *handle, bool array) {
  RingEntry **bucket{&arena.rings[handle[2] % kRingCacheSlots]};
  for (RingEntry *entry{__atomic_load_n(bucket, __ATOMIC_ACQUIRE)}; entry;
       entry = entry->next) {
    if (entry->slot == handle[2] && entry->generation == handle[3] &&
        entry->secret == handle[4]) {
      return entry;
    }
  }
  DeviceRing *ring{nullptr};
  uint64_t head{0};
  {
    rpc::Client::Port port{__llvm_rpc_client.open<rpcAttach>()};
    port.send([&](rpc::Buffer *buffer, uint32_t) {
      for (int i{0}; i < 5; ++i) {
        buffer->data[i] = handle[i];
      }
      buffer->data[5] = static_cast<uint64_t>(omp_get_device_num());
      buffer->data[6] = array;
    });
    port.recv([&](rpc::Buffer *buffer, uint32_t) {
      ring = reinterpret_cast<DeviceRing *>(buffer->data[0]);
      head = buffer->data[1];
    });
  }
  if (!ring) {
    return nullptr;
  }
  // Only a mailbox's one receiver adds its entry, but other mailboxes' may
  // be added to the same place at once.
  auto *entry{static_cast<RingEntry *>(merk_device_alloc(sizeof(RingEntry)))};
  *entry = {handle[2], handle[3], handle[4], ring, head,
      __atomic_load_n(bucket, __ATOMIC_RELAXED)};
  while (!__atomic_compare_exchange_n(bucket, &entry->next, entry, true,
      __ATOMIC_RELEASE, __ATOMIC_RELAXED)) {
  }
  return entry;
}

// Asks the host for the messages that did not fit in the ring.
void Drain(const uint64_t *handle) {
  rpc::Client::Port port{__llvm_rpc_client.open<rpcDrain>()};
  port.send([&](rpc::Buffer *buffer, uint32_t) {
    for (int i{0}; i < 5; ++i) {
      buffer->data[i] = handle[i];
    }
  });
  port.recv([](rpc::Buffer *, uint32_t) {});
}

// One look at the ring, with no call of the host unless messages wait
// outside it: whether it has a message, whose slot is then in `value`, and
// the ring's entry. The message stays until Taken.
bool Look(const uint64_t *handle, bool array, RingEntry *&entry,
    uint64_t *value) {
  entry = RingOf(handle, array);
  if (!entry) {
    return false;
  }
  DeviceRing *ring{entry->ring};
  if (entry->head == __atomic_load_n(&ring->tail, __ATOMIC_ACQUIRE)) {
    // What waits outside the ring is in it after the drain, so that a
    // receive with no time to wait still finds it.
    if (!__atomic_load_n(&ring->spilled, __ATOMIC_ACQUIRE)) {
      return false;
    }
    Drain(handle);
    if (entry->head == __atomic_load_n(&ring->tail, __ATOMIC_ACQUIRE)) {
      return false;
    }
  }
  *value = __atomic_load_n(
      &ring->slots[entry->head % kRingSlots], __ATOMIC_RELAXED);
  return true;
}

void Taken(RingEntry *entry) {
  __atomic_store_n(&entry->ring->head, ++entry->head, __ATOMIC_RELEASE);
}

} // namespace

// Takes the oldest message of the mailbox whose handle is the cell of five
// words at `handle`, looking at its ring (device-ring.h) until it has one
// or `milliseconds` pass, or forever if it is negative. It returns 1 with
// the message in `message`, or 0.
extern "C" int merk_device_receive(
    const uint64_t *handle, int64_t milliseconds, uint64_t *message) {
  RingEntry *entry{nullptr};
  if (!Wait(milliseconds,
          [&]() { return Look(handle, false, entry, message); })) {
    return 0;
  }
  Taken(entry);
  return 1;
}

// As merk_device_receive, for a mailbox of arrays of one-word elements: the
// ring gives the oldest message's size, the device allocates the array in
// its arena, and the host copies the message there, given its address.
extern "C" int merk_device_receive_array(
    const uint64_t *handle, int64_t milliseconds, uint64_t *message) {
  RingEntry *entry{nullptr};
  uint64_t size{0};
  if (!Wait(milliseconds, [&]() { return Look(handle, true, entry, &size); })) {
    return 0;
  }
  void *array{merk_device_alloc(sizeof(uint64_t) * (1 + size))};
  {
    rpc::Client::Port port{__llvm_rpc_client.open<rpcTakeArray>()};
    port.send([&](rpc::Buffer *buffer, uint32_t) {
      for (int i{0}; i < 5; ++i) {
        buffer->data[i] = handle[i];
      }
      buffer->data[5] = static_cast<uint64_t>(omp_get_device_num());
      buffer->data[6] = reinterpret_cast<uint64_t>(array);
    });
    port.recv([](rpc::Buffer *, uint32_t) {});
  }
  *message = reinterpret_cast<uint64_t>(array);
  Taken(entry);
  return 1;
}
