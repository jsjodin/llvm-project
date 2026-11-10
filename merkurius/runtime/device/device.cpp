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
extern "C" int omp_get_team_num(void);

using merkurius::runtime::kTeamsPlace;
using merkurius::runtime::kTeamTableSlots;
using merkurius::runtime::rpcChunk;
using merkurius::runtime::rpcFail;
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

struct TeamTable;

struct Arena {
  Chunk *current;
  RingEntry *rings[kRingCacheSlots];
  TeamTable *teams[kTeamTableSlots];
  uint32_t refilling;
};
Arena arena{nullptr, {}, {}, 0};

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

namespace {

// Reports an error that ends the program, as the host reports the same:
// `kind` is 0 for an uncaught exception and 1 for a feature not yet
// implemented (rpc-opcodes.h). The host ends the program, so this does not
// return.
[[noreturn]] void Fail(uint64_t kind, const char *text) {
  uint64_t size{0};
  while (text[size]) {
    ++size;
  }
  rpc::Client::Port port{__llvm_rpc_client.open<rpcFail>()};
  port.send([&](rpc::Buffer *buffer, uint32_t) { buffer->data[0] = kind; });
  port.send_n(text, size + 1);
  port.recv([](rpc::Buffer *, uint32_t) {});
  __builtin_trap();
}

// Text for Fail, built without a C library: each Append adds to the end.
struct Text {
  char characters[160];
  uint64_t size{0};

  Text &Append(const char *text) {
    for (; *text && size + 1 < sizeof characters; ++text) {
      characters[size++] = *text;
    }
    characters[size] = '\0';
    return *this;
  }

  Text &Append(int64_t value) {
    char digits[24];
    int count{0};
    uint64_t magnitude{value < 0 ? 0 - static_cast<uint64_t>(value)
                                 : static_cast<uint64_t>(value)};
    do {
      digits[count++] = static_cast<char>('0' + magnitude % 10);
      magnitude /= 10;
    } while (magnitude != 0);
    if (value < 0) {
      digits[count++] = '-';
    }
    char reversed[24];
    for (int i{0}; i < count; ++i) {
      reversed[i] = digits[count - 1 - i];
    }
    reversed[count] = '\0';
    return Append(reversed);
  }
};

// Team mailboxes (docs/INPROGRESS.md, Processes as teams, step 2): in a
// target teams region, each team has a mailbox for each message type, made
// when first used, which any team of the region sends to and only its team
// receives from, with no call of the host. Each is a ring of slots in the
// arena, so it lasts as long as the region. A sender takes the next position
// with an atomic add; the slot's sequence says which position it holds, as
// in Vyukov's bounded queue: a slot of position p is free when its sequence
// is p, and holds the message when it is p + 1. The receiver takes the
// positions in order, and frees a slot by the position a lap on.
//
// A send does not wait, as a send on the host does not: a mailbox has no
// bound, so a team may send more to its own mailbox than the ring holds
// before it receives. A sender whose slot still holds the message of the lap
// before puts its message in the slot's list of spilled messages instead,
// with its position, where the receiver finds it when it reaches that
// position. Only spilled messages cost memory beyond the ring, until the
// region ends.
constexpr uint64_t kTeamRingSlots{256};

struct TeamSlot {
  uint64_t sequence;
  uint64_t message;
};

struct TeamSpill {
  uint64_t position;
  uint64_t message;
  TeamSpill *next;
};

// The positions that senders and the receiver move, in cache lines of their
// own, which the allocation aligns.
struct TeamRing {
  uint64_t key;
  TeamRing *next;
  uint64_t tail;
  uint64_t pad0[5];
  uint64_t head;
  uint64_t pad1[7];
  TeamSlot slots[kTeamRingSlots];
  // By slot: senders push onto a list, and only the receiver removes.
  TeamSpill *spills[kTeamRingSlots];
};

// A target teams region's table, which the first of its teams to start
// makes: by team, its rings.
struct TeamTable {
  uint64_t id;
  uint64_t nodeHigh, nodeLow;
  int64_t teams;
  TeamTable *next;
  TeamRing **rings;
};

// The table of the target teams region that this team runs, or none in any
// other region: in the team's local memory, so each team of each kernel has
// its own. A region's code sets it at its start (merk_device_teams_enter and
// merk_device_no_teams), since local memory starts undefined.
template <typename T> using Local = __attribute__((address_space(3))) T;
[[clang::loader_uninitialized]] Local<TeamTable *> currentTeams;

} // namespace

// At the start of each team of a target teams region: finds the region's
// table by its id, which the host chose, or makes it, and makes it the
// team's.
extern "C" void merk_device_teams_enter(
    uint64_t id, uint64_t nodeHigh, uint64_t nodeLow, int32_t teams) {
  TeamTable **bucket{&arena.teams[id % kTeamTableSlots]};
  TeamTable *made{nullptr};
  for (;;) {
    TeamTable *first{__atomic_load_n(bucket, __ATOMIC_ACQUIRE)};
    for (TeamTable *table{first}; table; table = table->next) {
      if (table->id == id) {
        currentTeams = table;
        return;
      }
    }
    if (!made) {
      // Another team may make one too, wasting this one if it is first.
      uint64_t count{static_cast<uint64_t>(teams)};
      auto *bytes{static_cast<char *>(
          merk_device_alloc(sizeof(TeamTable) + count * sizeof(TeamRing *)))};
      made = reinterpret_cast<TeamTable *>(bytes);
      made->id = id;
      made->nodeHigh = nodeHigh;
      made->nodeLow = nodeLow;
      made->teams = teams;
      made->rings = reinterpret_cast<TeamRing **>(bytes + sizeof(TeamTable));
      for (uint64_t i{0}; i < count; ++i) {
        made->rings[i] = nullptr;
      }
    }
    made->next = first;
    if (__atomic_compare_exchange_n(bucket, &first, made, false,
            __ATOMIC_RELEASE, __ATOMIC_RELAXED)) {
      currentTeams = made;
      return;
    }
  }
}

// At the end of each team of a target teams region: nothing to do on a
// device, whose next region sets the team's table again.
extern "C" void merk_device_teams_leave(void) {}

// At the start of any other region that a module with team mailboxes runs.
extern "C" void merk_device_no_teams(void) { currentTeams = nullptr; }

namespace {

// Team `team`'s ring for the message type whose key is `key`, made the
// first time, by whichever team asks first.
TeamRing *TeamRingOf(TeamTable *table, int64_t team, uint64_t key) {
  TeamRing **bucket{&table->rings[team]};
  TeamRing *made{nullptr};
  for (;;) {
    TeamRing *first{__atomic_load_n(bucket, __ATOMIC_ACQUIRE)};
    for (TeamRing *ring{first}; ring; ring = ring->next) {
      if (ring->key == key) {
        return ring;
      }
    }
    if (!made) {
      auto at{reinterpret_cast<uint64_t>(
          merk_device_alloc(sizeof(TeamRing) + 56))};
      made = reinterpret_cast<TeamRing *>((at + 63) & ~uint64_t{63});
      made->key = key;
      made->tail = 0;
      made->head = 0;
      for (uint64_t i{0}; i < kTeamRingSlots; ++i) {
        made->slots[i].sequence = i;
        made->spills[i] = nullptr;
      }
    }
    made->next = first;
    if (__atomic_compare_exchange_n(bucket, &first, made, false,
            __ATOMIC_RELEASE, __ATOMIC_RELAXED)) {
      return made;
    }
  }
}

bool IsTeamHandle(const uint64_t *handle) {
  return handle[2] >> 48 == kTeamsPlace;
}

int64_t TeamOf(const uint64_t *handle) {
  return static_cast<int64_t>(handle[2] & ((uint64_t{1} << 48) - 1));
}

// The table of the region that this team runs, which must be that of the
// team mailbox's handle; `what` names the operation for the error.
TeamTable *TableOf(const uint64_t *handle, const char *what) {
  TeamTable *table{currentTeams};
  if (!table || table->id != handle[3]) {
    Fail(0, Text{}
                .Append("Software Error: ")
                .Append(what)
                .Append(": the mailbox of team ")
                .Append(TeamOf(handle))
                .Append(" is not of this target teams region")
                .characters);
  }
  return table;
}

void TeamSend(const uint64_t *handle, uint64_t message) {
  TableOf(handle, "send");
  auto *ring{reinterpret_cast<TeamRing *>(handle[4])};
  uint64_t position{__atomic_fetch_add(&ring->tail, 1, __ATOMIC_RELAXED)};
  uint64_t index{position % kTeamRingSlots};
  TeamSlot &slot{ring->slots[index]};
  if (__atomic_load_n(&slot.sequence, __ATOMIC_ACQUIRE) == position) {
    slot.message = message;
    __atomic_store_n(&slot.sequence, position + 1, __ATOMIC_RELEASE);
    return;
  }
  auto *spill{static_cast<TeamSpill *>(merk_device_alloc(sizeof(TeamSpill)))};
  spill->position = position;
  spill->message = message;
  spill->next = __atomic_load_n(&ring->spills[index], __ATOMIC_RELAXED);
  while (!__atomic_compare_exchange_n(&ring->spills[index], &spill->next,
      spill, true, __ATOMIC_RELEASE, __ATOMIC_RELAXED)) {
  }
}

// Takes the spilled message of the position from the slot's list, if its
// sender has put it there. Senders only push onto the list, so a spill that
// is not the first stays where it is while the receiver unlinks it.
bool TakeSpill(TeamRing *ring, uint64_t index, uint64_t position,
    uint64_t *message) {
  TeamSpill **list{&ring->spills[index]};
  TeamSpill *first{__atomic_load_n(list, __ATOMIC_ACQUIRE)};
  TeamSpill *before{nullptr};
  TeamSpill *spill{first};
  while (spill && spill->position != position) {
    before = spill;
    spill = spill->next;
  }
  if (!spill) {
    return false;
  }
  *message = spill->message;
  if (before) {
    before->next = spill->next;
    return true;
  }
  if (__atomic_compare_exchange_n(list, &first, spill->next, false,
          __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
    return true;
  }
  // Senders pushed in front of it: it has a spill before it now.
  for (before = __atomic_load_n(list, __ATOMIC_ACQUIRE);
       before->next != spill; before = before->next) {
  }
  before->next = spill->next;
  return true;
}

// One look at a team mailbox's ring: whether the message of the next
// position has arrived, in its slot or spilled, which it takes.
bool TeamTake(TeamRing *ring, uint64_t *message) {
  uint64_t position{__atomic_load_n(&ring->head, __ATOMIC_RELAXED)};
  uint64_t index{position % kTeamRingSlots};
  TeamSlot &slot{ring->slots[index]};
  // The slot is free for this position or holds its message: the receiver
  // has taken the position a lap before.
  uint64_t sequence{__atomic_load_n(&slot.sequence, __ATOMIC_ACQUIRE)};
  if (sequence == position + 1) {
    *message = slot.message;
  } else if (!TakeSpill(ring, index, position, message)) {
    return false;
  }
  __atomic_store_n(&ring->head, position + 1, __ATOMIC_RELAXED);
  __atomic_store_n(
      &slot.sequence, position + kTeamRingSlots, __ATOMIC_RELEASE);
  return true;
}

// Takes the oldest message of a team mailbox, which must be this team's,
// waiting until `milliseconds` pass, or forever if it is negative.
int TeamReceive(const uint64_t *handle, int64_t milliseconds,
    uint64_t *message) {
  TableOf(handle, "receive");
  int64_t team{TeamOf(handle)};
  if (team != omp_get_team_num()) {
    Fail(0, Text{}
                .Append("Software Error: receive: team ")
                .Append(static_cast<int64_t>(omp_get_team_num()))
                .Append(" received from the mailbox of team ")
                .Append(team)
                .characters);
  }
  auto *ring{reinterpret_cast<TeamRing *>(handle[4])};
  if (TeamTake(ring, message)) {
    return 1;
  }
  double deadline{omp_get_wtime() + static_cast<double>(milliseconds) / 1000};
  for (;;) {
    if (milliseconds >= 0 && omp_get_wtime() >= deadline) {
      return 0;
    }
    // About 64 clocks between looks: no host thread serves this ring.
    __builtin_amdgcn_s_sleep(1);
    if (TeamTake(ring, message)) {
      return 1;
    }
  }
}

} // namespace

// Team `team`'s mailbox for the message type whose key is `key`, in the
// target teams region that this team runs: a handle of five words in the
// arena, the node's id, the teams' place and the team, the region's id, and
// the ring.
extern "C" uint64_t *merk_device_team_mailbox(int64_t team, uint64_t key) {
  TeamTable *table{currentTeams};
  if (!table) {
    Fail(1, "team_mailbox on a device outside a target teams region");
  }
  if (team < 0 || team >= table->teams) {
    Fail(0, Text{}
                .Append("Software Error: team_mailbox: the team number, ")
                .Append(team)
                .Append(", is not from 0 to ")
                .Append(table->teams - 1)
                .characters);
  }
  TeamRing *ring{TeamRingOf(table, team, key)};
  auto *handle{static_cast<uint64_t *>(merk_device_alloc(5 * sizeof(uint64_t)))};
  handle[0] = table->nodeHigh;
  handle[1] = table->nodeLow;
  handle[2] = kTeamsPlace << 48 | static_cast<uint64_t>(team);
  handle[3] = table->id;
  handle[4] = reinterpret_cast<uint64_t>(ring);
  return handle;
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
  if (IsTeamHandle(handle)) {
    TeamSend(handle, message);
    return;
  }
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
// returns once it has queued it. To a team mailbox, the array goes by its
// pointer (docs/INPROGRESS.md, Processes as teams, step 3): every team of
// the region can address it, in the arena or mapped for the region, and
// neither team can update it in place, since a send makes the sender's
// array shared and a receive gives a shared one.
extern "C" void merk_device_send_array(
    const uint64_t *handle, uint64_t key, const uint64_t *array) {
  if (IsTeamHandle(handle)) {
    TeamSend(handle, reinterpret_cast<uint64_t>(array));
    return;
  }
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
  if (IsTeamHandle(handle)) {
    return TeamReceive(handle, milliseconds, message);
  }
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
// its arena, and the host copies the message there, given its address. From
// a team mailbox, the sender's array itself (merk_device_send_array).
extern "C" int merk_device_receive_array(
    const uint64_t *handle, int64_t milliseconds, uint64_t *message) {
  if (IsTeamHandle(handle)) {
    return TeamReceive(handle, milliseconds, message);
  }
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
