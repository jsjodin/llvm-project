//===-- runtime/device/device.cpp - The device's runtime ------------------===//
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
// image beside the OpenMP device runtime, whose RPC client it uses, and into
// the image of a module's GPU process kinds, whose start object defines the
// client (process-start.cpp). The host's merkrt has the same functions for
// a region that runs on the host.
//
// It uses gpuintrin.h and LLVM's GPU libc rather than OpenMP's API, so that
// it works in a GPU process's kernel, which has no OpenMP device runtime, as
// in a region's (docs/INPROGRESS.md, GPU processes without omp target,
// step 2).
//
//===----------------------------------------------------------------------===//

#include "shared/rpc.h"
#include "../copy.h"
#include "../device-ring.h"
#include "../rpc-opcodes.h"
#include <gpuintrin.h>
#include <time.h>

// The OpenMP device runtime's client, which the plugin points at the host's
// buffer when it loads the image.
extern "C" rpc::Client __llvm_rpc_client;

// OpenMP's number of the device, in a region's image only.
extern "C" int omp_get_device_num(void) __attribute__((weak));

// The device's number, as the host's runtime knows it, or -1 if not set.
// A GPU process's image defines it in its start object, so that it is
// exported (process-start.cpp). Temporary (step 4c of GPU processes without
// omp target, docs/INPROGRESS.md): the runtime sets it when it loads such an
// image. Until then it is -1, as in a region's image, which takes OpenMP's
// number.
extern "C" [[gnu::visibility("protected"), gnu::weak]] int merk_device_number{
    -1};

namespace {

int DeviceNumber() {
  int number{__scoped_atomic_load_n(
      &merk_device_number, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE)};
  if (number >= 0) {
    return number;
  }
  return omp_get_device_num ? omp_get_device_num() : 0;
}

// Nanoseconds by libc's steady clock. libc takes its frequency to be
// 100 MHz unless the loader sets __llvm_libc_clock_freq, which the offload
// plugin does not: right from gfx9 on, and too slow by up to four times
// before.
int64_t Now() {
  timespec time;
  clock_gettime(CLOCK_MONOTONIC, &time);
  return static_cast<int64_t>(time.tv_sec) * 1000000000 + time.tv_nsec;
}

// The time `milliseconds` from now, or the end of time for a wait of more
// than about 146 years.
int64_t Deadline(int64_t milliseconds) {
  constexpr int64_t kLongest{INT64_MAX / 2 / 1000000};
  if (milliseconds > kLongest) {
    return INT64_MAX;
  }
  return Now() + milliseconds * 1000000;
}

} // namespace

using merkurius::runtime::DeviceRing;
using merkurius::runtime::kRingCacheSlots;
using merkurius::runtime::kRingOfArrays;
using merkurius::runtime::kRingOfTerms;
using merkurius::runtime::kRingOfWords;
using merkurius::runtime::kRingSlots;
using merkurius::runtime::kTeamsPlace;
using merkurius::runtime::kTeamTableSlots;
using merkurius::runtime::rpcAttach;
using merkurius::runtime::rpcChunk;
using merkurius::runtime::rpcDrain;
using merkurius::runtime::rpcFail;
using merkurius::runtime::rpcGiveArray;
using merkurius::runtime::rpcPinnedChunk;
using merkurius::runtime::rpcSend;
using merkurius::runtime::rpcSendArray;
using merkurius::runtime::rpcSendTerm;
using merkurius::runtime::rpcTakeArray;
using merkurius::runtime::rpcTakeTerm;

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
      buffer->data[0] = static_cast<uint64_t>(DeviceNumber());
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
// `kind` is 0 for an uncaught exception, 1 for a feature not yet
// implemented and 2 for a limit of the runtime (rpc-opcodes.h). The host ends the program, so this does not
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
  // The teams that have started (merk_device_teams_wait_all), and whether
  // one of them is reporting that the others did not.
  int64_t started;
  uint32_t reporting;
  // The barrier's (merk_device_team_barrier): the teams that have arrived
  // in this round, and the rounds that have ended.
  int64_t arrived;
  int64_t rounds;
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
      made->started = 0;
      made->reporting = 0;
      made->arrived = 0;
      made->rounds = 0;
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

// How long the teams of a region with team mailboxes wait for each other to
// start. Teams that fit on the device start within microseconds.
constexpr int64_t kTeamsStartMilliseconds{1000};

// After merk_device_teams_enter, in each team of a target teams region whose
// code uses team mailboxes: waits until every team of the region has started
// (docs/INPROGRESS.md, Processes as teams, step 4). A team that is not
// running deadlocks the teams that wait for its messages, and a GPU runs no
// more teams at once than its compute units hold, which depends on the
// kernel's registers and on what else the GPU runs. So when not all have
// started in time, one team reports how many have, and the program ends.
extern "C" void merk_device_teams_wait_all(void) {
  TeamTable *table{currentTeams};
  int64_t started{__atomic_add_fetch(&table->started, 1, __ATOMIC_ACQ_REL)};
  int64_t deadline{Deadline(kTeamsStartMilliseconds)};
  while (started < table->teams) {
    if (Now() >= deadline) {
      uint32_t none{0};
      if (__atomic_compare_exchange_n(&table->reporting, &none, 1, false,
              __ATOMIC_ACQ_REL, __ATOMIC_RELAXED)) {
        Fail(2, Text{}
                    .Append("target teams: only ")
                    .Append(started)
                    .Append(" of the ")
                    .Append(table->teams)
                    .Append(" teams ran at once on the device, and the teams "
                            "of a region with team mailboxes must all run at "
                            "once")
                    .characters);
      }
      // The host ends the program for the team that reports.
      for (;;) {
        __builtin_amdgcn_s_sleep(127);
      }
    }
    __builtin_amdgcn_s_sleep(127);
    started = __atomic_load_n(&table->started, __ATOMIC_ACQUIRE);
  }
}

// process.team_barrier on a device: the last team to arrive ends the round,
// and the others wait until it has, which costs about one atomic round trip
// a team. Only a region whose teams all run at once can wait like this, so
// one that calls it waits for its teams to start (merk_device_teams_wait_all,
// MarkRegionsForTeamMailboxes), and reports it if they do not. Outside a
// teams region there is one team.
extern "C" void merk_device_team_barrier(void) {
  TeamTable *table{currentTeams};
  if (!table || table->teams <= 1) {
    return;
  }
  // Read before arriving: the round cannot end before this team arrives.
  int64_t round{__atomic_load_n(&table->rounds, __ATOMIC_ACQUIRE)};
  if (__atomic_add_fetch(&table->arrived, 1, __ATOMIC_ACQ_REL) ==
      table->teams) {
    // No team arrives in the next round until it sees this one end.
    __atomic_store_n(&table->arrived, 0, __ATOMIC_RELAXED);
    __atomic_add_fetch(&table->rounds, 1, __ATOMIC_RELEASE);
    return;
  }
  while (__atomic_load_n(&table->rounds, __ATOMIC_ACQUIRE) == round) {
    __builtin_amdgcn_s_sleep(1);
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
  if (team != __gpu_block_id_x()) {
    Fail(0, Text{}
                .Append("Software Error: receive: team ")
                .Append(static_cast<int64_t>(__gpu_block_id_x()))
                .Append(" received from the mailbox of team ")
                .Append(team)
                .characters);
  }
  auto *ring{reinterpret_cast<TeamRing *>(handle[4])};
  if (TeamTake(ring, message)) {
    return 1;
  }
  int64_t deadline{Deadline(milliseconds)};
  for (;;) {
    if (milliseconds >= 0 && Now() >= deadline) {
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
    buffer->data[0] = static_cast<uint64_t>(DeviceNumber());
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
        array[0] << 8 | static_cast<uint64_t>(DeviceNumber());
  });
  port.recv([](rpc::Buffer *, uint32_t) {});
}

namespace {

// Calls `poll` until it finds a message or `milliseconds` pass, or forever
// if it is negative; whether it found one.
template <typename Poll> bool Wait(int64_t milliseconds, Poll poll) {
  int64_t deadline{Deadline(milliseconds)};
  for (;;) {
    if (poll()) {
      return true;
    }
    if (milliseconds >= 0 && Now() >= deadline) {
      return false;
    }
    // About 8 microseconds (127 * 64 clocks at about 1 GHz) between polls,
    // so that the device does not keep the host's server thread busy.
    __builtin_amdgcn_s_sleep(127);
  }
}

// The entry of the mailbox's ring, attaching it the first time, or null if
// the mailbox is closed.
RingEntry *RingOf(const uint64_t *handle, uint64_t kind) {
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
      buffer->data[5] = static_cast<uint64_t>(DeviceNumber());
      buffer->data[6] = kind;
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
bool Look(
    const uint64_t *handle, uint64_t kind, RingEntry *&entry, uint64_t *value) {
  entry = RingOf(handle, kind);
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
          [&]() { return Look(handle, kRingOfWords, entry, message); })) {
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
  if (!Wait(milliseconds,
          [&]() { return Look(handle, kRingOfArrays, entry, &size); })) {
    return 0;
  }
  void *array{merk_device_alloc(sizeof(uint64_t) * (1 + size))};
  bool found{false};
  {
    rpc::Client::Port port{__llvm_rpc_client.open<rpcTakeArray>()};
    port.send([&](rpc::Buffer *buffer, uint32_t) {
      for (int i{0}; i < 5; ++i) {
        buffer->data[i] = handle[i];
      }
      buffer->data[5] = static_cast<uint64_t>(DeviceNumber());
      buffer->data[6] = reinterpret_cast<uint64_t>(array);
    });
    port.recv(
        [&](rpc::Buffer *buffer, uint32_t) { found = buffer->data[0] != 0; });
  }
  // A mailbox that closed meanwhile has no message.
  if (!found) {
    return 0;
  }
  *message = reinterpret_cast<uint64_t>(array);
  Taken(entry);
  return 1;
}

namespace {

using merkurius::runtime::CopyTyped;
using merkurius::runtime::Resolve;

// Reports a value that a copy of a message cannot copy (copy.h).
[[noreturn]] void CopyFailed(const char *message, const char *detail) {
  Fail(2, Text{}.Append(message).Append(detail).characters);
}

// Where the device's copy of a message it receives goes: its arena.
struct ArenaMemory {
  static constexpr bool kOtherImage{true};
  void *Allocate(uint64_t bytes) { return merk_device_alloc(bytes); }
  [[noreturn]] void Fail(const char *message, const char *detail) {
    CopyFailed(message, detail);
  }
};

// Where the device's copy of a message it sends goes: host memory that the
// host reads, in chunks that the host gives (rpcPinnedChunk), linked by
// their first word. Only the sending thread uses them.
struct PinnedChunk {
  PinnedChunk *next;
  uint64_t device;
  uint64_t bytes;
};

class PinnedMemory {
public:
  static constexpr bool kOtherImage{true};
  PinnedChunk *First() const { return first_; }

  void *Allocate(uint64_t bytes) {
    bytes = (bytes + 7) & ~uint64_t{7};
    if (static_cast<uint64_t>(limit_ - next_) < bytes) {
      PinnedChunk *chunk{nullptr};
      {
        rpc::Client::Port port{__llvm_rpc_client.open<rpcPinnedChunk>()};
        port.send([&](rpc::Buffer *buffer, uint32_t) {
          buffer->data[0] = static_cast<uint64_t>(DeviceNumber());
          buffer->data[1] = bytes;
        });
        port.recv([&](rpc::Buffer *buffer, uint32_t) {
          chunk = reinterpret_cast<PinnedChunk *>(buffer->data[0]);
        });
      }
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

  [[noreturn]] void Fail(const char *message, const char *detail) {
    CopyFailed(message, detail);
  }

private:
  PinnedChunk *first_{nullptr};
  PinnedChunk *last_{nullptr};
  char *next_{nullptr};
  char *limit_{nullptr};
};

} // namespace

// Sends a message of any type that is not one word or an array of one-word
// elements (docs/INPROGRESS.md, GPU processes without omp target, step 4e):
// the device copies it by its type_info, `typeInfo`, into host memory, and
// the host sends that copy, as merk_device_send sends a word, and frees it.
// To a team mailbox, the message goes by its pointer, as an array does.
extern "C" void merk_device_send_term(const uint64_t *handle, uint64_t key,
    uint64_t message, const void *typeInfo) {
  if (IsTeamHandle(handle)) {
    TeamSend(handle, message);
    return;
  }
  PinnedMemory memory;
  uint64_t copy{CopyTyped(
      message, Resolve(reinterpret_cast<uint64_t>(typeInfo), nullptr), memory)};
  // The host reads the copy once it has the buffer.
  __scoped_atomic_thread_fence(__ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
  rpc::Client::Port port{__llvm_rpc_client.open<rpcSendTerm>()};
  port.send([&](rpc::Buffer *buffer, uint32_t) {
    for (int i{0}; i < 5; ++i) {
      buffer->data[i] = handle[i];
    }
    buffer->data[5] = key;
    buffer->data[6] = copy;
    buffer->data[7] = reinterpret_cast<uint64_t>(memory.First());
  });
  port.recv([](rpc::Buffer *, uint32_t) {});
}

// As merk_device_receive, for a mailbox of messages of any other type: the
// ring says that a message is there, the host copies it into host memory
// that the device reads (rpcTakeTerm), and the device copies that into its
// arena by the message type's type_info, `typeInfo`, before it moves the
// ring's head, after which the host frees its copy. A mailbox that closed
// meanwhile has no message. From a team mailbox, the sender's message
// itself.
extern "C" int merk_device_receive_term(const uint64_t *handle,
    int64_t milliseconds, const void *typeInfo, uint64_t *message) {
  if (IsTeamHandle(handle)) {
    return TeamReceive(handle, milliseconds, message);
  }
  RingEntry *entry{nullptr};
  uint64_t present{0};
  if (!Wait(milliseconds,
          [&]() { return Look(handle, kRingOfTerms, entry, &present); })) {
    return 0;
  }
  bool found{false};
  uint64_t copy{0};
  {
    rpc::Client::Port port{__llvm_rpc_client.open<rpcTakeTerm>()};
    port.send([&](rpc::Buffer *buffer, uint32_t) {
      for (int i{0}; i < 5; ++i) {
        buffer->data[i] = handle[i];
      }
    });
    port.recv([&](rpc::Buffer *buffer, uint32_t) {
      found = buffer->data[0] != 0;
      copy = buffer->data[1];
    });
  }
  if (!found) {
    return 0;
  }
  // The host wrote the copy before it answered.
  __scoped_atomic_thread_fence(__ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM);
  ArenaMemory memory;
  *message = CopyTyped(
      copy, Resolve(reinterpret_cast<uint64_t>(typeInfo), nullptr), memory);
  Taken(entry);
  return 1;
}
