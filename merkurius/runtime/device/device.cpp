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
#include "../place.h"
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

// The stack, a lane, that the image's kernel was launched with, which its
// recursion checks (process-start.cpp, which defines it for a GPU process's
// image). In a region's image nothing sets it, and there is no limit.
extern "C" [[gnu::visibility("protected"),
    gnu::weak]] uint32_t merk_device_stack_bytes{0xffffffffu};

// The stack pointer, which is an offset in the lane's scratch.
extern "C" __attribute__((address_space(5))) void *merk_device_stack_save(
    void) __asm("llvm.stacksave.p5");

extern "C" [[noreturn]] void merk_device_stack_overflow(void);

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

using merkurius::runtime::ArrayWords;
using merkurius::runtime::ArrayWordsOfHeader;
using merkurius::runtime::DeviceRing;
using merkurius::runtime::kArenaClearWords;
using merkurius::runtime::kBumpWords;
using merkurius::runtime::kChunkBytes;
using merkurius::runtime::kNoRegionSlot;
using merkurius::runtime::kPlaceIdle;
using merkurius::runtime::kPlaceReserved;
using merkurius::runtime::kPlaceRunning;
using merkurius::runtime::kPlaceTeamThreads;
using merkurius::runtime::kPlaceWorking;
using merkurius::runtime::kProcessLoopReductions;
using merkurius::runtime::kProcessLoopWords;
using merkurius::runtime::kRegionRingSlots;
using merkurius::runtime::kRegionSlotClearing;
using merkurius::runtime::kRegionSlots;
using merkurius::runtime::kRegionSlotWords;
using merkurius::runtime::kRingCacheSlots;
using merkurius::runtime::kRingOfArrays;
using merkurius::runtime::kRingOfTerms;
using merkurius::runtime::kRingOfWords;
using merkurius::runtime::kRingSlots;
using merkurius::runtime::kTeamsPlace;
using merkurius::runtime::PlaceSlot;
using merkurius::runtime::PlaceSlots;
using merkurius::runtime::PlaceTable;
using merkurius::runtime::RegionSlotGeneration;
using merkurius::runtime::RegionSlotNumber;
using merkurius::runtime::rpcAttach;
using merkurius::runtime::rpcChunk;
using merkurius::runtime::rpcDrain;
using merkurius::runtime::rpcFail;
using merkurius::runtime::rpcGiveArray;
using merkurius::runtime::rpcPinnedChunk;
using merkurius::runtime::rpcProcessEnded;
using merkurius::runtime::rpcSend;
using merkurius::runtime::rpcSendTerm;
using merkurius::runtime::rpcTakeArray;
using merkurius::runtime::rpcTakeTerm;
using merkurius::runtime::rpcTeamsStall;

namespace {

// A chunk of the arena, which the host allocated on the device: its size
// in bytes, and how many of them are taken, this header's included. The
// count may pass the size, by the allocations that did not fit.
struct Chunk {
  uint64_t size;
  uint64_t used;
};

// Where cells are bumped from: the current chunk, or none, which the host
// clears when it frees the chunks; and whether a wave is asking the host for
// a new chunk. One asks at a time: a chunk that lost a race to become the
// current one would be wasted but for one cell, and a GPU has thousands of
// waves.
struct Bump {
  Chunk *current;
  uint32_t refilling;
  uint32_t unused;
};
static_assert(sizeof(Bump) == kBumpWords * sizeof(uint64_t));

// An arena: its chunks, freed when no region is running, or for a GPU
// process's when it ends. It has the table of the rings this device has
// attached, by the mailbox's slot, each in the arena, which the host clears
// with it.
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
  Bump bump;
  RingEntry *rings[kRingCacheSlots];
};
static_assert(sizeof(Arena) == kArenaClearWords * sizeof(uint64_t));
Arena arena{{nullptr, 0, 0}, {}};

// In a region's image, by the slot of each region that runs (device-ring.h):
// where its cells are bumped from, its teams' table if it is a target teams
// region, which its first team to start makes, and the table of the rings
// it has attached, by the mailbox's slot; and the generation of the region
// whose they are (device-ring.h). The host takes back a region's chunks when
// it ends, and frees a ring once no running region or GPU process has
// attached it; the next region of the slot clears its words as it starts. A
// GPU process keeps its table of rings in its arena.
struct RegionSlot {
  Bump bump;
  TeamTable *teams;
  uint64_t generation;
  RingEntry *rings[kRegionRingSlots];
};
static_assert(sizeof(RegionSlot) == kRegionSlotWords * sizeof(uint64_t));
RegionSlot regionSlots[kRegionSlots];

// The slot of the region that the team runs, which merk_device_region_enter
// sets at the start of each of its teams, since local memory starts
// undefined.
template <typename T> using Local = __attribute__((address_space(3))) T;
[[clang::loader_uninitialized]] Local<RegionSlot *> regionSlot;

// In the kernel of a GPU's processes (place.h), the arena of the process
// that each wave of the team runs, by the wave's place in the team, which
// the wave sets as the process starts.
constexpr uint32_t kMostPlaceWaves{kPlaceTeamThreads / 32};
// The most sleeps of an idle wave of merk_place between looks at its slot.
constexpr uint32_t kMostPlaceSleeps{64};
[[clang::loader_uninitialized]] Local<Arena *> placeArenas[kMostPlaceWaves];

// A process's control block, in its arena, through which it runs its loops
// across its worker teams (place.h): the counts of the barrier that the
// process's wave and its worker teams cross together, how many cross it, the
// loop that runs, its first counter, its step and its trip count, whether
// it is a distribute loop, which the teams' first waves run, and the words
// its body takes; whether the process has ended; and the process's arena,
// which the workers bump the cells of the loops' bodies from.
struct ProcessControl {
  int64_t arrived;
  int64_t rounds;
  int64_t crossing;
  uint64_t loop;
  int64_t first;
  int64_t step;
  int64_t count;
  uint64_t distribute;
  uint64_t end;
  Arena *arena;
  uint64_t values[kProcessLoopWords];
  // The loop's reductions: how many, their values before the loop, which the
  // process sets and which are their results after it, how many worker
  // teams have stored their partial results, and each team's partial
  // results, kProcessLoopReductions words a team, which follow the block.
  uint64_t reductions;
  uint64_t stored;
  uint64_t results[kProcessLoopReductions];
  uint64_t partials[];
};

// The control block of the process that each wave of the team runs, by the
// wave's place in the team, or null for one without worker teams, and for a
// worker team's waves.
[[clang::loader_uninitialized]] Local<ProcessControl *>
    placeControls[kMostPlaceWaves];

// The partial results of a loop's reductions of each wave of a worker team,
// which the team's first thread combines.
[[clang::loader_uninitialized]] Local<uint64_t>
    wavePartials[kMostPlaceWaves][kProcessLoopReductions];

// Whether the worker team is the last to store its partial results of a
// loop's reductions, which then combines all the teams'.
[[clang::loader_uninitialized]] Local<uint32_t> lastToStore;

// In a worker team, the loop under parallel do that the team's first wave,
// running a distribute loop's body, has the team's threads run
// (merk_device_team_loop), or that the team's share of the distribute loop
// has ended; with the words its body takes, and its reductions' values
// before it, which are their results after it.
struct TeamLoop {
  uint64_t loop;
  int64_t first;
  int64_t step;
  int64_t count;
  uint64_t reductions;
  uint64_t end;
};
[[clang::loader_uninitialized]] Local<TeamLoop> teamLoop;
[[clang::loader_uninitialized]] Local<uint64_t> teamValues[kProcessLoopWords];
[[clang::loader_uninitialized]] Local<uint64_t>
    teamResults[kProcessLoopReductions];

// By the wave's place in the team, whether it runs a distribute loop's body,
// as the first wave of a worker team does, all its lanes the same: they
// make one cell between them (merk_device_alloc), and may run a loop across
// the team's threads. Each wave sets its own when it starts a process or
// works for one.
[[clang::loader_uninitialized]] Local<uint32_t> teamMains[kMostPlaceWaves];

} // namespace

// Whether this is a GPU process's image, whose start object defines it as
// 1 (process-start.cpp). In a region's image it stays 0, and the compiler
// folds what depends on it.
extern "C"
    [[gnu::visibility("hidden"), gnu::weak]] int merk_device_process_image{0};

namespace {

// The arena of what runs, for its tables: a region's image has one, its
// global, and each GPU process one of its own, which the host allocates and
// frees when the process ends (gpu-processes.cpp). The wave that runs the
// process keeps it in local memory, so it is found from anywhere in the
// process.
Arena *CurrentArena() {
  if (!merk_device_process_image) {
    return &arena;
  }
  return placeArenas[__gpu_thread_id_x() / __gpu_num_lanes()];
}

// Tells the host that the process of the slot has ended (rpcProcessEnded).
void ProcessEnded(PlaceTable *table, uint64_t slot) {
  rpc::Client::Port port{__llvm_rpc_client.open<rpcProcessEnded>()};
  port.send([&](rpc::Buffer *buffer, uint32_t) {
    buffer->data[0] = static_cast<uint64_t>(DeviceNumber());
    buffer->data[1] = reinterpret_cast<uint64_t>(table);
    buffer->data[2] = slot;
  });
  port.recv([](rpc::Buffer *, uint32_t) {});
}

// Where the cells of what runs are bumped from: a GPU process's arena, or
// the slot of the region.
Bump *CurrentBump() {
  if (merk_device_process_image) {
    return &CurrentArena()->bump;
  }
  return &regionSlot->bump;
}

// A new chunk from the host, the current one from now on, of at least
// `bytes` after its header, of which the first `bytes` are taken. The host
// frees a region's chunks when it ends, and others when no region runs.
void *NewChunk(Bump *bump, uint64_t bytes) {
  uint64_t wanted{sizeof(Chunk) + bytes};
  Chunk *chunk{nullptr};
  uint64_t size{0};
  auto *region{reinterpret_cast<RegionSlot *>(bump)};
  uint64_t slot{region >= regionSlots && region < regionSlots + kRegionSlots
          ? static_cast<uint64_t>(region - regionSlots)
          : kNoRegionSlot};
  {
    rpc::Client::Port port{__llvm_rpc_client.open<rpcChunk>()};
    port.send([&](rpc::Buffer *buffer, uint32_t) {
      buffer->data[0] = static_cast<uint64_t>(DeviceNumber());
      buffer->data[1] = reinterpret_cast<uint64_t>(bump);
      buffer->data[2] = wanted > kChunkBytes ? wanted : kChunkBytes;
      buffer->data[3] = slot;
    });
    port.recv([&](rpc::Buffer *buffer, uint32_t) {
      chunk = reinterpret_cast<Chunk *>(buffer->data[0]);
      size = buffer->data[1];
    });
  }
  chunk->size = size;
  chunk->used = wanted;
  __atomic_store_n(&bump->current, chunk, __ATOMIC_RELEASE);
  return chunk + 1;
}

// `bytes` bumped from the chunks of `bump`. When they do not fit, one lane
// on the device asks the host for a new chunk and takes its cell from it;
// the others sleep and try again in it.
void *Allocate(Bump *bump, uint64_t bytes) {
  bytes = (bytes + 7) & ~uint64_t{7};
  for (;;) {
    Chunk *chunk{__atomic_load_n(&bump->current, __ATOMIC_ACQUIRE)};
    if (chunk) {
      uint64_t at{__atomic_fetch_add(&chunk->used, bytes, __ATOMIC_RELAXED)};
      if (at + bytes <= chunk->size) {
        return reinterpret_cast<char *>(chunk) + at;
      }
    }
    uint32_t idle{0};
    if (rpc::is_first_lane(rpc::get_lane_mask()) &&
        __atomic_compare_exchange_n(&bump->refilling, &idle, 1, false,
            __ATOMIC_ACQUIRE, __ATOMIC_RELAXED)) {
      // Another wave may have made one since this one looked.
      void *cell{nullptr};
      if (__atomic_load_n(&bump->current, __ATOMIC_ACQUIRE) == chunk) {
        cell = NewChunk(bump, bytes);
      }
      __atomic_store_n(&bump->refilling, 0, __ATOMIC_RELEASE);
      if (cell) {
        return cell;
      }
    } else {
      __builtin_amdgcn_s_sleep(127);
    }
  }
}

// The region's slot, cleared if its words are an earlier region's, by the
// first wave of the region to find so; the first lane of each wave asks, so
// that no lane waits for another of its own wave. A wave that finds another
// clearing it waits for that one.
void ClaimRegionSlot(RegionSlot *slot, uint64_t generation) {
  for (;;) {
    uint64_t seen{__atomic_load_n(&slot->generation, __ATOMIC_ACQUIRE)};
    if (seen == generation) {
      return;
    }
    if (seen != kRegionSlotClearing &&
        __atomic_compare_exchange_n(&slot->generation, &seen,
            kRegionSlotClearing, false, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED)) {
      slot->bump.current = nullptr;
      slot->bump.refilling = 0;
      slot->bump.unused = 0;
      slot->teams = nullptr;
      for (uint64_t i{0}; i < kRegionRingSlots; ++i) {
        slot->rings[i] = nullptr;
      }
      __atomic_store_n(&slot->generation, generation, __ATOMIC_RELEASE);
      return;
    }
    __builtin_amdgcn_s_sleep(1);
  }
}

} // namespace

// At the start of each team of a target region, before it makes a cell: the
// slot that the host gave the region, and its generation (merk_region_enter).
extern "C" void merk_device_region_enter(uint64_t word) {
  RegionSlot *slot{&regionSlots[RegionSlotNumber(word)]};
  if (rpc::is_first_lane(rpc::get_lane_mask())) {
    ClaimRegionSlot(slot, RegionSlotGeneration(word));
  }
  regionSlot = slot;
}

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

// A cell, or an array, on the device (docs/INPROGRESS.md, The device's side
// of the first slice, step 6): bumped from the chunks of the region that
// runs, or of the GPU process, with an atomic add. The lanes of a team's
// first wave that run a distribute loop's body all run the same code with
// the same values, so they make one cell between them, which their first
// active lane bumps: each its own would leave the lanes' arrays apart, and
// a loop across the team's threads would update only one of them.
extern "C" void *merk_device_alloc(uint64_t bytes) {
  if (merk_device_process_image &&
      teamMains[__gpu_thread_id_x() / __gpu_num_lanes()]) {
    uint64_t mask{__gpu_lane_mask()};
    uint64_t cell{0};
    if (__gpu_is_first_in_lane(mask)) {
      cell = reinterpret_cast<uint64_t>(Allocate(CurrentBump(), bytes));
    }
    return reinterpret_cast<void *>(__gpu_read_first_lane_u64(mask, cell));
  }
  return Allocate(CurrentBump(), bytes);
}

namespace {

// Reports an error that ends the program, as the host reports the same:
// `kind` is 0 for an uncaught exception, 1 for a feature not yet
// implemented and 2 for a limit of the runtime (rpc-opcodes.h), and the text
// is `text` followed by `detail`, which the host joins, so that no buffer on
// the stack holds them. The host ends the program, so this does not return.
[[noreturn]] void Fail(
    uint64_t kind, const char *text, const char *detail = "") {
  uint64_t size{0};
  while (text[size]) {
    ++size;
  }
  uint64_t detailSize{0};
  while (detail[detailSize]) {
    ++detailSize;
  }
  rpc::Client::Port port{__llvm_rpc_client.open<rpcFail>()};
  port.send([&](rpc::Buffer *buffer, uint32_t) { buffer->data[0] = kind; });
  port.send_n(text, size + 1);
  port.send_n(detail, detailSize + 1);
  port.recv([](rpc::Buffer *, uint32_t) {});
  // The host ends the program rather than answer, so the wave never gets
  // here. It ends rather than trap: llvm.trap is not nocallback, so no
  // function that calls it is norecurse, and a kernel that calls one has
  // the plugin's dynamic stack, whose scratch can keep some of a GPU
  // process kernel's teams from ever starting.
  __builtin_amdgcn_endpgm();
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
[[clang::loader_uninitialized]] Local<TeamTable *> currentTeams;

} // namespace

// At the start of each team of a target teams region, after
// merk_device_region_enter: finds the region's table in its slot, or makes
// it, and makes it the team's. Its id is the one the host chose, which its
// teams' mailboxes are named by.
extern "C" void merk_device_teams_enter(
    uint64_t id, uint64_t nodeHigh, uint64_t nodeLow, int32_t teams) {
  TeamTable **slot{&regionSlot->teams};
  if (TeamTable * table{__atomic_load_n(slot, __ATOMIC_ACQUIRE)}) {
    currentTeams = table;
    return;
  }
  // Another team may make one too, wasting this one if it is first.
  uint64_t count{static_cast<uint64_t>(teams)};
  auto *bytes{static_cast<char *>(
      merk_device_alloc(sizeof(TeamTable) + count * sizeof(TeamRing *)))};
  auto *made{reinterpret_cast<TeamTable *>(bytes)};
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
  TeamTable *none{nullptr};
  if (__atomic_compare_exchange_n(
          slot, &none, made, false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
    currentTeams = made;
  } else {
    currentTeams = none;
  }
}

// How long, unless the host says otherwise (rpcTeamsStall), the teams of a
// region with team mailboxes wait for another of them to start. Teams that
// fit on an idle device start within microseconds, but on a device that
// other programs' kernels fill they start as those end: a region whose
// teams waited a second from their own start for all to start failed so
// under a full test run, beside a test whose teams filled the GPU.
constexpr int64_t kTeamsStallMilliseconds{10000};

// The host's answer for how long teams wait for another to start.
int64_t TeamsStallMilliseconds() {
  int64_t milliseconds{kTeamsStallMilliseconds};
  rpc::Client::Port port{__llvm_rpc_client.open<rpcTeamsStall>()};
  port.send([](rpc::Buffer *, uint32_t) {});
  port.recv([&](rpc::Buffer *buffer, uint32_t) {
    milliseconds = static_cast<int64_t>(buffer->data[0]);
  });
  return milliseconds;
}

// After merk_device_teams_enter, in each team of a target teams region whose
// code uses team mailboxes: waits until every team of the region has started
// (docs/INPROGRESS.md, Processes as teams, step 4). A team that is not
// running deadlocks the teams that wait for its messages, and a GPU runs no
// more teams at once than its compute units hold, which depends on the
// kernel's registers and on what else the GPU runs. So while teams still
// start the others wait, and when none has started for the stall time, one
// team reports how many have, and the program ends. A team asks the host
// for the stall time only once it has waited a millisecond, which teams
// that fit do not.
extern "C" void merk_device_teams_wait_all(void) {
  TeamTable *table{currentTeams};
  int64_t started{__atomic_add_fetch(&table->started, 1, __ATOMIC_ACQ_REL)};
  int64_t seen{started};
  int64_t stall{kTeamsStallMilliseconds};
  bool asked{false};
  int64_t asking{Deadline(1)};
  int64_t deadline{Deadline(stall)};
  while (started < table->teams) {
    if (!asked && Now() >= asking) {
      asked = true;
      stall = TeamsStallMilliseconds();
      deadline = Deadline(stall);
    }
    if (started != seen) {
      seen = started;
      deadline = Deadline(stall);
    }
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

namespace {

// One team's arrival at a barrier of `teams` teams, whose counts are
// `arrived` and `rounds`: the last team to arrive ends the round, and the
// others wait until it has, which costs about one atomic round trip a team.
void ArriveAndWait(int64_t *arrived, int64_t *rounds, int64_t teams) {
  // Read before arriving: the round cannot end before this team arrives.
  int64_t round{__atomic_load_n(rounds, __ATOMIC_ACQUIRE)};
  if (__atomic_add_fetch(arrived, 1, __ATOMIC_ACQ_REL) == teams) {
    // No team arrives in the next round until it sees this one end.
    __atomic_store_n(arrived, 0, __ATOMIC_RELAXED);
    __atomic_add_fetch(rounds, 1, __ATOMIC_RELEASE);
    return;
  }
  while (__atomic_load_n(rounds, __ATOMIC_ACQUIRE) == round) {
    __builtin_amdgcn_s_sleep(1);
  }
}

} // namespace

// A barrier across the threads of the team, which the rounds of a loop that
// a GPU process runs across its teams end with (MLIRGenProcesses.cpp,
// GenerateProcessLoops).
extern "C" void merk_device_sync_threads(void) { __gpu_sync_threads(); }

// process.team_barrier on a device, which a team's initial thread calls. Only
// a region whose teams all run at once can wait like this, so one that calls
// it waits for its teams to start (merk_device_teams_wait_all,
// MarkRegionsForTeamMailboxes), and reports it if they do not. Outside a
// teams region there is one team.
extern "C" void merk_device_team_barrier(void) {
  TeamTable *table{currentTeams};
  if (!table || table->teams <= 1) {
    return;
  }
  ArriveAndWait(&table->arrived, &table->rounds, table->teams);
}

// A barrier across all the threads of all the teams of a cooperative kernel,
// a GPU process's (docs/INPROGRESS.md, GPU processes that run loops, (c)).
// Its counts are in device memory, zero when the kernel starts, one pair for
// each launch. Every thread of every team calls it, and every thread's writes
// before it are seen by every thread after it. The teams must all be
// resident at once, which a cooperative launch promises; there is no time
// limit, as a GPU process's other teams wait at it while its first team
// waits for a message.
struct MerkGridBarrier {
  int64_t arrived;
  int64_t rounds;
};

extern "C" void merk_device_grid_barrier(MerkGridBarrier *barrier) {
  // Each thread's writes are made visible to the device before its team's
  // barrier, so that thread 0's arrival publishes the team's.
  __scoped_atomic_thread_fence(__ATOMIC_RELEASE, __MEMORY_SCOPE_DEVICE);
  __gpu_sync_threads();
  if (__gpu_thread_id_x() == 0 && __gpu_thread_id_y() == 0 &&
      __gpu_thread_id_z() == 0) {
    int64_t teams{static_cast<int64_t>(__gpu_num_blocks_x()) *
        __gpu_num_blocks_y() * __gpu_num_blocks_z()};
    if (teams > 1) {
      ArriveAndWait(&barrier->arrived, &barrier->rounds, teams);
    }
  }
  __gpu_sync_threads();
  // And each thread sees the other teams' writes after it.
  __scoped_atomic_thread_fence(__ATOMIC_ACQUIRE, __MEMORY_SCOPE_DEVICE);
}

// At the end of each team of a target teams region: nothing to do on a
// device, whose next region sets the team's table again.
extern "C" void merk_device_teams_leave(void) {}

// A procedure of a GPU process that recurses other than by tail calls has
// found no room for its calls in merk_device_stack_bytes (Lowering.cpp,
// AddDeviceStackChecks): rather than overflow the stack that its kernel was
// launched with, the program ends, as on the host.
extern "C" [[noreturn]] void merk_device_stack_overflow(void) {
  Fail(2,
      "stack overflow in a GPU process: MERK_GPU_PROCESS_STACK_BYTES "
      "gives GPU processes more stack");
}

// The start of a process of the module by its kind's number, with the
// address of its argument words, which merkc makes in each image of GPU
// processes (MLIRGenProcesses.cpp).
extern "C" void merk_process_dispatch(uint64_t kind, uint64_t words);

// A thread's share of the module's loop with the number, which runs across a
// GPU process's teams (MLIRGenProcesses.cpp, GenerateProcessLoops): `rounds`
// rounds, the iteration `start` in the first and `stride` further in each
// next, those below `count` with the counter `first` + iteration * `step`,
// and merk_device_sync_threads after each round, with the words its body
// takes at `values` and the thread's partial results of the loop's
// reductions at `partials`, which it updates. Every thread of a team goes
// round as often.
extern "C" void merk_process_loop(uint64_t loop, int64_t start, int64_t rounds,
    int64_t stride, int64_t count, int64_t first, int64_t step, uint64_t values,
    uint64_t partials);

// Sets the partial results at the address to the identities of the loop's
// reductions.
extern "C" void merk_process_loop_start(uint64_t loop, uint64_t *partials);

// Combines the partial results at `from` into those at `into`, each by its
// reduction's operator.
extern "C" void merk_process_loop_reduce(
    uint64_t loop, uint64_t *into, const uint64_t *from);

// A thread's share of a loop under parallel do in the body of a distribute
// loop, as merk_process_loop's of another, with the words its body takes in
// the team's local memory.
extern "C" void merk_team_loop(uint64_t loop, int64_t start, int64_t rounds,
    int64_t stride, int64_t count, int64_t first, int64_t step, uint64_t values,
    uint64_t partials);

namespace {

// The process's wave arrives at its control block's barrier and waits there
// for its worker teams: its writes before are seen by every worker thread
// after, and theirs by it.
void ProcessArrive(ProcessControl *control) {
  __scoped_atomic_thread_fence(__ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
  ArriveAndWait(&control->arrived, &control->rounds, control->crossing);
  __scoped_atomic_thread_fence(__ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM);
}

// A worker team arrives at the barrier, as merk_device_grid_barrier's teams
// do: every thread of the team calls it.
void WorkerArrive(ProcessControl *control) {
  __scoped_atomic_thread_fence(__ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
  __gpu_sync_threads();
  if (__gpu_thread_id_x() == 0) {
    ArriveAndWait(&control->arrived, &control->rounds, control->crossing);
  }
  __gpu_sync_threads();
  __scoped_atomic_thread_fence(__ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM);
}

// A word of the table, which is in host memory, read by the wave's first
// active lane alone, for the whole wave: a read by every lane of every wave
// of a kernel on all of a GPU's units, each polling, slowed a process ten
// times (docs/INPROGRESS.md, Open after (p1)).
uint64_t LoadSlotWord(uint64_t *word) {
  uint64_t mask{__gpu_lane_mask()};
  uint64_t value{0};
  if (__gpu_is_first_in_lane(mask)) {
    value =
        __scoped_atomic_load_n(word, __ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM);
  }
  return __gpu_read_first_lane_u64(mask, value);
}

// Lane 0 of a wave whose slot the host has set running: the process, which
// starts its worker teams first if it has any, and ends them after.
[[gnu::noinline]] void RunProcess(
    PlaceTable *table, PlaceSlot *slot, uint64_t wave, uint32_t inTeam) {
  uint64_t kind{__scoped_atomic_load_n(
      &slot->kind, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM)};
  uint64_t words{__scoped_atomic_load_n(
      &slot->words, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM)};
  uint64_t workers{__scoped_atomic_load_n(
      &slot->workers, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM)};
  // The arena is in the device's memory, which only the device writes
  // while the kernel runs, so the wave clears it rather than the host.
  auto *arena{reinterpret_cast<Arena *>(__scoped_atomic_load_n(
      &slot->arena, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM))};
  auto *clear{reinterpret_cast<uint64_t *>(arena)};
  for (uint64_t i{0}; i < kArenaClearWords; ++i) {
    clear[i] = 0;
  }
  placeArenas[inTeam] = arena;
  teamMains[inTeam] = 0;
  ProcessControl *control{nullptr};
  if (workers > 0) {
    control = static_cast<ProcessControl *>(Allocate(&arena->bump,
        sizeof(ProcessControl) +
            workers * kProcessLoopReductions * sizeof(uint64_t)));
    control->arrived = 0;
    control->rounds = 0;
    control->crossing = static_cast<int64_t>(workers) + 1;
    control->end = 0;
    control->arena = arena;
    __scoped_atomic_thread_fence(__ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
    uint64_t slots{__scoped_atomic_load_n(
        &table->slots, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM)};
    for (uint64_t i{0}; i < slots; ++i) {
      PlaceSlot *other{PlaceSlots(table) + i};
      if (__scoped_atomic_load_n(&other->state, __ATOMIC_ACQUIRE,
              __MEMORY_SCOPE_SYSTEM) == kPlaceReserved &&
          __scoped_atomic_load_n(
              &other->owner, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM) == wave) {
        __scoped_atomic_store_n(&other->control,
            reinterpret_cast<uint64_t>(control), __ATOMIC_RELAXED,
            __MEMORY_SCOPE_SYSTEM);
        __scoped_atomic_store_n(&other->state, kPlaceWorking, __ATOMIC_RELEASE,
            __MEMORY_SCOPE_SYSTEM);
      }
    }
  }
  placeControls[inTeam] = control;
  merk_process_dispatch(kind, words);
  if (control) {
    // The workers see the end, set their slots reserved again, and cross
    // once more, after which none reads the control block, which goes with
    // the arena.
    __scoped_atomic_store_n(
        &control->end, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
    ProcessArrive(control);
    ProcessArrive(control);
    placeControls[inTeam] = nullptr;
  }
  __scoped_atomic_store_n(
      &slot->state, kPlaceIdle, __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
  ProcessEnded(table, wave);
}

// The partial results of a loop's reductions of the wave's lanes, at
// `partials`, combined into those of every lane, by exchanging them across
// halves of the wave.
void ReduceAcrossWave(uint64_t loop, uint64_t *partials) {
  uint64_t mask{__gpu_lane_mask()};
  uint32_t lanes{__gpu_num_lanes()};
  uint32_t lane{__gpu_lane_id()};
  uint64_t other[kProcessLoopReductions];
  for (uint32_t offset{lanes / 2}; offset > 0; offset /= 2) {
    for (uint32_t r{0}; r < kProcessLoopReductions; ++r) {
      other[r] = __gpu_shuffle_idx_u64(mask, lane ^ offset, partials[r], lanes);
    }
    merk_process_loop_reduce(loop, partials, other);
  }
}

static_assert(kMostPlaceWaves <= 32, "a team's waves fit in a wave's lanes");

// The partial results of a loop's reductions of the team's threads, at
// `partials`, combined into those of its first thread: in each wave, then
// the waves', which the first wave's lanes load from the team's local memory,
// one wave's a lane, the rest of its lanes the reductions' identities, and
// combine in the same way. The first thread combining them one after another
// took 5 µs a loop of 160 teams (design document, Measurements, Memory
// bandwidth).
void ReduceAcrossTeam(uint64_t loop, uint64_t *partials, uint32_t inTeam) {
  ReduceAcrossWave(loop, partials);
  uint32_t lanes{__gpu_num_lanes()};
  uint32_t lane{__gpu_lane_id()};
  if (lane == 0) {
    for (uint32_t r{0}; r < kProcessLoopReductions; ++r) {
      wavePartials[inTeam][r] = partials[r];
    }
  }
  __gpu_sync_threads();
  if (__gpu_thread_id_x() < lanes) {
    if (lane < __gpu_num_threads_x() / lanes) {
      for (uint32_t r{0}; r < kProcessLoopReductions; ++r) {
        partials[r] = wavePartials[lane][r];
      }
    } else {
      merk_process_loop_start(loop, partials);
    }
    ReduceAcrossWave(loop, partials);
  }
}

// The partial results of a loop's reductions of a worker team's threads,
// combined across the team and stored in the team's words of the control
// block. The last team to store them then combines every team's, each of
// its threads loading those of other teams, so that the loads overlap, into
// the values before the loop: one after another on the process's lane, a
// GPU of 160 teams took 26 µs a loop (design document, Measurements, Memory
// bandwidth). The process reads them after the barrier that ends the loop.
[[gnu::noinline]] void CombineTeamPartials(ProcessControl *control,
    uint64_t loop, const uint64_t *threadPartials, uint64_t worker,
    uint64_t workers, uint32_t inTeam) {
  // A copy of all kProcessLoopReductions words, so that the indices are
  // constants and the copy stays in registers.
  uint64_t partials[kProcessLoopReductions];
  for (uint32_t r{0}; r < kProcessLoopReductions; ++r) {
    partials[r] = threadPartials[r];
  }
  ReduceAcrossTeam(loop, partials, inTeam);
  if (__gpu_thread_id_x() == 0) {
    for (uint32_t r{0}; r < kProcessLoopReductions; ++r) {
      control->partials[worker * kProcessLoopReductions + r] = partials[r];
    }
    lastToStore = __scoped_atomic_fetch_add(&control->stored, 1,
                      __ATOMIC_ACQ_REL, __MEMORY_SCOPE_DEVICE) == workers - 1;
  }
  __gpu_sync_threads();
  if (!lastToStore) {
    return;
  }
  __scoped_atomic_thread_fence(__ATOMIC_ACQUIRE, __MEMORY_SCOPE_DEVICE);
  merk_process_loop_start(loop, partials);
  uint64_t threads{__gpu_num_threads_x()};
  for (uint64_t team{__gpu_thread_id_x()}; team < workers; team += threads) {
    merk_process_loop_reduce(
        loop, partials, control->partials + team * kProcessLoopReductions);
  }
  ReduceAcrossTeam(loop, partials, inTeam);
  if (__gpu_thread_id_x() == 0) {
    merk_process_loop_reduce(loop, control->results, partials);
  }
}

// The team's local memory as the generic address that generated code takes.
uint64_t *TeamWords(Local<uint64_t> *words) { return (uint64_t *)words; }

// Every thread of a worker team: its share of the loop under parallel do
// that the team's first wave, running a distribute loop's body, set in
// teamLoop, in rounds of the team's threads' consecutive iterations, with a
// barrier across the team after each, as a loop across the process's teams
// has (Work); its reductions combined across the team into teamResults; and
// a barrier, after which the first wave reads them, and may set the next.
[[gnu::noinline]] void TeamShare(uint32_t inTeam) {
  uint64_t loop{teamLoop.loop};
  int64_t first{teamLoop.first};
  int64_t step{teamLoop.step};
  int64_t count{teamLoop.count};
  uint64_t reductions{teamLoop.reductions};
  uint64_t partials[kProcessLoopReductions];
  if (reductions) {
    merk_process_loop_start(loop, partials);
  }
  int64_t threads{static_cast<int64_t>(__gpu_num_threads_x())};
  int64_t rounds{count > 0 ? (count + threads - 1) / threads : 0};
  merk_team_loop(loop, static_cast<int64_t>(__gpu_thread_id_x()), rounds,
      threads, count, first, step,
      reinterpret_cast<uint64_t>(TeamWords(teamValues)),
      reinterpret_cast<uint64_t>(partials));
  if (reductions) {
    ReduceAcrossTeam(loop, partials, inTeam);
    if (__gpu_thread_id_x() == 0) {
      merk_process_loop_reduce(loop, TeamWords(teamResults), partials);
    }
  }
  __gpu_sync_threads();
}

// Every thread of a worker team, for a distribute loop: the team's first
// wave runs the bodies of the team's iterations, `workers` apart from
// `worker`, all its lanes the same, and its other waves wait for the loops
// that the bodies run across the team's threads, until the first wave says
// that its iterations have ended. The first wave is told by a value that the
// compiler knows is the same in all its lanes, so that whole waves take each
// branch, and none crosses the other's barriers.
void Distribute(uint64_t loop, int64_t first, int64_t step, int64_t count,
    uint64_t values, uint64_t worker, uint64_t workers, uint32_t inTeam) {
  uint32_t wave{__gpu_read_first_lane_u32(__gpu_lane_mask(), inTeam)};
  if (wave == 0) {
    teamMains[0] = 1;
    auto start{static_cast<int64_t>(worker)};
    auto stride{static_cast<int64_t>(workers)};
    int64_t rounds{count > start ? (count - start + stride - 1) / stride : 0};
    merk_process_loop(
        loop, start, rounds, stride, count, first, step, values, 0);
    teamMains[0] = 0;
    if (__gpu_lane_id() == 0) {
      teamLoop.end = 1;
    }
    __gpu_sync_threads();
    return;
  }
  for (;;) {
    __gpu_sync_threads();
    if (teamLoop.end) {
      return;
    }
    TeamShare(wave);
  }
}

// Every thread of a wave whose slot its process has set working: the team
// runs the process's loops with its other worker teams, each thread taking
// iterations a team's threads apart, from its own in its team's block, until
// the process ends.
[[gnu::noinline]] void Work(PlaceSlot *slot, uint32_t inTeam) {
  auto *control{
      reinterpret_cast<ProcessControl *>(LoadSlotWord(&slot->control))};
  uint64_t worker{LoadSlotWord(&slot->worker)};
  uint64_t workers{LoadSlotWord(&slot->workers)};
  placeArenas[inTeam] = control->arena;
  placeControls[inTeam] = nullptr;
  teamMains[inTeam] = 0;
  int64_t threads{static_cast<int64_t>(__gpu_num_threads_x())};
  int64_t stride{static_cast<int64_t>(workers) * threads};
  int64_t base{static_cast<int64_t>(worker) * threads};
  int64_t thread{static_cast<int64_t>(__gpu_thread_id_x())};
  for (;;) {
    WorkerArrive(control);
    if (__scoped_atomic_load_n(
            &control->end, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM)) {
      break;
    }
    uint64_t loop{control->loop};
    int64_t first{control->first};
    int64_t step{control->step};
    int64_t count{control->count};
    uint64_t reductions{control->reductions};
    auto values{reinterpret_cast<uint64_t>(control->values)};
    if (control->distribute) {
      Distribute(loop, first, step, count, values, worker, workers, inTeam);
      WorkerArrive(control);
      continue;
    }
    uint64_t partials[kProcessLoopReductions];
    auto partialsAt{reinterpret_cast<uint64_t>(partials)};
    if (reductions) {
      merk_process_loop_start(loop, partials);
    }
    // In rounds, the team's threads `threads` consecutive iterations, the
    // teams `stride` apart, with a barrier across the team after each round,
    // so that its waves stay together, as the device runtime's loops do for
    // stencils, whose waves read the same rows (design document, A barrier
    // after each round). Every thread goes round as often as the team. The
    // loop's procedure runs the rounds, with its body inlined, so that only
    // the loops' registers count in it (design document, Measurements,
    // Memory bandwidth).
    int64_t rounds{count > base ? (count - base + stride - 1) / stride : 0};
    merk_process_loop(loop, base + thread, rounds, stride, count, first, step,
        values, partialsAt);
    if (reductions) {
      CombineTeamPartials(control, loop, partials, worker, workers, inTeam);
    }
    WorkerArrive(control);
  }
  if (__gpu_lane_id() == 0) {
    __scoped_atomic_store_n(
        &slot->state, kPlaceReserved, __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
  }
  WorkerArrive(control);
}

} // namespace

// Each wave of the kernel of a GPU's processes, merk_place (place.h): lane 0
// polls the wave's slot of the table, sleeping, and runs each process that
// the host starts there, while the wave's other lanes wait for it; the whole
// wave works when its team is a process's worker team; and it returns once
// the table's end is set while the slot is idle. The table is in host memory,
// which the host writes while the kernel runs, so it is read at system scope.
// Each team first counts itself started, which the host waits for.
extern "C" void merk_device_place(PlaceTable *table) {
  if (__gpu_thread_id_x() == 0) {
    __scoped_atomic_fetch_add(
        &table->started, 1, __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
  }
  uint32_t inTeam{__gpu_thread_id_x() / __gpu_num_lanes()};
  uint64_t wave{static_cast<uint64_t>(__gpu_block_id_x()) *
          (__gpu_num_threads_x() / __gpu_num_lanes()) +
      inTeam};
  if (inTeam >= kMostPlaceWaves || wave >= LoadSlotWord(&table->slots)) {
    return;
  }
  PlaceSlot *slot{PlaceSlots(table) + wave};
  // A wave whose slot stays as it is sleeps twice as long before each look,
  // up to kMostPlaceSleeps sleeps of about 3.5 us: every wave of a kernel on
  // all of a GPU's units looking at the table in host memory every sleep
  // slowed a process ten times (docs/INPROGRESS.md, Open after (p1)).
  uint64_t seen{kPlaceIdle};
  uint32_t sleeps{1};
  for (;;) {
    uint64_t state{LoadSlotWord(&slot->state)};
    if (state != seen) {
      seen = state;
      sleeps = 1;
    }
    if (state == kPlaceRunning) {
      if (__gpu_lane_id() == 0) {
        RunProcess(table, slot, wave, inTeam);
      }
      continue;
    }
    if (state == kPlaceWorking) {
      Work(slot, inTeam);
      continue;
    }
    if (state == kPlaceIdle && LoadSlotWord(&table->end)) {
      return;
    }
    for (uint32_t i{0}; i < sleeps; ++i) {
      __builtin_amdgcn_s_sleep(127);
    }
    if (sleeps < kMostPlaceSleeps) {
      sleeps *= 2;
    }
  }
}

// The words of the loop that the process is about to run across its worker
// teams (MLIRGenProcesses.cpp, ProcessLoop), in its control block.
extern "C" uint64_t *merk_device_loop_values(void) {
  ProcessControl *control{
      placeControls[__gpu_thread_id_x() / __gpu_num_lanes()]};
  if (!control) {
    Fail(1,
        "a loop under distribute parallel do or distribute in the body of "
        "another loop, or in a GPU process without worker teams");
  }
  return control->values;
}

// The words of the values of the loop's reductions before it, which the
// process sets, and which are their results after it.
extern "C" uint64_t *merk_device_loop_results(void) {
  ProcessControl *control{
      placeControls[__gpu_thread_id_x() / __gpu_num_lanes()]};
  if (!control) {
    Fail(1,
        "a loop under distribute parallel do or distribute in the body of "
        "another loop, or in a GPU process without worker teams");
  }
  return control->results;
}

// Runs the loop with the number across the process's worker teams, from the
// first counter by the step, `count` times, with the words that
// merk_device_loop_values gave, and `reductions` reductions: the workers
// start when the process's wave arrives at the barrier, and it returns when
// they arrive again, done, the last of them having combined their teams'
// partial results into those of merk_device_loop_results
// (CombineTeamPartials).
extern "C" void merk_device_process_loop(uint64_t loop, int64_t first,
    int64_t step, int64_t count, uint64_t reductions) {
  ProcessControl *control{
      placeControls[__gpu_thread_id_x() / __gpu_num_lanes()]};
  if (!control) {
    Fail(1,
        "a loop under distribute parallel do or distribute in the body of "
        "another loop, or in a GPU process without worker teams");
  }
  control->loop = loop;
  control->first = first;
  control->step = step;
  control->count = count;
  control->distribute = 0;
  control->reductions = reductions;
  control->stored = 0;
  ProcessArrive(control);
  ProcessArrive(control);
}

// Runs the distribute loop with the number across the process's worker
// teams, each team's first wave running the bodies of its iterations, from
// the first counter by the step, `count` times, with the words that
// merk_device_loop_values gave (Distribute).
extern "C" void merk_device_process_distribute(
    uint64_t loop, int64_t first, int64_t step, int64_t count) {
  ProcessControl *control{
      placeControls[__gpu_thread_id_x() / __gpu_num_lanes()]};
  if (!control) {
    Fail(1,
        "a loop under distribute parallel do or distribute in the body of "
        "another loop, or in a GPU process without worker teams");
  }
  control->loop = loop;
  control->first = first;
  control->step = step;
  control->count = count;
  control->distribute = 1;
  control->reductions = 0;
  control->stored = 0;
  ProcessArrive(control);
  ProcessArrive(control);
}

namespace {

// The wave runs a distribute loop's body, the only code that may run a loop
// across a team's threads; check-gpu-code reports any other at compile time.
void CheckTeamMain() {
  if (!merk_device_process_image ||
      !teamMains[__gpu_thread_id_x() / __gpu_num_lanes()]) {
    Fail(1,
        "a loop under parallel do in a GPU process's code outside the body "
        "of a distribute loop");
  }
}

} // namespace

// The words of the loop under parallel do that a distribute loop's body is
// about to run across the team's threads (MLIRGenProcesses.cpp,
// ProcessLoop), in the team's local memory.
extern "C" uint64_t *merk_device_team_values(void) {
  CheckTeamMain();
  return TeamWords(teamValues);
}

// The words of the values of its reductions before it, which the first wave
// sets, and which are their results after it.
extern "C" uint64_t *merk_device_team_results(void) {
  CheckTeamMain();
  return TeamWords(teamResults);
}

// Runs the loop with the number across the team's threads, from the first
// counter by the step, `count` times, with the words that
// merk_device_team_values gave, and `reductions` reductions, whose results
// are in those of merk_device_team_results when it returns: the team's
// other waves, waiting in Distribute, start when the first wave crosses the
// team's barrier, and every thread of the team takes its share (TeamShare).
// Every lane of the first wave calls it, with the same values.
extern "C" void merk_device_team_loop(uint64_t loop, int64_t first,
    int64_t step, int64_t count, uint64_t reductions) {
  CheckTeamMain();
  uint32_t inTeam{__gpu_thread_id_x() / __gpu_num_lanes()};
  if (__gpu_is_first_in_lane(__gpu_lane_mask())) {
    teamLoop.loop = loop;
    teamLoop.first = first;
    teamLoop.step = step;
    teamLoop.count = count;
    teamLoop.reductions = reductions;
    teamLoop.end = 0;
  }
  // The iterations make their own cells.
  teamMains[inTeam] = 0;
  __gpu_sync_threads();
  TeamShare(inTeam);
  teamMains[inTeam] = 1;
}

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
  RingEntry **bucket{merk_device_process_image
          ? &CurrentArena()->rings[handle[2] % kRingCacheSlots]
          : &regionSlot->rings[handle[2] % kRegionRingSlots]};
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
      // The region that attaches it, which the host counts as a user of the
      // ring until it ends.
      buffer->data[7] = merk_device_process_image
          ? kNoRegionSlot
          : static_cast<uint64_t>(regionSlot - regionSlots);
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
// ring gives the oldest message's size, the host copies the message into
// host memory that the device reads (rpcTakeArray), and the device copies
// that into an array in its arena before it moves the ring's head, after
// which the host frees its copy. From a team mailbox, the sender's array
// itself (merk_device_send_array).
// Temporary (docs/INPROGRESS.md, GPU processes without omp target, step 5,
// "Flat records"): one lane copies word by word, until messages travel as
// their bytes and a wave's lanes copy them together.
extern "C" int merk_device_receive_array(
    const uint64_t *handle, int64_t milliseconds, uint64_t *message) {
  if (IsTeamHandle(handle)) {
    return TeamReceive(handle, milliseconds, message);
  }
  RingEntry *entry{nullptr};
  // The slot is the array's header (array.h).
  uint64_t header{0};
  if (!Wait(milliseconds,
          [&]() { return Look(handle, kRingOfArrays, entry, &header); })) {
    return 0;
  }
  bool found{false};
  const uint64_t *copy{nullptr};
  {
    rpc::Client::Port port{__llvm_rpc_client.open<rpcTakeArray>()};
    port.send([&](rpc::Buffer *buffer, uint32_t) {
      for (int i{0}; i < 5; ++i) {
        buffer->data[i] = handle[i];
      }
    });
    port.recv([&](rpc::Buffer *buffer, uint32_t) {
      found = buffer->data[0] != 0;
      copy = reinterpret_cast<const uint64_t *>(buffer->data[1]);
    });
  }
  // A mailbox that closed meanwhile has no message.
  if (!found) {
    return 0;
  }
  // The host wrote the copy before it answered.
  __scoped_atomic_thread_fence(__ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM);
  uint64_t words{ArrayWordsOfHeader(header)};
  auto *array{
      static_cast<uint64_t *>(merk_device_alloc(sizeof(uint64_t) * words))};
  for (uint64_t i{0}; i < words; ++i) {
    array[i] = copy[i];
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
  Fail(2, message, detail);
}

// Before each value that a copy copies, which recurses on the value's
// structure: that the stack has room after the copy's frame for what its
// calls may take, as the image's own recursion checks (Lowering.cpp,
// AddDeviceStackChecks). It is inlined into the copy, whose callees the
// backend measures.
[[gnu::always_inline]] inline void CheckCopyStack() {
  auto at{static_cast<uint32_t>(
      reinterpret_cast<uintptr_t>(merk_device_stack_save()))};
  if (at + __builtin_amdgcn_callee_stack_size() > merk_device_stack_bytes)
      [[unlikely]] {
    merk_device_stack_overflow();
  }
}

// Where the device's copy of a message it receives goes: its arena.
struct ArenaMemory {
  static constexpr bool kOtherImage{true};
  void *Allocate(uint64_t bytes) { return merk_device_alloc(bytes); }
  [[gnu::always_inline]] void CheckStack() { CheckCopyStack(); }
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
  [[gnu::always_inline]] void CheckStack() { CheckCopyStack(); }

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

namespace {

// The array, copied into host memory, which the host reads once the device
// has passed it the copy: not read by the host from the device's memory,
// whose lines the device's L2 cache may hold unwritten while it runs, which
// the device's fences do not write back.
// Temporary (docs/INPROGRESS.md, GPU processes without omp target, step 5,
// "Flat records"): one lane copies word by word, until messages travel as
// their bytes and a wave's lanes copy them together.
uint64_t *CopyToHost(const uint64_t *array, PinnedMemory &memory) {
  uint64_t words{ArrayWords(array)};
  auto *copy{
      static_cast<uint64_t *>(memory.Allocate(sizeof(uint64_t) * words))};
  for (uint64_t i{0}; i < words; ++i) {
    copy[i] = array[i];
  }
  __scoped_atomic_thread_fence(__ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
  return copy;
}

} // namespace

// What a region gives for an output array: the array, if it is the given
// array numbered `number`, which is mapped back; or else, for -1, a copy on
// the host of one the device made, since the arena does not outlive the
// region.
extern "C" uint64_t *merk_device_give_array(uint64_t *array, int64_t number) {
  if (number >= 0) {
    return array;
  }
  PinnedMemory memory;
  uint64_t *pinned{CopyToHost(array, memory)};
  uint64_t *copy{nullptr};
  rpc::Client::Port port{__llvm_rpc_client.open<rpcGiveArray>()};
  port.send([&](rpc::Buffer *buffer, uint32_t) {
    buffer->data[0] = reinterpret_cast<uint64_t>(pinned);
    // Its header, from which the host finds its bytes.
    buffer->data[1] = array[0];
    buffer->data[2] = reinterpret_cast<uint64_t>(memory.First());
  });
  port.recv([&](rpc::Buffer *buffer, uint32_t) {
    copy = reinterpret_cast<uint64_t *>(buffer->data[0]);
  });
  return copy;
}

// Sends an array of one-word elements, as merk_device_send sends a word:
// the device copies the array into host memory (CopyToHost), and the host
// sends that copy, as merk_device_send_term sends a term's, and returns
// once it has queued it. To a team mailbox, the array goes by its
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
  PinnedMemory memory;
  uint64_t *copy{CopyToHost(array, memory)};
  rpc::Client::Port port{__llvm_rpc_client.open<rpcSendTerm>()};
  port.send([&](rpc::Buffer *buffer, uint32_t) {
    for (int i{0}; i < 5; ++i) {
      buffer->data[i] = handle[i];
    }
    buffer->data[5] = key;
    buffer->data[6] = reinterpret_cast<uint64_t>(copy);
    buffer->data[7] = reinterpret_cast<uint64_t>(memory.First());
  });
  port.recv([](rpc::Buffer *, uint32_t) {});
}

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
