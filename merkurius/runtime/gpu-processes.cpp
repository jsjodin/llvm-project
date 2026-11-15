//===-- runtime/gpu-processes.cpp - Processes that run on a GPU -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Spawning a process on a GPU (docs/INPROGRESS.md, GPU processes without
// omp target, step 4c, and GPU processes that run loops, one kernel for a
// GPU's processes). The first spawn of one of a module's kinds on a GPU
// loads the module's image there through liboffload (process-images.h), and
// sets the image's merk_device_number to the GPU's number, which the
// device's runtime sends with its opcodes.
//
// A GPU's processes run in one kernel, the image's merk_place, a wave for
// each process (place.h). The runtime launches it at the first spawn of a
// module's kind on the GPU, cooperatively, with as many teams as are
// resident at once on its compute units for processes (compute-units.h), on a
// queue limited to them. The kernel takes a percentage of the units that the
// GPU's queues may use, 100 by default, MERK_GPU_PROCESS_UNITS, or what the
// program sets between kernels (merk_gpu_set_units); a launch makes a new
// queue when the units change, and the reaper destroys the old one once no
// kernel uses it. A target region on a GPU whose processes' kernel holds
// every unit would never start, and fails instead (merk_check_region_units).
// A spawn
// writes the process's slot of the kernel's table. A spawn fails, rather than
// waits, when no slot is free, or when the GPU's kernel is another module's:
// the processes running there could be waiting for the new one. The kernel ends
// once its last process has ended, and the next module's kernel starts then,
// after it on the queue.
//
// The mailbox is the host's, owned by a record of the process's that is never
// run (NewGpuProcessRecord), so that monitors can watch the process and no
// process of the host's receives from it: the GPU receives from it through a
// ring, as a region does (offload.cpp). When the process ends, the device
// tells the host (rpcProcessEnded), which, on the plugin's thread, frees its
// slot, ends the kernel if it was the last, closes the mailbox and tells the
// process's monitors, in that order, so that a watcher may spawn the next
// module's kind at once. That thread must not call liboffload, so a thread
// of the runtime's, the reaper, then frees what the process had:
// its arena, its argument words, and its mailbox's ring and the copy of a
// message there; and frees each kernel's table once the kernel has ended.
// Temporary (step 5 of GPU processes without omp target,
// docs/INPROGRESS.md): its handle's place is the host's, 0, until handles
// are routed by place.
//
// A running GPU process counts as a running region until the reaper has
// freed what it had, so that the rings that it may read stay while it runs,
// and the program ends at once when main does (offload.cpp).
//
//===----------------------------------------------------------------------===//

#include "compute-units.h"
#include "device-ring.h"
#include "devices.h"
#include "fragment.h"
#include "liboffload.h"
#include "place.h"
#include "process-images.h"
#include "process.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <pthread.h>

extern "C" void merk_node(std::int64_t *high, std::int64_t *low);
extern "C" [[noreturn]] void merk_uncaught_exception(const char *exception);
// OpenMP's, in a program with target regions, the only one that asks.
extern "C" int omp_get_default_device(void) __attribute__((weak));
extern "C" void merk_mailbox_close(std::int64_t nodeHigh, std::int64_t nodeLow,
    std::int64_t placeAndSlot, std::int64_t generation, std::int64_t secret);

extern "C" void merk_mailbox_close_ended(std::int64_t nodeHigh,
    std::int64_t nodeLow, std::int64_t placeAndSlot, std::int64_t generation,
    std::int64_t secret, merkurius::runtime::DeviceRing **ring, int *device,
    void **copy, void (**release)(int device, void *copy));

namespace merkurius::runtime {
void GpuProcessStarted();
void GpuProcessEnded();
// Whether MERK_CHECK_DEVICE_MEMORY is set (offload.cpp).
bool CheckingDeviceMemory();
void *NewProcessArena(int device);
void FreeProcessArena(int device, void *arena);
void FreeRing(int device, DeviceRing *ring);
// How long teams that must all run at once wait for another of them to
// start (offload.cpp, MERK_TEAMS_STALL_MILLISECONDS).
std::uint64_t TeamsStallMilliseconds();
} // namespace merkurius::runtime

namespace {

using namespace merkurius::runtime;

constexpr int kHandleWords{5};

// A module's image, loaded on a GPU.
struct LoadedImage {
  const char *module;
  int gpu;
  ol_program_handle_t program;
  const ProcessImage *image;
  LoadedImage *next;
};

pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
// The images loaded, with a lock of their own, which `lock` is never taken
// inside: loading an image takes the plugin's RPC server's lock, which the
// server holds while it runs our opcodes, some of which take `lock`
// (rpcProcessEnded).
pthread_mutex_t loadLock = PTHREAD_MUTEX_INITIALIZER;
LoadedImage *loaded{nullptr};

struct Place;

// A running process: its GPU, arena and argument words, its kernel, its
// record, which owns its mailbox, and its mailbox's handle, which is closed
// when it ends, taking the ring and the copy of a message there for the
// reaper to free.
struct RunningProcess {
  int gpu;
  void *arena;
  std::uint64_t *words;
  Place *place;
  // Its slot of the kernel, its incarnation (place.h), and its queue, in the
  // device's memory, and its peer queue, in the device's fine-grained
  // memory, or 0 without the peer path (PeerQueue); and the next on the list
  // of routes, from its spawn, once its handle is made, until it ends
  // (PlaceRoute).
  std::uint64_t slot;
  std::uint64_t incarnation;
  std::uint64_t queue;
  std::uint64_t peerQueue;
  RunningProcess *nextRoute;
  Process *record;
  std::int64_t handle[kHandleWords];
  DeviceRing *ring;
  int ringDevice;
  void *copy;
  void (*release)(int device, void *copy);
  RunningProcess *next;
};

// A launch of a module's kernel on a GPU: its table, in host memory, the
// process in each slot, for each slot of a worker team the slot of its
// process plus one, or 0 (place.h), the slots of each team, and how many
// processes it runs, of which the last ends the kernel. Once the kernel has
// ended, the reaper frees the table.
struct Place {
  const char *module;
  int gpu;
  // Its queue, and whether its units are all those of the GPU's queues.
  ol_queue_handle_t queue;
  bool allUnits;
  PlaceTable *table;
  // The slots' queues, in the device's memory (place.h, ProcessQueue), and
  // their peer queues, a block of the GPU's, or none (PeerBlock).
  void *queues;
  struct PeerBlock *peerBlock;
  RunningProcess **processes;
  std::uint64_t *owners;
  std::uint64_t teamSlots;
  std::uint64_t running;
  bool ending;
  Place *next;
};

// For each GPU, from its first spawn: the pairs of units that its queues may
// use (compute-units.h), and how many units they may use; the percentage of
// them that the program set for its processes' kernels, or 0 for the
// default; the share of a kernel's teams that a kind with loops takes as its
// workers, K for a K'th, or 0 for every free team; the queue of its latest
// kernel, limited to that kernel's units, or none; and the kernel that
// spawns use, unless it is ending.
struct GpuPlaces {
  bool ready;
  UnitMask units;
  int usable;
  int percent;
  std::int64_t share;
  ol_queue_handle_t queue;
  UnitMask queueUnits;
  Place *current;
  // A spawn is waiting for the teams of a kernel it launched, without the
  // lock (Launch), and the GPU's other spawns wait for it.
  bool launching;
};
GpuPlaces gpuPlaces[kMostGpus];

// Direct sends between the GPUs of a node (docs/INPROGRESS.md, Direct sends
// between the GPUs of a node), with MERK_GPU_PEER=1; without it, every send
// to another GPU's process goes through the host, as before.
//
// THIS MEMORY IS NEVER FREED BEFORE THE PROGRAM ENDS, on purpose (place.h,
// PeerQueue): a sender on another GPU may hold the address of a peer queue
// after its kernel has ended, and a receiver may return a copy to a send
// pool long after its sender ended. What a GPU keeps is bounded: its send
// pool and return ring, made at its first kernel with the peer path, and as
// many blocks of peer queues as it ever ran kernels at once, each of the
// most slots it took, which later kernels take again.
//
// A block of peer queues, for a kernel's slots, and the next free one of its
// GPU.
struct PeerBlock {
  void *queues;
  std::uint64_t slots;
  PeerBlock *next;
};

// For each GPU, from its first kernel with the peer path: whether its peer
// memory was tried, and made; its send pool, its return ring, and its free
// blocks of peer queues. Under `lock`.
struct GpuPeer {
  bool tried;
  bool ready;
  void *sendPool;
  std::uint64_t sendPoolBytes;
  void *returns;
  PeerBlock *free;
};
GpuPeer gpuPeers[kMostGpus];

// The incarnation of the last GPU process started, of any GPU and kernel
// (place.h, PlaceSlot). Under `lock`.
std::uint64_t processIncarnations{0};

// The running processes that a GPU process may send to directly
// (PlaceRoute), with a lock of their own: the plugin's RPC server asks for a
// route while a spawn may hold `lock` across a call of liboffload, which
// waits for the server.
pthread_mutex_t routesLock = PTHREAD_MUTEX_INITIALIZER;
RunningProcess *routes{nullptr};

// The blocks of messages between GPU processes that the kernels the reaper
// has freed left allocated (PlaceTable), with MERK_CHECK_DEVICE_MEMORY.
std::int64_t endedPlaceBlocks{0};
// The same for their messages put in peer queues, and copies in send pools
// not given back (PlaceTable's `peerSends` and `peerCopies`).
std::int64_t endedPeerSends{0};
std::int64_t endedPeerCopies{0};

// Every kernel that has not ended, and those that have, for the reaper.
Place *places{nullptr};
Place *endedPlaces{nullptr};

// The processes that have ended, for the reaper.
pthread_cond_t endedCondition = PTHREAD_COND_INITIALIZER;
// Signalled when a launch has seen its teams start, or not (Launch).
pthread_cond_t launchedCondition = PTHREAD_COND_INITIALIZER;
RunningProcess *ended{nullptr};
pthread_once_t reaperOnce = PTHREAD_ONCE_INIT;

// The text of an error, for process.spawn to report: it is never freed,
// as the program ends with it.
const char *Error(const char *format, const char *a, long long n,
    const char *b, const char *c, const char *d) {
  char *text{static_cast<char *>(Malloc(512))};
  std::snprintf(text, 512, format, a, n, b, c, d);
  return text;
}

const char *Details(ol_result_t result) {
  return result && result->details ? result->details : "failed";
}

// Whether the program asked for direct sends between GPUs: MERK_GPU_PEER=1,
// read once for each GPU, as its peer memory is made (ReadyPeer). Off by
// default until they have a test with two GPUs, and large arrays a faster
// path than one wave's loads (docs/INPROGRESS.md, Direct sends between the
// GPUs of a node, steps 5 and 6).
bool PeerWanted() {
  const char *text{std::getenv("MERK_GPU_PEER")};
  return text && std::strcmp(text, "1") == 0;
}

// The bytes of a GPU's send pool: MERK_GPU_SEND_POOL_BYTES, from 1 MiB, or
// 64 MiB.
constexpr std::uint64_t kSendPoolBytes{std::uint64_t{64} << 20};

std::uint64_t SendPoolBytes() {
  if (const char *text{std::getenv("MERK_GPU_SEND_POOL_BYTES")}) {
    char *end{nullptr};
    unsigned long long bytes{std::strtoull(text, &end, 10)};
    if (end != text && *end == '\0' && bytes >= std::uint64_t{1} << 20) {
      return bytes;
    }
  }
  return kSendPoolBytes;
}

// Zeroed host memory of `bytes`, a multiple of 64, aligned as a cache line,
// for the image of a PeerQueue or ReturnRing; or null.
void *ZeroedAligned(std::size_t bytes) {
  void *memory{std::aligned_alloc(64, bytes)};
  if (memory) {
    std::memset(memory, 0, bytes);
  }
  return memory;
}

// Copies `bytes` of zeros to the device's memory at `to`: whether it could.
bool ZeroDevice(int device, void *to, std::size_t bytes) {
  void *zeros{std::calloc(1, bytes)};
  bool copied{zeros && CopyToDevice(device, to, zeros, bytes)};
  std::free(zeros);
  return copied;
}

// The GPU's send pool and return ring, made the first time a kernel of it
// has the peer path; if they cannot be, the GPU's processes send to other
// GPUs through the host, and no route to them is direct. Kept until the
// program ends (GpuPeer). With the lock held.
bool ReadyPeer(const Gpu &gpu) {
  GpuPeer &peer{gpuPeers[gpu.number]};
  if (peer.tried) {
    return peer.ready;
  }
  peer.tried = true;
  if (!PeerWanted()) {
    return false;
  }
  std::uint64_t bytes{SendPoolBytes()};
  std::uint64_t slots{ReturnRingSlots(bytes)};
  std::size_t ringBytes{sizeof(ReturnRing) + slots * sizeof(ReturnRingSlot)};
  ringBytes = (ringBytes + 63) & ~std::size_t{63};
  auto *returns{static_cast<ReturnRing *>(
      DeviceAllocateFine(gpu.number, ringBytes))};
  if (!returns) {
    return false;
  }
  auto *ring{static_cast<ReturnRing *>(ZeroedAligned(ringBytes))};
  if (ring) {
    ring->slots = slots;
    for (std::uint64_t i{0}; i < slots; ++i) {
      RingSlots(ring)[i].sequence = i;
    }
  }
  bool made{ring && CopyToDevice(gpu.number, returns, ring, ringBytes)};
  std::free(ring);
  void *pool{made ? DeviceAllocate(gpu.number, bytes) : nullptr};
  if (!pool || !ZeroDevice(gpu.number, pool, SendPoolClearBytes(bytes))) {
    if (pool) {
      DeviceFree(gpu.number, pool);
    }
    DeviceFreeFine(gpu.number, returns);
    return false;
  }
  peer.sendPool = pool;
  peer.sendPoolBytes = bytes;
  // Read by PeerGpuOfReturns on the RPC server's thread, without the lock.
  __atomic_store_n(&peer.returns, returns, __ATOMIC_RELEASE);
  peer.ready = true;
  return true;
}

// A block of peer queues for a kernel of `slots` slots on the GPU: a free
// one of at least that many, or a new one, made empty (place.h, PeerQueue);
// or null without the peer path, or if it cannot be made, when the kernel's
// processes take no direct route from other GPUs. With the lock held.
PeerBlock *TakePeerBlock(const Gpu &gpu, std::uint64_t slots) {
  if (!ReadyPeer(gpu)) {
    return nullptr;
  }
  GpuPeer &peer{gpuPeers[gpu.number]};
  for (PeerBlock **at{&peer.free}; *at; at = &(*at)->next) {
    if ((*at)->slots >= slots) {
      PeerBlock *block{*at};
      *at = block->next;
      block->next = nullptr;
      return block;
    }
  }
  std::size_t bytes{slots * sizeof(PeerQueue)};
  void *queues{DeviceAllocateFine(gpu.number, bytes)};
  auto *empty{static_cast<PeerQueue *>(ZeroedAligned(bytes))};
  if (!queues || !empty) {
    if (queues) {
      DeviceFreeFine(gpu.number, queues);
    }
    std::free(empty);
    return nullptr;
  }
  for (std::uint64_t q{0}; q < slots; ++q) {
    for (std::uint64_t i{0}; i < kPeerQueueSlots; ++i) {
      empty[q].slots[i].sequence = i;
    }
  }
  bool copied{CopyToDevice(gpu.number, queues, empty, bytes)};
  std::free(empty);
  if (!copied) {
    DeviceFreeFine(gpu.number, queues);
    return nullptr;
  }
  auto *block{static_cast<PeerBlock *>(Malloc(sizeof(PeerBlock)))};
  *block = {queues, slots, nullptr};
  return block;
}

// Puts an ended kernel's block of peer queues back on its GPU's list, for
// the GPU's next kernel: never freed (GpuPeer). With the lock held.
void ReturnPeerBlock(int gpu, PeerBlock *block) {
  if (block) {
    block->next = gpuPeers[gpu].free;
    gpuPeers[gpu].free = block;
  }
}

// Ends the kernel, whose processes have all ended: its idle waves return.
// With the lock held.
void EndPlace(Place *place) {
  place->ending = true;
  __atomic_store_n(&place->table->end, 1, __ATOMIC_RELEASE);
}

// Frees what an ended process had.
void Reap(RunningProcess *process) {
  if (process->copy) {
    process->release(process->ringDevice, process->copy);
  }
  if (process->ring) {
    FreeRing(process->ringDevice, process->ring);
  }
  FreeProcessArena(process->gpu, process->arena);
  DeviceFreeHost(process->gpu, process->words);
  std::free(process);
  GpuProcessEnded();
}

// Frees an ended kernel's table and queues, keeping the count of its blocks
// of messages that are still allocated (PlaceTable).
void ReapPlace(Place *place) {
  __atomic_add_fetch(&endedPlaceBlocks, place->table->blocks, __ATOMIC_RELAXED);
  __atomic_add_fetch(&endedPeerSends, place->table->peerSends, __ATOMIC_RELAXED);
  __atomic_add_fetch(
      &endedPeerCopies, place->table->peerCopies, __ATOMIC_RELAXED);
  DeviceFreeHost(place->gpu, place->table);
  DeviceFree(place->gpu, place->queues);
  pthread_mutex_lock(&lock);
  ReturnPeerBlock(place->gpu, place->peerBlock);
  pthread_mutex_unlock(&lock);
  std::free(place->processes);
  std::free(place->owners);
  std::free(place);
}

// Whether a kernel of the list was launched on the queue.
bool QueueIn(const Place *list, ol_queue_handle_t queue) {
  for (const Place *place{list}; place; place = place->next) {
    if (place->queue == queue) {
      return true;
    }
  }
  return false;
}

// Whether a kernel that the reaper has not taken was launched on the queue.
// With the lock held.
bool QueueUsed(ol_queue_handle_t queue) {
  return QueueIn(places, queue) || QueueIn(endedPlaces, queue);
}

// Destroys a queue on which no kernel runs any longer, once the host
// function that followed its last kernel has returned.
void DestroyQueue(ol_queue_handle_t queue) {
  olSyncQueue(queue);
  olDestroyQueue(queue);
}

// The reaper's thread, for the program's life. A kernel ends only after its
// processes have told the host that they ended, so the reaper has taken
// them, in the same round or an earlier one, before it takes the kernel;
// the processes no longer refer to it by then.
void *Reaper(void *) {
  for (;;) {
    pthread_mutex_lock(&lock);
    while (!ended && !endedPlaces) {
      pthread_cond_wait(&endedCondition, &lock);
    }
    RunningProcess *list{ended};
    ended = nullptr;
    Place *placeList{endedPlaces};
    endedPlaces = nullptr;
    pthread_mutex_unlock(&lock);
    while (list) {
      RunningProcess *next{list->next};
      Reap(list);
      list = next;
    }
    while (placeList) {
      Place *next{placeList->next};
      // Its queue goes with the last kernel on it, unless the GPU's next
      // kernel may use it.
      pthread_mutex_lock(&lock);
      ol_queue_handle_t queue{placeList->queue};
      placeList->queue = nullptr;
      bool unused{queue != gpuPlaces[placeList->gpu].queue &&
          !QueueUsed(queue) && !QueueIn(next, queue)};
      pthread_mutex_unlock(&lock);
      ReapPlace(placeList);
      if (unused) {
        DestroyQueue(queue);
      }
      placeList = next;
    }
  }
  return nullptr;
}

void StartReaper() {
  pthread_t thread;
  if (pthread_create(&thread, nullptr, Reaper, nullptr) != 0) {
    std::fflush(stdout);
    std::fputs("Merkurius runtime: could not start the thread that frees what "
               "GPU processes had\n",
        stderr);
    ExitProgram(1);
  }
  pthread_detach(thread);
}

// On liboffload's thread, after a kernel has ended: the reaper frees its
// table.
void PlaceDone(void *data) {
  auto *place{static_cast<Place *>(data)};
  pthread_mutex_lock(&lock);
  GpuPlaces &gpu{gpuPlaces[place->gpu]};
  if (gpu.current == place) {
    gpu.current = nullptr;
  }
  for (Place **link{&places}; *link; link = &(*link)->next) {
    if (*link == place) {
      *link = place->next;
      break;
    }
  }
  place->next = endedPlaces;
  endedPlaces = place;
  pthread_cond_signal(&endedCondition);
  pthread_mutex_unlock(&lock);
}

// The module's image on the GPU, loaded the first time, or the text of the
// error (Load, below). With loadLock held.
std::uint64_t ProcessStackBytes(const char *&error);

// Sets the stack that the image's kernel is launched with, a lane, in its
// merk_device_stack_bytes, which its procedures that recurse other than by
// tail calls check, with what their calls may take below them (Lowering.cpp,
// AddDeviceStackChecks), as the device runtime's copy of a message does. Or
// gives the text of the error, for a kernel whose calls outside any
// recursion take more than the stack, which nothing checks. A kernel whose
// stack is fixed has none to check.
const char *SetStackLimit(const char *module, const Gpu &gpu,
    ol_program_handle_t program, const ProcessImage &image) {
  if (image.frames == 0) {
    return nullptr;
  }
  const char *error{nullptr};
  std::uint64_t stack{ProcessStackBytes(error)};
  if (!stack) {
    return error;
  }
  if (image.frames > stack) {
    char *text{static_cast<char *>(Malloc(512))};
    std::snprintf(text, 512,
        "the module %s's processes take %llu bytes of stack a lane for their "
        "calls on GPU %d, and GPU processes have %llu: a larger "
        "MERK_GPU_PROCESS_STACK_BYTES gives them more, which may keep some "
        "of their kernel's teams from starting",
        module, static_cast<unsigned long long>(image.frames), gpu.number,
        static_cast<unsigned long long>(stack));
    return text;
  }
  ol_symbol_handle_t symbol;
  void *address{nullptr};
  std::uint32_t limit{
      stack > 0xffffffffu ? 0xffffffffu : static_cast<std::uint32_t>(stack)};
  if (olGetSymbol(program, "merk_device_stack_bytes",
          kOlSymbolKindGlobalVariable, &symbol) ||
      olGetSymbolInfo(symbol, kOlSymbolInfoGlobalVariableAddress,
          sizeof address, &address) ||
      olMemcpy(nullptr, address, gpu.device, &limit, nullptr, sizeof limit)) {
    return Error("could not set the stack's limit in the module %s's image "
                 "on GPU %lld%s%s%s",
        module, gpu.number, "", "", "");
  }
  return nullptr;
}

const char *Load(const char *module, const Gpu &gpu, LoadedImage *&entry) {
  for (entry = loaded; entry; entry = entry->next) {
    if (entry->gpu == gpu.number && std::strcmp(entry->module, module) == 0) {
      return nullptr;
    }
  }
  const ProcessImage *image{FindProcessImage(module, gpu.name)};
  if (!image) {
    return Error("the module %s has no image of its GPU process kinds for "
                 "GPU %lld, a %s: it must be compiled with --offload-arch=%s%s",
        module, gpu.number, gpu.name, gpu.name, "");
  }
  ol_program_handle_t program;
  if (ol_result_t result{olCreateProgram(gpu.context, gpu.device, image->image,
          image->bytes, &program)}) {
    return Error("could not load the module %s's image on GPU %lld: %s%s%s",
        module, gpu.number, Details(result), "", "");
  }
  // The device's number, which its opcodes carry.
  ol_symbol_handle_t symbol;
  void *address{nullptr};
  std::int32_t number{gpu.number};
  if (olGetSymbol(program, "merk_device_number", kOlSymbolKindGlobalVariable,
          &symbol) ||
      olGetSymbolInfo(symbol, kOlSymbolInfoGlobalVariableAddress,
          sizeof address, &address) ||
      olMemcpy(nullptr, address, gpu.device, &number, nullptr, sizeof number)) {
    return Error("could not set the device's number in the module %s's image "
                 "on GPU %lld%s%s%s",
        module, gpu.number, "", "", "");
  }
  if (const char *error{SetStackLimit(module, gpu, program, *image)}) {
    olDestroyProgram(program);
    return error;
  }
  entry = static_cast<LoadedImage *>(Malloc(sizeof(LoadedImage)));
  *entry = {module, gpu.number, program, image, loaded};
  loaded = entry;
  return nullptr;
}

// The GPU's compute units for processes, found the first time, or the text
// of the error. With the lock held.
const char *Ready(const Gpu &gpu) {
  GpuPlaces &places{gpuPlaces[gpu.number]};
  if (places.ready) {
    return nullptr;
  }
  std::uint32_t count{0};
  std::uint32_t mask[kUnitMaskWords + 1]{};
  if (ol_result_t result{olGetDeviceInfo(
          gpu.device, kOlDeviceInfoNumComputeUnits, sizeof count, &count)}) {
    return Error("could not count GPU %s%lld's compute units: %s%s%s", "",
        gpu.number, Details(result), "", "");
  }
  if (count > 32 * kUnitMaskWords) {
    return Error("GPU %s%lld has more than %s256 compute units, which "
                 "processes cannot share%s%s",
        "", gpu.number, "", "", "");
  }
  if (ol_result_t result{
          olGetDeviceComputeUnitMask(gpu.device, kUnitMaskWords + 1, mask)}) {
    return Error("could not find the compute units of GPU %s%lld that "
                 "processes may use: %s%s%s",
        "", gpu.number, Details(result), "", "");
  }
  if (!ProcessUnits(mask, kUnitMaskWords + 1, places.units)) {
    return Error("GPU %s%lld has no pair of compute units for processes, "
                 "units 2N and 2N+1%s%s%s",
        "", gpu.number, "", "", "");
  }
  UnitMask usable{};
  std::memcpy(usable.words, mask, sizeof usable.words);
  places.usable = UnitCount(usable);
  places.ready = true;
  return nullptr;
}

// The size from which a GPU process's worker teams fill and copy its memory
// for it (device.cpp, Bulk): MERK_GPU_BULK_BYTES, or where crossing their
// barrier twice costs less than the process's one lane doing it. On gfx9
// (gfx90a, measured 2026-11-14) a lane fills about 16 ns a word and reads
// host memory uncached, a word a round trip, so from 2 KiB; on gfx1030 it
// filled at about 3 GB/s, so from 64 KiB.
std::uint64_t BulkBytes(const Gpu &gpu) {
  if (const char *text{std::getenv("MERK_GPU_BULK_BYTES")}) {
    char *end{nullptr};
    unsigned long long bytes{std::strtoull(text, &end, 10)};
    if (end != text && *end == '\0') {
      return bytes;
    }
  }
  if (std::strncmp(gpu.name, "gfx9", 4) == 0) {
    return std::uint64_t{2} << 10;
  }
  return std::uint64_t{64} << 10;
}

// The percentage of the GPU's units that its processes' kernels take: what
// the program set, or MERK_GPU_PROCESS_UNITS, or 100; or 0, with the text of
// the error.
int UnitsPercent(const GpuPlaces &places, const char *&error) {
  if (places.percent != 0) {
    return places.percent;
  }
  const char *text{std::getenv("MERK_GPU_PROCESS_UNITS")};
  if (!text) {
    return 100;
  }
  char *end{nullptr};
  long percent{std::strtol(text, &end, 10)};
  if (end == text || *end != '\0' || percent < 1 || percent > 100) {
    char *message{static_cast<char *>(Malloc(256))};
    std::snprintf(message, 256,
        "MERK_GPU_PROCESS_UNITS is not a percentage from 1 to 100: %s", text);
    error = message;
    return 0;
  }
  return static_cast<int>(percent);
}

// The queue for the GPU's next processes' kernel, limited to the share of
// its units that the kernel takes: the GPU's latest queue if its units are
// the same, or else a new one; and whether the units are all those of the
// GPU's queues. Or the text of the error. With the lock held.
const char *KernelQueue(
    const Gpu &gpu, ol_queue_handle_t &queue, bool &allUnits) {
  GpuPlaces &places{gpuPlaces[gpu.number]};
  const char *error{nullptr};
  int percent{UnitsPercent(places, error)};
  if (!percent) {
    return error;
  }
  UnitMask units{places.units};
  KeepShare(units, percent);
  allUnits = UnitCount(units) == places.usable;
  if (!places.queue ||
      std::memcmp(units.words, places.queueUnits.words, sizeof units.words)) {
    ol_queue_handle_t made;
    if (ol_result_t result{olCreateQueueWithComputeUnitMask(
            gpu.context, gpu.device, units.words, kUnitMaskWords, &made)}) {
      return Error("could not make a queue for GPU %s%lld's processes: "
                   "%s%s%s",
          "", gpu.number, Details(result), "", "");
    }
    // The old queue goes now if no kernel that has not been freed ran on
    // it, or else when the reaper frees its last.
    if (places.queue && !QueueUsed(places.queue)) {
      DestroyQueue(places.queue);
    }
    places.queue = made;
    places.queueUnits = units;
  }
  queue = places.queue;
  return nullptr;
}

std::uint64_t MonotonicMilliseconds() {
  timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);
  return static_cast<std::uint64_t>(now.tv_sec) * 1000 +
      static_cast<std::uint64_t>(now.tv_nsec) / 1000000;
}

// Waits for the teams of the kernel to start, as a team's slots wait for
// processes only once it runs, and teams that cannot all be resident at once
// never do: the GPU holds a team back until another ends, which none does.
// The plugin counts how many fit by their registers and local memory, but
// not by their scratch, which the GPU may not hold for all of them. Each
// team that starts restarts the wait; whether they all started.
bool TeamsStarted(const PlaceTable *table, std::uint64_t teams) {
  std::uint64_t stall{TeamsStallMilliseconds()};
  std::uint64_t seen{0};
  std::uint64_t deadline{MonotonicMilliseconds() + stall};
  for (;;) {
    std::uint64_t started{__atomic_load_n(&table->started, __ATOMIC_ACQUIRE)};
    if (started >= teams) {
      return true;
    }
    if (started != seen) {
      seen = started;
      deadline = MonotonicMilliseconds() + stall;
    } else if (MonotonicMilliseconds() >= deadline) {
      return false;
    }
    timespec pause{0, 100000};
    nanosleep(&pause, nullptr);
  }
}

// The bytes of stack for each thread of a processes' kernel whose stack is
// not fixed, as one whose code recurses: the plugin's own, 16 KiB, has
// scratch that the GPU does not hold for all the kernel's teams, and with
// 1 KiB it did, as it did up to 1792 bytes beside another queue's scratch
// (docs/INPROGRESS.md, GPU processes that run loops). 1 KiB left no room for
// a message's copy below the fixed frames of process-gpu-messages' kernel
// (The stack is checked). MERK_GPU_PROCESS_STACK_BYTES replaces it. A kernel
// with a fixed stack size has its own. Or 0, with the text of the error.
constexpr std::uint64_t kProcessStackBytes{1536};

std::uint64_t ProcessStackBytes(const char *&error) {
  const char *given{std::getenv("MERK_GPU_PROCESS_STACK_BYTES")};
  if (!given || !*given) {
    return kProcessStackBytes;
  }
  char *end{nullptr};
  unsigned long long bytes{std::strtoull(given, &end, 10)};
  if (*end != '\0' || bytes == 0) {
    char *text{static_cast<char *>(Malloc(512))};
    std::snprintf(text, 512,
        "MERK_GPU_PROCESS_STACK_BYTES is not a positive number of bytes: %s",
        given);
    error = text;
    return 0;
  }
  return bytes;
}

// Launches the module's kernel on the GPU, with a slot for each of its
// waves, and makes it the kernel that spawns use once its teams have all
// started; or gives the text of the error. With the lock held, which it
// lets go while it waits for the teams.
const char *Launch(const Gpu &gpu, const LoadedImage &image) {
  GpuPlaces &gpuPlace{gpuPlaces[gpu.number]};
  ol_queue_handle_t queue;
  bool allUnits{false};
  if (const char *error{KernelQueue(gpu, queue, allUnits)}) {
    return error;
  }
  ol_symbol_handle_t kernel;
  if (olGetSymbol(image.program, "merk_place", kOlSymbolKindKernel, &kernel)) {
    return Error("the module %s's image for GPU %lld has no kernel "
                 "merk_place%s%s%s",
        image.module, gpu.number, "", "", "");
  }
  ol_kernel_launch_size_args_t size{1, {1, 1, 1}, {kPlaceTeamThreads, 1, 1}, 0};
  std::uint32_t teams{0};
  std::uint32_t lanes{0};
  if (ol_result_t result{olGetKernelMaxCooperativeGroupCountOnQueue(
          queue, kernel, &size, &teams)}) {
    return Error("could not count the teams of the module %s's processes "
                 "that GPU %lld holds at once: %s%s%s",
        image.module, gpu.number, Details(result), "", "");
  }
  if (ol_result_t result{olGetDeviceInfo(
          gpu.device, kOlDeviceInfoNumLanes, sizeof lanes, &lanes)}) {
    return Error("could not ask for GPU %s%lld's wave size: %s%s%s", "",
        gpu.number, Details(result), "", "");
  }
  if (teams == 0 || lanes == 0 || kPlaceTeamThreads % lanes != 0) {
    return Error("GPU %s%lld cannot run the module %s's processes: their "
                 "kernel has no room on its compute units%s%s",
        "", gpu.number, image.module, "", "");
  }
  std::uint64_t slots{
      static_cast<std::uint64_t>(teams) * (kPlaceTeamThreads / lanes)};
  std::size_t bytes{sizeof(PlaceTable) + slots * sizeof(PlaceSlot)};
  auto *table{static_cast<PlaceTable *>(DeviceAllocateHost(gpu.number, bytes))};
  if (!table) {
    return Error("out of host memory for the table of GPU %s%lld's "
                 "processes%s%s%s",
        "", gpu.number, "", "", "");
  }
  std::memset(static_cast<void *>(table), 0, bytes);
  table->slots = slots;
  // The device's memory, before the kernel runs, so that no line of the
  // device's caches is stale; each wave makes its own queue empty.
  void *queues{DeviceAllocate(gpu.number, slots * sizeof(ProcessQueue))};
  if (!queues) {
    DeviceFreeHost(gpu.number, table);
    return Error("out of memory on GPU %s%lld for the queues of its "
                 "processes%s%s%s",
        "", gpu.number, "", "", "");
  }
  table->queues = reinterpret_cast<std::uint64_t>(queues);
  table->check = CheckingDeviceMemory();
  table->bulkBytes = BulkBytes(gpu);
  PeerBlock *peerBlock{TakePeerBlock(gpu, slots)};
  if (peerBlock) {
    const GpuPeer &peer{gpuPeers[gpu.number]};
    table->peerQueues = reinterpret_cast<std::uint64_t>(peerBlock->queues);
    table->sendPool = reinterpret_cast<std::uint64_t>(peer.sendPool);
    table->sendPoolBytes = peer.sendPoolBytes;
    table->returns = reinterpret_cast<std::uint64_t>(peer.returns);
  }
  auto *place{static_cast<Place *>(Malloc(sizeof(Place)))};
  auto *processes{
      static_cast<RunningProcess **>(Malloc(slots * sizeof(RunningProcess *)))};
  std::memset(
      static_cast<void *>(processes), 0, slots * sizeof(RunningProcess *));
  auto *owners{
      static_cast<std::uint64_t *>(Malloc(slots * sizeof(std::uint64_t)))};
  std::memset(static_cast<void *>(owners), 0, slots * sizeof(std::uint64_t));
  *place = {image.module, gpu.number, queue, allUnits, table, queues,
      peerBlock, processes, owners, kPlaceTeamThreads / lanes, 0, false,
      places};
  size.numGroups.x = teams;
  bool cooperative{true};
  const char *stackError{nullptr};
  std::uint64_t stack{ProcessStackBytes(stackError)};
  if (!stack) {
    DeviceFreeHost(gpu.number, table);
    DeviceFree(gpu.number, queues);
    ReturnPeerBlock(gpu.number, peerBlock);
    std::free(processes);
    std::free(owners);
    std::free(place);
    return stackError;
  }
  ol_kernel_launch_prop_t properties[]{
      {kOlKernelLaunchPropTypeIsCooperative, &cooperative},
      {kOlKernelLaunchPropTypeStackSize, &stack},
      {kOlKernelLaunchPropTypeNone, nullptr}};
  void *argument{table};
  void *arguments[]{&argument};
  std::size_t sizes[]{sizeof argument};
  if (ol_result_t result{olLaunchKernel(
          queue, gpu.device, kernel, &size, properties, 1, arguments, sizes)}) {
    DeviceFreeHost(gpu.number, table);
    DeviceFree(gpu.number, queues);
    ReturnPeerBlock(gpu.number, peerBlock);
    std::free(processes);
    std::free(owners);
    std::free(place);
    return Error("could not launch the module %s's processes' kernel on GPU "
                 "%lld: %s%s%s",
        image.module, gpu.number, Details(result), "", "");
  }
  if (ol_result_t queued{olLaunchHostFunction(queue, PlaceDone, place)}) {
    std::fflush(stdout);
    std::fprintf(stderr,
        "Merkurius runtime: could not follow a GPU processes' kernel: %s\n",
        Details(queued));
    ExitProgram(1);
  }
  places = place;
  // The wait is without the lock: a kernel launched while the GPU's last
  // one ends starts after the host function that follows that one on the
  // queue (PlaceDone), which takes the lock.
  gpuPlace.launching = true;
  pthread_mutex_unlock(&lock);
  bool started{TeamsStarted(table, teams)};
  pthread_mutex_lock(&lock);
  gpuPlace.launching = false;
  pthread_cond_broadcast(&launchedCondition);
  if (!started) {
    // Its waves that run return, after which the others start and return
    // too, and the reaper frees the table.
    EndPlace(place);
    std::uint64_t started{__atomic_load_n(&table->started, __ATOMIC_ACQUIRE)};
    char *text{static_cast<char *>(Malloc(512))};
    std::snprintf(text, 512,
        "GPU %d started %llu of the %u teams of the module %s's processes' "
        "kernel, which must all run at once, and no other within %llu ms "
        "(MERK_TEAMS_STALL_MILLISECONDS): the GPU may not hold their scratch "
        "for all of them, as for a kernel with a dynamic stack",
        gpu.number, static_cast<unsigned long long>(started), teams,
        image.module,
        static_cast<unsigned long long>(TeamsStallMilliseconds()));
    return text;
  }
  gpuPlace.current = place;
  return nullptr;
}

// Whether no process runs in the slot and no process has it as a worker's.
bool Free(const Place &place, std::uint64_t slot) {
  return !place.processes[slot] && place.owners[slot] == 0;
}

// Reserves as worker teams for the process in `slot` the teams of the kernel
// whose slots are all free, other than the process's own: every one, or with
// a share of K, the first of them up to a K'th of the kernel's teams, at
// least one; and gives how many (place.h). With the lock held.
std::uint64_t ReserveWorkers(
    Place &place, std::uint64_t slot, std::int64_t share) {
  std::uint64_t teams{place.table->slots / place.teamSlots};
  std::uint64_t own{slot / place.teamSlots};
  std::uint64_t most{teams};
  if (share > 0) {
    most = teams / static_cast<std::uint64_t>(share);
    most = most < 1 ? 1 : most;
  }
  std::uint64_t workers{0};
  for (std::uint64_t team{0}; team < teams && workers < most; ++team) {
    std::uint64_t first{team * place.teamSlots};
    bool free{team != own};
    for (std::uint64_t i{first}; free && i < first + place.teamSlots; ++i) {
      free = Free(place, i);
    }
    if (!free) {
      continue;
    }
    for (std::uint64_t i{first}; i < first + place.teamSlots; ++i) {
      place.owners[i] = slot + 1;
    }
    ++workers;
  }
  std::uint64_t worker{0};
  for (std::uint64_t i{0}; i < place.table->slots; ++i) {
    if (place.owners[i] != slot + 1) {
      continue;
    }
    PlaceSlot &reserved{PlaceSlots(place.table)[i]};
    reserved.owner = slot;
    reserved.worker = worker;
    reserved.workers = workers;
    reserved.control = 0;
    __atomic_store_n(&reserved.state, kPlaceReserved, __ATOMIC_RELEASE);
    if ((i + 1) % place.teamSlots == 0) {
      ++worker;
    }
  }
  return workers;
}

// The kernel and a free slot of it for a process of the module's kind on
// the GPU, which it takes for `process`, with the free teams as its workers
// if the kind runs loops across teams, or the text of the error. With the
// lock held.
const char *TakeSlot(const Gpu &gpu, const LoadedImage &image, const char *kind,
    bool loops, RunningProcess *process, std::uint64_t &slot,
    std::uint64_t &workers) {
  GpuPlaces &gpuPlace{gpuPlaces[gpu.number]};
  while (gpuPlace.launching) {
    pthread_cond_wait(&launchedCondition, &lock);
  }
  Place *place{gpuPlace.current};
  if (place && !place->ending && std::strcmp(place->module, image.module)) {
    return Error("GPU %s%lld runs processes of the module %s, and a GPU runs "
                 "one module's processes at a time: those of %s may start "
                 "there once they have ended%s",
        "", gpu.number, place->module, image.module, "");
  }
  if (!place || place->ending) {
    if (const char *error{Launch(gpu, image)}) {
      return error;
    }
    place = gpuPlace.current;
  }
  workers = 0;
  for (slot = 0; slot < place->table->slots; ++slot) {
    if (Free(*place, slot)) {
      if (loops) {
        workers = ReserveWorkers(*place, slot, gpuPlace.share);
        if (workers == 0) {
          char *text{static_cast<char *>(Malloc(512))};
          std::snprintf(text, 512,
              "GPU %d has no team for the loops of the process kind %s: the "
              "module %s's kernel there has %llu teams, and none is free of "
              "processes, which a kind that runs loops across teams takes as "
              "its workers",
              gpu.number, kind, image.module,
              static_cast<unsigned long long>(
                  place->table->slots / place->teamSlots));
          return text;
        }
      }
      place->processes[slot] = process;
      process->place = place;
      process->slot = slot;
      process->incarnation = ++processIncarnations;
      process->queue = reinterpret_cast<std::uint64_t>(place->queues) +
          slot * sizeof(ProcessQueue);
      process->peerQueue = place->peerBlock
          ? reinterpret_cast<std::uint64_t>(place->peerBlock->queues) +
              slot * sizeof(PeerQueue)
          : 0;
      ++place->running;
      return nullptr;
    }
  }
  char *text{static_cast<char *>(Malloc(512))};
  std::snprintf(text, 512,
      "GPU %d has no room for another process: the module %s's kernel there "
      "runs %llu processes at once, and they are all running",
      gpu.number, image.module,
      static_cast<unsigned long long>(place->table->slots));
  return text;
}

} // namespace

namespace merkurius::runtime {

// On the plugin's thread, when the device tells the host that the process
// of the slot has ended (rpcProcessEnded): its mailbox closes, the slot is
// free for another, and the reaper frees the rest.
void PlaceProcessEnded(int, std::uint64_t table, std::uint64_t slot) {
  RunningProcess *process{nullptr};
  pthread_mutex_lock(&lock);
  for (Place *place{places}; place; place = place->next) {
    if (reinterpret_cast<std::uint64_t>(place->table) == table &&
        slot < place->table->slots) {
      process = place->processes[slot];
      place->processes[slot] = nullptr;
      // Its worker teams, which have set their slots reserved again before
      // it ended, are free for others.
      for (std::uint64_t i{0}; i < place->table->slots; ++i) {
        if (place->owners[i] == slot + 1) {
          place->owners[i] = 0;
          __atomic_store_n(
              &PlaceSlots(place->table)[i].state, kPlaceIdle, __ATOMIC_RELEASE);
        }
      }
      if (process && --place->running == 0 && !place->ending) {
        EndPlace(place);
      }
      break;
    }
  }
  pthread_mutex_unlock(&lock);
  if (!process) {
    std::fflush(stdout);
    std::fputs("Merkurius runtime: a GPU told the host that a process ended "
               "that it does not know\n",
        stderr);
    std::_Exit(1);
  }
  // No sender is given its queue from now on: a send to its handle goes
  // through the host, to its mailbox, which closes.
  pthread_mutex_lock(&routesLock);
  for (RunningProcess **link{&routes}; *link; link = &(*link)->nextRoute) {
    if (*link == process) {
      *link = process->nextRoute;
      break;
    }
  }
  pthread_mutex_unlock(&routesLock);
  const std::int64_t *h{process->handle};
  merk_mailbox_close_ended(h[0], h[1], h[2], h[3], h[4], &process->ring,
      &process->ringDevice, &process->copy, &process->release);
  FireMonitorsOf(process->record, nullptr);
  std::free(process->record);
  process->record = nullptr;
  pthread_mutex_lock(&lock);
  process->next = ended;
  ended = process;
  pthread_cond_signal(&endedCondition);
  pthread_mutex_unlock(&lock);
}

// With MERK_CHECK_DEVICE_MEMORY, once no GPU process runs: the blocks of
// messages between GPU processes still allocated, in kernels that have
// ended and those that run.
// The messages that GPU processes put in peer queues, and the copies in send
// pools not given back, of every kernel (PlaceTable), with
// MERK_CHECK_DEVICE_MEMORY.
// The GPU whose return ring is `returns`, for a copy out of its send pool
// (rpcCopyPeer); -1 if no GPU's is. On the plugin's RPC server, without
// the lock, which a spawn may hold while it waits for the server: a GPU's
// ring is set once, before any kernel can know it.
int PeerGpuOfReturns(std::uint64_t returns) {
  for (int i{0}; i < kMostGpus; ++i) {
    void *ring{__atomic_load_n(&gpuPeers[i].returns, __ATOMIC_ACQUIRE)};
    if (ring && reinterpret_cast<std::uint64_t>(ring) == returns) {
      return i;
    }
  }
  return -1;
}

void PlacePeerCounts(std::int64_t *sends, std::int64_t *copies) {
  pthread_mutex_lock(&lock);
  *sends = __atomic_load_n(&endedPeerSends, __ATOMIC_RELAXED);
  *copies = __atomic_load_n(&endedPeerCopies, __ATOMIC_RELAXED);
  const Place *lists[]{places, endedPlaces};
  for (const Place *list : lists) {
    for (const Place *place{list}; place; place = place->next) {
      *sends += __atomic_load_n(&place->table->peerSends, __ATOMIC_ACQUIRE);
      *copies += __atomic_load_n(&place->table->peerCopies, __ATOMIC_ACQUIRE);
    }
  }
  pthread_mutex_unlock(&lock);
}

std::int64_t PlaceMessageBlocks() {
  std::int64_t blocks{__atomic_load_n(&endedPlaceBlocks, __ATOMIC_RELAXED)};
  pthread_mutex_lock(&lock);
  for (const Place *place{places}; place; place = place->next) {
    blocks += __atomic_load_n(&place->table->blocks, __ATOMIC_ACQUIRE);
  }
  for (const Place *place{endedPlaces}; place; place = place->next) {
    blocks += __atomic_load_n(&place->table->blocks, __ATOMIC_ACQUIRE);
  }
  pthread_mutex_unlock(&lock);
  return blocks;
}

// Whether kernels on GPUs `a` and `b` of this node can each reach the
// other's memory, so that a process on either sends directly to one on the
// other (olGetDevicePeerAccess), asked once a pair: 0 not asked, 1 yes, 2
// no. Under routesLock.
std::uint8_t peerPairs[kMostGpus][kMostGpus];

bool PeerPair(int a, int b) {
  if (a < 0 || b < 0 || a >= kMostGpus || b >= kMostGpus || a == b) {
    return false;
  }
  std::uint8_t &known{peerPairs[a < b ? a : b][a < b ? b : a]};
  if (known == 0) {
    const Gpu *ga{GpuOf(a)};
    const Gpu *gb{GpuOf(b)};
    bool ab{false};
    bool ba{false};
    bool both{ga && gb && olGetDevicePeerAccess &&
        olGetDevicePeerAccess(ga->device, gb->device, &ab) == nullptr &&
        olGetDevicePeerAccess(gb->device, ga->device, &ba) == nullptr &&
        ab && ba};
    known = both ? 1 : 2;
  }
  return known == 1;
}

// The route from a process of the kernel whose table is `table`, on the
// device, to the mailbox of the handle (rpcRoute):
// - the queue of the mailbox's process, if it is a running process of the
//   same kernel, with its incarnation in `incarnation`;
// - or else, with the peer path (MERK_GPU_PEER=1), if the process runs on
//   another GPU of this node whose memory and the sender's GPU's each reach
//   the other's, and both GPUs have their peer memory, 0 and the process's
//   peer queue in `peerQueue`, with its incarnation (place.h, PeerQueue);
// - or 0, for a send through the host.
// A process's mailbox is made at its spawn, so a sender that asks after it
// has the handle finds it, and keeps one route for its life.
std::uint64_t PlaceRoute(const std::uint64_t *handle, int device,
    std::uint64_t table, std::uint64_t *incarnation,
    std::uint64_t *peerQueue) {
  std::uint64_t queue{0};
  *incarnation = 0;
  *peerQueue = 0;
  pthread_mutex_lock(&routesLock);
  for (const RunningProcess *process{routes}; process;
      process = process->nextRoute) {
    bool same{true};
    for (int i{0}; i < kHandleWords; ++i) {
      same =
          same && static_cast<std::uint64_t>(process->handle[i]) == handle[i];
    }
    if (!same) {
      continue;
    }
    if (process->gpu == device &&
        reinterpret_cast<std::uint64_t>(process->place->table) == table) {
      queue = process->queue;
      *incarnation = process->incarnation;
    } else if (process->peerQueue != 0 && process->gpu != device &&
        reinterpret_cast<const PlaceTable *>(table)->peerQueues != 0 &&
        PeerPair(process->gpu, device)) {
      *peerQueue = process->peerQueue;
      *incarnation = process->incarnation;
    }
    break;
  }
  pthread_mutex_unlock(&routesLock);
  return queue;
}

} // namespace merkurius::runtime

// How many GPUs this node has for processes (process.gpus).
extern "C" std::int64_t merk_gpu_count(void) { return GpuCount(); }

// process.set_gpu_units: the percentage of the GPU's units, those its queues
// may use, that its next processes' kernels take, rounded up to a pair of
// units. It changes only between kernels: while the GPU's kernel runs
// processes, it is the text of the error.
extern "C" const char *merk_gpu_set_units(
    std::int64_t number, std::int64_t percent) {
  if (number < 0 || number >= GpuCount()) {
    char *text{static_cast<char *>(Malloc(128))};
    std::snprintf(text, 128, "there is no GPU %lld for processes",
        static_cast<long long>(number));
    return text;
  }
  if (percent < 1 || percent > 100) {
    char *text{static_cast<char *>(Malloc(128))};
    std::snprintf(text, 128, "%lld is not a percentage from 1 to 100",
        static_cast<long long>(percent));
    return text;
  }
  pthread_mutex_lock(&lock);
  GpuPlaces &places{gpuPlaces[number]};
  while (places.launching) {
    pthread_cond_wait(&launchedCondition, &lock);
  }
  if (places.current && !places.current->ending) {
    char *text{static_cast<char *>(Malloc(256))};
    std::snprintf(text, 256,
        "GPU %lld runs processes of the module %s: the units of its "
        "processes' kernels change between kernels, once its processes have "
        "all ended",
        static_cast<long long>(number), places.current->module);
    pthread_mutex_unlock(&lock);
    return text;
  }
  places.percent = static_cast<int>(percent);
  pthread_mutex_unlock(&lock);
  return nullptr;
}

// process.set_worker_teams: the share of a kernel's teams that each kind
// with loops spawned on the GPU from now on takes as its workers, K for a
// K'th, or 0 for every free team. It applies to later spawns, and may change
// at any time; a negative share is the text of the error.
extern "C" const char *merk_gpu_set_worker_teams(
    std::int64_t number, std::int64_t share) {
  if (number < 0 || number >= GpuCount()) {
    char *text{static_cast<char *>(Malloc(128))};
    std::snprintf(text, 128, "there is no GPU %lld for processes",
        static_cast<long long>(number));
    return text;
  }
  if (share < 0) {
    char *text{static_cast<char *>(Malloc(128))};
    std::snprintf(text, 128, "a share of %lld is not one of a positive number",
        static_cast<long long>(share));
    return text;
  }
  pthread_mutex_lock(&lock);
  gpuPlaces[number].share = share;
  pthread_mutex_unlock(&lock);
  return nullptr;
}

// Before a target region on the device, -1 for OpenMP's default: one on a
// GPU whose processes' kernel holds every unit that its queues may use would
// never start, as the kernel holds every team slot of its units until its
// processes end, so it is a software error.
extern "C" void merk_check_region_units(std::int64_t device) {
  if (device < 0 && omp_get_default_device) {
    device = omp_get_default_device();
  }
  if (device < 0 || device >= kMostGpus) {
    return;
  }
  pthread_mutex_lock(&lock);
  const GpuPlaces &places{gpuPlaces[device]};
  bool held{
      places.current && !places.current->ending && places.current->allUnits};
  pthread_mutex_unlock(&lock);
  if (!held) {
    return;
  }
  char text[512];
  std::snprintf(text, sizeof text,
      "Software Error: target region: the GPU processes' kernel on GPU %lld "
      "holds every compute unit that its queues may use, so the region would "
      "never start: process.set_gpu_units, between kernels, or "
      "MERK_GPU_PROCESS_UNITS leaves it a share of them",
      static_cast<long long>(device));
  merk_uncaught_exception(text);
}

// Spawns a process of the module's kind on the GPU, with the words of its
// arguments, an array (array.cpp: its size, then its elements), and a new
// mailbox for messages of the type, whose handle it gives: null, or the
// text of the error, for process.spawn to report.
extern "C" const char *merk_gpu_process_spawn(std::uint64_t typeInfo,
    std::int64_t number, const char *module, const char *kind,
    const std::uint64_t *words, std::int64_t *placeAndSlot,
    std::int64_t *generation, std::int64_t *secret) {
  pthread_once(&reaperOnce, StartReaper);
  const Gpu *gpu{GpuOf(static_cast<int>(number))};
  if (!gpu) {
    return Error("there is no GPU %s%lld: this node has %s%s%s", "", number,
        GpuCount() == 0 ? "none for processes"
            : GpuCount() == 1 ? "one, GPU 0"
                              : "several, from GPU 0",
        "", "");
  }
  if (gpu->number >= kMostGpus) {
    return Error("GPU %s%lld is beyond the %s64 GPUs that the runtime runs "
                 "processes on%s%s",
        "", number, "", "", "");
  }
  // The process's argument words, in host memory that the device reads:
  // the kind's arguments and then its mailbox's handle.
  std::size_t count{static_cast<std::size_t>(words[0]) + kHandleWords};
  auto *process{static_cast<RunningProcess *>(Malloc(sizeof(RunningProcess)))};
  *process = {};
  process->gpu = gpu->number;
  process->words = static_cast<std::uint64_t *>(
      DeviceAllocateHost(gpu->number, count * sizeof(std::uint64_t)));
  process->arena = NewProcessArena(gpu->number);
  process->record = NewGpuProcessRecord();
  if (!process->words || !process->arena || !process->record) {
    if (process->words) {
      DeviceFreeHost(gpu->number, process->words);
    }
    if (process->arena) {
      FreeProcessArena(gpu->number, process->arena);
    }
    std::free(process->record);
    std::free(process);
    return Error("out of memory for a process on GPU %s%lld%s%s%s", "", number,
        "", "", "");
  }
  std::uint64_t slot{0};
  std::uint64_t workers{0};
  int kindNumber{-1};
  LoadedImage *image{nullptr};
  pthread_mutex_lock(&loadLock);
  const char *error{Load(module, *gpu, image)};
  pthread_mutex_unlock(&loadLock);
  pthread_mutex_lock(&lock);
  if (!error) {
    kindNumber = KindNumber(*image->image, kind);
    if (kindNumber < 0) {
      error = Error("the module %s's image for GPU %lld has no process kind "
                    "%s%s%s",
          module, number, kind, "", "");
    }
  }
  if (!error) {
    error = Ready(*gpu);
  }
  if (!error) {
    error = TakeSlot(*gpu, *image, kind,
        KindRunsLoops(*image->image, kindNumber), process, slot, workers);
  }
  pthread_mutex_unlock(&lock);
  if (error) {
    DeviceFreeHost(gpu->number, process->words);
    FreeProcessArena(gpu->number, process->arena);
    std::free(process->record);
    std::free(process);
    return error;
  }
  std::int64_t *h{process->handle};
  merk_node(&h[0], &h[1]);
  NewMailboxOf(process->record, typeInfo, &h[2], &h[3], &h[4]);
  for (std::size_t i{0}; i < count; ++i) {
    process->words[i] = i < words[0]
        ? words[i + 1]
        : static_cast<std::uint64_t>(h[i - words[0]]);
  }
  GpuProcessStarted();
  PlaceSlot &placeSlot{PlaceSlots(process->place->table)[slot]};
  placeSlot.kind = static_cast<std::uint64_t>(kindNumber);
  placeSlot.arena = reinterpret_cast<std::uint64_t>(process->arena);
  placeSlot.words = reinterpret_cast<std::uint64_t>(process->words);
  placeSlot.workers = workers;
  placeSlot.incarnation = process->incarnation;
  pthread_mutex_lock(&routesLock);
  process->nextRoute = routes;
  routes = process;
  pthread_mutex_unlock(&routesLock);
  __atomic_store_n(&placeSlot.state, kPlaceRunning, __ATOMIC_RELEASE);
  *placeAndSlot = h[2];
  *generation = h[3];
  *secret = h[4];
  return nullptr;
}
