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
// queue limited to them that the GPU keeps for the program's life. A spawn
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
#include <pthread.h>

extern "C" void merk_node(std::int64_t *high, std::int64_t *low);
extern "C" void merk_mailbox_close(std::int64_t nodeHigh, std::int64_t nodeLow,
    std::int64_t placeAndSlot, std::int64_t generation, std::int64_t secret);

extern "C" void merk_mailbox_close_ended(std::int64_t nodeHigh,
    std::int64_t nodeLow, std::int64_t placeAndSlot, std::int64_t generation,
    std::int64_t secret, merkurius::runtime::DeviceRing **ring, int *device,
    void **copy, void (**release)(int device, void *copy));

namespace merkurius::runtime {
void GpuProcessStarted();
void GpuProcessEnded();
void *NewProcessArena(int device);
void FreeProcessArena(int device, void *arena);
void FreeRing(int device, DeviceRing *ring);
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
  Process *record;
  std::int64_t handle[kHandleWords];
  DeviceRing *ring;
  int ringDevice;
  void *copy;
  void (*release)(int device, void *copy);
  RunningProcess *next;
};

// A launch of a module's kernel on a GPU: its table, in host memory, the
// process in each slot, and how many processes it runs, of which the last
// ends the kernel. Once the kernel has ended,
// the reaper frees the table.
struct Place {
  const char *module;
  int gpu;
  PlaceTable *table;
  RunningProcess **processes;
  std::uint64_t running;
  bool ending;
  Place *next;
};

// The queue of each GPU's processes' kernels, limited to its units for
// processes (compute-units.h), from its first spawn, and the kernel that
// spawns use, unless it is ending.
struct GpuPlaces {
  bool ready;
  ol_queue_handle_t queue;
  Place *current;
};
GpuPlaces gpuPlaces[kMostGpus];

// Every kernel that has not ended, and those that have, for the reaper.
Place *places{nullptr};
Place *endedPlaces{nullptr};

// The processes that have ended, for the reaper.
pthread_cond_t endedCondition = PTHREAD_COND_INITIALIZER;
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

// Frees an ended kernel's table.
void ReapPlace(Place *place) {
  DeviceFreeHost(place->gpu, place->table);
  std::free(place->processes);
  std::free(place);
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
      ReapPlace(placeList);
      placeList = next;
    }
  }
  return nullptr;
}

void StartReaper() {
  pthread_t thread;
  if (pthread_create(&thread, nullptr, Reaper, nullptr) != 0) {
    std::fflush(stdout);
    std::fputs("Mercury runtime: could not start the thread that frees what "
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
// error. With the lock held.
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
  entry = static_cast<LoadedImage *>(Malloc(sizeof(LoadedImage)));
  *entry = {module, gpu.number, program, image, loaded};
  loaded = entry;
  return nullptr;
}

// The GPU's compute units and its processes' queue, made the first time, or
// the text of the error. With the lock held.
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
  UnitMask units;
  if (!ProcessUnits(mask, kUnitMaskWords + 1, units)) {
    return Error("GPU %s%lld has no pair of compute units for processes, "
                 "units 2N and 2N+1%s%s%s",
        "", gpu.number, "", "", "");
  }
  // Temporary (one kernel for a GPU's processes, docs/INPROGRESS.md, open:
  // the units of the processes' kernel): the kernel takes only the first
  // pair, as it holds every team slot of its units, so that target regions
  // and other kernels keep the rest; which units it takes is to decide.
  FirstPair(units);
  if (ol_result_t result{olCreateQueueWithComputeUnitMask(gpu.context,
          gpu.device, units.words, kUnitMaskWords, &places.queue)}) {
    return Error("could not make a queue for GPU %s%lld's processes: %s%s%s",
        "", gpu.number, Details(result), "", "");
  }
  places.ready = true;
  return nullptr;
}

// Launches the module's kernel on the GPU, with a slot for each of its
// waves, and makes it the kernel that spawns use; or gives the text of the
// error. With the lock held.
const char *Launch(const Gpu &gpu, const LoadedImage &image) {
  GpuPlaces &gpuPlace{gpuPlaces[gpu.number]};
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
          gpuPlace.queue, kernel, &size, &teams)}) {
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
  auto *place{static_cast<Place *>(Malloc(sizeof(Place)))};
  auto *processes{
      static_cast<RunningProcess **>(Malloc(slots * sizeof(RunningProcess *)))};
  std::memset(
      static_cast<void *>(processes), 0, slots * sizeof(RunningProcess *));
  *place = {image.module, gpu.number, table, processes, 0, false, places};
  size.numGroups.x = teams;
  bool cooperative{true};
  ol_kernel_launch_prop_t properties[]{
      {kOlKernelLaunchPropTypeIsCooperative, &cooperative},
      {kOlKernelLaunchPropTypeNone, nullptr}};
  void *argument{table};
  void *arguments[]{&argument};
  std::size_t sizes[]{sizeof argument};
  if (ol_result_t result{olLaunchKernel(gpuPlace.queue, gpu.device, kernel,
          &size, properties, 1, arguments, sizes)}) {
    DeviceFreeHost(gpu.number, table);
    std::free(processes);
    std::free(place);
    return Error("could not launch the module %s's processes' kernel on GPU "
                 "%lld: %s%s%s",
        image.module, gpu.number, Details(result), "", "");
  }
  if (ol_result_t queued{
          olLaunchHostFunction(gpuPlace.queue, PlaceDone, place)}) {
    std::fflush(stdout);
    std::fprintf(stderr,
        "Mercury runtime: could not follow a GPU processes' kernel: %s\n",
        Details(queued));
    ExitProgram(1);
  }
  places = place;
  gpuPlace.current = place;
  return nullptr;
}

// The kernel and a free slot of it for a process of the module's kind on
// the GPU, which it takes for `process`, or the text of the error. With the
// lock held.
const char *TakeSlot(const Gpu &gpu, const LoadedImage &image,
    RunningProcess *process, std::uint64_t &slot) {
  GpuPlaces &gpuPlace{gpuPlaces[gpu.number]};
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
  for (slot = 0; slot < place->table->slots; ++slot) {
    if (!place->processes[slot]) {
      place->processes[slot] = process;
      process->place = place;
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
      if (process && --place->running == 0 && !place->ending) {
        EndPlace(place);
      }
      break;
    }
  }
  pthread_mutex_unlock(&lock);
  if (!process) {
    std::fflush(stdout);
    std::fputs("Mercury runtime: a GPU told the host that a process ended "
               "that it does not know\n",
        stderr);
    std::_Exit(1);
  }
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

} // namespace merkurius::runtime

// How many GPUs this node has for processes (process.gpus).
extern "C" std::int64_t merk_gpu_count(void) { return GpuCount(); }

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
  int kindNumber{-1};
  pthread_mutex_lock(&lock);
  LoadedImage *image{nullptr};
  const char *error{Load(module, *gpu, image)};
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
    error = TakeSlot(*gpu, *image, process, slot);
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
  __atomic_store_n(&placeSlot.state, kPlaceRunning, __ATOMIC_RELEASE);
  *placeAndSlot = h[2];
  *generation = h[3];
  *secret = h[4];
  return nullptr;
}
