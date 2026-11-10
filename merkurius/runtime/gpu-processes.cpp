//===-- runtime/gpu-processes.cpp - Processes that run on a GPU -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Spawning a process on a GPU (docs/INPROGRESS.md, GPU processes without
// omp target, step 4c). A process kind that runs on a GPU has a kernel,
// merk_process_<kind>, in its module's image for each GPU it was compiled
// for (process-images.h). The first spawn of one of a module's kinds on a
// GPU loads the module's image there through liboffload, and sets the
// image's merk_device_number to the GPU's number, which the device's runtime
// sends with its opcodes. Each process is a launch of its kind's kernel,
// one team of one thread, on a queue of its own, so the host does not wait
// for it. Its arguments are words, and the last five are its mailbox's
// handle.
//
// The mailbox is the host's, and no process of the host's owns it: the
// GPU receives from it through a ring, as a region does (offload.cpp). A
// host function queued after the kernel closes it when the kernel ends.
// Temporary (step 5 of GPU processes without omp target,
// docs/INPROGRESS.md): its handle's place is the host's, 0, until handles
// are routed by place.
//
// A running GPU process counts as a running region, so that the devices'
// arenas and rings stay while it runs, and the program ends at once when
// main does (offload.cpp).
//
//===----------------------------------------------------------------------===//

#include "devices.h"
#include "fragment.h"
#include "liboffload.h"
#include "process-images.h"
#include "process.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <pthread.h>

extern "C" void merk_node(std::int64_t *high, std::int64_t *low);
extern "C" void merk_mailbox_new_unowned(std::uint64_t typeInfo,
    std::int64_t *placeAndSlot, std::int64_t *generation,
    std::int64_t *secret);
extern "C" void merk_mailbox_close(std::int64_t nodeHigh, std::int64_t nodeLow,
    std::int64_t placeAndSlot, std::int64_t generation, std::int64_t secret);

namespace merkurius::runtime {
void GpuProcessStarted();
void GpuProcessEnded();
} // namespace merkurius::runtime

namespace {

using namespace merkurius::runtime;

constexpr int kHandleWords{5};

// A module's image, loaded on a GPU.
struct LoadedImage {
  const char *module;
  int gpu;
  ol_program_handle_t program;
  LoadedImage *next;
};

pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
LoadedImage *loaded{nullptr};

// A running process: its queue, and its mailbox's handle, which the host
// function that follows its kernel closes.
struct RunningProcess {
  ol_queue_handle_t queue;
  std::int64_t handle[kHandleWords];
  RunningProcess *next;
};

// The processes whose kernels have ended, whose queues the next spawn
// destroys: not their host functions, which run on the queue.
RunningProcess *ended{nullptr};

// The text of an error, for spawn_<kind>_at to report: it is never freed,
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

void DestroyEnded() {
  pthread_mutex_lock(&lock);
  RunningProcess *list{ended};
  ended = nullptr;
  pthread_mutex_unlock(&lock);
  while (list) {
    RunningProcess *next{list->next};
    olDestroyQueue(list->queue);
    std::free(list);
    list = next;
  }
}

// On liboffload's thread, after the process's kernel.
void Ended(void *data) {
  auto *process{static_cast<RunningProcess *>(data)};
  const std::int64_t *h{process->handle};
  merk_mailbox_close(h[0], h[1], h[2], h[3], h[4]);
  GpuProcessEnded();
  pthread_mutex_lock(&lock);
  process->next = ended;
  ended = process;
  pthread_mutex_unlock(&lock);
}

// The module's image on the GPU, loaded the first time, or the text of the
// error. With the lock held.
const char *Load(const char *module, const Gpu &gpu,
    ol_program_handle_t &program) {
  for (LoadedImage *image{loaded}; image; image = image->next) {
    if (image->gpu == gpu.number && std::strcmp(image->module, module) == 0) {
      program = image->program;
      return nullptr;
    }
  }
  const ProcessImage *image{FindProcessImage(module, gpu.name)};
  if (!image) {
    return Error("the module %s has no image of its GPU process kinds for "
                 "GPU %lld, a %s: it must be compiled with --offload-arch=%s%s",
        module, gpu.number, gpu.name, gpu.name, "");
  }
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
  auto *entry{static_cast<LoadedImage *>(Malloc(sizeof(LoadedImage)))};
  *entry = {module, gpu.number, program, loaded};
  loaded = entry;
  return nullptr;
}

} // namespace

// How many GPUs this node has for processes (process.gpus).
extern "C" std::int64_t merk_gpu_count(void) { return GpuCount(); }

// Spawns a process of the module's kind on the GPU, with the words of its
// arguments, an array (array.cpp: its size, then its elements), and a new
// mailbox for messages of the type, whose handle it gives: null, or the
// text of the error, for spawn_<kind>_at to report.
extern "C" const char *merk_gpu_process_spawn(std::uint64_t typeInfo,
    std::int64_t number, const char *module, const char *kind,
    const std::uint64_t *words, std::int64_t *placeAndSlot,
    std::int64_t *generation, std::int64_t *secret) {
  DestroyEnded();
  const Gpu *gpu{GpuOf(static_cast<int>(number))};
  if (!gpu) {
    return Error("there is no GPU %s%lld: this node has %s%s%s", "", number,
        GpuCount() == 0 ? "none for processes"
            : GpuCount() == 1 ? "one, GPU 0"
                              : "several, from GPU 0",
        "", "");
  }
  ol_program_handle_t program;
  pthread_mutex_lock(&lock);
  const char *error{Load(module, *gpu, program)};
  pthread_mutex_unlock(&lock);
  if (error) {
    return error;
  }
  char name[256];
  std::snprintf(name, sizeof name, "merk_process_%s", kind);
  ol_symbol_handle_t kernel;
  if (olGetSymbol(program, name, kOlSymbolKindKernel, &kernel)) {
    return Error("the module %s's image for GPU %lld has no kernel for the "
                 "process kind %s%s%s",
        module, number, kind, "", "");
  }
  auto *process{static_cast<RunningProcess *>(Malloc(sizeof(RunningProcess)))};
  std::int64_t *h{process->handle};
  merk_node(&h[0], &h[1]);
  merk_mailbox_new_unowned(typeInfo, &h[2], &h[3], &h[4]);
  if (ol_result_t result{
          olCreateQueue(gpu->context, gpu->device, &process->queue)}) {
    merk_mailbox_close(h[0], h[1], h[2], h[3], h[4]);
    std::free(process);
    return Error("could not make a queue on GPU %s%lld: %s%s%s", "", number,
        Details(result), "", "");
  }
  // Each parameter of the kernel is a word, of which the plugin copies as
  // many bytes as the parameter has.
  std::size_t count{static_cast<std::size_t>(words[0]) + kHandleWords};
  auto *values{static_cast<std::uint64_t *>(
      Malloc(count * (2 * sizeof(std::uint64_t) + sizeof(void *))))};
  auto *sizes{reinterpret_cast<std::size_t *>(values + count)};
  auto *pointers{reinterpret_cast<void **>(sizes + count)};
  for (std::size_t i{0}; i < count; ++i) {
    values[i] = i < words[0] ? words[1 + i]
                             : static_cast<std::uint64_t>(h[i - words[0]]);
    sizes[i] = sizeof(std::uint64_t);
    pointers[i] = &values[i];
  }
  ol_kernel_launch_size_args_t size{1, {1, 1, 1}, {1, 1, 1}, 0};
  GpuProcessStarted();
  ol_result_t result{olLaunchKernel(process->queue, gpu->device, kernel, &size,
      nullptr, count, pointers, sizes)};
  std::free(values);
  if (result) {
    GpuProcessEnded();
    merk_mailbox_close(h[0], h[1], h[2], h[3], h[4]);
    olDestroyQueue(process->queue);
    std::free(process);
    return Error("could not launch the process kind %s's kernel on GPU "
                 "%lld: %s%s%s",
        kind, number, Details(result), "", "");
  }
  *placeAndSlot = h[2];
  *generation = h[3];
  *secret = h[4];
  if (ol_result_t queued{olLaunchHostFunction(process->queue, Ended, process)}) {
    std::fflush(stdout);
    std::fprintf(stderr,
        "Mercury runtime: could not follow a GPU process's kernel: %s\n",
        Details(queued));
    ExitProgram(1);
  }
  return nullptr;
}
