//===-- runtime/process-images.cpp - GPU process kinds' images ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "process-images.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <pthread.h>

namespace {

using merkurius::runtime::ProcessImage;

// The registered images, the last first. Modules register them from their
// constructors, which run before main, so the list is short and does not
// change while processes run; the lock is for a program that loads a module
// later.
struct Entry {
  ProcessImage image;
  Entry *next;
};
Entry *images{nullptr};
pthread_mutex_t imagesLock = PTHREAD_MUTEX_INITIALIZER;

[[noreturn]] void Fail(
    const char *message, const char *module, const char *gpu) {
  std::fflush(stdout);
  std::fprintf(stderr, "Mercury runtime: %s%s for %s\n", message, module, gpu);
  std::_Exit(1);
}

Entry *Find(const char *module, const char *gpu) {
  for (Entry *entry{images}; entry; entry = entry->next) {
    if (std::strcmp(entry->image.module, module) == 0 &&
        std::strcmp(entry->image.gpu, gpu) == 0) {
      return entry;
    }
  }
  return nullptr;
}

} // namespace

// Registers the image of `module`'s process kinds that run on a GPU, for
// `gpu`: `bytes` at `image`, which stays where it is, in the program's
// data. A module registers each GPU's image once.
extern "C" void merk_register_process_image(const char *module, const char *gpu,
    const void *image, std::uint64_t bytes) {
  pthread_mutex_lock(&imagesLock);
  if (Find(module, gpu)) {
    Fail("GPU processes' image registered twice: ", module, gpu);
  }
  auto *entry{static_cast<Entry *>(std::malloc(sizeof(Entry)))};
  if (!entry) {
    Fail("out of memory for the GPU processes' image of ", module, gpu);
  }
  *entry = {{module, gpu, image, bytes}, images};
  images = entry;
  pthread_mutex_unlock(&imagesLock);
}

namespace merkurius::runtime {

const ProcessImage *FindProcessImage(const char *module, const char *gpu) {
  pthread_mutex_lock(&imagesLock);
  Entry *entry{Find(module, gpu)};
  pthread_mutex_unlock(&imagesLock);
  return entry ? &entry->image : nullptr;
}

} // namespace merkurius::runtime
