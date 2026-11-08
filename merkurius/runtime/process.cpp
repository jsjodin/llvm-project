//===-- runtime/process.cpp - Merkurius processes -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "process.h"
#include "stack.h"
#include <cstdio>
#include <cstdlib>
#include <pthread.h>

extern "C" {
// Constant-initialized, as merk_heap is.
thread_local char *merk_stack_limit{nullptr};
}

namespace merkurius::runtime {
namespace {
// Constant-initialized, as merk_heap is.
thread_local Process *current{nullptr};
Process firstProcess{{nullptr, nullptr}, 1};
} // namespace

Process *CurrentProcess() { return current; }

void RunProcess(Process *next) {
  if (current) {
    current->heap = merk_heap;
    current->stackLimit = merk_stack_limit;
  }
  merk_stack_limit = next ? next->stackLimit : nullptr;
  merk_heap = next ? next->heap : MerkHeap{nullptr, nullptr};
  current = next;
}

} // namespace merkurius::runtime

// The first process runs on the thread's own stack until there are
// process stacks to run it on (docs/INPROGRESS.md, The scheduler, step 3).
extern "C" void merk_process_start(void) {
  using namespace merkurius::runtime;
  pthread_attr_t attributes;
  if (pthread_getattr_np(pthread_self(), &attributes) == 0) {
    void *base{nullptr};
    std::size_t bytes{0};
    if (pthread_attr_getstack(&attributes, &base, &bytes) == 0 &&
        bytes > 2 * kStackMargin) {
      firstProcess.stackLimit = static_cast<char *>(base) + kStackMargin;
    }
    pthread_attr_destroy(&attributes);
  }
  RunProcess(&firstProcess);
}

// Temporary (chunk 2 step 6, docs/INPROGRESS.md): ends the program, until
// an uncaught error ends only its process, which its monitors see.
extern "C" void merk_stack_overflow(void) {
  std::fflush(stdout);
  std::fprintf(stderr, "Mercury runtime: stack overflow\n");
  std::exit(1);
}
