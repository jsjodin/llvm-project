//===-- runtime/process.h - Merkurius processes -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A process's record (design document, "Processes and scheduler";
// docs/INPROGRESS.md, The scheduler, step 1). Compiled code allocates through
// the thread's merk_heap (heap.h), which holds the running process's heap:
// a process's own heap is in its record while it does not run.
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_RUNTIME_PROCESS_H_
#define MERKURIUS_RUNTIME_PROCESS_H_

#include "heap.h"
#include <cstdint>

namespace merkurius::runtime {

struct Process {
  // The next free byte of its heap and that chunk's end, while it does not
  // run; null for a process that has not allocated.
  MerkHeap heap;
  std::uint64_t id;
};

// The process this thread runs, or null on a thread that runs none, as a
// parallel loop's team threads do not.
Process *CurrentProcess();

// This thread runs `next` from now on, which may be null: the current
// process's heap goes to its record, and next's comes from its own.
void RunProcess(Process *next);

} // namespace merkurius::runtime

extern "C" {
// At start-up: the thread that calls it runs the program's first process.
void merk_process_start(void);
}

#endif // MERKURIUS_RUNTIME_PROCESS_H_
