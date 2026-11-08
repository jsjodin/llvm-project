//===-- runtime/process.h - Merkurius processes -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Processes and their scheduler (design document, "Processes and scheduler";
// docs/INPROGRESS.md, The scheduler). Compiled code allocates through the
// thread's merk_heap (heap.h), which holds the running process's heap: a
// process's own heap is in its record while it does not run.
//
// Temporary (chunk 2 step 4, docs/INPROGRESS.md): one worker, the program's
// first thread, runs every process, each until it blocks, and one lock
// guards the scheduler and the mailboxes. A process holds the lock across a
// switch, and the process switched to releases it.
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_RUNTIME_PROCESS_H_
#define MERKURIUS_RUNTIME_PROCESS_H_

#include "heap.h"
#include "stack.h"
#include <cstddef>
#include <cstdint>
#include <pthread.h>

namespace merkurius::runtime {

struct Process {
  // The next free byte of its heap and that chunk's end, while it does not
  // run; null for a process that has not allocated.
  MerkHeap heap;
  std::uint64_t id;
  // Where its stack's overflow check stops, while it does not run; null for
  // no check.
  char *stackLimit{nullptr};
  // The size of the next chunk of its heap, which grows from small so that a
  // process costs little, or 0 for the heap's usual size.
  std::size_t chunkBytes{0};
  Stack *stack{nullptr};
  Context context{nullptr};
  // What it runs: a closure of pred(mailbox(M), io, io), and the mailbox.
  std::uint64_t closure{0};
  std::uint64_t argument{0};
  // Whether it is blocked, and until when, on the monotonic clock in
  // nanoseconds, negative for no end.
  bool waiting{false};
  std::int64_t deadline{-1};
  Process *nextRunnable{nullptr};
  Process *previousTimed{nullptr};
  Process *nextTimed{nullptr};
  // The first of its mailboxes, a slot of the table, or -1.
  std::int64_t firstMailbox{-1};
};

// Guards the scheduler and the mailboxes (mailbox.cpp).
extern pthread_mutex_t schedulerLock;

// The process this thread runs, or null on a thread that runs none, as a
// parallel loop's team threads do not.
Process *CurrentProcess();

// This thread runs `next` from now on, which may be null: the current
// process's heap goes to its record, and next's comes from its own.
void RunProcess(Process *next);

// A process that will run `closure` on `argument`, not yet runnable, or null
// if there is no address space for its stack. 0 bytes is the default size.
Process *NewProcess(
    std::size_t stackBytes, std::uint64_t closure, std::uint64_t argument);

// The rest take the lock held, and hold it on return.
// Makes a new process runnable.
void Start(Process *);
// Makes a blocked process runnable; one that is not blocked is left alone.
void Wake(Process *);
// Runs other processes until this one is woken, or its deadline, if not
// negative, has passed.
void Block(std::int64_t deadline);

// mailbox.cpp: closes the process's mailboxes, as it ends.
void CloseMailboxesOf(Process *);

} // namespace merkurius::runtime

extern "C" {
// What a procedure's entry compares the stack pointer with: the running
// process's limit, or null on a thread that runs none, which passes.
extern thread_local char *merk_stack_limit;
// A checked procedure found the stack pointer below the limit.
[[noreturn]] void merk_stack_overflow(void);

// Runs the program's first process, which calls `entry`, on a stack of its
// own, and returns when it does.
void merk_process_run_main(void (*entry)(void));
}

#endif // MERKURIUS_RUNTIME_PROCESS_H_
