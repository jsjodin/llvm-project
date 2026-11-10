//===-- runtime/process.h - Merkurius processes -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Processes and their scheduler (design document, "Processes and scheduler";
// design document, How the scheduler was built).
// Compiled code allocates through the
// thread's merk_heap (heap.h), which holds the running process's heap: a
// process's own heap is in its record while it does not run.
//
// Workers are OS threads, started as there is work for them, up to one for
// each core or MERK_WORKERS; a program that spawns nothing has only its
// first thread. A worker runs a process until it blocks. A worker with nothing
// to run does so on its thread's own stack, since another worker may resume
// the process it last ran.
//
// A process may resume on another thread after any call that can block. The
// runtime's C++ must not keep a thread-local's address across such a call.
//
// Each worker has a queue of runnable processes, with a lock of its own, and
// a worker with an empty queue takes half of another's.
//
// A process that blocks holds its waitLock across its switch, and what it
// switches to releases it, so that nothing can wake and resume it before its
// context is saved. Locks are taken in this order: a mailbox's, a process's
// waitLock, schedulerLock (timers and new stacks), a worker's queue.
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

// A process's messages are stamped in the order they arrive, over all its
// mailboxes. `arrivals` also counts its mailboxes' closes.
struct ArrivalCounts {
  std::uint64_t stamps{0};
  std::int64_t arrivals{0};
};

struct Monitor;
struct ChunkLink;

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
  // The chunks of its heap and of the messages it has read, which go when it
  // ends. Only the running process changes them, once it is running.
  ChunkLink *chunks{nullptr};
  // Bytes added to `chunks` since its last collection, and how many start the
  // next, or 0 for the first size.
  std::size_t allocatedBytes{0};
  std::size_t collectAtBytes{0};
  Stack *stack{nullptr};
  Context context{nullptr};
  // What it runs: a closure of pred(mailbox(M), io, io), and the mailbox.
  std::uint64_t closure{0};
  std::uint64_t argument{0};
  // The cores its parallel region has claimed, or 0.
  int team{0};
  // Whether it is on a thread of the blocking pool.
  bool moved{false};
  // Guards `waiting` and its lists of mailboxes and monitors.
  pthread_mutex_t waitLock = PTHREAD_MUTEX_INITIALIZER;
  // Who watches it (mailbox.cpp).
  Monitor *firstMonitor{nullptr};
  ArrivalCounts counts{};
  // Whether it is blocked, and until when, on the monotonic clock in
  // nanoseconds, negative for no end.
  bool waiting{false};
  std::int64_t deadline{-1};
  Process *nextRunnable{nullptr};
  // Its place in the heap of timers, if it has a deadline.
  std::int64_t timerIndex{-1};
  // The first of its mailboxes, a slot of the table, or -1.
  std::int64_t firstMailbox{-1};
  // How many target regions it is in, during which it stays on its thread,
  // to which libomp ties them, and blocks the thread.
  int pinned{0};
};

// Guards the timers, and a stack freed until its process has switched away.
extern pthread_mutex_t schedulerLock;

// The process this thread runs, or null on a thread that runs none, as a
// parallel loop's team threads do not.
Process *CurrentProcess();

// This thread runs `next` from now on, which may be null: the current
// process's heap goes to its record, and next's comes from its own.
void RunProcess(Process *next);

// A process that will run `closure` on `argument`, not yet runnable, or null
// if there is no address space for its stack. 0 bytes is the default size.
// With schedulerLock held.
Process *NewProcess(
    std::size_t stackBytes, std::uint64_t closure, std::uint64_t argument);

// Whether this thread's process, which is not null, can be collected: not
// while it is in a target or parallel region, whose threads hold its cells.
bool MayCollect(const Process *);

// Whether a receive on this thread blocks a process, letting others run,
// rather than the thread.
bool CanSuspend();

// A target region starts or ends. Once the program has spawned, the process
// runs the region on a thread of the blocking pool, so that it holds up no
// worker, and goes back to the workers afterwards.
void Pin();
void Unpin();
// The same for a parallel region, which also claims its team's cores: the
// first returns the team's size.
int EnterParallelRegion();
void ExitParallelRegion();

// Makes a new process runnable.
void Start(Process *);
// With the process's waitLock held, which it keeps: makes a blocked process
// runnable; one that is not blocked is left alone.
void Wake(Process *);
// With this process's waitLock held, which it releases: runs other processes
// until this one is woken, or its deadline, if not negative, has passed.
void Block(std::int64_t deadline);

// Ends the program with this status. A target region still running must not
// have libomptarget torn down under it, so then the library's destructors do
// not run. Standard output is flushed either way.
[[noreturn]] void ExitProgram(int status);
// Whether ExitProgram has been called: what a region does next would use what
// is being torn down, so it stops instead (offload.cpp).
bool ProgramEnding();
// offload.cpp: whether a target region is running.
bool RegionsRunning();
// With MERK_CHECK_PINNED, MERK_CHECK_MESSAGES or MERK_CHECK_DEVICE_MEMORY,
// reports at the program's normal end what of the messages and the devices'
// memory is still allocated (offload.cpp).
void CheckFreedAtEnd();
// mailbox.cpp: how many queued messages neither a process read nor a
// mailbox freed.
std::int64_t HeldMessages();

// After a failure, which has been reported, with its text as the reason that
// monitors see: ends the running process alone, and does not return. It returns if the program has to end instead: on a
// thread that runs no process, in `main`, or in a target or parallel region,
// which cannot be abandoned on their own.
void EndAfterFailure(const char *reason);

// mailbox.cpp: tells those who monitor a process that has closed its
// mailboxes that it ended, for `reason`, or normally if that is null.
void FireMonitorsOf(Process *, const char *reason);

// mailbox.cpp: closes the process's mailboxes, as it ends.
void CloseMailboxesOf(Process *);

} // namespace merkurius::runtime

extern "C" {
// What a procedure's entry compares the stack pointer with: the running
// process's limit, or null on a thread that runs none, which passes.
extern thread_local char *merk_stack_limit;
// A check found the stack pointer below the limit: the process is being
// preempted, and this returns, perhaps on another thread, or its stack has
// overflowed, and this does not.
void merk_stack_limit_passed(void);

// Runs the program's first process, which calls `entry`, on a stack of its
// own. `entry` ends the program, with merk_process_stop and exit, on whichever
// thread it then runs. If there is no address space for the stack, it is
// called on this thread, which then runs no process.
void merk_process_run_main(void (*entry)(void));

// As the program ends: no process is scheduled after this.
void merk_process_stop(void);
}

#endif // MERKURIUS_RUNTIME_PROCESS_H_
