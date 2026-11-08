//===-- runtime/process.cpp - Merkurius processes -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "process.h"
#include "node.h"
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <sched.h>

extern "C" {
// Constant-initialized, as merk_heap is.
thread_local char *merk_stack_limit{nullptr};
}

namespace merkurius::runtime {

pthread_mutex_t schedulerLock = PTHREAD_MUTEX_INITIALIZER;

namespace {
constexpr std::size_t kFirstChunkBytes{4096};

// Constant-initialized, as merk_heap is.
thread_local Process *current{nullptr};
Process firstProcess{{nullptr, nullptr}, 1};
Context threadContext{nullptr};
std::uint64_t nextId{2};

// The processes ready to run, oldest first, and how many, which the worker
// reads without the lock as it spins.
Process *firstRunnable{nullptr};
Process *lastRunnable{nullptr};
std::int64_t runnable{0};

// Temporary (chunk 2 step 4, docs/INPROGRESS.md): a list, until each worker
// has a timer wheel. The processes blocked with a deadline, and a time no
// later than the earliest, or -1 if there are none.
Process *firstTimed{nullptr};
std::int64_t earliest{-1};

// Where the worker sleeps when nothing can run.
pthread_cond_t ready;
bool sleeping{false};
pthread_once_t readyOnce = PTHREAD_ONCE_INIT;
// What a process that has ended saves into, since its record has gone.
Context ended{nullptr};

void InitializeReady() {
  pthread_condattr_t attributes;
  pthread_condattr_init(&attributes);
  pthread_condattr_setclock(&attributes, CLOCK_MONOTONIC);
  pthread_cond_init(&ready, &attributes);
  pthread_condattr_destroy(&attributes);
}

std::int64_t Now() {
  timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);
  return std::int64_t{now.tv_sec} * 1000000000 + now.tv_nsec;
}

void Enqueue(Process *process) {
  process->nextRunnable = nullptr;
  if (lastRunnable) {
    lastRunnable->nextRunnable = process;
  } else {
    firstRunnable = process;
  }
  lastRunnable = process;
  __atomic_add_fetch(&runnable, 1, __ATOMIC_RELEASE);
  if (sleeping) {
    pthread_cond_signal(&ready);
  }
}

void AddTimed(Process *process) {
  process->previousTimed = nullptr;
  process->nextTimed = firstTimed;
  if (firstTimed) {
    firstTimed->previousTimed = process;
  }
  firstTimed = process;
  if (earliest < 0 || process->deadline < earliest) {
    earliest = process->deadline;
  }
}

void RemoveTimed(Process *process) {
  if (process->previousTimed) {
    process->previousTimed->nextTimed = process->nextTimed;
  } else {
    firstTimed = process->nextTimed;
  }
  if (process->nextTimed) {
    process->nextTimed->previousTimed = process->previousTimed;
  }
  if (!firstTimed) {
    earliest = -1;
  }
}

// Wakes the processes whose deadlines have passed.
void FireTimers() {
  if (earliest < 0 || Now() < earliest) {
    return;
  }
  std::int64_t now{Now()};
  std::int64_t next{-1};
  Process *process{firstTimed};
  while (process) {
    Process *following{process->nextTimed};
    if (process->deadline <= now) {
      Wake(process);
    } else if (next < 0 || process->deadline < next) {
      next = process->deadline;
    }
    process = following;
  }
  earliest = next;
}

// Nothing is runnable: spins for a while, as a wake from sleeping costs
// more than a message's other steps, and then sleeps until something is.
void Idle() {
  std::int64_t end{Now() + SpinNanoseconds()};
  if (earliest >= 0 && earliest < end) {
    end = earliest;
  }
  pthread_mutex_unlock(&schedulerLock);
  // It yields, so that with fewer cores than threads the one it waits for
  // can run.
  while (__atomic_load_n(&runnable, __ATOMIC_ACQUIRE) == 0 && Now() < end) {
    sched_yield();
  }
  pthread_mutex_lock(&schedulerLock);
  if (firstRunnable || (earliest >= 0 && Now() >= earliest)) {
    return;
  }
  pthread_once(&readyOnce, InitializeReady);
  sleeping = true;
  if (earliest < 0) {
    pthread_cond_wait(&ready, &schedulerLock);
  } else {
    timespec until{static_cast<time_t>(earliest / 1000000000),
        static_cast<long>(earliest % 1000000000)};
    pthread_cond_timedwait(&ready, &schedulerLock, &until);
  }
  sleeping = false;
}

Process *PickNext() {
  for (;;) {
    FireTimers();
    if (Process *process{firstRunnable}) {
      firstRunnable = process->nextRunnable;
      if (!firstRunnable) {
        lastRunnable = nullptr;
      }
      __atomic_sub_fetch(&runnable, 1, __ATOMIC_RELAXED);
      return process;
    }
    Idle();
  }
}

// The stack and record go back while still in use, which is safe with one
// worker: nothing else takes them until this has switched away.
[[noreturn]] void Finish(Process *self) {
  CloseMailboxesOf(self);
  Process *next{PickNext()};
  RunProcess(next);
  FreeStack(self->stack);
  std::free(self);
  SwitchContext(ended, next->context);
  std::abort();
}

void ProcessBody(void *argument) {
  auto *self{static_cast<Process *>(argument)};
  pthread_mutex_unlock(&schedulerLock);
  CallClosure(self->closure, self->argument);
  pthread_mutex_lock(&schedulerLock);
  Finish(self);
}

void MainBody(void *entry) {
  reinterpret_cast<void (*)(void)>(entry)();
  SwitchContext(firstProcess.context, threadContext);
}
} // namespace

Process *CurrentProcess() { return current; }

void RunProcess(Process *next) {
  if (current) {
    current->heap = merk_heap;
    current->stackLimit = merk_stack_limit;
  }
  merk_heap = next ? next->heap : MerkHeap{nullptr, nullptr};
  merk_stack_limit = next ? next->stackLimit : nullptr;
  current = next;
}

Process *NewProcess(
    std::size_t stackBytes, std::uint64_t closure, std::uint64_t argument) {
  Stack *stack{NewStack(stackBytes ? stackBytes : kDefaultStackBytes)};
  if (!stack) {
    return nullptr;
  }
  auto *process{static_cast<Process *>(std::malloc(sizeof(Process)))};
  if (!process) {
    FreeStack(stack);
    return nullptr;
  }
  *process = Process{{nullptr, nullptr}, nextId++};
  process->stack = stack;
  process->stackLimit = stack->limit;
  process->chunkBytes = kFirstChunkBytes;
  process->closure = closure;
  process->argument = argument;
  process->context = NewContext(*stack, ProcessBody, process);
  return process;
}

void Start(Process *process) { Enqueue(process); }

void Wake(Process *process) {
  if (!process->waiting) {
    return;
  }
  process->waiting = false;
  if (process->deadline >= 0) {
    RemoveTimed(process);
  }
  Enqueue(process);
}

void Block(std::int64_t deadline) {
  Process *self{current};
  self->waiting = true;
  self->deadline = deadline;
  if (deadline >= 0) {
    AddTimed(self);
  }
  Process *next{PickNext()};
  if (next != self) {
    RunProcess(next);
    SwitchContext(self->context, next->context);
  }
}

} // namespace merkurius::runtime

extern "C" void merk_process_run_main(void (*entry)(void)) {
  using namespace merkurius::runtime;
  firstProcess.stack = NewStack(kDefaultStackBytes);
  if (!firstProcess.stack) {
    // No address space for it: the thread's own stack serves, unchecked.
    RunProcess(&firstProcess);
    entry();
    return;
  }
  firstProcess.stackLimit = firstProcess.stack->limit;
  firstProcess.context = NewContext(*firstProcess.stack, MainBody,
      reinterpret_cast<void *>(entry));
  RunProcess(&firstProcess);
  SwitchContext(threadContext, firstProcess.context);
}

// Temporary (chunk 2 step 6, docs/INPROGRESS.md): ends the program, until
// an uncaught error ends only its process, which its monitors see.
extern "C" void merk_stack_overflow(void) {
  std::fflush(stdout);
  std::fprintf(stderr, "Mercury runtime: stack overflow\n");
  std::exit(1);
}
