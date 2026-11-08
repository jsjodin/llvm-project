//===-- runtime/process.cpp - Merkurius processes -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "process.h"
#include "node.h"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <sched.h>
#include <unistd.h>

extern "C" {
// Constant-initialized, as merk_heap is.
thread_local char *merk_stack_limit{nullptr};
}

namespace merkurius::runtime {

pthread_mutex_t schedulerLock = PTHREAD_MUTEX_INITIALIZER;

namespace {
constexpr std::size_t kFirstChunkBytes{4096};
// Spinning workers beyond a few only take cores from each other.
constexpr int kMostSpinning{2};

struct Worker {
  pthread_mutex_t lock;
  Process *first;
  Process *last;
  std::int64_t count;
  // For choosing where to steal from.
  std::uint64_t random;
};

// One for each worker that may start; fixed once the program has started.
Worker *workerTable{nullptr};
int mostWorkers{1};
// How many have started, the first thread being one.
std::atomic<int> workers{1};
pthread_mutex_t startLock = PTHREAD_MUTEX_INITIALIZER;
// For a wake from a thread that is not a worker.
std::atomic<std::uint64_t> nextQueue{0};

// Constant-initialized, as merk_heap is.
thread_local Process *current{nullptr};
// Where this thread's worker goes on from when it has nothing to run.
thread_local Context workerContext{nullptr};
thread_local Worker *thisWorker{nullptr};
// The lock that what this thread switched to is to release, or null.
thread_local pthread_mutex_t *unlockAfterSwitch{nullptr};

Process firstProcess{{nullptr, nullptr}, 1};
std::uint64_t nextId{2};

// The processes queued on any worker. It is added to before a process is
// queued, so that a worker about to sleep cannot miss it.
std::atomic<std::int64_t> runnable{0};
std::atomic<int> idle{0};
std::atomic<int> spinning{0};
std::atomic<int> sleeping{0};
std::atomic<bool> stopping{false};

pthread_mutex_t sleepLock = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t ready;
pthread_once_t readyOnce = PTHREAD_ONCE_INIT;

// The processes blocked with a deadline, a binary heap by deadline, under
// schedulerLock, and the earliest deadline, or -1.
Process **timers{nullptr};
std::int64_t timerCount{0};
std::int64_t timerCapacity{0};
std::atomic<std::int64_t> earliest{-1};

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

[[noreturn]] void OutOfMemory() {
  std::fflush(stdout);
  std::fprintf(stderr, "Mercury runtime: out of memory\n");
  std::exit(1);
}

void *WorkerThread(void *);

void InitializeWorker(Worker &worker, std::uint64_t seed) {
  pthread_mutex_init(&worker.lock, nullptr);
  worker.first = nullptr;
  worker.last = nullptr;
  worker.count = 0;
  worker.random = seed * 2 + 1;
}

// More work than idle workers takes another worker, if one may start.
void StartWorkerIfNeeded() {
  if (workers.load(std::memory_order_relaxed) >= mostWorkers ||
      runnable.load(std::memory_order_relaxed) <=
          idle.load(std::memory_order_relaxed)) {
    return;
  }
  pthread_mutex_lock(&startLock);
  int index{workers.load(std::memory_order_relaxed)};
  if (index < mostWorkers) {
    pthread_t thread;
    pthread_attr_t attributes;
    pthread_attr_init(&attributes);
    pthread_attr_setdetachstate(&attributes, PTHREAD_CREATE_DETACHED);
    idle.fetch_add(1);
    if (pthread_create(&thread, &attributes, WorkerThread, &workerTable[index]) ==
        0) {
      workers.store(index + 1, std::memory_order_release);
    } else {
      idle.fetch_sub(1);
    }
    pthread_attr_destroy(&attributes);
  }
  pthread_mutex_unlock(&startLock);
}

// A sleeping worker is woken unless a spinning one will find the work.
void WakeWorker() {
  if (spinning.load() == 0 && sleeping.load() > 0) {
    pthread_mutex_lock(&sleepLock);
    pthread_cond_signal(&ready);
    pthread_mutex_unlock(&sleepLock);
  }
  StartWorkerIfNeeded();
}

void Enqueue(Process *process) {
  Worker *worker{thisWorker};
  if (!worker) {
    worker = &workerTable[nextQueue.fetch_add(1, std::memory_order_relaxed) %
        static_cast<std::uint64_t>(workers.load(std::memory_order_acquire))];
  }
  runnable.fetch_add(1);
  process->nextRunnable = nullptr;
  pthread_mutex_lock(&worker->lock);
  if (worker->last) {
    worker->last->nextRunnable = process;
  } else {
    worker->first = process;
  }
  worker->last = process;
  ++worker->count;
  pthread_mutex_unlock(&worker->lock);
  WakeWorker();
}

Process *PopLocal(Worker &worker) {
  pthread_mutex_lock(&worker.lock);
  Process *process{worker.first};
  if (process) {
    worker.first = process->nextRunnable;
    if (!worker.first) {
      worker.last = nullptr;
    }
    --worker.count;
  }
  pthread_mutex_unlock(&worker.lock);
  if (process) {
    runnable.fetch_sub(1);
  }
  return process;
}

// The older half of another worker's queue: the first of it, to run, and
// the rest on this worker's.
Process *Steal(Worker &self) {
  int started{workers.load(std::memory_order_acquire)};
  self.random ^= self.random << 13;
  self.random ^= self.random >> 7;
  self.random ^= self.random << 17;
  for (int i{0}; i < started; ++i) {
    Worker &victim{workerTable[(self.random + static_cast<std::uint64_t>(i)) %
        static_cast<std::uint64_t>(started)]};
    if (&victim == &self || !__atomic_load_n(&victim.first, __ATOMIC_RELAXED)) {
      continue;
    }
    pthread_mutex_lock(&victim.lock);
    Process *taken{victim.first};
    if (!taken) {
      pthread_mutex_unlock(&victim.lock);
      continue;
    }
    std::int64_t half{(victim.count + 1) / 2};
    Process *last{taken};
    for (std::int64_t n{1}; n < half; ++n) {
      last = last->nextRunnable;
    }
    victim.first = last->nextRunnable;
    if (!victim.first) {
      victim.last = nullptr;
    }
    victim.count -= half;
    pthread_mutex_unlock(&victim.lock);
    last->nextRunnable = nullptr;
    if (taken != last) {
      pthread_mutex_lock(&self.lock);
      if (self.last) {
        self.last->nextRunnable = taken->nextRunnable;
      } else {
        self.first = taken->nextRunnable;
      }
      self.last = last;
      self.count += half - 1;
      pthread_mutex_unlock(&self.lock);
    }
    runnable.fetch_sub(1);
    return taken;
  }
  return nullptr;
}

void SetTimer(std::int64_t index, Process *process) {
  timers[index] = process;
  process->timerIndex = index;
}

void SiftUp(std::int64_t index) {
  Process *process{timers[index]};
  while (index > 0) {
    std::int64_t parent{(index - 1) / 2};
    if (timers[parent]->deadline <= process->deadline) {
      break;
    }
    SetTimer(index, timers[parent]);
    index = parent;
  }
  SetTimer(index, process);
}

void SiftDown(std::int64_t index) {
  Process *process{timers[index]};
  for (;;) {
    std::int64_t child{2 * index + 1};
    if (child >= timerCount) {
      break;
    }
    if (child + 1 < timerCount &&
        timers[child + 1]->deadline < timers[child]->deadline) {
      ++child;
    }
    if (process->deadline <= timers[child]->deadline) {
      break;
    }
    SetTimer(index, timers[child]);
    index = child;
  }
  SetTimer(index, process);
}

void NoteEarliest() {
  earliest.store(timerCount ? timers[0]->deadline : -1);
}

void AddTimer(Process *process) {
  if (timerCount == timerCapacity) {
    timerCapacity = timerCapacity ? 2 * timerCapacity : 16;
    timers = static_cast<Process **>(
        std::realloc(timers, sizeof(Process *) * timerCapacity));
    if (!timers) {
      OutOfMemory();
    }
  }
  std::int64_t before{earliest.load()};
  SetTimer(timerCount, process);
  SiftUp(timerCount++);
  NoteEarliest();
  // A worker asleep until the old time must see the new one.
  if ((before < 0 || process->deadline < before) && sleeping.load() > 0) {
    pthread_mutex_lock(&sleepLock);
    pthread_cond_signal(&ready);
    pthread_mutex_unlock(&sleepLock);
  }
}

void RemoveTimer(Process *process) {
  std::int64_t index{process->timerIndex};
  process->timerIndex = -1;
  --timerCount;
  if (index != timerCount) {
    SetTimer(index, timers[timerCount]);
    SiftUp(index);
    SiftDown(timers[index]->timerIndex);
  }
  NoteEarliest();
}

bool TimerDue() {
  std::int64_t time{earliest.load(std::memory_order_relaxed)};
  return time >= 0 && Now() >= time;
}

// Wakes the processes whose deadlines have passed. A process whose waitLock
// is taken is being blocked or woken; it is tried again later.
void FireTimers() {
  if (!TimerDue()) {
    return;
  }
  pthread_mutex_lock(&schedulerLock);
  std::int64_t now{Now()};
  while (timerCount && timers[0]->deadline <= now) {
    Process *process{timers[0]};
    if (pthread_mutex_trylock(&process->waitLock) != 0) {
      break;
    }
    process->waiting = false;
    RemoveTimer(process);
    pthread_mutex_unlock(&process->waitLock);
    Enqueue(process);
  }
  pthread_mutex_unlock(&schedulerLock);
}

// What a thread does first after switching: on a process's stack it may be
// another thread than last time, so this must read its thread-local afresh.
[[gnu::noinline]] void AfterSwitch() {
  if (unlockAfterSwitch) {
    pthread_mutex_t *lock{unlockAfterSwitch};
    unlockAfterSwitch = nullptr;
    pthread_mutex_unlock(lock);
  }
}

// Nothing is runnable: spins for a while, as a wake from sleeping costs
// more than a message's other steps, and then sleeps until something is.
void Idle() {
  if (spinning.load() < kMostSpinning) {
    spinning.fetch_add(1);
    std::int64_t end{Now() + SpinNanoseconds()};
    std::int64_t time{earliest.load()};
    if (time >= 0 && time < end) {
      end = time;
    }
    // It yields, so that with fewer cores than threads the one it waits for
    // can run.
    while (runnable.load() == 0 && Now() < end) {
      sched_yield();
    }
    spinning.fetch_sub(1);
    if (runnable.load() > 0 || TimerDue()) {
      return;
    }
  }
  pthread_once(&readyOnce, InitializeReady);
  pthread_mutex_lock(&sleepLock);
  sleeping.fetch_add(1);
  if (runnable.load() == 0 && !TimerDue()) {
    std::int64_t time{earliest.load()};
    if (time < 0) {
      pthread_cond_wait(&ready, &sleepLock);
    } else {
      timespec until{static_cast<time_t>(time / 1000000000),
          static_cast<long>(time % 1000000000)};
      pthread_cond_timedwait(&ready, &sleepLock, &until);
    }
  }
  sleeping.fetch_sub(1);
  pthread_mutex_unlock(&sleepLock);
}

// On the thread's own stack, without the lock: runs processes for ever.
[[noreturn]] void WorkerLoop() {
  AfterSwitch();
  Worker &self{*thisWorker};
  for (;;) {
    idle.fetch_add(1);
    Process *next{nullptr};
    while (!next) {
      FireTimers();
      next = stopping.load() ? nullptr : PopLocal(self);
      if (!next && !stopping.load()) {
        next = Steal(self);
      }
      if (!next) {
        Idle();
      }
    }
    idle.fetch_sub(1);
    // Work may remain for a worker that is asleep.
    if (runnable.load() > 0) {
      WakeWorker();
    }
    RunProcess(next);
    SwitchContext(workerContext, next->context);
    AfterSwitch();
  }
}

void *WorkerThread(void *worker) {
  thisWorker = static_cast<Worker *>(worker);
  idle.fetch_sub(1);
  WorkerLoop();
}

// The next process to run on this worker, or null.
Process *TakeLocal() {
  return stopping.load() ? nullptr : PopLocal(*thisWorker);
}

// The stack and record go back while still in use, which is safe as
// schedulerLock is held until this has switched away, and NewProcess is
// called with it.
[[noreturn]] void Finish(Process *self) {
  pthread_mutex_lock(&schedulerLock);
  Process *next{TakeLocal()};
  RunProcess(next);
  FreeStack(self->stack);
  std::free(self);
  unlockAfterSwitch = &schedulerLock;
  SwitchContext(ended, next ? next->context : workerContext);
  std::abort();
}

void ProcessBody(void *argument) {
  AfterSwitch();
  auto *self{static_cast<Process *>(argument)};
  CallClosure(self->closure, self->argument);
  CloseMailboxesOf(self);
  Finish(self);
}

void MainBody(void *entry) {
  AfterSwitch();
  reinterpret_cast<void (*)(void)>(entry)();
  std::abort();
}

int ReadMostWorkers() {
  if (const char *text{std::getenv("MERK_WORKERS")}; text && *text) {
    char *end{nullptr};
    long value{std::strtol(text, &end, 10)};
    if (*end != '\0' || value < 1) {
      std::fflush(stdout);
      std::fprintf(stderr,
          "Mercury runtime: MERK_WORKERS=%s is not a positive number\n", text);
      std::exit(1);
    }
    return static_cast<int>(value);
  }
  long cores{sysconf(_SC_NPROCESSORS_ONLN)};
  return cores > 0 ? static_cast<int>(cores) : 1;
}
} // namespace

Process *CurrentProcess() { return current; }

bool CanSuspend() { return current && current->pinned == 0; }

void Pin() {
  if (current) {
    ++current->pinned;
  }
}

void Unpin() {
  if (current) {
    --current->pinned;
  }
}

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
  if (process->timerIndex >= 0) {
    pthread_mutex_lock(&schedulerLock);
    RemoveTimer(process);
    pthread_mutex_unlock(&schedulerLock);
  }
  Enqueue(process);
}

void Block(std::int64_t deadline) {
  Process *self{current};
  self->waiting = true;
  self->deadline = deadline;
  if (deadline >= 0) {
    pthread_mutex_lock(&schedulerLock);
    AddTimer(self);
    pthread_mutex_unlock(&schedulerLock);
  }
  FireTimers();
  Process *next{TakeLocal()};
  RunProcess(next);
  unlockAfterSwitch = &self->waitLock;
  SwitchContext(self->context, next ? next->context : workerContext);
  // It may be on another thread now.
  AfterSwitch();
}

} // namespace merkurius::runtime

extern "C" void merk_process_run_main(void (*entry)(void)) {
  using namespace merkurius::runtime;
  firstProcess.stack = NewStack(kDefaultStackBytes);
  if (!firstProcess.stack) {
    entry();
    return;
  }
  mostWorkers = ReadMostWorkers();
  workerTable = static_cast<Worker *>(std::malloc(sizeof(Worker) * mostWorkers));
  if (!workerTable) {
    OutOfMemory();
  }
  for (int i{0}; i < mostWorkers; ++i) {
    InitializeWorker(workerTable[i], static_cast<std::uint64_t>(i) + 1);
  }
  thisWorker = &workerTable[0];
  firstProcess.stackLimit = firstProcess.stack->limit;
  firstProcess.context = NewContext(*firstProcess.stack, MainBody,
      reinterpret_cast<void *>(entry));
  RunProcess(&firstProcess);
  SwitchContext(workerContext, firstProcess.context);
  WorkerLoop();
}

extern "C" void merk_process_stop(void) {
  merkurius::runtime::stopping.store(true);
  pthread_mutex_lock(&merkurius::runtime::schedulerLock);
}

// Temporary (chunk 2 step 6, docs/INPROGRESS.md): ends the program, until
// an uncaught error ends only its process, which its monitors see.
extern "C" void merk_stack_overflow(void) {
  std::fflush(stdout);
  std::fprintf(stderr, "Mercury runtime: stack overflow\n");
  std::exit(1);
}
