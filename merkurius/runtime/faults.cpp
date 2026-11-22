//===-- runtime/faults.cpp - The fault injector ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Faults injected from outside what the program supervises, to test and
// measure its recovery (docs/INPROGRESS.md, Failure, step 3). MERK_FAULTS
// turns it on, with settings separated by spaces or commas:
//
// - `seed=N`: the random numbers' seed; a random one if it is missing, which
//   the schedule records.
// - `mtbf=MS`: the mean time between faults in milliseconds. The times are
//   exponentially distributed, as failures that come independently are.
// - `fail=T+U`: the kinds of process to fail, by their spawn mailbox's
//   message type, `T` or `module.T`. A fault ends a random live process of
//   those kinds as a failure, with the reason failed("injected"), as
//   process.stop ends one with killed.
// - `partition`: a fault may also silence every connection to a random peer
//   node, as a partition would. Nothing more is written to them, and what is
//   read from them is dropped and not heard, so the ticks lose them within
//   a few tick times, and both nodes fence each other.
// - `kernel`: a fault may also end every process of a random GPU's kernel
//   with the reason kernel_ended, as a trap would end them all, through
//   their stop words (place.h): each ends at its next receive.
// - `count=N`: no more than N faults.
// - `log=PATH`: where the schedule is written, else to stderr.
// - `replay=PATH`: injects a schedule written before, and nothing else.
//
// Each fault picks one of the kinds it has turned on at random, and one of
// that kind's targets; a time with no target injects nothing. The schedule
// has a line for each fault: its time in milliseconds since the runtime
// started, and `fail` with the kind and the process's number among those of
// its kind in the order they were spawned, from 1, or `partition` with the
// peer's number in the order this node first connected to it, or `kernel`
// with the GPU's number. Replay
// repeats the times and the targets, which name the same processes and
// peers in a program that spawns and connects in the same order; it does
// not repeat the threads' interleaving.
//
//===----------------------------------------------------------------------===//

#include "process.h"

#include "descriptor.h"
#include "node.h"
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <pthread.h>
#include <sys/random.h>

using namespace merkurius::runtime;

namespace merkurius::runtime {
bool faultsOn{false};
} // namespace merkurius::runtime

namespace {

constexpr int kMostKinds{16};
constexpr std::size_t kNameBytes{256};

// A listed process, by its kind and number.
struct Entry {
  Entry *previous;
  Entry *next;
  Process *process;
  int kind;
  std::uint64_t number;
};

struct Settings {
  std::uint64_t seed{0};
  bool seeded{false};
  double mtbf{0};
  char kinds[kMostKinds][kNameBytes]{};
  int kindCount{0};
  bool partition{false};
  bool kernel{false};
  std::uint64_t count{UINT64_MAX};
  char log[kNameBytes]{};
  char replay[kNameBytes]{};
};

Settings settings;
// Guards the list and the numbers.
pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
Entry *first{nullptr};
std::uint64_t spawned[kMostKinds]{};
FILE *schedule{nullptr};
std::int64_t started{0};

[[noreturn]] void Fail(const char *message, const char *detail) {
  std::fflush(stdout);
  std::fprintf(stderr, "Merkurius runtime: %s%s\n", message, detail);
  ExitProgram(1);
}

std::int64_t Now() {
  timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);
  return std::int64_t{now.tv_sec} * 1000000000 + now.tv_nsec;
}

void SleepUntil(std::int64_t when) {
  timespec until{static_cast<time_t>(when / 1000000000),
      static_cast<long>(when % 1000000000)};
  while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &until, nullptr) ==
      EINTR) {
  }
}

// splitmix64: small, and the same everywhere, so a seed means one schedule.
std::uint64_t Next(std::uint64_t &state) {
  std::uint64_t z{state += 0x9e3779b97f4a7c15u};
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9u;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebu;
  return z ^ (z >> 31);
}

// Uniform in [0, n).
std::uint64_t Below(std::uint64_t &state, std::uint64_t n) {
  return Next(state) % n;
}

// An exponentially distributed wait, in nanoseconds.
std::int64_t Wait(std::uint64_t &state) {
  double u{static_cast<double>(Next(state) >> 11) * 0x1.0p-53};
  return static_cast<std::int64_t>(-std::log1p(-u) * settings.mtbf * 1e6);
}

bool ParseNumber(const char *text, std::uint64_t &value) {
  char *end;
  errno = 0;
  unsigned long long parsed{std::strtoull(text, &end, 10)};
  if (*text == '\0' || *end != '\0' || errno != 0 || *text == '-') {
    return false;
  }
  value = parsed;
  return true;
}

void Copy(char (&into)[kNameBytes], const char *text, const char *what) {
  if (std::strlen(text) >= kNameBytes) {
    Fail("MERK_FAULTS: too long: ", what);
  }
  std::strcpy(into, text);
}

void Setting(char *item) {
  char *equals{std::strchr(item, '=')};
  const char *value{equals ? equals + 1 : nullptr};
  if (equals) {
    *equals = '\0';
  }
  std::uint64_t number;
  if (std::strcmp(item, "seed") == 0 && value && ParseNumber(value, number)) {
    settings.seed = number;
    settings.seeded = true;
  } else if (std::strcmp(item, "mtbf") == 0 && value &&
      ParseNumber(value, number) && number > 0) {
    settings.mtbf = static_cast<double>(number);
  } else if (std::strcmp(item, "count") == 0 && value &&
      ParseNumber(value, number)) {
    settings.count = number;
  } else if (std::strcmp(item, "fail") == 0 && value && *value) {
    char names[kNameBytes];
    Copy(names, value, "fail");
    for (char *name{std::strtok(names, "+")}; name;
        name = std::strtok(nullptr, "+")) {
      if (settings.kindCount == kMostKinds) {
        Fail("MERK_FAULTS: more than 16 kinds to fail", "");
      }
      Copy(settings.kinds[settings.kindCount++], name, "fail");
    }
  } else if (std::strcmp(item, "partition") == 0 && !value) {
    settings.partition = true;
  } else if (std::strcmp(item, "kernel") == 0 && !value) {
    settings.kernel = true;
  } else if (std::strcmp(item, "log") == 0 && value && *value) {
    Copy(settings.log, value, "log");
  } else if (std::strcmp(item, "replay") == 0 && value && *value) {
    Copy(settings.replay, value, "replay");
  } else {
    if (equals) {
      *equals = '=';
    }
    Fail("MERK_FAULTS: not a setting: ", item);
  }
}

// The listed kind of a message type, or -1.
int KindOf(Word messageType) {
  const TypeCtor *ctor{(messageType & kTagMask) == 0
          ? reinterpret_cast<const TypeCtor *>(messageType)
          : reinterpret_cast<const TypeCtor *>(
                reinterpret_cast<const Word *>(messageType - 1)[0])};
  std::size_t module{std::strlen(ctor->module)};
  for (int i{0}; i < settings.kindCount; ++i) {
    const char *kind{settings.kinds[i]};
    if (std::strcmp(kind, ctor->name) == 0 ||
        (std::strncmp(kind, ctor->module, module) == 0 && kind[module] == '.' &&
            std::strcmp(kind + module + 1, ctor->name) == 0)) {
      return i;
    }
  }
  return -1;
}

void Record(std::int64_t at, const char *line) {
  long long milliseconds{static_cast<long long>((at - started) / 1000000)};
  if (settings.log[0]) {
    std::fprintf(schedule, "%lld %s\n", milliseconds, line);
    std::fflush(schedule);
  } else {
    std::fprintf(
        stderr, "Merkurius runtime: fault: %lld %s\n", milliseconds, line);
  }
}

bool Ending(const Entry *e) {
  return __atomic_load_n(&e->process->stopRequested, __ATOMIC_ACQUIRE);
}

// Fails the process of the kind with the number, or with `number` 0 a random
// one of any listed kind: false if there is none.
bool FailOne(std::uint64_t &state, int kind, std::uint64_t number,
    char (&line)[2 * kNameBytes]) {
  pthread_mutex_lock(&lock);
  Entry *chosen{nullptr};
  if (number == 0) {
    // Not one that a stop or a fault is ending already.
    std::uint64_t live{0};
    for (Entry *e{first}; e; e = e->next) {
      live += Ending(e) ? 0 : 1;
    }
    if (live > 0) {
      std::uint64_t index{Below(state, live)};
      for (Entry *e{first}; e && !chosen; e = e->next) {
        if (!Ending(e) && index-- == 0) {
          chosen = e;
        }
      }
    }
  } else {
    for (Entry *e{first}; e && !chosen; e = e->next) {
      if (e->kind == kind && e->number == number) {
        chosen = e;
      }
    }
  }
  if (chosen) {
    // On the list, so it has not ended: Finish takes it off with the lock.
    RequestStop(chosen->process, kDownFailed);
    std::snprintf(line, sizeof line, "fail %s %llu",
        settings.kinds[chosen->kind],
        static_cast<unsigned long long>(chosen->number));
  }
  pthread_mutex_unlock(&lock);
  return chosen != nullptr;
}

bool PartitionOne(std::uint64_t &state, char (&line)[2 * kNameBytes]) {
  std::uint64_t numbers[256];
  std::uint64_t count{LivePeers(numbers, 256)};
  if (count == 0) {
    return false;
  }
  std::uint64_t number{numbers[Below(state, count)]};
  if (!Partition(number)) {
    return false;
  }
  std::snprintf(line, sizeof line, "partition %llu",
      static_cast<unsigned long long>(number));
  return true;
}

bool KernelOne(std::uint64_t &state, char (&line)[2 * kNameBytes]) {
  int gpus[64];
  int count{LiveKernels(gpus, 64)};
  if (count == 0) {
    return false;
  }
  int gpu{gpus[Below(state, static_cast<std::uint64_t>(count))]};
  if (!EndKernel(gpu)) {
    return false;
  }
  std::snprintf(line, sizeof line, "kernel %d", gpu);
  return true;
}

void Inject() {
  std::uint64_t state{settings.seed};
  std::uint64_t injected{0};
  std::int64_t at{started};
  // The kinds of fault turned on: fail, partition and kernel, in that order.
  int kinds[3];
  int kindCount{0};
  if (settings.kindCount > 0) {
    kinds[kindCount++] = 0;
  }
  if (settings.partition) {
    kinds[kindCount++] = 1;
  }
  if (settings.kernel) {
    kinds[kindCount++] = 2;
  }
  while (injected < settings.count) {
    at += Wait(state);
    SleepUntil(at);
    int kind{kindCount == 1
            ? kinds[0]
            : kinds[Below(state, static_cast<std::uint64_t>(kindCount))]};
    char line[2 * kNameBytes];
    bool done{kind == 0 ? FailOne(state, 0, 0, line)
            : kind == 1 ? PartitionOne(state, line)
                        : KernelOne(state, line)};
    if (done) {
      Record(at, line);
      ++injected;
    }
  }
}

void Replay() {
  FILE *file{std::fopen(settings.replay, "r")};
  if (!file) {
    Fail("MERK_FAULTS: cannot read the schedule ", settings.replay);
  }
  char text[3 * kNameBytes];
  std::uint64_t unused{0};
  while (std::fgets(text, sizeof text, file)) {
    if (text[0] == '#' || text[0] == '\n') {
      continue;
    }
    long long milliseconds;
    char what[16];
    char name[kNameBytes];
    unsigned long long number;
    int fields{std::sscanf(
        text, "%lld %15s %255s %llu", &milliseconds, what, name, &number)};
    bool partition{fields >= 3 && std::strcmp(what, "partition") == 0};
    bool kernel{fields >= 3 && std::strcmp(what, "kernel") == 0};
    if (partition || kernel) {
      fields =
          std::sscanf(text, "%lld %15s %llu", &milliseconds, what, &number);
    }
    if (!(partition || kernel
                ? fields == 3
                : fields == 4 && std::strcmp(what, "fail") == 0)) {
      Fail("MERK_FAULTS: not a line of a schedule: ", text);
    }
    std::int64_t at{started + milliseconds * 1000000};
    SleepUntil(at);
    char line[2 * kNameBytes];
    bool done{false};
    if (partition) {
      done = Partition(number);
      std::snprintf(line, sizeof line, "partition %llu", number);
    } else if (kernel) {
      done = EndKernel(static_cast<int>(number));
      std::snprintf(line, sizeof line, "kernel %llu", number);
    } else {
      int kind{-1};
      for (int i{0}; i < settings.kindCount && kind < 0; ++i) {
        kind = std::strcmp(settings.kinds[i], name) == 0 ? i : -1;
      }
      done = kind >= 0 && FailOne(unused, kind, number, line);
      if (!done) {
        std::snprintf(line, sizeof line, "fail %s %llu", name, number);
      }
    }
    if (done) {
      Record(at, line);
    } else {
      std::fprintf(stderr,
          "Merkurius runtime: fault: the schedule's \"%s\" found no target\n",
          line);
    }
  }
  std::fclose(file);
}

void *Injector(void *) {
  if (settings.replay[0]) {
    Replay();
  } else {
    Inject();
  }
  return nullptr;
}

// Replay names its kinds in the schedule: they are listed from its lines.
void KindsOfSchedule() {
  FILE *file{std::fopen(settings.replay, "r")};
  if (!file) {
    Fail("MERK_FAULTS: cannot read the schedule ", settings.replay);
  }
  char text[3 * kNameBytes];
  while (std::fgets(text, sizeof text, file)) {
    long long milliseconds;
    char name[kNameBytes];
    if (std::sscanf(text, "%lld fail %255s", &milliseconds, name) != 2) {
      continue;
    }
    bool known{false};
    for (int i{0}; i < settings.kindCount && !known; ++i) {
      known = std::strcmp(settings.kinds[i], name) == 0;
    }
    if (!known) {
      if (settings.kindCount == kMostKinds) {
        Fail("MERK_FAULTS: more than 16 kinds to fail", "");
      }
      std::strcpy(settings.kinds[settings.kindCount++], name);
    }
  }
  std::fclose(file);
}

} // namespace

namespace merkurius::runtime {

void StartFaults() {
  const char *text{std::getenv("MERK_FAULTS")};
  if (!text) {
    return;
  }
  started = Now();
  std::size_t bytes{std::strlen(text)};
  auto *items{static_cast<char *>(std::malloc(bytes + 1))};
  std::memcpy(items, text, bytes + 1);
  char *rest{items};
  while (char *item{strsep(&rest, " ,")}) {
    if (*item) {
      Setting(item);
    }
  }
  std::free(items);
  if (settings.replay[0]) {
    if (settings.kindCount > 0 || settings.partition || settings.kernel ||
        settings.mtbf > 0 || settings.seeded) {
      Fail("MERK_FAULTS: replay takes no other settings but log", "");
    }
    KindsOfSchedule();
  } else {
    if (settings.mtbf == 0) {
      Fail("MERK_FAULTS: mtbf=MILLISECONDS is missing", "");
    }
    if (settings.kindCount == 0 && !settings.partition && !settings.kernel) {
      Fail("MERK_FAULTS: no faults to inject: fail=KINDS, partition or "
           "kernel",
          "");
    }
    if (!settings.seeded) {
      while (getrandom(&settings.seed, sizeof settings.seed, 0) !=
          static_cast<ssize_t>(sizeof settings.seed)) {
      }
    }
  }
  if (settings.log[0]) {
    schedule = std::fopen(settings.log, "w");
    if (!schedule) {
      Fail("MERK_FAULTS: cannot write the schedule ", settings.log);
    }
  }
  if (!settings.replay[0]) {
    char header[64];
    std::snprintf(header, sizeof header, "# seed=%llu",
        static_cast<unsigned long long>(settings.seed));
    if (schedule) {
      std::fprintf(schedule, "%s\n", header);
      std::fflush(schedule);
    } else {
      std::fprintf(stderr, "Merkurius runtime: fault: %s\n", header);
    }
  }
  faultsOn = true;
  pthread_attr_t attributes;
  pthread_attr_init(&attributes);
  pthread_attr_setdetachstate(&attributes, PTHREAD_CREATE_DETACHED);
  pthread_t thread;
  if (pthread_create(&thread, &attributes, Injector, nullptr) != 0) {
    Fail("MERK_FAULTS: cannot start the injector's thread", "");
  }
  pthread_attr_destroy(&attributes);
}

void FaultsSpawned(Process *process, Word messageType) {
  int kind{KindOf(messageType)};
  if (kind < 0) {
    return;
  }
  auto *entry{static_cast<Entry *>(std::malloc(sizeof(Entry)))};
  if (!entry) {
    Fail("out of memory", "");
  }
  pthread_mutex_lock(&lock);
  *entry = {nullptr, first, process, kind, ++spawned[kind]};
  if (first) {
    first->previous = entry;
  }
  first = entry;
  process->faultEntry = entry;
  pthread_mutex_unlock(&lock);
}

void FaultsEnded(Process *process) {
  pthread_mutex_lock(&lock);
  auto *entry{static_cast<Entry *>(process->faultEntry)};
  if (entry->previous) {
    entry->previous->next = entry->next;
  } else {
    first = entry->next;
  }
  if (entry->next) {
    entry->next->previous = entry->previous;
  }
  process->faultEntry = nullptr;
  pthread_mutex_unlock(&lock);
  std::free(entry);
}

} // namespace merkurius::runtime
