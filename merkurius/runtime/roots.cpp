//===-- runtime/roots.cpp - Finding a process's roots ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "roots.h"
#include "descriptor.h"
#include "walk.h"
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// The linker's bounds of the tables. Weak, for a program with none.
extern "C" const char __start_merk_root_types[] __attribute__((weak));
extern "C" const char __stop_merk_root_types[] __attribute__((weak));

namespace merkurius::runtime {
namespace {

// The order of Lowering.cpp's RootKind.
enum RootKind : std::int64_t {
  RootUnresolved,
  RootTerm,
  RootTypeInfo,
  RootTypeclassInfo,
  RootRef,
  RootClosure,
  RootString,
  RootArray,
  RootEnvironment,
  RootPolymorphicTerm,
  RootWord,
};

struct RootEntry {
  std::int64_t kind;
  Word pointer;
};

struct RootTable {
  const char *stackMaps;
  std::int64_t count;
  RootEntry entries[];
};

// LLVM's stack map format, version 3.
struct Location {
  std::uint8_t kind;
  std::uint8_t reserved;
  std::uint16_t size;
  std::uint16_t reg;
  std::uint16_t reserved2;
  std::int32_t offset;
};
enum LocationKind : std::uint8_t {
  Register = 1,
  Direct = 2,
  Indirect = 3,
  Constant = 4,
  ConstantIndex = 5,
};
struct Record {
  std::uint64_t id;
  std::uint32_t offset;
  std::uint16_t reserved;
  std::uint16_t locationCount;
  Location locations[];
};

struct Safepoint {
  Word address;
  const Record *record;
  const RootTable *table;
  const std::uint64_t *constants;
  std::uint64_t stackSize;
};

Safepoint *safepoints{nullptr};
std::size_t safepointMask{0};

std::atomic<std::uint64_t> safepointsChecked{0}, framesChecked{0}, rootsChecked{0},
    unresolvedRoots{0}, environmentsChecked{0}, notSafepoints{0};

[[noreturn]] void Bad(const char *what, Word value) {
  std::fflush(stdout);
  std::fprintf(stderr, "Mercury runtime: bad root: %s (%#llx)\n", what,
      static_cast<unsigned long long>(value));
  std::abort();
}

Word Hash(Word address) { return (address >> 3) * 0x9e3779b97f4a7c15ull; }

void Insert(const Safepoint &safepoint) {
  std::size_t place{Hash(safepoint.address) & safepointMask};
  while (safepoints[place].address != 0) {
    place = (place + 1) & safepointMask;
  }
  safepoints[place] = safepoint;
}

const Safepoint *Find(Word address) {
  std::size_t place{Hash(address) & safepointMask};
  while (safepoints[place].address != 0) {
    if (safepoints[place].address == address) {
      return &safepoints[place];
    }
    place = (place + 1) & safepointMask;
  }
  return nullptr;
}

const Record *NextRecord(const Record *record) {
  const char *end{reinterpret_cast<const char *>(record->locations + record->locationCount)};
  std::uintptr_t aligned{(reinterpret_cast<std::uintptr_t>(end) + 7) & ~std::uintptr_t{7}};
  const auto *liveOuts{reinterpret_cast<const std::uint16_t *>(aligned)};
  // Padding, and the number of live-outs, of 4 bytes each.
  end = reinterpret_cast<const char *>(liveOuts + 2) + 4 * liveOuts[1];
  aligned = (reinterpret_cast<std::uintptr_t>(end) + 7) & ~std::uintptr_t{7};
  return reinterpret_cast<const Record *>(aligned);
}

// Visits each of a stack map's records, with its function's address.
template <typename Visit>
void ForEachRecord(const RootTable *table, Visit visit) {
  const char *map{table->stackMaps};
  std::uint32_t functions, constants, records;
  std::memcpy(&functions, map + 4, 4);
  std::memcpy(&constants, map + 8, 4);
  std::memcpy(&records, map + 12, 4);
  const auto *function{reinterpret_cast<const std::uint64_t *>(map + 16)};
  const auto *constant{function + 3 * functions};
  const auto *record{reinterpret_cast<const Record *>(constant + constants)};
  for (std::uint32_t f{0}; f < functions; ++f, function += 3) {
    for (std::uint64_t r{0}; r < function[2]; ++r) {
      visit(function[0] + record->offset, record, constant, function[1]);
      record = NextRecord(record);
    }
  }
}

const RootTable *NextTable(const RootTable *table) {
  return reinterpret_cast<const RootTable *>(table->entries + table->count);
}

void Report() {
  std::fprintf(stderr,
      "Mercury runtime: roots: %llu safepoints, %llu frames, %llu roots, "
      "%llu unresolved, %llu environments, %llu slow paths not at a safepoint\n",
      static_cast<unsigned long long>(safepointsChecked.load()),
      static_cast<unsigned long long>(framesChecked.load()),
      static_cast<unsigned long long>(rootsChecked.load()),
      static_cast<unsigned long long>(unresolvedRoots.load()),
      static_cast<unsigned long long>(environmentsChecked.load()),
      static_cast<unsigned long long>(notSafepoints.load()));
}

void BuildSafepoints() {
  const auto *begin{reinterpret_cast<const RootTable *>(__start_merk_root_types)};
  const auto *end{reinterpret_cast<const RootTable *>(__stop_merk_root_types)};
  std::size_t count{0};
  for (const RootTable *table{begin}; table < end; table = NextTable(table)) {
    std::uint32_t records;
    std::memcpy(&records, table->stackMaps + 12, 4);
    count += records;
  }
  std::size_t capacity{16};
  while (capacity < 2 * count + 2) {
    capacity *= 2;
  }
  safepoints = static_cast<Safepoint *>(std::calloc(capacity, sizeof(Safepoint)));
  safepointMask = capacity - 1;
  for (const RootTable *table{begin}; table < end; table = NextTable(table)) {
    ForEachRecord(table, [table](Word address, const Record *record,
                             const std::uint64_t *constants,
                             std::uint64_t stackSize) {
      Insert({address, record, table, constants, stackSize});
    });
  }
  std::atexit(Report);
}

std::int64_t ConstantOf(const Safepoint &safepoint, const Location &location) {
  if (location.kind == Constant) {
    return location.offset;
  }
  if (location.kind == ConstantIndex) {
    return static_cast<std::int64_t>(safepoint.constants[location.offset]);
  }
  Bad("a stack map entry that is not a constant", location.kind);
}

// How many cells a root's check goes through, as a long list would otherwise
// take as long as a collection at each safepoint.
constexpr int kBudget{256};

void CheckTerm(Word value, const Type &type, int &budget);

void CheckField(Word value, Word type, const Environment *environment, int &budget) {
  CheckTerm(value, Resolve(type, environment), budget);
}

void CheckTerm(Word value, const Type &type, int &budget) {
  if (--budget < 0) {
    return;
  }
  const TypeCtor *ctor{type.ctor};
  switch (ctor->kind) {
  case Enum:
    if (value >= ctor->functorCount) {
      Bad("an enumeration's value out of range", value);
    }
    return;
  case NoTag:
  case Tuple:
  case Cells:
    break;
  default:
    return;
  }
  if (ctor->kind == Cells) {
    std::uint16_t index{ctor->byTag[value & kTagMask]};
    if (index != kConstantTag && index != kSharedTag &&
        index >= ctor->functorCount) {
      Bad("a tag no constructor of the type has", value);
    }
    if (index != kConstantTag && value < 4096) {
      Bad("a cell pointer that cannot be one", value);
    }
  }
  const Word *fields{nullptr};
  Word term{value};
  const Functor *functor{FunctorOf(term, ctor, fields)};
  if (!fields) {
    return;
  }
  Environment existential{type.environment.arguments, type.environment.parent,
      fields - functor->hidden};
  for (unsigned i{0}; i < functor->arity; ++i) {
    CheckField(fields[i], functor->fields[i], &existential, budget);
  }
}

constexpr int kMostTypeInfos{32};

void CheckRoot(Word value, const RootEntry &entry, const Word *infos = nullptr,
    std::int64_t count = 0) {
  rootsChecked.fetch_add(1, std::memory_order_relaxed);
  int budget{kBudget};
  switch (entry.kind) {
  case RootTerm:
    CheckTerm(value, Resolve(entry.pointer, nullptr), budget);
    return;
  case RootWord:
  case RootPolymorphicTerm:
    if (count == 0) {
      unresolvedRoots.fetch_add(1, std::memory_order_relaxed);
      return;
    }
    if (entry.kind == RootWord) {
      CheckTerm(value, Resolve(infos[0], nullptr), budget);
    } else {
      Environment environment{infos, nullptr, nullptr};
      CheckTerm(value, Resolve(entry.pointer, &environment), budget);
    }
    return;
  case RootUnresolved:
    unresolvedRoots.fetch_add(1, std::memory_order_relaxed);
    return;
  case RootTypeInfo:
    if (value < 4096 || (value & kTagMask) > 1) {
      Bad("a type_info with a tag it cannot have", value);
    }
    return;
  case RootClosure:
  case RootTypeclassInfo:
    if (value < 4096 || (value & kTagMask) != 0) {
      Bad("a closure or typeclass_info that is not a cell", value);
    }
    return;
  default:
    return;
  }
}

void CheckEnvironment(const Safepoint &safepoint, char *address, const RootEntry &entry) {
  if (entry.kind != RootEnvironment) {
    Bad("an environment whose layout is not one", entry.kind);
  }
  environmentsChecked.fetch_add(1, std::memory_order_relaxed);
  const auto *layout{reinterpret_cast<const std::int64_t *>(entry.pointer)};
  for (std::int64_t i{0}; i < layout[0]; ++i) {
    std::int64_t index{layout[1 + 2 * i]};
    if (index < 0) {
      continue;
    }
    Word value;
    std::memcpy(&value, address + layout[2 + 2 * i], sizeof value);
    // Null is a slot not filled yet, which is also a constant of tag 0.
    if (value != 0) {
      CheckRoot(value, safepoint.table->entries[index]);
    }
  }
}

void CheckFrame(const Safepoint &safepoint, char *stackPointer) {
  const Record *record{safepoint.record};
  const Location *locations{record->locations};
  std::int64_t deopt{ConstantOf(safepoint, locations[2])};
  const Location *state{locations + 3};
  std::int64_t pairs{ConstantOf(safepoint, state[0])};
  const Location *types{state + 1};
  const Location *environments{types + pairs};
  // A type's index is -1, all ones, for one to ignore, with no type_infos.
  std::int64_t typeInfos{0};
  for (std::int64_t i{0}; i < pairs; ++i) {
    std::int64_t bits{ConstantOf(safepoint, types[i])};
    if (bits >= 0) {
      typeInfos += bits >> 32;
    }
  }
  std::int64_t environmentCount{(deopt - 1 - pairs - typeInfos) / 2};
  const Location *infoAt{environments + 2 * environmentCount};
  const Location *gc{state + deopt};
  if (gc + 2 * pairs != locations + record->locationCount) {
    Bad("a stack map whose roots do not match its table", pairs);
  }
  for (std::int64_t i{0}; i < environmentCount; ++i) {
    std::int64_t layout{ConstantOf(safepoint, environments[2 * i])};
    const Location &where{environments[2 * i + 1]};
    if (where.kind != Direct) {
      Bad("an environment that is not in the frame", where.kind);
    }
    CheckEnvironment(safepoint, stackPointer + where.offset,
        safepoint.table->entries[layout]);
  }
  for (std::int64_t i{0}; i < pairs; ++i) {
    std::int64_t bits{ConstantOf(safepoint, types[i])};
    std::int64_t type{bits < 0 ? bits : bits & 0xffffffff};
    std::int64_t count{bits < 0 ? 0 : bits >> 32};
    Word infos[kMostTypeInfos];
    for (std::int64_t k{0}; k < count; ++k, ++infoAt) {
      if (k >= kMostTypeInfos) {
        continue;
      }
      if (infoAt->kind == Indirect) {
        std::memcpy(&infos[k], stackPointer + infoAt->offset, sizeof(Word));
      } else if (infoAt->kind == Constant || infoAt->kind == ConstantIndex) {
        infos[k] = static_cast<Word>(ConstantOf(safepoint, *infoAt));
      } else {
        Bad("a type_info that is not in a stack slot", infoAt->kind);
      }
    }
    const Location &derived{gc[2 * i + 1]};
    if (type < 0 || derived.kind == Constant || derived.kind == ConstantIndex) {
      continue;
    }
    if (derived.kind != Indirect) {
      Bad("a root that is not in a stack slot", derived.kind);
    }
    Word value;
    std::memcpy(&value, stackPointer + derived.offset, sizeof value);
    CheckRoot(value, safepoint.table->entries[type], infos,
        count > kMostTypeInfos ? 0 : count);
  }
}

std::atomic<int> rootsState{-1};
std::atomic<int> tableState{0};

} // namespace

bool CheckRootsEnabled() {
  int state{rootsState.load(std::memory_order_relaxed)};
  if (state < 0) {
    const char *value{std::getenv("MERK_CHECK_ROOTS")};
    state = value && *value;
    rootsState.store(state, std::memory_order_relaxed);
  }
  return state != 0;
}

void CheckRoots(void *frame) {
  int expected{0};
  if (tableState.compare_exchange_strong(expected, 1)) {
    BuildSafepoints();
    tableState.store(2);
  }
  while (tableState.load() != 2) {
  }
  auto *callee{static_cast<Word *>(frame)};
  // What the Mercury code that called merk_alloc_slow was running on.
  Word *expectedStack{callee + 2};
  bool first{true};
  for (;;) {
    const Safepoint *safepoint{Find(callee[1])};
    if (!safepoint) {
      if (first) {
        notSafepoints.fetch_add(1, std::memory_order_relaxed);
      }
      return;
    }
    if (first) {
      safepointsChecked.fetch_add(1, std::memory_order_relaxed);
    }
    framesChecked.fetch_add(1, std::memory_order_relaxed);
    auto *framePointer{reinterpret_cast<Word *>(callee[0])};
    // The frame's stack pointer, from its frame pointer, as a tail-calling
    // callee may have popped arguments.
    char *stackPointer{reinterpret_cast<char *>(framePointer) -
        (safepoint->stackSize - sizeof(Word))};
    if (first && stackPointer != reinterpret_cast<char *>(expectedStack)) {
      Bad("a frame whose size is not what the stack map says",
          reinterpret_cast<Word>(stackPointer));
    }
    CheckFrame(*safepoint, stackPointer);
    callee = framePointer;
    first = false;
  }
}

} // namespace merkurius::runtime
