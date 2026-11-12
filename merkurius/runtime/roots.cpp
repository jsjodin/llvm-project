//===-- runtime/roots.cpp - Finding a process's roots ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "roots.h"
#include "array.h"
#include "descriptor.h"
#include "fragment.h"
#include "frame-info.h"
#include "heap.h"
#include "process.h"
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
  RootBits,
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
  // The bytes the frame reserves between its return address and its saved
  // frame pointer for its tail calls' stack arguments (frame-info.h), which
  // stackSize includes.
  std::uint64_t reserve;
};

Safepoint *safepoints{nullptr};
std::size_t safepointMask{0};

std::atomic<std::uint64_t> safepointsChecked{0}, framesChecked{0}, rootsChecked{0},
    environmentsChecked{0}, notSafepoints{0};

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
      "%llu environments, %llu slow paths not at a safepoint\n",
      static_cast<unsigned long long>(safepointsChecked.load()),
      static_cast<unsigned long long>(framesChecked.load()),
      static_cast<unsigned long long>(rootsChecked.load()),
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
    ForEachRecord(table,
        [table](Word address, const Record *record,
            const std::uint64_t *constants, std::uint64_t stackSize) {
          std::int64_t reserve{ReserveAt(address)};
          if (reserve < 0) {
            Bad("a safepoint whose frame has no frame pointer rule in "
                ".eh_frame",
                address);
          }
          Insert({address, record, table, constants, stackSize,
              static_cast<std::uint64_t>(reserve)});
        });
  }
  if (CheckRootsEnabled()) {
    std::atexit(Report);
  }
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
// The stack map's index of a root to ignore, and of one only to update.
constexpr std::int64_t kIgnored{-1};
constexpr std::int64_t kUpdateOnly{-2};
// A process's first collection comes after this many bytes.
constexpr std::size_t kFirstCollectBytes{4 << 20};
// A collection walks the whole stack, so that the allocation between two
// pays for it. A round number, not tuned: 4 took the deep stack benchmark from
// 16.2 s to 3.5 s (design document, Measurements).
constexpr std::size_t kStackFactor{4};
constexpr std::size_t kMostToChunkBytes{1 << 20};
constexpr std::size_t kKeepBytes{1 << 16};

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
      Bad("a polymorphic root with no type_info", value);
    }
    if (entry.kind == RootWord) {
      CheckTerm(value, Resolve(infos[0], nullptr), budget);
    } else {
      Environment environment{infos, nullptr, nullptr};
      CheckTerm(value, Resolve(entry.pointer, &environment), budget);
    }
    return;
  case RootUnresolved:
    Bad("a root whose type has no descriptor", value);
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

// What a frame's environment and its roots are, for each of which `visit` is
// called: where it is, its type's index, negative for none, and its type_infos.
template <typename Visit>
void VisitEnvironment(
    const Safepoint &safepoint, char *address, const RootEntry &entry, Visit visit) {
  if (entry.kind != RootEnvironment) {
    Bad("an environment whose layout is not one", entry.kind);
  }
  environmentsChecked.fetch_add(1, std::memory_order_relaxed);
  const auto *layout{reinterpret_cast<const std::int64_t *>(entry.pointer)};
  const std::int64_t *field{layout + 1};
  for (std::int64_t i{0}; i < layout[0]; ++i) {
    std::int64_t index{field[0]};
    std::int64_t offset{field[1]};
    std::int64_t count{field[2]};
    const std::int64_t *places{field + 3};
    field = places + count;
    if (index < 0) {
      continue;
    }
    auto *slot{reinterpret_cast<Word *>(address + offset)};
    // Null is a slot not filled yet, which is also a constant of tag 0.
    if (*slot == 0) {
      continue;
    }
    if (count > kMostTypeInfos) {
      Bad("an environment's field with too many type_infos", count);
    }
    Word infos[kMostTypeInfos];
    for (std::int64_t k{0}; k < count; ++k) {
      std::memcpy(&infos[k], address + places[k], sizeof(Word));
    }
    visit(slot, index, infos, count);
  }
}

// The frame pointer of a frame whose stack pointer is `stackPointer`: below
// the return address, the reserve for tail calls' arguments, and the saved
// frame pointer.
char *FramePointerOf(const Safepoint &safepoint, char *stackPointer) {
  return stackPointer +
      (safepoint.stackSize - safepoint.reserve - sizeof(Word));
}

// Where a location of a frame is. A frame that pushes the arguments of a call
// has its slots at the frame pointer, as its stack pointer varies.
char *BaseOf(const Safepoint &safepoint, const Location &location,
    char *stackPointer) {
  constexpr std::uint16_t kFramePointer{6};
  constexpr std::uint16_t kStackPointer{7};
  if (location.reg == kFramePointer) {
    return FramePointerOf(safepoint, stackPointer);
  }
  if (location.reg != kStackPointer || safepoint.stackSize == ~std::uint64_t{0}) {
    Bad("a stack map location not in a frame of a known size", location.reg);
  }
  return stackPointer;
}

template <typename Visit>
void VisitFrame(const Safepoint &safepoint, char *stackPointer, Visit visit) {
  const Record *record{safepoint.record};
  const Location *locations{record->locations};
  std::int64_t deopt{ConstantOf(safepoint, locations[2])};
  const Location *state{locations + 3};
  // Above the number of pairs, how many heap pointers the state has after the
  // type_infos, which are dead after the call.
  std::int64_t counts{ConstantOf(safepoint, state[0])};
  std::int64_t pairs{counts & 0xffffffff};
  const Location *types{state + 1};
  const Location *environments{types + pairs};
  // A negative type's index is one to ignore or only to update, with no
  // type_infos.
  std::int64_t typeInfos{0};
  for (std::int64_t i{0}; i < pairs; ++i) {
    std::int64_t bits{ConstantOf(safepoint, types[i])};
    if (bits >= 0) {
      typeInfos += (bits >> 32) & 0xffffff;
    }
  }
  std::int64_t environmentCount{(deopt - 1 - pairs - typeInfos - (counts >> 32)) / 2};
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
    VisitEnvironment(safepoint,
        BaseOf(safepoint, where, stackPointer) + where.offset,
        safepoint.table->entries[layout], visit);
  }
  for (std::int64_t i{0}; i < pairs; ++i) {
    std::int64_t bits{ConstantOf(safepoint, types[i])};
    std::int64_t type{bits < 0 ? bits : bits & 0xffffffff};
    std::int64_t count{bits < 0 ? 0 : (bits >> 32) & 0xffffff};
    // Above them, the tag of a root that is its term's untagged cell.
    Word tag{bits < 0 ? Word{0} : static_cast<Word>(bits >> 56) & kTagMask};
    if (count > kMostTypeInfos) {
      Bad("a root with too many type_infos", count);
    }
    Word infos[kMostTypeInfos];
    for (std::int64_t k{0}; k < count; ++k, ++infoAt) {
      if (infoAt->kind == Indirect) {
        std::memcpy(&infos[k],
            BaseOf(safepoint, *infoAt, stackPointer) + infoAt->offset,
            sizeof(Word));
      } else if (infoAt->kind == Constant || infoAt->kind == ConstantIndex) {
        infos[k] = static_cast<Word>(ConstantOf(safepoint, *infoAt));
      } else {
        Bad("a type_info that is not in a stack slot", infoAt->kind);
      }
    }
    const Location &derived{gc[2 * i + 1]};
    if (type == kIgnored || derived.kind == Constant ||
        derived.kind == ConstantIndex) {
      continue;
    }
    if (derived.kind != Indirect) {
      Bad("a root that is not in a stack slot", derived.kind);
    }
    auto *slot{reinterpret_cast<Word *>(
        BaseOf(safepoint, derived, stackPointer) + derived.offset)};
    if (tag == 0) {
      visit(slot, type, infos, count);
    } else {
      Word term{*slot + tag};
      visit(&term, type, infos, count);
      *slot = term - tag;
    }
  }
}

// The frames from that of merk_alloc_slow, or false if its caller is not at a
// safepoint. `visit` takes each frame's safepoint and stack pointer.
template <typename Visit>
bool VisitFrames(void *frame, Visit visit) {
  // The callee's frame pointer, and where its return address is: next to
  // its saved frame pointer, or above the reserve for its tail calls'
  // arguments (frame-info.h). merk_alloc_slow reserves none.
  auto *callee{static_cast<Word *>(frame)};
  Word *returnAddress{callee + 1};
  bool first{true};
  for (;;) {
    const Safepoint *safepoint{Find(*returnAddress)};
    if (!safepoint) {
      return !first;
    }
    auto *framePointer{reinterpret_cast<Word *>(callee[0])};
    // The frame's stack pointer, from its frame pointer, as a tail-calling
    // callee may have popped arguments.
    char *stackPointer{reinterpret_cast<char *>(framePointer) -
        (safepoint->stackSize - safepoint->reserve - sizeof(Word))};
    visit(*safepoint, stackPointer);
    callee = framePointer;
    returnAddress =
        reinterpret_cast<Word *>(reinterpret_cast<char *>(framePointer) +
            sizeof(Word) + safepoint->reserve);
    first = false;
  }
}

// A copying collection of a process's heap. Cells have no header, so what is
// reached is copied by its type, a forwarding address is left in the first
// word of the old cell, and a side bitmap says which old words hold one.
class Collector {
public:
  explicit Collector(Process *process) : process_{process} { Prepare(); }

  void Root(Word *slot, std::int64_t type, const Safepoint &safepoint,
      const Word *infos, std::int64_t count);
  void UpdateOnly(Word *slot);
  void Drain();
  void Finish();
  // A frame of this many bytes was walked.
  void CountStack(std::uint64_t bytes) { stackBytes_ += bytes; }

private:
  bool KeepLarge(ChunkLink *chunk);
  struct FromChunk {
    char *begin;
    char *end;
    std::uint64_t *marks;
    bool large;
  };
  // What is left to do of a cell in to-space: its fields, each of its type in
  // `types`, or all of `single`'s for an array.
  struct Pending {
    Word *to;
    const Word *types;
    Word single;
    const Environment *environment;
    std::uint64_t count;
  };
  struct Block {
    Block *next;
  };

  static int CompareChunks(const void *a, const void *b);
  void Prepare();
  FromChunk *FindFrom(Word address);
  bool Forwarded(FromChunk *, const Word *cell, Word *&to) const;
  void Forward(FromChunk *, Word *cell, Word *to);
  Word *ToSpace(std::size_t words);
  Word *CopyOut(
      FromChunk *chunk, const Word *from, std::size_t words, std::size_t bytes);
  void *Keep(std::size_t bytes);
  const Environment *KeepEnvironment(const Environment &);
  void Push(const Pending &);

  Word ForwardTypeInfo(Word info);
  Word ForwardDictionary(Word dictionary);
  void Trace(Word *slot, Word type, const Environment *environment);
  void ForwardValue(Word *slot, const Type &type);
  void ForwardCell(Word *slot, const Type &type);
  void ForwardClosure(Word *slot);
  void ForwardString(Word *slot);
  void ForwardArray(Word *slot, const Type *type);

  Process *process_;
  FromChunk *chunks_{nullptr};
  std::size_t chunkCount_{0};
  ChunkLink *toChunks_{nullptr};
  char *toNext_{nullptr};
  char *toLimit_{nullptr};
  std::size_t toChunkBytes_{4096};
  std::size_t liveBytes_{0};
  std::size_t stackBytes_{0};
  Pending *pending_{nullptr};
  std::size_t pendingCount_{0};
  std::size_t pendingCapacity_{0};
  Block *blocks_{nullptr};
  char *keepNext_{nullptr};
  char *keepLimit_{nullptr};
  const Environment *lastEnvironment_{nullptr};
};

int Collector::CompareChunks(const void *a, const void *b) {
  const char *x{static_cast<const FromChunk *>(a)->begin};
  const char *y{static_cast<const FromChunk *>(b)->begin};
  return x < y ? -1 : x > y ? 1 : 0;
}

void Collector::Prepare() {
  for (ChunkLink *chunk{process_->chunks}; chunk; chunk = chunk->next) {
    chunkCount_ += !chunk->pinned;
  }
  chunks_ = static_cast<FromChunk *>(
      Malloc(sizeof(FromChunk) * (chunkCount_ + 1)));
  std::size_t i{0};
  for (ChunkLink *chunk{process_->chunks}; chunk; chunk = chunk->next) {
    if (chunk->pinned) {
      continue;
    }
    char *begin{reinterpret_cast<char *>(chunk + 1)};
    auto *marks{static_cast<std::uint64_t *>(
        std::calloc(chunk->bytes / 512 + 1, sizeof(std::uint64_t)))};
    if (!marks) {
      Bad("no memory for a collection's marks", chunk->bytes);
    }
    chunks_[i++] = {begin, begin + chunk->bytes, marks, chunk->large};
  }
  std::qsort(chunks_, chunkCount_, sizeof(FromChunk), CompareChunks);
}

Collector::FromChunk *Collector::FindFrom(Word address) {
  const char *at{reinterpret_cast<const char *>(address)};
  std::size_t low{0}, high{chunkCount_};
  while (low < high) {
    std::size_t middle{low + (high - low) / 2};
    if (at < chunks_[middle].begin) {
      high = middle;
    } else if (at >= chunks_[middle].end) {
      low = middle + 1;
    } else {
      return &chunks_[middle];
    }
  }
  return nullptr;
}

bool Collector::Forwarded(FromChunk *chunk, const Word *cell, Word *&to) const {
  std::size_t index{static_cast<std::size_t>(
      reinterpret_cast<const char *>(cell) - chunk->begin) / sizeof(Word)};
  if ((chunk->marks[index / 64] >> (index % 64) & 1) == 0) {
    return false;
  }
  to = chunk->large ? const_cast<Word *>(cell) : reinterpret_cast<Word *>(*cell);
  return true;
}

void Collector::Forward(FromChunk *chunk, Word *cell, Word *to) {
  std::size_t index{static_cast<std::size_t>(
      reinterpret_cast<const char *>(cell) - chunk->begin) / sizeof(Word)};
  chunk->marks[index / 64] |= std::uint64_t{1} << (index % 64);
  if (!chunk->large) {
    *cell = reinterpret_cast<Word>(to);
  }
}

// Where a reached object is from now on. A large object stays where it is.
Word *Collector::CopyOut(
    FromChunk *chunk, const Word *from, std::size_t words, std::size_t bytes) {
  if (chunk->large) {
    liveBytes_ += words * sizeof(Word);
    return const_cast<Word *>(from);
  }
  Word *to{ToSpace(words)};
  std::memcpy(to, from, bytes);
  return to;
}

Word *Collector::ToSpace(std::size_t words) {
  std::size_t bytes{words * sizeof(Word)};
  liveBytes_ += bytes;
  if (static_cast<std::size_t>(toLimit_ - toNext_) < bytes) {
    std::size_t size{bytes > toChunkBytes_ ? bytes : toChunkBytes_};
    if (toChunkBytes_ < kMostToChunkBytes) {
      toChunkBytes_ *= 2;
    }
    auto *chunk{static_cast<ChunkLink *>(Malloc(sizeof(ChunkLink) + size))};
    *chunk = {toChunks_, size, nullptr, false};
    toChunks_ = chunk;
    toNext_ = reinterpret_cast<char *>(chunk + 1);
    toLimit_ = toNext_ + size;
  }
  auto *cell{reinterpret_cast<Word *>(toNext_)};
  toNext_ += bytes;
  return cell;
}

// Memory for the rest of the collection.
void *Collector::Keep(std::size_t bytes) {
  if (static_cast<std::size_t>(keepLimit_ - keepNext_) < bytes) {
    std::size_t size{bytes > kKeepBytes ? bytes : kKeepBytes};
    auto *block{static_cast<Block *>(Malloc(sizeof(Block) + 8 + size))};
    block->next = blocks_;
    blocks_ = block;
    keepNext_ = reinterpret_cast<char *>(block) + sizeof(Block) + 8;
    keepLimit_ = keepNext_ + size;
  }
  void *kept{keepNext_};
  keepNext_ += bytes;
  return kept;
}

// One lasting as long as what is pending. A long list is all one, so that its
// length costs nothing here.
const Environment *Collector::KeepEnvironment(const Environment &environment) {
  if (lastEnvironment_ && lastEnvironment_->arguments == environment.arguments &&
      lastEnvironment_->parent == environment.parent &&
      lastEnvironment_->cell == environment.cell) {
    return lastEnvironment_;
  }
  auto *kept{static_cast<Environment *>(Keep(sizeof(Environment)))};
  *kept = environment;
  lastEnvironment_ = kept;
  return kept;
}

void Collector::Push(const Pending &entry) {
  if (pendingCount_ == pendingCapacity_) {
    pendingCapacity_ = pendingCapacity_ ? 2 * pendingCapacity_ : 256;
    auto *grown{static_cast<Pending *>(
        std::realloc(pending_, sizeof(Pending) * pendingCapacity_))};
    if (!grown) {
      Bad("no memory for a collection's work", pendingCapacity_);
    }
    pending_ = grown;
  }
  pending_[pendingCount_++] = entry;
}

Word Collector::ForwardTypeInfo(Word info) {
  if ((info & kTagMask) != 1) {
    return info; // A descriptor, which is static.
  }
  auto *cell{reinterpret_cast<Word *>(info - 1)};
  FromChunk *chunk{FindFrom(info - 1)};
  if (!chunk) {
    return info;
  }
  Word *to;
  if (Forwarded(chunk, cell, to)) {
    return reinterpret_cast<Word>(to) + 1;
  }
  const auto *ctor{reinterpret_cast<const TypeCtor *>(cell[0])};
  bool counted{ctor->kind == Tuple || ctor->kind == Closure};
  Word arity{counted ? cell[1] : ctor->arity};
  Word before{counted ? Word{2} : Word{1}};
  to = CopyOut(chunk, cell, before + arity, sizeof(Word) * (before + arity));
  Forward(chunk, cell, to);
  // As deep as the type is written, so a call is enough.
  for (Word i{0}; i < arity; ++i) {
    to[before + i] = ForwardTypeInfo(to[before + i]);
  }
  return reinterpret_cast<Word>(to) + 1;
}

Word Collector::ForwardDictionary(Word dictionary) {
  auto *cell{reinterpret_cast<Word *>(dictionary)};
  FromChunk *chunk{FindFrom(dictionary)};
  if (!chunk) {
    return dictionary;
  }
  Word *to;
  if (Forwarded(chunk, cell, to)) {
    return reinterpret_cast<Word>(to);
  }
  DictionaryShape shape{ShapeOfDictionary(cell)};
  Word total{shape.superclasses + shape.typeInfos + shape.constraints};
  to = CopyOut(chunk, cell, 1 + total, sizeof(Word) * (1 + total));
  Forward(chunk, cell, to);
  for (Word i{1}; i <= total; ++i) {
    bool isTypeInfo{i > shape.superclasses &&
        i <= shape.superclasses + shape.typeInfos};
    to[i] = isTypeInfo ? ForwardTypeInfo(to[i]) : ForwardDictionary(to[i]);
  }
  return reinterpret_cast<Word>(to);
}

void Collector::Trace(Word *slot, Word type, const Environment *environment) {
  Type resolved{Resolve(type, environment)};
  // A field whose type's arguments are all type parameters, as a list's tail
  // has, is resolved here. Otherwise each cell would add to the chain of
  // environments and a long list would cost its length squared.
  if (environment && (type & kTagMask) == 1 && resolved.ctor->kind != Tuple &&
      resolved.ctor->kind != Closure && resolved.ctor->arity != 0) {
    const Word *arguments{resolved.environment.arguments};
    unsigned arity{resolved.ctor->arity};
    bool variables{true};
    bool same{true};
    for (unsigned i{0}; i < arity; ++i) {
      variables = variables && (arguments[i] & kTagMask) == kVariableTag;
      same = same && arguments[i] == ((Word{i} << kTagBits) | kVariableTag);
    }
    if (same) {
      resolved.environment = {environment->arguments, environment->parent};
    } else if (variables) {
      auto *flat{static_cast<Word *>(Keep(sizeof(Word) * arity))};
      for (unsigned i{0}; i < arity; ++i) {
        flat[i] = environment->arguments[arguments[i] >> kTagBits];
      }
      resolved.environment = {flat, environment->parent};
    }
  }
  ForwardValue(slot, resolved);
}

void Collector::ForwardValue(Word *slot, const Type &type) {
  switch (type.ctor->kind) {
  case String:
    ForwardString(slot);
    return;
  case Array:
    ForwardArray(slot, &type);
    return;
  case Closure:
    ForwardClosure(slot);
    return;
  case NoTag:
  case Tuple:
  case Cells:
    ForwardCell(slot, type);
    return;
  case Void:
    // Nothing determined its type, so nothing can have made such a value.
    if (FindFrom(*slot & ~kTagMask)) {
      Bad("a value of the type void that points into the heap", *slot);
    }
    return;
  default:
    return; // A word that is not a pointer.
  }
}

void Collector::ForwardString(Word *slot) {
  Word value{*slot};
  FromChunk *chunk{FindFrom(value)};
  if (!chunk) {
    return; // Null, or a literal.
  }
  if ((value & 7) != 0) {
    Bad("a string in the heap that is not at a word", value);
  }
  auto *text{reinterpret_cast<Word *>(value)};
  Word *to;
  if (Forwarded(chunk, text, to)) {
    *slot = reinterpret_cast<Word>(to);
    return;
  }
  std::size_t bytes{std::strlen(reinterpret_cast<const char *>(text)) + 1};
  to = CopyOut(chunk, text, (bytes + 7) / sizeof(Word), bytes);
  Forward(chunk, text, to);
  *slot = reinterpret_cast<Word>(to);
}

// `type` is null for an array of numbers.
void Collector::ForwardArray(Word *slot, const Type *type) {
  Word value{*slot};
  FromChunk *chunk{FindFrom(value)};
  if (!chunk) {
    return;
  }
  auto *array{reinterpret_cast<Word *>(value)};
  Word *to;
  if (Forwarded(chunk, array, to)) {
    *slot = reinterpret_cast<Word>(to);
    return;
  }
  Word size{ArraySize(array)};
  Word words{ArrayWords(array)};
  to = CopyOut(chunk, array, words, sizeof(Word) * words);
  Forward(chunk, array, to);
  *slot = reinterpret_cast<Word>(to);
  // Narrow elements are numbers (array.h).
  if (type && size != 0 && IsWordArray(to)) {
    const Environment *kept{KeepEnvironment(type->environment)};
    Word element{(Word{0} << kTagBits) | kVariableTag};
    Push({to + 1, nullptr, element, kept, size});
  }
}

void Collector::ForwardClosure(Word *slot) {
  Word value{*slot};
  FromChunk *chunk{FindFrom(value)};
  if (!chunk) {
    return;
  }
  auto *cell{reinterpret_cast<Word *>(value)};
  Word *to;
  if (Forwarded(chunk, cell, to)) {
    *slot = reinterpret_cast<Word>(to);
    return;
  }
  const auto *layout{reinterpret_cast<const ClosureLayout *>(cell[1])};
  if (!layout) {
    Bad("a closure made on a device in a process's heap", value);
  }
  to = CopyOut(chunk, cell, 2 + layout->count, sizeof(Word) * (2 + layout->count));
  Forward(chunk, cell, to);
  *slot = reinterpret_cast<Word>(to);
  Word *captured{to + 2};
  // First, as the types of the others read them.
  for (Word i{0}; i < layout->count; ++i) {
    if (layout->captures[i] == kCaptureTypeInfo) {
      captured[i] = ForwardTypeInfo(captured[i]);
    } else if (layout->captures[i] == kCaptureTypeclassInfo) {
      captured[i] = ForwardDictionary(captured[i]);
    }
  }
  if (layout->count != 0) {
    Push({captured, layout->captures, 0,
        KeepEnvironment({nullptr, nullptr, captured}), layout->count});
  }
}

void Collector::ForwardCell(Word *slot, const Type &type) {
  const TypeCtor *ctor{type.ctor};
  Word value{*slot};
  const Functor *functor{nullptr};
  unsigned tag{0};
  if (ctor->kind == NoTag ||
      (ctor->kind == Tuple && ctor->functors[0].arity == 1)) {
    // The value is its only field.
    Trace(slot, ctor->functors[0].fields[0], KeepEnvironment(type.environment));
    return;
  }
  if (ctor->kind == Tuple) {
    functor = &ctor->functors[0];
    tag = functor->tag;
  } else {
    tag = static_cast<unsigned>(value & kTagMask);
    if (ctor->byTag[tag] == kConstantTag) {
      if (FindFrom(value & ~kTagMask)) {
        Bad("a constant that points into the heap", value);
      }
      return;
    }
  }
  FromChunk *chunk{FindFrom(value - tag)};
  if (!chunk) {
    return; // Null, in a hole not yet filled, or a static cell.
  }
  auto *cell{reinterpret_cast<Word *>(value - tag)};
  Word *to;
  if (Forwarded(chunk, cell, to)) {
    *slot = reinterpret_cast<Word>(to) + tag;
    return;
  }
  unsigned before;
  if (ctor->kind == Tuple) {
    before = 0;
  } else {
    std::uint16_t index{ctor->byTag[tag]};
    if (index == kSharedTag) {
      functor = &ctor->functors[ctor->shared[cell[0]]];
      before = 1 + functor->hidden;
    } else {
      functor = &ctor->functors[index];
      before = functor->hidden;
    }
  }
  unsigned arity{functor->arity};
  to = CopyOut(chunk, cell, before + arity, sizeof(Word) * (before + arity));
  Forward(chunk, cell, to);
  *slot = reinterpret_cast<Word>(to) + tag;
  Word *fields{to + before};
  unsigned hidden{functor->hidden};
  Word *hiddenTo{fields - hidden};
  for (unsigned i{0}; i < hidden; ++i) {
    hiddenTo[i] = i < functor->hiddenTypeInfos
        ? ForwardTypeInfo(hiddenTo[i])
        : ForwardDictionary(hiddenTo[i]);
  }
  if (arity != 0) {
    Environment environment{type.environment.arguments,
        type.environment.parent, hidden != 0 ? hiddenTo : nullptr};
    Push({fields, functor->fields, 0, KeepEnvironment(environment), arity});
  }
}

void Collector::Drain() {
  while (pendingCount_ != 0) {
    Pending &top{pending_[pendingCount_ - 1]};
    Word *slot{top.to};
    Word type{top.types ? *top.types : top.single};
    const Environment *environment{top.environment};
    if (--top.count == 0) {
      // Before the last field, so that a long list takes no more room.
      --pendingCount_;
    } else {
      ++top.to;
      if (top.types) {
        ++top.types;
      }
    }
    if (type != kCaptureTypeInfo && type != kCaptureTypeclassInfo) {
      Trace(slot, type, environment);
    }
  }
}

void Collector::Root(Word *slot, std::int64_t type, const Safepoint &safepoint,
    const Word *infos, std::int64_t count) {
  const RootEntry &entry{safepoint.table->entries[type]};
  Word forwarded[kMostTypeInfos];
  for (std::int64_t k{0}; k < count; ++k) {
    forwarded[k] = ForwardTypeInfo(infos[k]);
  }
  switch (entry.kind) {
  case RootTerm:
    ForwardValue(slot, Resolve(entry.pointer, nullptr));
    break;
  case RootWord:
    if (count == 0) {
      Bad("a polymorphic root with no type_info", *slot);
    }
    ForwardValue(slot, Resolve(forwarded[0], nullptr));
    break;
  case RootPolymorphicTerm: {
    if (count == 0) {
      Bad("a polymorphic root with no type_info", *slot);
    }
    auto *arguments{static_cast<Word *>(Keep(sizeof(Word) * count))};
    std::memcpy(arguments, forwarded, sizeof(Word) * count);
    Trace(slot, entry.pointer, KeepEnvironment({arguments, nullptr, nullptr}));
    break;
  }
  case RootTypeInfo:
    *slot = ForwardTypeInfo(*slot);
    break;
  case RootTypeclassInfo:
    *slot = ForwardDictionary(*slot);
    break;
  case RootClosure:
    ForwardClosure(slot);
    break;
  case RootString:
    ForwardString(slot);
    break;
  case RootArray:
    ForwardArray(slot, nullptr);
    break;
  case RootRef:
  case RootBits:
    break; // RootRef: UpdateOnly, afterwards.
  default:
    Bad("a root of a kind that cannot be collected", entry.kind);
  }
  Drain();
}

// A pointer to a cell that some typed root reaches: it follows that cell.
void Collector::UpdateOnly(Word *slot) {
  Word value{*slot};
  auto *cell{reinterpret_cast<Word *>(value & ~kTagMask)};
  FromChunk *chunk{FindFrom(value & ~kTagMask)};
  if (!chunk) {
    return;
  }
  Word *to;
  if (!Forwarded(chunk, cell, to)) {
    Bad("an update-only root whose cell nothing else reaches", value);
  }
  *slot = reinterpret_cast<Word>(to) + (value & kTagMask);
}

// Whether a chunk is a large object that was reached, which its first word's
// mark says, as it holds the one object.
bool Collector::KeepLarge(ChunkLink *chunk) {
  return chunk->large &&
      (FindFrom(reinterpret_cast<Word>(chunk + 1))->marks[0] & 1) != 0;
}

void Collector::Finish() {
  ChunkLink *kept{toChunks_};
  bool poison{CollectsEveryAllocation()};
  for (ChunkLink *chunk{process_->chunks}; chunk;) {
    ChunkLink *next{chunk->next};
    if (chunk->pinned || KeepLarge(chunk)) {
      chunk->next = kept;
      kept = chunk;
    } else {
      if (poison) {
        std::memset(chunk + 1, 0xdb, chunk->bytes);
      }
      std::free(chunk);
    }
    chunk = next;
  }
  process_->chunks = kept;
  for (std::size_t i{0}; i < chunkCount_; ++i) {
    std::free(chunks_[i].marks);
  }
  std::free(chunks_);
  std::free(pending_);
  while (blocks_) {
    Block *next{blocks_->next};
    std::free(blocks_);
    blocks_ = next;
  }
  merk_heap = poison ? MerkHeap{nullptr, nullptr} : MerkHeap{toNext_, toLimit_};
  process_->allocatedBytes = 0;
  // Enough allocation to pay for the next collection's work, as well as for
  // what the heap has: it copies what is live and walks the whole stack.
  std::size_t due{2 * liveBytes_ > kFirstCollectBytes ? 2 * liveBytes_
                                                      : kFirstCollectBytes};
  process_->collectAtBytes =
      kStackFactor * stackBytes_ > due ? kStackFactor * stackBytes_ : due;
}

std::atomic<int> rootsState{-1};
std::atomic<int> everyState{-1};
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

bool CollectsEveryAllocation() {
  int state{everyState.load(std::memory_order_relaxed)};
  if (state < 0) {
    const char *value{std::getenv("MERK_GC_EVERY_ALLOC")};
    state = value && *value ? std::atoi(value) : 0;
    everyState.store(state < 1 && value && *value ? 1 : state,
        std::memory_order_relaxed);
    state = everyState.load(std::memory_order_relaxed);
  }
  return state != 0;
}

namespace {
// Whether this allocation is the one of each N that MERK_GC_EVERY_ALLOC=N
// collects at, which reaches programs too long for a collection at each.
thread_local std::uint64_t allocationsSeen{0};

bool IsNthAllocation() {
  return allocationsSeen++ %
      static_cast<std::uint64_t>(everyState.load(std::memory_order_relaxed)) == 0;
}
} // namespace

namespace {
void EnsureSafepoints() {
  int expected{0};
  if (tableState.compare_exchange_strong(expected, 1)) {
    BuildSafepoints();
    tableState.store(2);
  }
  while (tableState.load() != 2) {
  }
}
} // namespace

void CheckRoots(void *frame) {
  EnsureSafepoints();
  bool first{true};
  bool safe{VisitFrames(frame, [&](const Safepoint &safepoint, char *stack) {
    if (first) {
      safepointsChecked.fetch_add(1, std::memory_order_relaxed);
      first = false;
    }
    framesChecked.fetch_add(1, std::memory_order_relaxed);
    VisitFrame(safepoint, stack,
        [&](Word *slot, std::int64_t type, const Word *infos, std::int64_t count) {
          if (type >= 0) {
            CheckRoot(*slot, safepoint.table->entries[type], infos, count);
          }
        });
  })};
  if (!safe) {
    notSafepoints.fetch_add(1, std::memory_order_relaxed);
  }
}

bool CollectIfDue(void *frame) {
  Process *process{CurrentProcess()};
  if (!process || !MayCollect(process)) {
    return false;
  }
  if (CollectsEveryAllocation()) {
    if (!IsNthAllocation()) {
      return false;
    }
  } else {
    std::size_t due{process->collectAtBytes ? process->collectAtBytes
                                            : kFirstCollectBytes};
    if (process->allocatedBytes < due) {
      return false;
    }
  }
  EnsureSafepoints();
  auto *callee{static_cast<Word *>(frame)};
  if (!Find(callee[1])) {
    return false;
  }
  Collector collector{process};
  VisitFrames(frame, [&](const Safepoint &safepoint, char *stack) {
    if (safepoint.stackSize != ~std::uint64_t{0}) {
      collector.CountStack(safepoint.stackSize);
    }
    VisitFrame(safepoint, stack,
        [&](Word *slot, std::int64_t type, const Word *infos, std::int64_t count) {
          if (type >= 0) {
            collector.Root(slot, type, safepoint, infos, count);
          }
        });
  });
  // After every typed root has gone, as these follow what those moved.
  VisitFrames(frame, [&](const Safepoint &safepoint, char *stack) {
    VisitFrame(safepoint, stack,
        [&](Word *slot, std::int64_t type, const Word *, std::int64_t) {
          if (type == kUpdateOnly ||
              (type >= 0 && safepoint.table->entries[type].kind == RootRef)) {
            collector.UpdateOnly(slot);
          }
        });
  });
  collector.Finish();
  return true;
}

} // namespace merkurius::runtime
