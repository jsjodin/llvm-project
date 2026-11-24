//===-- runtime/open.cpp - The open layout of terms -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "open.h"
#include "heap.h"
#include "process.h"
#include "trail.h"
#include "walk.h"
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <pthread.h>

namespace merkurius::runtime {

namespace {
Word *CellOf(Word reference) {
  return reinterpret_cast<Word *>(reference & ~kTagMask);
}

// The trail, allocated on its first use.
MerkTrail &Trail() {
  if (!merk_trail) {
    merk_trail = static_cast<MerkTrail *>(std::calloc(1, sizeof(MerkTrail)));
    if (!merk_trail) {
      std::fputs("merk: out of memory for the trail\n", stderr);
      std::abort();
    }
  }
  return *merk_trail;
}

[[gnu::noinline]] void GrowMarks(MerkTrail &trail) {
  std::uint64_t capacity{trail.markCapacity ? 2 * trail.markCapacity : 64};
  auto *marks{static_cast<std::uint64_t *>(
      std::realloc(trail.marks, capacity * sizeof(std::uint64_t)))};
  if (!marks) {
    std::fputs("merk: out of memory for the trail\n", stderr);
    std::abort();
  }
  trail.marks = marks;
  trail.markCapacity = capacity;
}

[[gnu::noinline]] void Grow(MerkTrail &trail) {
  std::uint64_t capacity{trail.capacity ? 2 * trail.capacity : 1024};
  auto *entries{static_cast<MerkTrailEntry *>(
      std::realloc(trail.entries, capacity * sizeof(MerkTrailEntry)))};
  if (!entries) {
    std::fputs("merk: out of memory for the trail\n", stderr);
    std::abort();
  }
  trail.entries = entries;
  trail.capacity = capacity;
}
} // namespace

} // namespace merkurius::runtime

using namespace merkurius::runtime;

extern "C" {
uint64_t merk_open_new_var(void) {
  auto *cell{static_cast<Word *>(Allocate(sizeof(Word)))};
  Word reference{reinterpret_cast<Word>(cell) | kOpenVariableTag};
  *cell = reference;
  return reference;
}

uint64_t merk_open_box(uint64_t word) {
  auto *cell{static_cast<Word *>(Allocate(sizeof(Word)))};
  *cell = word;
  return reinterpret_cast<Word>(cell);
}

uint64_t merk_open_unbox(uint64_t box) {
  return *reinterpret_cast<const Word *>(box);
}

uint64_t merk_open_deref(uint64_t word) {
  while (IsVariableReference(word)) {
    Word next{*CellOf(word)};
    if (next == word) {
      break;
    }
    word = next;
  }
  return word;
}

uint64_t merk_open_closed_value(uint64_t word) {
  Word value{merk_open_deref(word)};
  if (IsUnbound(value)) {
    std::fflush(stdout);
    std::fputs("Merkurius runtime: a closed(T) value is unbound where it must "
               "be ground\n",
        stderr);
    ExitProgram(1);
  }
  return *reinterpret_cast<const Word *>(value); // a box, tag 0
}

void merk_open_bind(uint64_t variable, uint64_t value) {
  Word *cell{CellOf(variable)};
  // Its caller dereferences it first. A variable bound already would have
  // its binding overwritten, and trailed, without a sign.
  assert(IsVariableReference(variable) && *cell == variable &&
      "merk_open_bind: not a dereferenced unbound variable");
  MerkTrail *trail{merk_trail};
  if (trail && trail->live != 0) {
    if (trail->top == trail->capacity) {
      Grow(*trail);
    }
    trail->entries[trail->top++] = {cell, *cell};
  }
  *cell = value;
}

uint64_t merk_open_trail_mark(void) {
  MerkTrail &trail{Trail()};
  if (trail.live == trail.markCapacity) {
    GrowMarks(trail);
  }
  trail.marks[trail.live] = trail.top;
  return trail.live++;
}

void merk_open_trail_undo(uint64_t mark) {
  MerkTrail &trail{*merk_trail};
  std::uint64_t to{trail.marks[mark]};
  while (trail.top > to) {
    MerkTrailEntry &entry{trail.entries[--trail.top]};
    *entry.address = entry.old;
  }
}

void merk_open_trail_release(uint64_t mark) {
  MerkTrail *trail{merk_trail};
  if (!trail || trail->live <= mark) {
    return;
  }
  trail->live = mark;
  if (mark == 0) {
    trail->top = 0;
  }
}

uint64_t merk_open_trail_marks(void) {
  return merk_trail ? merk_trail->live : 0;
}
}

namespace merkurius::runtime {

OpenEncoding OpenEncodingOf(const TypeCtor &type) {
  switch (type.kind) {
  case Int8:
  case Int16:
  case Int32:
  case Uint8:
  case Uint16:
  case Uint32:
  case Char:
  case Enum:
    return OpenEncoding::Shifted;
  case Int:
  case Uint:
  case Int64:
  case Uint64:
  case Float:
  case String:
    return OpenEncoding::Boxed;
  case Cells:
    return OpenEncoding::Cells;
  case Tuple:
    // A tuple of one argument is that argument, as a notag is.
    return type.functorCount == 1 && type.functors[0].arity == 1
        ? OpenEncoding::AsArgument
        : OpenEncoding::Cells;
  case NoTag:
    return (type.flags & kClosed) != 0 ? OpenEncoding::ClosedBox
                                       : OpenEncoding::AsArgument;
  case Open:
    return OpenEncoding::AsArgument;
  case Dummy:
  case Void:
    return OpenEncoding::Same;
  default:
    return OpenEncoding::Unsupported;
  }
}

OpenTag OpenTagOf(const TypeCtor &type, uint32_t functor) {
  if (type.kind == Tuple) {
    return {0, -1};
  }
  bool constants{false};
  unsigned cells{0}, cell{0};
  for (uint32_t i{0}; i < type.functorCount; ++i) {
    // An existentially typed constructor's hidden fields count: it has a
    // cell even if it has no arguments.
    if (type.functors[i].arity == 0 && type.functors[i].hidden == 0) {
      constants = true;
      continue;
    }
    if (i == functor) {
      cell = cells;
    }
    ++cells;
  }
  return CellTag(constants, cells, cell, kOpenTags);
}

namespace {
struct SmallIntBoxes {
  Word words[kSmallIntBoxMax - kSmallIntBoxMin + 1];
  constexpr SmallIntBoxes() : words{} {
    for (int64_t i{kSmallIntBoxMin}; i <= kSmallIntBoxMax; ++i) {
      words[i - kSmallIntBoxMin] = static_cast<Word>(i);
    }
  }
};
// Aligned to a word, so that a box's pointer has tag 0.
alignas(8) constexpr SmallIntBoxes kSmallIntBoxes{};
} // namespace

const Word *StaticIntBox(int64_t value) {
  if (value < kSmallIntBoxMin || value > kSmallIntBoxMax) {
    return nullptr;
  }
  return &kSmallIntBoxes.words[value - kSmallIntBoxMin];
}

//===----------------------------------------------------------------------===//
// The generic walkers: unify, to_open and from_open by the type_info
//===----------------------------------------------------------------------===//

namespace {

[[noreturn]] void OutOfMemory() {
  std::fputs("merk: out of memory for an open term's walk\n", stderr);
  std::abort();
}

// A hash table of malloc'd memory, from a key of two words to a word, as the
// runtime has no C++ library to take containers from. A key's first word is
// never 0. It has no destructor, so that it can be thread-local or global:
// Free gives its memory back.
struct Table {
  struct Slot {
    Word first, second, value;
  };
  Slot *slots{nullptr};
  std::size_t capacity{0}, count{0};

  static std::size_t Hash(Word first, Word second) {
    return static_cast<std::size_t>(
        (first * 0x9e3779b97f4a7c15ull) ^ (second * 0xc2b2ae3d27d4eb4full));
  }
  Slot *Find(Word first, Word second) const {
    if (capacity == 0) {
      return nullptr;
    }
    for (std::size_t i{Hash(first, second) & (capacity - 1)};;
         i = (i + 1) & (capacity - 1)) {
      Slot &slot{slots[i]};
      if (slot.first == 0) {
        return nullptr;
      }
      if (slot.first == first && slot.second == second) {
        return &slot;
      }
    }
  }
  // The key's slot, and whether it is new, its value then `value`.
  std::pair<Slot *, bool> Insert(Word first, Word second, Word value) {
    if (2 * (count + 1) > capacity) {
      Grow();
    }
    for (std::size_t i{Hash(first, second) & (capacity - 1)};;
         i = (i + 1) & (capacity - 1)) {
      Slot &slot{slots[i]};
      if (slot.first == 0) {
        slot = {first, second, value};
        ++count;
        return {&slot, true};
      }
      if (slot.first == first && slot.second == second) {
        return {&slot, false};
      }
    }
  }
  void Grow() {
    Slot *old{slots};
    std::size_t oldCapacity{capacity};
    capacity = capacity ? 2 * capacity : 64;
    slots = static_cast<Slot *>(std::calloc(capacity, sizeof(Slot)));
    if (!slots) {
      OutOfMemory();
    }
    count = 0;
    for (std::size_t i{0}; i < oldCapacity; ++i) {
      if (old[i].first != 0) {
        Insert(old[i].first, old[i].second, old[i].value);
      }
    }
    std::free(old);
  }
  void Free() {
    std::free(slots);
    slots = nullptr;
    capacity = count = 0;
  }
};

[[noreturn]] void Unsupported(const char *what, const TypeCtor *ctor) {
  std::fflush(stdout);
  std::fprintf(stderr,
      "Merkurius runtime: open terms of %s, such as %s.%s, are not supported "
      "yet\n",
      what, ctor->module ? ctor->module : "", ctor->name ? ctor->name : "");
  ExitProgram(1);
}

[[noreturn]] void BadOpenTerm(const char *what, Word value) {
  std::fflush(stdout);
  std::fprintf(stderr, "Merkurius runtime: bad open term: %s (%#llx)\n", what,
      static_cast<unsigned long long>(value));
  std::abort();
}

constexpr uint16_t kNoFunctor{0xfffd};

// A Cells or Tuple type's open layout: by tag, a functor's number, or
// kConstantTag, kSharedTag or kNoFunctor; and by secondary tag, a functor's
// number.
struct OpenLayout {
  uint16_t byTag[8];
  uint16_t *shared;
  OpenTag *tags;
};

// Made on a type's first use on the thread, as a table the compiler made would
// be (Temporary (docs/INPROGRESS.md, Open types: what is left): the compiler's
// own open layouts can give the descriptors these tables).
const OpenLayout &LayoutOf(const TypeCtor *ctor) {
  thread_local Table *layouts{nullptr};
  if (!layouts) {
    layouts = static_cast<Table *>(std::calloc(1, sizeof(Table)));
    if (!layouts) {
      OutOfMemory();
    }
  }
  Word key{reinterpret_cast<Word>(ctor)};
  if (Table::Slot * slot{layouts->Find(key, 0)}) {
    return *reinterpret_cast<const OpenLayout *>(slot->value);
  }
  uint32_t count{ctor->functorCount};
  auto *layout{static_cast<OpenLayout *>(std::calloc(1, sizeof(OpenLayout)))};
  auto *tags{static_cast<OpenTag *>(std::calloc(count + 1, sizeof(OpenTag)))};
  auto *shared{
      static_cast<uint16_t *>(std::calloc(count + 1, sizeof(uint16_t)))};
  if (!layout || !tags || !shared) {
    OutOfMemory();
  }
  layout->tags = tags;
  layout->shared = shared;
  std::fill(std::begin(layout->byTag), std::end(layout->byTag), kNoFunctor);
  for (uint32_t i{0}; i < count; ++i) {
    const Functor &functor{ctor->functors[i]};
    if (ctor->kind != Tuple && functor.arity == 0 && functor.hidden == 0) {
      layout->byTag[0] = kConstantTag;
      tags[i] = {0, -1};
      continue;
    }
    OpenTag tag{OpenTagOf(*ctor, i)};
    tags[i] = tag;
    if (tag.secondary < 0) {
      layout->byTag[tag.tag] = static_cast<uint16_t>(i);
    } else {
      layout->byTag[tag.tag] = kSharedTag;
      shared[tag.secondary] = static_cast<uint16_t>(i);
    }
  }
  layouts->Insert(key, 0, reinterpret_cast<Word>(layout));
  return *layout;
}

// The functor of an open term of a Cells or Tuple type, which is not a
// variable reference, and its number; `fields` is null for a constant.
const Functor *OpenFunctorOf(
    Word value, const TypeCtor *ctor, const Word *&fields, uint32_t &index) {
  const OpenLayout &layout{LayoutOf(ctor)};
  unsigned tag{static_cast<unsigned>(value & kTagMask)};
  uint16_t entry{layout.byTag[tag]};
  if (entry == kConstantTag) {
    fields = nullptr;
    index = ctor->constants[value >> kTagBits];
    return &ctor->functors[index];
  }
  if (entry == kNoFunctor) {
    BadOpenTerm("a tag no constructor of the type has", value);
  }
  const Word *cell{reinterpret_cast<const Word *>(value - tag)};
  if (entry == kSharedTag) {
    index = layout.shared[cell[0]];
    fields = cell + 1;
  } else {
    index = entry;
    fields = cell;
  }
  const Functor *functor{&ctor->functors[index]};
  if (functor->hidden != 0) {
    Unsupported("existentially typed constructors", ctor);
  }
  return functor;
}

// The encoding of a value of the type, reporting one open terms cannot hold.
OpenEncoding EncodingOf(const TypeCtor *ctor) {
  OpenEncoding encoding{OpenEncodingOf(*ctor)};
  if (encoding == OpenEncoding::Unsupported) {
    Unsupported(ctor->kind == Array ? "arrays"
            : ctor->kind == Closure ? "higher-order types"
                                    : "type_infos and typeclass_infos",
        ctor);
  }
  return encoding;
}

// An AsArgument type's argument's type: a notag's or a tuple of one's field,
// or open(T)'s T.
Type ArgumentType(const Type &type) {
  if (type.ctor->kind == Open) {
    return Resolve(type.environment.arguments[0], type.environment.parent);
  }
  return Resolve(type.ctor->functors[0].fields[0], &type.environment);
}

// closed(T)'s T.
Type ClosedArgument(const Type &type) {
  return Resolve(type.ctor->functors[0].fields[0], &type.environment);
}

unsigned ShiftedBits(const TypeCtor *ctor) {
  switch (ctor->kind) {
  case Int8:
  case Uint8:
    return 8;
  case Int16:
  case Uint16:
    return 16;
  default:
    return 32; // Int32, Uint32 and Char; an enum's number fits
  }
}

// Whether two boxes of a Boxed type's values hold equal values: a float's
// as == has it, a string's by its characters.
bool BoxesEqual(const TypeCtor *ctor, Word a, Word b) {
  Word x{*CellOf(a)}, y{*CellOf(b)};
  switch (ctor->kind) {
  case Float: {
    double p, q;
    std::memcpy(&p, &x, sizeof p);
    std::memcpy(&q, &y, sizeof q);
    return p == q;
  }
  case String:
    return std::strcmp(reinterpret_cast<const char *>(x),
               reinterpret_cast<const char *>(y)) == 0;
  default:
    return x == y;
  }
}

// After this many cells, a walker that may meet a cyclic term, which
// unification without the occurs check can make, starts remembering the
// cells it has been at.
constexpr uint64_t kCellsBeforeRemembering{1 << 14};

bool OpenUnify(const Type &type, Word a, Word b, bool occursCheck,
    uint64_t &cells, Table &seen);
bool Occurs(
    Word variable, const Type &type, Word value, uint64_t &cells, Table &seen);

// Two ground values in the ordinary layout, as closed(T)'s boxes hold them:
// whether they are equal. A value of open(U), which T can be in polymorphic
// code, is unified in U's open layout.
bool GroundEqual(const Type &type, Word a, Word b) {
  for (;;) {
    // Only its own unify entry may compare such a type's values, which a
    // walker cannot call yet.
    if (type.ctor->flags & kUserEquality) {
      Unsupported("types with a user-defined equality", type.ctor);
    }
    if (a == b) {
      return true;
    }
    switch (type.ctor->kind) {
    case Float:
    case String:
      return BoxesEqual(
          type.ctor, reinterpret_cast<Word>(&a), reinterpret_cast<Word>(&b));
    case Dummy:
      return true;
    case Open: {
      uint64_t cells{0};
      Table seen;
      bool equal{OpenUnify(ArgumentType(type), a, b, false, cells, seen)};
      seen.Free();
      return equal;
    }
    case Array:
    case Closure:
      Unsupported(type.ctor->kind == Array ? "arrays in closed(T)"
                                           : "higher-order types in closed(T)",
          type.ctor);
    case NoTag:
    case Tuple:
    case Cells:
      break;
    default:
      return false; // words of the same value are equal
    }
    const Word *fa{nullptr}, *fb{nullptr};
    Word ta{a}, tb{b};
    const Functor *functor{FunctorOf(ta, type.ctor, fa)};
    if (FunctorOf(tb, type.ctor, fb) != functor) {
      return false;
    }
    if (!fa) {
      return false; // two constants with unequal words
    }
    if (functor->hidden != 0) {
      Unsupported("existentially typed constructors", type.ctor);
    }
    if (fa == &ta) {
      // A notag's or a tuple of one's only field, the value itself.
      return GroundEqual(Resolve(functor->fields[0], &type.environment), a, b);
    }
    unsigned arity{functor->arity};
    if (arity == 0) {
      return true;
    }
    for (unsigned i{0}; i + 1 < arity; ++i) {
      if (!GroundEqual(
              Resolve(functor->fields[i], &type.environment), fa[i], fb[i])) {
        return false;
      }
    }
    Word last{functor->fields[arity - 1]};
    if (!IsSameType(last, type) || type.ctor->kind == Tuple) {
      return GroundEqual(
          Resolve(last, &type.environment), fa[arity - 1], fb[arity - 1]);
    }
    a = fa[arity - 1];
    b = fb[arity - 1];
  }
}

// Binds the unbound variable to the value, if the occurs check, when it is
// asked for, finds the variable is not in the value.
bool BindChecked(Word variable, const Type &type, Word value, bool occursCheck,
    uint64_t &cells) {
  if (occursCheck) {
    Table seen;
    bool occurs{Occurs(variable, type, value, cells, seen)};
    seen.Free();
    if (occurs) {
      return false;
    }
  }
  merk_open_bind(variable, value);
  return true;
}

// Unification of two open terms of the type (design document, "Open
// terms", Operations): each dereferenced, an unbound variable bound to the
// other side, and a functor's fields unified in turn. A field of the term's
// own type, as a list's tail, is unified in the loop.
bool OpenUnify(const Type &type, Word a, Word b, bool occursCheck,
    uint64_t &cells, Table &seen) {
  for (;;) {
    a = merk_open_deref(a);
    b = merk_open_deref(b);
    if (a == b) {
      return true;
    }
    OpenEncoding encoding{EncodingOf(type.ctor)};
    if (encoding == OpenEncoding::AsArgument) {
      return OpenUnify(ArgumentType(type), a, b, occursCheck, cells, seen);
    }
    if (IsVariableReference(a)) {
      return BindChecked(a, type, b, occursCheck, cells);
    }
    if (IsVariableReference(b)) {
      return BindChecked(b, type, a, occursCheck, cells);
    }
    // As in GroundEqual.
    if (type.ctor->flags & kUserEquality) {
      Unsupported("types with a user-defined equality", type.ctor);
    }
    switch (encoding) {
    case OpenEncoding::Same:
      return true;
    case OpenEncoding::Shifted:
      return false; // unequal words
    case OpenEncoding::Boxed:
      return BoxesEqual(type.ctor, a, b);
    case OpenEncoding::ClosedBox:
      // The occurs check stops here: closed(T) holds a ground T.
      return GroundEqual(ClosedArgument(type), *CellOf(a), *CellOf(b));
    default:
      break;
    }
    if (((a ^ b) & kTagMask) != 0) {
      return false;
    }
    const Word *fa{nullptr}, *fb{nullptr};
    uint32_t ia{0}, ib{0};
    const Functor *functor{OpenFunctorOf(a, type.ctor, fa, ia)};
    OpenFunctorOf(b, type.ctor, fb, ib);
    if (ia != ib || !fa) {
      return false;
    }
    if (++cells > kCellsBeforeRemembering && !seen.Insert(a, b, 0).second) {
      // A cyclic pair already being unified: the rest of it is, by the
      // time this returns.
      return true;
    }
    unsigned arity{functor->arity};
    for (unsigned i{0}; i + 1 < arity; ++i) {
      if (!OpenUnify(Resolve(functor->fields[i], &type.environment), fa[i],
              fb[i], occursCheck, cells, seen)) {
        return false;
      }
    }
    Word last{functor->fields[arity - 1]};
    if (!IsSameType(last, type) || type.ctor->kind == Tuple) {
      return OpenUnify(Resolve(last, &type.environment), fa[arity - 1],
          fb[arity - 1], occursCheck, cells, seen);
    }
    a = fa[arity - 1];
    b = fb[arity - 1];
  }
}

// Whether the unbound variable is in the open term. The check stops at
// closed(T), which holds a ground T.
bool Occurs(
    Word variable, const Type &type, Word value, uint64_t &cells, Table &seen) {
  for (;;) {
    value = merk_open_deref(value);
    if (value == variable) {
      return true;
    }
    if (IsVariableReference(value)) {
      return false;
    }
    OpenEncoding encoding{EncodingOf(type.ctor)};
    if (encoding == OpenEncoding::AsArgument) {
      return Occurs(variable, ArgumentType(type), value, cells, seen);
    }
    if (encoding != OpenEncoding::Cells) {
      return false;
    }
    const Word *fields{nullptr};
    uint32_t index{0};
    const Functor *functor{OpenFunctorOf(value, type.ctor, fields, index)};
    if (!fields || functor->arity == 0) {
      return false;
    }
    if (++cells > kCellsBeforeRemembering && !seen.Insert(value, 0, 0).second) {
      return false; // a cycle, whose cells are being looked at
    }
    unsigned arity{functor->arity};
    for (unsigned i{0}; i + 1 < arity; ++i) {
      if (Occurs(variable, Resolve(functor->fields[i], &type.environment),
              fields[i], cells, seen)) {
        return true;
      }
    }
    Word last{functor->fields[arity - 1]};
    if (!IsSameType(last, type) || type.ctor->kind == Tuple) {
      return Occurs(variable, Resolve(last, &type.environment),
          fields[arity - 1], cells, seen);
    }
    value = fields[arity - 1];
  }
}

// Whether values of the type have the same word in both layouts: a type
// marked kOpenSame whose arguments do, or a dummy type. The arguments of
// open(T), already open, do too.
bool SameInBothLayouts(const Type &type) {
  const TypeCtor *ctor{type.ctor};
  if (ctor->kind == Dummy || ctor->kind == Void || ctor->kind == Open) {
    return true;
  }
  if ((ctor->flags & kOpenSame) == 0) {
    return false;
  }
  unsigned arity{ctor->kind == Tuple ? ctor->functors[0].arity : ctor->arity};
  for (unsigned i{0}; i < arity; ++i) {
    Environment environment{type.environment};
    if (!SameInBothLayouts(
            Resolve((Word{i} << kTagBits) | kVariableTag, &environment))) {
      return false;
    }
  }
  return true;
}

Word *NewCell(unsigned words) {
  return static_cast<Word *>(Allocate(sizeof(Word) * words));
}

Word Box(Word value) {
  Word *box{NewCell(1)};
  *box = value;
  return reinterpret_cast<Word>(box);
}

// A ground value in the open layout, shared where its type's layouts are the
// same.
Word ToOpen(const Type &type, Word value) {
  Word result{0};
  Word *slot{&result};
  for (;;) {
    if (SameInBothLayouts(type)) {
      *slot = value;
      return result;
    }
    switch (EncodingOf(type.ctor)) {
    case OpenEncoding::Same:
      *slot = value;
      return result;
    case OpenEncoding::Shifted: {
      unsigned bits{ShiftedBits(type.ctor)};
      Word mask{type.ctor->kind == Enum ? ~Word{0} : (Word{1} << bits) - 1};
      *slot = (value & mask) << kTagBits;
      return result;
    }
    case OpenEncoding::Boxed: {
      const TypeCtor *ctor{type.ctor};
      bool small{ctor->kind == Int || ctor->kind == Int64
              ? StaticIntBox(static_cast<int64_t>(value)) != nullptr
              : (ctor->kind == Uint || ctor->kind == Uint64) &&
                  value <= static_cast<Word>(kSmallIntBoxMax)};
      *slot = small
          ? reinterpret_cast<Word>(StaticIntBox(static_cast<int64_t>(value)))
          : Box(value);
      return result;
    }
    case OpenEncoding::ClosedBox:
      *slot = Box(value);
      return result;
    case OpenEncoding::AsArgument:
      *slot = ToOpen(ArgumentType(type), value);
      return result;
    default:
      break;
    }
    // Cells: the cell copied, with its open tags, each field converted.
    const Word *fields{nullptr};
    Word term{value};
    const Functor *functor{FunctorOf(term, type.ctor, fields)};
    if (!fields) {
      *slot = value; // a constant's word is the same in both layouts
      return result;
    }
    if (functor->hidden != 0) {
      Unsupported("existentially typed constructors", type.ctor);
    }
    uint32_t index{static_cast<uint32_t>(functor - type.ctor->functors)};
    OpenTag tag{LayoutOf(type.ctor).tags[index]};
    unsigned before{tag.secondary >= 0 ? 1u : 0u};
    unsigned arity{functor->arity};
    Word *cell{NewCell(before + arity)};
    if (before != 0) {
      cell[0] = static_cast<Word>(tag.secondary);
    }
    *slot = reinterpret_cast<Word>(cell) | tag.tag;
    Word *to{cell + before};
    for (unsigned i{0}; i + 1 < arity; ++i) {
      to[i] = ToOpen(Resolve(functor->fields[i], &type.environment), fields[i]);
    }
    Word last{functor->fields[arity - 1]};
    if (!IsSameType(last, type) || type.ctor->kind == Tuple) {
      to[arity - 1] =
          ToOpen(Resolve(last, &type.environment), fields[arity - 1]);
      return result;
    }
    value = fields[arity - 1];
    slot = &to[arity - 1];
  }
}

// Whether the open term has no variable cell, bound or not, which from_open
// then need not copy if its type's layouts are the same. A term with no
// variable cell is not cyclic, since only a binding can make a cycle.
bool HasNoVariableCell(const Type &type, Word value) {
  for (;;) {
    if (IsVariableReference(value)) {
      return false;
    }
    OpenEncoding encoding{EncodingOf(type.ctor)};
    if (encoding == OpenEncoding::AsArgument) {
      return HasNoVariableCell(ArgumentType(type), value);
    }
    if (encoding != OpenEncoding::Cells) {
      return true;
    }
    const Word *fields{nullptr};
    uint32_t index{0};
    const Functor *functor{OpenFunctorOf(value, type.ctor, fields, index)};
    if (!fields || functor->arity == 0) {
      return true;
    }
    unsigned arity{functor->arity};
    for (unsigned i{0}; i + 1 < arity; ++i) {
      if (!HasNoVariableCell(
              Resolve(functor->fields[i], &type.environment), fields[i])) {
        return false;
      }
    }
    Word last{functor->fields[arity - 1]};
    if (!IsSameType(last, type) || type.ctor->kind == Tuple) {
      return HasNoVariableCell(
          Resolve(last, &type.environment), fields[arity - 1]);
    }
    value = fields[arity - 1];
  }
}

// Kept type_infos (merk_open_type_info, merk_kept_type_info_begin): cells
// that last as long as the program, one for each type, whose arguments are
// kept or static too, so that the collector, which leaves cells outside the
// heap as they are, never needs to follow one. A type is keyed by a chain:
// its descriptor, then a node for each argument, by the node before and the
// argument's kept type_info, then the cell, by the last node and 0. A node
// is the address of its record, tagged 1, so that it is never a descriptor,
// whose tag is 0. Constant-initialized, as merk_heap is.
pthread_mutex_t typeInfoLock = PTHREAD_MUTEX_INITIALIZER;
Table typeInfoCells;

struct KeptNode {
  Word before; // a node, or the descriptor
  Word argument;
  const TypeCtor *ctor;
  Word count;
};

// The node after `key` for an argument's kept type_info.
Word KeptNodeAfter(Word key, Word argument) {
  auto [slot, inserted]{typeInfoCells.Insert(key, argument, 0)};
  if (inserted) {
    auto *node{static_cast<KeptNode *>(std::malloc(sizeof(KeptNode)))};
    if (!node) {
      OutOfMemory();
    }
    bool first{(key & kTagMask) == 0};
    const KeptNode *previous{
        first ? nullptr : reinterpret_cast<const KeptNode *>(key - 1)};
    *node = {key, argument,
        first ? reinterpret_cast<const TypeCtor *>(key) : previous->ctor,
        first ? 1 : previous->count + 1};
    slot->value = reinterpret_cast<Word>(node) + 1;
  }
  return slot->value;
}

// The kept cell of the chain that ends at `key`: a descriptor, kept as it
// is when it has no arguments, or a node.
Word KeptCell(Word key) {
  if ((key & kTagMask) == 0) {
    return key;
  }
  auto [slot, inserted]{typeInfoCells.Insert(key, 0, 0)};
  if (inserted) {
    const auto *last{reinterpret_cast<const KeptNode *>(key - 1)};
    // A tuple's or closure's cell has the number of its arguments in word 1.
    bool counted{last->ctor->kind == Tuple || last->ctor->kind == Closure};
    Word before{counted ? Word{2} : Word{1}};
    auto *cell{static_cast<Word *>(
        std::malloc((before + last->count) * sizeof(Word)))};
    if (!cell) {
      OutOfMemory();
    }
    cell[0] = reinterpret_cast<Word>(last->ctor);
    if (counted) {
      cell[1] = last->count;
    }
    for (const KeptNode *node{last}; node; node = (node->before & kTagMask) == 0
             ? nullptr
             : reinterpret_cast<const KeptNode *>(node->before - 1)) {
      cell[before + node->count - 1] = node->argument;
    }
    slot->value = reinterpret_cast<Word>(cell) + 1;
  }
  return slot->value;
}

// A type_info as a kept one: a descriptor as it is, a cell as the kept cell
// of its type, made of its arguments' kept type_infos, so that one made on
// the heap, which a collection moves, can be in a kept one.
Word KeepTypeInfo(Word info) {
  if ((info & kTagMask) != 1) {
    return info;
  }
  const auto *cell{reinterpret_cast<const Word *>(info - 1)};
  const auto *ctor{reinterpret_cast<const TypeCtor *>(cell[0])};
  bool counted{ctor->kind == Tuple || ctor->kind == Closure};
  Word count{counted ? cell[1] : ctor->arity};
  const Word *arguments{cell + (counted ? 2 : 1)};
  Word key{cell[0]};
  for (Word i{0}; i < count; ++i) {
    key = KeptNodeAfter(key, KeepTypeInfo(arguments[i]));
  }
  return KeptCell(key);
}

// from_open's copy: the copy of each cell of the open term copied so far,
// by the cell's word, or kCopying while its fields are being copied, so
// that reaching it again is a cycle. A cell reached twice, as a term can
// share it, is copied once.
using Copies = Table;
constexpr Word kCopying{~Word{0}};

// The cells of a loop's list and their copies, recorded when it ends, by
// Done, which each return calls.
struct Spine {
  Copies &copies;
  Word *cells{nullptr};
  std::size_t count{0}, capacity{0};
  void Push(Word cell, Word copy) {
    if (count + 2 > capacity) {
      capacity = capacity ? 2 * capacity : 32;
      auto *grown{
          static_cast<Word *>(std::realloc(cells, capacity * sizeof(Word)))};
      if (!grown) {
        OutOfMemory();
      }
      cells = grown;
    }
    cells[count++] = cell;
    cells[count++] = copy;
  }
  bool Done(bool ok) {
    for (std::size_t i{0}; i < count; i += 2) {
      copies.Find(cells[i], 0)->value = cells[i + 1];
    }
    // No destructor, which would need the C++ library's unwinding.
    std::free(cells);
    cells = nullptr;
    count = capacity = 0;
    return ok;
  }
};

// The open term in the ordinary layout, or false if a variable in it is
// unbound or it is cyclic.
bool FromOpen(const Type &type, Word value, Word &out, Copies &copies) {
  Word *slot{&out};
  Spine spine{copies};
  for (;;) {
    value = merk_open_deref(value);
    if (IsVariableReference(value)) {
      return spine.Done(false);
    }
    switch (EncodingOf(type.ctor)) {
    case OpenEncoding::Same:
      *slot = value;
      return spine.Done(true);
    case OpenEncoding::Shifted:
      *slot = value >> kTagBits;
      return spine.Done(true);
    case OpenEncoding::Boxed:
    case OpenEncoding::ClosedBox:
      *slot = *CellOf(value);
      return spine.Done(true);
    case OpenEncoding::AsArgument:
      if (type.ctor->kind == Open) {
        Unsupported("from_open of an open type's values", type.ctor);
      }
      return spine.Done(FromOpen(ArgumentType(type), value, *slot, copies));
    default:
      break;
    }
    const Word *fields{nullptr};
    uint32_t index{0};
    const Functor *functor{OpenFunctorOf(value, type.ctor, fields, index)};
    if (!fields || functor->arity == 0) {
      *slot = value; // a constant's word is the same in both layouts
      return spine.Done(true);
    }
    auto [entry, inserted]{copies.Insert(value, 0, kCopying)};
    if (!inserted) {
      *slot = entry->value;
      return spine.Done(entry->value != kCopying);
    }
    unsigned secondary{functor->secondary >= 0 ? 1u : 0u};
    unsigned arity{functor->arity};
    Word *cell{NewCell(secondary + arity)};
    if (secondary != 0) {
      cell[0] = static_cast<Word>(functor->secondary);
    }
    Word copy{reinterpret_cast<Word>(cell) | functor->tag};
    *slot = copy;
    Word *to{cell + secondary};
    for (unsigned i{0}; i + 1 < arity; ++i) {
      if (!FromOpen(Resolve(functor->fields[i], &type.environment), fields[i],
              to[i], copies)) {
        return spine.Done(false);
      }
    }
    Word last{functor->fields[arity - 1]};
    if (!IsSameType(last, type) || type.ctor->kind == Tuple) {
      bool ok{FromOpen(Resolve(last, &type.environment), fields[arity - 1],
          to[arity - 1], copies)};
      copies.Find(value, 0)->value = copy;
      return spine.Done(ok);
    }
    spine.Push(value, copy);
    value = fields[arity - 1];
    slot = &to[arity - 1];
  }
}

} // namespace

} // namespace merkurius::runtime

using namespace merkurius::runtime;

extern "C" {
int32_t merk_open_unify(uint64_t typeInfo, uint64_t a, uint64_t b) {
  uint64_t cells{0};
  Table seen;
  int32_t ok{OpenUnify(Resolve(typeInfo, nullptr), a, b, false, cells, seen)};
  seen.Free();
  return ok;
}

int32_t merk_open_unify_oc(uint64_t typeInfo, uint64_t a, uint64_t b) {
  uint64_t cells{0};
  Table seen;
  int32_t ok{OpenUnify(Resolve(typeInfo, nullptr), a, b, true, cells, seen)};
  seen.Free();
  return ok;
}

uint64_t merk_open_to_open(uint64_t typeInfo, uint64_t value) {
  return ToOpen(Resolve(typeInfo, nullptr), value);
}

int32_t merk_open_from_open(uint64_t typeInfo, uint64_t value, uint64_t *out) {
  Type type{Resolve(typeInfo, nullptr)};
  if (SameInBothLayouts(type) && HasNoVariableCell(type, value)) {
    *out = value;
    return 1;
  }
  Copies copies;
  int32_t ok{FromOpen(type, value, *out, copies)};
  copies.Free();
  return ok;
}

uint64_t merk_open_type_info(uint64_t descriptor, uint64_t argument) {
  pthread_mutex_lock(&typeInfoLock);
  Word info{KeptCell(KeptNodeAfter(descriptor, KeepTypeInfo(argument)))};
  pthread_mutex_unlock(&typeInfoLock);
  return info;
}

uint64_t merk_kept_type_info_begin(uint64_t descriptor) { return descriptor; }

uint64_t merk_kept_type_info_argument(uint64_t key, uint64_t argument) {
  pthread_mutex_lock(&typeInfoLock);
  Word node{KeptNodeAfter(key, KeepTypeInfo(argument))};
  pthread_mutex_unlock(&typeInfoLock);
  return node;
}

uint64_t merk_kept_type_info_end(uint64_t key) {
  pthread_mutex_lock(&typeInfoLock);
  Word info{KeptCell(key)};
  pthread_mutex_unlock(&typeInfoLock);
  return info;
}

void merk_open_compared(void) {
  std::fflush(stdout);
  std::fprintf(
      stderr, "Merkurius runtime: called compare/3 for an open type\n");
  ExitProgram(1);
}

int32_t merk_open_is_var(uint64_t value) {
  return IsVariableReference(merk_open_deref(value));
}
}
