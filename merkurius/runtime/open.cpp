//===-- runtime/open.cpp - The open layout of terms -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "open.h"
#include "heap.h"
#include "trail.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

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

void merk_open_bind(uint64_t variable, uint64_t value) {
  Word *cell{CellOf(variable)};
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
  ++trail.live;
  return trail.top;
}

void merk_open_trail_undo(uint64_t mark) {
  MerkTrail &trail{*merk_trail};
  while (trail.top > mark) {
    MerkTrailEntry &entry{trail.entries[--trail.top]};
    *entry.address = entry.old;
  }
}

void merk_open_trail_release(uint64_t mark) {
  MerkTrail &trail{*merk_trail};
  if (--trail.live == 0) {
    trail.top = 0;
  }
  (void)mark;
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
  case Tuple:
    return OpenEncoding::Cells;
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
    if (type.functors[i].arity == 0) {
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

} // namespace merkurius::runtime
