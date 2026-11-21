//===-- runtime/open.cpp - The open layout of terms -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "open.h"

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
