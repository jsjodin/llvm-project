//===-- runtime/compute-units.cpp - A GPU's units for processes -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "compute-units.h"

namespace merkurius::runtime {

int UnitCount(const UnitMask &mask) {
  int count{0};
  for (std::uint32_t word : mask.words) {
    count += __builtin_popcount(word);
  }
  return count;
}

bool ProcessUnits(const std::uint32_t *usable, int words, UnitMask &mask) {
  mask = {};
  for (int word{0}; word < words; ++word) {
    if (word >= kUnitMaskWords) {
      if (usable[word] != 0) {
        return false;
      }
      continue;
    }
    // The low unit of each pair whose units are both usable, and its pair.
    std::uint32_t both{usable[word] & (usable[word] >> 1) & 0x55555555u};
    mask.words[word] = both | (both << 1);
  }
  return UnitCount(mask) != 0;
}

void FirstPair(UnitMask &mask) {
  bool found{false};
  for (std::uint32_t &word : mask.words) {
    if (found) {
      word = 0;
    } else if (word != 0) {
      word &= std::uint32_t{3} << (__builtin_ctz(word) & ~1);
      found = true;
    }
  }
}

} // namespace merkurius::runtime
