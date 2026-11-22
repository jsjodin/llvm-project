//===-- runtime/word-map.h - A hash map of words ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A hash map from a word that is never 0 to a word, of malloc'd memory, for
// the host's walks over open terms: a message's copy, which maps each
// variable cell and each cell it has copied to its copy (copy.h, CopyOpen),
// and io.write, which numbers variables and keeps the cells on its path
// (write.cpp). The runtime has no C++ library to take containers from. It
// has no destructor, so that it can be a member of what has none: Free
// gives its memory back.
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_RUNTIME_WORD_MAP_H_
#define MERKURIUS_RUNTIME_WORD_MAP_H_

#include "descriptor.h"
#include <cstdio>
#include <cstdlib>

namespace merkurius::runtime {

class WordMap {
public:
  // The value of `key`, or null.
  Word *Find(Word key) const {
    if (capacity_ == 0) {
      return nullptr;
    }
    for (std::size_t i{Home(key)};; i = (i + 1) & (capacity_ - 1)) {
      if (slots_[i].key == 0) {
        return nullptr;
      }
      if (slots_[i].key == key) {
        return &slots_[i].value;
      }
    }
  }

  // The value of `key`, `value` if it is new, and whether it is.
  Word *Insert(Word key, Word value, bool &inserted) {
    if (2 * (count_ + 1) > capacity_) {
      Grow();
    }
    for (std::size_t i{Home(key)};; i = (i + 1) & (capacity_ - 1)) {
      if (slots_[i].key == 0) {
        slots_[i] = {key, value};
        ++count_;
        inserted = true;
        return &slots_[i].value;
      }
      if (slots_[i].key == key) {
        inserted = false;
        return &slots_[i].value;
      }
    }
  }

  // Removes `key`, if it is there, moving back the keys after it that
  // would not be found past its empty slot.
  void Erase(Word key) {
    if (capacity_ == 0) {
      return;
    }
    std::size_t mask{capacity_ - 1};
    std::size_t i{Home(key)};
    for (;; i = (i + 1) & mask) {
      if (slots_[i].key == 0) {
        return;
      }
      if (slots_[i].key == key) {
        break;
      }
    }
    for (std::size_t j{(i + 1) & mask};; j = (j + 1) & mask) {
      if (slots_[j].key == 0) {
        break;
      }
      std::size_t home{Home(slots_[j].key)};
      // Whether j's home is cyclically outside (i, j]: then j may move to i.
      bool moves{i <= j ? (home <= i || home > j) : (home <= i && home > j)};
      if (moves) {
        slots_[i] = slots_[j];
        i = j;
      }
    }
    slots_[i] = {0, 0};
    --count_;
  }

  void Free() {
    std::free(slots_);
    slots_ = nullptr;
    capacity_ = count_ = 0;
  }

private:
  struct Slot {
    Word key, value;
  };

  std::size_t Home(Word key) const {
    Word mixed{key * 0x9e3779b97f4a7c15ull};
    return static_cast<std::size_t>(mixed ^ (mixed >> 32)) & (capacity_ - 1);
  }

  void Grow() {
    Slot *old{slots_};
    std::size_t oldCapacity{capacity_};
    capacity_ = capacity_ ? 2 * capacity_ : 64;
    slots_ = static_cast<Slot *>(std::calloc(capacity_, sizeof(Slot)));
    if (!slots_) {
      std::fputs(
          "Merkurius runtime: out of memory for an open term's walk\n", stderr);
      std::abort();
    }
    count_ = 0;
    bool inserted;
    for (std::size_t i{0}; i < oldCapacity; ++i) {
      if (old[i].key != 0) {
        Insert(old[i].key, old[i].value, inserted);
      }
    }
    std::free(old);
  }

  Slot *slots_{nullptr};
  std::size_t capacity_{0}, count_{0};
};

} // namespace merkurius::runtime

#endif // MERKURIUS_RUNTIME_WORD_MAP_H_
