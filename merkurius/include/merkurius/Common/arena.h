//===-- include/merkurius/Common/arena.h ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_COMMON_ARENA_H_
#define MERKURIUS_COMMON_ARENA_H_

// An arena for objects that are freed all at once and never destroyed one
// by one, so they must be trivially destructible: terms, which live until
// their item is converted, and parse tree nodes, which live as long as the
// parse tree.

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Allocator.h"
#include <algorithm>
#include <new>
#include <type_traits>
#include <utility>

namespace merkurius::common {

class Arena {
public:
  template <typename T, typename... A> const T *Make(A &&...args) {
    static_assert(std::is_trivially_destructible_v<T>);
    void *p{allocator_.Allocate<T>()};
    if constexpr (std::is_aggregate_v<T>) {
      return new (p) T{std::forward<A>(args)...};
    } else {
      return new (p) T(std::forward<A>(args)...);
    }
  }
  llvm::StringRef Save(llvm::StringRef s) {
    if (s.empty()) {
      return {};
    }
    char *p{allocator_.Allocate<char>(s.size())};
    std::copy(s.begin(), s.end(), p);
    return {p, s.size()};
  }
  template <typename T> llvm::ArrayRef<T> SaveArray(llvm::ArrayRef<T> array) {
    static_assert(std::is_trivially_copyable_v<T>);
    if (array.empty()) {
      return {};
    }
    T *p{allocator_.Allocate<T>(array.size())};
    std::copy(array.begin(), array.end(), p);
    return {p, array.size()};
  }
  // Frees everything made since the last reset.
  void Reset() { allocator_.Reset(); }

private:
  llvm::BumpPtrAllocator allocator_;
};

} // namespace merkurius::common
#endif // MERKURIUS_COMMON_ARENA_H_
