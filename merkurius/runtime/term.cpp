//===-- runtime/term.cpp - Heap cells of terms ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The cells of terms: headerless, a word for each argument and one before
// them for a secondary tag (design document, "Type representation").
//
// Temporary (chunk 1 step 3, docs/INPROGRESS.md): code generation calls
// merk_alloc for each cell, a malloc that is never freed. Step 3 replaces
// it with mrt.alloc, a bump pointer and limit check inlined in the code,
// whose slow path, merk_alloc_slow, takes another chunk; chunk 2 makes the
// chunks per-process heaps, and chunk 3 collects them.
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <cstdio>
#include <cstdlib>

// Temporary, see above. A cell of `bytes`, a multiple of the word, aligned to 8 bytes so that its
// address leaves room for a 3-bit primary tag.
extern "C" void *merk_alloc(std::int64_t bytes) {
  void *cell{std::malloc(static_cast<std::size_t>(bytes))};
  if (!cell) {
    std::fflush(stdout);
    std::fprintf(stderr, "Mercury runtime: out of memory\n");
    std::exit(1);
  }
  return cell;
}
