//===-- runtime/heap.h - The heap -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The heap, which terms' cells and arrays are allocated on (design document,
// "Memory management"). Code generation's mrt.alloc bumps merk_heap.next and
// calls merk_alloc_slow when that would pass merk_heap.limit; the runtime's
// own allocations, arrays', do the same through Allocate.
//
// Temporary (chunk 1 step 3, docs/INPROGRESS.md): each thread has a heap,
// in thread-local storage, until chunk 2 gives each process one and decides
// where its pointer and limit live (D2); the heap's chunks are never freed
// until chunk 3's collector.
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_RUNTIME_HEAP_H_
#define MERKURIUS_RUNTIME_HEAP_H_

#include <cstddef>
#include <cstdint>

extern "C" {
// The next free byte of the heap's current chunk, and its end. Both are
// null before a thread's first allocation, which then takes the slow path.
struct MerkHeap {
  char *next;
  char *limit;
};
extern thread_local MerkHeap merk_heap;

// An allocation of `bytes`, a multiple of the word, that does not fit what
// is left of the current chunk.
void *merk_alloc_slow(std::int64_t bytes);
}

namespace merkurius::runtime {
// `bytes` on the heap, a multiple of the word, as mrt.alloc allocates them.
inline void *Allocate(std::size_t bytes) {
  char *next{merk_heap.next};
  if (static_cast<std::size_t>(merk_heap.limit - next) >= bytes) {
    merk_heap.next = next + bytes;
    return next;
  }
  return merk_alloc_slow(static_cast<std::int64_t>(bytes));
}
} // namespace merkurius::runtime

#endif // MERKURIUS_RUNTIME_HEAP_H_
