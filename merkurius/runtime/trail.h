//===-- runtime/trail.h - The trail of open terms' bindings -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The value trail of a process (design document, Runtime system, "Support
// for open terms"; docs/INPROGRESS.md, Open types, step 3): the (address,
// old word) of each binding of a variable cell made while a choice point or
// try goal that can bind open terms is live, so that backtracking into it, or
// catching an exception there, can undo them. open.cpp's routines use it;
// the scheduler only swaps and frees it, so that a program without open
// terms links none of them.
//
// merk_trail is the trail of the process the thread runs, as merk_heap is its
// heap: null until the process's first open choice point, allocated then
// (merk_open_trail_mark), and freed when the process ends.
//
// A parallel loop's team thread runs no process, so an iteration's choice
// point on it allocates a trail that nothing frees. The thread keeps it, as
// it keeps its heap chunks (heap.cpp, Chunk). That is one trail for each
// thread of the pool, reused by later regions and empty between them, since
// releasing the last mark empties it. No collection runs in a parallel
// region, so the collector never sees it live.
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_RUNTIME_TRAIL_H_
#define MERKURIUS_RUNTIME_TRAIL_H_

#include <cstddef>
#include <cstdint>
#include <cstdlib>

extern "C" {
struct MerkTrailEntry {
  std::uint64_t *address;
  std::uint64_t old;
};

struct MerkTrail {
  MerkTrailEntry *entries;
  // The entries in use, and room for.
  std::uint64_t top;
  std::uint64_t capacity;
  // The live choice points and try goals that can bind open terms, oldest
  // first: the trail's top when each began. A binding is trailed only while
  // there is one, and when the last goes the trail is emptied.
  std::uint64_t *marks;
  std::uint64_t live;
  std::uint64_t markCapacity;
};

extern thread_local MerkTrail *merk_trail;
}

// Code generation's inline bind reads the live marks there
// (lib/CodeGen/Lowering.cpp, kTrailLiveOffset).
static_assert(offsetof(MerkTrail, live) == 32);

namespace merkurius::runtime {
inline void FreeTrail(MerkTrail *trail) {
  if (trail) {
    std::free(trail->entries);
    std::free(trail->marks);
    std::free(trail);
  }
}
} // namespace merkurius::runtime

#endif // MERKURIUS_RUNTIME_TRAIL_H_
