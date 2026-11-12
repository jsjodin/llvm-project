//===-- runtime/heap.cpp - The heap ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The heap's slow path (heap.h): a new chunk when the current one is full,
// and a block of its own for an allocation too large to share one, which a
// collection does not move.
//
// MERK_HEAP_CHUNK_BYTES, if set, is the chunk's size instead, for testing:
// with a small one, such as 64, nearly every allocation takes the slow
// path, and those of more than half a chunk have blocks of their own.
//
// A process's heap is collected from here, when the allocation comes from
// Mercury code at a safepoint (roots.h). The rest of a full chunk is wasted.
//
//===----------------------------------------------------------------------===//

#include "fragment.h"
#include "heap.h"
#include "process.h"
#include "roots.h"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Constant-initialized, so that code generation's accesses need no call of
// an initializer.
extern "C" {
thread_local MerkHeap merk_heap{nullptr, nullptr};
}

namespace {
// A chunk: to be tuned with chunk 3's nursery, which these will become.
constexpr std::size_t kChunkBytes{1 << 20};
// An allocation larger than this, or than half a chunk, has a block of its
// own, so that a large array does not waste the rest of a chunk.
constexpr std::size_t kLargeBytes{kChunkBytes / 8};

std::size_t ReadChunkBytes() {
  std::size_t chunk{kChunkBytes};
  if (const char *bytes{std::getenv("MERK_HEAP_CHUNK_BYTES")};
      bytes && *bytes) {
    char *end{nullptr};
    unsigned long long value{std::strtoull(bytes, &end, 10)};
    if (*end != '\0' || value < 16) {
      std::fflush(stdout);
      std::fprintf(stderr,
          "Mercury runtime: MERK_HEAP_CHUNK_BYTES=%s is not a number of at "
          "least 16\n",
          bytes);
      merkurius::runtime::ExitProgram(1);
    }
    // A multiple of the word, so that cells stay aligned.
    chunk = static_cast<std::size_t>(value) & ~std::size_t{7};
  }
  return chunk;
}

// The chunk's size, 0 until the first slow path of any thread reads it.
// An atomic rather than a local static, whose guard needs the C++ library,
// which programs are not linked with; threads that race read the same.
std::atomic<std::size_t> chunkBytes{0};

char *Chunk(std::size_t bytes, bool large = false) {
  // malloc aligns to at least a word, which the primary tags need. A thread
  // that runs no process keeps its chunks.
  auto *chunk{static_cast<merkurius::runtime::ChunkLink *>(
      merkurius::runtime::Malloc(sizeof(merkurius::runtime::ChunkLink) + bytes))};
  chunk->next = nullptr;
  chunk->bytes = bytes;
  chunk->sibling = nullptr;
  chunk->pinned = false;
  chunk->large = large;
  if (merkurius::runtime::Process *process{
          merkurius::runtime::CurrentProcess()}) {
    chunk->next = process->chunks;
    process->chunks = chunk;
    process->allocatedBytes += bytes;
  }
  return reinterpret_cast<char *>(chunk + 1);
}
} // namespace

// A safepoint before the runtime allocates an array, as the runtime's own
// allocations only grow the heap. The walk starts at this frame, which must
// stay while it runs: a tail call to CheckRoots would pop it, and CheckRoots'
// own frame would overwrite the saved frame pointer and return address.
extern "C" void merk_gc_poll() {
  if (merkurius::runtime::CollectIfDue(__builtin_frame_address(0)) &&
      merkurius::runtime::CheckRootsEnabled()) {
    merkurius::runtime::CheckRoots(__builtin_frame_address(0));
    asm volatile("" ::: "memory"); // no tail call
  }
}

extern "C" void *merk_alloc_slow(std::int64_t bytes) {
  if (merkurius::runtime::CheckRootsEnabled()) {
    merkurius::runtime::CheckRoots(__builtin_frame_address(0));
  }
  std::size_t size{static_cast<std::size_t>(bytes)};
  if (merkurius::runtime::CollectIfDue(__builtin_frame_address(0))) {
    if (merkurius::runtime::CheckRootsEnabled()) {
      merkurius::runtime::CheckRoots(__builtin_frame_address(0));
    }
    if (static_cast<std::size_t>(merk_heap.limit - merk_heap.next) >= size) {
      char *cell{merk_heap.next};
      merk_heap.next = cell + size;
      return cell;
    }
  }
  std::size_t chunkSize{chunkBytes.load(std::memory_order_relaxed)};
  if (chunkSize == 0) {
    chunkSize = ReadChunkBytes();
    chunkBytes.store(chunkSize, std::memory_order_relaxed);
  }
  std::size_t large{chunkSize / 2 < kLargeBytes ? chunkSize / 2 : kLargeBytes};
  if (size > large) {
    return Chunk(size, true);
  }
  if (merkurius::runtime::CollectsEveryAllocation()) {
    // So that the next allocation comes here too.
    return Chunk(size);
  }
  // A process's heap starts small and doubles, so that a process that
  // allocates little costs little.
  merkurius::runtime::Process *process{merkurius::runtime::CurrentProcess()};
  if (process && process->chunkBytes != 0 && process->chunkBytes < chunkSize) {
    if (size <= process->chunkBytes) {
      chunkSize = process->chunkBytes;
    }
    process->chunkBytes *= 2;
  }
  char *chunk{Chunk(chunkSize)};
  merk_heap.next = chunk + size;
  merk_heap.limit = chunk + chunkSize;
  return chunk;
}
