//===-- runtime/heap.cpp - The heap ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The heap's slow path (heap.h): a new chunk when the current one is full,
// and a block of its own for an allocation too large to share one.
//
// MERK_HEAP_CHUNK_BYTES, if set, is the chunk's size instead, for testing:
// with a small one, such as 64, nearly every allocation takes the slow
// path, and those of more than half a chunk have blocks of their own.
//
// Temporary (chunk 1 step 3, docs/INPROGRESS.md): chunks come from malloc
// and are never freed, and the rest of a full chunk is wasted, until chunk
// 3's collector empties the nursery instead.
//
//===----------------------------------------------------------------------===//

#include "heap.h"
#include "process.h"
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
      std::exit(1);
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

char *Chunk(std::size_t bytes) {
  // malloc aligns to at least a word, which the primary tags need.
  void *chunk{std::malloc(bytes)};
  if (!chunk) {
    std::fflush(stdout);
    std::fprintf(stderr, "Mercury runtime: out of memory\n");
    std::exit(1);
  }
  return static_cast<char *>(chunk);
}
} // namespace

extern "C" void *merk_alloc_slow(std::int64_t bytes) {
  std::size_t size{static_cast<std::size_t>(bytes)};
  std::size_t chunkSize{chunkBytes.load(std::memory_order_relaxed)};
  if (chunkSize == 0) {
    chunkSize = ReadChunkBytes();
    chunkBytes.store(chunkSize, std::memory_order_relaxed);
  }
  std::size_t large{chunkSize / 2 < kLargeBytes ? chunkSize / 2 : kLargeBytes};
  if (size > large) {
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
