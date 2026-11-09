//===-- runtime/fragment.h - A message's cells ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Where a message's cells are made, outside the heap: by the local copy of a
// send (mailbox.cpp) and by the decoder (encode.cpp). The receiver reads them
// there, so the fragment joins its heap.
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_RUNTIME_FRAGMENT_H_
#define MERKURIUS_RUNTIME_FRAGMENT_H_

#include <cstddef>
#include <cstdio>
#include <cstdlib>

namespace merkurius::runtime {

inline void *Malloc(std::size_t bytes) {
  void *block{std::malloc(bytes)};
  if (!block) {
    std::fflush(stdout);
    std::fputs("Mercury runtime: out of memory\n", stderr);
    std::exit(1);
  }
  return block;
}

// The start of a chunk of cells from malloc, of a fragment or a process's heap.
struct ChunkLink {
  ChunkLink *next;
  // Its cells' bytes, which follow this.
  std::size_t bytes;
  // A message's chunks, in a list that stays as it is when `next` changes.
  ChunkLink *sibling;
  // Whether a message still in its mailbox is in it: the collector leaves it.
  bool pinned;
  // One large object, which a collection marks where it is instead of copying.
  bool large{false};
};

inline void FreeChunks(ChunkLink *chunks) {
  while (chunks) {
    ChunkLink *next{chunks->next};
    std::free(chunks);
    chunks = next;
  }
}

// Cells bump-allocated in chunks from malloc, which align them to a word at
// least, as the primary tags need.
//
// A fragment's chunks are linked, and whoever takes them frees them: a process
// when it ends, once it has read the message, or a mailbox that closes with
// the message unread. A process's collector frees them, once the message has
// left its mailbox and what is still used of it has been copied.
class Fragment {
public:
  // Its chunks, which it no longer owns.
  ChunkLink *Take() {
    ChunkLink *chunks{chunks_};
    chunks_ = nullptr;
    return chunks;
  }

  void *Allocate(std::size_t bytes) {
    bytes = (bytes + 7) & ~std::size_t{7};
    if (static_cast<std::size_t>(limit_ - next_) < bytes) {
      std::size_t size{bytes > chunkBytes_ / 4 ? bytes : chunkBytes_};
      chunkBytes_ = chunkBytes_ < kMostChunkBytes ? 2 * chunkBytes_ : chunkBytes_;
      auto *chunk{static_cast<ChunkLink *>(Malloc(sizeof(ChunkLink) + size))};
      chunk->next = chunks_;
      chunk->bytes = size;
      chunk->sibling = nullptr;
      chunk->pinned = false;
      chunk->large = false;
      chunks_ = chunk;
      next_ = reinterpret_cast<char *>(chunk + 1);
      limit_ = next_ + size;
    }
    void *cell{next_};
    next_ += bytes;
    return cell;
  }

private:
  // Most messages are small, and a chunk that is used only in part still
  // costs a page fault for each page: 4096 bytes for every message took 85% of
  // a ping-pong's time. Chunks start small and double.
  static constexpr std::size_t kFirstChunkBytes{256};
  static constexpr std::size_t kMostChunkBytes{4096};
  std::size_t chunkBytes_{kFirstChunkBytes};
  char *next_{nullptr};
  char *limit_{nullptr};
  ChunkLink *chunks_{nullptr};
};

} // namespace merkurius::runtime

#endif // MERKURIUS_RUNTIME_FRAGMENT_H_
