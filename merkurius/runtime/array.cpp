//===-- runtime/array.cpp - Arrays ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Arrays: a pointer to the array's size, a word, followed by its elements,
// a word each, whatever their type (design document, "Type
// representation"). Code generation reads and writes elements itself; these
// are the operations that allocate, and those over many elements. Arrays
// are never freed until there is a garbage collector.
//
//===----------------------------------------------------------------------===//

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" [[noreturn]] void merk_uncaught_exception(const char *exception);

namespace {
using Word = std::uint64_t;

Word *Allocate(std::int64_t size) {
  void *array{std::malloc(sizeof(Word) * (1 + size))};
  if (!array) {
    std::fflush(stdout);
    std::fprintf(stderr, "Mercury runtime: out of memory\n");
    std::exit(1);
  }
  Word *words{static_cast<Word *>(array)};
  words[0] = static_cast<Word>(size);
  return words;
}
} // namespace

// An array of `size` elements, each `init`. The size is not negative: the
// library's check comes first.
extern "C" Word *merk_array_new(std::int64_t size, Word init) {
  Word *array{Allocate(size)};
  for (std::int64_t i = 0; i < size; ++i) {
    array[1 + i] = init;
  }
  return array;
}

extern "C" Word *merk_array_copy(const Word *array) {
  std::int64_t size{static_cast<std::int64_t>(array[0])};
  Word *copy{Allocate(size)};
  std::memcpy(copy + 1, array + 1, sizeof(Word) * size);
  return copy;
}

// A new array of `size` elements: as many of `array`'s as fit, then `init`.
extern "C" Word *merk_array_resize(
    const Word *array, std::int64_t size, Word init) {
  std::int64_t old{static_cast<std::int64_t>(array[0])};
  std::int64_t kept{old < size ? old : size};
  Word *resized{Allocate(size)};
  std::memcpy(resized + 1, array + 1, sizeof(Word) * kept);
  for (std::int64_t i = kept; i < size; ++i) {
    resized[1 + i] = init;
  }
  return resized;
}

// Sets the elements from `lo` to `hi`, inclusive, to `element`, in place.
extern "C" Word *merk_array_fill(
    Word *array, std::int64_t lo, std::int64_t hi, Word element) {
  for (std::int64_t i = lo; i <= hi; ++i) {
    array[1 + i] = element;
  }
  return array;
}

// The library's index_out_of_bounds exception, as mmc writes it.
extern "C" [[noreturn]] void merk_index_out_of_bounds(
    const char *what, std::int64_t index, std::int64_t max) {
  int length{std::snprintf(nullptr, 0,
      "index_out_of_bounds(\"%s: index %" PRId64 " not in range [0, %" PRId64
      "]\")",
      what, index, max)};
  char *exception{static_cast<char *>(std::malloc(length + 1))};
  std::snprintf(exception, length + 1,
      "index_out_of_bounds(\"%s: index %" PRId64 " not in range [0, %" PRId64
      "]\")",
      what, index, max);
  merk_uncaught_exception(exception);
}
