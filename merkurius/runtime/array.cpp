//===-- runtime/array.cpp - Arrays ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Arrays: a pointer to the array's header, its size and its elements'
// width, followed by its elements (array.h). Code generation reads and
// writes elements itself, but for narrow ones in polymorphic code; these are
// the operations that allocate, on the heap, as mrt.alloc allocates
// (heap.h), and those over many elements. The device's runtime is compiled
// from this file too, for the device code generation calls them from.
//
//===----------------------------------------------------------------------===//

#include "array.h"
#include "walk.h"
#include <stdint.h>

#ifdef __AMDGPU__
// The device has no heap of ours: its arrays are in its arena, as its
// cells are (device/device.cpp).
extern "C" void *merk_device_alloc(uint64_t bytes);
#else
#include "heap.h"
#include <cinttypes>
#include <cstdio>
#include <cstdlib>

extern "C" [[noreturn]] void merk_uncaught_exception(const char *exception);
#endif

using merkurius::runtime::ArrayHeader;
using merkurius::runtime::ArraySize;
using merkurius::runtime::ArrayWidthCode;
using merkurius::runtime::ArrayWidthOfKind;
using merkurius::runtime::ElementBytes;
using merkurius::runtime::ElementWords;
using merkurius::runtime::kArrayWords;
using merkurius::runtime::LoadElement;
using merkurius::runtime::StoreElement;
using merkurius::runtime::Word;

namespace {

// An array of `size` elements of the width's code, whose last word, which
// narrow elements may not fill, is 0, so that an array's bytes depend only
// on its elements.
Word *Allocate(int64_t size, Word code) {
  Word words{ElementWords(static_cast<Word>(size), code)};
  uint64_t bytes{sizeof(Word) * (1 + words)};
#ifdef __AMDGPU__
  Word *array{static_cast<Word *>(merk_device_alloc(bytes))};
#else
  Word *array{static_cast<Word *>(merkurius::runtime::Allocate(bytes))};
#endif
  array[0] = ArrayHeader(static_cast<Word>(size), code);
  if (words > 0) {
    array[words] = 0;
  }
  return array;
}

// Sets elements `lo` to `hi`, inclusive, to the word's bits.
void Fill(Word *array, int64_t lo, int64_t hi, Word element) {
  if (ArrayWidthCode(array) == kArrayWords) {
    for (int64_t i = lo; i <= hi; ++i) {
      array[1 + i] = element;
    }
    return;
  }
  for (int64_t i = lo; i <= hi; ++i) {
    StoreElement(array, static_cast<Word>(i), element);
  }
}

} // namespace

// An array of `size` elements, each `init`, of the width's code (array.h).
// The size is not negative: the library's check comes first.
extern "C" Word *merk_array_new(int64_t size, Word init, Word code) {
  Word *array{Allocate(size, code)};
  Fill(array, 0, size - 1, init);
  return array;
}

extern "C" Word *merk_array_copy(const Word *array) {
  Word size{ArraySize(array)};
  Word code{ArrayWidthCode(array)};
  Word *copy{Allocate(static_cast<int64_t>(size), code)};
  __builtin_memcpy(
      copy + 1, array + 1, sizeof(Word) * ElementWords(size, code));
  return copy;
}

// A new array of `size` elements: as many of `array`'s as fit, then `init`.
extern "C" Word *merk_array_resize(
    const Word *array, int64_t size, Word init) {
  auto old{static_cast<int64_t>(ArraySize(array))};
  Word code{ArrayWidthCode(array)};
  int64_t kept{old < size ? old : size};
  Word *resized{Allocate(size, code)};
  __builtin_memcpy(
      resized + 1, array + 1, static_cast<uint64_t>(kept) * ElementBytes(code));
  Fill(resized, kept, size - 1, init);
  return resized;
}

// A new array of a list's elements, in order, of the width's code. A cell of
// [|] is the word `list` when its primary tag is `tag`, and holds the head
// and then the tail; the list ends at [], whose tag is another.
extern "C" Word *merk_array_from_list(Word list, Word tag, Word code) {
  constexpr Word kTagMask{7};
  int64_t size{0};
  for (Word cell = list; (cell & kTagMask) == tag;
      cell = reinterpret_cast<const Word *>(cell - tag)[1]) {
    ++size;
  }
  Word *array{Allocate(size, code)};
  Word cell{list};
  for (int64_t i = 0; i < size; ++i) {
    const Word *fields{reinterpret_cast<const Word *>(cell - tag)};
    StoreElement(array, static_cast<Word>(i), fields[0]);
    cell = fields[1];
  }
  return array;
}

// A new array of a list's elements, the last first, as
// array.from_reverse_list has them.
extern "C" Word *merk_array_from_reverse_list(Word list, Word tag, Word code) {
  Word *array{merk_array_from_list(list, tag, code)};
  for (int64_t i = 0, j = static_cast<int64_t>(ArraySize(array)) - 1; i < j;
      ++i, --j) {
    Word element{LoadElement(array, static_cast<Word>(i))};
    StoreElement(
        array, static_cast<Word>(i), LoadElement(array, static_cast<Word>(j)));
    StoreElement(array, static_cast<Word>(j), element);
  }
  return array;
}

// Sets the elements from `lo` to `hi`, inclusive, to `element`, in place.
extern "C" Word *merk_array_fill(
    Word *array, int64_t lo, int64_t hi, Word element) {
  Fill(array, lo, hi, element);
  return array;
}

// Element `i` of an array of narrow elements, as a word, zero-extended, and
// the store of one, for polymorphic code, which reads a word array's
// elements itself (Lowering.cpp, LowerArray).
extern "C" Word merk_array_load(const Word *array, int64_t i) {
  return LoadElement(array, static_cast<Word>(i));
}

extern "C" void merk_array_store(Word *array, int64_t i, Word element) {
  StoreElement(array, static_cast<Word>(i), element);
}

// The width's code of an array whose elements are of the type_info's type,
// for polymorphic code that makes one.
extern "C" Word merk_array_width(Word typeInfo) {
  return ArrayWidthOfKind(
      merkurius::runtime::Resolve(typeInfo, nullptr).ctor->kind);
}

#ifndef __AMDGPU__
// The library's index_out_of_bounds exception, as mmc writes it.
extern "C" [[noreturn]] void merk_index_out_of_bounds(
    const char *what, int64_t index, int64_t max) {
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
#endif
