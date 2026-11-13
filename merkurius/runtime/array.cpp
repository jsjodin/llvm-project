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
#include "record.h"
#include <stdint.h>

#ifdef __AMDGPU__
// The device has no heap of ours: its arrays are in its arena, as its
// cells are (device/device.cpp).
extern "C" void *merk_device_alloc(uint64_t bytes);
extern "C" void *merk_device_alloc_aligned(uint64_t bytes, uint64_t before);
#else
#include "heap.h"
#include <cinttypes>
#include <cstdio>
#include <cstdlib>

extern "C" [[noreturn]] void merk_uncaught_exception(const char *exception);
#endif

using merkurius::runtime::ArrayCodeOf;
using merkurius::runtime::ArrayElementBytes;
using merkurius::runtime::ArrayElements;
using merkurius::runtime::ArrayHeader;
using merkurius::runtime::ArrayHeaderWords;
using merkurius::runtime::ArraySize;
using merkurius::runtime::ArrayWidthCode;
using merkurius::runtime::ArrayWords;
using merkurius::runtime::BoxRecord;
using merkurius::runtime::ElementBytes;
using merkurius::runtime::IsRecord;
using merkurius::runtime::IsRecordArray;
using merkurius::runtime::kArrayBytes1;
using merkurius::runtime::kArrayBytes2;
using merkurius::runtime::kArrayBytes4;
using merkurius::runtime::kArrayRecords;
using merkurius::runtime::kArrayWords;
using merkurius::runtime::LoadElement;
using merkurius::runtime::RecordLayout;
using merkurius::runtime::Resolve;
using merkurius::runtime::StoreElement;
using merkurius::runtime::Type;
using merkurius::runtime::UnboxRecord;
using merkurius::runtime::Word;

namespace {

void *AllocateBytes(Word bytes) {
#ifdef __AMDGPU__
  return merk_device_alloc(bytes);
#else
  return merkurius::runtime::Allocate(bytes);
#endif
}

// The bytes of an array, of `headers` header words. On the device a large
// array's elements start on 256 bytes (device/device.cpp,
// merk_device_alloc_aligned); a small one's would waste too much of its
// cell.
void *AllocateArrayBytes(Word bytes, Word headers) {
#ifdef __AMDGPU__
  constexpr Word kAlignFrom{4096};
  if (bytes >= kAlignFrom) {
    return merk_device_alloc_aligned(bytes, sizeof(Word) * headers);
  }
#else
  (void)headers;
#endif
  return AllocateBytes(bytes);
}

// An array of `size` elements of the width's code, of `recordBytes` each
// for records, whose last word, which the elements may not fill, is 0, so
// that an array's bytes depend only on its elements.
Word *Allocate(int64_t size, Word code, Word recordBytes = 0) {
  Word headers{code == kArrayRecords ? Word{2} : Word{1}};
  Word bytes{static_cast<Word>(size) *
      (code == kArrayRecords ? recordBytes : ElementBytes(code))};
  Word words{headers + (bytes + sizeof(Word) - 1) / sizeof(Word)};
  auto *array{
      static_cast<Word *>(AllocateArrayBytes(sizeof(Word) * words, headers))};
  array[0] = ArrayHeader(static_cast<Word>(size), code);
  if (code == kArrayRecords) {
    array[1] = recordBytes;
  }
  if (words > headers) {
    array[words - 1] = 0;
  }
  return array;
}

// Sets elements `lo` to `hi`, inclusive, to the word's bits, as many as the
// width has. The width is read once: StoreElement for each element read the
// header each time, a load from memory for every byte, which on a GPU
// process's one lane made array.init of 3.6 KB of cells take 370 us, a third
// of a step of Life on two places (design document, Measurements, "Life on
// several places"). Each width's loop is a plain one of stores.
void Fill(Word *array, int64_t lo, int64_t hi, Word element) {
  if (lo > hi) {
    return;
  }
  auto count{static_cast<Word>(hi - lo + 1)};
  auto first{static_cast<Word>(lo)};
  switch (ArrayWidthCode(array)) {
  case kArrayBytes1:
    __builtin_memset(reinterpret_cast<unsigned char *>(array + 1) + first,
        static_cast<unsigned char>(element), count);
    return;
  case kArrayBytes2: {
    auto *elements{reinterpret_cast<uint16_t *>(array + 1) + first};
    auto value{static_cast<uint16_t>(element)};
    for (Word i{0}; i < count; ++i) {
      elements[i] = value;
    }
    return;
  }
  case kArrayBytes4: {
    auto *elements{reinterpret_cast<uint32_t *>(array + 1) + first};
    auto value{static_cast<uint32_t>(element)};
    for (Word i{0}; i < count; ++i) {
      elements[i] = value;
    }
    return;
  }
  default: {
    Word *elements{array + 1 + first};
    for (Word i{0}; i < count; ++i) {
      elements[i] = element;
    }
  }
  }
}

// Sets elements `lo` to `hi`, inclusive, of an array of records, to the
// element's bytes, which may be element `lo - 1`'s.
void FillBytes(Word *array, int64_t lo, int64_t hi, const void *element) {
  Word bytes{array[1]};
  unsigned char *elements{ArrayElements(array)};
  for (int64_t i = lo; i <= hi; ++i) {
    __builtin_memcpy(elements + bytes * static_cast<Word>(i), element, bytes);
  }
}

// Sets elements `lo` to `hi`, inclusive, of an array of records, to a cell
// of the type, read into element `lo` and copied from it.
void FillUnboxed(
    Word *array, int64_t lo, int64_t hi, Word element, const Type &type) {
  if (lo > hi) {
    return;
  }
  unsigned char *first{ArrayElements(array) + array[1] * static_cast<Word>(lo)};
  UnboxRecord(type, element, first);
  FillBytes(array, lo + 1, hi, first);
}

// Swaps elements `i` and `j` of an array of any width.
void Swap(Word *array, Word i, Word j) {
  Word bytes{ArrayElementBytes(array)};
  unsigned char *x{ArrayElements(array) + bytes * i};
  unsigned char *y{ArrayElements(array) + bytes * j};
  for (Word k{0}; k < bytes; ++k) {
    unsigned char byte{x[k]};
    x[k] = y[k];
    y[k] = byte;
  }
}

void Reverse(Word *array) {
  for (Word i = 0, j = ArraySize(array); i + 1 < j; ++i) {
    --j;
    Swap(array, i, j);
  }
}

// How many elements a list has. A cell of [|] is the word `list` when its
// primary tag is `tag`, and holds the head and then the tail; the list ends
// at [], whose tag is another.
int64_t ListLength(Word list, Word tag) {
  constexpr Word kTagMask{7};
  int64_t size{0};
  for (Word cell = list; (cell & kTagMask) == tag;
      cell = reinterpret_cast<const Word *>(cell - tag)[1]) {
    ++size;
  }
  return size;
}

} // namespace

// An array of `size` elements, each `init`, of the width's code (array.h).
// The size is not negative: the library's check comes first.
extern "C" Word *merk_array_new(int64_t size, Word init, Word code) {
  Word *array{Allocate(size, code)};
  Fill(array, 0, size - 1, init);
  return array;
}

// An array of `size` records of `bytes` each, each a copy of the element's
// bytes, for code that knows the record's type.
extern "C" Word *merk_array_new_bytes(
    int64_t size, const void *element, Word bytes) {
  Word *array{Allocate(size, kArrayRecords, bytes)};
  FillBytes(array, 0, size - 1, element);
  return array;
}

extern "C" Word *merk_array_copy(const Word *array) {
  Word words{ArrayWords(array)};
  auto *copy{static_cast<Word *>(
      AllocateArrayBytes(sizeof(Word) * words, ArrayHeaderWords(array)))};
  __builtin_memcpy(copy, array, sizeof(Word) * words);
  return copy;
}

// A new array of `size` elements: as many of `array`'s as fit, then `init`.
// Not an array of records, whose code knows its type.
extern "C" Word *merk_array_resize(
    const Word *array, int64_t size, Word init) {
  auto old{static_cast<int64_t>(ArraySize(array))};
  Word code{ArrayWidthCode(array)};
  int64_t kept{old < size ? old : size};
  Word *resized{Allocate(size, code)};
  __builtin_memcpy(ArrayElements(resized), ArrayElements(array),
      static_cast<uint64_t>(kept) * ElementBytes(code));
  Fill(resized, kept, size - 1, init);
  return resized;
}

// The same for records of `bytes` each, with the element's bytes, which
// may be null if none are added. An empty array made where the type was not
// known may be of another code.
extern "C" Word *merk_array_resize_bytes(
    const Word *array, int64_t size, const void *element, Word bytes) {
  auto old{static_cast<int64_t>(ArraySize(array))};
  int64_t kept{old < size ? old : size};
  Word *resized{Allocate(size, kArrayRecords, bytes)};
  __builtin_memcpy(ArrayElements(resized), ArrayElements(array),
      static_cast<uint64_t>(kept) * bytes);
  FillBytes(resized, kept, size - 1, element);
  return resized;
}

// A new array of a list's elements, in order, of the width's code.
extern "C" Word *merk_array_from_list(Word list, Word tag, Word code) {
  int64_t size{ListLength(list, tag)};
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
  Reverse(array);
  return array;
}

// Sets the elements from `lo` to `hi`, inclusive, to `element`, in place.
extern "C" Word *merk_array_fill(
    Word *array, int64_t lo, int64_t hi, Word element) {
  Fill(array, lo, hi, element);
  return array;
}

// The same for records, with the element's bytes.
extern "C" Word *merk_array_fill_bytes(
    Word *array, int64_t lo, int64_t hi, const void *element) {
  FillBytes(array, lo, hi, element);
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

// For polymorphic code, which has the elements' type_info: element `i` of
// an array of records, boxed into its cell, and the store of a cell into
// one. Boxing allocates as the other operations do, only growing the heap;
// code generation polls for a collection first.
extern "C" Word merk_array_load_record(
    const Word *array, int64_t i, Word typeInfo) {
  return BoxRecord(Resolve(typeInfo, nullptr),
      ArrayElements(array) + array[1] * static_cast<Word>(i), AllocateBytes);
}

extern "C" void merk_array_store_record(
    Word *array, int64_t i, Word element, Word typeInfo) {
  UnboxRecord(Resolve(typeInfo, nullptr), element,
      ArrayElements(array) + array[1] * static_cast<Word>(i));
}

// The words of an array, its header words' and its elements', for a copy
// of it (merk.array_words).
extern "C" Word merk_array_words(const Word *array) {
  return ArrayWords(array);
}

// The making operations for polymorphic code, by the elements' type_info,
// which gives their width or their records' bytes.
extern "C" Word *merk_array_new_typed(int64_t size, Word init, Word typeInfo) {
  Type type{Resolve(typeInfo, nullptr)};
  if (!IsRecord(type)) {
    return merk_array_new(size, init, ArrayCodeOf(type));
  }
  Word *array{Allocate(size, kArrayRecords, RecordLayout(type).bytes)};
  FillUnboxed(array, 0, size - 1, init, type);
  return array;
}

extern "C" Word *merk_array_fill_typed(
    Word *array, int64_t lo, int64_t hi, Word element, Word typeInfo) {
  if (!IsRecordArray(array)) {
    Fill(array, lo, hi, element);
    return array;
  }
  FillUnboxed(array, lo, hi, element, Resolve(typeInfo, nullptr));
  return array;
}

extern "C" Word *merk_array_resize_typed(
    const Word *array, int64_t size, Word init, Word typeInfo) {
  Type type{Resolve(typeInfo, nullptr)};
  if (!IsRecord(type)) {
    return merk_array_resize(array, size, init);
  }
  Word bytes{RecordLayout(type).bytes};
  auto old{static_cast<int64_t>(ArraySize(array))};
  int64_t kept{old < size ? old : size};
  Word *resized{Allocate(size, kArrayRecords, bytes)};
  __builtin_memcpy(ArrayElements(resized), ArrayElements(array),
      static_cast<uint64_t>(kept) * bytes);
  FillUnboxed(resized, kept, size - 1, init, type);
  return resized;
}

extern "C" Word *merk_array_from_list_typed(
    Word list, Word tag, Word typeInfo) {
  Type type{Resolve(typeInfo, nullptr)};
  if (!IsRecord(type)) {
    return merk_array_from_list(list, tag, ArrayCodeOf(type));
  }
  int64_t size{ListLength(list, tag)};
  Word bytes{RecordLayout(type).bytes};
  Word *array{Allocate(size, kArrayRecords, bytes)};
  Word cell{list};
  for (int64_t i = 0; i < size; ++i) {
    const Word *fields{reinterpret_cast<const Word *>(cell - tag)};
    UnboxRecord(type, fields[0], ArrayElements(array) + bytes * i);
    cell = fields[1];
  }
  return array;
}

extern "C" Word *merk_array_from_reverse_list_typed(
    Word list, Word tag, Word typeInfo) {
  Word *array{merk_array_from_list_typed(list, tag, typeInfo)};
  Reverse(array);
  return array;
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
