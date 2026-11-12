//===-- runtime/array.h - Arrays' layout ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// An array is a pointer to its header word, followed by its elements
// (design document, "Type representation"). The header's low 56 bits are
// the number of elements, and its top byte the elements' width: 0 for a word
// each, the layout of every array of another type, so that such an array's
// header is its size; or the code of 1, 2 or 4 bytes each, for an array of
// int8, uint8, int16, uint16, int32, uint32, char or an enumeration, as C
// holds them; or 4 for flat records, inline as C holds them, whose bytes
// each, C's sizeof, are the array's word 1, so that their elements start at
// word 2 (record.h; docs/INPROGRESS.md, Flat records, step 2). A narrow element
// is its bits; as a word, in polymorphic code, it is zero-extended, as every
// narrow integer is (Lowering.cpp, ToBits), so the width says all there is to
// an element and its signedness is the type's. The elements take whole words,
// the last padded, as the heap and the arenas allocate words.
// lib/CodeGen/Lowering.cpp lowers the same layout; the device's runtime is
// compiled from this header too.
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_RUNTIME_ARRAY_H_
#define MERKURIUS_RUNTIME_ARRAY_H_

#include "descriptor.h"

namespace merkurius::runtime {

constexpr unsigned kArrayWidthShift{56};
constexpr Word kArraySizeMask{(Word{1} << kArrayWidthShift) - 1};
// The widths' codes: a word, 1, 2 and 4 bytes.
constexpr Word kArrayWords{0};
constexpr Word kArrayBytes1{1};
constexpr Word kArrayBytes2{2};
constexpr Word kArrayBytes4{3};
constexpr Word kArrayRecords{4};

inline Word ArraySize(const Word *array) { return array[0] & kArraySizeMask; }
inline Word ArrayWidthCode(const Word *array) {
  return array[0] >> kArrayWidthShift;
}
// The bytes of an element of the width's code, which is not records'.
inline Word ElementBytes(Word code) {
  return code == kArrayWords ? sizeof(Word) : Word{1} << (code - 1);
}
// The words that `size` elements of the code take, the header not counted.
inline Word ElementWords(Word size, Word code) {
  return (size * ElementBytes(code) + sizeof(Word) - 1) / sizeof(Word);
}
inline bool IsRecordArray(const Word *array) {
  return ArrayWidthCode(array) == kArrayRecords;
}
// The bytes of an element of any array.
inline Word ArrayElementBytes(const Word *array) {
  return IsRecordArray(array) ? array[1] : ElementBytes(ArrayWidthCode(array));
}
// The words before the elements: the header, and a record's bytes.
inline Word ArrayHeaderWords(const Word *array) {
  return IsRecordArray(array) ? 2 : 1;
}
// Where the elements are.
inline unsigned char *ArrayElements(Word *array) {
  return reinterpret_cast<unsigned char *>(array + ArrayHeaderWords(array));
}
inline const unsigned char *ArrayElements(const Word *array) {
  return reinterpret_cast<const unsigned char *>(
      array + ArrayHeaderWords(array));
}
// The words of the whole array, its header words and its elements, which
// the elements' last word pads.
inline Word ArrayWords(const Word *array) {
  Word bytes{ArraySize(array) * ArrayElementBytes(array)};
  return ArrayHeaderWords(array) + (bytes + sizeof(Word) - 1) / sizeof(Word);
}
inline Word ArrayHeader(Word size, Word code) {
  return size | code << kArrayWidthShift;
}
inline bool IsWordArray(const Word *array) {
  return ArrayWidthCode(array) == kArrayWords;
}

// The width's code of an array whose elements are of the descriptor's kind.
inline Word ArrayWidthOfKind(uint8_t kind) {
  switch (kind) {
  case Int8:
  case Uint8:
    return kArrayBytes1;
  case Int16:
  case Uint16:
    return kArrayBytes2;
  case Int32:
  case Uint32:
  case Char:
  case Enum:
    return kArrayBytes4;
  default:
    return kArrayWords;
  }
}

// Element `i` of an array of any width but records', as a word,
// zero-extended.
inline Word LoadElement(const Word *array, Word i) {
  const auto *bytes{reinterpret_cast<const unsigned char *>(array + 1)};
  switch (ArrayWidthCode(array)) {
  case kArrayBytes1:
    return bytes[i];
  case kArrayBytes2: {
    uint16_t value;
    __builtin_memcpy(&value, bytes + 2 * i, 2);
    return value;
  }
  case kArrayBytes4: {
    uint32_t value;
    __builtin_memcpy(&value, bytes + 4 * i, 4);
    return value;
  }
  default:
    return array[1 + i];
  }
}

// Sets element `i` to a word's low bits, as many as the width has.
inline void StoreElement(Word *array, Word i, Word element) {
  auto *bytes{reinterpret_cast<unsigned char *>(array + 1)};
  switch (ArrayWidthCode(array)) {
  case kArrayBytes1:
    bytes[i] = static_cast<unsigned char>(element);
    return;
  case kArrayBytes2: {
    auto value{static_cast<uint16_t>(element)};
    __builtin_memcpy(bytes + 2 * i, &value, 2);
    return;
  }
  case kArrayBytes4: {
    auto value{static_cast<uint32_t>(element)};
    __builtin_memcpy(bytes + 4 * i, &value, 4);
    return;
  }
  default:
    array[1 + i] = element;
  }
}

} // namespace merkurius::runtime

#endif // MERKURIUS_RUNTIME_ARRAY_H_
