//===-- runtime/record.h - Flat records' bytes ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A flat record inline, as an array of records holds it (array.h): its
// fields as C holds them, each at its C size and alignment, a nested record
// inline (docs/INPROGRESS.md, Flat records, step 2). Which types are records
// is the compiler's choice (lib/CodeGen/Records.h), which a term
// type's descriptor carries in kFlat; a tuple's descriptor, which every
// tuple of its arity shares, is one by its arguments, by the same rule. The
// layout is C's, which is LLVM's for a literal struct on every target code
// generation has, so no table of offsets is needed. Boxing makes the record's
// cell, as code generation makes it; unboxing reads one into the bytes. The
// device's runtime is compiled from this header too.
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_RUNTIME_RECORD_H_
#define MERKURIUS_RUNTIME_RECORD_H_

#include "array.h"
#include "walk.h"

namespace merkurius::runtime {

bool IsRecord(const Type &type);

// Whether a value of the type can be a record's field.
inline bool IsFlatField(const Type &type) {
  switch (type.ctor->kind) {
  case Int:
  case Uint:
  case Int8:
  case Int16:
  case Int32:
  case Int64:
  case Uint8:
  case Uint16:
  case Uint32:
  case Uint64:
  case Float:
  case Char:
    return true;
  case Enum:
  case NoTag:
  case Cells:
    return (type.ctor->flags & kFlat) != 0;
  case Tuple:
    return IsRecord(type);
  default:
    return false;
  }
}

// Whether the type's values are records: a term type's the compiler chose,
// or a tuple of two or more flat fields.
inline bool IsRecord(const Type &type) {
  if (type.ctor->kind == Cells) {
    return (type.ctor->flags & kFlat) != 0;
  }
  if (type.ctor->kind != Tuple) {
    return false;
  }
  const Functor &functor{type.ctor->functors[0]};
  if (functor.arity < 2) {
    return false;
  }
  for (uint32_t i{0}; i < functor.arity; ++i) {
    if (!IsFlatField(Resolve(functor.fields[i], &type.environment))) {
      return false;
    }
  }
  return true;
}

// The width's code of an array of the type's elements, which code
// generation gives an array of it too.
inline Word ArrayCodeOf(const Type &type) {
  return IsRecord(type) ? kArrayRecords : ArrayWidthOfKind(type.ctor->kind);
}

struct Layout {
  Word bytes;
  Word alignment;
};

Layout RecordLayout(const Type &type);

// A flat field's size and alignment, as C holds it.
inline Layout FieldLayout(const Type &type) {
  switch (type.ctor->kind) {
  case Int8:
  case Uint8:
    return {1, 1};
  case Int16:
  case Uint16:
    return {2, 2};
  case Int32:
  case Uint32:
  case Char:
  case Enum:
    return {4, 4};
  case NoTag:
    return FieldLayout(
        Resolve(type.ctor->functors[0].fields[0], &type.environment));
  case Cells:
  case Tuple:
    return RecordLayout(type);
  default:
    return {8, 8};
  }
}

inline Word AlignUp(Word offset, Word alignment) {
  return (offset + alignment - 1) / alignment * alignment;
}

// A record's: its fields in order, each at the next offset its alignment
// allows, the whole a multiple of its largest alignment.
inline Layout RecordLayout(const Type &type) {
  const Functor &functor{type.ctor->functors[0]};
  Word offset{0}, alignment{1};
  for (uint32_t i{0}; i < functor.arity; ++i) {
    Layout field{FieldLayout(Resolve(functor.fields[i], &type.environment))};
    offset = AlignUp(offset, field.alignment) + field.bytes;
    alignment = field.alignment > alignment ? field.alignment : alignment;
  }
  return {AlignUp(offset, alignment), alignment};
}

template <typename Allocate>
Word BoxRecord(
    const Type &type, const unsigned char *bytes, Allocate &allocate);

// A field's word, as a cell holds it, from its bytes: a narrow one
// zero-extended, and a nested record boxed.
template <typename Allocate>
Word FieldWord(
    const Type &type, const unsigned char *bytes, Allocate &allocate) {
  switch (type.ctor->kind) {
  case Int8:
  case Uint8:
    return bytes[0];
  case Int16:
  case Uint16: {
    uint16_t value;
    __builtin_memcpy(&value, bytes, 2);
    return value;
  }
  case Int32:
  case Uint32:
  case Char:
  case Enum: {
    uint32_t value;
    __builtin_memcpy(&value, bytes, 4);
    return value;
  }
  case NoTag:
    return FieldWord(
        Resolve(type.ctor->functors[0].fields[0], &type.environment), bytes,
        allocate);
  case Cells:
  case Tuple:
    return BoxRecord(type, bytes, allocate);
  default: {
    Word value;
    __builtin_memcpy(&value, bytes, 8);
    return value;
  }
  }
}

// A record's cell, made by `allocate(bytes)`, from its bytes, and its nested
// records' cells, as code generation makes them: the one functor's tag in
// the pointer, and each field a word.
template <typename Allocate>
Word BoxRecord(
    const Type &type, const unsigned char *bytes, Allocate &allocate) {
  const Functor &functor{type.ctor->functors[0]};
  auto *cell{static_cast<Word *>(allocate(sizeof(Word) * functor.arity))};
  Word offset{0};
  for (uint32_t i{0}; i < functor.arity; ++i) {
    Type field{Resolve(functor.fields[i], &type.environment)};
    Layout layout{FieldLayout(field)};
    offset = AlignUp(offset, layout.alignment);
    cell[i] = FieldWord(field, bytes + offset, allocate);
    offset += layout.bytes;
  }
  return reinterpret_cast<Word>(cell) + functor.tag;
}

void UnboxRecord(const Type &type, Word value, unsigned char *bytes);

// A field's bytes from its word.
inline void StoreField(const Type &type, Word value, unsigned char *bytes) {
  switch (type.ctor->kind) {
  case Int8:
  case Uint8:
    bytes[0] = static_cast<unsigned char>(value);
    return;
  case Int16:
  case Uint16: {
    auto narrow{static_cast<uint16_t>(value)};
    __builtin_memcpy(bytes, &narrow, 2);
    return;
  }
  case Int32:
  case Uint32:
  case Char:
  case Enum: {
    auto narrow{static_cast<uint32_t>(value)};
    __builtin_memcpy(bytes, &narrow, 4);
    return;
  }
  case NoTag:
    StoreField(Resolve(type.ctor->functors[0].fields[0], &type.environment),
        value, bytes);
    return;
  case Cells:
  case Tuple:
    UnboxRecord(type, value, bytes);
    return;
  default:
    __builtin_memcpy(bytes, &value, 8);
    return;
  }
}

// A record's bytes from its cell, with its padding 0, so that equal records
// have equal bytes.
inline void UnboxRecord(const Type &type, Word value, unsigned char *bytes) {
  Layout whole{RecordLayout(type)};
  __builtin_memset(bytes, 0, whole.bytes);
  const Word *fields{nullptr};
  const Functor *functor{FunctorOf(value, type.ctor, fields)};
  Word offset{0};
  for (uint32_t i{0}; i < functor->arity; ++i) {
    Type field{Resolve(functor->fields[i], &type.environment)};
    Layout layout{FieldLayout(field)};
    offset = AlignUp(offset, layout.alignment);
    StoreField(field, fields[i], bytes + offset);
    offset += layout.bytes;
  }
}

inline bool AllZero(const unsigned char *bytes, Word from, Word to) {
  for (Word i{from}; i < to; ++i) {
    if (bytes[i] != 0) {
      return false;
    }
  }
  return true;
}

// Why a record's bytes are not a value of its type, as UnboxRecord makes
// them, or null if they are: each character must be a code point, each
// enumeration's number one of its constants', and the padding zeros, as the
// decoder checks what another node sent.
inline const char *RecordValueError(
    const Type &type, const unsigned char *bytes) {
  constexpr const char *padding{"a record's padding is not zeros"};
  const Functor &functor{type.ctor->functors[0]};
  Word offset{0};
  for (uint32_t i{0}; i < functor.arity; ++i) {
    Type field{Resolve(functor.fields[i], &type.environment)};
    while (field.ctor->kind == NoTag) {
      field = Resolve(field.ctor->functors[0].fields[0], &field.environment);
    }
    Layout layout{FieldLayout(field)};
    Word end{offset};
    offset = AlignUp(offset, layout.alignment);
    if (!AllZero(bytes, end, offset)) {
      return padding;
    }
    const unsigned char *at{bytes + offset};
    uint32_t narrow;
    switch (field.ctor->kind) {
    case Char:
      __builtin_memcpy(&narrow, at, 4);
      if (narrow > 0x10ffff) {
        return "a character is not a code point";
      }
      break;
    case Enum:
      __builtin_memcpy(&narrow, at, 4);
      if (narrow >= field.ctor->functorCount) {
        return "a functor's number is not one of its type's";
      }
      break;
    case Cells:
    case Tuple:
      if (const char *error{RecordValueError(field, at)}) {
        return error;
      }
      break;
    default:
      break;
    }
    offset += layout.bytes;
  }
  return AllZero(bytes, offset, RecordLayout(type).bytes) ? nullptr : padding;
}

} // namespace merkurius::runtime

#endif // MERKURIUS_RUNTIME_RECORD_H_
