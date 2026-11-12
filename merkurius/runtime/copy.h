//===-- runtime/copy.h - Copying a value by its type_info -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A deep copy of a value by its type_info, as a send copies a message
// (mailbox.cpp, design document, "Message passing"). `Memory` says where
// the copy's cells go: `void *Allocate(Word bytes)` gives memory aligned to a
// word at least, `[[noreturn]] void Fail(const char *, const char *)`
// reports a value that cannot be copied, `void CheckStack()` checks the
// stack before each value, as the copy recurses on the value's structure, and
// `kOtherImage` says whether the
// copy is for code of another image, the host's for a GPU's or a GPU's for
// the host's. Such code has other procedures and descriptors, so a closure
// and an existentially typed constructor's type_infos and typeclass_infos
// cannot be copied for it. The host copies into a fragment,
// and into host memory that a device reads (offload.cpp); a GPU copies
// into its arena and into such host memory (device/device.cpp), so that a
// message of any type that the host can copy crosses between them
// (docs/INPROGRESS.md, GPU processes without omp target, step 4e). It uses
// no library, so that it builds for the GPU.
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_RUNTIME_COPY_H_
#define MERKURIUS_RUNTIME_COPY_H_

#include "array.h"
#include "walk.h"

namespace merkurius::runtime {

template <typename Memory>
Word CopyTyped(Word value, const Type &type, Memory &memory);

// Whether a value of the type is its word in a copy.
inline bool CopiedAsItIs(const TypeCtor *ctor) {
  switch (ctor->kind) {
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
  case Enum:
  case Dummy:
    return true;
  default:
    return false;
  }
}

// A type_info: a descriptor is static, and a cell of one and its arguments'
// type_infos is copied.
template <typename Memory> Word CopyTypeInfo(Word info, Memory &memory) {
  if ((info & kTagMask) == 0) {
    return info;
  }
  const Word *cell{reinterpret_cast<const Word *>(info - 1)};
  const auto *ctor{reinterpret_cast<const TypeCtor *>(cell[0])};
  bool counted{ctor->kind == Tuple || ctor->kind == Closure};
  Word arity{counted ? cell[1] : ctor->arity};
  Word before{counted ? Word{2} : Word{1}};
  auto *copy{
      static_cast<Word *>(memory.Allocate(sizeof(Word) * (before + arity)))};
  for (Word i{0}; i < before; ++i) {
    copy[i] = cell[i];
  }
  for (Word i{0}; i < arity; ++i) {
    copy[before + i] = CopyTypeInfo(cell[before + i], memory);
  }
  return reinterpret_cast<Word>(copy) + 1;
}

// A typeclass_info, by its method table's shape. The table is static.
template <typename Memory>
Word CopyDictionary(Word dictionary, Memory &memory) {
  const Word *cell{reinterpret_cast<const Word *>(dictionary)};
  DictionaryShape shape{ShapeOfDictionary(cell)};
  Word total{shape.superclasses + shape.typeInfos + shape.constraints};
  auto *copy{static_cast<Word *>(memory.Allocate(sizeof(Word) * (1 + total)))};
  copy[0] = cell[0];
  for (Word i{1}; i <= total; ++i) {
    bool isTypeInfo{
        i > shape.superclasses && i <= shape.superclasses + shape.typeInfos};
    copy[i] = isTypeInfo ? CopyTypeInfo(cell[i], memory)
                         : CopyDictionary(cell[i], memory);
  }
  return reinterpret_cast<Word>(copy);
}

template <typename Memory>
Word CopyValue(
    Word value, Word type, const Environment *environment, Memory &memory) {
  Type resolved{Resolve(type, environment)};
  // A field that is its word takes no call, so that a flat record's copy
  // does not recurse, nor take a frame for each field.
  if (CopiedAsItIs(resolved.ctor)) {
    return value;
  }
  return CopyTyped(value, resolved, memory);
}

// A term of constructors: its cell copied, field by field, and its tag kept.
// A field of the term's own type is copied in the loop rather than by a
// call, so that the length of a list uses no stack.
template <typename Memory>
Word CopyTerm(Word value, const Type &type, Memory &memory) {
  Word result{0};
  Word *slot{&result};
  for (;;) {
    const Word *fields{nullptr};
    Word term{value};
    const Functor *functor{FunctorOf(term, type.ctor, fields)};
    if (!fields || fields == &term) {
      // A constant, or a notag's or a tuple of one's only field, which is
      // the value itself.
      if (!fields) {
        *slot = value;
        return result;
      }
      *slot = CopyValue(value, functor->fields[0], &type.environment, memory);
      return result;
    }
    unsigned tag{type.ctor->kind == Tuple
            ? functor->tag
            : static_cast<unsigned>(value & kTagMask)};
    // A constructor that shares its tag has its secondary tag in word 0.
    const Word *from{reinterpret_cast<const Word *>(value - tag)};
    unsigned before{static_cast<unsigned>(fields - from)};
    unsigned arity{functor->arity};
    Word *cell{
        static_cast<Word *>(memory.Allocate(sizeof(Word) * (before + arity)))};
    // Its secondary tag, and then the type_infos and typeclass_infos of an
    // existentially typed constructor, which the fields' types read.
    unsigned hidden{functor->hidden};
    if (Memory::kOtherImage && hidden != 0) {
      memory.Fail("cannot copy an existentially typed value between the host "
                  "and a GPU: ",
          functor->name);
    }
    for (unsigned i{0}; i < before - hidden; ++i) {
      cell[i] = from[i];
    }
    const Word *hiddenFrom{fields - hidden};
    Word *hiddenTo{cell + before - hidden};
    for (unsigned i{0}; i < hidden; ++i) {
      hiddenTo[i] = i < functor->hiddenTypeInfos
          ? CopyTypeInfo(hiddenFrom[i], memory)
          : CopyDictionary(hiddenFrom[i], memory);
    }
    Environment existential{
        type.environment.arguments, type.environment.parent, hiddenFrom};
    const Environment *environment{
        hidden != 0 ? &existential : &type.environment};
    *slot = reinterpret_cast<Word>(cell) + tag;
    Word *copied{cell + before};
    if (arity == 0) {
      return result;
    }
    for (unsigned i{0}; i + 1 < arity; ++i) {
      copied[i] = CopyValue(fields[i], functor->fields[i], environment, memory);
    }
    Word last{functor->fields[arity - 1]};
    if (!IsSameType(last, type) || type.ctor->kind == Tuple) {
      copied[arity - 1] =
          CopyValue(fields[arity - 1], last, environment, memory);
      return result;
    }
    value = fields[arity - 1];
    slot = &copied[arity - 1];
  }
}

// A closure's cell, by its layout: its code is static, and each capture is
// copied by what the layout says. A capture's type is found in the original
// cell, whose captured type_infos are the ones it names.
template <typename Memory> Word CopyClosure(Word value, Memory &memory) {
  const Word *cell{reinterpret_cast<const Word *>(value)};
  const auto *layout{reinterpret_cast<const ClosureLayout *>(cell[1])};
  if (!layout) {
    memory.Fail("cannot copy a closure made on a device", "");
  }
  Word *copy{
      static_cast<Word *>(memory.Allocate(sizeof(Word) * (2 + layout->count)))};
  copy[0] = cell[0];
  copy[1] = cell[1];
  const Word *captured{cell + 2};
  Environment environment{nullptr, nullptr, captured};
  for (Word i{0}; i < layout->count; ++i) {
    Word capture{layout->captures[i]};
    if (capture == kCaptureTypeInfo) {
      copy[2 + i] = CopyTypeInfo(captured[i], memory);
    } else if (capture == kCaptureTypeclassInfo) {
      copy[2 + i] = CopyDictionary(captured[i], memory);
    } else {
      copy[2 + i] = CopyValue(captured[i], capture, &environment, memory);
    }
  }
  return reinterpret_cast<Word>(copy);
}

template <typename Memory>
Word CopyTyped(Word value, const Type &type, Memory &memory) {
  // Inlined here, as the device's check measures this frame's callees.
  memory.CheckStack();
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
  case Enum:
  case Dummy:
    return value;
  case String: {
    const char *text{reinterpret_cast<const char *>(value)};
    Word bytes{__builtin_strlen(text) + 1};
    char *copy{static_cast<char *>(memory.Allocate(bytes))};
    __builtin_memcpy(copy, text, bytes);
    return reinterpret_cast<Word>(copy);
  }
  case Array: {
    const Word *array{reinterpret_cast<const Word *>(value)};
    Word size{ArraySize(array)};
    Word words{ArrayWords(array)};
    Word *copy{static_cast<Word *>(memory.Allocate(sizeof(Word) * words))};
    copy[0] = array[0];
    Word element{(Word{0} << kTagBits) | kVariableTag};
    // What is copied as it is, as CopyTyped returns it, is copied as a
    // block, as narrow elements and records always are, with a records
    // array's word 1 (array.h).
    if (!IsWordArray(array) ||
        (size != 0 && CopiedAsItIs(Resolve(element, &type.environment).ctor))) {
      __builtin_memcpy(copy + 1, array + 1, sizeof(Word) * (words - 1));
      return reinterpret_cast<Word>(copy);
    }
    for (Word i{0}; i < size; ++i) {
      copy[1 + i] = CopyValue(array[1 + i], element, &type.environment, memory);
    }
    return reinterpret_cast<Word>(copy);
  }
  case NoTag:
  case Tuple:
  case Cells:
    return CopyTerm(value, type, memory);
  case Closure:
    if (Memory::kOtherImage) {
      memory.Fail("cannot copy a closure between the host and a GPU", "");
    }
    return CopyClosure(value, memory);
  case Void:
    // Nothing has a value of this type. A trap would make every kernel that
    // copies have a stack that is not fixed (device/device.cpp, Fail).
    memory.Fail("a value of a type that has none: ", type.ctor->name);
  }
  memory.Fail("cannot send a value of type ", type.ctor->name);
}

} // namespace merkurius::runtime

#endif // MERKURIUS_RUNTIME_COPY_H_
