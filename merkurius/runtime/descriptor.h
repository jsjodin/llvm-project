//===-- runtime/descriptor.h - Type descriptors -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The static records that type_infos point to (design document, "Type
// descriptors"), as lib/CodeGen/Lowering.cpp's LowerTypeCtors makes them, and
// the words of type_infos and of fields' types.
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_RUNTIME_DESCRIPTOR_H_
#define MERKURIUS_RUNTIME_DESCRIPTOR_H_

#include <cstdint>

namespace merkurius::runtime {

using Word = std::uint64_t;

// The fields' order and types are LowerTypeCtors'. An existentially typed
// constructor's cell has `hidden` fields before its arguments: the type_infos
// and typeclass_infos of its existentially quantified type variables.
struct Functor {
  const char *name;
  std::uint32_t arity;
  std::uint8_t tag;
  std::uint8_t hidden;
  std::int32_t secondary;
  std::int64_t constant;
  const Word *fields;
};

struct TypeCtor {
  std::uint8_t kind;
  std::uint8_t flags;
  std::uint16_t arity;
  std::uint32_t functorCount;
  const char *module;
  const char *name;
  const void *unify;
  const void *compare;
  const Functor *functors;
  std::uint16_t byTag[8];
  const std::uint16_t *constants;
  const std::uint16_t *shared;
};

// The order of Lowering.cpp's DescriptorKind.
enum Kind : std::uint8_t {
  Int,
  Uint,
  Int8,
  Int16,
  Int32,
  Int64,
  Uint8,
  Uint16,
  Uint32,
  Uint64,
  Float,
  Char,
  String,
  Array,
  Dummy,
  Void,
  Enum,
  NoTag,
  Cells,
  Tuple,
  // A higher-order type's, named pred or func, for any number of arguments.
  Closure,
};

// byTag's marks, for a tag whose word is a constant, and for one that
// constructors share.
constexpr std::uint16_t kConstantTag{0xffff};
constexpr std::uint16_t kSharedTag{0xfffe};

constexpr Word kTagMask{7};
constexpr unsigned kTagBits{3};
// A type_info is a descriptor, tagged 0, or a cell of one and its arguments'
// type_infos, tagged 1, whose word 1 is a tuple's number of arguments. A
// field's type is also a type parameter of its term's type, by its number,
// tagged 2, or where an existentially typed constructor's cell has an
// existentially quantified type variable's type_info, tagged 3: in the hidden
// field of its low 16 bits, or in the word of that typeclass_info its high
// bits give, if they are not 0.
constexpr Word kVariableTag{2};
constexpr Word kHiddenTag{3};

// Word 1 of a closure's cell, after its entry's code; the captured values
// follow it. Each capture's word is a field's type, in which a type variable
// is hidden-tagged: the place of its type_info among the captures. Or one of
// the marks below. The layout is null on a device.
struct ClosureLayout {
  Word count;
  Word captures[];
};
// What a typeclass_info holds after its method table, which is its word 0:
// the table's word 0, in 16 bits each from the low end.
struct DictionaryShape {
  Word superclasses;
  Word typeInfos;
  Word constraints;
};
inline DictionaryShape ShapeOfDictionary(const Word *dictionary) {
  Word shape{*reinterpret_cast<const Word *>(dictionary[0])};
  return {shape & 0xffff, (shape >> 16) & 0xffff, (shape >> 32) & 0xffff};
}
constexpr Word kCaptureUnknown{0};
constexpr Word kCaptureTypeInfo{8};
constexpr Word kCaptureTypeclassInfo{16};

} // namespace merkurius::runtime

#endif // MERKURIUS_RUNTIME_DESCRIPTOR_H_
