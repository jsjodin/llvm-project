//===-- runtime/walk.h - Walking a value by its type ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// What the runtime's walkers share, io.write's (write.cpp) and the message
// copy's (mailbox.cpp): a field's type resolved in the environment of the
// term it is in, and a term's functor and fields (design document, "Type
// descriptors").
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_RUNTIME_WALK_H_
#define MERKURIUS_RUNTIME_WALK_H_

#include "descriptor.h"

namespace merkurius::runtime {

// What a type parameter stands for, in the type whose arguments these are,
// and, for the fields of an existentially typed constructor's cell, the cell,
// whose hidden fields have the type_infos of its existentially quantified type
// variables.
struct Environment {
  const Word *arguments;
  const Environment *parent;
  const Word *cell{nullptr};
};

// A type: its constructor, and its arguments, in the environment they were
// made in. A walker does not allocate, so a field's type stays as it is.
struct Type {
  const TypeCtor *ctor;
  Environment environment;
};

inline Type Resolve(Word type, const Environment *environment) {
  for (;;) {
    if ((type & kTagMask) == kVariableTag) {
      type = environment->arguments[type >> kTagBits];
      environment = environment->parent;
      continue;
    }
    if ((type & kTagMask) == kHiddenTag) {
      Word place{type >> kTagBits};
      type = environment->cell[place & 0xffff];
      if (Word word{place >> 16}) {
        type = reinterpret_cast<const Word *>(type)[word];
      }
      // A type_info has no type parameters.
      environment = nullptr;
      continue;
    }
    break;
  }
  if ((type & kTagMask) == 0) {
    return {reinterpret_cast<const TypeCtor *>(type), {nullptr, environment}};
  }
  const Word *cell{reinterpret_cast<const Word *>(type - 1)};
  const auto *ctor{reinterpret_cast<const TypeCtor *>(cell[0])};
  // A tuple's or closure's cell has the number of its arguments in word 1.
  return {ctor,
      {cell + (ctor->kind == Tuple || ctor->kind == Closure ? 2 : 1), environment}};
}

// Whether a field's type is the type of the term it is in, as a list's tail
// is, so that a loop can go on with that type.
inline bool IsSameType(Word field, const Type &type) {
  if ((field & kTagMask) == 0) {
    return reinterpret_cast<const TypeCtor *>(field) == type.ctor;
  }
  if ((field & kTagMask) != 1) {
    return false;
  }
  const Word *cell{reinterpret_cast<const Word *>(field - 1)};
  if (reinterpret_cast<const TypeCtor *>(cell[0]) != type.ctor ||
      type.ctor->kind == Tuple) {
    return false;
  }
  for (unsigned i{0}; i < type.ctor->arity; ++i) {
    if (cell[1 + i] != ((Word{i} << kTagBits) | kVariableTag)) {
      return false;
    }
  }
  return true;
}

// The functor of a term, and where its fields are.
inline const Functor *FunctorOf(Word &value, const TypeCtor *ctor, const Word *&fields) {
  switch (ctor->kind) {
  case Enum:
    return &ctor->functors[value];
  case Dummy:
    return &ctor->functors[0];
  case NoTag:
    fields = &value;
    return &ctor->functors[0];
  case Tuple: {
    const Functor *functor{&ctor->functors[0]};
    // A tuple of one argument is that argument, as a NoTag is.
    fields = functor->arity == 1
        ? &value
        : reinterpret_cast<const Word *>(value - functor->tag);
    return functor;
  }
  default:
    break;
  }
  unsigned tag{static_cast<unsigned>(value & kTagMask)};
  uint16_t index{ctor->byTag[tag]};
  if (index == kConstantTag) {
    return &ctor->functors[ctor->constants[value >> kTagBits]];
  }
  const Word *cell{reinterpret_cast<const Word *>(value - tag)};
  // An existentially typed constructor's arguments are after its hidden
  // fields.
  if (index == kSharedTag) {
    const Functor *functor{&ctor->functors[ctor->shared[cell[0]]]};
    fields = cell + 1 + functor->hidden;
    return functor;
  }
  const Functor *functor{&ctor->functors[index]};
  fields = cell + functor->hidden;
  return functor;
}

} // namespace merkurius::runtime

#endif // MERKURIUS_RUNTIME_WALK_H_
