//===-- runtime/type-info.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The order of types, by their type_infos, by which an existentially typed
// constructor's argument is compared before its value, and unified, as mmc's
// are: by the type constructors' modules, names and arities, a primitive
// type's module being builtin and a tuple's name tuple, and then by their
// arguments in order.
//
//===----------------------------------------------------------------------===//

#include "descriptor.h"
#include "process.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace merkurius::runtime;

namespace {

// A type_info's constructor, and its arguments' type_infos.
struct Parts {
  const TypeCtor *ctor;
  const Word *arguments;
  Word arity;
};

Parts PartsOf(Word info) {
  if ((info & kTagMask) == 0) {
    const auto *ctor{reinterpret_cast<const TypeCtor *>(info)};
    return {ctor, nullptr, 0};
  }
  const Word *cell{reinterpret_cast<const Word *>(info - 1)};
  const auto *ctor{reinterpret_cast<const TypeCtor *>(cell[0])};
  if (ctor->kind == Tuple || ctor->kind == Closure) {
    return {ctor, cell + 2, cell[1]};
  }
  return {ctor, cell + 1, ctor->arity};
}

int Sign(int order) { return (order > 0) - (order < 0); }

int CompareTypes(Word x, Word y) {
  if (x == y) {
    return 0;
  }
  Parts a{PartsOf(x)}, b{PartsOf(y)};
  if (a.ctor != b.ctor) {
    // A descriptor of the primitive types and tuples has no module.
    const char *aModule{*a.ctor->module ? a.ctor->module : "builtin"};
    const char *bModule{*b.ctor->module ? b.ctor->module : "builtin"};
    if (int order{std::strcmp(aModule, bModule)}) {
      return Sign(order);
    }
    const char *aName{a.ctor->kind == Tuple ? "tuple" : a.ctor->name};
    const char *bName{b.ctor->kind == Tuple ? "tuple" : b.ctor->name};
    if (int order{std::strcmp(aName, bName)}) {
      return Sign(order);
    }
  }
  if (a.arity != b.arity) {
    return a.arity < b.arity ? -1 : 1;
  }
  for (Word i{0}; i < a.arity; ++i) {
    if (int order{CompareTypes(a.arguments[i], b.arguments[i])}) {
      return order;
    }
  }
  return 0;
}

} // namespace

extern "C" std::int32_t merk_compare_type_infos(Word x, Word y) {
  return CompareTypes(x, y);
}

// A unification or comparison of closures, through a type variable, stops the
// program as mmc's does, which names a function's type pred too.
extern "C" [[noreturn]] void merk_higher_order_called(std::int32_t compare) {
  std::fflush(stdout);
  std::fprintf(stderr, "Mercury runtime: called %s for `pred' type\n",
      compare ? "compare/3" : "unify/2");
  merkurius::runtime::ExitProgram(1);
}
