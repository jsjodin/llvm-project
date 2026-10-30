//===-- include/merkurius/Semantics/types.h ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_SEMANTICS_TYPES_H_
#define MERKURIUS_SEMANTICS_TYPES_H_

// Types (reference manual, "Types"), interned, so that equal types are the
// same pointer (design document, "What semantic analysis writes"): a type
// variable, a type constructor applied to arguments, a tuple, or a
// higher-order type. Equivalence types are expanded before a type is
// made, so a Named type's constructor is never an equivalence.

#include "merkurius/Common/arena.h"
#include "merkurius/Common/enum-class.h"
#include "merkurius/Parser/parse-tree.h"
#include "merkurius/Semantics/symbols.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace merkurius::semantics {

ENUM_CLASS(TypeKind, Variable, Named, Tuple, HigherOrder)

struct Type {
  TypeKind kind;
  unsigned var; // for a Variable, its number
  const Symbol *constructor; // for a Named type
  // A Named type's arguments, a tuple's elements, or a higher-order type's
  // arguments, without a function's result.
  llvm::ArrayRef<const Type *> arguments;
  parser::PredOrFunc predOrFunc; // for a HigherOrder type
  parser::Purity purity; // for a HigherOrder type
  const Type *result; // for a higher-order function type
};

class TypeContext {
public:
  const Type *Variable(unsigned);
  const Type *Named(const Symbol &, llvm::ArrayRef<const Type *> arguments);
  const Type *Tuple(llvm::ArrayRef<const Type *> elements);
  const Type *HigherOrder(parser::PredOrFunc, parser::Purity,
      llvm::ArrayRef<const Type *> arguments, const Type *result);

  // The type with each variable replaced by `replace`'s type for it.
  const Type *Substitute(
      const Type *, llvm::function_ref<const Type *(unsigned)> replace);

private:
  const Type *Intern(const Type &);

  common::Arena arena_;
  std::map<std::vector<std::uintptr_t>, const Type *> interned_;
};

// Calls `f` once on each variable of the type, in the order they occur.
void ForEachVariable(const Type &, llvm::function_ref<void(unsigned)> f);

// The number of nodes the type has as a tree, which can be exponential in
// the number it has, since types share their parts; at most `limit`.
std::size_t TreeSize(const Type &, std::size_t limit);

// Prints a type as Mercury writes it, module-qualified, as mmc's messages
// do: list.list(int), {int, string}, pred(int), (func(int) = int), and
// the primitive char as character. A
// variable is printed by `name`, or as T<number> without one. TypeToString
// gives a placeholder for a type too large to read.
void PrintType(llvm::raw_ostream &, const Type &,
    llvm::function_ref<std::string(unsigned)> name = nullptr);
std::string TypeToString(
    const Type &, llvm::function_ref<std::string(unsigned)> name = nullptr);

// The types of a predicate, a function or a function symbol: a function's
// result is its last argument, and a constructor, field selector or field
// update is a function. Its type variables are numbered from 0.
struct Signature {
  parser::PredOrFunc kind;
  parser::Purity purity;
  unsigned typeVars;
  llvm::ArrayRef<llvm::StringRef> varNames; // as declared, by number
  llvm::ArrayRef<const Type *> arguments;
};

} // namespace merkurius::semantics
#endif // MERKURIUS_SEMANTICS_TYPES_H_
