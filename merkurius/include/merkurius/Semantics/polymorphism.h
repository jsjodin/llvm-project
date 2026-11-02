//===-- include/merkurius/Semantics/polymorphism.h --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_SEMANTICS_POLYMORPHISM_H_
#define MERKURIUS_SEMANTICS_POLYMORPHISM_H_

// Polymorphism analysis (design document, "Semantic analysis", pass 7):
// where each type_info a procedure needs comes from, a run-time
// description of a type that a polymorphic procedure, compiled once for
// every type, is passed for each of its type variables, as mmc does. A
// procedure has a hidden type_info argument for each type variable of its
// signature, in their order, which is that of their first appearance in
// its declaration, before its other arguments. A call passes one for each
// of its callee's, for the type the callee's variable stands for there:
// the caller's own hidden argument for one of its type variables; a
// constant for a type without variables; one built from the type's
// constructor and its arguments' type_infos for another; and a constant
// for void for a type variable that nothing determines. The body is not
// rewritten: code generation makes them.

#include "merkurius/Common/enum-class.h"
#include "merkurius/Parser/token.h"
#include "merkurius/Semantics/check-determinism.h"
#include "merkurius/Semantics/check-types.h"
#include "merkurius/Semantics/types.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <deque>
#include <memory>
#include <string>
#include <vector>

namespace merkurius::semantics {

ENUM_CLASS(TypeInfoKind, Parameter, Constant, Cell, Void)

// Where a type_info comes from: the procedure's hidden argument for one
// of its type variables, a constant for a type without type variables, a
// cell built from the type's constructor and its arguments' type_infos, or
// the constant for void, the type of a type variable nothing determines.
struct TypeInfo {
  TypeInfoKind kind;
  const Type *type{nullptr}; // the type described; null for void
  unsigned parameter{0}; // for a Parameter, its hidden argument's number
  std::vector<const TypeInfo *> arguments{}; // for a Cell
};

// Why a procedure is not analysed yet: it, or one it calls, has type class
// constraints; it, or one it calls, has existentially quantified types;
// it is an instance's method; or it makes a closure of a polymorphic
// predicate or function.
ENUM_CLASS(PolymorphismUnchecked, None, Constraints, Existential, Method,
    Closure)

struct ProcedurePolymorphism {
  const ProcedureBody *body;
  // The type variables of the procedure's signature, by number, each with
  // a hidden type_info argument, in this order.
  unsigned parameters{0};
  llvm::ArrayRef<llvm::StringRef> names;
  // The type_infos that a call or a function application passes, for its
  // callee's type variables in order, and that a test of two variables of
  // a type with type variables needs, for the type.
  llvm::DenseMap<const Body *, std::vector<const TypeInfo *>> goals;
  PolymorphismUnchecked unchecked{PolymorphismUnchecked::None};
};

class PolymorphismAnalysis {
public:
  // In the order of DeterminismChecking's bodies.
  const std::deque<ProcedurePolymorphism> &procedures() const {
    return procedures_;
  }

private:
  friend class PolymorphismAnalyser;
  friend std::unique_ptr<PolymorphismAnalysis> AnalysePolymorphism(
      TypeChecking &, const DeterminismChecking &);
  std::deque<TypeInfo> nodes_;
  std::deque<ProcedurePolymorphism> procedures_;
};

std::unique_ptr<PolymorphismAnalysis> AnalysePolymorphism(
    TypeChecking &, const DeterminismChecking &);

// Prints a procedure's hidden arguments, and the type_infos of each goal
// that has them, in the order of its body.
void DumpProcedurePolymorphism(llvm::raw_ostream &,
    const ProcedurePolymorphism &, const parser::TokenSequence &,
    llvm::function_ref<std::string(parser::CharBlock)> position);

} // namespace merkurius::semantics
#endif // MERKURIUS_SEMANTICS_POLYMORPHISM_H_
