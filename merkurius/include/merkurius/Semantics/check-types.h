//===-- include/merkurius/Semantics/check-types.h ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_SEMANTICS_CHECK_TYPES_H_
#define MERKURIUS_SEMANTICS_CHECK_TYPES_H_

// Type checking (design document, "Semantic analysis", pass 2): finds each
// clause's type assignment, a type for every variable and a declaration
// for every overloaded name, which must be unique up to renaming
// (reference manual, "Predicate and function type declarations").

#include "merkurius/Parser/message.h"
#include "merkurius/Parser/token.h"
#include "merkurius/Semantics/resolve-names.h"
#include "merkurius/Semantics/semantic-form.h"
#include "merkurius/Semantics/types.h"
#include "llvm/ADT/DenseMap.h"
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace merkurius::semantics {

// A clause's type assignment. The type variables of its predicate's or
// function's declaration are numbered from 0, in their declared order;
// any other variable left in a type is one that nothing determined.
struct ClauseTypes {
  const ClauseForm *clause;
  const Symbol *definition; // what the clause defines
  bool declared; // its types were declared, and the clause checked by them
  bool inferred; // its types were inferred, and the clause checked by them
  // The names of the declared or inferred signature's type variables.
  llvm::ArrayRef<llvm::StringRef> typeVarNames;
  llvm::ArrayRef<const Type *> vars; // by VarId
  // By goal number, the constructor, function or predicate chosen for the
  // functor of a unification or the name of a call, and that name; a
  // character constant's symbol is null, and a goal with a type error has
  // neither.
  llvm::ArrayRef<const Symbol *> symbols;
  llvm::ArrayRef<const parser::SymName *> names;
};

class TypeChecking {
public:
  TypeChecking();
  ~TypeChecking();
  TypeContext &types() { return types_; }
  llvm::ArrayRef<ClauseTypes> clauses() const { return clauses_; }
  // The signature of a predicate, function, constructor or field: for a
  // predicate or function without a declaration, its inferred one.
  const Signature *SignatureOf(const Symbol &symbol) const {
    auto it{signatures_.find(&symbol)};
    return it == signatures_.end() || !it->second ? nullptr : &*it->second;
  }
  // Prints a type of the clause, its type variables by their names.
  std::string Print(const ClauseTypes &, const Type &) const;

private:
  friend class TypeChecker;
  common::Arena arena_;
  TypeContext types_;
  // The type of an undefined name, which matches every type, so that it
  // draws no more errors. Signatures and clause types may contain it.
  Symbol error_{SymbolKind::Type, "", "<error>", 0, std::monostate{}, {},
      false};
  llvm::DenseMap<const Symbol *, std::optional<Signature>> signatures_;
  std::vector<ClauseTypes> clauses_;
};

// Prints a symbol with its signature: func int.'+'(int, int) = int, or
// pred m.p(T) <= (m.c(T)).
std::string SignatureToString(const Symbol &, const Signature &);
// Prints a constraint, module-qualified: m.c(list.list(T)).
void PrintConstraint(llvm::raw_ostream &, const ClassConstraint &,
    llvm::function_ref<std::string(unsigned)> name = nullptr);

struct TypeCheckOptions {
  // Infer the types of predicates and functions without declarations,
  // rather than report each as an error too (mmc's --infer-types).
  bool inferTypes{true};
  // At most this many passes of inference (mmc's
  // --type-inference-iteration-limit).
  unsigned inferenceIterationLimit{60};
};

// Checks the types of the module's clauses. The types of predicates and
// functions without declarations are inferred, one strongly connected
// component of the call graph at a time, callees first: each component's
// signatures start most general, and are recomputed from its clauses until
// they do not change.
std::unique_ptr<TypeChecking> CheckTypes(const SemanticForm &,
    const NameResolution &, const parser::TokenSequence &, parser::Messages &,
    const TypeCheckOptions & = {});

} // namespace merkurius::semantics
#endif // MERKURIUS_SEMANTICS_CHECK_TYPES_H_
