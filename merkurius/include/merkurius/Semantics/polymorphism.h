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
// where each type_info and typeclass_info a procedure needs comes from, as
// mmc does it (Jeffery's thesis, chapter 3). A type_info is a run-time
// description of a type, which a polymorphic procedure, compiled once for
// every type, is passed for each of its type variables; a typeclass_info,
// or dictionary, is one of an instance, with its methods, which it is
// passed for each of its class constraints. A procedure's hidden
// arguments, before its other arguments, are a type_info for each type
// variable of its signature that no constraint has, in their order, which
// is that of their first appearance in its declaration, and then a
// typeclass_info for each constraint, in theirs; an instance method's are
// the instance's, and then the method's. A constrained type variable's
// type_info is in the typeclass_info of the first constraint that has it.
// A call passes those of its callee: for a type variable, the caller's own
// type_info for one of its type variables, a constant for a type without
// them, one built from the type's constructor and its arguments' type_infos
// for another, and a constant for void for one that nothing determines;
// for a constraint, as the type checker proved it, the caller's own
// typeclass_info, a superclass's taken out of one, a constant for a
// constraint without type variables, and one built from the instance's
// with the typeclass_infos of the instance's own constraints for another.
// An existentially typed predicate or function gives its caller, after
// those, a type_info for each of its existentially quantified type
// variables that its existential constraints do not have, and then a
// typeclass_info for each existential constraint, which its clauses make
// as for a call; a construction with an existentially typed constructor
// passes them as a call does, and a deconstruction gets them. The body is
// not rewritten: code generation makes them.

#include "merkurius/Common/enum-class.h"
#include "merkurius/Parser/token.h"
#include "merkurius/Semantics/check-determinism.h"
#include "merkurius/Semantics/check-modes.h"
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

ENUM_CLASS(TypeInfoKind, Parameter, Constant, Cell, Void, Dictionary,
    Argument, Output)

struct Dictionary;

// Where a type_info comes from: the procedure's hidden argument for one
// of its type variables, a constant for a type without type variables, a
// cell built from the type's constructor and its arguments' type_infos,
// the constant for void, the type of a type variable nothing determines,
// the type_info in a hidden typeclass_info at a place, an argument of
// another type_info, or one that a goal gives, for a type its callee or
// the constructor it takes apart has existentially quantified. The types
// that goals give are printed as type variables, from kExistentialVar on.
inline constexpr unsigned kExistentialVar{1u << 30};

struct TypeInfo {
  TypeInfoKind kind;
  const Type *type{nullptr}; // the type described; null for void
  unsigned parameter{0}; // for a Parameter, its type variable
  // For a Cell, its arguments' type_infos; for an Argument, the one it is
  // an argument of.
  std::vector<const TypeInfo *> arguments{};
  const Dictionary *dictionary{nullptr}; // for a Dictionary
  // For a Dictionary, its place in it, from 1, after the type class's
  // superclasses, as mmc numbers them; for an Argument, which, from 1.
  unsigned index{0};
  const Body *goal{nullptr}; // for an Output, the goal that gives it
  const Symbol *callee{nullptr}; // and its callee or constructor
};

ENUM_CLASS(DictionaryKind, Parameter, Constant, Superclass, Instance, Output)

// Where a typeclass_info comes from: the procedure's hidden argument for one
// of its constraints, a constant for a constraint without type variables,
// a superclass's taken out of another, or a cell built from an instance's
// with the type_infos of the instance's type variables that its
// constraints do not have, the typeclass_infos of its constraints, those of
// its type class's superclasses on its types, and the type_infos of the
// constraint's types; or one that a goal gives, for an existential
// constraint of its callee or of the constructor it takes apart.
struct Dictionary {
  DictionaryKind kind;
  ClassConstraint constraint;
  unsigned parameter{0}; // for a Parameter, its constraint's number
  // For a Superclass, the typeclass_info it is taken out of, and which of
  // its type class's superclasses it is, from 1.
  const Dictionary *from{nullptr};
  unsigned index{0};
  const InstanceHead *instance{nullptr}; // for an Instance
  std::vector<const TypeInfo *> typeInfos{};
  std::vector<const Dictionary *> premises{};
  std::vector<const Dictionary *> supers{};
  std::vector<const TypeInfo *> types{};
  const Body *goal{nullptr}; // for an Output, the goal that gives it
  const Symbol *callee{nullptr}; // and its callee or constructor
};

// A hidden argument: a type_info or a typeclass_info, which the
// procedure is given, or gives for an existentially quantified type
// variable or an existential constraint.
struct HiddenArgument {
  const TypeInfo *typeInfo{nullptr};
  const Dictionary *dictionary{nullptr};
  bool output{false};
};

// What a goal passes: type_infos for its callee's type variables that its
// constraints do not have, in order, and typeclass_infos for its
// constraints, those for its existentially quantified ones and its
// existential constraints being the ones it gives; a test of two variables
// of a type with type variables, the type's type_info. What a clause of an
// existentially typed procedure gives for its own.
struct PassedInfos {
  std::vector<const TypeInfo *> typeInfos;
  std::vector<const Dictionary *> dictionaries;
  const Symbol *constructor{nullptr}; // a construction's or deconstruction's
};

// Why a procedure is not analysed yet: it is an instance method with
// existentially quantified types, or it uses a type that a goal it does not
// analyse gives; a call's constraints were not proved; or it makes a
// closure of a polymorphic predicate or function.
ENUM_CLASS(PolymorphismUnchecked, None, Constraints, Existential, Closure)

struct ProcedurePolymorphism {
  const ProcedureBody *body;
  // The number of type variables of the procedure's signature, and their
  // names, by number.
  unsigned parameters{0};
  llvm::ArrayRef<llvm::StringRef> names;
  std::vector<HiddenArgument> hidden;
  // By type variable, its type_info.
  std::vector<const TypeInfo *> sources;
  llvm::DenseMap<const Body *, PassedInfos> goals;
  // For an existentially typed procedure, what each clause gives.
  std::vector<PassedInfos> results;
  // The types that goals give which the clauses give for the procedure's
  // own existentially quantified type variables, by their variables: they
  // are printed by those names.
  llvm::DenseMap<unsigned, unsigned> givenAs;
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
      TypeChecking &, const ModeChecking &, const DeterminismChecking &);
  std::deque<TypeInfo> nodes_;
  std::deque<Dictionary> dictionaries_;
  std::deque<std::vector<const Type *>> typeLists_;
  std::deque<ProcedurePolymorphism> procedures_;
};

std::unique_ptr<PolymorphismAnalysis> AnalysePolymorphism(
    TypeChecking &, const ModeChecking &, const DeterminismChecking &);

// Prints a procedure's hidden arguments, where the type_infos of its
// constrained type variables come from, and what each goal that passes
// type_infos or typeclass_infos passes, in the order of its body.
void DumpProcedurePolymorphism(llvm::raw_ostream &,
    const ProcedurePolymorphism &, const parser::TokenSequence &,
    llvm::function_ref<std::string(parser::CharBlock)> position);

} // namespace merkurius::semantics
#endif // MERKURIUS_SEMANTICS_POLYMORPHISM_H_
