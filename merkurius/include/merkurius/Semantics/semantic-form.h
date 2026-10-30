//===-- include/merkurius/Semantics/semantic-form.h -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_SEMANTICS_SEMANTIC_FORM_H_
#define MERKURIUS_SEMANTICS_SEMANTIC_FORM_H_

// The semantic form: the desugared body of each clause, in a few goal
// kinds (design document, "Semantic form"). Bodies are in superhomogeneous
// form: every argument is a variable, and each unification holds at most
// one functor. Nodes are immutable and live in a SemanticArena, like the
// parse tree's, and each records the tokens of the parse-tree node it came
// from, which diagnostics use.
//
// Variables are numbered densely within their clause, and every goal has a
// number, also dense within its clause, by which semantic analysis indexes
// its tables.

#include "merkurius/Common/arena.h"
#include "merkurius/Common/enum-class.h"
#include "merkurius/Parser/parse-tree.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <optional>
#include <variant>

namespace merkurius::semantics {

using SemanticArena = common::Arena;
using parser::TokenRange;

// A variable of a clause, an index into its ClauseForm's variables.
using VarId = std::uint32_t;

ENUM_CLASS(VarKind, Program, Anonymous, Fresh, State)

// A program variable has the name it was written with, and each _ is an
// anonymous variable of its own; desugaring makes the fresh ones, and one
// variable for each version of a state variable, numbered from 0 within the
// clause.
struct VarInfo {
  VarKind kind;
  llvm::StringRef name; // for a program variable or a state variable
  TokenRange tokens; // where it first appears
  unsigned version{0}; // for a state variable
};

struct Conj;
struct Disj;
struct IfThenElse;
struct Not;
struct Unify;
struct Call;
struct HOCall;
struct Scope;
struct Try;
struct Atomic;
using Goal = std::variant<const Conj *, const Disj *, const IfThenElse *,
    const Not *, const Unify *, const Call *, const HOCall *, const Scope *,
    const Try *, const Atomic *>;

// G1, G2, ... or G1 & G2 & ...; true is the empty conjunction.
struct Conj {
  TokenRange tokens;
  unsigned number;
  bool parallel;
  llvm::ArrayRef<Goal> goals;
};

// G1 ; G2 ; ...; fail is the empty disjunction.
struct Disj {
  TokenRange tokens;
  unsigned number;
  llvm::ArrayRef<Goal> goals;
};

struct IfThenElse {
  TokenRange tokens;
  unsigned number;
  Goal condition;
  Goal then;
  Goal otherwise;
};

// not G, which implications, all and \= become too.
struct Not {
  TokenRange tokens;
  unsigned number;
  Goal goal;
};

// The function symbol of a unification: a name, which semantic analysis
// resolves to a constructor or a function, the list constructors, a tuple,
// a literal, an implementation-defined literal such as $line, coerce, or a
// lambda expression.
ENUM_CLASS(FunctorKind, Name, Cons, Nil, Tuple, Literal,
    ImplementationLiteral, Coerce, Lambda)

struct Lambda;

struct Functor {
  FunctorKind kind;
  const parser::SymName *name; // for Name
  const parser::Literal *literal; // for Literal
  const parser::ImplementationLiteral *implementation;
  const Lambda *lambda{nullptr}; // for Lambda
};

// X = Y, or X = f(Y1, ..., Yn).
struct Unify {
  TokenRange tokens;
  unsigned number;
  VarId left;
  bool isVar; // X = Y
  VarId var; // Y
  Functor functor; // f
  llvm::ArrayRef<VarId> arguments; // Y1, ..., Yn
};

// pred(X1::M1, ...) is Det :- Body, or func(...) = (Y::M) is Det :- Body,
// the functor of X = Lambda. Its head's arguments are distinct variables,
// local to it, like a clause's: a function's result is the last of them,
// and a DCG lambda's two implicit arguments the last two. Each argument
// has the mode it was written with, if any. The variables of the body that
// are not arguments may be shared with the enclosing clause.
struct Lambda {
  TokenRange tokens;
  parser::PredOrFunc kind;
  parser::Purity purity;
  llvm::ArrayRef<VarId> arguments;
  llvm::ArrayRef<std::optional<parser::Mode>> modes;
  std::optional<parser::Determinism> determinism;
  Goal body;
};

// p(X1, ..., Xn); event p(...) is a call too.
struct Call {
  TokenRange tokens;
  unsigned number;
  const parser::SymName *name;
  llvm::ArrayRef<VarId> arguments;
  bool event;
};

// P(X1, ..., Xn) or call(P, X1, ..., Xn), and, for a function, Y = F(X1,
// ..., Xn) or Y = apply(F, X1, ..., Xn), whose result is the last argument.
struct HOCall {
  TokenRange tokens;
  unsigned number;
  parser::PredOrFunc kind;
  VarId closure;
  llvm::ArrayRef<VarId> arguments;
};

// A goal under a quantifier, a promise, a requirement, a purity
// annotation, a trace goal or an OpenMP directive. The parse-tree goal
// holds the scope's parameters; vars are the variables it names, which for
// a trace goal are the first and last versions of the state variable of
// each io(!S) and state(M, !S) parameter.
ENUM_CLASS(ScopeKind, Some, PromisePurity, PromiseSolutions,
    RequireDeterminism, RequireSwitch, DisableWarnings, Trace, Purity, Omp)

// The goal or DCG goal that a node came from.
using ParseGoal = std::variant<parser::Goal, parser::DcgGoal>;

struct Scope {
  TokenRange tokens;
  unsigned number;
  ScopeKind kind;
  ParseGoal origin;
  llvm::ArrayRef<VarId> vars;
  Goal goal;
};

// catch Pattern -> Goal, or catch_any Var -> Goal, whose pattern is
// empty: the exception is unified with `exception`, by `pattern`, which
// chooses this branch if it succeeds.
struct Catch {
  TokenRange tokens;
  VarId exception;
  Goal pattern;
  Goal goal;
};

// try [io(!S), ...] Goal then Then else Else catch ... (reference manual,
// "Exception handling"). `io` holds the versions of each io(!S)'s state
// variable before and after Goal, in pairs. Goal's updates of the other
// state variables are seen in Then only, as in an if-then-else.
struct Try {
  TokenRange tokens;
  unsigned number;
  llvm::ArrayRef<VarId> io;
  Goal goal;
  Goal then;
  std::optional<Goal> otherwise;
  llvm::ArrayRef<Catch> catches;
  std::optional<Catch> catchAny;
};

// One alternative of an atomic goal, with the first and last versions of
// its inner state, which is its own.
struct AtomicAlternative {
  VarId innerInitial;
  VarId innerFinal;
  Goal goal;
};

// atomic [outer(...), inner(...), vars(...)] G1 or_else G2 ..., a
// transaction of mmc's software transactional memory. The outer state's
// versions are those before and after the goal; it is not visible inside.
struct Atomic {
  TokenRange tokens;
  unsigned number;
  VarId outerInitial;
  VarId outerFinal;
  llvm::ArrayRef<VarId> vars;
  llvm::ArrayRef<AtomicAlternative> alternatives;
};

// Term : Type or Term `with_type` Type, recorded against the variable that
// holds Term.
struct TypeQualification {
  TokenRange tokens;
  VarId var;
  parser::Type type;
};

// A clause, as Head :- Body with distinct variables as the head's
// arguments; a function's result is the last of them, and a DCG rule's two
// implicit arguments are the last two.
struct ClauseForm {
  TokenRange tokens;
  parser::PredOrFunc kind;
  const parser::SymName *name;
  llvm::ArrayRef<VarId> head;
  Goal body;
  llvm::ArrayRef<VarInfo> vars;
  llvm::ArrayRef<TypeQualification> typeQualifications;
  unsigned goals; // the number of goals
};

// The clauses of a file that desugared without errors, in order.
struct SemanticForm {
  llvm::ArrayRef<const ClauseForm *> clauses;
};

// Prints the semantic form, one goal per line, indented by depth. Program
// variables print as their names, anonymous ones as _$N and fresh ones as
// $N, where N is the variable's number, and version V of state variable S
// as S!V. A program variable with the name of an earlier one, which a
// lambda's argument may have, prints as Name$N. A lambda's body is printed
// under the unification whose functor it is.
void DumpSemanticForm(llvm::raw_ostream &, const SemanticForm &);

} // namespace merkurius::semantics
#endif // MERKURIUS_SEMANTICS_SEMANTIC_FORM_H_
