//===-- include/merkurius/Semantics/check-determinism.h ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_SEMANTICS_CHECK_DETERMINISM_H_
#define MERKURIUS_SEMANTICS_CHECK_DETERMINISM_H_

// Determinism analysis (design document, "Semantic analysis", pass 5). It
// works on the body of each procedure that mode analysis checked, in the
// form mmc gives it after mode analysis: one set of variables for the
// procedure, its clauses the branches of a disjunction that bind the head
// variables, HeadVar__N, its conjunctions in execution order, and each
// unification of a bound X with f(Y1, ..., Yn) a deconstruction into
// distinct new variables, each followed by a test of the new variable and
// the argument it stands for if that was bound. Switch detection turns a
// disjunction whose branches test a variable bound before it against
// different functors into a switch (reference manual, "Determinism
// checking and inference"; Henderson, Somogyi and Conway, "Determinism
// analysis in the Mercury compiler", 1996, section 3), and common
// subexpression elimination moves a deconstruction that every branch
// makes out of the disjunction, so that the variables it binds may be
// switched on (ibid., section 4). The determinism of each goal is then
// inferred from its parts' (reference manual, "Determinism checking and
// inference"; ibid., section 5.1), and a procedure's declared determinism
// checked against its body's, with the goals that make the difference
// (ibid., section 5.3). A committed choice call must be where only its
// first solution is wanted (reference manual, "Committed choice
// nondeterminism").

#include "merkurius/Parser/message.h"
#include "merkurius/Parser/token.h"
#include "merkurius/Semantics/check-modes.h"
#include "merkurius/Semantics/check-types.h"
#include "merkurius/Semantics/insts.h"
#include "merkurius/Semantics/semantic-form.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include <deque>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace merkurius::semantics {

// A variable of a procedure's body.
using BodyVar = std::uint32_t;

struct BodyVarInfo {
  // As mmc names it: the program's name, HeadVar__N, STATE_VARIABLE_S_N,
  // or V_N for the others.
  std::string name;
  const Type *type;
};

ENUM_CLASS(BodyKind, Unify, Call, HOCall, Conj, Disj, Switch, IfThenElse,
    Not, Scope)

struct Body;

// An arm of a switch: the goal run when the variable has the functor, or
// one of the others.
struct SwitchCase {
  InstFunctor functor;
  const Body *goal;
  std::vector<InstFunctor> others{};

  bool Has(const InstFunctor &f) const {
    return functor == f || llvm::is_contained(others, f);
  }
};

struct Body {
  BodyKind kind;
  // Those of the goal it comes from.
  TokenRange tokens;
  // For a goal of the clauses, its modes: the procedure a call calls.
  const GoalModes *modes{nullptr};
  // For a unification: X = Y if isVar, else X = f(Y1, ..., Yn), and how it
  // runs; for a function application or a field access, a call, whose
  // result is X.
  UnifyKind unify{UnifyKind::Test};
  bool isVar{false};
  BodyVar left{0};
  BodyVar var{0};
  const Unify *source{nullptr};
  InstFunctor functor{};
  // Whether it tests the variable of the switch whose case it is in, so
  // that it cannot fail.
  bool switchTest{false};
  // A lambda expression's body, and its head variables.
  const Body *lambda{nullptr};
  std::vector<BodyVar> lambdaHead;
  // The arguments of a unification's functor, of a call, or of a
  // higher-order call, after its closure.
  std::vector<BodyVar> arguments;
  const Call *call{nullptr};
  const HOCall *hoCall{nullptr};
  BodyVar closure{0};
  // A conjunction's or a disjunction's goals; an if-then-else's condition,
  // then part and else part; the goal of a negation or a scope.
  std::vector<const Body *> goals;
  ScopeKind scope{ScopeKind::Some};
  // For a disjunction, and a switch: the insts before it of the variables
  // bound outside it that it uses.
  std::vector<std::pair<BodyVar, const Inst *>> entry;
  // For a switch: its variable, whether it can fail because its cases
  // leave out functors the variable may have, and its cases, by functor.
  // For a deconstruction, whether the variable's inst before it allows
  // another functor, or its type has no known set of constructors.
  BodyVar switchVar{0};
  bool canFail{false};
  std::vector<SwitchCase> cases;
};

// Why a procedure's determinism is not inferred yet: it calls one whose
// determinism is to be inferred, or it has a scope that requires or
// promises a determinism.
ENUM_CLASS(DeterminismUnchecked, None, InferredCallee, Requirement)

// A procedure's body, or an instance method's in one of its modes.
struct ProcedureBody {
  const Procedure *procedure{nullptr};
  const MethodModes *method{nullptr};
  std::vector<BodyVarInfo> vars;
  std::vector<BodyVar> head;
  const Body *body{nullptr};
  // Whether common subexpression elimination changed the body, after
  // which mmc checks its modes again, and so knows the functor of a
  // switch's variable in each case, and of a variable after a
  // deconstruction or an assignment from one known.
  bool rechecked{false};
  // The determinism inferred for the body, and for each of its goals; none
  // if `unchecked`.
  std::optional<parser::Determinism> determinism;
  llvm::DenseMap<const Body *, parser::Determinism> goals;
  DeterminismUnchecked unchecked{DeterminismUnchecked::None};
};

class DeterminismChecking {
public:
  // The bodies of the procedures that mode analysis checked, in the order
  // of ModeChecking's procedures, then the inferred ones, then instance
  // methods'.
  const std::deque<ProcedureBody> &bodies() const { return bodies_; }
  const ProcedureBody *BodyOf(const Procedure &procedure) const {
    return byProcedure_.lookup(&procedure);
  }

private:
  friend class BodyBuilder;
  friend class SwitchDetector;
  friend class DeterminismAnalyser;
  friend std::unique_ptr<DeterminismChecking> CheckDeterminism(
      const parser::TokenSequence &, TypeChecking &, ModeChecking &,
      parser::Messages &);
  std::deque<Body> nodes_;
  std::deque<ProcedureBody> bodies_;
  llvm::DenseMap<const Procedure *, const ProcedureBody *> byProcedure_;
};

// Prints a procedure's body, each goal with its position and its
// determinism, as determinism analysis sees it.
void DumpProcedureBody(llvm::raw_ostream &, const ProcedureBody &,
    const parser::TokenSequence &,
    llvm::function_ref<std::string(parser::CharBlock)> position);

std::unique_ptr<DeterminismChecking> CheckDeterminism(
    const parser::TokenSequence &, TypeChecking &, ModeChecking &,
    parser::Messages &);

} // namespace merkurius::semantics
#endif // MERKURIUS_SEMANTICS_CHECK_DETERMINISM_H_
