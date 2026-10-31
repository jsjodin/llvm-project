//===-- include/merkurius/Semantics/check-modes.h ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_SEMANTICS_CHECK_MODES_H_
#define MERKURIUS_SEMANTICS_CHECK_MODES_H_

// Mode analysis (design document, "Semantic analysis", pass 4). It reads
// the module's inst and mode definitions and its predicates' and
// functions' mode declarations into procedures (reference manual,
// "Predicate and function mode declarations"), and reports the errors in
// them that mmc reports. Then it checks each procedure's clauses: it
// schedules each conjunction, delaying a goal that would give a mode error
// until the goals after it bind what it needs, classifies each unification
// and chooses a procedure for each call, checks the branches of
// disjunctions and if-then-elses from the same insts and merges their
// results, and keeps negations and conditions from binding what is used
// outside them (Overton, "Precise and Expressive Mode Systems for Typed
// Logic Programming Languages", 2003, chapter 3). A predicate's clauses
// are branches of a disjunction too. A lambda expression's body is
// checked as a clause is, and a higher-order term, and a call of one, by
// its higher-order inst (reference manual, "Higher-order insts and
// modes"). Only the first mode error of a procedure is reported, as by
// mmc.

#include "merkurius/Parser/message.h"
#include "merkurius/Parser/parse-tree.h"
#include "merkurius/Parser/token.h"
#include "merkurius/Semantics/check-types.h"
#include "merkurius/Semantics/insts.h"
#include "merkurius/Semantics/resolve-names.h"
#include "merkurius/Semantics/semantic-form.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include <deque>
#include <memory>
#include <optional>
#include <vector>

namespace merkurius::semantics {

// One mode of a predicate or function: from a mode declaration, a
// predicate or function declaration that combines types and modes, or,
// for a function without mode declarations, the default mode, with every
// argument in and the result out (reference manual, "Predicate and
// function mode declarations").
struct Procedure {
  const Symbol *symbol;
  unsigned number; // in the order of the declarations
  llvm::ArrayRef<ArgMode> modes; // a function's result last
  std::optional<parser::Determinism> determinism;
  // The name in the declaration; empty for a default mode.
  parser::CharBlock source;
  // The inst parameters of the declaration, by number (reference manual,
  // "Constrained polymorphic modes").
  llvm::ArrayRef<const Inst *> parameters{};
};

// What a unification does, by the insts of its sides: X = f(...) with X
// free constructs, and with X bound deconstructs, or tests if f's
// arguments are bound too; X = Y with one side free assigns, and with
// neither tests. A function application or a field access is a call.
ENUM_CLASS(UnifyKind, Construct, Deconstruct, Assign, Test, Call)

// The modes of a goal of a checked clause, by goal number.
struct GoalModes {
  UnifyKind unify{UnifyKind::Test};
  // For an assignment, the variable assigned.
  VarId assigned{0};
  // For a call or a function application: the procedure called; for a
  // higher-order term made from a predicate's or function's name, its
  // procedure.
  const Procedure *callee{nullptr};
  // Whether the call outputs to a fresh variable an argument that is
  // already bound, and tests the two (an implied mode).
  bool implied{false};
  // For a conjunction: its goals in execution order, those of the
  // conjunctions in it among them.
  llvm::ArrayRef<Goal> order{};
};

// A clause, as checked for one procedure.
struct ClauseModes {
  const ClauseTypes *clause;
  const Procedure *procedure;
  llvm::ArrayRef<GoalModes> goals; // by goal number
  // The insts of the head's variables at the end.
  llvm::ArrayRef<const Inst *> final;
};

// Why a procedure's clauses were not checked: they use what mode analysis
// does not check yet, or their predicate has errors of other kinds.
ENUM_CLASS(Unchecked, None, TypeErrors, ModeSpecificClauses,
    ParallelConjunction, FieldClosure, Coerce, Scope, Try, Atomic,
    InstParameters, InferredModes, InferredDeterminism)

class ModeChecking {
public:
  ModeChecking();
  ~ModeChecking();
  InstContext &insts() { return insts_; }
  // The procedures of the module's predicates and functions, in the order
  // of their symbols.
  llvm::ArrayRef<Procedure> procedures() const { return procedures_; }
  // Those of a predicate or function of the module or of an interface it
  // loaded.
  llvm::ArrayRef<const Procedure *> ProceduresOf(const Symbol &) const;
  // The module's own inst definitions, with their bodies.
  llvm::ArrayRef<const Symbol *> instDefinitions() const { return instDefinitions_; }
  // A procedure's clauses as checked, unless it was not checked or has a
  // mode error; why it was not checked.
  llvm::ArrayRef<ClauseModes> ClausesOf(const Procedure &procedure) const {
    auto it{clauses_.find(&procedure)};
    return it == clauses_.end() ? llvm::ArrayRef<ClauseModes>{}
                                : llvm::ArrayRef<ClauseModes>{it->second};
  }
  Unchecked UncheckedReason(const Procedure &procedure) const {
    return unchecked_.lookup(&procedure);
  }
  bool HasError(const Procedure &procedure) const {
    return errors_.count(&procedure);
  }

private:
  friend class ModeChecker;
  friend class ProcedureChecker;
  common::Arena arena_;
  InstContext insts_;
  std::vector<Procedure> procedures_;
  llvm::DenseMap<const Symbol *, std::vector<const Procedure *>> bySymbol_;
  std::vector<const Symbol *> instDefinitions_;
  // Those of imported predicates and functions, and those of fields.
  std::deque<Procedure> imported_;
  llvm::DenseMap<const Procedure *, std::vector<ClauseModes>> clauses_;
  llvm::DenseMap<const Procedure *, Unchecked> unchecked_;
  llvm::DenseSet<const Procedure *> errors_;
  // The clauses of predicates with several, as checked: with head
  // unifications, as mmc checks them.
  std::deque<ClauseForm> forms_;
  std::deque<ClauseTypes> clauseTypes_;
  std::deque<Unify> unifies_;
  std::deque<Conj> conjs_;
};

// Prints a checked clause's goals in execution order, each with its
// position, and the head's final insts.
void DumpClauseModes(llvm::raw_ostream &, const ClauseModes &,
    const parser::TokenSequence &,
    llvm::function_ref<std::string(parser::CharBlock)> position);
// Prints a procedure's name and modes, as mmc does: p(in, out), or
// f(in) = out.
std::string ProcedureToString(const Procedure &);

struct ModeCheckOptions {
  // Infer the modes of predicates without mode declarations, rather than
  // report each (mmc's --infer-modes).
  bool inferModes{false};
  // At most this many passes of mode inference (mmc's
  // --mode-inference-iteration-limit).
  unsigned inferenceIterationLimit{30};
};

std::unique_ptr<ModeChecking> CheckModes(const parser::ParseTree &,
    const parser::TokenSequence &, const SemanticForm &,
    const NameResolution &, TypeChecking &, parser::Messages &,
    const ModeCheckOptions & = {});

} // namespace merkurius::semantics
#endif // MERKURIUS_SEMANTICS_CHECK_MODES_H_
