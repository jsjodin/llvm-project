//===-- lib/Semantics/check-uniqueness.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "merkurius/Semantics/check-uniqueness.h"
#include "llvm/ADT/STLExtras.h"
#include <map>
#include <optional>
#include <set>
#include <string>

namespace merkurius::semantics {

using namespace parser::literals;

namespace {

using Vars = std::set<BodyVar>;

class UniquenessChecker {
public:
  UniquenessChecker(const parser::TokenSequence &tokens, ModeChecking &modes,
      const ProcedureBody &body)
      : tokens_{tokens}, modes_{modes}, insts_{modes.insts()},
        ops_{modes.ops()}, body_{body} {}

  // The first error in the body, if any: where, and its text.
  std::optional<std::pair<parser::CharBlock, std::string>> Check();

private:
  parser::CharBlock Source(TokenRange range) const {
    return {tokens_.tokens[range.first].spelling.begin(),
        tokens_.tokens[range.last].spelling.end()};
  }
  const Type *TypeOf(BodyVar var) const { return body_.vars[var].type; }
  std::string Name(BodyVar var) const { return body_.vars[var].name; }
  std::string InClause() const;
  // The variables a goal uses.
  const Vars &Uses(const Body *);
  std::optional<parser::Determinism> DeterminismOf(const Body *goal) const {
    auto it{body_.goals.find(goal)};
    return it == body_.goals.end() ? std::nullopt
                                   : std::make_optional(it->second);
  }
  bool CanFail(const Body *) const;
  bool CanBacktrackInto(const Body *goal, bool failsAfter) const;
  // Makes mostly_unique the variables of `vars` bound before the goal.
  void Mark(const Vars &vars, const Vars &bound) {
    for (BodyVar var : vars) {
      if (bound.count(var)) {
        mostly_.insert(var);
        conditional_.erase(var);
      }
    }
  }
  // Whether a variable is mostly_unique, as what is made from it is.
  bool Shares(BodyVar var) const {
    return mostly_.count(var) && !conditional_.count(var);
  }
  // Checks a goal, after which the variables of `live` are used; with
  // `failsAfter`, a goal after it in the committed choice context it is
  // in, if any, can fail.
  void Walk(const Body *, const Vars &live, bool failsAfter);
  void Leaf(const Body *);
  // That argument i of a call, `var`, is mostly_unique where its mode
  // wants it unique.
  void CheckArgument(const Body *, const ArgMode &, BodyVar var,
      const std::string &where);
  void CheckFinal();
  // Whether `inst`, made mostly_unique, matches `expected`, as in mmc.
  bool Matches(const Inst *inst, const Inst *expected, const Type *);

  const parser::TokenSequence &tokens_;
  ModeChecking &modes_;
  InstContext &insts_;
  InstOps &ops_;
  const ProcedureBody &body_;
  std::map<const Body *, Vars> uses_;
  // The variables bound, and those that are mostly_unique, at the point
  // being checked; whether only its first solution is wanted.
  Vars bound_, mostly_;
  // Those mostly_unique only because the else part of an if-then-else
  // uses them, which mmc does not pass on to what is made from them.
  Vars conditional_;
  bool committed_{false};
  std::optional<std::pair<parser::CharBlock, std::string>> error_;
  // The goal run last, where an error in the final insts is reported, as
  // in mmc.
  parser::CharBlock last_{};
};

std::string UniquenessChecker::InClause() const {
  return "in the clause for " +
      (body_.method ? "the method " + ProcedureToString(*body_.procedure) +
                  " of instance " + body_.method->method->instanceText
                    : ProcedureToString(*body_.procedure));
}

const Vars &UniquenessChecker::Uses(const Body *goal) {
  if (auto it{uses_.find(goal)}; it != uses_.end()) {
    return it->second;
  }
  Vars vars;
  switch (goal->kind) {
  case BodyKind::Unify:
    vars.insert(goal->left);
    if (goal->isVar) {
      vars.insert(goal->var);
    }
    if (goal->lambda) {
      // Its nonlocals, those of its body that are not its head's.
      Vars inner{Uses(goal->lambda)};
      for (BodyVar var : goal->lambdaHead) {
        inner.erase(var);
      }
      vars.insert(inner.begin(), inner.end());
    }
    break;
  case BodyKind::HOCall:
    vars.insert(goal->closure);
    break;
  case BodyKind::Switch:
    vars.insert(goal->switchVar);
    for (const SwitchCase &c : goal->cases) {
      const Vars &inner{Uses(c.goal)};
      vars.insert(inner.begin(), inner.end());
    }
    break;
  default:
    break;
  }
  vars.insert(goal->arguments.begin(), goal->arguments.end());
  vars.insert(goal->failed.begin(), goal->failed.end());
  vars.insert(goal->promised.begin(), goal->promised.end());
  for (const Body *g : goal->goals) {
    const Vars &inner{Uses(g)};
    vars.insert(inner.begin(), inner.end());
  }
  if (goal->kind == BodyKind::Disj && goal->goals.empty()) {
    // A fail that stands for goals left out.
    for (const auto &[var, inst] : goal->entry) {
      vars.insert(var);
    }
  }
  return uses_[goal] = std::move(vars);
}

bool UniquenessChecker::CanFail(const Body *goal) const {
  std::optional<parser::Determinism> d{DeterminismOf(goal)};
  return d &&
      (*d == parser::Determinism::Semidet ||
          *d == parser::Determinism::Nondet ||
          *d == parser::Determinism::Failure ||
          *d == parser::Determinism::CcNondet);
}

// Whether the goal can succeed more than once, so that what follows may be
// run again on backtracking into it. Where only the first solution is
// wanted, that is only when a goal after it there can fail, as mmc's
// determinism analysis gives a goal before one that can fail all its
// solutions.
bool UniquenessChecker::CanBacktrackInto(
    const Body *goal, bool failsAfter) const {
  // Nor is a call of a committed choice procedure, even where all its
  // solutions are wanted, which is an error.
  if (goal->modes &&
      (goal->kind == BodyKind::Call || goal->kind == BodyKind::HOCall)) {
    std::optional<parser::Determinism> callee{goal->kind == BodyKind::HOCall
            ? std::make_optional(goal->modes->determinism)
            : goal->modes->callee ? goal->modes->callee->determinism
                                  : std::nullopt};
    if (callee == parser::Determinism::CcMulti ||
        callee == parser::Determinism::CcNondet) {
      return false;
    }
  }
  std::optional<parser::Determinism> d{DeterminismOf(goal)};
  return d &&
      (*d == parser::Determinism::Multi ||
          *d == parser::Determinism::Nondet) &&
      (!committed_ || failsAfter);
}

void UniquenessChecker::Walk(
    const Body *goal, const Vars &live, bool failsAfter) {
  if (error_) {
    return;
  }
  Vars before{bound_};
  switch (goal->kind) {
  case BodyKind::Conj: {
    // What is used after each goal: the goals after it, and what is used
    // after the conjunction.
    std::vector<Vars> after(goal->goals.size(), live);
    std::vector<bool> fails(goal->goals.size(), failsAfter);
    for (std::size_t i{goal->goals.size()}; i-- > 1;) {
      after[i - 1] = after[i];
      const Vars &uses{Uses(goal->goals[i])};
      after[i - 1].insert(uses.begin(), uses.end());
      fails[i - 1] = fails[i] || CanFail(goal->goals[i]);
    }
    for (std::size_t i{0}; i < goal->goals.size(); ++i) {
      Walk(goal->goals[i], after[i], fails[i]);
    }
    break;
  }
  case BodyKind::Disj:
  case BodyKind::Switch: {
    // mmc leaves out the disjuncts whose end mode analysis does not
    // reach, and one with a single disjunct left is a conjunction.
    std::vector<const Body *> branches;
    for (const Body *g : goal->goals) {
      if (Reached(g)) {
        branches.push_back(g);
      }
    }
    // One that can fail or succeed more than once may run a later branch
    // after an earlier one has used a value. Where only its first solution
    // is wanted, as in one with more than one branch that has at most one
    // solution because its outputs are not used, a branch that cannot
    // fail is the last one run, and only in a branch that can fail are the
    // values it uses, as in mmc.
    std::optional<parser::Determinism> d{DeterminismOf(goal)};
    bool marks{goal->kind == BodyKind::Disj && branches.size() > 1 && d &&
        *d != parser::Determinism::Erroneous};
    bool byBranch{marks &&
        ((committed_ && !failsAfter) || *d == parser::Determinism::Det ||
            *d == parser::Determinism::Semidet ||
            *d == parser::Determinism::CcMulti ||
            *d == parser::Determinism::CcNondet ||
            *d == parser::Determinism::Failure)};
    if (marks && !byBranch) {
      Mark(Uses(goal), before);
    }
    for (const SwitchCase &c : goal->cases) {
      branches.push_back(c.goal);
    }
    // The insts after it are those of its branches merged: a variable is
    // mostly_unique after it if it is so after any.
    Vars mostly{mostly_}, bound{bound_}, conditional{conditional_},
        mergedMostly, mergedBound, shared;
    for (const Body *branch : branches) {
      mostly_ = mostly;
      bound_ = bound;
      conditional_ = conditional;
      if (byBranch && CanFail(branch)) {
        Mark(Uses(branch), before);
      }
      Walk(branch, live, failsAfter);
      mergedMostly.insert(mostly_.begin(), mostly_.end());
      mergedBound.insert(bound_.begin(), bound_.end());
      for (BodyVar var : mostly_) {
        if (Shares(var)) {
          shared.insert(var);
        }
      }
    }
    if (!branches.empty()) {
      mostly_ = std::move(mergedMostly);
      bound_ = std::move(mergedBound);
      conditional_.clear();
      for (BodyVar var : mostly_) {
        if (!shared.count(var)) {
          conditional_.insert(var);
        }
      }
    }
    break;
  }
  case BodyKind::IfThenElse: {
    const Body *condition{goal->goals[0]};
    const Body *then{goal->goals[1]};
    const Body *otherwise{goal->goals[2]};
    if (!Reached(condition)) {
      // mmc's mode analysis makes it its else part.
      Walk(otherwise, live, failsAfter);
      break;
    }
    Vars mostly{mostly_}, conditional{conditional_};
    // The condition's variables that the else part uses are needed if
    // the condition fails after using them.
    const Vars &inElse{Uses(otherwise)};
    Vars both;
    for (BodyVar var : Uses(condition)) {
      if (inElse.count(var)) {
        both.insert(var);
      }
    }
    for (BodyVar var : both) {
      if (before.count(var) && mostly_.insert(var).second) {
        conditional_.insert(var);
      }
    }
    Vars afterCondition{live};
    const Vars &inThen{Uses(then)};
    afterCondition.insert(inThen.begin(), inThen.end());
    Walk(condition, afterCondition, failsAfter || CanFail(then));
    Walk(then, live, failsAfter);
    Vars thenMostly{std::move(mostly_)}, thenBound{std::move(bound_)},
        thenConditional{std::move(conditional_)};
    mostly_ = std::move(mostly);
    bound_ = before;
    conditional_ = std::move(conditional);
    Walk(otherwise, live, failsAfter);
    mostly_.insert(thenMostly.begin(), thenMostly.end());
    bound_.insert(thenBound.begin(), thenBound.end());
    conditional_.insert(thenConditional.begin(), thenConditional.end());
    break;
  }
  case BodyKind::Not: {
    // The negated goal binds nothing outside it, the variables used after
    // the negation are needed when it fails, and only its first solution
    // is wanted. One whose end mode analysis does not reach mmc makes
    // true.
    if (!Reached(goal->goals.front())) {
      break;
    }
    Vars mostly{mostly_}, conditional{conditional_};
    bool committed{committed_};
    Mark(live, before);
    committed_ = true;
    Walk(goal->goals.front(), live, false);
    committed_ = committed;
    mostly_ = std::move(mostly);
    conditional_ = std::move(conditional);
    bound_ = std::move(before);
    break;
  }
  case BodyKind::Scope:
    if (goal->promise) {
      // Only its first solution is wanted.
      bool committed{committed_};
      committed_ = true;
      Walk(goal->goals.front(), live, false);
      committed_ = committed;
    } else {
      Walk(goal->goals.front(), live, failsAfter);
    }
    break;
  default:
    Leaf(goal);
    // A call's outputs may be needed by the callee itself to find its next
    // solution.
    if ((goal->kind == BodyKind::Call || goal->kind == BodyKind::HOCall) &&
        CanBacktrackInto(goal, failsAfter)) {
      const Vars &uses{Uses(goal)};
      mostly_.insert(uses.begin(), uses.end());
    }
    break;
  }
  if (CanBacktrackInto(goal, failsAfter)) {
    Mark(live, before);
  }
}

void UniquenessChecker::Leaf(const Body *goal) {
  last_ = tokens_.tokens[goal->tokens.first].spelling;
  const Vars &uses{Uses(goal)};
  if (goal->kind == BodyKind::Call ||
      (goal->kind == BodyKind::Unify && goal->unify == UnifyKind::Call)) {
    const Procedure *callee{goal->modes ? goal->modes->callee : nullptr};
    if (callee) {
      bool function{goal->kind == BodyKind::Unify};
      std::vector<BodyVar> arguments{goal->arguments};
      if (function) {
        arguments.push_back(goal->left);
      }
      const Symbol &symbol{*callee->symbol};
      std::string what{std::string{function ? "func " : "pred "} +
          symbol.module.str() + '.' + symbol.name.str() + '/' +
          std::to_string(symbol.arity)};
      for (std::size_t i{0};
           i < arguments.size() && i < callee->modes.size() && !error_; ++i) {
        bool result{function && i + 1 == arguments.size()};
        CheckArgument(goal, callee->modes[i], arguments[i],
            (result ? std::string{"the result"}
                    : "argument " + std::to_string(i + 1)) +
                " of the call to " + what);
      }
    }
  } else if (goal->kind == BodyKind::Unify) {
    switch (goal->unify) {
    case UnifyKind::Assign:
      if (Shares(goal->var)) {
        mostly_.insert(goal->left);
      }
      break;
    case UnifyKind::Construct:
      if (llvm::any_of(goal->arguments,
              [&](BodyVar var) { return Shares(var); })) {
        mostly_.insert(goal->left);
      }
      break;
    case UnifyKind::Deconstruct:
      if (Shares(goal->left)) {
        mostly_.insert(goal->arguments.begin(), goal->arguments.end());
      }
      break;
    default:
      break;
    }
  }
  // A goal binds those of its variables that were free.
  bound_.insert(uses.begin(), uses.end());
}

bool UniquenessChecker::Matches(
    const Inst *inst, const Inst *expected, const Type *type) {
  // mmc does not match a mostly_unique bound inst with a part bound too,
  // as mostly_unique(t(mostly_unique(3))), to ground, though one with a
  // constant or a ground part it does (see the design document's
  // register).
  while (expected->kind == InstKind::Defined) {
    expected = insts_.Expand(expected);
  }
  if (inst->kind == InstKind::Bound &&
      inst->uniqueness == Uniqueness::MostlyUnique &&
      expected->kind == InstKind::Ground &&
      llvm::any_of(inst->functors, [](const BoundAlternative &alternative) {
        return llvm::any_of(alternative.arguments, [](const Inst *argument) {
          return argument->kind == InstKind::Bound;
        });
      })) {
    return false;
  }
  return ops_.Matches(inst, expected, type);
}

void UniquenessChecker::CheckArgument(const Body *goal, const ArgMode &mode,
    BodyVar var, const std::string &where) {
  if (!mostly_.count(var)) {
    return;
  }
  auto it{llvm::find_if(
      goal->entry, [&](const auto &entry) { return entry.first == var; })};
  if (it == goal->entry.end() || !it->second) {
    return;
  }
  const Inst *inst{insts_.Share(it->second, Uniqueness::MostlyUnique)};
  if (inst == it->second || Matches(inst, mode.initial, TypeOf(var))) {
    return;
  }
  // mmc: "in argument 2 of call to predicate `m.p'/3: mode error: variable
  // `T0' has instantiatedness `mostly_unique', expected instantiatedness
  // was `unique'", as in its first phase.
  error_ = {Source(goal->tokens),
      InClause() + ": mode error in " + where + ": " + Name(var) +
          " has inst " + InstToString(*inst) + ", but " +
          InstToString(*mode.initial) + " is expected"};
}

void UniquenessChecker::CheckFinal() {
  const Procedure &procedure{*body_.procedure};
  // An inferred procedure's final insts are those its clauses give.
  if (procedure.inferred) {
    return;
  }
  llvm::ArrayRef<ClauseModes> clauses{body_.method
          ? llvm::ArrayRef<ClauseModes>{body_.method->clauses}
          : modes_.ClausesOf(procedure)};
  if (clauses.empty()) {
    return;
  }
  for (std::size_t i{0}; i < body_.head.size() && i < procedure.modes.size() &&
       i < clauses.front().final.size();
       ++i) {
    BodyVar var{body_.head[i]};
    if (!mostly_.count(var)) {
      continue;
    }
    const Inst *final{clauses.front().final[i]};
    const Inst *inst{insts_.Share(final, Uniqueness::MostlyUnique)};
    const Inst *expected{procedure.modes[i].final};
    if (inst == final || Matches(inst, expected, TypeOf(var))) {
      continue;
    }
    // mmc: "argument 2 did not get sufficiently instantiated. Final
    // instantiatedness of `T' was `mostly_unique', expected final
    // instantiatedness was `unique'".
    const char *how{ops_.Matches(expected, inst, TypeOf(var))
            ? "did not get sufficiently instantiated"
            : "had the wrong inst"};
    error_ = {last_,
        InClause() + ": mode error: argument " + std::to_string(i + 1) +
            ", " + Name(var) + ", " + how + ": its final inst is " +
            InstToString(*inst) + ", but " + InstToString(*expected) +
            " is expected"};
    return;
  }
}

std::optional<std::pair<parser::CharBlock, std::string>>
UniquenessChecker::Check() {
  const Procedure &procedure{*body_.procedure};
  for (std::size_t i{0}; i < body_.head.size() && i < procedure.modes.size();
       ++i) {
    if (!ops_.IsFree(procedure.modes[i].initial)) {
      bound_.insert(body_.head[i]);
    }
  }
  // The head's variables are used after the body, but for those that their
  // modes clobber.
  Vars live;
  for (std::size_t i{0}; i < body_.head.size() && i < procedure.modes.size();
       ++i) {
    const Inst *final{procedure.modes[i].final};
    while (final->kind == InstKind::Defined) {
      final = insts_.Expand(final);
    }
    if (!((final->kind == InstKind::Ground || final->kind == InstKind::Any) &&
            (final->uniqueness == Uniqueness::Clobbered ||
                final->uniqueness == Uniqueness::MostlyClobbered))) {
      live.insert(body_.head[i]);
    }
  }
  // A committed choice procedure wants only its first solution.
  std::optional<parser::Determinism> d{
      procedure.determinism ? procedure.determinism : body_.determinism};
  committed_ = d &&
      (*d == parser::Determinism::CcMulti ||
          *d == parser::Determinism::CcNondet);
  Walk(body_.body, live, false);
  if (!error_) {
    CheckFinal();
  }
  return error_;
}

} // namespace

void CheckUniqueness(const parser::TokenSequence &tokens, ModeChecking &modes,
    const DeterminismChecking &determinism, parser::Messages &messages) {
  for (const ProcedureBody &body : determinism.bodies()) {
    // Only bodies whose determinisms were inferred, so that it is known
    // where backtracking can happen.
    if (!body.body || !body.determinism) {
      continue;
    }
    UniquenessChecker checker{tokens, modes, body};
    if (auto error{checker.Check()}) {
      messages.Say(error->first, "%s"_err_en_US, error->second);
    }
  }
}

} // namespace merkurius::semantics
