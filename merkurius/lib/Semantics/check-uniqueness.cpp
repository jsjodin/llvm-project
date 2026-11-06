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

// The parts of a value that are mostly_unique: all of it, or those of the
// arguments of its functor, by position, that are, so that a term made
// from a mostly_unique value and a unique one is mostly_unique only in
// the first's part, as in mmc.
struct Parts {
  bool whole{false};
  std::map<std::size_t, Parts> arguments;

  bool Empty() const { return !whole && arguments.empty(); }
  void Merge(const Parts &that) {
    if (whole) {
      return;
    }
    if (that.whole) {
      whole = true;
      arguments.clear();
      return;
    }
    for (const auto &[i, argument] : that.arguments) {
      arguments[i].Merge(argument);
    }
  }
  bool operator==(const Parts &that) const {
    return whole == that.whole && arguments == that.arguments;
  }
};

// The variables with mostly_unique parts.
using Marks = std::map<BodyVar, Parts>;

void Merge(Marks &marks, const Marks &more) {
  for (const auto &[var, mark] : more) {
    marks[var].Merge(mark);
  }
}

class UniquenessChecker {
public:
  UniquenessChecker(const parser::TokenSequence &tokens, ModeChecking &modes,
      const ProcedureBody &body, bool mmc)
      : tokens_{tokens}, modes_{modes}, insts_{modes.insts()},
        ops_{modes.ops()}, body_{body}, mmc_{mmc} {}

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
        MarkWhole(var);
      }
    }
  }
  void MarkWhole(BodyVar var) { mostly_[var] = Parts{true, {}}; }
  // The parts of a variable's value that are mostly_unique, if any.
  const Parts *MarkOf(BodyVar var) const {
    auto it{mostly_.find(var)};
    return it == mostly_.end() || it->second.Empty() ? nullptr : &it->second;
  }
  // `inst` with the parts of `mark` made mostly_unique.
  const Inst *Apply(const Inst *inst, const Parts &mark);
  // Whether a goal clobbers one of `vars`, passing it where its final
  // inst is clobbered or mostly_clobbered.
  bool Clobbers(const Body *goal, const Vars &vars);
  bool ClobbersAny(llvm::ArrayRef<ArgMode> modes,
      llvm::ArrayRef<BodyVar> arguments, const Vars &vars) const;
  // `var`, and the variables that the unifications of `goal` make from it
  // or it from, whatever their order.
  Vars MadeFrom(const Body *goal, BodyVar var);
  // Checks a goal, after which the variables of `live` are used; with
  // `failsAfter`, a goal after it in the committed choice context it is
  // in, if any, can fail.
  void Walk(const Body *, const Vars &live, bool failsAfter);
  void Lambda(const Body *);
  void Closure(const Body *);
  void Leaf(const Body *);
  // That argument i of a call, `var`, is mostly_unique where its mode
  // wants it unique.
  void CheckArgument(const Body *, const ArgMode &, BodyVar var,
      const std::string &where);
  // Whether a final inst clobbers the value, so that it is not used after.
  bool Clobbers(const Inst *final) const;
  // That the head variables `head`, with the final insts `finals`, made
  // mostly_unique where they are, match their modes' final insts; those
  // of a lambda expression are named as in mmc.
  void CheckFinal(llvm::ArrayRef<BodyVar> head, llvm::ArrayRef<ArgMode> modes,
      llvm::ArrayRef<const Inst *> finals, bool lambda);
  // Whether `inst`, made mostly_unique, matches `expected`, as in mmc;
  // `var` is the variable whose inst it is, if any.
  bool Matches(const Inst *inst, const Inst *expected, const Type *,
      std::optional<BodyVar> var = std::nullopt);

  const parser::TokenSequence &tokens_;
  ModeChecking &modes_;
  InstContext &insts_;
  InstOps &ops_;
  const ProcedureBody &body_;
  // Whether to miss what mmc misses.
  bool mmc_;
  std::map<const Body *, Vars> uses_;
  // The variables bound, and those that are mostly_unique, at the point
  // being checked; whether only its first solution is wanted.
  Vars bound_;
  Marks mostly_;
  bool committed_{false};
  // The variables that were arguments of calls, whose insts mmc has
  // found ground then.
  Vars called_;
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
  if (!committed_ && goal->kind != BodyKind::Not && body_.unseen.count(goal)) {
    // Only the first solution of a goal with no outputs is wanted, as mmc
    // infers it so, whatever its determinism, and backtracking into its
    // parts does not run what follows it again, as for a determinism cast.
    committed_ = true;
    Walk(goal, {}, false);
    committed_ = false;
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
    Marks mostly{mostly_}, mergedMostly;
    Vars bound{bound_}, mergedBound;
    for (const Body *branch : branches) {
      mostly_ = mostly;
      bound_ = bound;
      if (byBranch && CanFail(branch)) {
        Mark(Uses(branch), before);
      }
      Walk(branch, live, failsAfter);
      Merge(mergedMostly, mostly_);
      mergedBound.insert(bound_.begin(), bound_.end());
    }
    if (!branches.empty()) {
      mostly_ = std::move(mergedMostly);
      bound_ = std::move(mergedBound);
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
      return;
    }
    Marks mostly{mostly_};
    Vars bound{bound_};
    // A value bound before that the else part uses is needed while the
    // condition runs, if the condition clobbers it, or what it makes from
    // it, by an assignment, a construction or a deconstruction; mmc looks
    // only for the variable itself (see the design document's register).
    // Once the condition has succeeded, the else part is not run, so it
    // is no longer needed; mmc has it so in the then part too.
    const Vars &inElse{Uses(otherwise)};
    for (BodyVar var : Uses(condition)) {
      if (inElse.count(var) && before.count(var) &&
          Clobbers(condition, mmc_ ? Vars{var} : MadeFrom(condition, var))) {
        MarkWhole(var);
      }
    }
    // What is used after the if-then-else is used after its parts, as
    // backtracking into one runs it again; mmc takes as used after the
    // condition only what the then part uses, and after the then and else
    // parts nothing (see the design document's register).
    Vars afterCondition{mmc_ ? Vars{} : live}, afterParts{afterCondition};
    const Vars &inThen{Uses(then)};
    afterCondition.insert(inThen.begin(), inThen.end());
    Walk(condition, afterCondition, failsAfter || CanFail(then));
    if (!mmc_ && !error_ && mostly_ != mostly) {
      // Again, without them, for what the then part starts from.
      mostly_ = mostly;
      bound_ = std::move(bound);
      Walk(condition, afterCondition, failsAfter || CanFail(then));
    }
    Walk(then, afterParts, failsAfter);
    Marks thenMostly{std::move(mostly_)};
    Vars thenBound{std::move(bound_)};
    mostly_ = std::move(mostly);
    bound_ = before;
    Walk(otherwise, afterParts, failsAfter);
    Merge(mostly_, thenMostly);
    bound_.insert(thenBound.begin(), thenBound.end());
    if (mmc_) {
      return;
    }
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
    Marks mostly{mostly_};
    bool committed{committed_};
    Mark(live, before);
    committed_ = true;
    Walk(goal->goals.front(), live, false);
    committed_ = committed;
    mostly_ = std::move(mostly);
    bound_ = std::move(before);
    break;
  }
  case BodyKind::Scope:
    if (goal->promise &&
        *goal->promise !=
            parser::PromiseSolutionsKind::EquivalentSolutionSets) {
      // Only its first solution is wanted, and backtracking into its goal
      // does not run what follows it again.
      bool committed{committed_};
      committed_ = true;
      Walk(goal->goals.front(), {}, false);
      committed_ = committed;
    } else {
      Walk(goal->goals.front(), live, failsAfter);
    }
    break;
  default:
    if (goal->lambda) {
      Lambda(goal);
    } else if (goal->kind == BodyKind::Unify &&
        goal->unify == UnifyKind::Construct && goal->modes &&
        goal->modes->callee) {
      Closure(goal);
    }
    Leaf(goal);
    // A call's outputs may be needed by the callee itself to find its next
    // solution.
    if ((goal->kind == BodyKind::Call || goal->kind == BodyKind::HOCall) &&
        CanBacktrackInto(goal, failsAfter)) {
      for (BodyVar var : Uses(goal)) {
        MarkWhole(var);
      }
    }
    break;
  }
  if (CanBacktrackInto(goal, failsAfter)) {
    Mark(live, before);
  }
}

// A lambda expression's body, which mmc checks where the closure is made,
// as a procedure of its own: its head variables are bound as their modes
// have them, those its modes do not clobber are used after it, and a
// committed choice one wants only its first solution.
void UniquenessChecker::Lambda(const Body *goal) {
  if (!goal->modes) {
    return;
  }
  llvm::ArrayRef<ArgMode> modes{goal->modes->lambda};
  Vars bound{bound_}, live;
  Marks mostly{mostly_};
  bool committed{committed_};
  parser::CharBlock last{last_};
  for (std::size_t i{0}; i < goal->lambdaHead.size() && i < modes.size();
       ++i) {
    if (!ops_.IsFree(modes[i].initial)) {
      bound_.insert(goal->lambdaHead[i]);
    }
    if (!Clobbers(modes[i].final)) {
      live.insert(goal->lambdaHead[i]);
    }
  }
  parser::Determinism d{goal->modes->determinism};
  committed_ = d == parser::Determinism::CcMulti ||
      d == parser::Determinism::CcNondet;
  Walk(goal->lambda, live, false);
  if (!error_) {
    // Mode analysis has checked that its final insts match its modes'.
    std::vector<const Inst *> finals;
    for (const ArgMode &mode : modes) {
      finals.push_back(mode.final);
    }
    CheckFinal(goal->lambdaHead, modes, finals, true);
  }
  bound_ = std::move(bound);
  mostly_ = std::move(mostly);
  committed_ = committed;
  last_ = last;
}

// A closure made from a predicate's or function's name, which mmc makes
// a lambda expression whose body calls it: if the callee can succeed more
// than once, its outputs are needed on backtracking into the call, and so
// are mostly_unique at the end.
void UniquenessChecker::Closure(const Body *goal) {
  const Procedure &callee{*goal->modes->callee};
  if (callee.determinism != parser::Determinism::Multi &&
      callee.determinism != parser::Determinism::Nondet) {
    return;
  }
  const Type *type{TypeOf(goal->left)};
  std::size_t curried{goal->arguments.size()};
  for (std::size_t i{curried}; i < callee.modes.size() && !error_; ++i) {
    const Inst *final{callee.modes[i].final};
    const Inst *inst{insts_.Share(final, Uniqueness::MostlyUnique)};
    std::size_t k{i - curried};
    const Type *argument{nullptr};
    if (type->kind == TypeKind::HigherOrder) {
      argument =
          k < type->arguments.size() ? type->arguments[k] : type->result;
    }
    if (inst == final || !argument || Matches(inst, final, argument)) {
      continue;
    }
    // mmc: "argument 2 did not get sufficiently instantiated. Final
    // instantiatedness of `V_9' was `mostly_unique', expected final
    // instantiatedness was `unique'", of its lambda expression.
    const char *how{ops_.Matches(final, inst, argument)
            ? "did not get sufficiently instantiated"
            : "had the wrong inst"};
    error_ = {Source(goal->tokens),
        InClause() + ": mode error: argument " + std::to_string(k + 1) +
            " of the closure of " + ProcedureToString(callee) + " " + how +
            ": its final inst is " + InstToString(*inst) + ", but " +
            InstToString(*final) + " is expected"};
  }
}

// Whether a call clobbers one of `vars` among its arguments, whose modes are
// `modes`.
bool UniquenessChecker::ClobbersAny(llvm::ArrayRef<ArgMode> modes,
    llvm::ArrayRef<BodyVar> arguments, const Vars &vars) const {
  for (std::size_t i{0}; i < modes.size() && i < arguments.size(); ++i) {
    if (vars.count(arguments[i]) && Clobbers(modes[i].final)) {
      return true;
    }
  }
  return false;
}

bool UniquenessChecker::Clobbers(const Body *goal, const Vars &vars) {
  if (goal->modes && goal->kind == BodyKind::HOCall &&
      ClobbersAny(goal->modes->lambda, goal->arguments, vars)) {
    return true;
  }
  if (goal->modes && goal->modes->callee && !goal->lambda &&
      (goal->kind == BodyKind::Call ||
          (goal->kind == BodyKind::Unify &&
              goal->unify == UnifyKind::Call))) {
    std::vector<BodyVar> arguments{goal->arguments};
    if (goal->kind == BodyKind::Unify) {
      arguments.push_back(goal->left);
    }
    if (ClobbersAny(goal->modes->callee->modes, arguments, vars)) {
      return true;
    }
  }
  for (const SwitchCase &c : goal->cases) {
    if (Clobbers(c.goal, vars)) {
      return true;
    }
  }
  return llvm::any_of(
      goal->goals, [&](const Body *g) { return Clobbers(g, vars); });
}

// The variables of each assignment, construction and deconstruction in a
// goal, one list for each.
void CollectLinks(const Body *goal, std::vector<std::vector<BodyVar>> &links) {
  if (goal->kind == BodyKind::Unify && !goal->lambda &&
      (goal->unify == UnifyKind::Assign ||
          goal->unify == UnifyKind::Construct ||
          goal->unify == UnifyKind::Deconstruct)) {
    std::vector<BodyVar> link{goal->arguments};
    link.push_back(goal->left);
    if (goal->isVar) {
      link.push_back(goal->var);
    }
    links.push_back(std::move(link));
  }
  for (const SwitchCase &c : goal->cases) {
    CollectLinks(c.goal, links);
  }
  for (const Body *g : goal->goals) {
    CollectLinks(g, links);
  }
}

Vars UniquenessChecker::MadeFrom(const Body *goal, BodyVar var) {
  // The unifications' variables, joined until nothing changes.
  std::vector<std::vector<BodyVar>> links;
  CollectLinks(goal, links);
  Vars family{var};
  for (bool changed{true}; changed;) {
    changed = false;
    for (const std::vector<BodyVar> &link : links) {
      if (llvm::any_of(link, [&](BodyVar v) { return family.count(v); })) {
        for (BodyVar v : link) {
          changed |= family.insert(v).second;
        }
      }
    }
  }
  return family;
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
  } else if (goal->kind == BodyKind::HOCall && goal->modes) {
    llvm::ArrayRef<ArgMode> modes{goal->modes->lambda};
    bool function{goal->hoCall->kind == parser::PredOrFunc::Function};
    const char *kind{function ? "function" : "predicate"};
    for (std::size_t i{0}; i < goal->arguments.size() && i < modes.size() &&
         !error_;
         ++i) {
      bool result{function && i + 1 == goal->arguments.size()};
      CheckArgument(goal, modes[i], goal->arguments[i],
          "argument " + std::to_string(i + 2) + " (" +
              (result ? std::string{"the result of the called function"}
                      : "argument " + std::to_string(i + 1) +
                          " of the called " + kind) +
              ") of the higher-order " + kind + " call");
    }
  } else if (goal->kind == BodyKind::Unify) {
    switch (goal->unify) {
    // What is made from a mostly_unique value, or taken from one, is
    // mostly_unique where it is.
    case UnifyKind::Assign:
      if (const Parts *mark{MarkOf(goal->var)}) {
        Parts copy{*mark};
        mostly_[goal->left].Merge(copy);
      }
      break;
    case UnifyKind::Construct: {
      Parts term;
      for (std::size_t i{0}; i < goal->arguments.size(); ++i) {
        if (const Parts *mark{MarkOf(goal->arguments[i])}) {
          term.arguments[i] = *mark;
        }
      }
      if (!term.Empty()) {
        mostly_[goal->left].Merge(term);
      }
      break;
    }
    case UnifyKind::Deconstruct:
      if (const Parts *mark{MarkOf(goal->left)}) {
        Parts term{*mark};
        for (std::size_t i{0}; i < goal->arguments.size(); ++i) {
          if (term.whole) {
            MarkWhole(goal->arguments[i]);
          } else if (auto it{term.arguments.find(i)};
                     it != term.arguments.end()) {
            mostly_[goal->arguments[i]].Merge(it->second);
          }
        }
      }
      break;
    default:
      break;
    }
  }
  if (goal->kind == BodyKind::Call || goal->kind == BodyKind::HOCall ||
      (goal->kind == BodyKind::Unify && goal->unify == UnifyKind::Call)) {
    called_.insert(uses.begin(), uses.end());
  }
  // A goal binds those of its variables that were free.
  bound_.insert(uses.begin(), uses.end());
}

bool UniquenessChecker::Matches(const Inst *inst, const Inst *expected,
    const Type *type, std::optional<BodyVar> var) {
  // mmc does not match a mostly_unique bound inst with a part bound too,
  // as mostly_unique(t(mostly_unique(3))), to mostly_unique, though one
  // with a constant or a ground part it does, and one of a variable that
  // was an argument of a call before (see the design document's
  // register).
  while (expected->kind == InstKind::Defined) {
    expected = insts_.Expand(expected);
  }
  if (inst->kind == InstKind::Bound &&
      inst->uniqueness == Uniqueness::MostlyUnique &&
      expected->kind == InstKind::Ground &&
      expected->uniqueness == Uniqueness::MostlyUnique &&
      !(var && called_.count(*var)) &&
      llvm::any_of(inst->functors, [](const BoundAlternative &alternative) {
        return llvm::any_of(alternative.arguments, [](const Inst *argument) {
          return argument->kind == InstKind::Bound;
        });
      })) {
    return false;
  }
  return ops_.Matches(inst, expected, type);
}

const Inst *UniquenessChecker::Apply(const Inst *inst, const Parts &mark) {
  if (mark.whole) {
    return insts_.Share(inst, Uniqueness::MostlyUnique);
  }
  const Inst *body{inst};
  while (body->kind == InstKind::Defined) {
    body = insts_.Expand(body);
  }
  if (body->kind != InstKind::Bound) {
    // Parts of a value whose inst has none: all of it.
    return insts_.Share(inst, Uniqueness::MostlyUnique);
  }
  std::vector<BoundAlternative> alternatives;
  std::vector<std::vector<const Inst *>> arguments;
  for (const BoundAlternative &alternative : body->functors) {
    arguments.emplace_back(
        alternative.arguments.begin(), alternative.arguments.end());
    for (const auto &[i, part] : mark.arguments) {
      if (i < arguments.back().size()) {
        arguments.back()[i] = Apply(arguments.back()[i], part);
      }
    }
  }
  bool changed{false};
  for (std::size_t j{0}; j < body->functors.size(); ++j) {
    changed |= !llvm::equal(arguments[j], body->functors[j].arguments);
    alternatives.push_back({body->functors[j].functor, arguments[j]});
  }
  return changed ? insts_.Bound(body->uniqueness, alternatives) : inst;
}

void UniquenessChecker::CheckArgument(const Body *goal, const ArgMode &mode,
    BodyVar var, const std::string &where) {
  const Parts *mark{MarkOf(var)};
  if (!mark) {
    return;
  }
  auto it{llvm::find_if(
      goal->entry, [&](const auto &entry) { return entry.first == var; })};
  if (it == goal->entry.end() || !it->second) {
    return;
  }
  const Inst *inst{Apply(it->second, *mark)};
  if (inst == it->second || Matches(inst, mode.initial, TypeOf(var), var)) {
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

bool UniquenessChecker::Clobbers(const Inst *final) const {
  while (final->kind == InstKind::Defined) {
    final = insts_.Expand(final);
  }
  return (final->kind == InstKind::Ground || final->kind == InstKind::Any) &&
      (final->uniqueness == Uniqueness::Clobbered ||
          final->uniqueness == Uniqueness::MostlyClobbered);
}

void UniquenessChecker::CheckFinal(llvm::ArrayRef<BodyVar> head,
    llvm::ArrayRef<ArgMode> modes, llvm::ArrayRef<const Inst *> finals,
    bool lambda) {
  for (std::size_t i{0};
       i < head.size() && i < modes.size() && i < finals.size(); ++i) {
    BodyVar var{head[i]};
    const Parts *mark{MarkOf(var)};
    if (!mark) {
      continue;
    }
    const Inst *final{finals[i]};
    const Inst *inst{Apply(final, *mark)};
    const Inst *expected{modes[i].final};
    if (inst == final || Matches(inst, expected, TypeOf(var), var)) {
      continue;
    }
    // mmc: "argument 2 did not get sufficiently instantiated. Final
    // instantiatedness of `T' was `mostly_unique', expected final
    // instantiatedness was `unique'".
    const char *how{ops_.Matches(expected, inst, TypeOf(var))
            ? "did not get sufficiently instantiated"
            : "had the wrong inst"};
    std::string name{lambda ? "LambdaHeadVar__" + std::to_string(i + 1)
                            : Name(var)};
    error_ = {last_,
        InClause() + ": mode error: argument " + std::to_string(i + 1) +
            ", " + name + ", " + how + ": its final inst is " +
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
    if (!Clobbers(procedure.modes[i].final)) {
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
  // An inferred procedure's final insts are those its clauses give.
  llvm::ArrayRef<ClauseModes> clauses{body_.method
          ? llvm::ArrayRef<ClauseModes>{body_.method->clauses}
          : modes_.ClausesOf(procedure)};
  if (!error_ && !procedure.inferred && !clauses.empty()) {
    CheckFinal(body_.head, procedure.modes, clauses.front().final, false);
  }
  return error_;
}

} // namespace

void CheckUniqueness(const parser::TokenSequence &tokens, ModeChecking &modes,
    const DeterminismChecking &determinism, parser::Messages &messages,
    bool mmc) {
  for (const ProcedureBody &body : determinism.bodies()) {
    // Only bodies whose determinisms were inferred, so that it is known
    // where backtracking can happen.
    if (!body.body || !body.determinism) {
      continue;
    }
    UniquenessChecker checker{tokens, modes, body, mmc};
    if (auto error{checker.Check()}) {
      messages.Say(error->first, "%s"_err_en_US, error->second);
    }
  }
}

} // namespace merkurius::semantics
