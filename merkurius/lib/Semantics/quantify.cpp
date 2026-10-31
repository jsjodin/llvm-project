//===-- lib/Semantics/quantify.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Implicit quantification (design document, "Semantic form", step 7, and
// the reference manual, "Implicit quantification").
//
// A goal's nonlocal variables are those that occur both in it and outside
// it, where "outside" leaves out the goals parallel to it: the other
// disjuncts of a disjunction, the else part for the condition and then
// part of an if-then-else and the other way round, and the other branches
// of a try or an atomic goal. A variable that occurs in a goal and not
// outside it is local to it, and where the goal is one of those parallel
// ones, or a negation or a lambda expression's body, the variable's
// occurrences there are a variable of their own. Desugaring gives each name
// one variable, so quantification renames apart those occurrences: the
// first goal to which a variable is local keeps it, and each later one
// parallel to it gets a copy.
//
// The goals of two lambda expressions are parallel too, so whether a
// lambda expression captures a variable is decided from the occurrences
// outside lambda expressions only; a goal beside the lambda expression then
// sees the variables it captures as occurrences in it.
//
// The variables that some quantifies are new ones, apart from any of the
// same name outside the scope; at the top of an if-then-else's condition,
// some quantifies over the condition and the then part (manual, "Goals").
//
// The work is done in three passes over the clause: the variables that
// occur in each goal, bottom-up; the variables each lambda expression
// captures, top-down, which gives each goal's free variables; and the
// renaming, top-down, which rebuilds the goals and records their nonlocal
// variables.

#include "merkurius/Semantics/quantify.h"
#include "merkurius/Common/idioms.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include <vector>

namespace merkurius::semantics {
namespace {

using namespace parser::literals;
using Set = llvm::BitVector;

bool Has(const Set &set, VarId var) {
  return var < set.size() && set.test(var);
}
void Add(Set &set, VarId var) {
  if (var >= set.size()) {
    set.resize(var + 1);
  }
  set.set(var);
}
void Add(Set &set, llvm::ArrayRef<VarId> vars) {
  for (VarId var : vars) {
    Add(set, var);
  }
}

unsigned GoalNumber(const Goal &goal) {
  return std::visit([](const auto *x) { return x->number; }, goal);
}

// The quantifier at the top of an if-then-else's condition, if any.
const Scope *ConditionScope(const IfThenElse &ite) {
  const auto *scope{std::get_if<const Scope *>(&ite.condition)};
  return scope && (*scope)->kind == ScopeKind::Some ? *scope : nullptr;
}

class Quantifier {
public:
  Quantifier(const ClauseForm &clause, const parser::TokenSequence &tokens,
      SemanticArena &arena, parser::Messages &messages)
      : clause_{clause}, tokens_{tokens}, arena_{arena}, messages_{messages},
        vars_{clause.vars.begin(), clause.vars.end()}, all_(clause.goals),
        direct_(clause.goals), free_(clause.goals),
        nonlocals_(clause.goals) {
    for (const TypeQualification &qualification : clause.typeQualifications) {
      qualified_.push_back(qualification.var);
    }
  }

  const ClauseForm &Run();

private:
  template <typename T, typename... A> const T *Make(A &&...args) {
    return arena_.Make<T>(std::forward<A>(args)...);
  }
  template <typename T> llvm::ArrayRef<T> Save(llvm::ArrayRef<T> array) {
    return arena_.SaveArray(array);
  }
  parser::CharBlock Source(TokenRange range) const {
    return {tokens_.tokens[range.first].spelling.begin(),
        tokens_.tokens[range.last].spelling.end()};
  }

  // Pass 1: all_ holds the variables that occur in each goal, and direct_
  // those that occur outside its lambda expressions' bodies; both leave
  // out the variables of an inner some, which are not the same variables.
  void Collect(const Goal &);
  void CollectInto(const Goal &goal, Set &all, Set &direct) {
    Collect(goal);
    all |= all_[GoalNumber(goal)];
    direct |= direct_[GoalNumber(goal)];
  }
  // Pass 2: free_ holds each goal's free variables, those that occur in it
  // outside its lambda expressions' bodies or that the lambda expressions
  // capture. `outside` holds the variables that occur outside the goal and
  // outside lambda expressions.
  void Capture(const Goal &, const Set &outside);
  // Pass 3: rebuilds the goal with its variables renamed, and records its
  // nonlocal variables. `outside` holds the variables, renamed, that occur
  // outside it.
  Goal Rebuild(const Goal &, const Set &outside);

  // The outside of each conjunct of `goals`, from `table`.
  std::vector<Set> ConjunctOutsides(llvm::ArrayRef<Goal> goals,
      const Set &outside, const std::vector<Set> &table, bool map);

  // The renaming in effect, from the clause's variables to their copies.
  VarId MapVar(VarId var) const {
    auto it{map_.find(var)};
    return it == map_.end() ? var : it->second;
  }
  Set Map(const Set &set) const {
    Set result{set};
    for (const auto &[from, to] : map_) {
      if (Has(set, from)) {
        result.reset(from);
        Add(result, to);
      }
    }
    return result;
  }
  llvm::SmallVector<VarId, 4> MapVars(llvm::ArrayRef<VarId> vars) const {
    llvm::SmallVector<VarId, 4> result;
    for (VarId var : vars) {
      result.push_back(MapVar(var));
    }
    return result;
  }
  llvm::ArrayRef<VarId> SaveMapped(llvm::ArrayRef<VarId> vars) {
    return Save(llvm::ArrayRef<VarId>{MapVars(vars)});
  }
  // A new variable as `original` was; renames `var` in the scope.
  void Rename(VarId var) {
    VarId current{MapVar(var)};
    vars_.push_back(vars_[current]);
    map_[var] = vars_.size() - 1;
  }

  // A goal to which the variables of `occurring` that are not `outside`
  // are local: each keeps its variable if no goal parallel to this one has
  // taken it, and is renamed otherwise. The renaming lasts until End.
  struct Local {
    llvm::DenseMap<VarId, VarId> map;
    unsigned id;
  };
  Local Begin(const Set &occurring, const Set &outside);
  void Claim(const Set &occurring, const Set &outside, unsigned id);
  void End(Local &&local) {
    map_ = std::move(local.map);
    active_.erase(local.id);
  }
  // some Vars quantifies new variables: renames each that is outside,
  // which mmc warns about.
  void Quantified(llvm::ArrayRef<VarId> vars, const Set &outside,
      TokenRange range) {
    for (VarId var : vars) {
      if (Has(outside, MapVar(var))) {
        Overlapping(MapVar(var), range);
        Rename(var);
      }
    }
  }
  void Overlapping(VarId var, TokenRange range) {
    // mmc: "warning: variable `X' has overlapping scopes".
    messages_.Say(Source(range), "variable %s has overlapping scopes"_warn_en_US,
        vars_[var].name);
  }
  void Record(const Goal &goal, const Set &outside) {
    unsigned number{GoalNumber(goal)};
    Set nonlocals{Map(free_[number])};
    nonlocals &= outside;
    llvm::SmallVector<VarId, 8> vars;
    for (unsigned var : nonlocals.set_bits()) {
      vars.push_back(var);
    }
    nonlocals_[number] = Save(llvm::ArrayRef<VarId>{vars});
  }
  // The type qualifications in the goal's tokens follow its renaming.
  void Qualifications(TokenRange range) {
    for (std::size_t j{0}; j < qualified_.size(); ++j) {
      const TypeQualification &q{clause_.typeQualifications[j]};
      if (q.tokens.first >= range.first && q.tokens.last <= range.last) {
        qualified_[j] = MapVar(q.var);
      }
    }
  }

  const ClauseForm &clause_;
  const parser::TokenSequence &tokens_;
  SemanticArena &arena_;
  parser::Messages &messages_;
  std::vector<VarInfo> vars_;
  std::vector<Set> all_, direct_, free_;
  llvm::DenseMap<unsigned, Set> captured_; // by the lambda's unification
  std::vector<llvm::ArrayRef<VarId>> nonlocals_;
  std::vector<VarId> qualified_;
  llvm::DenseMap<VarId, VarId> map_;
  // The goal each variable is local to, and the goals being rebuilt.
  llvm::DenseMap<VarId, unsigned> owner_;
  llvm::DenseSet<unsigned> active_;
  unsigned locals_{0};
};

void Quantifier::Collect(const Goal &goal) {
  Set all, direct;
  auto both{[&](llvm::ArrayRef<VarId> vars) {
    Add(all, vars);
    Add(direct, vars);
  }};
  std::visit(
      common::visitors{
          [&](const Conj *conj) {
            for (const Goal &g : conj->goals) {
              CollectInto(g, all, direct);
            }
          },
          [&](const Disj *disj) {
            for (const Goal &g : disj->goals) {
              CollectInto(g, all, direct);
            }
          },
          [&](const IfThenElse *ite) {
            if (const Scope * scope{ConditionScope(*ite)}) {
              Collect(scope->goal);
              all_[scope->number] = all_[GoalNumber(scope->goal)];
              direct_[scope->number] = direct_[GoalNumber(scope->goal)];
              Set pairAll{all_[scope->number]}, pairDirect{direct_[scope->number]};
              CollectInto(ite->then, pairAll, pairDirect);
              for (VarId var : scope->vars) {
                if (Has(pairAll, var)) {
                  pairAll.reset(var);
                }
                if (Has(pairDirect, var)) {
                  pairDirect.reset(var);
                }
              }
              all |= pairAll;
              direct |= pairDirect;
            } else {
              CollectInto(ite->condition, all, direct);
              CollectInto(ite->then, all, direct);
            }
            CollectInto(ite->otherwise, all, direct);
          },
          [&](const Not *negation) { CollectInto(negation->goal, all, direct); },
          [&](const Unify *unify) {
            both(unify->left);
            if (unify->isVar) {
              both(unify->var);
            } else {
              both(unify->arguments);
            }
            if (const Lambda * lambda{unify->functor.lambda}) {
              Collect(lambda->body);
              all |= all_[GoalNumber(lambda->body)];
              Add(all, lambda->arguments);
            }
          },
          [&](const Call *call) { both(call->arguments); },
          [&](const HOCall *call) {
            both(call->closure);
            both(call->arguments);
          },
          [&](const Scope *scope) {
            CollectInto(scope->goal, all, direct);
            if (scope->kind == ScopeKind::Some) {
              for (VarId var : scope->vars) {
                if (Has(all, var)) {
                  all.reset(var);
                }
                if (Has(direct, var)) {
                  direct.reset(var);
                }
              }
            }
          },
          [&](const Try *t) {
            both(t->io);
            CollectInto(t->goal, all, direct);
            CollectInto(t->then, all, direct);
            if (t->otherwise) {
              CollectInto(*t->otherwise, all, direct);
            }
            auto branch{[&](const Catch &c) {
              both(c.exception);
              CollectInto(c.pattern, all, direct);
              CollectInto(c.goal, all, direct);
            }};
            for (const Catch &c : t->catches) {
              branch(c);
            }
            if (t->catchAny) {
              branch(*t->catchAny);
            }
          },
          [&](const Atomic *atomic) {
            both({atomic->outerInitial, atomic->outerFinal});
            for (const AtomicAlternative &alternative : atomic->alternatives) {
              both({alternative.innerInitial, alternative.innerFinal});
              CollectInto(alternative.goal, all, direct);
            }
          },
      },
      goal);
  all_[GoalNumber(goal)] = std::move(all);
  direct_[GoalNumber(goal)] = std::move(direct);
}

std::vector<Set> Quantifier::ConjunctOutsides(llvm::ArrayRef<Goal> goals,
    const Set &outside, const std::vector<Set> &table, bool map) {
  // Each conjunct's outside is the goal's and the other conjuncts'.
  std::size_t n{goals.size()};
  std::vector<Set> before(n + 1), after(n + 1), result(n);
  for (std::size_t j{0}; j < n; ++j) {
    before[j + 1] = before[j];
    before[j + 1] |= table[GoalNumber(goals[j])];
  }
  for (std::size_t j{n}; j > 0; --j) {
    after[j - 1] = after[j];
    after[j - 1] |= table[GoalNumber(goals[j - 1])];
  }
  for (std::size_t j{0}; j < n; ++j) {
    Set others{before[j]};
    others |= after[j + 1];
    result[j] = map ? Map(others) : others;
    result[j] |= outside;
  }
  return result;
}

void Quantifier::Capture(const Goal &goal, const Set &outside) {
  Set free;
  auto with{[&](const Set &extra) {
    Set result{outside};
    result |= extra;
    return result;
  }};
  auto into{[&](const Goal &g, const Set &out) {
    Capture(g, out);
    free |= free_[GoalNumber(g)];
  }};
  auto without{[&](Set set, llvm::ArrayRef<VarId> vars) {
    for (VarId var : vars) {
      if (Has(set, var)) {
        set.reset(var);
      }
    }
    return set;
  }};
  auto leaf{[&](llvm::ArrayRef<VarId> vars) { Add(free, vars); }};
  std::visit(
      common::visitors{
          [&](const Conj *conj) {
            std::vector<Set> outsides{
                ConjunctOutsides(conj->goals, outside, direct_, false)};
            for (std::size_t j{0}; j < conj->goals.size(); ++j) {
              into(conj->goals[j], outsides[j]);
            }
          },
          [&](const Disj *disj) {
            for (const Goal &g : disj->goals) {
              into(g, outside);
            }
          },
          [&](const IfThenElse *ite) {
            const Scope *scope{ConditionScope(*ite)};
            if (scope) {
              Set out{without(outside, scope->vars)};
              Set conditionOut{out};
              conditionOut |= direct_[GoalNumber(ite->then)];
              Capture(scope->goal, conditionOut);
              free_[scope->number] = free_[GoalNumber(scope->goal)];
              Set thenOut{out};
              thenOut |= direct_[GoalNumber(scope->goal)];
              Capture(ite->then, thenOut);
              Set pair{free_[scope->number]};
              pair |= free_[GoalNumber(ite->then)];
              free |= without(pair, scope->vars);
            } else {
              into(ite->condition, with(direct_[GoalNumber(ite->then)]));
              into(ite->then, with(direct_[GoalNumber(ite->condition)]));
            }
            into(ite->otherwise, outside);
          },
          [&](const Not *negation) { into(negation->goal, outside); },
          [&](const Unify *unify) {
            leaf(unify->left);
            if (unify->isVar) {
              leaf(unify->var);
            } else {
              leaf(unify->arguments);
            }
            if (const Lambda * lambda{unify->functor.lambda}) {
              Set captured{all_[GoalNumber(lambda->body)]};
              captured &= outside;
              free |= captured;
              Set body{captured};
              Add(body, lambda->arguments);
              Capture(lambda->body, body);
              captured_[unify->number] = std::move(captured);
            }
          },
          [&](const Call *call) { leaf(call->arguments); },
          [&](const HOCall *call) {
            leaf(call->closure);
            leaf(call->arguments);
          },
          [&](const Scope *scope) {
            if (scope->kind == ScopeKind::Some) {
              Capture(scope->goal, without(outside, scope->vars));
              free |= without(free_[GoalNumber(scope->goal)], scope->vars);
            } else {
              into(scope->goal, outside);
            }
          },
          [&](const Try *t) {
            leaf(t->io);
            Set out{outside};
            Add(out, t->io);
            auto and_{[&](const Set &extra) {
              Set s{out};
              s |= extra;
              return s;
            }};
            into(t->goal, and_(direct_[GoalNumber(t->then)]));
            into(t->then, and_(direct_[GoalNumber(t->goal)]));
            if (t->otherwise) {
              into(*t->otherwise, out);
            }
            auto branch{[&](const Catch &c) {
              leaf(c.exception);
              Set pattern{and_(direct_[GoalNumber(c.goal)])};
              Add(pattern, c.exception);
              Set body{and_(direct_[GoalNumber(c.pattern)])};
              Add(body, c.exception);
              into(c.pattern, pattern);
              into(c.goal, body);
            }};
            for (const Catch &c : t->catches) {
              branch(c);
            }
            if (t->catchAny) {
              branch(*t->catchAny);
            }
          },
          [&](const Atomic *atomic) {
            leaf({atomic->outerInitial, atomic->outerFinal});
            Set out{outside};
            Add(out, {atomic->outerInitial, atomic->outerFinal});
            for (const AtomicAlternative &alternative : atomic->alternatives) {
              leaf({alternative.innerInitial, alternative.innerFinal});
              Set inner{out};
              Add(inner, {alternative.innerInitial, alternative.innerFinal});
              into(alternative.goal, inner);
            }
          },
      },
      goal);
  free_[GoalNumber(goal)] = std::move(free);
}

Quantifier::Local Quantifier::Begin(const Set &occurring, const Set &outside) {
  Local local{map_, locals_++};
  active_.insert(local.id);
  Claim(occurring, outside, local.id);
  return local;
}

void Quantifier::Claim(
    const Set &occurring, const Set &outside, unsigned id) {
  for (unsigned var : occurring.set_bits()) {
    VarId current{MapVar(var)};
    if (Has(outside, current)) {
      continue;
    }
    auto it{owner_.find(current)};
    if (it != owner_.end() && !active_.contains(it->second)) {
      Rename(var);
      current = MapVar(var);
    }
    owner_[current] = id;
  }
}

Goal Quantifier::Rebuild(const Goal &goal, const Set &outside) {
  Record(goal, outside);
  auto with{[&](const Set &base, const Set &extra) {
    Set result{base};
    result |= Map(extra);
    return result;
  }};
  auto local{[&](const Goal &g, const Set &out) {
    Local scope{Begin(all_[GoalNumber(g)], out)};
    Goal result{Rebuild(g, out)};
    End(std::move(scope));
    return result;
  }};
  return std::visit(
      common::visitors{
          [&](const Conj *conj) -> Goal {
            std::vector<Set> outsides{
                ConjunctOutsides(conj->goals, outside, free_, true)};
            llvm::SmallVector<Goal, 4> goals;
            for (std::size_t j{0}; j < conj->goals.size(); ++j) {
              goals.push_back(Rebuild(conj->goals[j], outsides[j]));
            }
            return Make<Conj>(conj->tokens, conj->number, conj->parallel,
                Save(llvm::ArrayRef<Goal>{goals}));
          },
          [&](const Disj *disj) -> Goal {
            llvm::SmallVector<Goal, 4> goals;
            for (const Goal &g : disj->goals) {
              goals.push_back(local(g, outside));
            }
            return Make<Disj>(
                disj->tokens, disj->number, Save(llvm::ArrayRef<Goal>{goals}));
          },
          [&](const IfThenElse *ite) -> Goal {
            const Scope *scope{ConditionScope(*ite)};
            const Goal &condition{scope ? scope->goal : ite->condition};
            unsigned c{GoalNumber(condition)}, t{GoalNumber(ite->then)};
            // The condition and the then part are one goal to which
            // variables may be local; a quantifier at the top of the
            // condition quantifies over both.
            Set pair{all_[c]};
            pair |= all_[t];
            Local both{map_, locals_++};
            active_.insert(both.id);
            if (scope) {
              Quantified(scope->vars, outside, scope->tokens);
              Add(pair, scope->vars);
            }
            Claim(pair, outside, both.id);
            Set conditionOutside{with(outside, free_[t])};
            Goal newCondition{Rebuild(condition, conditionOutside)};
            if (scope) {
              nonlocals_[scope->number] = nonlocals_[c];
              newCondition = Make<Scope>(scope->tokens, scope->number,
                  scope->kind, scope->origin, SaveMapped(scope->vars),
                  newCondition);
            }
            Goal then{Rebuild(ite->then, with(outside, free_[c]))};
            End(std::move(both));
            Goal otherwise{local(ite->otherwise, outside)};
            return Make<IfThenElse>(
                ite->tokens, ite->number, newCondition, then, otherwise);
          },
          [&](const Not *negation) -> Goal {
            return Make<Not>(negation->tokens, negation->number,
                local(negation->goal, outside));
          },
          [&](const Unify *unify) -> Goal {
            Qualifications(unify->tokens);
            Functor functor{unify->functor};
            if (const Lambda * lambda{unify->functor.lambda}) {
              Set occurring{all_[GoalNumber(lambda->body)]};
              Add(occurring, lambda->arguments);
              Set captured{Map(captured_[unify->number])};
              // mmc warns of an argument with the name of a variable
              // outside the lambda expression.
              llvm::StringSet<> names;
              for (unsigned var : outside.set_bits()) {
                if (vars_[var].kind == VarKind::Program) {
                  names.insert(vars_[var].name);
                }
              }
              Local scope{Begin(occurring, captured)};
              for (unsigned var : occurring.set_bits()) {
                VarId current{MapVar(var)};
                if (!Has(captured, current) &&
                    vars_[current].kind == VarKind::Program &&
                    names.contains(vars_[current].name)) {
                  Overlapping(current, lambda->tokens);
                }
              }
              llvm::ArrayRef<VarId> arguments{SaveMapped(lambda->arguments)};
              Set body{captured};
              Add(body, arguments);
              Goal newBody{Rebuild(lambda->body, body)};
              End(std::move(scope));
              functor.lambda = Make<Lambda>(lambda->tokens, lambda->kind,
                  lambda->any, lambda->purity, arguments, lambda->modes,
                  lambda->determinism, newBody);
            }
            return Make<Unify>(unify->tokens, unify->number,
                MapVar(unify->left), unify->isVar, MapVar(unify->var), functor,
                SaveMapped(unify->arguments));
          },
          [&](const Call *call) -> Goal {
            Qualifications(call->tokens);
            return Make<Call>(call->tokens, call->number, call->name,
                SaveMapped(call->arguments), call->event);
          },
          [&](const HOCall *call) -> Goal {
            Qualifications(call->tokens);
            return Make<HOCall>(call->tokens, call->number, call->kind,
                MapVar(call->closure), SaveMapped(call->arguments),
                call->purity);
          },
          [&](const Scope *scope) -> Goal {
            if (scope->kind != ScopeKind::Some) {
              return Make<Scope>(scope->tokens, scope->number, scope->kind,
                  scope->origin, SaveMapped(scope->vars),
                  Rebuild(scope->goal, outside));
            }
            Local some{map_, locals_++};
            active_.insert(some.id);
            Quantified(scope->vars, outside, scope->tokens);
            Set occurring{all_[GoalNumber(scope->goal)]};
            Add(occurring, scope->vars);
            Claim(occurring, outside, some.id);
            llvm::ArrayRef<VarId> vars{SaveMapped(scope->vars)};
            Goal inner{Rebuild(scope->goal, outside)};
            End(std::move(some));
            return Make<Scope>(scope->tokens, scope->number, scope->kind,
                scope->origin, vars, inner);
          },
          [&](const Try *t) -> Goal {
            llvm::ArrayRef<VarId> io{SaveMapped(t->io)};
            Set out{outside};
            Add(out, io);
            unsigned g{GoalNumber(t->goal)}, th{GoalNumber(t->then)};
            Set pair{all_[g]};
            pair |= all_[th];
            Local both{Begin(pair, out)};
            Goal tried{Rebuild(t->goal, with(out, free_[th]))};
            Goal then{Rebuild(t->then, with(out, free_[g]))};
            End(std::move(both));
            std::optional<Goal> otherwise;
            if (t->otherwise) {
              otherwise = local(*t->otherwise, out);
            }
            auto branch{[&](const Catch &c) {
              unsigned p{GoalNumber(c.pattern)}, b{GoalNumber(c.goal)};
              Set occurring{all_[p]};
              occurring |= all_[b];
              Add(occurring, c.exception);
              Local scope{Begin(occurring, out)};
              VarId exception{MapVar(c.exception)};
              Set patternOut{with(out, free_[b])};
              Add(patternOut, exception);
              Set bodyOut{with(out, free_[p])};
              Add(bodyOut, exception);
              Catch result{c.tokens, exception, Rebuild(c.pattern, patternOut),
                  Rebuild(c.goal, bodyOut)};
              End(std::move(scope));
              return result;
            }};
            llvm::SmallVector<Catch, 2> catches;
            for (const Catch &c : t->catches) {
              catches.push_back(branch(c));
            }
            std::optional<Catch> catchAny;
            if (t->catchAny) {
              catchAny = branch(*t->catchAny);
            }
            return Make<Try>(t->tokens, t->number, io, tried, then, otherwise,
                Save(llvm::ArrayRef<Catch>{catches}), catchAny);
          },
          [&](const Atomic *atomic) -> Goal {
            VarId initial{MapVar(atomic->outerInitial)};
            VarId final{MapVar(atomic->outerFinal)};
            llvm::ArrayRef<VarId> vars{SaveMapped(atomic->vars)};
            Set out{outside};
            Add(out, {initial, final});
            llvm::SmallVector<AtomicAlternative, 2> alternatives;
            for (const AtomicAlternative &alternative : atomic->alternatives) {
              Set occurring{all_[GoalNumber(alternative.goal)]};
              Add(occurring, {alternative.innerInitial, alternative.innerFinal});
              Local scope{Begin(occurring, out)};
              VarId innerInitial{MapVar(alternative.innerInitial)};
              VarId innerFinal{MapVar(alternative.innerFinal)};
              Set inner{out};
              Add(inner, {innerInitial, innerFinal});
              alternatives.push_back(AtomicAlternative{
                  innerInitial, innerFinal, Rebuild(alternative.goal, inner)});
              End(std::move(scope));
            }
            return Make<Atomic>(atomic->tokens, atomic->number, initial, final,
                vars, Save(llvm::ArrayRef<AtomicAlternative>{alternatives}));
          },
      },
      goal);
}

const ClauseForm &Quantifier::Run() {
  Collect(clause_.body);
  Set head;
  Add(head, clause_.head);
  Capture(clause_.body, head);
  Goal body{Rebuild(clause_.body, head)};
  std::vector<TypeQualification> qualifications{
      clause_.typeQualifications.begin(), clause_.typeQualifications.end()};
  for (std::size_t j{0}; j < qualifications.size(); ++j) {
    qualifications[j].var = qualified_[j];
  }
  return *Make<ClauseForm>(clause_.tokens, clause_.kind, clause_.name,
      clause_.head, body, Save(llvm::ArrayRef<VarInfo>{vars_}),
      Save(llvm::ArrayRef<TypeQualification>{qualifications}), clause_.goals,
      Save(llvm::ArrayRef<llvm::ArrayRef<VarId>>{nonlocals_}),
      clause_.modeSpecific);
}

} // namespace

const ClauseForm &Quantify(const ClauseForm &clause,
    const parser::TokenSequence &tokens, SemanticArena &arena,
    parser::Messages &messages) {
  return Quantifier{clause, tokens, arena, messages}.Run();
}

} // namespace merkurius::semantics
