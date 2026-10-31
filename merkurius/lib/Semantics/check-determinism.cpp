//===-- lib/Semantics/check-determinism.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "merkurius/Semantics/check-determinism.h"
#include "merkurius/Common/idioms.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <functional>
#include <map>
#include <optional>
#include <set>

namespace merkurius::semantics {

namespace {

// The goals of a conjunction, or the goal itself.
std::vector<const Body *> Conjuncts(const Body *goal) {
  if (goal->kind == BodyKind::Conj) {
    return goal->goals;
  }
  return {goal};
}

} // namespace

// Turns the clauses of a procedure, as mode analysis checked them, into
// its body in mmc's form.
class BodyBuilder {
public:
  BodyBuilder(DeterminismChecking &result, ProcedureBody &out,
      InstOps &ops)
      : result_{result}, out_{out}, ops_{ops} {}

  void Build(const Procedure &procedure, llvm::ArrayRef<ClauseModes> clauses) {
    std::size_t arity{procedure.modes.size()};
    if (clauses.size() == 1) {
      const ClauseForm &form{*clauses.front().clause->clause};
      for (std::size_t i{0}; i < form.head.size(); ++i) {
        out_.head.push_back(i);
      }
      Start(clauses.front(), true);
      out_.body = Whole(form.body);
      return;
    }
    for (std::size_t i{0}; i < arity; ++i) {
      out_.head.push_back(NewVar("HeadVar__" + std::to_string(i + 1),
          clauses.front().clause->vars[clauses.front().clause->clause->head[i]]));
    }
    Body disj{BodyKind::Disj, clauses.front().clause->clause->tokens};
    for (std::size_t i{0}; i < arity; ++i) {
      disj.entry.emplace_back(out_.head[i], procedure.modes[i].initial);
    }
    // A clause that is fail is left out, and if one is left, the body is
    // that clause, as in mmc.
    for (const ClauseModes &clause : clauses) {
      Start(clause, false);
      const Body *goal{Whole(clause.clause->clause->body)};
      if (goal->kind != BodyKind::Disj || !goal->goals.empty()) {
        disj.goals.push_back(goal);
      }
    }
    out_.body = disj.goals.size() == 1 ? disj.goals.front() : New(std::move(disj));
  }

private:
  const Body *New(Body body) { return &result_.nodes_.emplace_back(std::move(body)); }

  BodyVar NewVar(std::string name, const Type *type) {
    BodyVar var{static_cast<BodyVar>(out_.vars.size())};
    out_.vars.push_back({name.empty() ? "V_" + std::to_string(var) : name, type});
    return var;
  }

  // Gives each of the clause's variables one of the body's: the head's
  // to those of the procedure, and the argument of each head unification
  // that assigns it, if desugaring made it for a pattern, to that head
  // variable, since mmc unifies the head variable with the pattern; a new
  // one to each other, in the clause's order, which is mmc's.
  void Start(const ClauseModes &clause, bool single) {
    clause_ = &clause;
    const ClauseForm &form{*clause.clause->clause};
    map_.assign(form.vars.size(), ~BodyVar{0});
    for (std::size_t i{0}; i < form.head.size(); ++i) {
      if (single) {
        out_.vars.push_back({Name(form, form.head[i], i),
            clause.clause->vars[form.head[i]]});
      }
      map_[form.head[i]] = out_.head[i];
    }
    dropped_.clear();
    if (!single) {
      if (const auto *conj{std::get_if<const Conj *>(&form.body)}) {
        for (const semantics::Goal &goal : (*conj)->goals) {
          const auto *unify{std::get_if<const Unify *>(&goal)};
          if (!unify) {
            continue;
          }
          const GoalModes &modes{clause.goals[(*unify)->number]};
          if (modes.head && modes.unify == UnifyKind::Assign &&
              form.vars[(*unify)->var].kind != VarKind::Program &&
              form.vars[(*unify)->var].kind != VarKind::State) {
            map_[(*unify)->var] = map_[(*unify)->left];
            dropped_.insert(*unify);
          }
        }
      }
    }
    for (VarId var{0}; var < form.vars.size(); ++var) {
      if (map_[var] == ~BodyVar{0}) {
        map_[var] = NewVar(Name(form, var, std::nullopt), clause.clause->vars[var]);
      }
    }
  }

  static std::string Name(const ClauseForm &form, VarId var,
      std::optional<std::size_t> head) {
    const VarInfo &info{form.vars[var]};
    if (info.kind == VarKind::Program) {
      return info.name.str();
    }
    if (info.kind == VarKind::State) {
      return "STATE_VARIABLE_" + info.name.str() + "_" +
          std::to_string(info.version);
    }
    if (head) {
      return "HeadVar__" + std::to_string(*head + 1);
    }
    return {};
  }

  const Type *TypeOf(BodyVar var) const { return out_.vars[var].type; }

  // The insts of a goal's nonlocals before it, by the body's variables.
  std::vector<std::pair<BodyVar, const Inst *>> Before(unsigned number) const {
    const ClauseForm &form{*clause_->clause->clause};
    const GoalModes &modes{clause_->goals[number]};
    std::vector<std::pair<BodyVar, const Inst *>> before;
    for (std::size_t i{0};
         i < form.nonlocals[number].size() && i < modes.before.size(); ++i) {
      before.emplace_back(map_[form.nonlocals[number][i]], modes.before[i]);
    }
    return before;
  }

  // A goal, or true if it is left out.
  const Body *Whole(const semantics::Goal &goal) {
    if (const Body *out{Goal(goal)}) {
      return out;
    }
    TokenRange tokens{std::visit([](const auto *g) { return g->tokens; }, goal)};
    return New(Body{BodyKind::Conj, tokens});
  }

  // A goal in mmc's form, or nullptr if mmc leaves it out.
  const Body *Goal(const semantics::Goal &goal) {
    unsigned number{std::visit([](const auto *g) { return g->number; }, goal)};
    TokenRange tokens{std::visit([](const auto *g) { return g->tokens; }, goal)};
    const GoalModes &modes{clause_->goals[number]};
    return common::visit(
        common::visitors{
            [&](const Conj *) {
              Body conj{BodyKind::Conj, tokens, &modes};
              if (modes.order.empty() && !std::get<const Conj *>(goal)->goals.empty()) {
                // Not run: a goal before it cannot succeed.
                return New(std::move(conj));
              }
              // If a unification cannot succeed, the goals written after it
              // are not run in mmc, which runs them in the order written
              // when it can.
              // A head unification, which mode analysis numbers after the
              // body, is run when it is written before the unification.
              const semantics::Goal *failing{nullptr};
              for (const semantics::Goal &inner : modes.order) {
                const auto *unify{std::get_if<const Unify *>(&inner)};
                if (unify && clause_->goals[(*unify)->number].fails) {
                  failing = &inner;
                }
              }
              auto number{[](const semantics::Goal &g) {
                return std::visit([](const auto *g) { return g->number; }, g);
              }};
              auto first{[](const semantics::Goal &g) {
                return std::visit([](const auto *g) { return g->tokens.first; }, g);
              }};
              for (const semantics::Goal &inner : modes.order) {
                if (failing && number(inner) > number(*failing) &&
                    !(clause_->goals[number(inner)].head &&
                        first(inner) < first(*failing))) {
                  continue;
                }
                const Body *goal{Goal(inner)};
                if (!goal) {
                  continue;
                }
                for (const Body *g : Conjuncts(goal)) {
                  conj.goals.push_back(g);
                }
              }
              if (conj.goals.size() == 1) {
                return conj.goals.front();
              }
              return New(std::move(conj));
            },
            [&](const Disj *disj) {
              // A branch that is fail is left out, and one that is a
              // disjunction is its branches, as in mmc.
              Body out{BodyKind::Disj, tokens, &modes};
              for (const semantics::Goal &branch : disj->goals) {
                const Body *goal{Whole(branch)};
                if (goal->kind == BodyKind::Disj) {
                  llvm::append_range(out.goals, goal->goals);
                } else {
                  out.goals.push_back(goal);
                }
              }
              if (out.goals.size() == 1 && !disj->goals.empty()) {
                return out.goals.front();
              }
              // A fail uses no variables.
              if (!out.goals.empty()) {
                out.entry = Before(number);
              }
              return New(std::move(out));
            },
            [&](const IfThenElse *ite) {
              Body out{BodyKind::IfThenElse, tokens, &modes};
              out.goals = {Whole(ite->condition), Whole(ite->then),
                  Whole(ite->otherwise)};
              return New(std::move(out));
            },
            [&](const Not *negation) {
              Body out{BodyKind::Not, tokens, &modes};
              out.goals = {Whole(negation->goal)};
              return New(std::move(out));
            },
            [&](const Scope *scope) {
              Body out{BodyKind::Scope, tokens, &modes};
              out.scope = scope->kind;
              out.goals = {Whole(scope->goal)};
              return New(std::move(out));
            },
            [&](const Unify *unify) -> const Body * {
              // As in mmc, which deletes a construction or assignment of a
              // variable not used after it, and replaces a unification
              // that cannot succeed with fail.
              if (dropped_.count(unify) || modes.dead) {
                return nullptr;
              }
              if (modes.fails) {
                return New(Body{BodyKind::Disj, tokens, &modes});
              }
              return Unification(*unify, modes, tokens);
            },
            [&](const Call *call) {
              Body out{BodyKind::Call, tokens, &modes};
              out.call = call;
              for (VarId var : call->arguments) {
                out.arguments.push_back(map_[var]);
              }
              return New(std::move(out));
            },
            [&](const HOCall *call) {
              Body out{BodyKind::HOCall, tokens, &modes};
              out.hoCall = call;
              out.closure = map_[call->closure];
              for (VarId var : call->arguments) {
                out.arguments.push_back(map_[var]);
              }
              return New(std::move(out));
            },
            [&](const auto *) -> const Body * { return nullptr; },
        },
        goal);
  }

  // A deconstruction binds new variables, and each argument that was
  // bound is tested against its own, as mmc does.
  const Body *Unification(
      const Unify &unify, const GoalModes &modes, TokenRange tokens) {
    Body out{BodyKind::Unify, tokens, &modes};
    out.source = &unify;
    out.unify = modes.unify;
    out.isVar = unify.isVar;
    out.left = map_[unify.left];
    out.var = map_[unify.var];
    // An assignment's left side is the variable assigned.
    if (unify.isVar && modes.unify == UnifyKind::Assign &&
        modes.assigned == unify.var) {
      std::swap(out.left, out.var);
    }
    out.functor = modes.functor;
    for (VarId var : unify.arguments) {
      out.arguments.push_back(map_[var]);
    }
    if (const Lambda *lambda{unify.functor.lambda}) {
      out.lambda = Whole(lambda->body);
      for (VarId var : lambda->arguments) {
        out.lambdaHead.push_back(map_[var]);
      }
    }
    if (unify.isVar || unify.functor.kind == FunctorKind::Coerce ||
        unify.functor.kind == FunctorKind::Lambda || modes.callee ||
        (modes.unify != UnifyKind::Deconstruct &&
            modes.unify != UnifyKind::Test)) {
      return New(std::move(out));
    }
    out.unify = UnifyKind::Deconstruct;
    std::vector<std::pair<BodyVar, const Inst *>> before{Before(unify.number)};
    Body conj{BodyKind::Conj, tokens, &modes};
    std::vector<Body> tests;
    for (BodyVar &argument : out.arguments) {
      auto it{llvm::find_if(
          before, [&](const auto &b) { return b.first == argument; })};
      if (it == before.end() || ops_.IsFree(it->second)) {
        continue;
      }
      BodyVar fresh{NewVar({}, TypeOf(argument))};
      Body test{BodyKind::Unify, tokens, &modes};
      test.source = &unify;
      test.unify = UnifyKind::Test;
      test.isVar = true;
      test.left = argument;
      test.var = fresh;
      tests.push_back(std::move(test));
      argument = fresh;
    }
    if (tests.empty()) {
      return New(std::move(out));
    }
    conj.goals.push_back(New(std::move(out)));
    for (Body &test : tests) {
      conj.goals.push_back(New(std::move(test)));
    }
    return New(std::move(conj));
  }

  DeterminismChecking &result_;
  ProcedureBody &out_;
  InstOps &ops_;
  const ClauseModes *clause_{nullptr};
  std::vector<BodyVar> map_;
  std::set<const Unify *> dropped_;
};

// Switch detection and common subexpression elimination on a procedure's
// body, as mmc does them.
class SwitchDetector {
public:
  SwitchDetector(DeterminismChecking &result, ProcedureBody &body, InstOps &ops)
      : result_{result}, body_{body}, ops_{ops} {}

  // Detects switches, then moves the deconstructions out of disjunctions
  // that common subexpression elimination can, and if it changed anything,
  // simplifies the body and detects switches in all of it again, until it
  // cannot, as mmc does.
  void Run() {
    body_.body = Detect(body_.body);
    while (true) {
      changed_ = false;
      body_.body = Eliminate(body_.body);
      if (!changed_) {
        break;
      }
      std::map<BodyVar, InstFunctor> known;
      body_.body = Prune(body_.body, known);
      std::set<BodyVar> live{body_.head.begin(), body_.head.end()};
      if (const Body *kept{Simplify(body_.body, live)}) {
        body_.body = kept;
      } else {
        body_.body = New(Body{BodyKind::Conj, body_.body->tokens});
      }
      body_.body = Detect(body_.body);
    }
  }

private:
  using Entry = std::vector<std::pair<BodyVar, const Inst *>>;
  // The tests a disjunct makes of a variable: one, or one for each branch
  // of a disjunction in it, with the disjunct as each branch makes it.
  using Found = std::vector<std::pair<InstFunctor, const Body *>>;

  const Body *New(Body body) { return &result_.nodes_.emplace_back(std::move(body)); }

  BodyVar NewVar(const Type *type) {
    BodyVar var{static_cast<BodyVar>(body_.vars.size())};
    body_.vars.push_back({"V_" + std::to_string(var), type});
    return var;
  }

  const Body *Conj(TokenRange tokens, const std::vector<const Body *> &goals) {
    if (goals.size() == 1) {
      return goals.front();
    }
    Body conj{BodyKind::Conj, tokens};
    for (const Body *goal : goals) {
      for (const Body *g : Conjuncts(goal)) {
        conj.goals.push_back(g);
      }
    }
    return New(std::move(conj));
  }

  // A copy of a compound goal with other goals.
  const Body *With(const Body *goal, std::vector<const Body *> goals) {
    Body copy{*goal};
    copy.goals = std::move(goals);
    return New(std::move(copy));
  }

  const Body *Detect(const Body *goal) {
    switch (goal->kind) {
    case BodyKind::Disj:
      if (goal->goals.empty()) {
        return goal;
      }
      return Disjunction(goal->tokens, goal->goals, goal->entry);
    case BodyKind::Switch: {
      Body copy{*goal};
      for (SwitchCase &c : copy.cases) {
        std::optional<Cased> cased;
        if (c.others.empty()) {
          cased.emplace(*this, goal->switchVar, c.functor);
        }
        c.goal = Detect(c.goal);
      }
      return New(std::move(copy));
    }
    case BodyKind::Unify:
      if (goal->lambda) {
        Body copy{*goal};
        copy.lambda = Detect(goal->lambda);
        return New(std::move(copy));
      }
      return goal;
    case BodyKind::Call:
    case BodyKind::HOCall:
      return goal;
    default: {
      std::vector<const Body *> goals;
      for (const Body *g : goal->goals) {
        goals.push_back(Detect(g));
      }
      if (goal->kind == BodyKind::Conj) {
        return Conj(goal->tokens, goals);
      }
      return With(goal, std::move(goals));
    }
    }
  }

  // A disjunction is a switch on a variable bound before it that each
  // disjunct tests against a functor, with at least two functors: the
  // first that cannot fail, else the first of a type without constructors,
  // such as int, else the first. If there is none, each
  // variable that some disjuncts test, against at least two functors,
  // makes a partial switch, and the one that cannot fail, else the one
  // with the most functors, else the first of them, is made, followed by
  // the other disjuncts as a disjunction, in which switches are detected
  // in the same way (Henderson, Somogyi and Conway, figure 2). As in mmc,
  // if every disjunct tests a variable against the same functor, the only
  // switches are those that cannot fail and those on a variable of a type
  // without constructors, and without one the disjuncts are left as they
  // are, for common subexpression elimination.
  const Body *Disjunction(TokenRange tokens,
      const std::vector<const Body *> &disjuncts, const Entry &entry) {
    struct Partial {
      BodyVar var;
      std::vector<std::pair<InstFunctor, std::vector<const Body *>>> cases;
      std::vector<const Body *> left;
    };
    // Those that are switches, and those that are partial switches.
    std::vector<Partial> fulls;
    std::vector<Partial> partials;
    Entry bound;
    for (const auto &[var, inst] : entry) {
      if (!ops_.IsFree(inst)) {
        bound.emplace_back(var, inst);
      }
    }
    std::sort(bound.begin(), bound.end());
    // Whether a variable is tested in every disjunct against one functor.
    bool single{false};
    for (const auto &[var, inst] : bound) {
      Partial partial{var};
      auto add{[&](const InstFunctor &functor, const Body *goal) {
        auto it{llvm::find_if(
            partial.cases, [&](const auto &c) { return c.first == functor; })};
        if (it == partial.cases.end()) {
          partial.cases.push_back({functor, {}});
          it = std::prev(partial.cases.end());
        }
        it->second.push_back(goal);
      }};
      for (const Body *disjunct : disjuncts) {
        std::optional<Found> found{Scan(Conjuncts(disjunct), var, {var}, false)};
        if (!found || found->empty()) {
          partial.left.push_back(disjunct);
          continue;
        }
        for (auto &[functor, goal] : *found) {
          add(functor, goal);
        }
      }
      // mmc keeps the other disjuncts last first. If each can be split at
      // a disjunction into tests, the switch has all of them, those of
      // each after the others in its case.
      std::reverse(partial.left.begin(), partial.left.end());
      std::vector<Found> splits;
      for (const Body *disjunct : partial.left) {
        std::optional<Found> found{Scan(Conjuncts(disjunct), var, {var}, true)};
        if (!found || found->empty()) {
          splits.clear();
          break;
        }
        splits.push_back(std::move(*found));
      }
      if (!splits.empty()) {
        partial.left.clear();
        for (Found &found : splits) {
          for (auto &[functor, goal] : found) {
            add(functor, goal);
          }
        }
      }
      if (partial.cases.size() == 1 && partial.left.empty()) {
        single = true;
        continue;
      }
      if (partial.cases.size() < 2) {
        continue;
      }
      (partial.left.empty() ? fulls : partials).push_back(std::move(partial));
    }
    auto canFail{[&](const Partial &p) {
      std::vector<InstFunctor> functors;
      for (const auto &c : p.cases) {
        functors.push_back(c.first);
      }
      return CanFail(p.var, functors, entry);
    }};
    if (single) {
      llvm::erase_if(fulls, [&](const Partial &p) {
        return ops_.ConstructorsOf(body_.vars[p.var].type) != nullptr &&
            canFail(p);
      });
    }
    if (!fulls.empty()) {
      auto rank{[&](const Partial &p) {
        return std::make_pair(!canFail(p),
            ops_.ConstructorsOf(body_.vars[p.var].type) == nullptr);
      }};
      const Partial *best{&fulls.front()};
      for (const Partial &full : fulls) {
        if (rank(full) > rank(*best)) {
          best = &full;
        }
      }
      return Switch(tokens, best->var, best->cases, entry);
    }
    if (partials.empty() || single) {
      // If a variable is tested against one functor in every disjunct, mmc
      // leaves the disjuncts as they are, for common subexpression
      // elimination to take the test out of the disjunction.
      Body disj{BodyKind::Disj, tokens};
      for (const Body *disjunct : disjuncts) {
        disj.goals.push_back(single ? disjunct : Detect(disjunct));
      }
      disj.entry = entry;
      return New(std::move(disj));
    }
    // One that cannot fail first, then the one with the most functors,
    // then the first, as in mmc.
    auto rank{[&](const Partial &p) {
      return std::make_pair(!canFail(p), p.cases.size());
    }};
    const Partial *best{&partials.front()};
    for (const Partial &partial : partials) {
      if (rank(partial) > rank(*best)) {
        best = &partial;
      }
    }
    Body disj{BodyKind::Disj, tokens};
    disj.entry = entry;
    disj.goals.push_back(Switch(tokens, best->var, best->cases, entry));
    // The other disjuncts are a disjunction, even if there is one.
    const Body *rest{Disjunction(tokens, best->left, entry)};
    if (rest->kind == BodyKind::Disj) {
      llvm::append_range(disj.goals, rest->goals);
    } else {
      disj.goals.push_back(rest);
    }
    return New(std::move(disj));
  }

  // A switch's cases are in the order of their functors, as in mmc. It
  // can fail if the variable's inst before it allows a functor of its
  // type that no case has, or its type has no known set of constructors.
  const Body *Switch(TokenRange tokens, BodyVar var,
      std::vector<std::pair<InstFunctor, std::vector<const Body *>>> cases,
      const Entry &entry) {
    std::stable_sort(cases.begin(), cases.end(), [](const auto &a, const auto &b) {
      return FunctorBefore(a.first, b.first);
    });
    // In a case of a switch on the same variable, only that case's
    // functor can be reached, as the manual says and mmc does.
    auto known{cased_.find(var)};
    if (known != cased_.end()) {
      llvm::erase_if(
          cases, [&](const auto &c) { return !(c.first == known->second); });
      if (cases.empty()) {
        return New(Body{BodyKind::Disj, tokens});
      }
    }
    Body out{BodyKind::Switch, tokens};
    out.switchVar = var;
    out.entry = entry;
    std::vector<InstFunctor> functors;
    for (auto &[functor, goals] : cases) {
      functors.push_back(functor);
      if (goals.size() == 1 && merged_.count(goals.front())) {
        auto it{llvm::find_if(out.cases,
            [&](const SwitchCase &c) { return c.goal == goals.front(); })};
        if (it != out.cases.end()) {
          it->others.push_back(functor);
          continue;
        }
        // Detected when the case is complete.
        out.cases.push_back({functor, goals.front()});
        continue;
      }
      Cased cased{*this, var, functor};
      out.cases.push_back({functor,
          goals.size() == 1 ? Detect(goals.front())
                            : Disjunction(tokens, goals, entry)});
    }
    for (SwitchCase &c : out.cases) {
      if (merged_.count(c.goal)) {
        std::optional<Cased> cased;
        if (c.others.empty()) {
          cased.emplace(*this, var, c.functor);
        }
        c.goal = Detect(c.goal);
      }
    }
    out.canFail = known == cased_.end() && CanFail(var, functors, entry);
    return New(std::move(out));
  }

  // While it lives, the functor that a switch's variable has in the case
  // being detected.
  class Cased {
  public:
    Cased(SwitchDetector &detector, BodyVar var, const InstFunctor &functor)
        : detector_{detector}, var_{var} {
      auto it{detector_.cased_.find(var)};
      if (it != detector_.cased_.end()) {
        old_ = it->second;
      }
      detector_.cased_[var] = functor;
    }
    ~Cased() {
      if (old_) {
        detector_.cased_[var_] = *old_;
      } else {
        detector_.cased_.erase(var_);
      }
    }

  private:
    SwitchDetector &detector_;
    BodyVar var_;
    std::optional<InstFunctor> old_;
  };

  // Whether the variable's inst before the switch allows a functor of its
  // type that is not one of `functors`, or its type has no known set of
  // constructors.
  bool CanFail(BodyVar var, const std::vector<InstFunctor> &functors,
      const Entry &entry) {
    const Type *type{body_.vars[var].type};
    const Inst *inst{nullptr};
    for (const auto &[v, i] : entry) {
      if (v == var) {
        inst = i;
      }
    }
    const std::vector<ConstructorTypes> *constructors{ops_.ConstructorsOf(type)};
    if (!constructors || !inst) {
      return true;
    }
    for (const ConstructorTypes &c : *constructors) {
      if (!llvm::is_contained(functors, c.functor) &&
          ops_.Arguments(inst, c.functor, type)) {
        return true;
      }
    }
    return false;
  }

  // By name and arity, integers by value.
  static bool FunctorBefore(const InstFunctor &a, const InstFunctor &b) {
    long long x, y;
    if (a.literal && b.literal && !llvm::StringRef{a.name}.getAsInteger(0, x) &&
        !llvm::StringRef{b.name}.getAsInteger(0, y)) {
      return x < y;
    }
    return std::make_tuple(a.name, a.arity) < std::make_tuple(b.name, b.arity);
  }

  // Looks for a test of `var`, or of a variable unified with it, against a
  // functor, in a disjunct's goals, preceded only by unifications, and
  // goals in `some` scopes, as in mmc. A disjunction may come before it if
  // each of its branches, followed by the goals after it, has such a test;
  // the disjunct is then split, one for each test, with those goals in
  // each (reference manual, "Switches"). None if a goal of another kind
  // comes first; empty if the goals end first, with the variables found to
  // be unified with `var` in `aliases`.
  std::optional<Found> Scan(std::vector<const Body *> goals, BodyVar var,
      std::set<BodyVar> aliases, bool split, std::set<BodyVar> *out = nullptr) {
    // The aliases not made in a `some` scope, which are the only ones a
    // disjunction is split on in mmc.
    std::set<BodyVar> unscoped{aliases};
    for (std::size_t i{0}; i < goals.size(); ++i) {
      const Body *goal{goals[i]};
      auto rebuild{[&](const Body *replacement) {
        std::vector<const Body *> list{goals.begin(), goals.begin() + i};
        list.push_back(replacement);
        list.insert(list.end(), goals.begin() + i + 1, goals.end());
        return Conj(goal->tokens, list);
      }};
      switch (goal->kind) {
      case BodyKind::Unify:
        if (goal->isVar) {
          if (aliases.count(goal->left) || aliases.count(goal->var)) {
            aliases.insert(goal->left);
            aliases.insert(goal->var);
          }
          if (unscoped.count(goal->left) || unscoped.count(goal->var)) {
            unscoped.insert(goal->left);
            unscoped.insert(goal->var);
          }
          continue;
        }
        // A test of a switch's variable in its case against a constant is
        // true in mmc.
        if (goal->switchTest && goal->arguments.empty()) {
          continue;
        }
        if (goal->unify == UnifyKind::Deconstruct && aliases.count(goal->left)) {
          Body test{*goal};
          test.switchTest = true;
          return Found{{goal->functor, rebuild(New(std::move(test)))}};
        }
        if (goal->unify != UnifyKind::Call) {
          continue;
        }
        return std::nullopt;
      case BodyKind::Scope: {
        if (goal->scope != ScopeKind::Some) {
          return std::nullopt;
        }
        std::set<BodyVar> inner;
        std::optional<Found> found{
            Scan(Conjuncts(goal->goals.front()), var, aliases, false, &inner)};
        if (!found) {
          return std::nullopt;
        }
        if (found->empty()) {
          aliases = std::move(inner);
          continue;
        }
        Found out;
        for (auto &[functor, g] : *found) {
          out.push_back({functor, rebuild(With(goal, {g}))});
        }
        return out;
      }
      case BodyKind::Disj: {
        if (!split || goal->goals.empty()) {
          return std::nullopt;
        }
        if (i == 0) {
          if (std::optional<Found> found{Tests(goals, var)}) {
            return found;
          }
        }
        std::vector<const Body *> before{goals.begin(), goals.begin() + i};
        Found out;
        for (const Body *branch : goal->goals) {
          std::vector<const Body *> list{Conjuncts(branch)};
          list.insert(list.end(), goals.begin() + i + 1, goals.end());
          // A disjunction in a branch is not split, as in mmc.
          std::optional<Found> found{Scan(list, var, unscoped, false)};
          if (!found || found->empty()) {
            return std::nullopt;
          }
          for (auto &[functor, g] : *found) {
            std::vector<const Body *> whole{before};
            whole.push_back(g);
            out.push_back({functor, Conj(goal->tokens, whole)});
          }
        }
        return out;
      }
      default:
        return std::nullopt;
      }
    }
    if (out) {
      *out = std::move(aliases);
    }
    return Found{};
  }

  // A disjunct that starts with a disjunction whose branches each only
  // test the variable, with arguments not used after, is in mmc the goals
  // after it for each functor, and a case that has only it is merged with
  // the others that do.
  std::optional<Found> Tests(
      const std::vector<const Body *> &goals, BodyVar var) {
    std::vector<const Body *> rest{goals.begin() + 1, goals.end()};
    std::set<BodyVar> used;
    for (const Body *g : rest) {
      Uses(g, used);
    }
    for (const Body *branch : goals.front()->goals) {
      if (branch->kind != BodyKind::Unify || branch->isVar || branch->lambda ||
          branch->unify != UnifyKind::Deconstruct || branch->left != var ||
          llvm::any_of(branch->arguments,
              [&](BodyVar arg) { return used.count(arg); })) {
        return std::nullopt;
      }
    }
    const Body *after{Conj(goals.front()->tokens, rest)};
    merged_.insert(after);
    Found out;
    for (const Body *branch : goals.front()->goals) {
      out.push_back({branch->functor, after});
    }
    return out;
  }

  // Common subexpression elimination: in a disjunction, a variable bound
  // before it that each disjunct deconstructs with the same functor is
  // deconstructed before it instead, into new variables, which each
  // disjunct assigns to its own (Henderson, Somogyi and Conway, section
  // 4); the first such variable. The same for the cases of a switch, as in
  // mmc. A variable whose inst before the disjunction has free parts in
  // the functor's arguments is left alone.
  const Body *Eliminate(const Body *goal) {
    switch (goal->kind) {
    case BodyKind::Disj: {
      if (const Body *pulled{Pull(goal)}) {
        changed_ = true;
        return pulled;
      }
      std::vector<const Body *> goals;
      for (const Body *g : goal->goals) {
        goals.push_back(Eliminate(g));
      }
      return With(goal, std::move(goals));
    }
    case BodyKind::Switch: {
      if (const Body *pulled{Pull(goal)}) {
        changed_ = true;
        return pulled;
      }
      Body copy{*goal};
      for (SwitchCase &c : copy.cases) {
        c.goal = Eliminate(c.goal);
      }
      return New(std::move(copy));
    }
    case BodyKind::Unify:
      if (goal->lambda) {
        Body copy{*goal};
        copy.lambda = Eliminate(goal->lambda);
        return New(std::move(copy));
      }
      return goal;
    case BodyKind::Call:
    case BodyKind::HOCall:
      return goal;
    default: {
      std::vector<const Body *> goals;
      for (const Body *g : goal->goals) {
        goals.push_back(Eliminate(g));
      }
      if (goal->kind == BodyKind::Conj) {
        return Conj(goal->tokens, goals);
      }
      return With(goal, std::move(goals));
    }
    }
  }

  const Body *Pull(const Body *disj) {
    std::vector<const Body *> arms{disj->goals};
    if (disj->kind == BodyKind::Switch) {
      for (const SwitchCase &c : disj->cases) {
        arms.push_back(c.goal);
      }
    }
    if (arms.size() < 2) {
      return nullptr;
    }
    Entry bound;
    for (const auto &[var, inst] : disj->entry) {
      if (!ops_.IsFree(inst)) {
        bound.emplace_back(var, inst);
      }
    }
    std::sort(bound.begin(), bound.end());
    for (const auto &[var, inst] : bound) {
      if (disj->kind == BodyKind::Switch && var == disj->switchVar) {
        continue;
      }
      std::vector<const Body *> tests;
      for (const Body *disjunct : arms) {
        const Body *test{FirstTest(Conjuncts(disjunct), {var})};
        if (!test || (!tests.empty() && !(test->functor == tests.front()->functor))) {
          tests.clear();
          break;
        }
        tests.push_back(test);
      }
      if (tests.empty()) {
        continue;
      }
      const Body *first{tests.front()};
      const Type *type{body_.vars[var].type};
      std::optional<std::vector<const Inst *>> parts{
          ops_.Arguments(inst, first->functor, type)};
      if (!parts ||
          llvm::any_of(llvm::seq<std::size_t>(0, parts->size()), [&](std::size_t i) {
            return ops_.HasFree((*parts)[i], body_.vars[first->arguments[i]].type);
          })) {
        continue;
      }
      Body pulled{*first};
      pulled.left = var;
      pulled.switchTest = false;
      pulled.arguments.clear();
      Entry entry{disj->entry};
      for (std::size_t i{0}; i < first->arguments.size(); ++i) {
        BodyVar fresh{NewVar(body_.vars[first->arguments[i]].type)};
        pulled.arguments.push_back(fresh);
        entry.emplace_back(fresh, (*parts)[i]);
      }
      Body inner{*disj};
      inner.entry = std::move(entry);
      inner.goals.clear();
      for (std::size_t d{0}; d < arms.size(); ++d) {
        const Body *test{tests[d]};
        std::vector<const Body *> assigns;
        for (std::size_t i{0}; i < test->arguments.size(); ++i) {
          Body assign{BodyKind::Unify, test->tokens, test->modes};
          assign.source = test->source;
          assign.unify = UnifyKind::Assign;
          assign.isVar = true;
          assign.left = test->arguments[i];
          assign.var = pulled.arguments[i];
          assigns.push_back(New(std::move(assign)));
        }
        const Body *arm{Replace(arms[d], test, assigns)};
        if (disj->kind == BodyKind::Switch) {
          inner.cases[d].goal = arm;
        } else {
          inner.goals.push_back(arm);
        }
      }
      return Conj(disj->tokens, {New(std::move(pulled)), New(std::move(inner))});
    }
    return nullptr;
  }

  // The first deconstruction of a variable in `aliases`, or of one unified
  // with them, in a disjunct's goals after only unifications and goals in
  // `some` scopes, as Scan finds it.
  static const Body *FirstTest(
      const std::vector<const Body *> &goals, std::set<BodyVar> aliases) {
    for (const Body *goal : goals) {
      if (goal->kind == BodyKind::Scope && goal->scope == ScopeKind::Some) {
        if (const Body *test{FirstTest(Conjuncts(goal->goals.front()), aliases)}) {
          return test;
        }
        // The aliases it makes are local to it but still aliases.
        continue;
      }
      if (goal->kind != BodyKind::Unify || goal->unify == UnifyKind::Call) {
        return nullptr;
      }
      // A test of a switch's variable in its case against a constant is
      // true in mmc.
      if (goal->switchTest && goal->arguments.empty()) {
        continue;
      }
      if (goal->isVar) {
        if (aliases.count(goal->left) || aliases.count(goal->var)) {
          aliases.insert(goal->left);
          aliases.insert(goal->var);
        }
      } else if (goal->unify == UnifyKind::Deconstruct &&
          aliases.count(goal->left)) {
        return goal;
      }
    }
    return nullptr;
  }

  // A goal with `test`, one of its conjuncts or of those in its scopes,
  // replaced by `with`; true if nothing is left.
  const Body *Replace(const Body *goal, const Body *test,
      const std::vector<const Body *> &with) {
    if (goal == test) {
      return with.empty() ? New(Body{BodyKind::Conj, goal->tokens})
                          : Conj(goal->tokens, with);
    }
    std::vector<const Body *> goals;
    for (const Body *g : Conjuncts(goal)) {
      if (g == test) {
        llvm::append_range(goals, with);
      } else if (g->kind == BodyKind::Scope) {
        goals.push_back(With(g, {Replace(g->goals.front(), test, with)}));
      } else {
        goals.push_back(g);
      }
    }
    if (goals.empty()) {
      return New(Body{BodyKind::Conj, goal->tokens});
    }
    return Conj(goal->tokens, goals);
  }

  // As mmc's mode analysis again after common subexpression elimination,
  // which knows the functor of a switch's variable in each case: a
  // deconstruction of a variable known to have another functor is fail,
  // and so is the rest of its conjunction. `known` has the functors of the
  // variables before the goal, and gets those after it.
  const Body *Prune(const Body *goal, std::map<BodyVar, InstFunctor> &known) {
    // A fail that keeps the variables of the goal it replaces as its
    // entry, since mmc deletes unused goals by the uses before.
    auto fail{[&] {
      Body out{BodyKind::Disj, goal->tokens};
      std::set<BodyVar> uses;
      Uses(goal, uses);
      for (BodyVar var : uses) {
        out.entry.emplace_back(var, nullptr);
      }
      return New(std::move(out));
    }};
    auto isFail{[](const Body *g) {
      return g->kind == BodyKind::Disj && g->goals.empty();
    }};
    switch (goal->kind) {
    case BodyKind::Unify: {
      const Body *out{goal};
      if (goal->lambda) {
        std::map<BodyVar, InstFunctor> inner{known};
        Body copy{*goal};
        copy.lambda = Prune(goal->lambda, inner);
        out = New(std::move(copy));
      }
      if (goal->isVar) {
        auto from{known.find(goal->var)};
        auto to{known.find(goal->left)};
        if (goal->unify == UnifyKind::Assign && from != known.end()) {
          known[goal->left] = from->second;
        } else if (goal->unify == UnifyKind::Test && from != known.end() &&
            to != known.end() && !(from->second == to->second)) {
          return fail();
        }
      } else if (!goal->lambda && (goal->unify == UnifyKind::Deconstruct ||
                                      goal->unify == UnifyKind::Construct)) {
        auto it{known.find(goal->left)};
        if (goal->unify == UnifyKind::Deconstruct && it != known.end() &&
            !(it->second == goal->functor)) {
          return fail();
        }
        known[goal->left] = goal->functor;
      }
      return out;
    }
    case BodyKind::Call:
    case BodyKind::HOCall:
      return goal;
    case BodyKind::Conj: {
      std::vector<const Body *> goals;
      for (const Body *g : goal->goals) {
        goals.push_back(Prune(g, known));
        if (isFail(goals.back())) {
          break;
        }
      }
      return With(goal, std::move(goals));
    }
    case BodyKind::Disj: {
      std::vector<const Body *> goals;
      for (const Body *g : goal->goals) {
        std::map<BodyVar, InstFunctor> inner{known};
        goals.push_back(Prune(g, inner));
      }
      return With(goal, std::move(goals));
    }
    case BodyKind::Switch: {
      Body copy{*goal};
      // On a variable whose functor is known, the other cases cannot be
      // reached, and mmc deletes their goals.
      auto it{known.find(goal->switchVar)};
      for (SwitchCase &c : copy.cases) {
        if (it != known.end() && !c.Has(it->second)) {
          c.goal = New(Body{BodyKind::Conj, c.goal->tokens});
          continue;
        }
        std::map<BodyVar, InstFunctor> inner{known};
        if (c.others.empty()) {
          inner[goal->switchVar] = c.functor;
        }
        c.goal = Prune(c.goal, inner);
      }
      return New(std::move(copy));
    }
    case BodyKind::IfThenElse: {
      std::map<BodyVar, InstFunctor> then{known};
      std::map<BodyVar, InstFunctor> otherwise{known};
      const Body *c{Prune(goal->goals[0], then)};
      const Body *t{Prune(goal->goals[1], then)};
      const Body *e{Prune(goal->goals[2], otherwise)};
      return With(goal, {c, t, e});
    }
    default: {
      std::map<BodyVar, InstFunctor> inner{known};
      return With(goal, {Prune(goal->goals.front(), inner)});
    }
    }
  }

  // As mmc's mode analysis again after common subexpression elimination:
  // it deletes a construction or assignment of a variable not used after
  // it, and a test of a switch's variable in its case against a constant,
  // and a branch of a disjunction that is fail, and a disjunction with
  // one branch is that branch. `live` has the variables used after the
  // goal, and gets those it uses.
  const Body *Simplify(const Body *goal, std::set<BodyVar> &live) {
    const Body *out{SimplifyGoal(goal, live)};
    // mmc deletes a goal by the variables that the compound goals after it
    // had before their goals were deleted.
    switch (goal->kind) {
    case BodyKind::Disj:
    case BodyKind::Switch:
    case BodyKind::IfThenElse:
    case BodyKind::Not:
    case BodyKind::Scope:
      Uses(goal, live);
      break;
    default:
      break;
    }
    return out;
  }

  const Body *SimplifyGoal(const Body *goal, std::set<BodyVar> &live) {
    auto use{[&](BodyVar var) { live.insert(var); }};
    switch (goal->kind) {
    case BodyKind::Unify: {
      if ((goal->unify == UnifyKind::Construct ||
              goal->unify == UnifyKind::Assign) &&
          !goal->switchTest && !live.count(goal->left)) {
        return nullptr;
      }
      // A test of a switch's variable in its case against a constant is
      // true; one through a variable unified with it is kept, as in mmc.
      if (goal->switchTest && goal->arguments.empty() &&
          switching_.count(goal->left)) {
        return nullptr;
      }
      if (!goal->lambda) {
        Uses(goal, live);
        return goal;
      }
      // A lambda expression's body, whose head variables are used after
      // it.
      std::set<BodyVar> inner{goal->lambdaHead.begin(), goal->lambdaHead.end()};
      const Body *lambda{Simplify(goal->lambda, inner)};
      Body copy{*goal};
      copy.lambda =
          lambda ? lambda : New(Body{BodyKind::Conj, goal->lambda->tokens});
      const Body *out{New(std::move(copy))};
      Uses(out, live);
      return out;
    }
    case BodyKind::Call:
    case BodyKind::HOCall:
      Uses(goal, live);
      return goal;
    case BodyKind::Conj: {
      std::vector<const Body *> goals;
      for (const Body *g : llvm::reverse(goal->goals)) {
        if (const Body *kept{Simplify(g, live)}) {
          goals.insert(goals.begin(), kept);
        }
      }
      if (goals.size() == 1) {
        return goals.front();
      }
      return With(goal, std::move(goals));
    }
    case BodyKind::Disj: {
      if (goal->goals.empty()) {
        Uses(goal, live);
        return goal;
      }
      std::set<BodyVar> after{live};
      std::vector<const Body *> goals;
      for (const Body *g : goal->goals) {
        std::set<BodyVar> inner{after};
        const Body *kept{Simplify(g, inner)};
        if (!kept) {
          kept = New(Body{BodyKind::Conj, g->tokens});
        }
        // A branch that is fail is deleted, with the uses in it.
        if (kept->kind == BodyKind::Disj && kept->goals.empty()) {
          continue;
        }
        live.insert(inner.begin(), inner.end());
        if (kept->kind == BodyKind::Disj) {
          llvm::append_range(goals, kept->goals);
        } else {
          goals.push_back(kept);
        }
      }
      if (goals.size() == 1 && !goal->goals.empty()) {
        return goals.front();
      }
      if (goals.empty()) {
        // A fail in mmc that uses no variables.
        return New(Body{BodyKind::Disj, goal->tokens});
      }
      return With(goal, std::move(goals));
    }
    case BodyKind::Switch: {
      std::set<BodyVar> after{live};
      Body copy{*goal};
      bool added{switching_.insert(goal->switchVar).second};
      for (SwitchCase &c : copy.cases) {
        std::set<BodyVar> inner{after};
        const Body *kept{Simplify(c.goal, inner)};
        c.goal = kept ? kept : New(Body{BodyKind::Conj, c.goal->tokens});
        live.insert(inner.begin(), inner.end());
      }
      if (added) {
        switching_.erase(goal->switchVar);
      }
      use(goal->switchVar);
      return New(std::move(copy));
    }
    case BodyKind::IfThenElse: {
      std::set<BodyVar> after{live};
      std::set<BodyVar> otherwise{after};
      const Body *e{Simplify(goal->goals[2], otherwise)};
      std::set<BodyVar> then{after};
      const Body *t{Simplify(goal->goals[1], then)};
      const Body *c{Simplify(goal->goals[0], then)};
      live.insert(then.begin(), then.end());
      live.insert(otherwise.begin(), otherwise.end());
      auto orTrue{[&](const Body *g, const Body *old) {
        return g ? g : New(Body{BodyKind::Conj, old->tokens});
      }};
      return With(goal, {orTrue(c, goal->goals[0]), orTrue(t, goal->goals[1]),
                            orTrue(e, goal->goals[2])});
    }
    default: {
      // A negation or a scope.
      const Body *inner{Simplify(goal->goals.front(), live)};
      return With(goal,
          {inner ? inner : New(Body{BodyKind::Conj, goal->goals.front()->tokens})});
    }
    }
  }

  // The variables a goal uses.
  static void Uses(const Body *goal, std::set<BodyVar> &live) {
    switch (goal->kind) {
    case BodyKind::Unify:
      live.insert(goal->left);
      if (goal->isVar) {
        live.insert(goal->var);
      }
      live.insert(goal->arguments.begin(), goal->arguments.end());
      if (goal->lambda) {
        Uses(goal->lambda, live);
      }
      return;
    case BodyKind::HOCall:
      live.insert(goal->closure);
      [[fallthrough]];
    case BodyKind::Call:
      live.insert(goal->arguments.begin(), goal->arguments.end());
      return;
    case BodyKind::Switch:
      live.insert(goal->switchVar);
      for (const SwitchCase &c : goal->cases) {
        Uses(c.goal, live);
      }
      return;
    default:
      for (const Body *g : goal->goals) {
        Uses(g, live);
      }
      if (goal->kind == BodyKind::Disj && goal->goals.empty()) {
        for (const auto &[var, inst] : goal->entry) {
          live.insert(var);
        }
      }
    }
  }

  DeterminismChecking &result_;
  ProcedureBody &body_;
  InstOps &ops_;
  bool changed_{false};
  std::map<BodyVar, InstFunctor> cased_;
  // The goals after a disjunction of tests, from Tests.
  std::set<const Body *> merged_;
  // The variables of the switches that Simplify is in.
  std::set<BodyVar> switching_;
};

std::unique_ptr<DeterminismChecking> CheckDeterminism(
    const parser::TokenSequence &, TypeChecking &, ModeChecking &modes,
    parser::Messages &) {
  auto result{std::make_unique<DeterminismChecking>()};
  auto add{[&](const Procedure &procedure, const MethodModes *method,
               llvm::ArrayRef<ClauseModes> clauses) {
    if (clauses.empty()) {
      return;
    }
    ProcedureBody &body{result->bodies_.emplace_back()};
    body.procedure = &procedure;
    body.method = method;
    BodyBuilder{*result, body, modes.ops()}.Build(procedure, clauses);
    SwitchDetector{*result, body, modes.ops()}.Run();
    if (!method) {
      result->byProcedure_[&procedure] = &body;
    }
  }};
  for (const Procedure &procedure : modes.procedures()) {
    add(procedure, nullptr, modes.ClausesOf(procedure));
  }
  for (const Procedure &procedure : modes.inferred()) {
    add(procedure, nullptr, modes.ClausesOf(procedure));
  }
  for (const MethodModes &method : modes.methods()) {
    add(*method.procedure, &method, method.clauses);
  }
  return result;
}

void DumpProcedureBody(llvm::raw_ostream &os, const ProcedureBody &body,
    const parser::TokenSequence &tokens,
    llvm::function_ref<std::string(parser::CharBlock)> position) {
  auto name{[&](BodyVar var) { return body.vars[var].name; }};
  auto names{[&](llvm::ArrayRef<BodyVar> vars) {
    std::string text;
    llvm::raw_string_ostream s{text};
    llvm::interleave(vars, s, [&](BodyVar v) { s << name(v); }, ", ");
    return text;
  }};
  auto functor{[](const InstFunctor &f) {
    std::string text;
    if (!f.module.empty()) {
      text = f.module.str() + '.';
    }
    text += f.name.str();
    if (!f.literal) {
      text += '/' + std::to_string(f.arity);
    }
    return text;
  }};
  std::function<void(const Body *, unsigned)> dump{[&](const Body *goal,
                                                      unsigned depth) {
    std::string indent(2 * depth + 4, ' ');
    std::string at{position(tokens.tokens[goal->tokens.first].spelling)};
    switch (goal->kind) {
    case BodyKind::Conj:
      if (goal->goals.empty()) {
        os << indent << at << " true\n";
      }
      for (const Body *g : goal->goals) {
        dump(g, depth);
      }
      return;
    case BodyKind::Disj:
      os << indent << at << (goal->goals.empty() ? " fail\n" : " disjunction\n");
      for (const Body *g : goal->goals) {
        os << indent << "  branch\n";
        dump(g, depth + 2);
      }
      return;
    case BodyKind::Switch:
      os << indent << at << " switch on " << name(goal->switchVar)
         << (goal->canFail ? ", can fail\n" : ", cannot fail\n");
      for (const SwitchCase &c : goal->cases) {
        os << indent << "  case " << functor(c.functor);
        for (const InstFunctor &other : c.others) {
          os << " or " << functor(other);
        }
        os << '\n';
        dump(c.goal, depth + 2);
      }
      return;
    case BodyKind::IfThenElse:
      os << indent << at << " if-then-else\n";
      for (std::size_t i{0}; i < 3; ++i) {
        os << indent << "  " << (i == 0 ? "if" : i == 1 ? "then" : "else")
           << '\n';
        dump(goal->goals[i], depth + 2);
      }
      return;
    case BodyKind::Not:
      os << indent << at << " not\n";
      dump(goal->goals.front(), depth + 1);
      return;
    case BodyKind::Scope:
      os << indent << at << ' '
         << llvm::StringRef{EnumToString(goal->scope)}.lower() << '\n';
      dump(goal->goals.front(), depth + 1);
      return;
    case BodyKind::Call:
      os << indent << at << " call " << *goal->modes->callee->symbol << " mode "
         << goal->modes->callee->number << (goal->modes->implied ? " implied" : "")
         << ": " << goal->call->name->name << '(' << names(goal->arguments)
         << ")\n";
      return;
    case BodyKind::HOCall: {
      llvm::ArrayRef<BodyVar> arguments{goal->arguments};
      os << indent << at << " higher-order call: ";
      if (goal->hoCall->kind == parser::PredOrFunc::Function) {
        os << name(arguments.back()) << " = ";
        arguments = arguments.drop_back();
      }
      os << name(goal->closure) << '(' << names(arguments) << ")\n";
      return;
    }
    case BodyKind::Unify:
      os << indent << at << ' ';
      if (goal->isVar) {
        if (goal->unify == UnifyKind::Assign) {
          os << "assign " << name(goal->left) << " := " << name(goal->var);
        } else {
          os << "test " << name(goal->left) << " = " << name(goal->var);
        }
      } else if (goal->unify == UnifyKind::Call) {
        os << "call " << *goal->modes->callee->symbol << " mode "
           << goal->modes->callee->number
           << (goal->modes->implied ? " implied" : "") << ": "
           << name(goal->left) << " = "
           << goal->source->functor.name->name << '(' << names(goal->arguments)
           << ')';
      } else if (goal->lambda || goal->source->functor.kind == FunctorKind::Coerce ||
          goal->modes->callee) {
        os << llvm::StringRef{EnumToString(goal->unify)}.lower() << ' '
           << name(goal->left) << " = "
           << (goal->lambda ? "lambda"
                   : goal->modes->callee ? "closure"
                                         : "coerce")
           << '(' << names(goal->arguments) << ')';
      } else {
        os << llvm::StringRef{EnumToString(goal->unify)}.lower() << ' '
           << name(goal->left) << " = " << functor(goal->functor);
        if (!goal->arguments.empty()) {
          os << '(' << names(goal->arguments) << ')';
        }
        if (goal->switchTest) {
          os << ", switched on";
        }
      }
      os << '\n';
      if (goal->lambda) {
        dump(goal->lambda, depth + 1);
      }
      return;
    }
  }};
  dump(body.body, 0);
}

} // namespace merkurius::semantics
