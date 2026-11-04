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
#include <limits>
#include <map>
#include <optional>
#include <set>

namespace merkurius::semantics {

using namespace parser::literals;

namespace {

// The goals of a conjunction, or the goal itself.
std::vector<const Body *> Conjuncts(const Body *goal) {
  // A true that keeps the variables of the goals it replaces is one.
  if (goal->kind == BodyKind::Conj &&
      !(goal->goals.empty() && !goal->failed.empty())) {
    return goal->goals;
  }
  return {goal};
}

// Whether a value of the type with the inst may have a functor that is not
// one of `functors`, or the type has no known set of constructors; with no
// inst, it may.
bool AllowsOthers(InstOps &ops, const Type *type, const Inst *inst,
    llvm::ArrayRef<InstFunctor> functors) {
  if (!inst) {
    return true;
  }
  const std::vector<ConstructorTypes> *constructors{ops.ConstructorsOf(type)};
  if (!constructors) {
    // A value of a type without constructors, such as int, bound to some
    // only, as by a test before.
    const Inst *plain{ops.Plain(inst)};
    return plain->kind != InstKind::Bound ||
        llvm::any_of(plain->functors, [&](const BoundAlternative &b) {
          return !llvm::is_contained(functors, b.functor);
        });
  }
  for (const ConstructorTypes &c : *constructors) {
    if (!llvm::is_contained(functors, c.functor) &&
        ops.Arguments(inst, c.functor, type)) {
      return true;
    }
  }
  return false;
}

// Whether a goal is true, or a conjunction of trues.
bool IsTrue(const semantics::Goal &goal) {
  const auto *conj{std::get_if<const Conj *>(&goal)};
  return conj && !(*conj)->parallel &&
      llvm::all_of((*conj)->goals,
          [](const semantics::Goal &g) { return IsTrue(g); });
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

  // Whether mode analysis deleted the goal, a construction of a variable
  // not used after it, whose variables mmc's quantification still finds,
  // as for a unification replaced with fail, since mmc quantifies the body
  // again only after deleting an assignment.
  bool DeadConstruction(const semantics::Goal &goal) const {
    const auto *unify{std::get_if<const Unify *>(&goal)};
    if (!unify || dropped_.count(*unify)) {
      return false;
    }
    const GoalModes &modes{clause_->goals[(*unify)->number]};
    return modes.dead && modes.unify != UnifyKind::Assign;
  }

  // A goal, or true if it is left out.
  const Body *Whole(const semantics::Goal &goal) {
    if (const Body *out{Goal(goal)}) {
      return out;
    }
    TokenRange tokens{std::visit([](const auto *g) { return g->tokens; }, goal)};
    // A construction mmc deletes is true, with its variables, which mmc's
    // quantification still finds, since it does not quantify the body
    // again after it.
    if (DeadConstruction(goal)) {
      Body out{BodyKind::Conj, tokens};
      unsigned number{(*std::get_if<const Unify *>(&goal))->number};
      for (VarId var : clause_->clause->clause->nonlocals[number]) {
        out.failed.push_back(map_[var]);
      }
      return New(std::move(out));
    }
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
              // The variables of the goals left out, which mmc's
              // quantification still finds until it runs again, as for a
              // unification replaced with fail.
              std::vector<BodyVar> dropped;
              const ClauseForm &form{*clause_->clause->clause};
              auto drop{[&](const semantics::Goal &g) {
                for (VarId var : form.nonlocals[number(g)]) {
                  dropped.push_back(map_[var]);
                }
              }};
              for (const semantics::Goal &inner :
                  std::get<const Conj *>(goal)->goals) {
                if (!llvm::any_of(modes.order, [&](const semantics::Goal &g) {
                      return number(g) == number(inner);
                    })) {
                  drop(inner);
                }
              }
              for (const semantics::Goal &inner : modes.order) {
                if (failing && number(inner) > number(*failing) &&
                    !(clause_->goals[number(inner)].head &&
                        first(inner) < first(*failing))) {
                  drop(inner);
                  continue;
                }
                const Body *goal{Goal(inner)};
                if (!goal) {
                  if (DeadConstruction(inner)) {
                    drop(inner);
                  }
                  continue;
                }
                for (const Body *g : Conjuncts(goal)) {
                  conj.goals.push_back(g);
                }
              }
              if (!dropped.empty() && !conj.goals.empty()) {
                Body last{*conj.goals.back()};
                llvm::append_range(last.failed, dropped);
                conj.goals.back() = New(std::move(last));
              } else if (!dropped.empty()) {
                conj.failed = std::move(dropped);
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
              // Mode analysis does not run the then part after a condition
              // whose end it does not reach, which mmc leaves out.
              const Body *condition{Whole(ite->condition)};
              const Body *then{Whole(ite->then)};
              if (!Reached(condition)) {
                then = New(Body{BodyKind::Conj, then->tokens});
              }
              out.goals = {condition, then, Whole(ite->otherwise)};
              return New(std::move(out));
            },
            [&](const Not *negation) {
              // mmc takes the negation of true, or of a conjunction of
              // trues, as fail.
              if (IsTrue(negation->goal)) {
                return New(Body{BodyKind::Disj, tokens, &modes});
              }
              Body out{BodyKind::Not, tokens, &modes};
              out.goals = {Whole(negation->goal)};
              return New(std::move(out));
            },
            [&](const Scope *scope) {
              Body out{BodyKind::Scope, tokens, &modes};
              out.scope = scope->kind;
              if (scope->kind == ScopeKind::Omp) {
                out.directive = std::get<const parser::OmpGoal *>(
                    std::get<parser::Goal>(scope->origin))->directive;
                for (VarId var :
                    clause_->clause->clause->nonlocals[scope->number]) {
                  out.nonlocals.push_back(map_[var]);
                }
              }
              if (scope->kind == ScopeKind::PromiseSolutions) {
                const parser::Goal &origin{std::get<parser::Goal>(scope->origin)};
                out.promise = std::get<const parser::PromiseSolutions *>(
                    origin)->kind;
                for (VarId var : scope->vars) {
                  out.promised.push_back(map_[var]);
                }
                out.entry = Before(number);
              } else if (scope->kind == ScopeKind::RequireDeterminism) {
                const parser::Goal &origin{std::get<parser::Goal>(scope->origin)};
                out.required = std::get<const parser::RequireDeterminism *>(
                    origin)->determinism;
              } else if (scope->kind == ScopeKind::RequireSwitch) {
                const parser::Goal &origin{std::get<parser::Goal>(scope->origin)};
                out.required = std::get<const parser::RequireSwitch *>(origin)
                                   ->armDeterminism;
                if (!scope->vars.empty()) {
                  out.switchOn = map_[scope->vars.front()];
                }
              }
              out.goals = {Whole(scope->goal)};
              return New(std::move(out));
            },
            [&](const Unify *unify) -> const Body * {
              // As in mmc, which deletes a construction or assignment of a
              // variable not used after it, and replaces a unification
              // that cannot succeed with fail.
              if (dropped_.count(unify)) {
                return nullptr;
              }
              if (modes.dead) {
                if (modes.unify == UnifyKind::Assign) {
                  out_.requantified = true;
                }
                return nullptr;
              }
              if (modes.fails) {
                Body out{BodyKind::Disj, tokens, &modes};
                out.failed.push_back(map_[unify->left]);
                if (unify->isVar) {
                  out.failed.push_back(map_[unify->var]);
                }
                for (VarId var : unify->arguments) {
                  out.failed.push_back(map_[var]);
                }
                return New(std::move(out));
              }
              return Unification(*unify, modes, tokens);
            },
            [&](const Call *call) {
              Body out{BodyKind::Call, tokens, &modes};
              out.call = call;
              for (VarId var : call->arguments) {
                out.arguments.push_back(map_[var]);
              }
              out.entry = Before(number);
              return New(std::move(out));
            },
            [&](const HOCall *call) {
              Body out{BodyKind::HOCall, tokens, &modes};
              out.hoCall = call;
              out.closure = map_[call->closure];
              for (VarId var : call->arguments) {
                out.arguments.push_back(map_[var]);
              }
              out.entry = Before(number);
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
    if (modes.unify == UnifyKind::Call) {
      out.entry = Before(unify.number);
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
    auto x{llvm::find_if(before, [&](const auto &b) { return b.first == out.left; })};
    out.canFail = AllowsOthers(ops_, TypeOf(out.left),
        x == before.end() ? nullptr : x->second, {out.functor});
    Body conj{BodyKind::Conj, tokens, &modes};
    std::vector<Body> tests;
    // A bound argument is a test, unless X's argument is free, which it
    // binds, as in mmc.
    std::optional<std::vector<const Inst *>> fields;
    if (x != before.end() && x->second) {
      fields = ops_.Arguments(x->second, out.functor, TypeOf(out.left));
    }
    for (std::size_t i{0}; i < out.arguments.size(); ++i) {
      BodyVar &argument{out.arguments[i]};
      auto it{llvm::find_if(
          before, [&](const auto &b) { return b.first == argument; })};
      if (it == before.end() || ops_.IsFree(it->second) ||
          (fields && i < fields->size() && ops_.IsFree((*fields)[i]))) {
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
      body_.rechecked = true;
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

  // A conjunction with its goals rebuilt; a true, which may keep the
  // variables of the goals it replaces, as it is.
  const Body *Rebuilt(
      const Body *goal, const std::vector<const Body *> &goals) {
    return goal->goals.empty() ? goal : Conj(goal->tokens, goals);
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
        return Rebuilt(goal, goals);
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
        const Body *merged{goals.front()};
        auto it{llvm::find_if(out.cases,
            [&](const SwitchCase &c) { return c.goal == merged; })};
        if (it != out.cases.end()) {
          it->others.push_back(functor);
          continue;
        }
        // Detected when the case is complete.
        out.cases.push_back({functor, goals.front()});
        continue;
      }
      Cased cased{*this, var, functor};
      // A case's disjunction is where its first disjunct in the source
      // is, as in mmc's messages.
      TokenRange range{std::numeric_limits<std::uint32_t>::max(), 0};
      for (const Body *g : goals) {
        range.first = std::min(range.first, g->tokens.first);
        range.last = std::max(range.last, g->tokens.last);
      }
      out.cases.push_back({functor,
          goals.size() == 1 ? Detect(goals.front())
                            : Disjunction(range, goals, entry)});
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
    const Inst *inst{nullptr};
    for (const auto &[v, i] : entry) {
      if (v == var) {
        inst = i;
      }
    }
    return AllowsOthers(ops_, body_.vars[var].type, inst, functors);
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
  // goals in scopes, as in mmc. A disjunction may come before it if
  // each of its branches, followed by the goals after it, has such a test;
  // the disjunct is then split, one for each test, with those goals in
  // each (reference manual, "Switches"). None if a goal of another kind
  // comes first; empty if the goals end first, with the variables found to
  // be unified with `var` in `aliases`.
  std::optional<Found> Scan(std::vector<const Body *> goals, BodyVar var,
      std::set<BodyVar> aliases, bool split, std::set<BodyVar> *out = nullptr) {
    // The aliases not made in a scope, which are the only ones a
    // disjunction is split on in mmc.
    std::set<BodyVar> unscoped{aliases};
    // Whether a test of a case's own variable against a constant came
    // before, which is `true` in mmc, so that a disjunction after it is
    // not split; one through a variable unified with it is kept.
    bool truth{false};
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
          truth |= cased_.count(goal->left) != 0;
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
        // mmc looks into every scope, as a promise or a trace goal.
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
        if (!split || truth || goal->goals.empty()) {
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
      bool added{switching_.insert(goal->switchVar).second};
      for (SwitchCase &c : copy.cases) {
        c.goal = Eliminate(c.goal);
      }
      if (added) {
        switching_.erase(goal->switchVar);
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
        return Rebuilt(goal, goals);
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
      // In a case of a switch on the variable, it cannot fail.
      pulled.canFail = !switching_.count(var) &&
          CanFail(var, {first->functor}, disj->entry);
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
  // scopes, as Scan finds it; `stopped` is set if a goal of another kind
  // comes first.
  const Body *FirstTest(const std::vector<const Body *> &goals,
      std::set<BodyVar> aliases, bool *stopped = nullptr) {
    for (const Body *goal : goals) {
      if (goal->kind == BodyKind::Scope) {
        bool inner{false};
        if (const Body *test{
                FirstTest(Conjuncts(goal->goals.front()), aliases, &inner)}) {
          return test;
        }
        if (inner) {
          return Stop(stopped);
        }
        // The aliases it makes are local to it but still aliases.
        continue;
      }
      if (goal->kind != BodyKind::Unify || goal->unify == UnifyKind::Call) {
        return Stop(stopped);
      }
      // A test of a switch's variable in its case against a constant is
      // true in mmc; one through a variable unified with it is kept, and
      // can be moved out.
      if (goal->switchTest && goal->arguments.empty() &&
          switching_.count(goal->left)) {
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
  static const Body *Stop(bool *stopped) {
    if (stopped) {
      *stopped = true;
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
    return Rebuilt(goal, goals);
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
      // A conjunction in it, as a disjunction left with one branch, is
      // part of it, as in mmc.
      std::vector<const Body *> goals;
      for (const Body *g : llvm::reverse(goal->goals)) {
        if (const Body *kept{Simplify(g, live)}) {
          std::vector<const Body *> parts{Conjuncts(kept)};
          goals.insert(goals.begin(), parts.begin(), parts.end());
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
  // The variables of the switches that Simplify or Eliminate is in.
  std::set<BodyVar> switching_;
};

// The two components of a determinism (reference manual, "Determinism
// categories"): whether it can fail before its first solution, and how
// many solutions it can have at most.
enum class Solutions { Zero, One, Many };

struct Components {
  bool canFail;
  Solutions most;
};

Components ComponentsOf(parser::Determinism determinism) {
  switch (determinism) {
  case parser::Determinism::Det:
    return {false, Solutions::One};
  case parser::Determinism::Semidet:
    return {true, Solutions::One};
  case parser::Determinism::Multi:
  case parser::Determinism::CcMulti:
    return {false, Solutions::Many};
  case parser::Determinism::Nondet:
  case parser::Determinism::CcNondet:
    return {true, Solutions::Many};
  case parser::Determinism::Failure:
    return {true, Solutions::Zero};
  case parser::Determinism::Erroneous:
    return {false, Solutions::Zero};
  }
  return {true, Solutions::Many};
}

parser::Determinism DeterminismOf(Components c) {
  switch (c.most) {
  case Solutions::Zero:
    return c.canFail ? parser::Determinism::Failure : parser::Determinism::Erroneous;
  case Solutions::One:
    return c.canFail ? parser::Determinism::Semidet : parser::Determinism::Det;
  case Solutions::Many:
    return c.canFail ? parser::Determinism::Nondet : parser::Determinism::Multi;
  }
  return parser::Determinism::Nondet;
}

bool IsCommittedChoice(parser::Determinism determinism) {
  return determinism == parser::Determinism::CcMulti ||
      determinism == parser::Determinism::CcNondet;
}

// A goal's determinism, which in a context where only its first solution
// is wanted is the committed choice one if it has more than one.
parser::Determinism DeterminismOf(Components c, bool committed) {
  if (committed && c.most == Solutions::Many) {
    return c.canFail ? parser::Determinism::CcNondet
                     : parser::Determinism::CcMulti;
  }
  return DeterminismOf(c);
}

// Whether `a` is at least as tight as `b` in the lattice of determinisms.
bool Within(Components a, Components b) {
  return (!a.canFail || b.canFail) && a.most <= b.most;
}

// (A, B) can fail if A can, or if A can succeed and B can fail; it has no
// solutions if either has none, and more than one if either has more and
// neither has none (reference manual, "Determinism checking and
// inference"). det is its identity.
Components Conjunction(Components a, Components b) {
  Solutions most{a.most == Solutions::Zero || b.most == Solutions::Zero
          ? Solutions::Zero
          : a.most == Solutions::Many || b.most == Solutions::Many
          ? Solutions::Many
          : Solutions::One};
  return {a.canFail || (a.most != Solutions::Zero && b.canFail), most};
}

// (A ; B) can fail if both can; it has more than one solution if either
// has, or both have one. failure is its identity.
Components Disjunction(Components a, Components b) {
  Solutions most{a.most == Solutions::Zero ? b.most
          : b.most == Solutions::Zero      ? a.most
                                           : Solutions::Many};
  return {a.canFail && b.canFail, most};
}

// At most one arm of a switch runs: it can fail if any arm can, and has as
// many solutions as the arm with most (Henderson, Somogyi and Conway,
// section 5.1). erroneous is its identity.
Components Switch(Components a, Components b) {
  return {a.canFail || b.canFail, std::max(a.most, b.most)};
}

// Whether mode analysis reaches the end of a goal, as in mmc: not after a
// unification that cannot succeed, which is fail, or a call to a
// procedure declared to have no solutions, since mode analysis comes
// before inference; a negation always succeeds.
bool Reached(const Body *goal) {
  switch (goal->kind) {
  case BodyKind::Unify:
    return goal->unify != UnifyKind::Call ||
        !goal->modes->callee->determinism ||
        ComponentsOf(*goal->modes->callee->determinism).most != Solutions::Zero;
  case BodyKind::Call:
    return !goal->modes->callee->determinism ||
        ComponentsOf(*goal->modes->callee->determinism).most != Solutions::Zero;
  case BodyKind::HOCall:
    return ComponentsOf(goal->modes->determinism).most != Solutions::Zero;
  case BodyKind::Conj:
    return llvm::all_of(goal->goals, [&](const Body *g) { return Reached(g); });
  case BodyKind::Disj:
    return llvm::any_of(goal->goals, [&](const Body *g) { return Reached(g); });
  case BodyKind::Switch:
    return llvm::any_of(
        goal->cases, [&](const SwitchCase &c) { return Reached(c.goal); });
  case BodyKind::IfThenElse:
    return (Reached(goal->goals[0]) && Reached(goal->goals[1])) ||
        Reached(goal->goals[2]);
  case BodyKind::Not:
    return true;
  case BodyKind::Scope:
    return Reached(goal->goals.front());
  }
  return true;
}

// Infers the determinism of each goal of a procedure's body, from its
// parts' (reference manual, "Determinism checking and inference";
// Henderson, Somogyi and Conway, section 5.1), checks it against the
// declared determinism, and if it is not as tight finds the goals that
// make the difference (ibid., section 5.3).
class DeterminismAnalyser {
public:
  DeterminismAnalyser(const DeterminismChecking &checking, ProcedureBody &body,
      TypeChecking &types, InstOps &ops, const parser::TokenSequence &tokens,
      parser::Messages &messages)
      : checking_{checking}, body_{body}, types_{types}, ops_{ops},
        tokens_{tokens}, messages_{messages} {}

  void Run() {
    std::set<BodyVar> bound;
    for (std::size_t i{0}; i < body_.head.size(); ++i) {
      if (!ops_.IsFree(body_.procedure->modes[i].initial)) {
        bound.insert(body_.head[i]);
      }
    }
    Components d{Infer(body_.body, bound,
        std::set<BodyVar>{body_.head.begin(), body_.head.end()})
                     .d};
    if (body_.unchecked != DeterminismUnchecked::None) {
      body_.goals.clear();
      return;
    }
    // The body of a committed choice procedure is in a context where only
    // its first solution is wanted (reference manual, "Committed choice
    // nondeterminism").
    std::optional<parser::Determinism> declared{body_.procedure->determinism};
    bool committed{declared && IsCommittedChoice(*declared)};
    Contexts(body_.body, {committed, {}});
    Requirements(body_.body);
    for (auto &[goal, determinism] : body_.goals) {
      determinism = DeterminismOf(ComponentsOf(determinism), first_.count(goal));
    }
    body_.determinism = DeterminismOf(d, committed);
    body_.unseen = unseen_;
    Check();
  }

private:
  struct Inferred {
    Components d;
    bool impure{false};
  };
  // A goal that makes a determinism not as tight as declared, and why.
  struct Cause {
    TokenRange tokens;
    std::string text;
  };
  // Whether only the first solution of a goal is wanted, and if not
  // because goals after it can fail, those goals.
  struct Context {
    bool first;
    std::vector<const Body *> failing;
  };

  // The occurrences of each variable in a goal, each goal counted once
  // though a merged case's goal is in several cases.
  void Count(const Body *goal, std::map<BodyVar, unsigned> &counts,
      std::set<const Body *> &visited) {
    if (!visited.insert(goal).second) {
      return;
    }
    auto add{[&](BodyVar var) { ++counts[var]; }};
    // Those of the goals deleted or replaced with fail, until mmc
    // quantifies the body again.
    if (!body_.rechecked && !body_.requantified) {
      for (BodyVar var : Sorted(goal->failed)) {
        add(var);
      }
    }
    switch (goal->kind) {
    case BodyKind::Unify:
      add(goal->left);
      if (goal->isVar) {
        add(goal->var);
      }
      llvm::for_each(goal->arguments, add);
      if (goal->lambda) {
        llvm::for_each(goal->lambdaHead, add);
        Count(goal->lambda, counts, visited);
      }
      return;
    case BodyKind::Call:
      llvm::for_each(goal->arguments, add);
      return;
    case BodyKind::HOCall:
      add(goal->closure);
      llvm::for_each(goal->arguments, add);
      return;
    case BodyKind::Switch:
      add(goal->switchVar);
      for (const SwitchCase &c : goal->cases) {
        Count(c.goal, counts, visited);
      }
      return;
    default:
      for (const Body *g : goal->goals) {
        Count(g, counts, visited);
      }
    }
  }

  // The variables of a goal.
  const std::set<BodyVar> &Vars(const Body *goal) {
    auto [it, inserted]{vars_.try_emplace(goal)};
    if (inserted) {
      std::map<BodyVar, unsigned> counts;
      std::set<const Body *> visited;
      Count(goal, counts, visited);
      for (const auto &entry : counts) {
        it->second.insert(entry.first);
      }
    }
    return it->second;
  }

  // Whether a goal binds a variable that is used outside it, which makes
  // its solutions different (reference manual, "Interfacing
  // nondeterministic code with the real world"). `outside` has the
  // variables used outside the goal, as mmc's quantification finds them
  // after switch detection: in a branch, those of its parent; in a
  // conjunct, those and the other conjuncts'.
  bool HasOutputs(const Body *goal, const std::set<BodyVar> &before,
      const std::set<BodyVar> &outside) {
    return llvm::any_of(Vars(goal), [&](BodyVar var) {
      return !before.count(var) && outside.count(var);
    });
  }

  // A callee's determinism, declared or inferred; none for one whose modes
  // were not checked.
  std::optional<parser::Determinism> Of(const Procedure &callee) const {
    return checking_.DeterminismOf(callee);
  }

  // A call in an implied mode tests the argument that is already bound
  // against the callee's output, so it can fail.
  static Components Implied(Components components, const GoalModes &modes) {
    components.canFail = components.canFail || modes.implied;
    return components;
  }

  Components Callee(const Procedure &callee) {
    std::optional<parser::Determinism> determinism{Of(callee)};
    if (!determinism) {
      if (body_.unchecked == DeterminismUnchecked::None) {
        body_.unchecked = DeterminismUnchecked::UncheckedCallee;
      }
      return {true, Solutions::Many};
    }
    return ComponentsOf(*determinism);
  }

  bool Impure(const Symbol &symbol) const {
    const Signature *sig{types_.SignatureOf(symbol)};
    return sig && sig->purity == parser::Purity::Impure;
  }

  // The determinism of a goal; the variables it binds are added to
  // `bound`. A goal that can succeed more than once but binds no variable
  // used outside it, and is not impure, succeeds at most once, as its
  // solutions are the same to the rest (ibid.). As in mmc for a
  // disjunction, this goes for the innermost such goal; mmc infers a
  // conjunct other than a disjunction in a context where only its first
  // solution is wanted, so the disjunctions inside it keep more than one
  // solution, but this changes no procedure's determinism (see the design
  // document's known differences).
  Inferred Infer(const Body *goal, std::set<BodyVar> &bound,
      const std::set<BodyVar> &outside) {
    std::set<BodyVar> before{bound};
    Inferred r{InferGoal(goal, bound, outside)};
    if (!r.impure && !HasOutputs(goal, before, outside)) {
      unseen_.insert(goal);
      if (r.d.most == Solutions::Many) {
        r.d.most = Solutions::One;
        pruned_.insert(goal);
      }
    }
    body_.goals[goal] = DeterminismOf(r.d);
    return r;
  }

  // A goal's determinism as the goals around it see it.
  Components Seen(const Body *goal) const {
    auto it{body_.goals.find(goal)};
    if (it == body_.goals.end()) {
      return {false, Solutions::One};
    }
    return ComponentsOf(it->second);
  }

  Inferred InferGoal(const Body *goal, std::set<BodyVar> &bound,
      const std::set<BodyVar> &outside) {
    const Components det{false, Solutions::One};
    switch (goal->kind) {
    case BodyKind::Unify:
      return Unification(goal, bound);
    case BodyKind::Call:
      bound.insert(goal->arguments.begin(), goal->arguments.end());
      return {Implied(Callee(*goal->modes->callee), *goal->modes),
          Impure(*goal->modes->callee->symbol)};
    case BodyKind::HOCall: {
      bound.insert(goal->arguments.begin(), goal->arguments.end());
      return {Implied(ComponentsOf(goal->modes->determinism), *goal->modes),
          goal->hoCall->purity == parser::Purity::Impure};
    }
    case BodyKind::Conj: {
      Inferred r{det};
      // The number of conjuncts that each variable is in.
      std::map<BodyVar, unsigned> in;
      for (const Body *g : goal->goals) {
        for (BodyVar var : Vars(g)) {
          ++in[var];
        }
      }
      // The variables of the unifications replaced with fail are still
      // used, in the other conjuncts and outside a determinism cast in
      // them, until mmc quantifies the body again.
      std::set<BodyVar> failed{failed_};
      if (!body_.rechecked && !body_.requantified) {
        for (const Body *g : goal->goals) {
          failed_.insert(g->failed.begin(), g->failed.end());
        }
      }
      // The constructions and assignments that mmc's mode analysis deletes,
      // of variables not used after them, are already gone; the variables
      // of those left are used, even by one that is not used after a goal
      // deleted later, as in mmc, whose liveness is from before.
      for (const Body *g : goal->goals) {
        std::set<BodyVar> used{outside};
        for (BodyVar var : Vars(g)) {
          if (in[var] > 1) {
            used.insert(var);
          }
        }
        Inferred i{Infer(g, bound, used)};
        r.d = Conjunction(r.d, i.d);
        r.impure |= i.impure;
      }
      failed_ = std::move(failed);
      return r;
    }
    case BodyKind::Disj: {
      Inferred r{{true, Solutions::Zero}};
      std::set<BodyVar> after{bound};
      Known known{*this};
      for (const Body *g : goal->goals) {
        std::set<BodyVar> inner{bound};
        known.Reset();
        Inferred i{Infer(g, inner, outside)};
        r.d = Disjunction(r.d, i.d);
        r.impure |= i.impure;
        after.insert(inner.begin(), inner.end());
      }
      bound = std::move(after);
      return r;
    }
    case BodyKind::Switch: {
      Inferred r{{false, Solutions::Zero}};
      std::set<BodyVar> after{bound};
      Known known{*this};
      for (const SwitchCase &c : goal->cases) {
        std::set<BodyVar> inner{bound};
        known.Reset();
        if (body_.rechecked && c.others.empty()) {
          known_[goal->switchVar] = c.functor;
        }
        Inferred i{Infer(c.goal, inner, outside)};
        r.d = Switch(r.d, i.d);
        r.impure |= i.impure;
        after.insert(inner.begin(), inner.end());
      }
      if (goal->canFail) {
        r.d.canFail = true;
      }
      bound = std::move(after);
      return r;
    }
    case BodyKind::IfThenElse: {
      std::set<BodyVar> then{bound};
      std::set<BodyVar> condition{outside}, thenOutside{outside};
      condition.insert(Vars(goal->goals[1]).begin(), Vars(goal->goals[1]).end());
      thenOutside.insert(Vars(goal->goals[0]).begin(), Vars(goal->goals[0]).end());
      Known known{*this};
      Inferred c{Infer(goal->goals[0], then, condition)};
      Inferred t{Infer(goal->goals[1], then, thenOutside)};
      std::set<BodyVar> otherwise{bound};
      known.Reset();
      Inferred e{Infer(goal->goals[2], otherwise, outside)};
      bool impure{c.impure || t.impure || e.impure};
      bound.insert(then.begin(), then.end());
      // If the condition cannot fail, the else part is not run.
      if (!c.d.canFail) {
        return {Conjunction(c.d, t.d), impure};
      }
      // If it has no solutions, the then part is not run either, and the
      // if-then-else is its else part, as in mmc.
      if (c.d.most == Solutions::Zero) {
        return {e.d, impure};
      }
      Components both{Conjunction(c.d, t.d)};
      return {{t.d.canFail || e.d.canFail, std::max(both.most, e.d.most)},
          impure};
    }
    case BodyKind::Not: {
      std::set<BodyVar> inner{bound};
      Known known{*this};
      Inferred i{Infer(goal->goals.front(), inner, outside)};
      switch (DeterminismOf(i.d)) {
      case parser::Determinism::Erroneous:
        return {{false, Solutions::Zero}, i.impure};
      case parser::Determinism::Failure:
        return {det, i.impure};
      case parser::Determinism::Det:
      case parser::Determinism::Multi:
        return {{true, Solutions::Zero}, i.impure};
      default:
        return {{true, Solutions::One}, i.impure};
      }
    }
    case BodyKind::Scope:
      if (goal->promise) {
        return Cast(goal, bound, outside);
      }
      return Infer(goal->goals.front(), bound, outside);
    }
    return {det};
  }

  Inferred Unification(const Body *goal, std::set<BodyVar> &bound) {
    const Components det{false, Solutions::One};
    bound.insert(goal->left);
    if (goal->isVar) {
      bound.insert(goal->var);
    }
    bound.insert(goal->arguments.begin(), goal->arguments.end());
    if (goal->lambda) {
      Lambda(goal, bound);
      return {det};
    }
    switch (goal->unify) {
    case UnifyKind::Call:
      return {Implied(Callee(*goal->modes->callee), *goal->modes),
          Impure(*goal->modes->callee->symbol)};
    case UnifyKind::Test:
      return {{true, Solutions::One}};
    case UnifyKind::Deconstruct: {
      // In a case of a switch on its variable, as in mmc, which knows its
      // functor there, or after a deconstruction of it with the same
      // functor. One with another functor is fail already where mmc knows
      // that it cannot succeed (SwitchDetector::Prune); elsewhere it goes
      // by the inst, as in mmc.
      auto it{known_.find(goal->left)};
      bool known{goal->switchTest ||
          (it != known_.end() && it->second == goal->functor)};
      if (body_.rechecked) {
        known_[goal->left] = goal->functor;
      }
      return {{!known && goal->canFail, Solutions::One}};
    }
    case UnifyKind::Assign: {
      auto it{known_.find(goal->var)};
      if (it != known_.end()) {
        known_[goal->left] = it->second;
      } else {
        known_.erase(goal->left);
      }
      return {det};
    }
    default:
      return {det};
    }
  }

  // While it lives, the functors known before a branching goal, which
  // each branch starts from and which are known after it. As in mmc, if
  // it checks the modes again after common subexpression elimination
  // (ProcedureBody::rechecked), a case gives the functor of its switch's
  // variable, a deconstruction before, in the same conjunction or around
  // it, its variable's, and an assignment passes one on; a construction
  // does not. Otherwise only the insts that mode analysis found count.
  class Known {
  public:
    explicit Known(DeterminismAnalyser &analyser)
        : analyser_{analyser}, saved_{analyser.known_} {}
    ~Known() { Reset(); }
    void Reset() { analyser_.known_ = saved_; }

  private:
    DeterminismAnalyser &analyser_;
    std::map<BodyVar, InstFunctor> saved_;
  };

  // A determinism cast (reference manual, "Goals"): promise_equivalent_
  // solutions and arbitrary succeed at most once where their goal can
  // succeed more than once; promise_equivalent_solution_sets is as its
  // goal, and the arbitrary scopes in it are the casts. Each lists the
  // variables its goal binds, those with free parts before it that are
  // used outside it, as mmc's quantification finds them after mode
  // analysis, none if its end is not reached; mmc reports those not listed, then those
  // listed but not bound, in the order of the variables. An arbitrary
  // scope must be in a promise_equivalent_solution_sets scope, outside
  // lambda expressions, without its variables, and that scope must not
  // be in another; mmc casts it all the same.
  Inferred Cast(const Body *goal, std::set<BodyVar> &bound,
      const std::set<BodyVar> &outside) {
    using Kind = parser::PromiseSolutionsKind;
    Kind kind{*goal->promise};
    const Body *outer{sets_.empty() ? nullptr : sets_.back()};
    // A scope copied by switch detection is checked once.
    bool check{checked_.insert(goal->tokens.first).second};
    if (check && kind == Kind::Arbitrary) {
      if (!outer) {
        // mmc: "Error: this `arbitrary' scope is not nested inside a
        // `promise_equivalent_solution_sets' scope."
        messages_.Say(Where(goal->tokens),
            "this arbitrary scope is not nested inside a promise_equivalent_solution_sets scope"_err_en_US);
      } else {
        std::vector<BodyVar> both;
        for (BodyVar var : Sorted(goal->promised)) {
          if (llvm::is_contained(outer->promised, var)) {
            both.push_back(var);
          }
        }
        if (!both.empty()) {
          // mmc: "Error: this `arbitrary' scope and the
          // `promise_equivalent_solution_sets' scope it is nested inside
          // overlap on the variable X. This is the outer
          // `promise_equivalent_solution_sets' scope."
          messages_
              .Say(Where(goal->tokens),
                  "this arbitrary scope and the promise_equivalent_solution_sets scope it is nested inside overlap on %s"_err_en_US,
                  both.size() == 1 ? "the variable " + Name(both.front())
                                   : "the following variables: " + Names(both))
              .Attach(Where(outer->tokens),
                  "this is the outer promise_equivalent_solution_sets scope"_en_US);
        }
      }
    }
    if (check && kind == Kind::EquivalentSolutionSets && outer) {
      // mmc: "Error: `promise_equivalent_solution_sets' scope is nested
      // inside another. This is the outer
      // `promise_equivalent_solution_sets' scope."
      messages_
          .Say(Where(goal->tokens),
              "this promise_equivalent_solution_sets scope is nested inside another"_err_en_US)
          .Attach(Where(outer->tokens),
              "this is the outer promise_equivalent_solution_sets scope"_en_US);
    }
    if (kind == Kind::EquivalentSolutionSets) {
      sets_.push_back(goal);
    }
    Inferred r{Infer(goal->goals.front(), bound, outside)};
    if (kind == Kind::EquivalentSolutionSets) {
      sets_.pop_back();
    }
    if (check) {
      std::vector<BodyVar> binds;
      if (Reached(goal->goals.front())) {
        for (const auto &[var, inst] : goal->entry) {
          if (inst && !ops_.IsGround(inst, body_.vars[var].type) &&
              (outside.count(var) || failed_.count(var))) {
            binds.push_back(var);
          }
        }
      }
      std::vector<BodyVar> unlisted;
      for (BodyVar var : Sorted(binds)) {
        if (!llvm::is_contained(goal->promised, var)) {
          unlisted.push_back(var);
        }
      }
      std::vector<BodyVar> extra;
      for (BodyVar var : Sorted(goal->promised)) {
        if (!llvm::is_contained(binds, var)) {
          extra.push_back(var);
        }
      }
      std::string what{kind == Kind::EquivalentSolutions
              ? "promise_equivalent_solutions"
              : kind == Kind::EquivalentSolutionSets
              ? "promise_equivalent_solution_sets"
              : "arbitrary"};
      if (!unlisted.empty()) {
        // mmc: "Error: the `promise_equivalent_solutions' goal binds a
        // variable that is not listed: Y."
        messages_.Say(Where(goal->tokens),
            "the %s goal binds %s: %s"_err_en_US, what,
            unlisted.size() == 1 ? "a variable that is not listed"
                                 : "some variables that are not listed",
            Names(unlisted));
      }
      if (!extra.empty()) {
        // mmc: "Error: the `promise_equivalent_solutions' goal lists an
        // extra variable: X."
        messages_.Say(Where(goal->tokens), "the %s goal lists %s: %s"_err_en_US,
            what,
            extra.size() == 1 ? "an extra variable" : "some extra variables",
            Names(extra));
      }
    }
    if (kind != Kind::EquivalentSolutionSets && r.d.most == Solutions::Many) {
      r.d.most = Solutions::One;
    }
    return r;
  }

  // Variables in their order, once each.
  static std::vector<BodyVar> Sorted(std::vector<BodyVar> vars) {
    llvm::sort(vars);
    vars.erase(std::unique(vars.begin(), vars.end()), vars.end());
    return vars;
  }

  // X, Y and Z.
  std::string Names(const std::vector<BodyVar> &vars) const {
    std::string text;
    for (std::size_t i{0}; i < vars.size(); ++i) {
      if (i > 0) {
        text += i + 1 == vars.size() ? " and " : ", ";
      }
      text += Name(vars[i]);
    }
    return text;
  }

  // A lambda expression's body, which runs when the closure is called,
  // with the arguments that its modes make inputs bound.
  void Lambda(const Body *goal, const std::set<BodyVar> &bound) {
    std::set<BodyVar> inner{bound};
    Known known{*this};
    llvm::ArrayRef<ArgMode> modes{goal->modes->lambda};
    // It is not in the promise_equivalent_solution_sets scopes around it.
    std::vector<const Body *> sets;
    std::swap(sets, sets_);
    for (std::size_t i{0}; i < goal->lambdaHead.size(); ++i) {
      if (i < modes.size() && !ops_.IsFree(modes[i].initial)) {
        inner.insert(goal->lambdaHead[i]);
      }
    }
    Components d{Infer(goal->lambda, inner,
        std::set<BodyVar>{goal->lambdaHead.begin(), goal->lambdaHead.end()})
                     .d};
    std::swap(sets, sets_);
    lambdas_.push_back({goal, d});
  }

  std::string Name(BodyVar var) const { return body_.vars[var].name; }

  parser::CharBlock Where(TokenRange tokens) const {
    return tokens_.tokens[tokens.first].spelling;
  }

  static std::string FunctorText(const InstFunctor &f) {
    std::string text{f.name.str()};
    if (!f.literal) {
      text += '/' + std::to_string(f.arity);
    }
    return text;
  }

  // What a call, or a higher-order call, does beyond what is allowed:
  // with no solutions allowed, as in an erroneous procedure, any is too
  // many.
  static std::string Beyond(bool fails, bool many, Solutions most) {
    std::string succeeds{most == Solutions::Zero ? "can succeed"
                                                 : "can succeed more than once"};
    return fails && many ? "can fail and " + succeeds
        : fails          ? std::string{"can fail"}
                         : succeeds;
  }

  // The goals in `goal` that make it fail, or succeed more than `most`
  // times, beyond what is allowed (Henderson, Somogyi and Conway, section
  // 5.3).
  void Diagnose(const Body *goal, bool canFail, Solutions most,
      const std::string &context, std::vector<Cause> &causes) {
    if (!body_.goals.count(goal)) {
      return;
    }
    Components d{Seen(goal)};
    bool fails{d.canFail && !canFail};
    bool many{d.most > most};
    if (!fails && !many) {
      return;
    }
    // Inside a goal whose solutions are the same to the rest, more than
    // one is allowed, and so inside one that has no more than allowed, as
    // after a goal that cannot succeed, as in mmc; but where none is
    // allowed, the goals that can succeed are still why.
    Solutions inner{pruned_.count(goal) || (!many && most != Solutions::Zero)
            ? Solutions::Many
            : most};
    auto cause{[&](std::string text) {
      causes.push_back({goal->tokens, context + text});
    }};
    switch (goal->kind) {
    case BodyKind::Unify:
      // A test of a case's own constant is true in mmc, in the case of a
      // switch, though not in an arm that a require_switch_arms_<det>
      // scope checks.
      if (goal->lambda ||
          (goal->switchTest && goal->arguments.empty() && !context.empty())) {
        return;
      }
      if (goal->unify == UnifyKind::Call) {
        cause("the call to " + ProcedureToString(*goal->modes->callee) + ' ' +
            Beyond(fails, many, most));
      } else if (goal->isVar) {
        cause("the unification of " + Name(goal->left) + " and " +
            Name(goal->var) + ' ' + Beyond(fails, many, most));
      } else {
        cause("the unification of " + Name(goal->left) + " and " +
            FunctorText(goal->functor) + ' ' + Beyond(fails, many, most));
      }
      return;
    case BodyKind::Call:
      cause("the call to " + ProcedureToString(*goal->modes->callee) + ' ' +
          Beyond(fails, many, most));
      return;
    case BodyKind::HOCall:
      cause(std::string{"the higher-order call "} + Beyond(fails, many, most));
      return;
    case BodyKind::Conj:
      // Not the goals after one whose end mode analysis does not reach, as
      // in mmc.
      for (const Body *g : goal->goals) {
        Diagnose(g, canFail, inner, context, causes);
        if (!Reached(g)) {
          break;
        }
      }
      return;
    case BodyKind::Disj: {
      std::size_t solutions{0};
      for (const Body *g : goal->goals) {
        if (Seen(g).most != Solutions::Zero) {
          ++solutions;
        }
      }
      if (many && solutions > 1) {
        cause("the disjunction has more than one disjunct with solutions");
      }
      // A disjunct's failure counts only if the disjunction can fail.
      for (const Body *g : goal->goals) {
        Diagnose(g, canFail || !d.canFail, inner, context, causes);
      }
      return;
    }
    case BodyKind::Switch: {
      if (fails && goal->canFail) {
        std::vector<std::string> missing{Missing(goal)};
        cause("the switch on " + Name(goal->switchVar) +
            (missing.empty() ? std::string{" can fail"}
                             : " does not cover " + llvm::join(missing, ", ")));
      }
      for (const SwitchCase &c : goal->cases) {
        std::string functors{FunctorText(c.functor)};
        for (const InstFunctor &other : c.others) {
          functors += " or " + FunctorText(other);
        }
        Diagnose(c.goal, canFail, inner,
            context + "in the case " + functors + " of the switch on " +
                Name(goal->switchVar) + ": ",
            causes);
      }
      return;
    }
    case BodyKind::IfThenElse: {
      bool condition{Seen(goal->goals[0]).canFail};
      if (condition && Seen(goal->goals[0]).most == Solutions::Zero) {
        // Only the else part is run.
        Diagnose(goal->goals[2], canFail, inner, context, causes);
        return;
      }
      Diagnose(goal->goals[0], canFail || condition, inner, context, causes);
      Diagnose(goal->goals[1], canFail, inner, context, causes);
      if (condition) {
        Diagnose(goal->goals[2], canFail, inner, context, causes);
      }
      return;
    }
    case BodyKind::Not:
      if (fails) {
        cause("the negated goal can succeed");
      }
      // Where no solution is allowed, as mmc puts it.
      if (many) {
        cause("the negated goal can fail");
      }
      return;
    case BodyKind::Scope:
      // Inside a determinism cast whose goal can succeed more than once,
      // as in mmc, only what makes it fail.
      if (goal->promise &&
          *goal->promise != parser::PromiseSolutionsKind::EquivalentSolutionSets &&
          Seen(goal->goals.front()).most == Solutions::Many) {
        inner = Solutions::Many;
      }
      Diagnose(goal->goals.front(), canFail, inner, context, causes);
      return;
    }
  }

  // Whether the procedure has I/O state arguments, as mmc has it: exactly
  // two of type io.state, one bound at the start and one free.
  bool HasIOState() const {
    const Procedure &procedure{*body_.procedure};
    int in{0}, out{0};
    for (std::size_t i{0}; i < body_.head.size() && i < procedure.modes.size();
         ++i) {
      const Type *type{body_.vars[body_.head[i]].type};
      if (type && type->kind == TypeKind::Named && type->arguments.empty() &&
          type->constructor->module == "io" &&
          type->constructor->name == "state") {
        ++(ops_.IsFree(procedure.modes[i].initial) ? out : in);
      }
    }
    return in == 1 && out == 1;
  }

  // Where the procedure's determinism is declared, and its name in
  // messages.
  // mmc reports an instance method's at its definition in the instance,
  // since its procedure is one made for it there.
  parser::CharBlock Declaration() const {
    if (body_.method && !body_.method->clauses.empty()) {
      return Where(body_.method->clauses.front().clause->clause->tokens);
    }
    const Procedure &procedure{*body_.procedure};
    return procedure.source.empty() ? procedure.symbol->source : procedure.source;
  }

  std::string Text() const {
    if (body_.method) {
      return "the method " + ProcedureToString(*body_.procedure) +
          " of instance " + body_.method->method->instanceText;
    }
    return ProcedureToString(*body_.procedure);
  }

  void Report(parser::Message &message, const Body *goal,
      parser::Determinism declared) {
    std::vector<Cause> causes;
    Components d{ComponentsOf(declared)};
    Diagnose(goal, d.canFail, d.most, "", causes);
    for (const Cause &c : causes) {
      message.Attach(Where(c.tokens), "%s"_because_en_US, c.text);
    }
  }

  // The functors of its variable's type that a switch has no case for.
  std::vector<std::string> Missing(const Body *goal) {
    std::vector<std::string> missing;
    const Type *type{body_.vars[goal->switchVar].type};
    if (const std::vector<ConstructorTypes> *constructors{
            ops_.ConstructorsOf(type)}) {
      for (const ConstructorTypes &c : *constructors) {
        if (llvm::none_of(goal->cases,
                [&](const SwitchCase &s) { return s.Has(c.functor); })) {
          missing.push_back(FunctorText(c.functor));
        }
      }
    }
    return missing;
  }

  // Checks the require_<det>, require_complete_switch and
  // require_switch_arms_<det> scopes (reference manual, "Goals"), each
  // once though switch detection copies it, as mmc does.
  void Requirements(const Body *goal) {
    switch (goal->kind) {
    case BodyKind::Unify:
      if (goal->lambda) {
        Requirements(goal->lambda);
      }
      return;
    case BodyKind::Switch:
      for (const SwitchCase &c : goal->cases) {
        Requirements(c.goal);
      }
      return;
    case BodyKind::Scope:
      if ((goal->required || goal->switchOn) &&
          required_.insert(goal->tokens.first).second) {
        Require(goal);
      }
      [[fallthrough]];
    default:
      for (const Body *g : goal->goals) {
        Requirements(g);
      }
    }
  }

  void Require(const Body *goal) {
    // A goal with more than one solution is committed choice where only
    // its first solution is wanted, as in mmc.
    bool first{requiredFirst_[goal]};
    const Body *inner{goal->goals.front()};
    if (goal->scope == ScopeKind::RequireDeterminism) {
      parser::Determinism actual{DeterminismOf(Seen(inner), first)};
      // mmc: "Error: the required determinism of the goal in this scope is
      // `det', but its actual determinism is `semidet'."
      if (actual != *goal->required) {
        auto &message{messages_.Say(Where(goal->tokens),
            "the goal in this scope is required to be %s, but its determinism is %s"_err_en_US,
            DeterminismName(*goal->required), DeterminismName(actual))};
        Report(message, inner, *goal->required);
      }
      return;
    }
    if (!goal->switchOn) {
      return;
    }
    while (inner->kind == BodyKind::Conj && inner->goals.size() == 1) {
      inner = inner->goals.front();
    }
    std::string var{Name(*goal->switchOn)};
    // mmc checks the arms of a require_switch_arms_<det> scope's switch on
    // another variable as if it were on this one (see the register).
    if (inner->kind != BodyKind::Switch ||
        (inner->switchVar != *goal->switchOn && !goal->required)) {
      // mmc: "Error: the goal inside the require_complete_switch [X] scope
      // is not a switch on `X'."
      messages_.Say(Where(goal->tokens),
          "the goal inside the %s [%s] scope is not a switch on %s"_err_en_US,
          goal->required ? std::string{"require_switch_arms_"} +
                  DeterminismName(*goal->required)
                         : std::string{"require_complete_switch"},
          var, var);
      return;
    }
    if (!goal->required) {
      // mmc: "Error: the switch on `X' is required to be complete, but it
      // does not cover `c'/1."
      if (inner->canFail) {
        std::vector<std::string> missing{Missing(inner)};
        if (missing.empty()) {
          messages_.Say(Where(goal->tokens),
              "the switch on %s is required to be complete, but it is not"_err_en_US,
              var);
        } else {
          messages_.Say(Where(goal->tokens),
              "the switch on %s is required to be complete, but it does not cover %s"_err_en_US,
              var, llvm::join(missing, ", "));
        }
      }
      return;
    }
    // An arm is acceptable if its determinism makes the promises the
    // required one does; one with more than one solution is committed
    // choice only where only its first solution is wanted.
    Components required{ComponentsOf(*goal->required)};
    var = Name(inner->switchVar);
    for (const SwitchCase &c : inner->cases) {
      Components d{Seen(c.goal)};
      if (Within(d, required) &&
          !(d.most == Solutions::Many && IsCommittedChoice(*goal->required) &&
              !first)) {
        continue;
      }
      std::string functors{FunctorText(c.functor)};
      for (const InstFunctor &other : c.others) {
        functors += " and " + FunctorText(other);
      }
      // mmc: "Error: the arms of the switch on X are required have a
      // determinism that is acceptable in a `det' context, but the actual
      // determinism of the arm for r1.b/0 is `semidet'." At the arm.
      auto &message{messages_.Say(Where(c.goal->tokens),
          "the arms of the switch on %s are required to have a determinism acceptable in a %s context, but the arm for %s is %s"_err_en_US,
          var, DeterminismName(*goal->required), functors,
          DeterminismName(DeterminismOf(d, first)))};
      Report(message, c.goal, *goal->required);
    }
  }

  // Finds the goals where only the first solution is wanted (reference
  // manual, "Committed choice nondeterminism"): the body of a committed
  // choice procedure or lambda expression, a goal with no outputs, as a
  // negated one, the last conjunct of a conjunction where it is wanted,
  // and those before conjuncts that cannot fail, the then and else parts
  // of an if-then-else where it is wanted, and its condition if the then
  // part cannot fail, and the parts of the other goals where it is. A
  // committed choice call elsewhere is an error, with the goals after it
  // that can fail, as in mmc.
  // As in mmc, a call is committed choice if its callee is and only its
  // first solution is wanted, and a compound goal with more than one
  // solution if it is and so is each part with more than one as its
  // parent sees it, which it is where only its first solution is wanted,
  // as mmc commits to it there. Returns whether the goal has more than
  // one solution as its parent sees it.
  bool Contexts(const Body *goal, Context context) {
    // Only the first solution of a goal with no outputs is wanted, but the
    // goals after it that can fail are still why a committed choice in it
    // needs all, if something in it asks for them, as in mmc.
    if (unseen_.count(goal)) {
      context.first = true;
    }
    bool loose{ContextsIn(goal, context)};
    if (Seen(goal).most != Solutions::Many) {
      return false;
    }
    std::optional<parser::Determinism> own;
    if (goal->kind == BodyKind::Call ||
        (goal->kind == BodyKind::Unify && goal->unify == UnifyKind::Call)) {
      own = Of(*goal->modes->callee);
    } else if (goal->kind == BodyKind::HOCall) {
      own = goal->modes->determinism;
    }
    if (context.first && (own ? IsCommittedChoice(*own) : !loose)) {
      first_.insert(goal);
    }
    return !context.first;
  }

  // The goals in a goal that can fail, as mmc records them, which are the
  // reasons a committed choice call before it is in a context that
  // requires all solutions: a goal that can fail, or else those in the
  // branches of a disjunction, a switch or an if-then-else, though not in
  // its condition, or in a scope, even when it cannot fail.
  void Failing(const Body *goal, std::vector<const Body *> &out) {
    if (!body_.goals.count(goal)) {
      return;
    }
    if (Seen(goal).canFail) {
      out.push_back(goal);
      return;
    }
    switch (goal->kind) {
    case BodyKind::Conj:
    case BodyKind::Disj:
    case BodyKind::Scope:
      for (const Body *g : goal->goals) {
        Failing(g, out);
      }
      return;
    case BodyKind::Switch:
      for (const SwitchCase &c : goal->cases) {
        Failing(c.goal, out);
      }
      return;
    case BodyKind::IfThenElse:
      Failing(goal->goals[1], out);
      Failing(goal->goals[2], out);
      return;
    default:
      return;
    }
  }

  bool ContextsIn(const Body *goal, const Context &context) {
    auto committed{[&](parser::Determinism determinism, std::string what) {
      if (context.first || !IsCommittedChoice(determinism)) {
        return;
      }
      // mmc: "Error: call to predicate `p'/1 with determinism `cc_multi'
      // occurs in a context which requires all solutions."
      auto &message{messages_.Say(Where(goal->tokens),
          "%s with determinism %s is in a context that requires all solutions"_err_en_US,
          what, DeterminismName(determinism))};
      std::vector<Cause> causes;
      for (const Body *g : context.failing) {
        Diagnose(g, false, Solutions::Many, "", causes);
      }
      for (const Cause &c : causes) {
        message.Attach(Where(c.tokens), "%s"_because_en_US, c.text);
      }
    }};
    switch (goal->kind) {
    case BodyKind::Unify:
      if (goal->lambda) {
        parser::Determinism declared{goal->modes->determinism};
        Contexts(goal->lambda, {IsCommittedChoice(declared), {}});
      } else if (goal->unify == UnifyKind::Call) {
        const Procedure &callee{*goal->modes->callee};
        committed(*Of(callee), "the call to " + ProcedureToString(callee));
      }
      return false;
    case BodyKind::Call: {
      const Procedure &callee{*goal->modes->callee};
      committed(*Of(callee), "the call to " + ProcedureToString(callee));
      return false;
    }
    case BodyKind::HOCall:
      committed(goal->modes->determinism, "the higher-order call");
      return false;
    case BodyKind::Conj: {
      bool loose{false};
      Context inner{context};
      for (const Body *g : llvm::reverse(goal->goals)) {
        loose |= Contexts(g, inner);
        Failing(g, inner.failing);
        if (Seen(g).canFail) {
          inner.first = false;
        } else if (!Reached(g)) {
          // Nothing after an erroneous goal asks for another solution, as
          // mmc has it, where mode analysis finds it is: by declared
          // determinisms, not inferred ones.
          inner = {true, {}};
        }
      }
      return loose;
    }
    case BodyKind::Switch: {
      bool loose{false};
      for (const SwitchCase &c : goal->cases) {
        loose |= Contexts(c.goal, context);
      }
      return loose;
    }
    case BodyKind::IfThenElse: {
      Context condition{context};
      Failing(goal->goals[1], condition.failing);
      if (Seen(goal->goals[1]).canFail) {
        condition.first = false;
      }
      bool loose{Contexts(goal->goals[0], condition)};
      loose |= Contexts(goal->goals[1], context);
      loose |= Contexts(goal->goals[2], context);
      return loose;
    }
    case BodyKind::Not:
      Contexts(goal->goals.front(), {true, {}});
      return false;
    case BodyKind::Scope:
      // Only the first solution of a determinism cast's goal is wanted, but
      // the goals after it that can fail are still why a committed choice
      // call in it needs all, if something in it asks for them, as in mmc,
      // unlike a negation's.
      if (goal->promise &&
          *goal->promise != parser::PromiseSolutionsKind::EquivalentSolutionSets) {
        Contexts(goal->goals.front(), {true, context.failing});
        return false;
      }
      if (goal->scope == ScopeKind::RequireDeterminism ||
          goal->scope == ScopeKind::RequireSwitch) {
        requiredFirst_[goal] = context.first;
      }
      return Contexts(goal->goals.front(), context);
    default: {
      bool loose{false};
      for (const Body *g : goal->goals) {
        loose |= Contexts(g, context);
      }
      return loose;
    }
    }
  }

  // Whether the procedure is a function's mode whose arguments are all in
  // and whose result is out.
  bool PrimaryFunctionMode() {
    const Procedure &procedure{*body_.procedure};
    if (procedure.symbol->kind != SymbolKind::Function ||
        procedure.modes.size() != body_.head.size()) {
      return false;
    }
    for (std::size_t i{0}; i < procedure.modes.size(); ++i) {
      const ArgMode &mode{procedure.modes[i]};
      const Type *type{body_.vars[body_.head[i]].type};
      bool result{i + 1 == procedure.modes.size()};
      if ((result ? !ops_.IsFree(mode.initial)
                  : !ops_.IsGround(mode.initial, type)) ||
          !ops_.IsGround(mode.final, type)) {
        return false;
      }
    }
    return true;
  }

  // Compares the inferred determinisms with the declared ones, of the
  // procedure and of its lambda expressions.
  void Check() {
    for (const auto &[goal, d] : lambdas_) {
      parser::Determinism declared{goal->modes->determinism};
      if (!Within(d, ComponentsOf(declared))) {
        // mmc: "Determinism error in lambda expression. Declared `det',
        // inferred `semidet''."
        auto &message{messages_.Say(Where(goal->tokens),
            "the lambda expression is declared %s, but its determinism is inferred to be %s"_err_en_US,
            DeterminismName(declared),
            DeterminismName(DeterminismOf(d, IsCommittedChoice(declared))))};
        Report(message, goal->lambda, declared);
      }
    }
    std::optional<parser::Determinism> declared{body_.procedure->determinism};
    parser::Determinism d{declared ? *declared : *body_.determinism};
    if (HasIOState() && d != parser::Determinism::Det &&
        d != parser::Determinism::CcMulti &&
        d != parser::Determinism::Erroneous) {
      // mmc: "error: invalid determinism for a predicate with I/O state
      // arguments." By the declared determinism, or the inferred one.
      messages_.Say(Declaration(),
          "invalid determinism for %s: a procedure with I/O state arguments must be det, cc_multi or erroneous, but it is %s"_err_en_US,
          Text(), DeterminismName(d));
    }
    if (!declared) {
      return;
    }
    if (PrimaryFunctionMode() &&
        ComponentsOf(*body_.determinism).most == Solutions::Many) {
      // mmc: "Error: invalid determinism for `f'(in) = out: the primary
      // mode of a function cannot be `multi'." By the inferred
      // determinism, then as for other procedures. Only for this mode,
      // though the manual rules out more than one solution for any mode
      // whose arguments are all input (see the register).
      messages_.Say(Declaration(),
          "invalid determinism for %s: the primary mode of a function cannot be %s"_err_en_US,
          Text(), DeterminismName(*body_.determinism));
    }
    Components inferred{ComponentsOf(*body_.determinism)};
    if (!Within(inferred, ComponentsOf(*declared))) {
      // mmc: "error: determinism declaration not satisfied. Declared
      // `det', inferred `semidet'. The reason for the difference is the
      // following."
      auto &message{messages_.Say(Declaration(),
          "the determinism declaration of %s is not satisfied: it is declared %s, but inferred %s"_err_en_US,
          Text(), DeterminismName(*declared), DeterminismName(*body_.determinism))};
      Report(message, body_.body, *declared);
    } else if (*body_.determinism != *declared &&
        !body_.procedure->source.empty() && !body_.method) {
      // mmc: "warning: determinism declaration could be tighter." Not
      // for an instance's method, whose declaration is its type class's,
      // as in mmc.
      messages_.Say(Declaration(),
          "the determinism declaration of %s could be tighter: it is declared %s, but inferred %s"_warn_en_US,
          Text(), DeterminismName(*declared), DeterminismName(*body_.determinism));
    }
  }

  const DeterminismChecking &checking_;
  ProcedureBody &body_;
  TypeChecking &types_;
  InstOps &ops_;
  const parser::TokenSequence &tokens_;
  parser::Messages &messages_;
  std::map<const Body *, std::set<BodyVar>> vars_;
  // The functors that variables are known to have (see Known).
  std::map<BodyVar, InstFunctor> known_;
  // The goals that have no outputs, and those of them that succeed at most
  // once because of that.
  std::set<const Body *> unseen_;
  std::set<const Body *> pruned_;
  // The promise_equivalent_solution_sets scopes that the goal is in, and
  // the determinism casts checked, by their first token.
  std::vector<const Body *> sets_;
  // The require_* scopes checked, by their first tokens, and whether only
  // the first solution of each is wanted.
  std::set<unsigned> required_;
  std::map<const Body *, bool> requiredFirst_;
  // The variables of the unifications replaced with fail in the
  // conjunctions around a goal, before mmc quantifies the body again.
  std::set<BodyVar> failed_;
  std::set<unsigned> checked_;
  // The goals with more than one solution where only the first is wanted,
  // which are committed choice.
  std::set<const Body *> first_;
  std::vector<std::pair<const Body *, Components>> lambdas_;
};

std::unique_ptr<DeterminismChecking> CheckDeterminism(
    const parser::TokenSequence &tokens, TypeChecking &types,
    ModeChecking &modes, parser::Messages &messages, bool analyse) {
  auto result{std::make_unique<DeterminismChecking>()};
  auto analyseBody{[&](ProcedureBody &body, parser::Messages &messages) {
    body.goals.clear();
    body.determinism.reset();
    body.unchecked = DeterminismUnchecked::None;
    DeterminismAnalyser{
        *result, body, types, modes.ops(), tokens, messages}
        .Run();
  }};
  auto add{[&](const Procedure &procedure, const MethodModes *method,
               llvm::ArrayRef<ClauseModes> clauses) {
    // A procedure whose modes are not all known, as one declared with an
    // undefined inst in `with_inst`, has an error already.
    if (clauses.empty() ||
        procedure.modes.size() != clauses.front().clause->clause->head.size()) {
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
  if (!analyse) {
    return result;
  }
  // The determinisms of the procedures declared without one are the least
  // fixpoint of inference, starting from erroneous, as in mmc, where the
  // manual says det (see the register). Each step is joined with the last,
  // so that the fixpoint is reached in at most three steps per procedure.
  // A procedure that calls one with none, and so has none, takes its
  // callers' away in turn. One with a mode error is not analysed, and so
  // stays erroneous, as in mmc.
  for (const Procedure &procedure : modes.procedures()) {
    if (!procedure.determinism && modes.HasError(procedure)) {
      result->inferred_[&procedure] = parser::Determinism::Erroneous;
    }
  }
  for (const Procedure &procedure : modes.inferred()) {
    if (!procedure.determinism && modes.HasError(procedure)) {
      result->inferred_[&procedure] = parser::Determinism::Erroneous;
    }
  }
  std::vector<ProcedureBody *> undeclared;
  for (ProcedureBody &body : result->bodies_) {
    if (!body.method && !body.procedure->determinism) {
      undeclared.push_back(&body);
      result->inferred_[body.procedure] = parser::Determinism::Erroneous;
    }
  }
  for (bool changed{!undeclared.empty()}; changed;) {
    changed = false;
    for (ProcedureBody *body : undeclared) {
      if (!result->inferred_.count(body->procedure)) {
        continue;
      }
      parser::Messages ignored;
      analyseBody(*body, ignored);
      if (!body->determinism) {
        result->inferred_.erase(body->procedure);
        changed = true;
        continue;
      }
      parser::Determinism &inferred{result->inferred_[body->procedure]};
      Components old{ComponentsOf(inferred)};
      Components now{ComponentsOf(*body->determinism)};
      parser::Determinism joined{DeterminismOf(
          {old.canFail || now.canFail, std::max(old.most, now.most)})};
      if (joined != inferred) {
        inferred = joined;
        changed = true;
      }
    }
  }
  for (ProcedureBody &body : result->bodies_) {
    analyseBody(body, messages);
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
  // A goal's determinism, if it was inferred.
  auto tag{[&](const Body *goal) {
    auto it{body.goals.find(goal)};
    return it == body.goals.end()
        ? std::string{}
        : std::string{" ["} + DeterminismName(it->second) + ']';
  }};
  std::function<void(const Body *, unsigned)> dump{[&](const Body *goal,
                                                      unsigned depth) {
    std::string indent(2 * depth + 4, ' ');
    std::string at{position(tokens.tokens[goal->tokens.first].spelling)};
    switch (goal->kind) {
    case BodyKind::Conj:
      if (goal->goals.empty()) {
        os << indent << at << " true" << tag(goal) << '\n';
      }
      for (const Body *g : goal->goals) {
        dump(g, depth);
      }
      return;
    case BodyKind::Disj:
      os << indent << at << (goal->goals.empty() ? " fail" : " disjunction")
         << tag(goal) << '\n';
      for (const Body *g : goal->goals) {
        os << indent << "  branch\n";
        dump(g, depth + 2);
      }
      return;
    case BodyKind::Switch:
      os << indent << at << " switch on " << name(goal->switchVar)
         << (goal->canFail ? ", can fail" : ", cannot fail") << tag(goal)
         << '\n';
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
      os << indent << at << " if-then-else" << tag(goal) << '\n';
      for (std::size_t i{0}; i < 3; ++i) {
        os << indent << "  " << (i == 0 ? "if" : i == 1 ? "then" : "else")
           << '\n';
        dump(goal->goals[i], depth + 2);
      }
      return;
    case BodyKind::Not:
      os << indent << at << " not" << tag(goal) << '\n';
      dump(goal->goals.front(), depth + 1);
      return;
    case BodyKind::Scope:
      os << indent << at << ' ';
      if (goal->promise) {
        using Kind = parser::PromiseSolutionsKind;
        os << (*goal->promise == Kind::EquivalentSolutions
                      ? "promise_equivalent_solutions"
                      : *goal->promise == Kind::EquivalentSolutionSets
                      ? "promise_equivalent_solution_sets"
                      : "arbitrary")
           << " [" << names(goal->promised) << ']';
      } else if (goal->scope == ScopeKind::RequireDeterminism) {
        os << "require_" << DeterminismName(*goal->required);
      } else if (goal->scope == ScopeKind::RequireSwitch) {
        if (goal->required) {
          os << "require_switch_arms_" << DeterminismName(*goal->required);
        } else {
          os << "require_complete_switch";
        }
        if (goal->switchOn) {
          os << " [" << name(*goal->switchOn) << ']';
        }
      } else {
        os << llvm::StringRef{EnumToString(goal->scope)}.lower();
      }
      os << tag(goal) << '\n';
      dump(goal->goals.front(), depth + 1);
      return;
    case BodyKind::Call:
      os << indent << at << " call " << *goal->modes->callee->symbol << " mode "
         << goal->modes->callee->number << (goal->modes->implied ? " implied" : "")
         << ": " << goal->call->name->name << '(' << names(goal->arguments)
         << ')' << tag(goal) << '\n';
      return;
    case BodyKind::HOCall: {
      llvm::ArrayRef<BodyVar> arguments{goal->arguments};
      os << indent << at << " higher-order call: ";
      if (goal->hoCall->kind == parser::PredOrFunc::Function) {
        os << name(arguments.back()) << " = ";
        arguments = arguments.drop_back();
      }
      os << name(goal->closure) << '(' << names(arguments) << ')' << tag(goal)
         << '\n';
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
      os << tag(goal) << '\n';
      if (goal->lambda) {
        dump(goal->lambda, depth + 1);
      }
      return;
    }
  }};
  dump(body.body, 0);
}

} // namespace merkurius::semantics
