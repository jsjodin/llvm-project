//===-- lib/Semantics/desugar.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Desugaring into the semantic form (design document, "Semantic form").
// Each clause is desugared on its own: its variables are renumbered, its
// head's arguments become distinct variables, and its data-terms become
// chains of unifications through fresh variables.
//
// State variables are threaded through the clause as it is desugared, which
// gives the result of the reference manual's transform_goal and
// transform_clause ("State variables") without its chains of X1 = X2: each
// state variable in scope has a current version, an atomic goal that names
// !:X makes the next one, and where branches leave it at different
// versions, each branch ends by unifying its version with a new one.
//
// A DCG rule's two implicit arguments are threaded the same way, as the
// versions of a hidden state variable, which gives the result of the
// manual's DCG-transform ("DCG-goals").

#include "merkurius/Semantics/desugar.h"
#include "merkurius/Common/idioms.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>
#include <vector>

namespace merkurius::semantics {
namespace {

using namespace parser::literals;
using Goals = llvm::SmallVector<Goal, 4>;

TokenRange GoalTokens(const parser::Goal &goal) {
  return std::visit([](const auto *x) { return x->tokens; }, goal);
}
TokenRange GoalTokens(const parser::DcgGoal &goal) {
  return std::visit([](const auto *x) { return x->tokens; }, goal);
}

// The hidden state variable of a DCG rule. A Mercury variable's name starts
// with a capital letter or _, so this one cannot be named in the source.
constexpr llvm::StringLiteral dcg{"dcg"};

// A slot of the head that a state variable's last version fills.
constexpr VarId unfilled{~VarId{0}};

class ClauseDesugarer {
public:
  ClauseDesugarer(const parser::TokenSequence &tokens, SemanticArena &arena,
      parser::Messages &messages)
      : tokens_{tokens}, arena_{arena}, messages_{messages} {}

  std::optional<const ClauseForm *> Desugar(const parser::FactOrRule &);
  std::optional<const ClauseForm *> Desugar(const parser::DcgRule &);

private:
  template <typename T, typename... A> const T *Make(A &&...args) {
    return arena_.Make<T>(std::forward<A>(args)...);
  }
  template <typename T> llvm::ArrayRef<T> Save(llvm::ArrayRef<T> array) {
    return arena_.SaveArray(array);
  }
  unsigned Number() { return goals_++; }
  parser::CharBlock Source(TokenRange range) const {
    return {tokens_.tokens[range.first].spelling.begin(),
        tokens_.tokens[range.last].spelling.end()};
  }
  template <typename... A> void Error(TokenRange range, A &&...args) {
    messages_.Say(Source(range), std::forward<A>(args)...);
    ok_ = false;
  }
  // Reports the first form in the clause that is not desugared yet.
  void Todo(TokenRange range, const char *what) {
    if (ok_) {
      messages_.Say(Source(range), "desugaring %s"_todo_en_US, what);
      ok_ = false;
    }
  }

  // The state variables in scope, in the order they came into scope, each
  // with its current version, which it lacks until it is first given one.
  using StateMap = llvm::MapVector<llvm::StringRef, std::optional<VarId>>;
  // A state variable brought into scope by a quantifier or a trace
  // parameter, hiding any of the same name, until Restore.
  struct Shadowed {
    llvm::StringRef name;
    bool wasInScope;
    std::optional<VarId> was;
  };

  VarId ProgramVar(const parser::Variable &);
  VarId Fresh(TokenRange);
  VarId NewVersion(llvm::StringRef name, TokenRange);
  // The version that !.X or !:X stands for in the atomic goal being
  // desugared.
  VarId StateRef(const parser::StateVar &, bool next);
  void StateVarBothError(const parser::StateVar &);
  // Makes the !:X of the atomic goal just desugared the current versions.
  void FinishAtomic();
  Shadowed Shadow(llvm::StringRef name, std::optional<VarId> version);
  void Restore(llvm::ArrayRef<Shadowed>);
  // Desugars a branch from `start` into `out`, and returns the state at its
  // end.
  template <typename G>
  StateMap Branch(const G &goal, const StateMap &start, Goals &out) {
    state_ = start;
    DesugarInto(goal, out);
    return state_;
  }
  // Joins branches that started from `start` and ended in `ends`.
  void Merge(const StateMap &start, llvm::ArrayRef<StateMap> ends,
      llvm::MutableArrayRef<Goals> arms, TokenRange);
  // A call's arguments, in which !X is the two arguments !.X and !:X.
  llvm::ArrayRef<VarId> CallArguments(
      llvm::ArrayRef<parser::DataTerm>, Goals &out);
  // The variable that a data-term is, if it is a variable or !.X or !:X.
  std::optional<VarId> SimpleVar(const parser::DataTerm &);
  // A variable that holds the term, after the goals added to `out`.
  VarId Flatten(const parser::DataTerm &, Goals &out);
  llvm::ArrayRef<VarId> FlattenAll(
      llvm::ArrayRef<parser::DataTerm>, Goals &out);
  // Adds the goals for Var = Term to `out`.
  void UnifyWith(
      VarId, const parser::DataTerm &, TokenRange, Goals &out);
  // A list whose tail is `tail`, `tailVar` or, with neither, [].
  void UnifyList(VarId, TokenRange, llvm::ArrayRef<parser::DataTerm> elements,
      const std::optional<parser::DataTerm> &tail, Goals &out,
      std::optional<VarId> tailVar = std::nullopt);
  void AddUnify(Goals &out, std::size_t at, TokenRange range, VarId left,
      Functor functor, llvm::ArrayRef<VarId> arguments);

  template <typename G> Goal DesugarGoal(const G &goal) {
    Goals goals;
    DesugarInto(goal, goals);
    return Conjoin(GoalTokens(goal), goals);
  }
  // Adds the goal to `out`, as conjuncts when it is a conjunction.
  void DesugarInto(const parser::Goal &, Goals &out);
  void DesugarInto(const parser::DcgGoal &, Goals &out);
  // some Vars G or all Vars G, for a goal or a DCG goal.
  template <typename G>
  void Quantify(const ParseGoal &origin, TokenRange, parser::Quantifier,
      llvm::ArrayRef<parser::DataTerm> vars, const G &goal, Goals &out);
  // The DCG argument's current version, and a new next one.
  VarId DcgCurrent() { return *state_.lookup(dcg); }
  VarId DcgNext(TokenRange range) {
    VarId next{NewVersion(dcg, range)};
    state_[dcg] = next;
    return next;
  }
  // The head's arguments and, for a DCG rule, its implicit ones; and the
  // clause, once its body is desugared into `goals`.
  void Head(llvm::ArrayRef<const parser::ModedTerm *> arguments,
      const parser::ModedTerm *result, Goals &goals);
  void DcgHead(TokenRange);
  const ClauseForm *Finish(TokenRange, parser::PredOrFunc,
      const parser::SymName *, Goals &goals);
  Goal Conjoin(TokenRange, const Goals &);
  Goal Negate(TokenRange range, Goal goal) {
    return Make<Not>(range, Number(), goal);
  }
  std::optional<llvm::ArrayRef<VarId>> ScopeVars(
      llvm::ArrayRef<parser::DataTerm>);
  Goal MakeScope(TokenRange range, ScopeKind kind, const ParseGoal &origin,
      llvm::ArrayRef<VarId> vars, const parser::Goal &goal) {
    Goal inner{DesugarGoal(goal)};
    return Make<Scope>(range, Number(), kind, origin, vars, inner);
  }

  const parser::TokenSequence &tokens_;
  SemanticArena &arena_;
  parser::Messages &messages_;
  std::vector<VarInfo> vars_;
  llvm::DenseMap<unsigned, VarId> numbers_; // the parse tree's numbering
  std::vector<TypeQualification> typeQualifications_;
  StateMap state_;
  // The !:X versions that the atomic goal being desugared names.
  llvm::SmallVector<std::pair<llvm::StringRef, VarId>, 2> next_;
  llvm::StringMap<unsigned> versions_; // each state variable's next version
  // The head being built, and each state variable whose last version fills
  // a slot of it once the body is desugared.
  struct Last {
    std::size_t slot;
    llvm::StringRef name;
    TokenRange tokens;
  };
  llvm::SmallVector<VarId, 4> head_;
  llvm::SmallVector<Last, 2> lasts_;
  unsigned goals_{0};
  bool ok_{true};
};

VarId ClauseDesugarer::ProgramVar(const parser::Variable &var) {
  auto [it, added]{numbers_.try_emplace(var.number, vars_.size())};
  if (added) {
    vars_.push_back(VarInfo{
        var.name == "_" ? VarKind::Anonymous : VarKind::Program,
        var.name == "_" ? llvm::StringRef{} : var.name, var.tokens});
  }
  return it->second;
}

VarId ClauseDesugarer::Fresh(TokenRange range) {
  vars_.push_back(VarInfo{VarKind::Fresh, {}, range});
  return vars_.size() - 1;
}

VarId ClauseDesugarer::NewVersion(llvm::StringRef name, TokenRange range) {
  unsigned &version{versions_[name]};
  vars_.push_back(VarInfo{VarKind::State, name, range, version++});
  return vars_.size() - 1;
}

VarId ClauseDesugarer::StateRef(const parser::StateVar &var, bool next) {
  auto it{state_.find(var.name)};
  if (it == state_.end()) {
    // mmc: "state variable !.S is not visible in this context".
    Error(var.tokens,
        "state variable %s%s is not visible in this context"_err_en_US,
        next ? "!:" : "!.", var.name);
    return Fresh(var.tokens);
  }
  if (next) {
    for (const auto &[name, version] : next_) {
      if (name == var.name) {
        return version;
      }
    }
    VarId version{NewVersion(var.name, var.tokens)};
    next_.emplace_back(var.name, version);
    return version;
  }
  if (!it->second) {
    // As in mmc, a warning: the version is a variable that nothing binds.
    messages_.Say(Source(var.tokens),
        "reference to uninitialized state variable !.%s"_warn_en_US,
        var.name);
    it->second = NewVersion(var.name, var.tokens);
  }
  return *it->second;
}

void ClauseDesugarer::StateVarBothError(const parser::StateVar &var) {
  Error(var.tokens,
      "!%s may be only an argument of a call or a clause head, since it "
      "stands for two arguments, !.%s and !:%s"_err_en_US,
      var.name, var.name, var.name);
}

void ClauseDesugarer::FinishAtomic() {
  for (const auto &[name, version] : next_) {
    state_[name] = version;
  }
  next_.clear();
}

ClauseDesugarer::Shadowed ClauseDesugarer::Shadow(
    llvm::StringRef name, std::optional<VarId> version) {
  auto it{state_.find(name)};
  Shadowed shadowed{name, it != state_.end(),
      it != state_.end() ? it->second : std::nullopt};
  state_[name] = version;
  return shadowed;
}

void ClauseDesugarer::Restore(llvm::ArrayRef<Shadowed> shadowed) {
  for (const Shadowed &s : llvm::reverse(shadowed)) {
    if (s.wasInScope) {
      state_[s.name] = s.was;
    } else {
      state_.erase(s.name);
    }
  }
}

void ClauseDesugarer::Merge(const StateMap &start,
    llvm::ArrayRef<StateMap> ends, llvm::MutableArrayRef<Goals> arms,
    TokenRange range) {
  state_ = start;
  for (auto &[name, current] : state_) {
    std::optional<VarId> first{ends.front().lookup(name)};
    if (llvm::all_of(ends.drop_front(),
            [&](const StateMap &end) { return end.lookup(name) == first; })) {
      current = first;
      continue;
    }
    VarId merged{NewVersion(name, range)};
    for (std::size_t j{0}; j < ends.size(); ++j) {
      if (std::optional<VarId> last{ends[j].lookup(name)}) {
        arms[j].push_back(Make<Unify>(range, Number(), merged, true, *last,
            Functor{}, llvm::ArrayRef<VarId>{}));
      }
    }
    current = merged;
  }
}

llvm::ArrayRef<VarId> ClauseDesugarer::CallArguments(
    llvm::ArrayRef<parser::DataTerm> terms, Goals &out) {
  llvm::SmallVector<VarId, 4> vars;
  for (const parser::DataTerm &term : terms) {
    const auto *state{std::get_if<const parser::StateVar *>(&term)};
    if (state && (*state)->kind == parser::StateVarKind::Both) {
      vars.push_back(StateRef(**state, false));
      vars.push_back(StateRef(**state, true));
    } else {
      vars.push_back(Flatten(term, out));
    }
  }
  return Save(llvm::ArrayRef<VarId>{vars});
}

std::optional<VarId> ClauseDesugarer::SimpleVar(const parser::DataTerm &term) {
  if (const auto *var{std::get_if<const parser::Variable *>(&term)}) {
    return ProgramVar(**var);
  }
  const auto *state{std::get_if<const parser::StateVar *>(&term)};
  if (state && (*state)->kind != parser::StateVarKind::Both) {
    return StateRef(**state, (*state)->kind == parser::StateVarKind::Next);
  }
  return std::nullopt;
}

VarId ClauseDesugarer::Flatten(const parser::DataTerm &term, Goals &out) {
  if (std::optional<VarId> var{SimpleVar(term)}) {
    return *var;
  }
  TokenRange range{std::visit([](const auto *x) { return x->tokens; }, term)};
  VarId fresh{Fresh(range)};
  UnifyWith(fresh, term, range, out);
  return fresh;
}

llvm::ArrayRef<VarId> ClauseDesugarer::FlattenAll(
    llvm::ArrayRef<parser::DataTerm> terms, Goals &out) {
  llvm::SmallVector<VarId, 4> vars;
  for (const parser::DataTerm &term : terms) {
    vars.push_back(Flatten(term, out));
  }
  return Save(llvm::ArrayRef<VarId>{vars});
}

// Puts X = f(...) before the unifications that its arguments needed, at
// `at`, so a chain reads from the outside in, as it was written.
void ClauseDesugarer::AddUnify(Goals &out, std::size_t at, TokenRange range,
    VarId left, Functor functor, llvm::ArrayRef<VarId> arguments) {
  out.insert(out.begin() + at,
      Make<Unify>(range, Number(), left, false, VarId{0}, functor, arguments));
}

void ClauseDesugarer::UnifyWith(VarId left, const parser::DataTerm &term,
    TokenRange range, Goals &out) {
  std::size_t at{out.size()};
  auto functor{[&](FunctorKind kind) { return Functor{kind, nullptr,
                                           nullptr, nullptr}; }};
  std::visit(
      common::visitors{
          [&](const parser::Variable *var) {
            out.push_back(Make<Unify>(range, Number(), left, true,
                ProgramVar(*var), Functor{}, llvm::ArrayRef<VarId>{}));
          },
          [&](const parser::Literal *literal) {
            AddUnify(out, at, range, left,
                Functor{FunctorKind::Literal, nullptr, literal, nullptr}, {});
          },
          [&](const parser::ImplementationLiteral *literal) {
            AddUnify(out, at, range, left,
                Functor{FunctorKind::ImplementationLiteral, nullptr, nullptr,
                    literal},
                {});
          },
          [&](const parser::FunctorData *data) {
            // apply(F, X1, ..., Xn) is a higher-order function call, as
            // call(P, ...) is a higher-order predicate call.
            if (data->name->qualifiers.empty() && data->name->name == "apply" &&
                !data->arguments.empty()) {
              VarId closure{Flatten(data->arguments.front(), out)};
              llvm::SmallVector<VarId, 4> arguments{
                  FlattenAll(data->arguments.drop_front(), out)};
              arguments.push_back(left);
              out.push_back(Make<HOCall>(range, Number(),
                  parser::PredOrFunc::Function, closure,
                  Save(llvm::ArrayRef<VarId>{arguments})));
              return;
            }
            llvm::ArrayRef<VarId> arguments{FlattenAll(data->arguments, out)};
            AddUnify(out, at, range, left,
                Functor{FunctorKind::Name, data->name, nullptr, nullptr},
                arguments);
          },
          [&](const parser::ListData *list) {
            UnifyList(left, range, list->elements, list->tail, out);
          },
          [&](const parser::TupleData *tuple) {
            llvm::ArrayRef<VarId> arguments{FlattenAll(tuple->elements, out)};
            AddUnify(out, at, range, left, functor(FunctorKind::Tuple),
                arguments);
          },
          [&](const parser::ApplyData *apply) {
            VarId closure{Flatten(apply->function, out)};
            llvm::SmallVector<VarId, 4> arguments{
                FlattenAll(apply->arguments, out)};
            arguments.push_back(left);
            out.push_back(Make<HOCall>(range, Number(),
                parser::PredOrFunc::Function, closure,
                Save(llvm::ArrayRef<VarId>{arguments})));
          },
          [&](const parser::UnifyData *both) {
            UnifyWith(left, both->left, range, out);
            UnifyWith(left, both->right, range, out);
          },
          [&](const parser::TypeQualified *qualified) {
            typeQualifications_.push_back(
                TypeQualification{qualified->tokens, left, qualified->type});
            UnifyWith(left, qualified->term, range, out);
          },
          [&](const parser::Coerce *coerce) {
            VarId argument{Flatten(coerce->term, out)};
            AddUnify(out, at, range, left, functor(FunctorKind::Coerce),
                Save(llvm::ArrayRef<VarId>{argument}));
          },
          [&](const parser::StateVar *var) {
            if (var->kind == parser::StateVarKind::Both) {
              StateVarBothError(*var);
              return;
            }
            out.push_back(Make<Unify>(range, Number(), left, true,
                StateRef(*var, var->kind == parser::StateVarKind::Next),
                Functor{}, llvm::ArrayRef<VarId>{}));
          },
          [&](const parser::ConditionalData *x) {
            Todo(x->tokens, "conditional expressions");
          },
          [&](const parser::FieldSelection *x) {
            Todo(x->tokens, "field selection");
          },
          [&](const parser::FieldUpdate *x) {
            Todo(x->tokens, "field update");
          },
          [&](const parser::Lambda *x) {
            Todo(x->tokens, "lambda expressions");
          },
      },
      term);
}

// [E1, E2 | T] is '[|]'(E1, V), V = '[|]'(E2, T); without a tail, the last
// cell's is [].
void ClauseDesugarer::UnifyList(VarId left, TokenRange range,
    llvm::ArrayRef<parser::DataTerm> elements,
    const std::optional<parser::DataTerm> &tail, Goals &out,
    std::optional<VarId> tailVar) {
  std::size_t at{out.size()};
  if (elements.empty()) {
    if (tail) {
      UnifyWith(left, *tail, range, out);
    } else if (tailVar) {
      out.push_back(Make<Unify>(range, Number(), left, true, *tailVar,
          Functor{}, llvm::ArrayRef<VarId>{}));
    } else {
      AddUnify(out, at, range, left,
          Functor{FunctorKind::Nil, nullptr, nullptr, nullptr}, {});
    }
    return;
  }
  VarId head{Flatten(elements.front(), out)};
  VarId rest;
  if (elements.size() == 1 && tail) {
    rest = Flatten(*tail, out);
  } else if (elements.size() == 1 && tailVar) {
    rest = *tailVar;
  } else {
    rest = Fresh(range);
    UnifyList(rest, range, elements.drop_front(), tail, out, tailVar);
  }
  AddUnify(out, at, range, left,
      Functor{FunctorKind::Cons, nullptr, nullptr, nullptr},
      Save(llvm::ArrayRef<VarId>{head, rest}));
}

Goal ClauseDesugarer::Conjoin(TokenRange range, const Goals &goals) {
  if (goals.size() == 1) {
    return goals.front();
  }
  return Make<Conj>(range, Number(), false, Save(llvm::ArrayRef<Goal>{goals}));
}

std::optional<llvm::ArrayRef<VarId>> ClauseDesugarer::ScopeVars(
    llvm::ArrayRef<parser::DataTerm> terms) {
  llvm::SmallVector<VarId, 4> vars;
  for (const parser::DataTerm &term : terms) {
    if (const auto *var{std::get_if<const parser::Variable *>(&term)}) {
      vars.push_back(ProgramVar(**var));
    } else {
      Todo(std::visit([](const auto *x) { return x->tokens; }, term),
          "state variables in a scope's list of variables");
      return std::nullopt;
    }
  }
  return Save(llvm::ArrayRef<VarId>{vars});
}

void ClauseDesugarer::DesugarInto(const parser::Goal &goal, Goals &out) {
  std::visit(
      common::visitors{
          [&](const parser::Conjunction *conj) {
            if (!conj->parallel) {
              for (const parser::Goal &conjunct : conj->goals) {
                DesugarInto(conjunct, out);
              }
              return;
            }
            Goals conjuncts;
            for (const parser::Goal &conjunct : conj->goals) {
              conjuncts.push_back(DesugarGoal(conjunct));
            }
            out.push_back(Make<Conj>(conj->tokens, Number(), true,
                Save(llvm::ArrayRef<Goal>{conjuncts})));
          },
          [&](const parser::Disjunction *disj) {
            StateMap start{state_};
            llvm::SmallVector<Goals, 2> arms(disj->goals.size());
            llvm::SmallVector<StateMap, 2> ends;
            for (std::size_t j{0}; j < disj->goals.size(); ++j) {
              ends.push_back(Branch(disj->goals[j], start, arms[j]));
            }
            Merge(start, ends, arms, disj->tokens);
            Goals disjuncts;
            for (std::size_t j{0}; j < arms.size(); ++j) {
              disjuncts.push_back(Conjoin(GoalTokens(disj->goals[j]), arms[j]));
            }
            out.push_back(Make<Disj>(
                disj->tokens, Number(), Save(llvm::ArrayRef<Goal>{disjuncts})));
          },
          [&](const parser::BuiltinGoal *builtin) {
            if (builtin->kind == parser::BuiltinGoalKind::Fail) {
              out.push_back(Make<Disj>(
                  builtin->tokens, Number(), llvm::ArrayRef<Goal>{}));
            } // true adds no conjunct
          },
          [&](const parser::Negation *negation) {
            // Nothing a negated goal binds is seen outside it.
            StateMap start{state_};
            Goal negated{DesugarGoal(negation->goal)};
            state_ = start;
            out.push_back(Negate(negation->tokens, negated));
          },
          [&](const parser::Implication *implication) {
            // G1 => G2 is not (G1, not G2), and G1 <=> G2 is both ways.
            auto implies{[&](const parser::Goal &from, const parser::Goal &to) {
              Goals both;
              DesugarInto(from, both);
              both.push_back(Negate(implication->tokens, DesugarGoal(to)));
              return Negate(
                  implication->tokens, Conjoin(implication->tokens, both));
            }};
            StateMap start{state_};
            switch (implication->kind) {
            case parser::ImplicationKind::Implies:
              out.push_back(implies(implication->left, implication->right));
              break;
            case parser::ImplicationKind::ImpliedBy:
              out.push_back(implies(implication->right, implication->left));
              break;
            case parser::ImplicationKind::Equivalent:
              out.push_back(implies(implication->left, implication->right));
              out.push_back(implies(implication->right, implication->left));
              break;
            }
            state_ = start;
          },
          [&](const parser::IfThenElse *ite) {
            // The condition's updates are seen in the then part only.
            StateMap start{state_};
            Goal condition{DesugarGoal(ite->condition)};
            StateMap afterCondition{state_};
            Goals arms[2];
            StateMap ends[2]{Branch(ite->then, afterCondition, arms[0]),
                Branch(ite->otherwise, start, arms[1])};
            Merge(start, ends, arms, ite->tokens);
            out.push_back(Make<IfThenElse>(ite->tokens, Number(), condition,
                Conjoin(GoalTokens(ite->then), arms[0]),
                Conjoin(GoalTokens(ite->otherwise), arms[1])));
          },
          [&](const parser::Unification *unification) {
            // mmc: "!S cannot appear as a unification argument".
            for (const parser::DataTerm *side :
                {&unification->left, &unification->right}) {
              const auto *state{std::get_if<const parser::StateVar *>(side)};
              if (state && (*state)->kind == parser::StateVarKind::Both) {
                Error((*state)->tokens,
                    "!%s cannot be a side of a unification; you probably "
                    "meant !.%s or !:%s"_err_en_US,
                    (*state)->name, (*state)->name, (*state)->name);
                return;
              }
            }
            if (std::optional<VarId> left{SimpleVar(unification->left)}) {
              UnifyWith(*left, unification->right, unification->tokens, out);
            } else if (std::optional<VarId> right{
                           SimpleVar(unification->right)}) {
              UnifyWith(*right, unification->left, unification->tokens, out);
            } else {
              VarId both{Flatten(unification->left, out)};
              UnifyWith(both, unification->right, unification->tokens, out);
            }
            FinishAtomic();
          },
          [&](const parser::Inequality *inequality) {
            // X \= Y is not (X = Y).
            StateMap start{state_};
            Goals unify;
            VarId left{Flatten(inequality->left, unify)};
            UnifyWith(left, inequality->right, inequality->tokens, unify);
            next_.clear();
            state_ = start;
            out.push_back(Negate(
                inequality->tokens, Conjoin(inequality->tokens, unify)));
          },
          [&](const parser::StateVarFieldUpdate *x) {
            Todo(x->tokens, "field update");
          },
          [&](const parser::Call *call) {
            if (call->name->qualifiers.empty() && call->name->name == "call" &&
                !call->arguments.empty()) {
              VarId closure{Flatten(call->arguments.front(), out)};
              llvm::ArrayRef<VarId> arguments{
                  CallArguments(call->arguments.drop_front(), out)};
              out.push_back(Make<HOCall>(call->tokens, Number(),
                  parser::PredOrFunc::Predicate, closure, arguments));
              FinishAtomic();
              return;
            }
            llvm::ArrayRef<VarId> arguments{
                CallArguments(call->arguments, out)};
            out.push_back(Make<Call>(
                call->tokens, Number(), call->name, arguments, false));
            FinishAtomic();
          },
          [&](const parser::HigherOrderCall *call) {
            VarId closure{Flatten(call->closure, out)};
            llvm::ArrayRef<VarId> arguments{
                CallArguments(call->arguments, out)};
            out.push_back(Make<HOCall>(call->tokens, Number(),
                parser::PredOrFunc::Predicate, closure, arguments));
            FinishAtomic();
          },
          [&](const parser::QuantifiedGoal *quantified) {
            Quantify(goal, quantified->tokens, quantified->quantifier,
                quantified->vars, quantified->goal, out);
          },
          [&](const parser::PurityGoal *x) {
            out.push_back(
                MakeScope(x->tokens, ScopeKind::Purity, goal, {}, x->goal));
          },
          [&](const parser::PromisePurity *x) {
            out.push_back(MakeScope(
                x->tokens, ScopeKind::PromisePurity, goal, {}, x->goal));
          },
          [&](const parser::PromiseSolutions *x) {
            if (auto vars{ScopeVars(x->vars)}) {
              out.push_back(MakeScope(
                  x->tokens, ScopeKind::PromiseSolutions, goal, *vars, x->goal));
            }
          },
          [&](const parser::RequireDeterminism *x) {
            out.push_back(MakeScope(
                x->tokens, ScopeKind::RequireDeterminism, goal, {}, x->goal));
          },
          [&](const parser::RequireSwitch *x) {
            if (auto vars{ScopeVars(llvm::ArrayRef<parser::DataTerm>{x->var})}) {
              out.push_back(MakeScope(
                  x->tokens, ScopeKind::RequireSwitch, goal, *vars, x->goal));
            }
          },
          [&](const parser::DisableWarnings *x) {
            out.push_back(MakeScope(
                x->tokens, ScopeKind::DisableWarnings, goal, {}, x->goal));
          },
          [&](const parser::TraceGoal *x) {
            // io(!IO) and state(M, !S) bring a new state variable into
            // scope inside the trace goal; its first and last versions
            // are the scope's variables.
            llvm::SmallVector<Shadowed, 2> shadowed;
            llvm::SmallVector<VarId, 4> first;
            for (const parser::TraceParam *param : x->params) {
              if (param->var) {
                const auto *state{
                    std::get<const parser::StateVar *>(*param->var)};
                first.push_back(NewVersion(state->name, state->tokens));
                shadowed.push_back(Shadow(state->name, first.back()));
              }
            }
            Goal inner{DesugarGoal(x->goal)};
            llvm::SmallVector<VarId, 4> vars;
            for (std::size_t j{0}; j < shadowed.size(); ++j) {
              vars.push_back(first[j]);
              vars.push_back(state_.lookup(shadowed[j].name).value_or(first[j]));
            }
            Restore(shadowed);
            out.push_back(Make<Scope>(x->tokens, Number(), ScopeKind::Trace,
                goal, Save(llvm::ArrayRef<VarId>{vars}), inner));
          },
          [&](const parser::TryGoal *x) { Todo(x->tokens, "try goals"); },
          [&](const parser::AtomicGoal *x) {
            Todo(x->tokens, "atomic goals");
          },
          [&](const parser::EventGoal *x) {
            llvm::ArrayRef<VarId> arguments{
                CallArguments(x->call->arguments, out)};
            out.push_back(Make<Call>(
                x->tokens, Number(), x->call->name, arguments, true));
            FinishAtomic();
          },
          [&](const parser::OmpGoal *x) {
            out.push_back(MakeScope(x->tokens, ScopeKind::Omp, goal, {}, x->goal));
          },
      },
      goal);
}

template <typename G>
void ClauseDesugarer::Quantify(const ParseGoal &origin, TokenRange range,
    parser::Quantifier quantifier, llvm::ArrayRef<parser::DataTerm> terms,
    const G &goal, Goals &out) {
  // A state variable !X among the variables of some is a new one inside
  // it, which starts with no version.
  bool some{quantifier == parser::Quantifier::Some};
  llvm::SmallVector<VarId, 4> vars;
  llvm::SmallVector<Shadowed, 2> shadowed;
  for (const parser::DataTerm &term : terms) {
    if (const auto *var{std::get_if<const parser::Variable *>(&term)}) {
      vars.push_back(ProgramVar(**var));
      continue;
    }
    const auto *state{std::get_if<const parser::StateVar *>(&term)};
    if (some && (*state)->kind == parser::StateVarKind::Both) {
      shadowed.push_back(Shadow((*state)->name, std::nullopt));
    } else {
      Error((*state)->tokens,
          some ? "only a variable or a state variable !X may be quantified "
                 "by some"_err_en_US
               : "only a variable may be quantified by all"_err_en_US);
    }
  }
  StateMap start{state_};
  Goal inner{DesugarGoal(goal)};
  Restore(shadowed);
  if (some) {
    out.push_back(Make<Scope>(range, Number(), ScopeKind::Some, origin,
        Save(llvm::ArrayRef<VarId>{vars}), inner));
    return;
  }
  // all Vars G is not (some Vars (not G)).
  state_ = start;
  Goal negated{Negate(range, inner)};
  Goal exists{Make<Scope>(range, Number(), ScopeKind::Some, origin,
      Save(llvm::ArrayRef<VarId>{vars}), negated)};
  out.push_back(Negate(range, exists));
}

void ClauseDesugarer::DesugarInto(const parser::DcgGoal &goal, Goals &out) {
  std::visit(
      common::visitors{
          [&](const parser::DcgConjunction *conj) {
            if (!conj->parallel) {
              for (const parser::DcgGoal &conjunct : conj->goals) {
                DesugarInto(conjunct, out);
              }
              return;
            }
            Goals conjuncts;
            for (const parser::DcgGoal &conjunct : conj->goals) {
              conjuncts.push_back(DesugarGoal(conjunct));
            }
            out.push_back(Make<Conj>(conj->tokens, Number(), true,
                Save(llvm::ArrayRef<Goal>{conjuncts})));
          },
          [&](const parser::DcgDisjunction *disj) {
            StateMap start{state_};
            llvm::SmallVector<Goals, 2> arms(disj->goals.size());
            llvm::SmallVector<StateMap, 2> ends;
            for (std::size_t j{0}; j < disj->goals.size(); ++j) {
              ends.push_back(Branch(disj->goals[j], start, arms[j]));
            }
            Merge(start, ends, arms, disj->tokens);
            Goals disjuncts;
            for (std::size_t j{0}; j < arms.size(); ++j) {
              disjuncts.push_back(Conjoin(GoalTokens(disj->goals[j]), arms[j]));
            }
            out.push_back(Make<Disj>(
                disj->tokens, Number(), Save(llvm::ArrayRef<Goal>{disjuncts})));
          },
          [&](const parser::DcgIfThenElse *ite) {
            StateMap start{state_};
            Goal condition{DesugarGoal(ite->condition)};
            StateMap afterCondition{state_};
            Goals arms[2];
            StateMap ends[2]{Branch(ite->then, afterCondition, arms[0]),
                Branch(ite->otherwise, start, arms[1])};
            Merge(start, ends, arms, ite->tokens);
            out.push_back(Make<IfThenElse>(ite->tokens, Number(), condition,
                Conjoin(GoalTokens(ite->then), arms[0]),
                Conjoin(GoalTokens(ite->otherwise), arms[1])));
          },
          [&](const parser::DcgNegation *negation) {
            StateMap start{state_};
            Goal negated{DesugarGoal(negation->goal)};
            state_ = start;
            out.push_back(Negate(negation->tokens, negated));
          },
          [&](const parser::DcgQuantified *quantified) {
            Quantify(goal, quantified->tokens, quantified->quantifier,
                quantified->vars, quantified->goal, out);
          },
          [&](const parser::DcgBraces *braces) {
            DesugarInto(braces->goal, out);
          },
          [&](const parser::DcgTerminals *terminals) {
            // [T1, T2] is V_in = [T1, T2 | V_out]; [] changes nothing.
            if (terminals->terms.empty()) {
              return;
            }
            VarId current{DcgCurrent()};
            VarId next{NewVersion(dcg, terminals->tokens)};
            UnifyList(current, terminals->tokens, terminals->terms,
                std::nullopt, out, next);
            FinishAtomic();
            state_[dcg] = next;
          },
          [&](const parser::DcgUnify *unify) {
            // =(T) is T = V_in; :=(T) is V_out = T.
            VarId var{unify->output ? NewVersion(dcg, unify->tokens)
                                    : DcgCurrent()};
            UnifyWith(var, unify->term, unify->tokens, out);
            FinishAtomic();
            state_[dcg] = var;
          },
          [&](const parser::DcgFieldSelection *x) {
            Todo(x->tokens, "field selection");
          },
          [&](const parser::DcgFieldUpdate *x) {
            Todo(x->tokens, "field update");
          },
          [&](const parser::DcgCall *call) {
            // p(A1, ..., An) is p(A1, ..., An, V_in, V_out); call(P, ...)
            // is a higher-order call.
            bool higherOrder{call->name->qualifiers.empty() &&
                call->name->name == "call" && !call->arguments.empty()};
            llvm::ArrayRef<parser::DataTerm> terms{call->arguments};
            std::optional<VarId> closure;
            if (higherOrder) {
              closure = Flatten(terms.front(), out);
              terms = terms.drop_front();
            }
            llvm::SmallVector<VarId, 4> arguments{CallArguments(terms, out)};
            arguments.push_back(DcgCurrent());
            arguments.push_back(DcgNext(call->tokens));
            llvm::ArrayRef<VarId> saved{Save(llvm::ArrayRef<VarId>{arguments})};
            if (closure) {
              out.push_back(Make<HOCall>(call->tokens, Number(),
                  parser::PredOrFunc::Predicate, *closure, saved));
            } else {
              out.push_back(
                  Make<Call>(call->tokens, Number(), call->name, saved, false));
            }
            FinishAtomic();
          },
          [&](const parser::DcgHigherOrderCall *call) {
            VarId closure{Flatten(call->closure, out)};
            llvm::SmallVector<VarId, 4> arguments{
                CallArguments(call->arguments, out)};
            arguments.push_back(DcgCurrent());
            arguments.push_back(DcgNext(call->tokens));
            out.push_back(Make<HOCall>(call->tokens, Number(),
                parser::PredOrFunc::Predicate, closure,
                Save(llvm::ArrayRef<VarId>{arguments})));
            FinishAtomic();
          },
      },
      goal);
}

void ClauseDesugarer::Head(llvm::ArrayRef<const parser::ModedTerm *> arguments,
    const parser::ModedTerm *result, Goals &goals) {
  // A state variable that the head names as !X, !.X or !:X is in scope in
  // the whole clause. Its first version is the head's !.X; its last is the
  // head's !:X, which is filled in after the body.
  auto headState{[](const parser::ModedTerm &moded) {
    const auto *state{std::get_if<const parser::StateVar *>(&moded.term)};
    return state ? *state : nullptr;
  }};
  for (const parser::ModedTerm *moded : arguments) {
    if (const parser::StateVar * state{headState(*moded)}) {
      std::optional<VarId> &first{state_[state->name]};
      if (state->kind != parser::StateVarKind::Next && !first) {
        first = NewVersion(state->name, state->tokens);
      }
    }
  }
  if (result) {
    if (const parser::StateVar * state{headState(*result)};
        state && state->kind == parser::StateVarKind::Both) {
      // mmc: "!X is not a legal function result".
      Error(state->tokens,
          "!%s cannot be a function's result, since it stands for two "
          "arguments, !.%s and !:%s"_err_en_US,
          state->name, state->name, state->name);
    }
  }
  // A head argument that is a variable not already among the head's stays
  // as it is; any other becomes a fresh variable, unified with the
  // argument at the start of the body.
  auto addHead{[&](VarId id, TokenRange range) {
    if (!llvm::is_contained(head_, id)) {
      head_.push_back(id);
      return;
    }
    VarId fresh{Fresh(range)};
    head_.push_back(fresh);
    goals.push_back(Make<Unify>(range, Number(), fresh, true, id, Functor{},
        llvm::ArrayRef<VarId>{}));
  }};
  for (const parser::ModedTerm *moded : arguments) {
    if (const parser::StateVar * state{headState(*moded)}) {
      if (state->kind != parser::StateVarKind::Next) {
        addHead(*state_[state->name], moded->tokens);
      }
      if (state->kind != parser::StateVarKind::Current) {
        lasts_.push_back(Last{head_.size(), state->name, state->tokens});
        head_.push_back(unfilled);
      }
    } else if (const auto *var{
                   std::get_if<const parser::Variable *>(&moded->term)}) {
      addHead(ProgramVar(**var), moded->tokens);
    } else {
      VarId fresh{Fresh(moded->tokens)};
      head_.push_back(fresh);
      UnifyWith(fresh, moded->term, moded->tokens, goals);
      FinishAtomic();
    }
  }
}

void ClauseDesugarer::DcgHead(TokenRange range) {
  VarId first{NewVersion(dcg, range)};
  state_[dcg] = first;
  head_.push_back(first);
  lasts_.push_back(Last{head_.size(), dcg, range});
  head_.push_back(unfilled);
}

const ClauseForm *ClauseDesugarer::Finish(TokenRange range,
    parser::PredOrFunc kind, const parser::SymName *name, Goals &goals) {
  // A state variable's last version is the head's !:X, unless it is
  // already in the head, as when the body does not update it.
  for (const Last &last : lasts_) {
    std::optional<VarId> version{state_.lookup(last.name)};
    if (version && !llvm::is_contained(head_, *version)) {
      head_[last.slot] = *version;
      continue;
    }
    VarId final{NewVersion(last.name, last.tokens)};
    head_[last.slot] = final;
    if (version) {
      goals.push_back(Make<Unify>(last.tokens, Number(), final, true,
          *version, Functor{}, llvm::ArrayRef<VarId>{}));
    }
  }
  Goal body{Conjoin(range, goals)};
  return Make<ClauseForm>(range, kind, name,
      Save(llvm::ArrayRef<VarId>{head_}), body,
      Save(llvm::ArrayRef<VarInfo>{vars_}),
      Save(llvm::ArrayRef<TypeQualification>{typeQualifications_}), goals_);
}

std::optional<const ClauseForm *> ClauseDesugarer::Desugar(
    const parser::FactOrRule &clause) {
  llvm::SmallVector<const parser::ModedTerm *, 4> arguments{
      clause.arguments.begin(), clause.arguments.end()};
  if (clause.result) {
    arguments.push_back(clause.result);
  }
  Goals goals;
  Head(arguments, clause.result, goals);
  if (clause.body) {
    DesugarInto(*clause.body, goals);
  }
  if (!ok_) {
    return std::nullopt;
  }
  return Finish(clause.tokens, clause.kind, clause.name, goals);
}

// Head --> Body is Head(V_in, V_out) :- Body, with V_in and V_out the
// first and last versions of the hidden state variable.
std::optional<const ClauseForm *> ClauseDesugarer::Desugar(
    const parser::DcgRule &rule) {
  Goals goals;
  Head(rule.arguments, nullptr, goals);
  DcgHead(rule.tokens);
  DesugarInto(rule.body, goals);
  if (!ok_) {
    return std::nullopt;
  }
  return Finish(rule.tokens, parser::PredOrFunc::Predicate, rule.name, goals);
}

} // namespace

const SemanticForm &Desugar(const parser::ParseTree &tree,
    const parser::TokenSequence &tokens, SemanticArena &arena,
    parser::Messages &messages) {
  llvm::SmallVector<const ClauseForm *, 16> clauses;
  for (const parser::Item &item : tree.items) {
    if (const auto *rule{std::get_if<const parser::FactOrRule *>(&item)}) {
      ClauseDesugarer desugarer{tokens, arena, messages};
      if (auto clause{desugarer.Desugar(**rule)}) {
        clauses.push_back(*clause);
      }
    } else if (const auto *rule{std::get_if<const parser::DcgRule *>(&item)}) {
      ClauseDesugarer desugarer{tokens, arena, messages};
      if (auto clause{desugarer.Desugar(**rule)}) {
        clauses.push_back(*clause);
      }
    }
  }
  return *arena.Make<SemanticForm>(
      arena.SaveArray(llvm::ArrayRef<const ClauseForm *>{clauses}));
}

} // namespace merkurius::semantics
