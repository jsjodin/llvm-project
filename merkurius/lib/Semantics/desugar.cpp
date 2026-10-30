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

#include "merkurius/Semantics/desugar.h"
#include "merkurius/Common/idioms.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>
#include <vector>

namespace merkurius::semantics {
namespace {

using namespace parser::literals;
using Goals = llvm::SmallVector<Goal, 4>;

class ClauseDesugarer {
public:
  ClauseDesugarer(const parser::TokenSequence &tokens, SemanticArena &arena,
      parser::Messages &messages)
      : tokens_{tokens}, arena_{arena}, messages_{messages} {}

  std::optional<const ClauseForm *> Desugar(const parser::FactOrRule &);

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
  // Reports the first form in the clause that is not desugared yet.
  void Todo(TokenRange range, const char *what) {
    if (ok_) {
      messages_.Say(Source(range), "desugaring %s"_todo_en_US, what);
      ok_ = false;
    }
  }

  VarId ProgramVar(const parser::Variable &);
  VarId Fresh(TokenRange);
  // A variable that holds the term, after the goals added to `out`.
  VarId Flatten(const parser::DataTerm &, Goals &out);
  llvm::ArrayRef<VarId> FlattenAll(
      llvm::ArrayRef<parser::DataTerm>, Goals &out);
  // Adds the goals for Var = Term to `out`.
  void UnifyWith(
      VarId, const parser::DataTerm &, TokenRange, Goals &out);
  void UnifyList(VarId, TokenRange, llvm::ArrayRef<parser::DataTerm> elements,
      const std::optional<parser::DataTerm> &tail, Goals &out);
  void AddUnify(Goals &out, std::size_t at, TokenRange range, VarId left,
      Functor functor, llvm::ArrayRef<VarId> arguments);

  Goal DesugarGoal(const parser::Goal &);
  // Adds the goal to `out`, as conjuncts when it is a conjunction.
  void DesugarInto(const parser::Goal &, Goals &out);
  Goal Conjoin(TokenRange, const Goals &);
  Goal Negate(TokenRange range, Goal goal) {
    return Make<Not>(range, Number(), goal);
  }
  std::optional<llvm::ArrayRef<VarId>> ScopeVars(
      llvm::ArrayRef<parser::DataTerm>);
  Goal MakeScope(TokenRange range, ScopeKind kind, const parser::Goal &origin,
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

VarId ClauseDesugarer::Flatten(const parser::DataTerm &term, Goals &out) {
  if (const auto *var{std::get_if<const parser::Variable *>(&term)}) {
    return ProgramVar(**var);
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
          [&](const parser::StateVar *x) { Todo(x->tokens, "state variables"); },
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
    const std::optional<parser::DataTerm> &tail, Goals &out) {
  std::size_t at{out.size()};
  if (elements.empty()) {
    if (tail) {
      UnifyWith(left, *tail, range, out);
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
  } else {
    rest = Fresh(range);
    UnifyList(rest, range, elements.drop_front(), tail, out);
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

Goal ClauseDesugarer::DesugarGoal(const parser::Goal &goal) {
  Goals goals;
  DesugarInto(goal, goals);
  return Conjoin(
      std::visit([](const auto *x) { return x->tokens; }, goal), goals);
}

std::optional<llvm::ArrayRef<VarId>> ClauseDesugarer::ScopeVars(
    llvm::ArrayRef<parser::DataTerm> terms) {
  llvm::SmallVector<VarId, 4> vars;
  for (const parser::DataTerm &term : terms) {
    if (const auto *var{std::get_if<const parser::Variable *>(&term)}) {
      vars.push_back(ProgramVar(**var));
    } else {
      Todo(std::visit([](const auto *x) { return x->tokens; }, term),
          "state variables");
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
            Goals disjuncts;
            for (const parser::Goal &disjunct : disj->goals) {
              disjuncts.push_back(DesugarGoal(disjunct));
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
            out.push_back(
                Negate(negation->tokens, DesugarGoal(negation->goal)));
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
          },
          [&](const parser::IfThenElse *ite) {
            Goal condition{DesugarGoal(ite->condition)};
            Goal then{DesugarGoal(ite->then)};
            Goal otherwise{DesugarGoal(ite->otherwise)};
            out.push_back(Make<IfThenElse>(
                ite->tokens, Number(), condition, then, otherwise));
          },
          [&](const parser::Unification *unification) {
            const auto *left{
                std::get_if<const parser::Variable *>(&unification->left)};
            if (left) {
              UnifyWith(ProgramVar(**left), unification->right,
                  unification->tokens, out);
            } else if (const auto *right{std::get_if<const parser::Variable *>(
                           &unification->right)}) {
              UnifyWith(ProgramVar(**right), unification->left,
                  unification->tokens, out);
            } else {
              VarId both{Flatten(unification->left, out)};
              UnifyWith(both, unification->right, unification->tokens, out);
            }
          },
          [&](const parser::Inequality *inequality) {
            // X \= Y is not (X = Y).
            Goals unify;
            VarId left{Flatten(inequality->left, unify)};
            UnifyWith(left, inequality->right, inequality->tokens, unify);
            out.push_back(Negate(
                inequality->tokens, Conjoin(inequality->tokens, unify)));
          },
          [&](const parser::Call *call) {
            if (call->name->qualifiers.empty() && call->name->name == "call" &&
                !call->arguments.empty()) {
              VarId closure{Flatten(call->arguments.front(), out)};
              llvm::ArrayRef<VarId> arguments{
                  FlattenAll(call->arguments.drop_front(), out)};
              out.push_back(Make<HOCall>(call->tokens, Number(),
                  parser::PredOrFunc::Predicate, closure, arguments));
              return;
            }
            llvm::ArrayRef<VarId> arguments{FlattenAll(call->arguments, out)};
            out.push_back(Make<Call>(
                call->tokens, Number(), call->name, arguments, false));
          },
          [&](const parser::HigherOrderCall *call) {
            VarId closure{Flatten(call->closure, out)};
            llvm::ArrayRef<VarId> arguments{FlattenAll(call->arguments, out)};
            out.push_back(Make<HOCall>(call->tokens, Number(),
                parser::PredOrFunc::Predicate, closure, arguments));
          },
          [&](const parser::QuantifiedGoal *quantified) {
            auto vars{ScopeVars(quantified->vars)};
            if (!vars) {
              return;
            }
            if (quantified->quantifier == parser::Quantifier::Some) {
              out.push_back(MakeScope(quantified->tokens, ScopeKind::Some, goal,
                  *vars, quantified->goal));
              return;
            }
            // all Vars G is not (some Vars (not G)).
            Goal inner{Negate(quantified->tokens, DesugarGoal(quantified->goal))};
            Goal some{Make<Scope>(quantified->tokens, Number(), ScopeKind::Some,
                goal, *vars, inner)};
            out.push_back(Negate(quantified->tokens, some));
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
            // io(!IO) and state(M, !S) parameters name state variables.
            for (const parser::TraceParam *param : x->params) {
              if (param->var) {
                Todo(param->tokens, "state variables");
                return;
              }
            }
            out.push_back(
                MakeScope(x->tokens, ScopeKind::Trace, goal, {}, x->goal));
          },
          [&](const parser::TryGoal *x) { Todo(x->tokens, "try goals"); },
          [&](const parser::AtomicGoal *x) {
            Todo(x->tokens, "atomic goals");
          },
          [&](const parser::EventGoal *x) {
            llvm::ArrayRef<VarId> arguments{
                FlattenAll(x->call->arguments, out)};
            out.push_back(Make<Call>(
                x->tokens, Number(), x->call->name, arguments, true));
          },
          [&](const parser::OmpGoal *x) {
            out.push_back(MakeScope(x->tokens, ScopeKind::Omp, goal, {}, x->goal));
          },
      },
      goal);
}

std::optional<const ClauseForm *> ClauseDesugarer::Desugar(
    const parser::FactOrRule &clause) {
  // A head argument that is a variable not already among the head's stays
  // as it is; any other becomes a fresh variable, unified with the
  // argument at the start of the body.
  Goals goals;
  llvm::SmallVector<VarId, 4> head;
  auto argument{[&](const parser::ModedTerm &moded) {
    if (const auto *var{std::get_if<const parser::Variable *>(&moded.term)}) {
      VarId id{ProgramVar(**var)};
      if (!llvm::is_contained(head, id)) {
        head.push_back(id);
        return;
      }
    }
    VarId fresh{Fresh(moded.tokens)};
    head.push_back(fresh);
    UnifyWith(fresh, moded.term, moded.tokens, goals);
  }};
  for (const parser::ModedTerm *moded : clause.arguments) {
    argument(*moded);
  }
  if (clause.result) {
    argument(*clause.result);
  }
  if (clause.body) {
    DesugarInto(*clause.body, goals);
  }
  if (!ok_) {
    return std::nullopt;
  }
  Goal body{Conjoin(clause.tokens, goals)};
  return Make<ClauseForm>(clause.tokens, clause.kind, clause.name,
      Save(llvm::ArrayRef<VarId>{head}), body,
      Save(llvm::ArrayRef<VarInfo>{vars_}),
      Save(llvm::ArrayRef<TypeQualification>{typeQualifications_}), goals_);
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
    } else if (const auto *dcg{std::get_if<const parser::DcgRule *>(&item)}) {
      messages.Say(parser::CharBlock{tokens.tokens[(*dcg)->tokens.first]
                                         .spelling.begin(),
                       tokens.tokens[(*dcg)->tokens.last].spelling.end()},
          "desugaring %s"_todo_en_US, "DCG rules");
    }
  }
  return *arena.Make<SemanticForm>(
      arena.SaveArray(llvm::ArrayRef<const ClauseForm *>{clauses}));
}

} // namespace merkurius::semantics
