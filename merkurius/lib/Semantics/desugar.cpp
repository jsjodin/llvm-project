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
#include "merkurius/Semantics/quantify.h"
#include "merkurius/Common/idioms.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
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
  // pred(m/N) is p as the clause m(X1, ..., XN) :- p(X1, ..., XN), and
  // func(m/N) is f as m(X1, ..., XN) = Y :- Y = f(X1, ..., XN), as mmc
  // checks them: f is a data-term.
  const ClauseForm *Desugar(const parser::MethodByName &);

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
  // The state variables in scope, in the order they came into scope, each
  // with its current version, which it lacks until it is first given one.
  using StateMap = llvm::MapVector<llvm::StringRef, std::optional<VarId>>;
  // A state variable brought into scope by a quantifier or a trace
  // parameter, hiding any of the same name, until Restore.
  struct Shadowed {
    llvm::StringRef name;
    bool wasInScope;
    std::optional<VarId> was;
    bool wasReadOnly;
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
  // Term ^ F1 ^ ... ^ Fn as Fn(..., F1(..., Term)), and Term ^ Fields :=
  // Value as calls to 'F :=' (reference manual, "Record syntax"). The term
  // is `term` or, if that is null, `termVar`.
  void Select(VarId left, const parser::DataTerm *term, VarId termVar,
      llvm::ArrayRef<const parser::FieldSpec *> fields, TokenRange,
      Goals &out);
  void Update(VarId left, const parser::DataTerm *term, VarId termVar,
      llvm::ArrayRef<const parser::FieldSpec *> fields,
      const parser::DataTerm &value, TokenRange, Goals &out);
  // 'F :=', with the qualifiers of F.
  const parser::SymName *UpdateName(const parser::FieldSpec &);
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
      const parser::SymName *, Goals &goals, bool modeSpecific = false);
  // Fills the head's slots for the state variables' last versions.
  void FillLasts(Goals &goals);
  const Lambda *DesugarLambda(const parser::Lambda &);
  void DesugarTry(const parser::TryGoal &, Goals &out);
  void DesugarAtomic(const parser::AtomicGoal &, Goals &out);
  Goal Conjoin(TokenRange, const Goals &);
  // As mmc has it, the negation of a conjunction of negations is the
  // disjunction of their goals, and the negation of a negation its goal,
  // which may then bind variables used outside (see the register).
  Goal Negate(TokenRange range, Goal goal) {
    std::vector<Goal> conjuncts{goal};
    for (std::size_t i{0}; i < conjuncts.size();) {
      const auto *conj{std::get_if<const Conj *>(&conjuncts[i])};
      if (conj && !(*conj)->parallel) {
        llvm::ArrayRef<Goal> inner{(*conj)->goals};
        conjuncts.erase(conjuncts.begin() + i);
        conjuncts.insert(conjuncts.begin() + i, inner.begin(), inner.end());
      } else {
        ++i;
      }
    }
    if (!conjuncts.empty() && llvm::all_of(conjuncts, [](const Goal &g) {
          return std::holds_alternative<const Not *>(g);
        })) {
      Goals disjuncts;
      for (const Goal &g : conjuncts) {
        disjuncts.push_back(std::get<const Not *>(g)->goal);
      }
      if (disjuncts.size() == 1) {
        return disjuncts.front();
      }
      return Make<Disj>(range, Number(), Save(llvm::ArrayRef<Goal>{disjuncts}));
    }
    return Make<Not>(range, Number(), goal);
  }
  Goal ListScope(TokenRange, ScopeKind, const parser::Goal &origin,
      llvm::ArrayRef<parser::DataTerm> vars, const parser::Goal &goal);
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
  // The annotation of the goal being desugared, which a higher-order
  // predicate call in it has.
  parser::Purity annotation_{parser::Purity::Pure};
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
  // The state variables of the scopes outside a lambda expression, which
  // may only be read as !.X in it.
  llvm::StringSet<> readOnly_;
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
  if (next && readOnly_.contains(var.name)) {
    // mmc: "cannot use !:S here due to the surrounding lambda expression".
    Error(var.tokens,
        "!:%s cannot be used in a lambda expression, which may only read "
        "the state variables outside it, as !.%s"_err_en_US,
        var.name, var.name);
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
      it != state_.end() ? it->second : std::nullopt, readOnly_.erase(name)};
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
    if (s.wasReadOnly) {
      readOnly_.insert(s.name);
    }
  }
}

void ClauseDesugarer::Merge(const StateMap &start,
    llvm::ArrayRef<StateMap> ends, llvm::MutableArrayRef<Goals> arms,
    TokenRange range) {
  state_ = start;
  for (auto &entry : state_) {
    llvm::StringRef name{entry.first};
    std::optional<VarId> &current{entry.second};
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
            // call(P, ...) is a higher-order predicate call, and
            // impure_apply and semipure_apply apply impure and semipure
            // functions.
            llvm::StringRef name{data->name->name};
            if (data->name->qualifiers.empty() &&
                (name == "apply" || name == "impure_apply" ||
                    name == "semipure_apply") &&
                !data->arguments.empty()) {
              VarId closure{Flatten(data->arguments.front(), out)};
              llvm::SmallVector<VarId, 4> arguments{
                  FlattenAll(data->arguments.drop_front(), out)};
              arguments.push_back(left);
              out.push_back(Make<HOCall>(range, Number(),
                  parser::PredOrFunc::Function, closure,
                  Save(llvm::ArrayRef<VarId>{arguments}),
                  name == "impure_apply"         ? parser::Purity::Impure
                      : name == "semipure_apply" ? parser::Purity::Semipure
                                                 : parser::Purity::Pure));
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
          [&](const parser::ConditionalData *conditional) {
            // As in mmc, X = (if C then T else E) is if C then X = T else
            // X = E, with the state variables threaded as in that goal.
            auto next{std::move(next_)};
            next_.clear();
            StateMap start{state_};
            Goal condition{DesugarGoal(conditional->condition)};
            Goals arms[2];
            StateMap ends[2];
            UnifyWith(left, conditional->then, range, arms[0]);
            ends[0] = state_;
            state_ = start;
            UnifyWith(left, conditional->otherwise, range, arms[1]);
            ends[1] = state_;
            Merge(start, ends, arms, conditional->tokens);
            next_ = std::move(next);
            out.push_back(Make<IfThenElse>(conditional->tokens, Number(),
                condition, Conjoin(conditional->tokens, arms[0]),
                Conjoin(conditional->tokens, arms[1])));
          },
          [&](const parser::FieldSelection *selection) {
            Select(left, &selection->term, VarId{0}, selection->fields, range,
                out);
          },
          [&](const parser::FieldUpdate *update) {
            Update(left, &update->term, VarId{0}, update->fields,
                update->value, range, out);
          },
          [&](const parser::Lambda *lambda) {
            Functor functor{FunctorKind::Lambda, nullptr, nullptr, nullptr,
                DesugarLambda(*lambda)};
            AddUnify(out, at, range, left, functor, {});
          },
      },
      term);
}

void ClauseDesugarer::Select(VarId left, const parser::DataTerm *term,
    VarId termVar, llvm::ArrayRef<const parser::FieldSpec *> fields,
    TokenRange range, Goals &out) {
  std::size_t at{out.size()};
  const parser::FieldSpec &last{*fields.back()};
  VarId inner;
  if (fields.size() > 1) {
    inner = Fresh(range);
    Select(inner, term, termVar, fields.drop_back(), range, out);
  } else {
    inner = term ? Flatten(*term, out) : termVar;
  }
  llvm::SmallVector<VarId, 4> arguments{FlattenAll(last.arguments, out)};
  arguments.push_back(inner);
  AddUnify(out, at, range, left,
      Functor{FunctorKind::Name, last.name, nullptr, nullptr},
      Save(llvm::ArrayRef<VarId>{arguments}));
}

// Term ^ F := Value is 'F :='(Term, Value); with more fields,
// Term0 ^ F ^ Rest := Value is
//   Old = F(Term0), New = (Old ^ Rest := Value), Term = 'F :='(Term0, New).
void ClauseDesugarer::Update(VarId left, const parser::DataTerm *term,
    VarId termVar, llvm::ArrayRef<const parser::FieldSpec *> fields,
    const parser::DataTerm &value, TokenRange range, Goals &out) {
  std::size_t at{out.size()};
  const parser::FieldSpec &first{*fields.front()};
  VarId whole{term ? Flatten(*term, out) : termVar};
  llvm::SmallVector<VarId, 4> arguments{FlattenAll(first.arguments, out)};
  const parser::SymName *update{UpdateName(first)};
  if (fields.size() == 1) {
    arguments.push_back(whole);
    arguments.push_back(Flatten(value, out));
    AddUnify(out, at, range, left,
        Functor{FunctorKind::Name, update, nullptr, nullptr},
        Save(llvm::ArrayRef<VarId>{arguments}));
    return;
  }
  llvm::SmallVector<VarId, 4> selectArguments{arguments};
  selectArguments.push_back(whole);
  VarId old{Fresh(range)};
  AddUnify(out, out.size(), range, old,
      Functor{FunctorKind::Name, first.name, nullptr, nullptr},
      Save(llvm::ArrayRef<VarId>{selectArguments}));
  VarId updated{Fresh(range)};
  Update(updated, nullptr, old, fields.drop_front(), value, range, out);
  arguments.push_back(whole);
  arguments.push_back(updated);
  AddUnify(out, out.size(), range, left,
      Functor{FunctorKind::Name, update, nullptr, nullptr},
      Save(llvm::ArrayRef<VarId>{arguments}));
}

const parser::SymName *ClauseDesugarer::UpdateName(
    const parser::FieldSpec &field) {
  return Make<parser::SymName>(field.name->tokens, field.name->qualifiers,
      arena_.Save((field.name->name + " :=").str()));
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

// As in mmc, !.S in the list is the version before the goal, !:S the one
// after it, and !S both.
Goal ClauseDesugarer::ListScope(TokenRange range, ScopeKind kind,
    const parser::Goal &origin, llvm::ArrayRef<parser::DataTerm> terms,
    const parser::Goal &goal) {
  llvm::SmallVector<std::optional<VarId>, 4> before;
  for (const parser::DataTerm &term : terms) {
    const auto *state{std::get_if<const parser::StateVar *>(&term)};
    before.push_back(state && (*state)->kind != parser::StateVarKind::Next
            ? std::optional<VarId>{StateRef(**state, false)}
            : std::nullopt);
  }
  Goal inner{DesugarGoal(goal)};
  llvm::SmallVector<VarId, 4> vars;
  for (std::size_t j{0}; j < terms.size(); ++j) {
    const auto *state{std::get_if<const parser::StateVar *>(&terms[j])};
    if (!state) {
      vars.push_back(*SimpleVar(terms[j]));
      continue;
    }
    if (before[j]) {
      vars.push_back(*before[j]);
    }
    if ((*state)->kind != parser::StateVarKind::Current) {
      std::optional<VarId> after{state_.lookup((*state)->name)};
      vars.push_back(after ? *after : StateRef(**state, false));
    }
  }
  return Make<Scope>(
      range, Number(), kind, origin, Save(llvm::ArrayRef<VarId>{vars}), inner);
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
            // X = f(...) has the tokens of f(...), as a subterm's does, and
            // X = Y those of the goal.
            auto range{[&](const parser::DataTerm &term) {
              return std::holds_alternative<const parser::Variable *>(term) ||
                      std::holds_alternative<const parser::StateVar *>(term)
                  ? unification->tokens
                                     : std::visit([](const auto *x) {
                                         return x->tokens;
                                       }, term);
            }};
            if (std::optional<VarId> left{SimpleVar(unification->left)}) {
              UnifyWith(*left, unification->right, range(unification->right),
                  out);
            } else if (std::optional<VarId> right{
                           SimpleVar(unification->right)}) {
              UnifyWith(*right, unification->left, range(unification->left),
                  out);
            } else {
              VarId both{Flatten(unification->left, out)};
              UnifyWith(both, unification->right, range(unification->right),
                  out);
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
          [&](const parser::StateVarFieldUpdate *update) {
            // !S ^ Fields := Value is !:S = !.S ^ Fields := Value.
            const auto &var{*std::get<const parser::StateVar *>(update->var)};
            VarId current{StateRef(var, false)};
            VarId next{StateRef(var, true)};
            Update(next, nullptr, current, update->fields, update->value,
                update->tokens, out);
            FinishAtomic();
          },
          [&](const parser::Call *call) {
            if (call->name->qualifiers.empty() && call->name->name == "call" &&
                !call->arguments.empty()) {
              VarId closure{Flatten(call->arguments.front(), out)};
              llvm::ArrayRef<VarId> arguments{
                  CallArguments(call->arguments.drop_front(), out)};
              out.push_back(Make<HOCall>(call->tokens, Number(),
                  parser::PredOrFunc::Predicate, closure, arguments,
                  annotation_));
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
                parser::PredOrFunc::Predicate, closure, arguments,
                annotation_));
            FinishAtomic();
          },
          [&](const parser::QuantifiedGoal *quantified) {
            Quantify(goal, quantified->tokens, quantified->quantifier,
                quantified->vars, quantified->goal, out);
          },
          [&](const parser::PurityGoal *x) {
            parser::Purity saved{annotation_};
            annotation_ = x->purity;
            out.push_back(
                MakeScope(x->tokens, ScopeKind::Purity, goal, {}, x->goal));
            annotation_ = saved;
          },
          [&](const parser::PromisePurity *x) {
            out.push_back(MakeScope(
                x->tokens, ScopeKind::PromisePurity, goal, {}, x->goal));
          },
          [&](const parser::PromiseSolutions *x) {
            out.push_back(ListScope(
                x->tokens, ScopeKind::PromiseSolutions, goal, x->vars, x->goal));
          },
          [&](const parser::RequireDeterminism *x) {
            out.push_back(MakeScope(
                x->tokens, ScopeKind::RequireDeterminism, goal, {}, x->goal));
          },
          [&](const parser::RequireSwitch *x) {
            out.push_back(ListScope(x->tokens, ScopeKind::RequireSwitch, goal,
                llvm::ArrayRef<parser::DataTerm>{x->var}, x->goal));
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
          [&](const parser::TryGoal *t) { DesugarTry(*t, out); },
          [&](const parser::AtomicGoal *atomic) {
            DesugarAtomic(*atomic, out);
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
          [&](const parser::DcgFieldSelection *selection) {
            // Term =^ Fields is Term = V_in ^ Fields, V_out = V_in.
            std::optional<VarId> var{SimpleVar(selection->term)};
            VarId left{var ? *var : Fresh(selection->tokens)};
            Select(left, nullptr, DcgCurrent(), selection->fields,
                selection->tokens, out);
            if (!var) {
              UnifyWith(left, selection->term, selection->tokens, out);
            }
            FinishAtomic();
          },
          [&](const parser::DcgFieldUpdate *update) {
            // ^ Fields := Term is V_out = V_in ^ Fields := Term.
            VarId current{DcgCurrent()};
            VarId next{NewVersion(dcg, update->tokens)};
            Update(next, nullptr, current, update->fields, update->value,
                update->tokens, out);
            FinishAtomic();
            state_[dcg] = next;
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
    parser::PredOrFunc kind, const parser::SymName *name, Goals &goals,
    bool modeSpecific) {
  FillLasts(goals);
  Goal body{Conjoin(range, goals)};
  return Make<ClauseForm>(range, kind, name,
      Save(llvm::ArrayRef<VarId>{head_}), body,
      Save(llvm::ArrayRef<VarInfo>{vars_}),
      Save(llvm::ArrayRef<TypeQualification>{typeQualifications_}), goals_,
      llvm::ArrayRef<llvm::ArrayRef<VarId>>{}, modeSpecific);
}

void ClauseDesugarer::FillLasts(Goals &goals) {
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
}

// The goal's updates of state variables are seen in the then part only, as
// in an if-then-else: as in mmc, a catch branch sees the versions before
// the goal. An io(!S) parameter's state is the exception: it is threaded
// through the goal, which cannot undo it, so every branch continues from
// the version after the goal.
void ClauseDesugarer::DesugarTry(const parser::TryGoal &t, Goals &out) {
  llvm::SmallVector<std::pair<llvm::StringRef, VarId>, 2> io;
  for (const parser::DataTerm &term : t.io) {
    const auto &var{*std::get<const parser::StateVar *>(term)};
    io.emplace_back(var.name, StateRef(var, false));
  }
  StateMap start{state_};
  Goal goal{DesugarGoal(t.goal)};
  StateMap afterGoal{state_};
  llvm::SmallVector<VarId, 4> ioVersions;
  for (auto &[name, before] : io) {
    // Not in scope, which StateRef reported, the state stays as it was.
    std::optional<VarId> after{afterGoal.lookup(name)};
    ioVersions.push_back(before);
    ioVersions.push_back(after ? *after : before);
  }
  // The state that the else part and the catch branches start from.
  StateMap failed{start};
  for (auto &[name, before] : io) {
    failed[name] = afterGoal.lookup(name);
  }
  llvm::SmallVector<Goals, 4> arms;
  llvm::SmallVector<StateMap, 4> ends;
  arms.emplace_back();
  ends.push_back(Branch(t.then, afterGoal, arms.back()));
  if (t.otherwise) {
    arms.emplace_back();
    ends.push_back(Branch(*t.otherwise, failed, arms.back()));
  }
  // A catch's pattern is unified with the exception, a variable of its own.
  llvm::SmallVector<std::pair<VarId, Goal>, 4> patterns;
  auto catchBranch{[&](const parser::CatchClause &c) {
    state_ = failed;
    Goals pattern;
    std::optional<VarId> var{SimpleVar(c.pattern)};
    VarId exception{var ? *var : Fresh(c.tokens)};
    if (!var) {
      UnifyWith(exception, c.pattern, c.tokens, pattern);
      FinishAtomic();
    }
    patterns.emplace_back(exception, Conjoin(c.tokens, pattern));
    arms.emplace_back();
    DesugarInto(c.goal, arms.back());
    ends.push_back(state_);
  }};
  for (const parser::CatchClause *c : t.catches) {
    catchBranch(*c);
  }
  if (t.catchAny) {
    catchBranch(*t.catchAny);
  }
  Merge(start, ends, arms, t.tokens);
  std::size_t arm{0};
  Goal then{Conjoin(GoalTokens(t.then), arms[arm++])};
  std::optional<Goal> otherwise;
  if (t.otherwise) {
    otherwise = Conjoin(GoalTokens(*t.otherwise), arms[arm++]);
  }
  llvm::SmallVector<Catch, 4> catches;
  std::optional<Catch> catchAny;
  std::size_t j{0};
  for (const parser::CatchClause *c : t.catches) {
    catches.push_back(Catch{c->tokens, patterns[j].first, patterns[j].second,
        Conjoin(GoalTokens(c->goal), arms[arm++])});
    ++j;
  }
  if (t.catchAny) {
    catchAny = Catch{t.catchAny->tokens, patterns[j].first,
        patterns[j].second, Conjoin(GoalTokens(t.catchAny->goal), arms[arm])};
  }
  out.push_back(Make<Try>(t.tokens, Number(),
      Save(llvm::ArrayRef<VarId>{ioVersions}), goal, then, otherwise,
      Save(llvm::ArrayRef<Catch>{catches}), catchAny));
}

// The outer state is read before the goal and made new after it, and, as
// in mmc, it is not visible inside. Each alternative has inner state of
// its own: !S names a new state variable in it, and S0, S the same two
// variables in each.
void ClauseDesugarer::DesugarAtomic(
    const parser::AtomicGoal &atomic, Goals &out) {
  const parser::AtomicParam *outer{nullptr}, *inner{nullptr};
  llvm::SmallVector<VarId, 4> vars;
  for (const parser::AtomicParam *param : atomic.params) {
    switch (param->kind) {
    case parser::AtomicParamKind::Outer:
      outer = param;
      break;
    case parser::AtomicParamKind::Inner:
      inner = param;
      break;
    case parser::AtomicParamKind::Vars:
      for (const parser::DataTerm &var : param->vars) {
        vars.push_back(*SimpleVar(var));
      }
      break;
    }
  }
  // The converter requires both outer and inner.
  auto stateVar{[](const parser::AtomicParam &param) {
    const auto *var{std::get_if<const parser::StateVar *>(&*param.initial)};
    return var ? *var : nullptr;
  }};
  const parser::StateVar *outerState{stateVar(*outer)};
  VarId outerInitial{outerState ? StateRef(*outerState, false)
                                : *SimpleVar(*outer->initial)};
  VarId outerFinal{outerState ? NewVersion(outerState->name, outer->tokens)
                              : *SimpleVar(*outer->final)};
  llvm::SmallVector<Shadowed, 1> hidden;
  if (outerState) {
    auto it{state_.find(outerState->name)};
    if (it != state_.end()) {
      hidden.push_back(Shadowed{outerState->name, true, it->second,
          readOnly_.erase(outerState->name)});
      state_.erase(it);
    }
  }
  const parser::StateVar *innerState{stateVar(*inner)};
  StateMap start{state_};
  llvm::SmallVector<Goals, 2> arms(atomic.goals.size());
  llvm::SmallVector<StateMap, 2> ends;
  llvm::SmallVector<std::pair<VarId, VarId>, 2> innerVersions;
  for (std::size_t j{0}; j < atomic.goals.size(); ++j) {
    const parser::Goal &alternative{atomic.goals[j]};
    state_ = start;
    Goals &goals{arms[j]};
    VarId innerInitial, innerFinal;
    if (innerState) {
      Shadowed shadowed{
          Shadow(innerState->name, NewVersion(innerState->name, inner->tokens))};
      innerInitial = *state_.lookup(innerState->name);
      DesugarInto(alternative, goals);
      std::optional<VarId> last{state_.lookup(innerState->name)};
      if (last && *last != innerInitial) {
        innerFinal = *last;
      } else {
        innerFinal = NewVersion(innerState->name, inner->tokens);
        goals.push_back(Make<Unify>(inner->tokens, Number(), innerFinal, true,
            innerInitial, Functor{}, llvm::ArrayRef<VarId>{}));
      }
      Restore(shadowed);
    } else {
      innerInitial = *SimpleVar(*inner->initial);
      innerFinal = *SimpleVar(*inner->final);
      DesugarInto(alternative, goals);
    }
    innerVersions.emplace_back(innerInitial, innerFinal);
    ends.push_back(state_);
  }
  // The other state variables are threaded as through a disjunction.
  Merge(start, ends, arms, atomic.tokens);
  llvm::SmallVector<AtomicAlternative, 2> alternatives;
  for (std::size_t j{0}; j < arms.size(); ++j) {
    alternatives.push_back(AtomicAlternative{innerVersions[j].first,
        innerVersions[j].second,
        Conjoin(GoalTokens(atomic.goals[j]), arms[j])});
  }
  Restore(hidden);
  if (outerState) {
    state_[outerState->name] = outerFinal;
  }
  out.push_back(Make<Atomic>(atomic.tokens, Number(), outerInitial,
      outerFinal, Save(llvm::ArrayRef<VarId>{vars}),
      Save(llvm::ArrayRef<AtomicAlternative>{alternatives})));
}

// A lambda expression is desugared like a clause, with a head of its own.
// The variables of its arguments are local to it, and so are the state
// variables its head names; the other state variables in scope may only be
// read in it.
const Lambda *ClauseDesugarer::DesugarLambda(const parser::Lambda &lambda) {
  // The variables of the arguments, as new variables while in the lambda.
  struct Collector {
    llvm::SmallVector<unsigned, 4> numbers;
    void Post(const parser::Variable &var) { numbers.push_back(var.number); }
  } collector;
  for (const parser::ModedTerm *argument : lambda.arguments) {
    parser::Walk(argument->term, collector);
  }
  llvm::SmallVector<std::pair<unsigned, std::optional<VarId>>, 4> outer;
  for (unsigned number : collector.numbers) {
    auto it{numbers_.find(number)};
    if (it != numbers_.end()) {
      outer.emplace_back(number, it->second);
      numbers_.erase(it);
    } else {
      outer.emplace_back(number, std::nullopt);
    }
  }
  StateMap state{state_};
  auto next{std::move(next_)};
  next_.clear();
  auto head{std::move(head_)};
  head_.clear();
  auto lasts{std::move(lasts_)};
  lasts_.clear();
  llvm::StringSet<> readOnly{readOnly_};
  for (const auto &entry : state_) {
    readOnly_.insert(entry.first);
  }
  // The head's !.X and !:X name new state variables; !X cannot be an
  // argument, since a lambda has no syntax for the modes of its two parts.
  for (const parser::ModedTerm *argument : lambda.arguments) {
    if (const auto *var{std::get_if<const parser::StateVar *>(&argument->term)}) {
      if ((*var)->kind == parser::StateVarKind::Both) {
        // mmc: "!S cannot be a lambda argument".
        Error((*var)->tokens,
            "!%s cannot be an argument of a lambda expression; you probably "
            "meant !.%s or !:%s"_err_en_US,
            (*var)->name, (*var)->name, (*var)->name);
      }
      state_[(*var)->name] = std::nullopt;
      readOnly_.erase((*var)->name);
    }
  }
  llvm::SmallVector<const parser::ModedTerm *, 4> arguments{
      lambda.arguments.begin(), lambda.arguments.end()};
  if (lambda.result) {
    arguments.push_back(lambda.result);
  }
  Goals goals;
  // The body is not under the goal's annotation.
  parser::Purity savedAnnotation{annotation_};
  annotation_ = parser::Purity::Pure;
  Head(arguments, lambda.result, goals);
  llvm::SmallVector<std::optional<parser::Mode>, 4> modes;
  for (const parser::ModedTerm *argument : arguments) {
    modes.push_back(argument->mode);
  }
  if (lambda.dcgBody) {
    state_.erase(dcg);
    DcgHead(lambda.tokens);
    llvm::append_range(modes, lambda.dcgModes);
    DesugarInto(*lambda.dcgBody, goals);
  } else if (lambda.body) {
    DesugarInto(*lambda.body, goals);
  }
  FillLasts(goals);
  annotation_ = savedAnnotation;
  const Lambda *result{Make<Lambda>(lambda.tokens, lambda.kind,
      lambda.any, lambda.purity, Save(llvm::ArrayRef<VarId>{head_}),
      Save(llvm::ArrayRef<std::optional<parser::Mode>>{modes}),
      lambda.determinism, Conjoin(lambda.tokens, goals))};
  for (const auto &[number, var] : outer) {
    if (var) {
      numbers_[number] = *var;
    } else {
      numbers_.erase(number);
    }
  }
  state_ = std::move(state);
  next_ = std::move(next);
  head_ = std::move(head);
  lasts_ = std::move(lasts);
  readOnly_ = std::move(readOnly);
  return result;
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
  bool modeSpecific{llvm::any_of(arguments,
      [](const parser::ModedTerm *argument) { return argument->mode; })};
  return Finish(clause.tokens, clause.kind, clause.name, goals, modeSpecific);
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

const ClauseForm *ClauseDesugarer::Desugar(const parser::MethodByName &method) {
  bool function{method.kind == parser::PredOrFunc::Function};
  for (unsigned j{0}; j < method.method->arity + (function ? 1 : 0); ++j) {
    head_.push_back(Fresh(method.tokens));
  }
  Goals goals;
  llvm::ArrayRef<VarId> head{Save(llvm::ArrayRef<VarId>{head_})};
  // The goal is the implementation's name, which its messages show.
  TokenRange name{method.implementation->tokens};
  if (function) {
    goals.push_back(Make<Unify>(name, Number(), head.back(), false,
        VarId{0},
        Functor{FunctorKind::Name, method.implementation, nullptr, nullptr},
        head.drop_back()));
  } else {
    goals.push_back(
        Make<Call>(name, Number(), method.implementation, head, false));
  }
  return Finish(method.tokens, method.kind, method.method->name, goals);
}

} // namespace

const SemanticForm &Desugar(const parser::ParseTree &tree,
    const parser::TokenSequence &tokens, SemanticArena &arena,
    parser::Messages &messages) {
  llvm::SmallVector<const ClauseForm *, 16> clauses;
  llvm::SmallVector<MethodClause, 16> methods;
  for (const parser::Item &item : tree.items) {
    if (const auto *decl{std::get_if<const parser::InstanceDecl *>(&item)}) {
      for (const parser::InstanceMethod &method : (*decl)->methods) {
        ClauseDesugarer desugarer{tokens, arena, messages};
        std::optional<const ClauseForm *> clause;
        const parser::MethodByName *byName{nullptr};
        common::visit(
            common::visitors{
                [&](const parser::MethodByName *m) {
                  byName = m;
                  clause = desugarer.Desugar(*m);
                },
                [&](const parser::FactOrRule *rule) {
                  clause = desugarer.Desugar(*rule);
                },
                [&](const parser::DcgRule *rule) {
                  clause = desugarer.Desugar(*rule);
                },
            },
            method);
        if (clause) {
          methods.push_back(MethodClause{
              *decl, &Quantify(**clause, tokens, arena, messages), byName});
        }
      }
    }
    if (const auto *rule{std::get_if<const parser::FactOrRule *>(&item)}) {
      ClauseDesugarer desugarer{tokens, arena, messages};
      if (auto clause{desugarer.Desugar(**rule)}) {
        clauses.push_back(&Quantify(**clause, tokens, arena, messages));
      }
    } else if (const auto *rule{std::get_if<const parser::DcgRule *>(&item)}) {
      ClauseDesugarer desugarer{tokens, arena, messages};
      if (auto clause{desugarer.Desugar(**rule)}) {
        clauses.push_back(&Quantify(**clause, tokens, arena, messages));
      }
    } else if (const auto *decl{std::get_if<const parser::MutableDecl *>(&item)}) {
      // A mutable declaration defines access predicates (reference manual,
      // "Module-local mutable variables"), which are still to come.
      messages.Say(parser::CharBlock{tokens.tokens[(*decl)->tokens.first]
                                         .spelling.begin(),
                       tokens.tokens[(*decl)->tokens.last].spelling.end()},
          "desugaring %s"_todo_en_US, "mutable declarations");
    }
  }
  return *arena.Make<SemanticForm>(
      arena.SaveArray(llvm::ArrayRef<const ClauseForm *>{clauses}),
      arena.SaveArray(llvm::ArrayRef<MethodClause>{methods}));
}

} // namespace merkurius::semantics
