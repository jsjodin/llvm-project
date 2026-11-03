//===-- lib/Semantics/translate-loops.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The translation of loops (design document, "Loops", Meaning, and "From a
// nest to omp.loop_nest", step 4). A loop
//
//   ( for(I, Lo, Hi), foreach(X, L), fromto(F, In, Out, Last) do Body )
//
// in a clause of p becomes a call of p__loop_Line_Column, its loop
// predicate, with the iterators' and the threaded state variables' first
// values and the shared variables that the body uses:
//
//   Trip = (the number of iterations from Lo to Hi),
//   p__loop(Lo, Trip, L, F, Last, ...)
//
//   p__loop(I, Count, L0, In, Last, ...) :-
//       ( if Count = 0 then
//           L0 = [], Last = In                          % Ends
//       else
//           L0 = [X | L1],                              % Steps
//           p__body(I, X, In, Out, ...),
//           p__loop(I + 1, Count - 1, L1, Out, Last, ...)
//       ).
//   p__body(I, X, In, Out, ...) :- Body.
//
// Without a for it is the disjunction ( Ends ; Steps, ... ), which mode
// analysis makes a switch on a list bound before the loop, or on a fromto
// whose Last has no variables, by Last's functor against In's pattern,
// which is taken from the start of the body into Steps. The for's trip
// count is computed before the loop and counted down, so that a loop to
// max_int ends; I advances beside it. A count's last value is unified with
// its Hi in Ends, and a threaded state variable's version after the loop
// with its version in the iteration that does not happen.
//
// The new clauses share the clause's variables, numbered as in it, and
// its goal numbers, with new ones after them, so that the clause's types
// are theirs: the variables that only the translation makes are appended
// to them, with their types. Quantification then gives each new clause its
// nonlocal variables. The predicates are local to the module and have the
// signatures of their heads' types, with the loop's purity, and no modes,
// so that mode analysis infers theirs from their calls.

#include "merkurius/Semantics/translate-loops.h"
#include "merkurius/Common/idioms.h"
#include "merkurius/Semantics/check-types.h"
#include "merkurius/Semantics/quantify.h"
#include "merkurius/Semantics/resolve-names.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include <functional>
#include <optional>
#include <string>

namespace merkurius::semantics {

using namespace parser::literals;
using parser::LoopIteratorKind;

namespace {

unsigned GoalNumber(const Goal &goal) {
  return std::visit([](const auto *x) { return x->number; }, goal);
}

// Calls f on each goal of `goal`, outermost first, and on those of its
// lambda expressions' bodies.
void ForEachGoal(const Goal &goal, llvm::function_ref<void(const Goal &)> f) {
  f(goal);
  common::visit(
      common::visitors{
          [&](const Conj *x) {
            for (const Goal &g : x->goals) {
              ForEachGoal(g, f);
            }
          },
          [&](const Disj *x) {
            for (const Goal &g : x->goals) {
              ForEachGoal(g, f);
            }
          },
          [&](const IfThenElse *x) {
            ForEachGoal(x->condition, f);
            ForEachGoal(x->then, f);
            ForEachGoal(x->otherwise, f);
          },
          [&](const Not *x) { ForEachGoal(x->goal, f); },
          [&](const Unify *x) {
            if (x->functor.lambda) {
              ForEachGoal(x->functor.lambda->body, f);
            }
          },
          [&](const Call *) {},
          [&](const HOCall *) {},
          [&](const Scope *x) { ForEachGoal(x->goal, f); },
          [&](const Try *x) {
            ForEachGoal(x->goal, f);
            ForEachGoal(x->then, f);
            if (x->otherwise) {
              ForEachGoal(*x->otherwise, f);
            }
            for (const Catch &c : x->catches) {
              ForEachGoal(c.pattern, f);
              ForEachGoal(c.goal, f);
            }
            if (x->catchAny) {
              ForEachGoal(x->catchAny->pattern, f);
              ForEachGoal(x->catchAny->goal, f);
            }
          },
          [&](const Atomic *x) {
            for (const AtomicAlternative &a : x->alternatives) {
              ForEachGoal(a.goal, f);
            }
          },
          [&](const Loop *x) { ForEachGoal(x->body, f); },
      },
      goal);
}

bool HasLoop(const Goal &goal) {
  bool found{false};
  ForEachGoal(goal, [&](const Goal &g) {
    found |= std::holds_alternative<const Loop *>(g);
  });
  return found;
}

// The variables that occur in a goal, in its lambda expressions too.
void Occurring(const Goal &goal, llvm::BitVector &set) {
  auto add{[&](VarId var) {
    if (var >= set.size()) {
      set.resize(var + 1);
    }
    set.set(var);
  }};
  auto addAll{[&](llvm::ArrayRef<VarId> vars) {
    for (VarId var : vars) {
      add(var);
    }
  }};
  ForEachGoal(goal, [&](const Goal &g) {
    common::visit(
        common::visitors{
            [&](const Unify *x) {
              add(x->left);
              if (x->isVar) {
                add(x->var);
              }
              addAll(x->arguments);
              if (x->functor.lambda) {
                addAll(x->functor.lambda->arguments);
              }
            },
            [&](const Call *x) { addAll(x->arguments); },
            [&](const HOCall *x) {
              add(x->closure);
              addAll(x->arguments);
            },
            [&](const Scope *x) { addAll(x->vars); },
            [&](const Try *x) {
              addAll(x->io);
              for (const Catch &c : x->catches) {
                add(c.exception);
              }
              if (x->catchAny) {
                add(x->catchAny->exception);
              }
            },
            [&](const Atomic *x) {
              add(x->outerInitial);
              add(x->outerFinal);
              addAll(x->vars);
              for (const AtomicAlternative &a : x->alternatives) {
                add(a.innerInitial);
                add(a.innerFinal);
              }
            },
            [&](const Loop *x) {
              for (const LoopIterator &iterator : x->iterators) {
                addAll(iterator.arguments);
              }
              for (const LoopState &state : x->states) {
                addAll({state.before, state.in, state.out, state.after});
              }
            },
            [&](const auto *) {},
        },
        g);
  });
}

bool Has(const llvm::BitVector &set, VarId var) {
  return var < set.size() && set.test(var);
}

// Whether a constraint is satisfied, directly or not, by one that the
// clause's declaration assumes.
bool UsesAssumed(const ConstraintProof &proof) {
  return proof.assumed ||
      llvm::any_of(proof.premises, [](const ConstraintProof &premise) {
        return UsesAssumed(premise);
      });
}

// A for's step: 1, or the literal written, which is a nonzero int.
std::int64_t StepValue(const LoopIterator &iterator) {
  if (!iterator.step) {
    return 1;
  }
  llvm::StringRef text{iterator.step->text};
  bool negative{text.consume_front("-")};
  unsigned radix{10};
  if (text.consume_front("0b")) {
    radix = 2;
  } else if (text.consume_front("0o")) {
    radix = 8;
  } else if (text.consume_front("0x")) {
    radix = 16;
  }
  std::string digits{text};
  llvm::erase(digits, '_');
  std::uint64_t magnitude{0};
  if (llvm::StringRef{digits}.getAsInteger(radix, magnitude) ||
      magnitude > static_cast<std::uint64_t>(INT64_MAX)) {
    return 0;
  }
  auto value{static_cast<std::int64_t>(magnitude)};
  return negative ? -value : value;
}

} // namespace

class LoopTranslator {
public:
  LoopTranslator(NameResolution &resolution, TypeChecking &types,
      const LoopPurities &purities, const parser::TokenSequence &tokens,
      SemanticArena &arena, parser::Messages &messages)
      : resolution_{resolution}, types_{types}, purities_{purities},
        tokens_{tokens}, arena_{arena}, messages_{messages} {}

  LoopTranslation Run();

private:
  // A predicate made for a loop: its symbol and its clause, before
  // quantification.
  struct Made {
    const Symbol *symbol;
    const ClauseForm *clause;
  };

  // Whether the loops of the clause can be translated; if not, why not is
  // reported, or the loops are left for mode checking to report.
  bool Translatable(const ClauseTypes &);
  void TranslateClause(std::size_t index);
  // The goal with its loops translated: the same goal if it has none.
  Goal Rebuild(const Goal &);
  // The goals that replace a loop: the computation of its trip count, if
  // it has a for, and the call of its loop predicate.
  Goal TranslateLoop(const Loop &);
  // The number of iterations of a for, into a new variable; returns the
  // test it starts with.
  const Call *TripCount(const LoopIterator &, VarId trip, std::vector<Goal> &);

  // The new predicate's symbol, p__kind_Line_Column.
  const Symbol &NewPredicate(llvm::StringRef kind, const Loop &, unsigned arity);
  const parser::SymName *NameOf(const Symbol &);
  const parser::Literal *IntLiteral(TokenRange, llvm::StringRef text);
  // A function or predicate of int.
  const Symbol *IntSymbol(SymbolKind, llvm::StringRef name, unsigned arity);
  const Symbol *ListConstructor(llvm::StringRef name, unsigned arity);

  VarId NewVar(const Type *, TokenRange);
  unsigned NewNumber(const Symbol * = nullptr, const parser::SymName * = nullptr);
  Goal Conjunction(TokenRange, llvm::ArrayRef<Goal>);
  Goal UnifyVars(TokenRange, VarId left, VarId right);
  Goal UnifyLiteral(TokenRange, VarId left, const parser::Literal *);
  // X = f(Y1, ..., Yn) for a constructor or a function f.
  Goal UnifyFunctor(TokenRange, VarId left, FunctorKind, const Symbol &,
      llvm::ArrayRef<VarId> arguments);
  Goal CallOf(TokenRange, const Symbol &, llvm::ArrayRef<VarId> arguments);
  // A new int variable that holds `text`, by a goal added to `goals`.
  VarId Constant(TokenRange, llvm::StringRef text, std::vector<Goal> &goals);
  // Result = f(X, Y) for a function f of int, into a new variable.
  VarId Apply(TokenRange, llvm::StringRef f, VarId x, VarId y,
      std::vector<Goal> &goals);

  parser::CharBlock Source(TokenRange range) const {
    return {tokens_.tokens[range.first].spelling.begin(),
        tokens_.tokens[range.last].spelling.end()};
  }
  template <typename T, typename... A> const T *Make(A &&...args) {
    return arena_.Make<T>(std::forward<A>(args)...);
  }
  template <typename T> llvm::ArrayRef<T> Save(llvm::ArrayRef<T> array) {
    return arena_.SaveArray(array);
  }

  NameResolution &resolution_;
  TypeChecking &types_;
  const LoopPurities &purities_;
  const parser::TokenSequence &tokens_;
  SemanticArena &arena_;
  parser::Messages &messages_;
  llvm::DenseMap<const Symbol *, const parser::SymName *> names_;
  const Type *int_{nullptr};

  // The clause being translated, and its tables, which its loops'
  // predicates' clauses share, extended with what the translation makes.
  const ClauseTypes *clause_{nullptr};
  std::vector<VarInfo> vars_;
  std::vector<const Type *> varTypes_;
  std::vector<const Symbol *> symbols_;
  std::vector<const parser::SymName *> callNames_;
  unsigned goals_{0};
  // Each loop's predicates, in the order of the result, a loop's before
  // those of the loops in its body.
  std::vector<std::optional<Made>> made_;
  // The purities of the predicates made, by symbol.
  llvm::DenseMap<const Symbol *, parser::Purity> purity_;
  LoopTranslation result_;
};

const Symbol *LoopTranslator::IntSymbol(
    SymbolKind kind, llvm::StringRef name, unsigned arity) {
  const ModuleSymbols *module{resolution_.Module("int")};
  if (!module) {
    return nullptr;
  }
  for (const Symbol *symbol : module->Lookup(name)) {
    if (symbol->kind == kind && symbol->arity == arity) {
      return symbol;
    }
  }
  return nullptr;
}

const Symbol *LoopTranslator::ListConstructor(
    llvm::StringRef name, unsigned arity) {
  const ModuleSymbols *module{resolution_.Module("list")};
  if (!module) {
    return nullptr;
  }
  for (const Symbol *symbol : module->Lookup(name)) {
    if (symbol->kind == SymbolKind::Constructor && symbol->arity == arity) {
      return symbol;
    }
  }
  return nullptr;
}

// The names of the clause's loops' predicates, and the int functions and
// predicates their goals use, with the symbol as their only candidate.
const parser::SymName *LoopTranslator::NameOf(const Symbol &symbol) {
  const parser::SymName *&name{names_[&symbol]};
  if (!name) {
    llvm::ArrayRef<llvm::StringRef> qualifiers;
    if (symbol.module != resolution_.module().name()) {
      qualifiers = arena_.SaveArray(llvm::ArrayRef<llvm::StringRef>{symbol.module});
    }
    const Symbol *candidate{&symbol};
    name = Make<parser::SymName>(parser::SymName{clause_->clause->tokens,
        qualifiers, symbol.name,
        arena_.SaveArray(llvm::ArrayRef<const Symbol *>{candidate})});
  }
  return name;
}

const parser::Literal *LoopTranslator::IntLiteral(
    TokenRange range, llvm::StringRef text) {
  return Make<parser::Literal>(parser::Literal{
      range, parser::LiteralKind::Integer, arena_.Save(text)});
}

VarId LoopTranslator::NewVar(const Type *type, TokenRange range) {
  vars_.push_back(VarInfo{VarKind::Fresh, {}, range});
  varTypes_.push_back(type);
  return static_cast<VarId>(vars_.size() - 1);
}

unsigned LoopTranslator::NewNumber(
    const Symbol *symbol, const parser::SymName *name) {
  symbols_.push_back(symbol);
  callNames_.push_back(name);
  return goals_++;
}

Goal LoopTranslator::Conjunction(TokenRange range, llvm::ArrayRef<Goal> goals) {
  if (goals.size() == 1) {
    return goals.front();
  }
  return Make<Conj>(range, NewNumber(), false, Save(goals));
}

Goal LoopTranslator::UnifyVars(TokenRange range, VarId left, VarId right) {
  return Make<Unify>(range, NewNumber(), left, true, right, Functor{},
      llvm::ArrayRef<VarId>{});
}

Goal LoopTranslator::UnifyLiteral(
    TokenRange range, VarId left, const parser::Literal *literal) {
  return Make<Unify>(range, NewNumber(), left, false, VarId{0},
      Functor{FunctorKind::Literal, nullptr, literal, nullptr},
      llvm::ArrayRef<VarId>{});
}

Goal LoopTranslator::UnifyFunctor(TokenRange range, VarId left,
    FunctorKind kind, const Symbol &symbol, llvm::ArrayRef<VarId> arguments) {
  const parser::SymName *name{NameOf(symbol)};
  return Make<Unify>(range, NewNumber(&symbol, name), left, false, VarId{0},
      Functor{kind, name, nullptr, nullptr}, Save(arguments));
}

Goal LoopTranslator::CallOf(
    TokenRange range, const Symbol &symbol, llvm::ArrayRef<VarId> arguments) {
  const parser::SymName *name{NameOf(symbol)};
  return Make<Call>(range, NewNumber(&symbol, name), name, Save(arguments),
      false);
}

VarId LoopTranslator::Constant(
    TokenRange range, llvm::StringRef text, std::vector<Goal> &goals) {
  VarId var{NewVar(int_, range)};
  goals.push_back(UnifyLiteral(range, var, IntLiteral(range, text)));
  return var;
}

VarId LoopTranslator::Apply(TokenRange range, llvm::StringRef f, VarId x,
    VarId y, std::vector<Goal> &goals) {
  VarId result{NewVar(int_, range)};
  goals.push_back(UnifyFunctor(range, result, FunctorKind::Name,
      *IntSymbol(SymbolKind::Function, f, 2), {x, y}));
  return result;
}

const Symbol &LoopTranslator::NewPredicate(
    llvm::StringRef kind, const Loop &loop, unsigned arity) {
  // The loop's line and column, counted from the start of the file.
  const parser::Token &first{tokens_.tokens.front()};
  const char *start{first.trivia.empty() ? first.spelling.begin()
                                         : first.trivia.begin()};
  const char *at{tokens_.tokens[loop.tokens.first].spelling.begin()};
  unsigned line{1}, column{1};
  for (const char *p{start}; p < at; ++p) {
    if (*p == '\n') {
      ++line;
      column = 1;
    } else {
      ++column;
    }
  }
  std::string name{(clause_->clause->name->name + "__" + kind + "_" +
      llvm::Twine{line} + "_" + llvm::Twine{column})
          .str()};
  return resolution_.module().Add(SymbolKind::Predicate, name, arity,
      std::monostate{}, Source(loop.tokens), false);
}

bool LoopTranslator::Translatable(const ClauseTypes &clause) {
  if (clause.typeError || !clause.definition) {
    return false;
  }
  bool translatable{true};
  ForEachGoal(clause.clause->body, [&](const Goal &goal) {
    const auto *loop{std::get_if<const Loop *>(&goal)};
    if (!loop) {
      return;
    }
    // A loop that purity checking did not check is in a predicate with a
    // type error.
    if (!purities_.count(*loop)) {
      translatable = false;
    }
    for (const LoopIterator &iterator : (*loop)->iterators) {
      bool counter{iterator.kind == LoopIteratorKind::For ||
          iterator.kind == LoopIteratorKind::Count};
      // Without int or list, which were reported missing, the loop is left
      // for mode checking to report.
      if ((counter && !IntSymbol(SymbolKind::Function, "+", 2)) ||
          (iterator.kind == LoopIteratorKind::Foreach &&
              !ListConstructor("[|]", 2))) {
        translatable = false;
      }
    }
  });
  if (!translatable) {
    return false;
  }
  // The type class constraints of the goals in a loop's body must not need
  // the clause's own, which the loop's predicates would need to be given.
  llvm::DenseSet<unsigned> inLoops;
  ForEachGoal(clause.clause->body, [&](const Goal &goal) {
    if (const auto *loop{std::get_if<const Loop *>(&goal)}) {
      ForEachGoal((*loop)->body,
          [&](const Goal &g) { inLoops.insert(GoalNumber(g)); });
    }
  });
  for (const GoalConstraint &constraint : clause.constraints) {
    if (inLoops.count(constraint.goal) && UsesAssumed(constraint.proof)) {
      ForEachGoal(clause.clause->body, [&](const Goal &goal) {
        if (const auto *loop{std::get_if<const Loop *>(&goal)}) {
          messages_.Say(Source((*loop)->tokens),
              "%s"_todo_en_US,
              "loops whose body needs a type class constraint of its "
              "predicate");
        }
      });
      return false;
    }
  }
  return true;
}

LoopTranslation LoopTranslator::Run() {
  for (const Symbol *symbol : resolution_.primitive().Lookup("int")) {
    if (symbol->kind == SymbolKind::Type) {
      int_ = types_.types().Named(*symbol, {});
    }
  }
  std::vector<std::size_t> &result{result_.clauses};
  std::size_t clauses{types_.clauses_.size()};
  for (std::size_t i{0}; i < clauses; ++i) {
    const ClauseTypes &clause{types_.clauses_[i]};
    if (!HasLoop(clause.clause->body) || !Translatable(clause)) {
      continue;
    }
    std::size_t first{types_.clauses_.size()};
    TranslateClause(i);
    result.push_back(i);
    for (std::size_t j{first}; j < types_.clauses_.size(); ++j) {
      result.push_back(j);
    }
  }
  for (const TranslatedLoop &loop : result_.loops) {
    result_.byPredicate[loop.loopPredicate] = &loop;
  }
  return std::move(result_);
}

void LoopTranslator::TranslateClause(std::size_t index) {
  clause_ = &types_.clauses_[index];
  const ClauseForm &form{*clause_->clause};
  vars_.assign(form.vars.begin(), form.vars.end());
  varTypes_.assign(clause_->vars.begin(), clause_->vars.end());
  symbols_.assign(clause_->symbols.begin(), clause_->symbols.end());
  callNames_.assign(clause_->names.begin(), clause_->names.end());
  goals_ = form.goals;
  made_.clear();
  Goal body{Rebuild(form.body)};
  ClauseTypes outer{*clause_};

  // The tables of the clause and its loops' predicates' clauses.
  llvm::ArrayRef<VarInfo> vars{Save(llvm::ArrayRef<VarInfo>{vars_})};
  llvm::ArrayRef<const Type *> varTypes{
      types_.arena_.SaveArray(llvm::ArrayRef<const Type *>{varTypes_})};
  llvm::ArrayRef<const Symbol *> symbols{
      types_.arena_.SaveArray(llvm::ArrayRef<const Symbol *>{symbols_})};
  llvm::ArrayRef<const parser::SymName *> names{
      types_.arena_.SaveArray(llvm::ArrayRef<const parser::SymName *>{callNames_})};
  // Quantification makes no variable, since each new one occurs in one
  // goal of those parallel to each other; its messages were given for the
  // clause.
  auto quantify{[&](const ClauseForm &clause) {
    parser::Messages ignored;
    const ClauseForm &quantified{Quantify(clause, tokens_, arena_, ignored)};
    CHECK(quantified.vars.size() == vars.size() &&
        "loop translation made variables that quantification renamed");
    return &quantified;
  }};
  // Each constraint goes with the clause that has its goal; the clause's
  // existential constraints, numbered by its number of goals, stay with it.
  auto constraintsOf{[&](const ClauseForm &clause, bool isOuter) {
    llvm::DenseSet<unsigned> numbers;
    ForEachGoal(clause.body, [&](const Goal &g) { numbers.insert(GoalNumber(g)); });
    std::vector<GoalConstraint> constraints;
    for (GoalConstraint constraint : outer.constraints) {
      if (constraint.goal == form.goals) {
        if (isOuter) {
          constraint.goal = goals_;
          constraints.push_back(constraint);
        }
      } else if (numbers.count(constraint.goal)) {
        constraints.push_back(constraint);
      }
    }
    return types_.arena_.SaveArray(llvm::ArrayRef<GoalConstraint>{constraints});
  }};

  const ClauseForm *translated{quantify(*Make<ClauseForm>(form.tokens,
      form.kind, form.name, form.head, body, vars, form.typeQualifications,
      goals_, llvm::ArrayRef<llvm::ArrayRef<VarId>>{}, form.modeSpecific))};
  ClauseTypes &own{types_.clauses_[index]};
  own.clause = translated;
  own.vars = varTypes;
  own.symbols = symbols;
  own.names = names;
  own.constraints = constraintsOf(*translated, true);

  for (const std::optional<Made> &made : made_) {
    const ClauseForm &clause{*quantify(*Make<ClauseForm>(made->clause->tokens,
        made->clause->kind, made->clause->name, made->clause->head,
        made->clause->body, vars, llvm::ArrayRef<TypeQualification>{},
        goals_, llvm::ArrayRef<llvm::ArrayRef<VarId>>{}, false))};
    // The signature of the head's types, with the clause's type
    // variables.
    std::vector<const Type *> arguments;
    for (VarId var : clause.head) {
      arguments.push_back(varTypes[var]);
    }
    unsigned typeVars(outer.typeVarNames.size());
    std::function<void(const Type &)> count{[&](const Type &type) {
      if (type.kind == TypeKind::Variable) {
        typeVars = std::max(typeVars, type.var + 1);
      }
      for (const Type *argument : type.arguments) {
        count(*argument);
      }
      if (type.result) {
        count(*type.result);
      }
    }};
    for (const Type *type : arguments) {
      count(*type);
    }
    types_.signatures_[made->symbol] = Signature{parser::PredOrFunc::Predicate,
        purity_.lookup(made->symbol), typeVars, outer.typeVarNames,
        types_.arena_.SaveArray(llvm::ArrayRef<const Type *>{arguments})};
    ClauseTypes types{&clause, made->symbol, true, false, outer.typeVarNames,
        varTypes, symbols, names, constraintsOf(clause, false)};
    types_.clauses_.push_back(types);
    clause_ = &types_.clauses_[index];
  }
}

Goal LoopTranslator::Rebuild(const Goal &goal) {
  auto rebuilt{[&](llvm::ArrayRef<Goal> goals, std::vector<Goal> &out) {
    bool changed{false};
    for (const Goal &g : goals) {
      out.push_back(Rebuild(g));
      changed |= out.back() != g;
    }
    return changed;
  }};
  return common::visit(
      common::visitors{
          [&](const Conj *x) -> Goal {
            std::vector<Goal> goals;
            if (!rebuilt(x->goals, goals)) {
              return x;
            }
            return Make<Conj>(x->tokens, x->number, x->parallel,
                Save(llvm::ArrayRef<Goal>{goals}));
          },
          [&](const Disj *x) -> Goal {
            std::vector<Goal> goals;
            if (!rebuilt(x->goals, goals)) {
              return x;
            }
            return Make<Disj>(
                x->tokens, x->number, Save(llvm::ArrayRef<Goal>{goals}));
          },
          [&](const IfThenElse *x) -> Goal {
            Goal condition{Rebuild(x->condition)}, then{Rebuild(x->then)},
                otherwise{Rebuild(x->otherwise)};
            if (condition == x->condition && then == x->then &&
                otherwise == x->otherwise) {
              return x;
            }
            return Make<IfThenElse>(
                x->tokens, x->number, condition, then, otherwise);
          },
          [&](const Not *x) -> Goal {
            Goal inner{Rebuild(x->goal)};
            if (inner == x->goal) {
              return x;
            }
            return Make<Not>(x->tokens, x->number, inner);
          },
          [&](const Unify *x) -> Goal {
            const Lambda *lambda{x->functor.lambda};
            if (!lambda) {
              return x;
            }
            Goal body{Rebuild(lambda->body)};
            if (body == lambda->body) {
              return x;
            }
            Functor functor{x->functor};
            functor.lambda = Make<Lambda>(lambda->tokens, lambda->kind,
                lambda->any, lambda->purity, lambda->arguments, lambda->modes,
                lambda->determinism, body);
            return Make<Unify>(x->tokens, x->number, x->left, x->isVar, x->var,
                functor, x->arguments);
          },
          [&](const Call *x) -> Goal { return x; },
          [&](const HOCall *x) -> Goal { return x; },
          [&](const Scope *x) -> Goal {
            Goal inner{Rebuild(x->goal)};
            if (inner == x->goal) {
              return x;
            }
            return Make<Scope>(
                x->tokens, x->number, x->kind, x->origin, x->vars, inner);
          },
          [&](const Try *x) -> Goal {
            Goal tried{Rebuild(x->goal)}, then{Rebuild(x->then)};
            bool changed{tried != x->goal || then != x->then};
            std::optional<Goal> otherwise;
            if (x->otherwise) {
              otherwise = Rebuild(*x->otherwise);
              changed |= *otherwise != *x->otherwise;
            }
            auto branch{[&](const Catch &c) {
              Catch result{c.tokens, c.exception, Rebuild(c.pattern),
                  Rebuild(c.goal)};
              changed |= result.pattern != c.pattern || result.goal != c.goal;
              return result;
            }};
            std::vector<Catch> catches;
            for (const Catch &c : x->catches) {
              catches.push_back(branch(c));
            }
            std::optional<Catch> catchAny;
            if (x->catchAny) {
              catchAny = branch(*x->catchAny);
            }
            if (!changed) {
              return x;
            }
            return Make<Try>(x->tokens, x->number, x->io, tried, then,
                otherwise, Save(llvm::ArrayRef<Catch>{catches}), catchAny);
          },
          [&](const Atomic *x) -> Goal {
            bool changed{false};
            std::vector<AtomicAlternative> alternatives;
            for (const AtomicAlternative &a : x->alternatives) {
              alternatives.push_back(
                  AtomicAlternative{a.innerInitial, a.innerFinal, Rebuild(a.goal)});
              changed |= alternatives.back().goal != a.goal;
            }
            if (!changed) {
              return x;
            }
            return Make<Atomic>(x->tokens, x->number, x->outerInitial,
                x->outerFinal, x->vars,
                Save(llvm::ArrayRef<AtomicAlternative>{alternatives}));
          },
          [&](const Loop *x) -> Goal { return TranslateLoop(*x); },
      },
      goal);
}

// The number of iterations of for(I, Lo, Hi, S), which is 0 if Hi is
// below Lo, or above it for a negative S, and is otherwise, with M = |S|
// and Small and Big the lesser and the greater bound,
// (Big - Small) div M + 1. That is computed so that it cannot overflow:
// for M = 1, Big - Small + 1 wraps only for the whole range of int, whose
// count, 2^64, is the one it cannot represent; otherwise as
// Big div M - Small div M, plus one unless Big mod M < Small mod M.
const Call *LoopTranslator::TripCount(
    const LoopIterator &iterator, VarId trip, std::vector<Goal> &goals) {
  TokenRange range{iterator.tokens};
  std::int64_t step{StepValue(iterator)};
  VarId lo{iterator.arguments[1]}, hi{iterator.arguments[2]};
  VarId small{step > 0 ? lo : hi}, big{step > 0 ? hi : lo};
  std::vector<Goal> none;
  none.push_back(UnifyLiteral(range, trip, IntLiteral(range, "0")));
  std::vector<Goal> some;
  std::uint64_t magnitude{step > 0 ? static_cast<std::uint64_t>(step)
                                   : 0 - static_cast<std::uint64_t>(step)};
  if (magnitude == 1) {
    VarId span{Apply(range, "-", big, small, some)};
    VarId one{Constant(range, "1", some)};
    some.push_back(UnifyFunctor(range, trip, FunctorKind::Name,
        *IntSymbol(SymbolKind::Function, "+", 2), {span, one}));
  } else {
    VarId m{Constant(range, std::to_string(magnitude), some)};
    VarId q1{Apply(range, "div", big, m, some)};
    VarId q2{Apply(range, "div", small, m, some)};
    VarId r1{Apply(range, "mod", big, m, some)};
    VarId r2{Apply(range, "mod", small, m, some)};
    VarId span{Apply(range, "-", q1, q2, some)};
    std::vector<Goal> plus;
    VarId one{Constant(range, "1", plus)};
    plus.push_back(UnifyFunctor(range, trip, FunctorKind::Name,
        *IntSymbol(SymbolKind::Function, "+", 2), {span, one}));
    some.push_back(Make<IfThenElse>(range, NewNumber(),
        CallOf(range, *IntSymbol(SymbolKind::Predicate, "<", 2), {r1, r2}),
        UnifyVars(range, trip, span), Conjunction(range, plus)));
  }
  Goal test{
      CallOf(range, *IntSymbol(SymbolKind::Predicate, "<", 2), {big, small})};
  goals.push_back(Make<IfThenElse>(range, NewNumber(), test,
      Conjunction(range, none), Conjunction(range, some)));
  return std::get<const Call *>(test);
}

Goal LoopTranslator::TranslateLoop(const Loop &loop) {
  TokenRange range{loop.tokens};
  // The slot of this loop's predicates, before those of the loops in its
  // body.
  std::size_t slot{made_.size()};
  made_.emplace_back();
  made_.emplace_back();

  // In's pattern, for a fromto whose Last has no variables, is taken from
  // the start of the body, so that Ends and Steps are a switch on it.
  llvm::DenseSet<VarId> ending;
  for (const LoopIterator &iterator : loop.iterators) {
    if (iterator.kind == LoopIteratorKind::Fromto && iterator.ends) {
      ending.insert(iterator.arguments[1]);
    }
  }
  llvm::DenseMap<VarId, const Unify *> patterns;
  Goal rest{loop.body};
  if (!ending.empty()) {
    auto pattern{[&](const Goal &g) -> const Unify * {
      const auto *unify{std::get_if<const Unify *>(&g)};
      return unify && !(*unify)->isVar && ending.count((*unify)->left) &&
              !patterns.count((*unify)->left)
          ? *unify
          : nullptr;
    }};
    if (const Unify *unify{pattern(loop.body)}) {
      patterns[unify->left] = unify;
      rest = Make<Conj>(unify->tokens, NewNumber(), false, llvm::ArrayRef<Goal>{});
    } else if (const auto *conj{std::get_if<const Conj *>(&loop.body)};
               conj && !(*conj)->parallel) {
      std::size_t taken{0};
      while (taken < (*conj)->goals.size()) {
        const Unify *unify{pattern((*conj)->goals[taken])};
        if (!unify) {
          break;
        }
        patterns[unify->left] = unify;
        ++taken;
      }
      if (taken > 0) {
        rest = Make<Conj>((*conj)->tokens, (*conj)->number, false,
            (*conj)->goals.drop_front(taken));
      }
    }
  }
  Goal body{Rebuild(rest)};

  // The body predicate's arguments: the iterators' own variables and
  // those of the patterns taken from the body, the threaded state
  // variables' versions in an iteration, and the shared variables, those
  // of the goal around the loop, that the body uses.
  llvm::BitVector occurring;
  Occurring(body, occurring);
  llvm::BitVector nonlocal;
  for (VarId var : clause_->clause->nonlocals[GoalNumber(loop.body)]) {
    if (var >= nonlocal.size()) {
      nonlocal.resize(var + 1);
    }
    nonlocal.set(var);
  }
  llvm::DenseSet<VarId> own;
  std::vector<VarId> candidates;
  auto ownVar{[&](VarId var) {
    own.insert(var);
    candidates.push_back(var);
  }};
  for (const LoopIterator &iterator : loop.iterators) {
    switch (iterator.kind) {
    case LoopIteratorKind::For:
    case LoopIteratorKind::Count:
    case LoopIteratorKind::Foreach:
      ownVar(iterator.arguments[0]);
      break;
    case LoopIteratorKind::Fromto:
      ownVar(iterator.arguments[1]);
      ownVar(iterator.arguments[2]);
      if (const Unify *unify{patterns.lookup(iterator.arguments[1])}) {
        for (VarId var : unify->arguments) {
          ownVar(var);
        }
      }
      break;
    }
  }
  for (const LoopState &state : loop.states) {
    ownVar(state.in);
    ownVar(state.out);
  }
  for (VarId var : clause_->clause->nonlocals[GoalNumber(loop.body)]) {
    candidates.push_back(var);
  }
  std::vector<VarId> bodyHead;
  llvm::DenseSet<VarId> inBodyHead;
  for (VarId var : candidates) {
    if (Has(occurring, var) && (own.count(var) || Has(nonlocal, var)) &&
        inBodyHead.insert(var).second) {
      bodyHead.push_back(var);
    }
  }
  const Symbol &bodySymbol{NewPredicate("body", loop, bodyHead.size())};

  // The loop predicate's arguments, with the values the loop starts with
  // and those the next iteration gets; a variable passed through is the
  // same in all three.
  std::vector<VarId> head, initial, next;
  llvm::DenseSet<VarId> inHead;
  auto through{[&](VarId var) {
    if (inHead.insert(var).second) {
      head.push_back(var);
      initial.push_back(var);
      next.push_back(var);
    }
  }};
  // Two iterators that share an own variable give the head two variables
  // that are unified.
  std::vector<Goal> unifications;
  auto argument{[&](VarId var, VarId first, VarId then) {
    if (!inHead.insert(var).second) {
      VarId copy{NewVar(varTypes_[var], range)};
      unifications.push_back(UnifyVars(range, copy, var));
      var = copy;
    }
    head.push_back(var);
    initial.push_back(first);
    next.push_back(then);
  }};
  std::vector<Goal> before, ends, steps, advance;
  std::optional<Goal> condition;
  TranslatedLoop translated{&loop, nullptr, nullptr};
  auto position{[&](VarId var) {
    return static_cast<unsigned>(llvm::find(head, var) - head.begin());
  }};
  for (const LoopIterator &iterator : loop.iterators) {
    TokenRange at{iterator.tokens};
    llvm::ArrayRef<VarId> a{iterator.arguments};
    switch (iterator.kind) {
    case LoopIteratorKind::For: {
      VarId trip{NewVar(int_, at)}, count{NewVar(int_, at)};
      translated.tripTest = TripCount(iterator, trip, before)->number;
      translated.step = StepValue(iterator);
      VarId step{NewVar(int_, at)};
      advance.push_back(UnifyLiteral(at, step,
          iterator.step ? iterator.step : IntLiteral(at, "1")));
      VarId nextI{Apply(at, "+", a[0], step, advance)};
      VarId one{Constant(at, "1", advance)};
      VarId nextCount{Apply(at, "-", count, one, advance)};
      argument(a[0], a[1], nextI);
      argument(count, trip, nextCount);
      condition = UnifyLiteral(at, count, IntLiteral(at, "0"));
      break;
    }
    case LoopIteratorKind::Count: {
      VarId one{Constant(at, "1", advance)};
      VarId nextI{Apply(at, "+", a[0], one, advance)};
      argument(a[0], a[1], nextI);
      through(a[2]);
      VarId last{Constant(at, "1", ends)};
      ends.push_back(UnifyFunctor(at, a[2], FunctorKind::Name,
          *IntSymbol(SymbolKind::Function, "-", 2), {a[0], last}));
      break;
    }
    case LoopIteratorKind::Foreach: {
      const Type *type{varTypes_[a[1]]};
      VarId list{NewVar(type, at)}, tail{NewVar(type, at)};
      argument(list, a[1], tail);
      ends.push_back(UnifyFunctor(
          at, list, FunctorKind::Nil, *ListConstructor("[]", 0), {}));
      steps.push_back(UnifyFunctor(at, list, FunctorKind::Cons,
          *ListConstructor("[|]", 2), {a[0], tail}));
      break;
    }
    case LoopIteratorKind::Fromto: {
      auto in{static_cast<unsigned>(head.size())};
      argument(a[1], a[0], a[2]);
      const Unify *last{nullptr};
      if (iterator.ends) {
        // Last's functor, from the goal before the loop that binds it.
        ForEachGoal(clause_->clause->body, [&](const Goal &g) {
          const auto *unify{std::get_if<const Unify *>(&g)};
          if (unify && (*unify)->left == a[3] && !(*unify)->isVar &&
              !(*unify)->functor.lambda) {
            last = *unify;
          }
        });
      }
      if (last) {
        for (VarId var : last->arguments) {
          through(var);
        }
        ends.push_back(Make<Unify>(last->tokens,
            NewNumber(symbols_[last->number], callNames_[last->number]),
            a[1], false, VarId{0}, last->functor, last->arguments));
      } else {
        through(a[3]);
        translated.threaded.push_back({in, position(a[3]), &iterator});
        ends.push_back(UnifyVars(at, a[3], a[1]));
      }
      if (const Unify *pattern{patterns.lookup(a[1])}) {
        steps.push_back(pattern);
      }
      break;
    }
    }
  }
  for (const LoopState &state : loop.states) {
    auto in{static_cast<unsigned>(head.size())};
    argument(state.in, state.before, state.out);
    through(state.after);
    translated.threaded.push_back({in, position(state.after), nullptr});
    ends.push_back(UnifyVars(range, state.after, state.in));
  }
  for (VarId var : bodyHead) {
    if (!own.count(var)) {
      through(var);
    }
  }
  const Symbol &loopSymbol{NewPredicate("loop", loop, head.size())};
  translated.loopPredicate = &loopSymbol;
  translated.bodyPredicate = &bodySymbol;
  result_.loops.push_back(std::move(translated));
  parser::Purity purity{purities_.lookup(&loop)};
  purity_[&loopSymbol] = purity;
  purity_[&bodySymbol] = purity;

  steps.push_back(CallOf(range, bodySymbol, bodyHead));
  steps.insert(steps.end(), advance.begin(), advance.end());
  steps.push_back(CallOf(range, loopSymbol, next));
  Goal iterate{Conjunction(range, steps)};
  Goal end{Conjunction(range, ends)};
  Goal choice{condition
          ? Goal{Make<IfThenElse>(range, NewNumber(), *condition, end, iterate)}
          : Goal{Make<Disj>(range, NewNumber(),
                Save(llvm::ArrayRef<Goal>{end, iterate}))}};
  unifications.push_back(choice);

  made_[slot] = Made{&loopSymbol,
      Make<ClauseForm>(range, parser::PredOrFunc::Predicate, NameOf(loopSymbol),
          Save(llvm::ArrayRef<VarId>{head}), Conjunction(range, unifications),
          llvm::ArrayRef<VarInfo>{}, llvm::ArrayRef<TypeQualification>{}, 0u)};
  made_[slot + 1] = Made{&bodySymbol,
      Make<ClauseForm>(range, parser::PredOrFunc::Predicate, NameOf(bodySymbol),
          Save(llvm::ArrayRef<VarId>{bodyHead}), body,
          llvm::ArrayRef<VarInfo>{}, llvm::ArrayRef<TypeQualification>{}, 0u)};

  before.push_back(CallOf(range, loopSymbol, initial));
  return Conjunction(range, before);
}

llvm::SmallVector<llvm::StringRef, 1> LoopModules(const SemanticForm &form) {
  bool counters{false};
  auto look{[&](const ClauseForm &clause) {
    ForEachGoal(clause.body, [&](const Goal &goal) {
      if (const auto *loop{std::get_if<const Loop *>(&goal)}) {
        for (const LoopIterator &iterator : (*loop)->iterators) {
          counters |= iterator.kind == LoopIteratorKind::For ||
              iterator.kind == LoopIteratorKind::Count;
        }
      }
    });
  }};
  for (const ClauseForm *clause : form.clauses) {
    look(*clause);
  }
  for (const MethodClause &method : form.methods) {
    look(*method.clause);
  }
  llvm::SmallVector<llvm::StringRef, 1> modules;
  if (counters) {
    modules.push_back("int");
  }
  return modules;
}

LoopTranslation TranslateLoops(NameResolution &resolution,
    TypeChecking &types, const LoopPurities &purities,
    const parser::TokenSequence &tokens, SemanticArena &arena,
    parser::Messages &messages) {
  return LoopTranslator{resolution, types, purities, tokens, arena, messages}
      .Run();
}

} // namespace merkurius::semantics
