//===-- lib/Semantics/check-purity.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "merkurius/Semantics/check-purity.h"
#include "merkurius/Common/idioms.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <optional>
#include <string>

namespace merkurius::semantics {

using namespace parser::literals;
using parser::Purity;

namespace {

const char *Name(Purity purity) {
  switch (purity) {
  case Purity::Pure:
    return "pure";
  case Purity::Semipure:
    return "semipure";
  case Purity::Impure:
    return "impure";
  }
  return "";
}

// The less pure of two purities: pure > semipure > impure (reference
// manual, "Purity ordering"), in the enumeration's reverse order.
Purity Least(Purity x, Purity y) { return std::max(x, y); }

std::string Text(const Symbol &symbol) {
  std::string text;
  llvm::raw_string_ostream os{text};
  os << symbol;
  return text;
}

class PurityChecker {
public:
  PurityChecker(const parser::TokenSequence &tokens,
      const NameResolution &resolution, const TypeChecking &types,
      parser::Messages &messages, LoopPurities *loops)
      : tokens_{tokens}, resolution_{resolution}, types_{types},
        messages_{messages}, loops_{loops} {}

  void Run(const parser::ParseTree &, const SemanticForm &);

private:
  // The purity of a goal, the least of its parts': a call's and a
  // function application's are their callees' declared ones; a promise's,
  // the one it promises; a trace goal's, pure.
  Purity Check(const Goal &);
  Purity CheckUnify(const Unify &);
  Purity CheckCall(const Call &);
  Purity CheckScope(const Scope &);
  // A goal annotated `impure` or `semipure`.
  Purity CheckAnnotated(const Scope &, Purity annotation);
  // The purity that a goal's callee is declared with.
  Purity Declared(const Symbol *) const;
  // The function that a unification applies, if any.
  const Symbol *Function(const Unify &) const;
  bool IsNamed(VarId var) const {
    VarKind kind{clause_->clause->vars[var].kind};
    return kind == VarKind::Program || kind == VarKind::State;
  }
  void CheckPredicates(const parser::ParseTree &);

  parser::CharBlock Source(TokenRange range) const {
    return {tokens_.tokens[range.first].spelling.begin(),
        tokens_.tokens[range.last].spelling.end()};
  }

  const parser::TokenSequence &tokens_;
  const NameResolution &resolution_;
  const TypeChecking &types_;
  parser::Messages &messages_;
  LoopPurities *loops_;
  const ClauseTypes *clause_{nullptr};
  // Inside a compound goal annotated in error, the annotation, which the
  // goals in it are taken to have.
  std::optional<Purity> assumed_;
  // The purity of each predicate's or function's clauses, and its first;
  // and those of each instance's method definitions.
  struct Clauses {
    Purity purity{Purity::Pure};
    const ClauseForm *first{nullptr};
    bool modeSpecific{false}; // a clause is for one mode
  };
  llvm::MapVector<const Symbol *, Clauses> clauses_;
  llvm::MapVector<const MethodDefinition *, Clauses> methods_;
};

void PurityChecker::Run(
    const parser::ParseTree &tree, const SemanticForm &form) {
  // The clauses that methods defined by name stand for: their calls have
  // the method's declared purity, as in mmc.
  llvm::DenseSet<const ClauseForm *> byName;
  for (const MethodClause &method : form.methods) {
    if (method.byName) {
      byName.insert(method.clause);
    }
  }
  // A predicate or function with a type error in a clause is not checked,
  // as mmc does not check purity after type errors.
  llvm::DenseSet<const void *> mistyped;
  for (const ClauseTypes &clause : types_.clauses()) {
    if (clause.typeError) {
      mistyped.insert(clause.method ? static_cast<const void *>(clause.method)
                                    : clause.definition);
    }
  }
  for (const ClauseTypes &clause : types_.clauses()) {
    if (!clause.definition ||
        mistyped.count(clause.method ? static_cast<const void *>(clause.method)
                                     : clause.definition)) {
      continue;
    }
    clause_ = &clause;
    std::optional<Purity> saved{assumed_};
    if (clause.method && byName.count(clause.clause)) {
      assumed_ = Declared(clause.method->method);
    }
    Purity purity{Check(clause.clause->body)};
    assumed_ = saved;
    Clauses &of{clause.method ? methods_[clause.method]
                              : clauses_[clause.definition]};
    of.purity = Least(of.purity, purity);
    if (!of.first) {
      of.first = clause.clause;
    }
    of.modeSpecific |= clause.clause->modeSpecific();
  }
  clause_ = nullptr;
  CheckPredicates(tree);
}

Purity PurityChecker::Declared(const Symbol *symbol) const {
  const Signature *sig{symbol ? types_.SignatureOf(*symbol) : nullptr};
  return sig ? sig->purity : Purity::Pure;
}

const Symbol *PurityChecker::Function(const Unify &unify) const {
  if (unify.isVar || unify.functor.kind != FunctorKind::Name ||
      unify.number >= clause_->symbols.size()) {
    return nullptr;
  }
  // With fewer arguments, it makes a closure.
  const Symbol *symbol{clause_->symbols[unify.number]};
  return symbol && symbol->kind == SymbolKind::Function &&
          unify.arguments.size() == symbol->arity
      ? symbol
      : nullptr;
}

Purity PurityChecker::Check(const Goal &goal) {
  return common::visit(
      common::visitors{
          [&](const Conj *x) {
            Purity purity{Purity::Pure};
            for (const Goal &g : x->goals) {
              purity = Least(purity, Check(g));
            }
            return purity;
          },
          [&](const Disj *x) {
            Purity purity{Purity::Pure};
            for (const Goal &g : x->goals) {
              purity = Least(purity, Check(g));
            }
            return purity;
          },
          [&](const IfThenElse *x) {
            return Least(Check(x->condition),
                Least(Check(x->then), Check(x->otherwise)));
          },
          [&](const Not *x) { return Check(x->goal); },
          [&](const Unify *x) { return CheckUnify(*x); },
          [&](const Call *x) { return CheckCall(*x); },
          // Its annotation, or impure_apply or semipure_apply, gives its
          // purity, which type checking made the closure's; mmc needs no
          // annotation for impure_apply.
          [&](const HOCall *x) { return x->purity; },
          [&](const Scope *x) { return CheckScope(*x); },
          [&](const Try *x) {
            Purity purity{Least(Check(x->goal), Check(x->then))};
            if (x->otherwise) {
              purity = Least(purity, Check(*x->otherwise));
            }
            for (const Catch &c : x->catches) {
              purity = Least(purity, Least(Check(c.pattern), Check(c.goal)));
            }
            if (x->catchAny) {
              purity = Least(purity, Check(x->catchAny->goal));
            }
            return purity;
          },
          [&](const Atomic *x) {
            Purity purity{Purity::Pure};
            for (const AtomicAlternative &alternative : x->alternatives) {
              purity = Least(purity, Check(alternative.goal));
            }
            return purity;
          },
          [&](const Loop *x) {
            Purity purity{Check(x->body)};
            if (loops_) {
              (*loops_)[x] = purity;
            }
            return purity;
          },
      },
      goal);
}

Purity PurityChecker::CheckUnify(const Unify &unify) {
  if (unify.functor.kind == FunctorKind::Lambda) {
    // Its body is at most as impure as its annotation (reference manual,
    // "Purity annotations on lambda expressions"); making it is pure.
    const Lambda &lambda{*unify.functor.lambda};
    std::optional<Purity> saved{assumed_};
    assumed_.reset();
    Purity body{Check(lambda.body)};
    assumed_ = saved;
    if (body > lambda.purity) {
      // mmc: "closure body is impure, but closure was not declared
      // impure".
      messages_.Say(Source(lambda.tokens),
          "the body of this lambda expression is %s, so it must be annotated `%s'"_err_en_US,
          Name(body), Name(body));
    }
    return Purity::Pure;
  }
  const Symbol *function{Function(unify)};
  Purity purity{Declared(function)};
  if (purity != Purity::Pure && !(assumed_ && purity <= *assumed_)) {
    // mmc: "call must be in an explicit unification which is preceded
    // by `impure' indicator".
    messages_.Say(Source(unify.tokens),
        "%s %s must be applied in a unification of its own, annotated `%s'"_err_en_US,
        Name(purity), Text(*function), Name(purity));
  }
  return purity;
}

Purity PurityChecker::CheckCall(const Call &call) {
  const Symbol *callee{call.number < clause_->symbols.size()
          ? clause_->symbols[call.number]
          : nullptr};
  Purity purity{Declared(callee)};
  if (purity != Purity::Pure && !(assumed_ && purity <= *assumed_)) {
    // mmc: "call must be preceded by `impure' indicator".
    messages_.Say(Source(call.tokens),
        "the call of %s %s must be annotated `%s'"_err_en_US, Name(purity),
        Text(*callee), Name(purity));
  }
  return purity;
}

Purity PurityChecker::CheckScope(const Scope &scope) {
  const auto *origin{std::get_if<parser::Goal>(&scope.origin)};
  switch (scope.kind) {
  case ScopeKind::Purity:
    if (origin) {
      if (const auto *annotated{std::get_if<const parser::PurityGoal *>(origin)}) {
        return CheckAnnotated(scope, (*annotated)->purity);
      }
    }
    break;
  case ScopeKind::PromisePurity:
    (void)Check(scope.goal);
    if (origin) {
      if (const auto *promise{
              std::get_if<const parser::PromisePurity *>(origin)}) {
        return (*promise)->purity;
      }
    }
    return Purity::Pure;
  case ScopeKind::Trace:
    // A trace goal's impurity does not leak (mmc).
    (void)Check(scope.goal);
    return Purity::Pure;
  default:
    break;
  }
  return Check(scope.goal);
}

Purity PurityChecker::CheckAnnotated(const Scope &scope, Purity annotation) {
  const auto &annotated{
      *std::get<const parser::PurityGoal *>(std::get<parser::Goal>(scope.origin))};
  bool unification{
      std::holds_alternative<const parser::Unification *>(annotated.goal)};
  if (!unification &&
      !std::holds_alternative<const parser::Call *>(annotated.goal) &&
      !std::holds_alternative<const parser::HigherOrderCall *>(annotated.goal)) {
    // mmc: "`impure' marker in an inappropriate place".
    messages_.Say(Source(scope.tokens),
        "`%s' annotates a call or a unification, not a compound goal"_err_en_US,
        Name(annotation));
    std::optional<Purity> saved{assumed_};
    assumed_ = annotation;
    Purity purity{Check(scope.goal)};
    assumed_ = saved;
    return purity;
  }
  // The goal desugars to its call, or its unification, after those of its
  // arguments' terms; the annotated unification's result is the variable
  // it names, and its nested function applications' are fresh variables.
  llvm::SmallVector<Goal, 4> parts;
  if (const auto *conj{std::get_if<const Conj *>(&scope.goal)}) {
    parts.append((*conj)->goals.begin(), (*conj)->goals.end());
  } else {
    parts.push_back(scope.goal);
  }
  std::optional<Purity> top;
  const Symbol *callee{nullptr};
  bool applies{false}, higherOrder{false};
  Purity purity{Purity::Pure};
  for (const Goal &part : parts) {
    if (const auto *call{std::get_if<const Call *>(&part)}) {
      callee = (*call)->number < clause_->symbols.size()
          ? clause_->symbols[(*call)->number]
          : nullptr;
      top = Declared(callee);
      purity = Least(purity, *top);
    } else if (const auto *call{std::get_if<const HOCall *>(&part)}) {
      higherOrder = true; // type checking checked its annotation
      purity = Least(purity, (*call)->purity);
    } else if (const auto *unify{std::get_if<const Unify *>(&part)}) {
      applies |= !(*unify)->isVar;
      const Symbol *function{Function(**unify)};
      if (function && unification && IsNamed((*unify)->left) && !top) {
        callee = function;
        top = Declared(function);
        purity = Least(purity, *top);
      } else {
        purity = Least(purity, Check(part));
      }
    } else {
      purity = Least(purity, Check(part));
    }
  }
  if (higherOrder && !top) {
    return purity;
  }
  if (!top) {
    if (applies) {
      // mmc: "unification with expression was declared impure, but
      // expression was not a function call".
      messages_.Say(Source(scope.tokens),
          "`%s' annotates a unification that applies no function"_err_en_US,
          Name(annotation));
    }
    return purity;
  }
  if (!callee) {
    return purity; // a type error
  }
  if (*top > annotation) {
    messages_.Say(Source(scope.tokens),
        unification
            ? "%s %s must be applied in a unification of its own, annotated `%s'"_err_en_US
            : "the call of %s %s must be annotated `%s'"_err_en_US,
        Name(*top), Text(*callee), Name(*top));
  } else if (*top < annotation) {
    // mmc: "warning: unnecessary `impure' indicator".
    messages_.Say(Source(scope.tokens),
        "the `%s' annotation is unnecessary, since %s is %s"_warn_en_US,
        Name(annotation), Text(*callee), Name(*top));
  }
  return purity;
}

// The module's own predicate or function with the name and arity.
const Symbol *FindPredicate(const ModuleSymbols &own, parser::PredOrFunc kind,
    llvm::StringRef name, unsigned arity) {
  for (const Symbol *symbol : own.Lookup(name)) {
    if (symbol->kind ==
            (kind == parser::PredOrFunc::Function ? SymbolKind::Function
                                                  : SymbolKind::Predicate) &&
        symbol->arity == arity) {
      return symbol;
    }
  }
  return nullptr;
}

void PurityChecker::CheckPredicates(const parser::ParseTree &tree) {
  const ModuleSymbols &own{resolution_.module()};
  // The number of modes of each predicate or function; those promised to
  // have equivalent clauses; and the purity of each foreign_proc for C,
  // the language merkc's foreign code is in, by its attributes (reference
  // manual, "Foreign code attributes").
  llvm::DenseMap<const Symbol *, unsigned> modes;
  llvm::DenseSet<const Symbol *> equivalent;
  struct Foreign {
    const parser::ForeignProcPragma *pragma;
    Purity purity;
  };
  llvm::MapVector<const Symbol *, llvm::SmallVector<Foreign, 1>> foreign;
  for (const parser::Item &item : tree.items) {
    if (const auto *decl{std::get_if<const parser::PredDecl *>(&item)}) {
      bool moded{(*decl)->arguments.empty() && (*decl)->determinism};
      for (const parser::TypeAndMode *argument : (*decl)->arguments) {
        moded |= argument->mode.has_value();
      }
      if (const Symbol *
          symbol{moded ? FindPredicate(own, (*decl)->kind, (*decl)->name->name,
                             (*decl)->arguments.size())
                       : nullptr}) {
        ++modes[symbol];
      }
    } else if (const auto *mode{std::get_if<const parser::ModeDecl *>(&item)}) {
      if (const Symbol *
          symbol{FindPredicate(own, (*mode)->kind, (*mode)->name->name,
              (*mode)->arguments.size())}) {
        ++modes[symbol];
      }
    } else if (const auto *pragma{
                   std::get_if<const parser::PredPragma *>(&item)}) {
      if ((*pragma)->kind == parser::PredPragmaKind::PromiseEquivalentClauses) {
        for (const Symbol *symbol :
            own.Lookup((*pragma)->pred->nameArity->name->name)) {
          if (symbol->arity == (*pragma)->pred->nameArity->arity) {
            equivalent.insert(symbol);
          }
        }
      }
    } else if (const auto *proc{
                   std::get_if<const parser::ForeignProcPragma *>(&item)}) {
      if ((*proc)->language != parser::ForeignLanguage::C) {
        continue;
      }
      if (const Symbol *
          symbol{FindPredicate(own, (*proc)->kind, (*proc)->name->name,
              (*proc)->arguments.size())}) {
        // Impure unless promised otherwise.
        Purity purity{Purity::Impure};
        for (parser::ForeignProcAttribute attribute : (*proc)->attributes) {
          if (attribute == parser::ForeignProcAttribute::PromisePure) {
            purity = Purity::Pure;
          } else if (attribute ==
              parser::ForeignProcAttribute::PromiseSemipure) {
            purity = Purity::Semipure;
          }
        }
        foreign[symbol].push_back({*proc, purity});
        Clauses &of{clauses_[symbol]};
        of.purity = Least(of.purity, purity);
      }
    }
  }
  // The promises of the pragmas, by predicate or function.
  llvm::DenseMap<const Symbol *, Purity> promised;
  for (const parser::Item &item : tree.items) {
    const auto *pragma{std::get_if<const parser::PredPragma *>(&item)};
    if (!pragma ||
        ((*pragma)->kind != parser::PredPragmaKind::PromisePure &&
            (*pragma)->kind != parser::PredPragmaKind::PromiseSemipure)) {
      continue;
    }
    Purity promise{(*pragma)->kind == parser::PredPragmaKind::PromisePure
            ? Purity::Pure
            : Purity::Semipure};
    const parser::PredNameArity &pred{*(*pragma)->pred};
    llvm::StringRef name{pred.nameArity->name->name};
    unsigned arity{pred.nameArity->arity};
    bool found{false};
    std::string arities;
    for (const Symbol *symbol : own.Lookup(name)) {
      if (symbol->kind != SymbolKind::Predicate &&
          symbol->kind != SymbolKind::Function) {
        continue;
      }
      parser::PredOrFunc kind{symbol->kind == SymbolKind::Function
              ? parser::PredOrFunc::Function
              : parser::PredOrFunc::Predicate};
      if (pred.kind && *pred.kind != kind) {
        continue;
      }
      if (symbol->arity != arity) {
        arities += (arities.empty() ? "" : ", ") + std::to_string(symbol->arity);
        continue;
      }
      promised[symbol] = promise;
      found = true;
    }
    if (!found) {
      // mmc: "`:- pragma promise_pure' declaration for `f2'/2 without
      // corresponding `:- pred' or `:- func' declaration".
      messages_.Say(Source((*pragma)->tokens),
          "`:- pragma promise_%s' names %s/%u, which this module does not declare%s"_err_en_US,
          promise == Purity::Pure ? "pure" : "semipure", name.str(), arity,
          arities.empty() ? "" : "; its arity is " + arities);
    }
  }
  // An instance's method is at least as pure as its declaration (reference
  // manual, "Declaring impure functions and predicates").
  for (const auto &[method, of] : methods_) {
    Purity declared{Declared(method->method)};
    if (of.purity > declared) {
      // mmc: "In type class method implementation: purity error:
      // predicate is impure".
      messages_.Say(Source(of.first->tokens),
          "the implementation of %s in instance %s is %s, but the method is declared %s"_err_en_US,
          Text(*method->method), method->instanceText, Name(of.purity),
          Name(declared));
    }
  }
  for (const auto &[symbol, clauses] : clauses_) {
    const auto *decl{std::get_if<const parser::PredDecl *>(&symbol->declaration)};
    if (!decl && !clauses.first) {
      continue; // foreign code only, for an undeclared predicate
    }
    Purity declared{decl ? (*decl)->purity : Purity::Pure};
    parser::CharBlock where{
        decl ? symbol->source : Source(clauses.first->tokens)};
    std::string text{Text(*symbol)};
    // Clauses for different modes may mean different things, so they are
    // impure unless promised equivalent (reference manual, "Different
    // clauses for different modes").
    Clauses of{clauses};
    if (of.modeSpecific && modes.lookup(symbol) > 1 &&
        !equivalent.count(symbol)) {
      of.purity = Purity::Impure;
    }
    if (auto it{promised.find(symbol)}; it != promised.end()) {
      Purity promise{it->second};
      if (declared != promise) {
        // mmc: "declared impure but promised pure".
        messages_.Say(where, "%s is declared %s, but promised %s"_err_en_US,
            text, Name(declared), Name(promise));
      } else if (of.purity <= declared) {
        // mmc: "warning: unnecessary `promise_pure' pragma".
        messages_.Say(where,
            "the `promise_%s' pragma is unnecessary, since %s is %s"_warn_en_US,
            Name(promise), text, Name(of.purity));
      }
      continue;
    }
    // A foreign_proc's purity is the declared one.
    bool mismatched{false};
    for (const Foreign &code : foreign.lookup(symbol)) {
      if (code.purity != declared) {
        // mmc: "foreign clause for predicate `f1'/1 has purity impure but
        // that predicate has been declared pure".
        messages_.Say(Source(code.pragma->tokens),
            "this foreign_proc for %s is %s, but %s is declared %s"_err_en_US,
            text, Name(code.purity), text, Name(declared));
        mismatched = true;
      }
    }
    if (of.purity > declared) {
      // mmc: "purity error: predicate is impure. It must be declared
      // `impure' or promised pure".
      messages_.Say(where,
          "%s is %s, so it must be declared `%s' or promised %s"_err_en_US,
          text, Name(of.purity), Name(of.purity), Name(declared));
    } else if (of.purity < declared && !mismatched) {
      // mmc: "warning: declared impure but actually pure".
      messages_.Say(where, "%s is declared %s, but it is %s"_warn_en_US, text,
          Name(declared), Name(of.purity));
    }
  }
}

} // namespace

void CheckPurity(const parser::ParseTree &tree,
    const parser::TokenSequence &tokens, const SemanticForm &form,
    const NameResolution &resolution, const TypeChecking &types,
    parser::Messages &messages, LoopPurities *loops) {
  PurityChecker{tokens, resolution, types, messages, loops}.Run(tree, form);
}

} // namespace merkurius::semantics
