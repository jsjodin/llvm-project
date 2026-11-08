//===-- lib/Semantics/check-receives.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "merkurius/Semantics/check-receives.h"
#include "merkurius/Common/idioms.h"
#include "merkurius/Semantics/semantic-form.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <string>
#include <utility>
#include <vector>

namespace merkurius::semantics {

using namespace parser::literals;

namespace {

// A predicate's receive goals' arms for one mailbox.
struct MailboxArms {
  const ClauseTypes *clause; // where the first is
  TokenRange mailbox; // the first's
  const Type *type; // of its messages
  bool all{false}; // some arm selects every message
  llvm::SmallPtrSet<const Symbol *, 8> selected;
};

class ReceiveChecker {
public:
  ReceiveChecker(const parser::TokenSequence &tokens, TypeChecking &types,
      InstTypes &constructors, parser::Messages &messages)
      : tokens_{tokens}, types_{types}, constructors_{constructors},
        messages_{messages} {}

  void Check();

private:
  parser::CharBlock Source(TokenRange range) const {
    return {tokens_.tokens[range.first].spelling.begin(),
        tokens_.tokens[range.last].spelling.end()};
  }
  void Walk(const Goal &goal);
  void Case(const Lambda &lambda);
  const Unify *PatternOf(const Goal &condition, VarId message) const;
  void Warn(const MailboxArms &arms);

  const parser::TokenSequence &tokens_;
  TypeChecking &types_;
  InstTypes &constructors_;
  parser::Messages &messages_;
  const ClauseTypes *clause_{nullptr};
  llvm::MapVector<std::pair<const Symbol *, llvm::StringRef>, MailboxArms>
      mailboxes_;
  llvm::DenseSet<const Symbol *> typeErrors_;
};

void ReceiveChecker::Check() {
  for (const ClauseTypes &clause : types_.clauses()) {
    if (clause.typeError) {
      typeErrors_.insert(clause.definition);
      continue;
    }
    clause_ = &clause;
    Walk(clause.clause->body);
  }
  for (const auto &[key, arms] : mailboxes_) {
    if (!arms.all && !typeErrors_.count(key.first)) {
      Warn(arms);
    }
  }
}

void ReceiveChecker::Walk(const Goal &goal) {
  common::visit(
      common::visitors{
          [&](const Conj *x) {
            for (const Goal &g : x->goals) {
              Walk(g);
            }
          },
          [&](const Disj *x) {
            for (const Goal &g : x->goals) {
              Walk(g);
            }
          },
          [&](const IfThenElse *x) {
            Walk(x->condition);
            Walk(x->then);
            Walk(x->otherwise);
          },
          [&](const Not *x) { Walk(x->goal); },
          [&](const Unify *x) {
            if (x->functor.kind == FunctorKind::Lambda) {
              if (x->functor.lambda->receive) {
                Case(*x->functor.lambda);
              }
              Walk(x->functor.lambda->body);
            }
          },
          [&](const Call *) {},
          [&](const HOCall *) {},
          [&](const Scope *x) { Walk(x->goal); },
          [&](const Try *x) {
            Walk(x->goal);
            Walk(x->then);
            if (x->otherwise) {
              Walk(*x->otherwise);
            }
            for (const Catch &c : x->catches) {
              Walk(c.goal);
            }
            if (x->catchAny) {
              Walk(x->catchAny->goal);
            }
          },
          [&](const Atomic *x) {
            for (const AtomicAlternative &a : x->alternatives) {
              Walk(a.goal);
            }
          },
          [&](const Loop *x) { Walk(x->body); },
      },
      goal);
}

// The unification of the message with a pattern's functor, in the
// condition that tries it.
const Unify *ReceiveChecker::PatternOf(
    const Goal &condition, VarId message) const {
  if (const auto *unify{std::get_if<const Unify *>(&condition)}) {
    return (*unify)->left == message && !(*unify)->isVar ? *unify : nullptr;
  }
  if (const auto *conj{std::get_if<const Conj *>(&condition)}) {
    for (const Goal &goal : (*conj)->goals) {
      if (const Unify *unify{PatternOf(goal, message)}) {
        return unify;
      }
    }
  }
  return nullptr;
}

void ReceiveChecker::Case(const Lambda &lambda) {
  const ReceiveCase &receive{*lambda.receive};
  VarId message{lambda.arguments.front()};
  auto [it, added]{mailboxes_.insert({{clause_->definition, receive.text},
      MailboxArms{clause_, receive.mailbox, clause_->vars[message]}})};
  MailboxArms &arms{it->second};
  // The arms, in order, are the conditions of the chain of if-then-elses;
  // a chain of another shape is taken to select everything, so that it
  // draws no warning.
  Goal goal{lambda.body};
  for (bool takesAll : receive.takesAll) {
    const auto *chain{std::get_if<const IfThenElse *>(&goal)};
    if (!chain) {
      arms.all = true;
      return;
    }
    if (takesAll) {
      arms.all = true;
    } else if (const Unify *pattern{PatternOf((*chain)->condition, message)}) {
      if (pattern->functor.kind == FunctorKind::Tuple) {
        arms.all = true;
      } else if (const Symbol *symbol{clause_->symbols[pattern->number]};
                 symbol && symbol->kind == SymbolKind::Constructor) {
        arms.selected.insert(symbol);
      }
    }
    goal = (*chain)->otherwise;
  }
}

// mmc has no receive goal.
void ReceiveChecker::Warn(const MailboxArms &arms) {
  std::vector<ConstructorTypes> constructors;
  std::string text{Source(arms.mailbox).ToString()};
  if (!constructors_.Constructors(arms.type, constructors)) {
    messages_.Say(Source(arms.mailbox),
        "no arm selects every message of the mailbox %s, of type %s, so the others stay queued for another receive [-Wunselected-message]"_warn_en_US,
        text, types_.Print(*arms.clause, *arms.type));
    return;
  }
  std::string missing;
  std::size_t count{0};
  for (const ConstructorTypes &constructor : constructors) {
    if (constructor.symbol && !arms.selected.count(constructor.symbol)) {
      missing += (missing.empty() ? "" : ", ") + constructor.functor.name.str() +
          "/" + std::to_string(constructor.functor.arity);
      ++count;
    }
  }
  if (count == 0) {
    return;
  }
  messages_.Say(Source(arms.mailbox),
      "no arm selects the messages of the mailbox %s whose functor is %s %s, so they stay queued for another receive [-Wunselected-message]"_warn_en_US,
      text, count == 1 ? "the constructor" : "one of the constructors",
      missing);
}

} // namespace

void CheckReceives(const parser::TokenSequence &tokens, TypeChecking &types,
    InstTypes &constructors, parser::Messages &messages) {
  ReceiveChecker{tokens, types, constructors, messages}.Check();
}

} // namespace merkurius::semantics
