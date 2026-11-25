//===-- lib/Semantics/check-pragmas.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "merkurius/Semantics/check-pragmas.h"
#include "merkurius/Common/idioms.h"
#include "merkurius/Semantics/semantic-form.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include <string>

namespace merkurius::semantics {

using namespace parser::literals;

namespace {

std::string Text(const Symbol &symbol) {
  std::string text;
  llvm::raw_string_ostream os{text};
  os << symbol;
  return text;
}

// The type variables a declaration's types mention.
struct TypeVarCollector {
  llvm::StringSet<> names;
  template <typename T> bool Pre(const T &) { return true; }
  template <typename T> void Post(const T &) {}
  void Post(const parser::TypeVar &var) { names.insert(var.name); }
};

class PragmaChecker {
public:
  PragmaChecker(const parser::TokenSequence &tokens,
      const NameResolution &resolution, const ModeChecking &modes,
      parser::Messages &messages)
      : tokens_{tokens}, resolution_{resolution}, modes_{modes},
        messages_{messages} {}

  void Run(const parser::ParseTree &);

private:
  parser::CharBlock Source(parser::TokenRange range) const {
    return {tokens_.tokens[range.first].spelling.begin(),
        tokens_.tokens[range.last].spelling.end()};
  }
  // The module's predicates or functions that a pragma names, which it
  // reports if there are none; for a mode, also its procedure, or null if
  // the predicate has no such mode, which it reports too.
  llvm::SmallVector<const Symbol *, 1> Targets(const parser::PredSpec &,
      llvm::StringRef pragma, parser::TokenRange, const Procedure **);
  void CheckObsolete(const parser::ObsoletePragma &);
  void CheckTypeSpec(const parser::TypeSpecPragma &);
  void ImportedObsolete(const ModuleSymbols &, const parser::ObsoletePragma &);
  void WarnObsoleteCalls();
  void WarnObsoleteCall(const Goal &, const GoalModes &);

  const parser::TokenSequence &tokens_;
  const NameResolution &resolution_;
  const ModeChecking &modes_;
  parser::Messages &messages_;
  // The obsolete predicates and functions, and procedures, with the
  // replacements their pragmas suggest.
  llvm::DenseMap<const Symbol *, const parser::ObsoletePragma *> obsolete_;
  llvm::DenseMap<const Procedure *, const parser::ObsoletePragma *>
      obsoleteProcs_;
  llvm::DenseSet<const void *> warned_; // goals, once each
};

llvm::SmallVector<const Symbol *, 1> PragmaChecker::Targets(
    const parser::PredSpec &spec, llvm::StringRef pragma,
    parser::TokenRange tokens, const Procedure **procedure) {
  const ModuleSymbols &own{resolution_.module()};
  llvm::SmallVector<const Symbol *, 1> targets;
  std::string name, arities;
  unsigned arity{0};
  if (const auto *pred{std::get_if<const parser::PredNameArity *>(&spec)}) {
    targets = PragmaTargets(own, **pred, &arities);
    name = (*pred)->nameArity->name->name.str();
    arity = (*pred)->nameArity->arity;
  } else {
    const parser::ProcSpec &proc{*std::get<const parser::ProcSpec *>(spec)};
    SymbolKind kind{proc.kind == parser::PredOrFunc::Function
            ? SymbolKind::Function
            : SymbolKind::Predicate};
    name = proc.name->name.str();
    arity = proc.modes.size();
    for (const Symbol *symbol : own.Lookup(proc.name->name)) {
      if (symbol->kind == kind && symbol->arity == arity) {
        targets.push_back(symbol);
      }
    }
    if (!targets.empty()) {
      *procedure = modes_.ProcedureOf(proc);
      if (!*procedure) {
        // mmc: "`:- pragma obsolete_proc' declaration for undeclared mode
        // of predicate `p'/2", "`:- pragma type_spec' declaration for
        // `p'/2 specifies non-existent mode".
        messages_.Say(Source(tokens),
            "`:- pragma %s' names a mode that %s does not have"_err_en_US,
            pragma.str(), Text(*targets.front()));
      }
    }
  }
  if (targets.empty()) {
    // mmc: "`:- pragma type_spec' declaration for `p'/2 without
    // corresponding `:- pred' or `:- func' declaration".
    messages_.Say(Source(tokens),
        "`:- pragma %s' names %s/%u, which this module does not declare%s"_err_en_US,
        pragma.str(), name, arity,
        arities.empty() ? "" : "; its arity is " + arities);
  }
  return targets;
}

void PragmaChecker::CheckObsolete(const parser::ObsoletePragma &pragma) {
  bool proc{std::holds_alternative<const parser::ProcSpec *>(pragma.pred)};
  const Procedure *procedure{nullptr};
  for (const Symbol *symbol : Targets(pragma.pred,
           proc ? "obsolete_proc" : "obsolete", pragma.tokens, &procedure)) {
    if (!proc) {
      obsolete_[symbol] = &pragma;
    }
  }
  if (procedure) {
    obsoleteProcs_[procedure] = &pragma;
  }
}

void PragmaChecker::ImportedObsolete(
    const ModuleSymbols &symbols, const parser::ObsoletePragma &pragma) {
  if (const auto *pred{
          std::get_if<const parser::PredNameArity *>(&pragma.pred)}) {
    for (const Symbol *symbol : PragmaTargets(symbols, **pred)) {
      obsolete_[symbol] = &pragma;
    }
  } else if (const Procedure *procedure{modes_.ProcedureOf(
                 *std::get<const parser::ProcSpec *>(pragma.pred))}) {
    obsoleteProcs_[procedure] = &pragma;
  }
}

void PragmaChecker::CheckTypeSpec(const parser::TypeSpecPragma &pragma) {
  const Procedure *procedure{nullptr};
  for (const Symbol *symbol :
      Targets(pragma.pred, "type_spec", pragma.tokens, &procedure)) {
    const auto *decl{
        std::get_if<const parser::PredDecl *>(&symbol->declaration)};
    if (!decl) {
      continue; // its types are inferred
    }
    TypeVarCollector declared;
    parser::Walk(**decl, declared);
    llvm::StringSet<> replaced;
    for (const parser::TypeSubstitution *binding : pragma.substitution) {
      replaced.insert(binding->var);
    }
    llvm::StringSet<> seen;
    for (const parser::TypeSubstitution *binding : pragma.substitution) {
      if (!declared.names.contains(binding->var)) {
        // mmc: "variable `T' does not occur in the `:- pred' declaration".
        messages_.Say(Source(pragma.tokens),
            "the variable %s of the `:- pragma type_spec' substitution does not occur in the declaration of %s"_err_en_US,
            binding->var.str(), Text(*symbol));
      }
      if (!seen.insert(binding->var).second) {
        // mmc: "variable `T' has multiple replacement types".
        messages_.Say(Source(pragma.tokens),
            "the `:- pragma type_spec' substitution gives the variable %s more than one type"_err_en_US,
            binding->var.str());
      }
      TypeVarCollector mentioned;
      parser::Walk(binding->type, mentioned);
      for (const auto &var : mentioned.names) {
        if (replaced.contains(var.getKey())) {
          // mmc: "variable `T' occurs on both sides of the substitution".
          messages_.Say(Source(pragma.tokens),
              "the variable %s occurs on both sides of the `:- pragma type_spec' substitution"_err_en_US,
              var.getKey().str());
        }
      }
    }
  }
}

void PragmaChecker::WarnObsoleteCall(const Goal &goal, const GoalModes &modes) {
  const Procedure *callee{modes.callee};
  if (!callee) {
    return;
  }
  const parser::ObsoletePragma *pragma{obsoleteProcs_.lookup(callee)};
  bool proc{pragma != nullptr};
  if (!pragma) {
    pragma = obsolete_.lookup(callee->symbol);
  }
  const void *key{
      std::visit([](const auto *x) -> const void * { return x; }, goal)};
  if (!pragma || !warned_.insert(key).second) {
    return;
  }
  parser::CharBlock at{
      Source(std::visit([](const auto *x) { return x->tokens; }, goal))};
  // mmc: "Warning: call to obsolete predicate `m.p'/1", "call to obsolete
  // `m.p'(out, in)".
  parser::Message &message{proc
          ? messages_.Say(at, "call to obsolete procedure %s of %s"_warn_en_US,
                ProcedureToString(*callee), Text(*callee->symbol))
          : messages_.Say(
                at, "call to obsolete %s"_warn_en_US, Text(*callee->symbol))};
  if (pragma->replacements.empty()) {
    return;
  }
  std::string list;
  for (std::size_t i{0}; i < pragma->replacements.size(); ++i) {
    const parser::NameAndArity &replacement{*pragma->replacements[i]};
    if (i > 0) {
      list += i + 1 == pragma->replacements.size() ? " and " : ", ";
    }
    list +=
        replacement.name->name.str() + "/" + std::to_string(replacement.arity);
  }
  // mmc: "The suggested replacement is `new'/2", "The possible suggested
  // replacements are `new'/2 and `newer'/2".
  message.Attach(at,
      pragma->replacements.size() == 1
          ? "the suggested replacement is %s"_en_US
          : "the possible suggested replacements are %s"_en_US,
      list);
}

// Calls and closures of obsolete predicates, functions and procedures, in
// the module's procedures that are not obsolete themselves, as in mmc.
void PragmaChecker::WarnObsoleteCalls() {
  if (obsolete_.empty() && obsoleteProcs_.empty()) {
    return;
  }
  for (const Procedure &procedure : modes_.procedures()) {
    if (obsolete_.count(procedure.symbol) || obsoleteProcs_.count(&procedure)) {
      continue;
    }
    for (const ClauseModes &clause : modes_.ClausesOf(procedure)) {
      ForEachGoal(clause.clause->clause->body, [&](const Goal &goal) {
        unsigned number{
            std::visit([](const auto *x) { return x->number; }, goal)};
        if (number < clause.goals.size()) {
          WarnObsoleteCall(goal, clause.goals[number]);
        }
      });
    }
  }
}

void PragmaChecker::Run(const parser::ParseTree &tree) {
  for (const ModuleInterface *module : resolution_.interfaces()) {
    const ModuleSymbols *symbols{resolution_.Module(module->name)};
    if (!symbols) {
      continue;
    }
    for (const parser::Item &item : module->items) {
      if (const auto *pragma{
              std::get_if<const parser::ObsoletePragma *>(&item)}) {
        ImportedObsolete(*symbols, **pragma);
      }
    }
  }
  for (const parser::Item &item : tree.items) {
    if (const auto *pragma{
            std::get_if<const parser::ObsoletePragma *>(&item)}) {
      CheckObsolete(**pragma);
    } else if (const auto *pragma{
                   std::get_if<const parser::TypeSpecPragma *>(&item)}) {
      CheckTypeSpec(**pragma);
    }
  }
  WarnObsoleteCalls();
}

} // namespace

void CheckPragmas(const parser::ParseTree &tree,
    const parser::TokenSequence &tokens, const NameResolution &resolution,
    const ModeChecking &modes, parser::Messages &messages) {
  PragmaChecker{tokens, resolution, modes, messages}.Run(tree);
}

} // namespace merkurius::semantics
