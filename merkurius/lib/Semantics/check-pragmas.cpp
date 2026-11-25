//===-- lib/Semantics/check-pragmas.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "merkurius/Semantics/check-pragmas.h"
#include "merkurius/Common/idioms.h"
#include "merkurius/Semantics/insts.h"
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

// Whether an inst has a unique or mostly unique part, or a clobbered one.
bool HasUniquePart(const Inst *inst) {
  if (!inst) {
    return false;
  }
  switch (inst->kind) {
  case InstKind::Any:
  case InstKind::Ground:
  case InstKind::HigherOrder:
    return inst->uniqueness != Uniqueness::Shared;
  case InstKind::Bound:
    return inst->uniqueness != Uniqueness::Shared ||
        llvm::any_of(inst->functors, [](const BoundAlternative &functor) {
          return llvm::any_of(functor.arguments, HasUniquePart);
        });
  default:
    return false;
  }
}

bool HasUniqueMode(const Procedure &procedure) {
  return llvm::any_of(procedure.modes, [](const ArgMode &mode) {
    return HasUniquePart(mode.initial) || HasUniquePart(mode.final);
  });
}

bool IsOutput(const ArgMode &mode) {
  return mode.initial && mode.initial->kind == InstKind::Free;
}

// Whether a fact table may have an argument of the type, as mmc allows:
// int, float or string ("Fact tables").
bool IsFactTableType(const parser::Type &type) {
  const auto *named{std::get_if<const parser::NamedType *>(&type)};
  if (!named || !(*named)->arguments.empty() ||
      (*named)->name->candidates.size() != 1) {
    return false;
  }
  const Symbol &symbol{*(*named)->name->candidates.front()};
  return symbol.module.empty() &&
      (symbol.name == "int" || symbol.name == "float" ||
          symbol.name == "string");
}

std::string Snake(parser::Determinism determinism) {
  return common::SnakeName(parser::EnumToString(determinism));
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
  void CheckTabling(const parser::TablingPragma &);
  void ReportTablingNotImplemented();
  bool CheckTabledProcedure(
      const parser::TablingPragma &, llvm::StringRef pragma, const Procedure &);
  void CheckFactTable(const parser::FactTablePragma &);
  void CheckFeatureSet(const parser::RequireFeatureSetPragma &);
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
  // The tabling pragmas so far, by predicate or function, each with its
  // procedure if it names a mode.
  llvm::DenseMap<const Symbol *,
      llvm::SmallVector<
          std::pair<const Procedure *, const parser::TablingPragma *>, 1>>
      tabled_;
  // The predicates and functions with an error in a tabling pragma, and the
  // pragmas without one, which are reported as not yet implemented unless
  // their predicate has an error in another.
  llvm::DenseSet<const Symbol *> tablingErrors_;
  llvm::SmallVector<std::pair<const parser::TablingPragma *,
      llvm::SmallVector<const Symbol *, 1>>>
      tablingOk_;
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

// The rules of "Tabled evaluation", as mmc checks them: a tabling pragma
// names a predicate or function of the module, or a mode it has; at most
// one names each procedure; none names a procedure with unique modes; a
// specified attribute gives a method for each argument, `output` exactly
// for the outputs; and minimal_model names only procedures that can
// succeed more than once. Tabling itself is not yet implemented.
void PragmaChecker::CheckTabling(const parser::TablingPragma &pragma) {
  std::string name{common::SnakeName(parser::EnumToString(pragma.method))};
  bool modeForm{std::holds_alternative<const parser::ProcSpec *>(pragma.pred)};
  const Procedure *procedure{nullptr};
  auto targets{Targets(pragma.pred, name, pragma.tokens, &procedure)};
  bool ok{!targets.empty() && (!modeForm || procedure)};
  for (const Symbol *symbol : modeForm && !procedure
          ? llvm::ArrayRef<const Symbol *>{}
          : llvm::ArrayRef<const Symbol *>{targets}) {
    auto &earlier{tabled_[symbol]};
    for (const auto &[proc, other] : earlier) {
      if (proc && procedure && proc != procedure) {
        continue; // another mode
      }
      if (other->method == pragma.method) {
        // mmc: "predicate `m.p'/2 has duplicate memo pragmas specified".
        messages_.Say(Source(pragma.tokens),
            "%s already has a `:- pragma %s'"_err_en_US, Text(*symbol), name);
      } else {
        // mmc: "predicate `m.p'/2 has both memo and loop_check pragmas
        // specified. Only one kind of tabling pragma may be applied to
        // it".
        messages_.Say(Source(pragma.tokens),
            "%s has both a `:- pragma %s' and a `:- pragma %s'; only one kind of tabling may be applied to it"_err_en_US,
            Text(*symbol),
            common::SnakeName(parser::EnumToString(other->method)), name);
      }
      ok = false;
      tablingErrors_.insert(symbol);
      break;
    }
    earlier.emplace_back(procedure, &pragma);
    llvm::SmallVector<const Procedure *, 2> procedures;
    if (procedure) {
      procedures.push_back(procedure);
    } else {
      llvm::append_range(procedures, modes_.ProceduresOf(*symbol));
    }
    for (const Procedure *proc : procedures) {
      if (!CheckTabledProcedure(pragma, name, *proc)) {
        ok = false;
        tablingErrors_.insert(symbol);
      }
    }
  }
  if (ok) {
    tablingOk_.emplace_back(&pragma, targets);
  }
}

void PragmaChecker::ReportTablingNotImplemented() {
  for (const auto &[pragma, symbols] : tablingOk_) {
    if (llvm::any_of(symbols, [&](const Symbol *symbol) {
          return tablingErrors_.contains(symbol);
        })) {
      continue;
    }
    messages_.Say(Source(pragma->tokens), "%s"_todo_en_US,
        "tabled evaluation (`:- pragma " +
            common::SnakeName(parser::EnumToString(pragma->method)) + "')");
  }
}

bool PragmaChecker::CheckTabledProcedure(const parser::TablingPragma &pragma,
    llvm::StringRef name, const Procedure &procedure) {
  parser::CharBlock at{Source(pragma.tokens)};
  // mmc reports these two at the procedure's declaration.
  parser::CharBlock declared{
      procedure.source.empty() ? procedure.symbol->source : procedure.source};
  if (HasUniqueMode(procedure)) {
    // mmc: "`:- pragma memo' declaration not allowed for procedure with
    // unique modes".
    messages_
        .Say(declared,
            "%s of %s has unique modes, so it cannot be tabled; tabling records its arguments"_err_en_US,
            ProcedureToString(procedure), Text(*procedure.symbol))
        .Attach(at, "the `:- pragma %s' is here"_en_US, name.str());
    return false;
  }
  if (pragma.method == parser::TablingMethod::MinimalModel &&
      procedure.determinism &&
      *procedure.determinism != parser::Determinism::Nondet &&
      *procedure.determinism != parser::Determinism::Multi) {
    // mmc: "`:- pragma minimal_model' declaration not allowed for
    // procedure with determinism `semidet'".
    messages_
        .Say(declared,
            "%s of %s is %s, so it cannot have minimal model tabling, which is for procedures that can succeed more than once"_err_en_US,
            ProcedureToString(procedure), Text(*procedure.symbol),
            Snake(*procedure.determinism))
        .Attach(at, "the `:- pragma minimal_model' is here"_en_US);
    return false;
  }
  if (!pragma.attributes || !pragma.attributes->specified) {
    return true;
  }
  llvm::ArrayRef<parser::TabledArg> methods{*pragma.attributes->specified};
  if (methods.size() != procedure.modes.size()) {
    // mmc: "not enough argument tabling methods specified", "too many
    // argument tabling methods specified".
    messages_.Say(at,
        "`:- pragma %s' specifies %zu argument tabling methods, but %s of %s has %zu arguments"_err_en_US,
        name.str(), methods.size(), ProcedureToString(procedure),
        Text(*procedure.symbol), procedure.modes.size());
    return false;
  }
  bool ok{true};
  for (std::size_t i{0}; i < methods.size(); ++i) {
    bool output{IsOutput(procedure.modes[i])};
    if (output != (methods[i] == parser::TabledArg::Output)) {
      // mmc: "argument 1: argument tabling method `value' is not
      // compatible with output modes".
      messages_.Say(at,
          "argument %zu of %s of %s is an %s, so its tabling method cannot be '%s'"_err_en_US,
          i + 1, ProcedureToString(procedure), Text(*procedure.symbol),
          output ? "output" : "input",
          common::SnakeName(parser::EnumToString(methods[i])));
      ok = false;
    }
  }
  return ok;
}

// A fact table names a predicate or function of the module whose arguments
// are of the types mmc allows ("Fact tables"); fact tables themselves are
// not yet implemented.
void PragmaChecker::CheckFactTable(const parser::FactTablePragma &pragma) {
  std::string arities;
  auto targets{PragmaTargets(resolution_.module(), *pragma.pred, &arities)};
  if (targets.empty()) {
    // mmc: "`:- pragma fact_table' declaration for `p'/2 without
    // corresponding `:- pred' or `:- func' declaration".
    messages_.Say(Source(pragma.tokens),
        "`:- pragma fact_table' names %s/%u, which this module does not declare%s"_err_en_US,
        pragma.pred->nameArity->name->name.str(), pragma.pred->nameArity->arity,
        arities.empty() ? "" : "; its arity is " + arities);
    return;
  }
  bool ok{true};
  for (const Symbol *symbol : targets) {
    const auto *decl{
        std::get_if<const parser::PredDecl *>(&symbol->declaration)};
    if (!decl) {
      continue;
    }
    llvm::SmallVector<const parser::Type *, 4> types;
    for (const parser::TypeAndMode *argument : (*decl)->arguments) {
      types.push_back(&argument->type);
    }
    if ((*decl)->result) {
      types.push_back(&(*decl)->result->type);
    }
    for (std::size_t i{0}; i < types.size(); ++i) {
      if (!IsFactTableType(*types[i])) {
        // mmc: "type `list.list(int)' is not allowed in fact tables. The
        // only types allowed in fact tables are `int', `float', and
        // `string'".
        // mmc reports it at the declaration.
        messages_
            .Say(symbol->source,
                "argument %zu of %s is not an int, a float or a string, the only types a fact table may have"_err_en_US,
                i + 1, Text(*symbol))
            .Attach(Source(pragma.tokens),
                "the `:- pragma fact_table' is here"_en_US);
        ok = false;
      }
    }
  }
  if (ok) {
    messages_.Say(Source(pragma.tokens), "%s"_todo_en_US,
        "fact tables (`:- pragma fact_table')");
  }
}

// The features merkc's one build has ("Feature sets"): concurrency, double
// precision floats and the strict sequential semantics. It has neither
// single precision floats nor a conservative collector, and tabling,
// parallel conjunctions and trailing are not yet implemented.
void PragmaChecker::CheckFeatureSet(
    const parser::RequireFeatureSetPragma &pragma) {
  parser::CharBlock at{Source(pragma.tokens)};
  for (parser::Feature feature : pragma.features) {
    switch (feature) {
    case parser::Feature::Concurrency:
    case parser::Feature::DoublePrecFloat:
    case parser::Feature::StrictSequential:
      break;
    case parser::Feature::SinglePrecFloat:
      // mmc: "this module must be compiled in a grade that uses single
      // precision floats".
      messages_.Say(at,
          "this module requires single precision floats, and merkc's floats are double precision"_err_en_US);
      break;
    case parser::Feature::ConservativeGc:
      messages_.Say(at,
          "this module requires conservative garbage collection, and merkc's collector is precise"_err_en_US);
      break;
    case parser::Feature::Memo:
      messages_.Say(
          at, "%s"_todo_en_US, "tabled evaluation (the feature 'memo')");
      break;
    case parser::Feature::ParallelConj:
      messages_.Say(at, "%s"_todo_en_US,
          "parallel conjunctions (the feature 'parallel_conj')");
      break;
    case parser::Feature::Trailing:
      messages_.Say(at, "%s"_todo_en_US, "trailing (the feature 'trailing')");
      break;
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
    } else if (const auto *pragma{
                   std::get_if<const parser::TablingPragma *>(&item)}) {
      CheckTabling(**pragma);
    } else if (const auto *pragma{
                   std::get_if<const parser::FactTablePragma *>(&item)}) {
      CheckFactTable(**pragma);
    } else if (const auto *pragma{
                   std::get_if<const parser::RequireFeatureSetPragma *>(
                       &item)}) {
      CheckFeatureSet(**pragma);
    }
  }
  ReportTablingNotImplemented();
  WarnObsoleteCalls();
}

} // namespace

void CheckPragmas(const parser::ParseTree &tree,
    const parser::TokenSequence &tokens, const NameResolution &resolution,
    const ModeChecking &modes, parser::Messages &messages) {
  PragmaChecker{tokens, resolution, modes, messages}.Run(tree);
}

} // namespace merkurius::semantics
