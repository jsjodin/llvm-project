//===-- lib/Semantics/check-foreign.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "merkurius/Semantics/check-foreign.h"
#include "merkurius/Common/idioms.h"
#include "merkurius/Semantics/insts.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace merkurius::semantics {

using namespace parser::literals;

namespace {

class ForeignChecker {
public:
  ForeignChecker(const parser::TokenSequence &tokens,
      const NameResolution &resolution, const ModeChecking &modes,
      const DeterminismChecking &determinism, parser::Messages &messages)
      : tokens_{tokens}, resolution_{resolution}, modes_{modes},
        determinism_{determinism}, messages_{messages} {}

  void Run(const parser::ParseTree &);

private:
  parser::CharBlock Source(parser::TokenRange range) const {
    return {tokens_.tokens[range.first].spelling.begin(),
        tokens_.tokens[range.last].spelling.end()};
  }
  void CheckForeignProcs();
  void CheckExport(const parser::ForeignExportPragma &);
  void CheckForeignType(const parser::ForeignTypePragma &, bool interface);
  void CheckSupertypes();
  void CheckForeignEnum(const parser::ForeignEnumPragma &);
  void CheckExportEnum(const parser::ForeignExportEnumPragma &);
  void CheckForeignImport(const parser::ForeignImportModulePragma &);
  // A pragma that passes its checks but is not yet implemented, reported
  // only if no pragma is in error, as an error would show what to fix first.
  void Todo(parser::CharBlock at, llvm::StringRef what) {
    todo_.emplace_back(at, what.str());
  }
  // The declarations and definitions of a type of the module, by name and
  // arity, as the items give them: its symbol has only the first.
  struct TypeItems {
    const parser::DuTypeDefn *du{nullptr};
    const parser::EquivalenceTypeDefn *equivalence{nullptr};
    const parser::AbstractTypeDecl *abstract{nullptr};
    bool exported{false}; // a `:- type` of it is in the interface
    parser::CharBlock declared; // the first `:- type` of it
    llvm::SmallVector<const parser::ForeignTypePragma *, 1> foreign;
  };
  const TypeItems *ItemsOf(llvm::StringRef name, unsigned arity) const;
  // The discriminated union definition of a type, if it has one.
  const parser::DuTypeDefn *DuOf(const Symbol &type) const;
  // The type that a pragma names as t/N: of the module, or with
  // `imported`, of a module it imports too.
  const Symbol *NamedType(const parser::NameAndArity &, bool imported) const;
  // The constructors of a discriminated union type, in order.
  llvm::SmallVector<const Symbol *> Constructors(const Symbol &type) const;
  // A constructor of non-zero arity of a discriminated union type, which
  // is then not an enumeration.
  const Symbol *NonConstant(const Symbol &type) const;

  const parser::TokenSequence &tokens_;
  const NameResolution &resolution_;
  const ModeChecking &modes_;
  const DeterminismChecking &determinism_;
  parser::Messages &messages_;
  std::vector<std::pair<parser::CharBlock, std::string>> todo_;
  std::map<std::pair<std::string, unsigned>, TypeItems> types_;
  // The foreign_enum of each type for each language.
  std::map<std::pair<const Symbol *, parser::ForeignLanguage>,
      const parser::ForeignEnumPragma *>
      enums_;
};

std::string TypeText(const Symbol &type) {
  return "`" + type.name.str() + "'/" + std::to_string(type.arity);
}

// A constructor named in a pragma, unqualified or qualified with the type's
// module, which mmc accepts though the manual says it must not be qualified
// (see the register).
bool Names(const parser::SymName &name, const Symbol &constructor) {
  return name.name == constructor.name &&
      (name.qualifiers.empty() ||
          llvm::join(name.qualifiers, ".") == constructor.module);
}

void ForeignChecker::Run(const parser::ParseTree &tree) {
  bool interface{false};
  for (const parser::Item &item : tree.items) {
    if (const auto *section{std::get_if<const parser::SectionDecl *>(&item)}) {
      interface = (*section)->section == parser::Section::Interface;
    } else if (const auto *du{std::get_if<const parser::DuTypeDefn *>(&item)}) {
      TypeItems &type{
          types_[{(*du)->name->name.str(), (*du)->parameters.size()}]};
      type.du = *du;
      type.exported |= interface;
      if (type.declared.empty()) {
        type.declared = Source((*du)->name->tokens);
      }
    } else if (const auto *equivalence{
                   std::get_if<const parser::EquivalenceTypeDefn *>(&item)}) {
      TypeItems &type{types_[{(*equivalence)->name->name.str(),
          (*equivalence)->parameters.size()}]};
      type.equivalence = *equivalence;
      type.exported |= interface;
      if (type.declared.empty()) {
        type.declared = Source((*equivalence)->name->tokens);
      }
    } else if (const auto *abstract{
                   std::get_if<const parser::AbstractTypeDecl *>(&item)}) {
      TypeItems &type{types_[{
          (*abstract)->name->name.str(), (*abstract)->parameters.size()}]};
      type.abstract = *abstract;
      type.exported |= interface;
      if (type.declared.empty()) {
        type.declared = Source((*abstract)->name->tokens);
      }
    } else if (const auto *foreign{
                   std::get_if<const parser::ForeignTypePragma *>(&item)}) {
      types_[{(*foreign)->name->name.str(), (*foreign)->parameters.size()}]
          .foreign.push_back(*foreign);
    }
  }
  CheckForeignProcs();
  interface = false;
  for (const parser::Item &item : tree.items) {
    if (const auto *section{std::get_if<const parser::SectionDecl *>(&item)}) {
      interface = (*section)->section == parser::Section::Interface;
    } else if (const auto *exported{
                   std::get_if<const parser::ForeignExportPragma *>(&item)}) {
      CheckExport(**exported);
    } else if (const auto *type{
                   std::get_if<const parser::ForeignTypePragma *>(&item)}) {
      CheckForeignType(**type, interface);
    } else if (const auto *enumeration{
                   std::get_if<const parser::ForeignEnumPragma *>(&item)}) {
      CheckForeignEnum(**enumeration);
    } else if (const auto *exportEnum{
                   std::get_if<const parser::ForeignExportEnumPragma *>(
                       &item)}) {
      CheckExportEnum(**exportEnum);
    } else if (const auto *import{
                   std::get_if<const parser::ForeignImportModulePragma *>(
                       &item)}) {
      CheckForeignImport(**import);
    }
  }
  CheckSupertypes();
  if (!messages_.AnyFatalError()) {
    for (const auto &[at, what] : todo_) {
      messages_.Say(at, "%s"_todo_en_US, what);
    }
  }
}

const ForeignChecker::TypeItems *ForeignChecker::ItemsOf(
    llvm::StringRef name, unsigned arity) const {
  auto it{types_.find({name.str(), arity})};
  return it == types_.end() ? nullptr : &it->second;
}

const parser::DuTypeDefn *ForeignChecker::DuOf(const Symbol &type) const {
  if (const auto *du{
          std::get_if<const parser::DuTypeDefn *>(&type.declaration)}) {
    return *du;
  }
  if (type.module == resolution_.module().name()) {
    if (const TypeItems *items{ItemsOf(type.name, type.arity)}) {
      return items->du;
    }
  }
  return nullptr;
}

const Symbol *ForeignChecker::NamedType(
    const parser::NameAndArity &named, bool imported) const {
  const parser::SymName &name{*named.name};
  std::string qualifier{llvm::join(name.qualifiers, ".")};
  llvm::SmallVector<const ModuleSymbols *> modules;
  if (qualifier.empty() || qualifier == resolution_.module().name()) {
    modules.push_back(&resolution_.module());
  }
  if (imported) {
    for (const ModuleInterface *interface : resolution_.interfaces()) {
      if (const ModuleSymbols *module{resolution_.Module(interface->name)}) {
        if (qualifier.empty() || qualifier == module->name()) {
          modules.push_back(module);
        }
      }
    }
  }
  for (const ModuleSymbols *module : modules) {
    // Its definition, if it has one, else any declaration.
    const Symbol *found{nullptr};
    for (const Symbol *symbol : module->Lookup(name.name)) {
      if (symbol->kind == SymbolKind::Type && symbol->arity == named.arity &&
          (!found ||
              std::holds_alternative<const parser::DuTypeDefn *>(
                  symbol->declaration))) {
        found = symbol;
      }
    }
    if (found) {
      return found;
    }
  }
  return nullptr;
}

llvm::SmallVector<const Symbol *> ForeignChecker::Constructors(
    const Symbol &type) const {
  llvm::SmallVector<const Symbol *> result;
  const parser::DuTypeDefn *defn{DuOf(type)};
  if (!defn) {
    return result;
  }
  const ModuleSymbols *module{type.module == resolution_.module().name()
          ? &resolution_.module()
          : resolution_.Module(type.module)};
  if (!module) {
    return result;
  }
  for (const parser::Constructor *constructor : defn->constructors) {
    for (const Symbol *symbol : module->Lookup(constructor->name->name)) {
      if (symbol->kind == SymbolKind::Constructor && symbol->owner == &type &&
          symbol->arity == constructor->arguments.size()) {
        result.push_back(symbol);
        break;
      }
    }
  }
  return result;
}

const Symbol *ForeignChecker::NonConstant(const Symbol &type) const {
  for (const Symbol *constructor : Constructors(type)) {
    if (constructor->arity != 0) {
      return constructor;
    }
  }
  return nullptr;
}

// A foreign_proc may not be multi or nondet (reference manual, "pragma
// foreign_proc"); mmc accepts one that nothing calls, and aborts compiling
// a call of one (see the register). An erroneous one may not promise
// will_not_throw_exception ("Foreign code attributes").
void ForeignChecker::CheckForeignProcs() {
  for (const Procedure &procedure : modes_.procedures()) {
    const parser::ForeignProcPragma *pragma{modes_.ForeignProcOf(procedure)};
    if (!pragma) {
      continue;
    }
    std::optional<parser::Determinism> d{determinism_.DeterminismOf(procedure)};
    if (!d) {
      continue;
    }
    if (*d == parser::Determinism::Multi || *d == parser::Determinism::Nondet) {
      messages_.Say(Source(pragma->tokens),
          "%s is %s, but a foreign_proc may not be multi or nondet"_err_en_US,
          ProcedureToString(procedure), DeterminismName(*d));
    }
    if (*d == parser::Determinism::Erroneous &&
        llvm::is_contained(pragma->attributes,
            parser::ForeignProcAttribute::WillNotThrowException)) {
      // mmc: "`p'(out) has determinism erroneous but also has foreign
      // clauses that have a `will_not_throw_exception' attribute."
      parser::CharBlock at{procedure.source.empty() ? procedure.symbol->source
                                                    : procedure.source};
      messages_
          .Say(at,
              "%s is erroneous, so its foreign_proc may not be will_not_throw_exception"_err_en_US,
              ProcedureToString(procedure))
          .Attach(
              Source(pragma->tokens), "the foreign_proc is here"_because_en_US);
    }
  }
}

// A procedure exported to foreign code may not be multi or nondet
// (reference manual, "Calling Merkurius from foreign code").
void ForeignChecker::CheckExport(const parser::ForeignExportPragma &pragma) {
  const Procedure *procedure{modes_.ExportOf(pragma)};
  if (!procedure) {
    return; // reported by mode analysis
  }
  std::optional<parser::Determinism> d{determinism_.DeterminismOf(*procedure)};
  if (d &&
      (*d == parser::Determinism::Multi || *d == parser::Determinism::Nondet)) {
    // mmc: "`:- pragma foreign_export' declaration for a procedure that has
    // a declared determinism of multi."
    messages_.Say(Source(pragma.tokens),
        "%s is %s, but a procedure exported to foreign code may not be multi or nondet"_err_en_US,
        ProcedureToString(*procedure), DeterminismName(*d));
    return;
  }
  Todo(Source(pragma.tokens), "pragma foreign_export");
}

// A foreign_type defines a type that a `:- type` declares or defines, as
// an abstract or discriminated union type, and is no more visible than it
// (reference manual, "Using foreign types from Merkurius").
void ForeignChecker::CheckForeignType(
    const parser::ForeignTypePragma &pragma, bool interface) {
  const parser::SymName &name{*pragma.name};
  std::string text{
      "`" + name.name.str() + "'/" + std::to_string(pragma.parameters.size())};
  const TypeItems *type{ItemsOf(name.name, pragma.parameters.size())};
  if (type && type->equivalence) {
    // mmc: "this definition of `t'/0 is incompatible with the equivalence
    // type definition".
    messages_
        .Say(Source(pragma.tokens),
            "%s is an equivalence type, so it may not have a foreign_type"_err_en_US,
            text)
        .Attach(Source(type->equivalence->name->tokens),
            "the equivalence is defined here"_because_en_US);
  } else if (type && type->du && type->du->supertype) {
    // mmc: "`s'/0 is defined to be a subtype, so there must not be any
    // `:- pragma foreign_type' declarations for it."
    messages_
        .Say(Source(pragma.tokens),
            "%s is a subtype, so it may not have a foreign_type"_err_en_US,
            text)
        .Attach(Source(type->du->name->tokens),
            "the subtype is defined here"_because_en_US);
  } else if (!type || (!type->du && !type->abstract)) {
    // mmc: "a `:- pragma foreign_type' definition for `t'/0 without either
    // a Mercury definition or a Mercury declaration for `t'/0."
    messages_.Say(Source(pragma.tokens),
        "this foreign_type is for %s, which the module does not declare with `:- type'"_err_en_US,
        text);
  } else if (interface && !type->exported) {
    // mmc: "since the Mercury declaration of `t'/0 is in the
    // implementation section, any foreign type definition for it must be
    // in the implementation section as well."
    messages_
        .Say(Source(pragma.tokens),
            "%s is declared in the implementation section, so its foreign_type must be there too"_err_en_US,
            text)
        .Attach(type->declared, "it is declared here"_because_en_US);
  } else if (type->du) {
    // An abstract type's foreign_type is its definition, whose values code
    // generation reports; a discriminated union's would replace it, which
    // it would ignore.
    Todo(Source(pragma.tokens),
        "pragma foreign_type for a discriminated union type");
  }
}

// A type with a foreign_type may not be the supertype of a subtype
// (reference manual, "Using foreign types from Merkurius").
void ForeignChecker::CheckSupertypes() {
  for (const auto &[key, type] : types_) {
    if (!type.du || !type.du->supertype) {
      continue;
    }
    const auto *named{
        std::get_if<const parser::NamedType *>(&*type.du->supertype)};
    if (!named) {
      continue;
    }
    for (const Symbol *super : (*named)->name->candidates) {
      if (super->kind != SymbolKind::Type ||
          super->module != resolution_.module().name()) {
        continue;
      }
      const TypeItems *items{ItemsOf(super->name, super->arity)};
      if (items && !items->foreign.empty()) {
        // mmc: "`t'/0 cannot be a supertype because it is a foreign
        // type."
        messages_.Say(Source(type.du->name->tokens),
            "%s may not be a supertype, since it has a foreign_type"_err_en_US,
            TypeText(*super));
        break;
      }
    }
  }
}

// A foreign_enum gives each constructor of an enumeration type that the
// module defines a value of its own, at most once for each language
// (reference manual, "Using foreign enumerations in Merkurius code").
void ForeignChecker::CheckForeignEnum(const parser::ForeignEnumPragma &pragma) {
  parser::CharBlock at{Source(pragma.tokens)};
  const Symbol *type{NamedType(*pragma.type, /*imported=*/false)};
  if (!type) {
    // mmc: "`:- pragma foreign_enum' declaration for the undeclared type
    // `u'/0."
    std::string text;
    llvm::raw_string_ostream os{text};
    pragma.type->PrintInline(os);
    messages_.Say(at,
        "this foreign_enum is for `%s', which the module does not define"_err_en_US,
        text);
    return;
  }
  auto [it, added]{enums_.try_emplace({type, pragma.language}, &pragma)};
  if (!added) {
    // mmc: "duplicate foreign enum definition in C for `t'/0."
    messages_
        .Say(at, "%s has another foreign_enum for this language"_err_en_US,
            TypeText(*type))
        .Attach(Source(it->second->tokens),
            "the other foreign_enum is here"_because_en_US);
    return;
  }
  if (!DuOf(*type)) {
    messages_.Say(at,
        "%s is not a discriminated union type, so it may not have a foreign_enum"_err_en_US,
        TypeText(*type));
    return;
  }
  if (const Symbol *constructor{NonConstant(*type)}) {
    // mmc: "the Mercury definition of `t'/0 is not an enumeration type, so
    // there must not be any `:- pragma foreign_enum' declarations for it."
    messages_
        .Say(at,
            "%s is not an enumeration type, so it may not have a foreign_enum"_err_en_US,
            TypeText(*type))
        .Attach(constructor->source,
            "it has this constructor of non-zero arity: `%s'/%u"_because_en_US,
            constructor->name.str(), constructor->arity);
    return;
  }
  llvm::SmallVector<const Symbol *> constructors{Constructors(*type)};
  llvm::DenseSet<const Symbol *> valued;
  llvm::StringSet<> values;
  for (const parser::ForeignEnumValue *value : pragma.values) {
    const Symbol *constructor{nullptr};
    for (const Symbol *c : constructors) {
      if (Names(*value->constructor, *c)) {
        constructor = c;
      }
    }
    if (!constructor) {
      // mmc: "the following symbol is not a constructor of the type: `c'."
      messages_.Say(Source(value->tokens),
          "`%s' is not a constructor of %s"_err_en_US,
          value->constructor->name.str(), TypeText(*type));
      continue;
    }
    if (!valued.insert(constructor).second) {
      // mmc: "the specified mapping between the names of Mercury
      // constructors and the corresponding foreign values is
      // inconsistent. The following Mercury constructor name is repeated."
      messages_.Say(Source(value->tokens),
          "`%s' is given more than one foreign value"_err_en_US,
          constructor->name.str());
    }
    if (!values.insert(value->value).second) {
      // mmc: "... The following foreign value is repeated: `1'."
      messages_.Say(Source(value->tokens),
          "the foreign value `%s' is given to more than one constructor"_err_en_US,
          value->value.str());
    }
  }
  for (const Symbol *constructor : constructors) {
    if (!valued.contains(constructor)) {
      // mmc: "the following constructor does not have a foreign value:
      // `b'."
      messages_.Say(at, "`%s' of %s has no foreign value"_err_en_US,
          constructor->name.str(), TypeText(*type));
    }
  }
  Todo(at, "pragma foreign_enum");
}

// A foreign_export_enum is for an enumeration type, of the module or one
// it imports, and gives its constructors distinct symbolic names
// (reference manual, "Using Merkurius enumerations in foreign code").
void ForeignChecker::CheckExportEnum(
    const parser::ForeignExportEnumPragma &pragma) {
  parser::CharBlock at{Source(pragma.tokens)};
  const Symbol *type{NamedType(*pragma.type, /*imported=*/true)};
  if (!type) {
    std::string text;
    llvm::raw_string_ostream os{text};
    pragma.type->PrintInline(os);
    messages_.Say(at,
        "this foreign_export_enum is for `%s', which is not a type the module can see"_err_en_US,
        text);
    return;
  }
  if (!DuOf(*type)) {
    messages_.Say(
        at, "%s is not an enumeration type"_err_en_US, TypeText(*type));
    return;
  }
  if (const Symbol *constructor{NonConstant(*type)}) {
    // mmc: "`m.t'/0 is not an enumeration type. It has this non-zero arity
    // constructor: `b'/1."
    messages_
        .Say(at, "%s is not an enumeration type"_err_en_US, TypeText(*type))
        .Attach(constructor->source,
            "it has this constructor of non-zero arity: `%s'/%u"_because_en_US,
            constructor->name.str(), constructor->arity);
    return;
  }
  llvm::SmallVector<const Symbol *> constructors{Constructors(*type)};
  llvm::DenseMap<const Symbol *, std::string> names;
  for (const parser::ForeignEnumValue *override : pragma.overrides) {
    const Symbol *constructor{nullptr};
    for (const Symbol *c : constructors) {
      if (Names(*override->constructor, *c)) {
        constructor = c;
      }
    }
    if (!constructor) {
      // mmc: "the following symbol is not a constructor of the type: `c'."
      messages_.Say(Source(override->tokens),
          "`%s' is not a constructor of %s"_err_en_US,
          override->constructor->name.str(), TypeText(*type));
      continue;
    }
    names[constructor] =
        pragma.prefix.value_or("").str() + override->value.str();
  }
  // The prefix goes after the default name is made upper case.
  llvm::StringMap<const Symbol *> named;
  for (const Symbol *constructor : constructors) {
    std::string name;
    if (auto it{names.find(constructor)}; it != names.end()) {
      name = it->second;
    } else {
      name = pragma.prefix.value_or("").str() +
          (pragma.uppercase ? llvm::StringRef{constructor->name}.upper()
                            : constructor->name.str());
    }
    if (!named.try_emplace(name, constructor).second) {
      // mmc: "the specified mapping between the names of Mercury
      // constructors and the corresponding foreign names is inconsistent.
      // The following foreign name is repeated: `X'."
      messages_.Say(at,
          "the foreign name `%s' is given to more than one constructor of %s"_err_en_US,
          name, TypeText(*type));
    }
  }
  Todo(at, "pragma foreign_export_enum");
}

// A foreign_import_module makes another module's foreign_export
// declarations visible to the module's foreign code (reference manual,
// "Declaring Mercury exports to other modules"), with its shared
// foreign_decls, as mmc's header of the module, its .mh file, has them.
// Code generation includes the foreign_decls of a module the program loads;
// foreign_exports are not yet implemented, nor is loading another module for
// it alone.
void ForeignChecker::CheckForeignImport(
    const parser::ForeignImportModulePragma &pragma) {
  if (pragma.language != parser::ForeignLanguage::C) {
    return;
  }
  std::string name{pragma.module->name.str()};
  if (!pragma.module->qualifiers.empty()) {
    name = llvm::join(pragma.module->qualifiers, ".") + "." + name;
  }
  if (name == resolution_.module().name()) {
    return;
  }
  if (resolution_.Module(name)) {
    for (const ModuleInterface *interface : resolution_.interfaces()) {
      if (interface->name == name && interface->foreignExports) {
        Todo(Source(pragma.tokens),
            "pragma foreign_import_module of a module with foreign_exports");
      }
    }
    return;
  }
  Todo(Source(pragma.tokens),
      "pragma foreign_import_module of a module that is not imported");
}

} // namespace

void CheckForeign(const parser::ParseTree &tree,
    const parser::TokenSequence &tokens, const NameResolution &resolution,
    const ModeChecking &modes, const DeterminismChecking &determinism,
    parser::Messages &messages) {
  ForeignChecker{tokens, resolution, modes, determinism, messages}.Run(tree);
}

} // namespace merkurius::semantics
