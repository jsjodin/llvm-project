//===-- lib/Semantics/resolve-names.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Name resolution. Each module's symbols are collected from its items
// first, so that a name may be used before its declaration; then every
// name is looked up in the scope of the section it is in:
//
// - A module's implementation sees all its own symbols, and the exported
//   symbols of every module it imports or uses, with builtin among them.
// - Its interface sees only what it exports, and the modules its
//   interface section imports or uses (reference manual, "The module
//   system": entities used in the interface must be imported there).
// - An imported module's declarations are resolved in its own interface's
//   scope.
//
// A name qualified by a module refers to that module's symbols, if the
// module is visible; an unqualified name to those of every module imported
// with import_module, the module itself, and, for a type, the primitive
// types. A module imported with use_module is visible only qualified.

#include "merkurius/Semantics/resolve-names.h"
#include "merkurius/Common/idioms.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include <string>

namespace merkurius::semantics {

using namespace parser::literals;

NameResolution::NameResolution() = default;
NameResolution::~NameResolution() = default;

const ModuleSymbols *NameResolution::Module(llvm::StringRef name) const {
  for (const auto &[module, symbols] : imported_) {
    if (symbols->name() == name) {
      return symbols.get();
    }
  }
  return nullptr;
}

const parser::SymName *NameResolution::FunctorName(const Unify &unify) const {
  if (unify.functor.kind == FunctorKind::Name) {
    return unify.functor.name;
  }
  auto it{functors_.find(&unify)};
  return it == functors_.end() ? nullptr : it->second;
}

namespace {

// The primitive types (reference manual, "Primitive types"); mmc also
// accepts character, the name the library's char module defines char as.
constexpr const char *primitiveTypes[]{"int", "int8", "int16", "int32",
    "int64", "uint", "uint8", "uint16", "uint32", "uint64", "float", "char",
    "character", "string"};

std::string FullName(const parser::SymName &name) {
  std::string result{llvm::join(name.qualifiers, ".")};
  if (!result.empty()) {
    result += '.';
  }
  return result + name.name.str();
}

// A module visible in a scope.
struct Visible {
  const ModuleSymbols *symbols;
  bool qualifiedOnly; // imported with use_module
  bool exportedOnly; // an imported module, or the module in its interface
};
using Visibility = std::vector<Visible>;

bool Names(const Visible &visible, llvm::StringRef qualifier) {
  llvm::StringRef module{visible.symbols->name()};
  return module == qualifier ||
      (module.ends_with(qualifier) &&
          module[module.size() - qualifier.size() - 1] == '.');
}

const char *Describe(NameUse use) {
  switch (use) {
  case NameUse::Type:
    return "type";
  case NameUse::Inst:
    return "inst";
  case NameUse::Mode:
    return "mode";
  case NameUse::Typeclass:
    return "type class";
  case NameUse::Call:
    return "predicate";
  case NameUse::DataTerm:
    return "symbol";
  case NameUse::Definition:
    return "predicate or function";
  }
  return "name";
}

} // namespace

class Resolver {
public:
  Resolver(NameResolution &result, ModuleLoader &loader,
      parser::Messages &messages)
      : result_{result}, loader_{loader}, messages_{messages} {}

  void Run(const parser::ParseTree &, const parser::TokenSequence &,
      const SemanticForm &);

private:
  // Collects a module's symbols from its items.
  void Declare(ModuleSymbols &, const parser::Item &, bool exported,
      const parser::TokenSequence &);
  // Declares the predicates and functions whose arity their `with_type`
  // gives, in the module's scope.
  void DeclareWithTypes(ModuleSymbols &, const Visibility &);
  // Whether a declaration of the module's own repeats one of its earlier
  // ones, which it reports, as mmc does: a predicate or function declared
  // twice, a method among them, and a type, inst or mode defined twice. A
  // type's abstract declaration, or a foreign type, goes with its
  // definition.
  bool Duplicate(const ModuleSymbols &, SymbolKind, llvm::StringRef name,
      unsigned arity, Declaration, parser::CharBlock source,
      const Symbol *owner = nullptr);
  std::optional<unsigned> WithTypeArity(
      const parser::Type &, const Visibility &, int depth);
  struct Pending {
    ModuleSymbols *symbols;
    const parser::PredDecl *decl;
    bool exported;
    const Symbol *owner;
    const parser::TokenSequence *tokens;
  };
  std::vector<Pending> pending_;
  // The scope of an imported module's interface.
  Visibility InterfaceScope(const ModuleInterface &);
  ModuleSymbols &Symbols(const ModuleInterface &);

  // Resolves the names in an item.
  void ResolveItem(const parser::Item &, const Visibility &, const Visibility *outer,
      bool local, const parser::TokenSequence &);
  void ResolveClause(const ClauseForm &, const Visibility &);
  // A method clause's name is its type class's method, if it has one.
  void ResolveMethodClause(const MethodClause &, const Visibility &);
  void ResolveGoal(const Goal &, const Visibility &);
  // Resolves the type, inst, mode and class names below `x`.
  template <typename T>
  void ResolveBelow(
      const T &x, const Visibility &scope, const Visibility *outer = nullptr);

  // Looks the name up, records what it finds, and reports it if it finds
  // nothing. `outer` is the implementation's scope when `scope` is the
  // interface's, for a better message.
  void Resolve(const parser::SymName &, NameUse, unsigned arity,
      const Visibility &scope, const Visibility *outer = nullptr);
  llvm::SmallVector<const Symbol *, 4> Lookup(const parser::SymName &,
      NameUse, unsigned arity, const Visibility &, bool anyArity = false,
      bool qualifiedOnly = false) const;
  static bool Accepts(const Symbol &, NameUse, unsigned arity, bool anyArity);
  void Undefined(const parser::SymName &, NameUse, unsigned arity,
      const Visibility &, const Visibility *outer);

  parser::CharBlock Source(const parser::SymName &name) const {
    return {tokens_->tokens[name.tokens.first].spelling.begin(),
        tokens_->tokens[name.tokens.last].spelling.end()};
  }

  NameResolution &result_;
  ModuleLoader &loader_;
  parser::Messages &messages_;
  const parser::TokenSequence *tokens_{nullptr}; // of the names resolved
  bool local_{false}; // resolving the module's own names
  const ModuleSymbols *own_{nullptr}; // the module whose names these are
};

void Resolver::Declare(ModuleSymbols &symbols, const parser::Item &item,
    bool exported, const parser::TokenSequence &tokens) {
  auto source{[&](const parser::SymName &name) {
    return parser::CharBlock{tokens.tokens[name.tokens.first].spelling.begin(),
        tokens.tokens[name.tokens.last].spelling.end()};
  }};
  auto type{[&](const parser::SymName &name, std::size_t arity,
                Declaration declaration) -> const Symbol * {
    if (Duplicate(symbols, SymbolKind::Type, name.name, arity, declaration,
            source(name))) {
      return nullptr; // its constructors are not the type's
    }
    return &symbols.Add(SymbolKind::Type, name.name, arity, declaration,
        source(name), exported);
  }};
  auto predicate{[&](const parser::PredDecl &decl, const Symbol *owner) {
    if (decl.withType) {
      // Its arity depends on its type's, which is known once the module's
      // types are.
      pending_.push_back({&symbols, &decl, exported, owner, &tokens});
      return;
    }
    SymbolKind kind{decl.kind == parser::PredOrFunc::Function
            ? SymbolKind::Function
            : SymbolKind::Predicate};
    if (Duplicate(symbols, kind, decl.name->name, decl.arguments.size(),
            &decl, source(*decl.name), owner)) {
      return;
    }
    symbols.Add(decl.kind == parser::PredOrFunc::Function
            ? SymbolKind::Function
            : SymbolKind::Predicate,
        decl.name->name, decl.arguments.size(), &decl, source(*decl.name),
        exported, owner);
  }};
  std::visit(
      common::visitors{
          [&](const parser::DuTypeDefn *defn) {
            const Symbol *owned{
                type(*defn->name, defn->parameters.size(), defn)};
            if (!owned) {
              return;
            }
            const Symbol &owner{*owned};
            for (const parser::Constructor *constructor : defn->constructors) {
              symbols.Add(SymbolKind::Constructor, constructor->name->name,
                  constructor->arguments.size(), constructor,
                  source(*constructor->name), exported, &owner);
              for (const parser::ConstructorArg *argument :
                  constructor->arguments) {
                if (argument->field) {
                  symbols.Add(SymbolKind::Field, argument->field->name, 1,
                      constructor, source(*argument->field), exported,
                      &owner);
                  symbols.Add(SymbolKind::FieldUpdate,
                      (argument->field->name + " :=").str(), 2, constructor,
                      source(*argument->field), exported, &owner);
                }
              }
            }
          },
          [&](const parser::EquivalenceTypeDefn *defn) {
            type(*defn->name, defn->parameters.size(), defn);
          },
          [&](const parser::AbstractTypeDecl *decl) {
            type(*decl->name, decl->parameters.size(), decl);
          },
          [&](const parser::ForeignTypePragma *pragma) {
            type(*pragma->name, pragma->parameters.size(), pragma);
          },
          [&](const parser::PredDecl *decl) { predicate(*decl, nullptr); },
          [&](const parser::InstDefn *defn) {
            if (!Duplicate(symbols, SymbolKind::Inst, defn->name->name,
                    defn->parameters.size(), defn, source(*defn->name))) {
              symbols.Add(SymbolKind::Inst, defn->name->name,
                  defn->parameters.size(), defn, source(*defn->name),
                  exported);
            }
          },
          [&](const parser::ModeDefn *defn) {
            if (!Duplicate(symbols, SymbolKind::Mode, defn->name->name,
                    defn->parameters.size(), defn, source(*defn->name))) {
              symbols.Add(SymbolKind::Mode, defn->name->name,
                  defn->parameters.size(), defn, source(*defn->name),
                  exported);
            }
          },
          [&](const parser::TypeclassDecl *decl) {
            const Symbol &owner{symbols.Add(SymbolKind::Typeclass,
                decl->name->name, decl->parameters.size(), decl,
                source(*decl->name), exported)};
            for (const parser::ClassMethod &method : decl->methods) {
              if (const auto *pred{
                      std::get_if<const parser::PredDecl *>(&method)}) {
                predicate(**pred, &owner);
              }
            }
          },
          [](const auto *) {},
      },
      item);
}

bool Resolver::Duplicate(const ModuleSymbols &symbols, SymbolKind kind,
    llvm::StringRef name, unsigned arity, Declaration declaration,
    parser::CharBlock source, const Symbol *owner) {
  if (&symbols != own_) {
    return false; // an imported module's are its own business
  }
  auto definition{[](const Declaration &d) {
    return std::holds_alternative<const parser::DuTypeDefn *>(d) ||
        std::holds_alternative<const parser::EquivalenceTypeDefn *>(d);
  }};
  for (const Symbol *symbol : symbols.Lookup(name)) {
    if (symbol->kind != kind || symbol->arity != arity) {
      continue;
    }
    std::string text;
    llvm::raw_string_ostream os{text};
    os << *symbol;
    // The later in the source is the duplicate: a declaration with
    // `with_type` is declared after the others.
    parser::CharBlock first{symbol->source}, second{source};
    if (second.begin() < first.begin()) {
      std::swap(first, second);
    }
    switch (kind) {
    case SymbolKind::Predicate:
    case SymbolKind::Function:
      if (!std::holds_alternative<const parser::PredDecl *>(
              symbol->declaration)) {
        continue;
      }
      if (owner && symbol->owner == owner) {
        return true; // a type class defined twice, which is reported
      }
      // mmc: "pred `m.p'/1 multiply defined".
      messages_
          .Say(second, "%s is declared more than once"_err_en_US, os.str())
          .Attach(first, "it is first declared here"_because_en_US);
      return true;
    case SymbolKind::Type:
      if (definition(declaration) && definition(symbol->declaration)) {
        bool equivalence{std::holds_alternative<
            const parser::EquivalenceTypeDefn *>(declaration)};
        if (equivalence !=
            std::holds_alternative<const parser::EquivalenceTypeDefn *>(
                symbol->declaration)) {
          // mmc: "this definition of `t'/0 is incompatible with the
          // equivalence type definition", at the other.
          parser::CharBlock union_{equivalence ? symbol->source : source};
          parser::CharBlock other{equivalence ? source : symbol->source};
          messages_
              .Say(union_,
                  "%s is defined both as a discriminated union type and as an equivalence type"_err_en_US,
                  os.str())
              .Attach(other, "the equivalence is defined here"_because_en_US);
        } else {
          // mmc: "duplicate discriminated union type definition".
          messages_
              .Say(second, "%s is defined more than once"_err_en_US, os.str())
              .Attach(first, "it is first defined here"_because_en_US);
        }
        return true;
      }
      if (std::holds_alternative<const parser::AbstractTypeDecl *>(
              declaration) &&
          std::holds_alternative<const parser::AbstractTypeDecl *>(
              symbol->declaration)) {
        // mmc: "duplicate abstract type declaration".
        messages_
            .Say(second, "%s is declared abstractly more than once"_warn_en_US,
                os.str())
            .Attach(first, "it is first declared here"_because_en_US);
        return true;
      }
      return false;
    case SymbolKind::Inst:
    case SymbolKind::Mode:
      // mmc: "duplicate inst definition", "duplicate mode definition".
      messages_.Say(second, "%s is defined more than once"_err_en_US, os.str())
          .Attach(first, "it is first defined here"_because_en_US);
      return true;
    default:
      return false;
    }
  }
  return false;
}

ModuleSymbols &Resolver::Symbols(const ModuleInterface &module) {
  if (auto it{result_.imported_.find(&module)};
      it != result_.imported_.end()) {
    return *it->second;
  }
  auto owned{std::make_unique<ModuleSymbols>(module.name, module.tokens)};
  ModuleSymbols &symbols{*owned};
  result_.imported_[&module] = std::move(owned);
  for (const parser::Item &item : module.items) {
    Declare(symbols, item, true, *module.tokens);
  }
  DeclareWithTypes(symbols, InterfaceScope(module));
  return symbols;
}

void Resolver::DeclareWithTypes(
    ModuleSymbols &symbols, const Visibility &scope) {
  std::vector<Pending> rest;
  for (const Pending &p : pending_) {
    if (p.symbols != &symbols) {
      rest.push_back(p);
      continue;
    }
    const parser::PredDecl &decl{*p.decl};
    unsigned arity(decl.arguments.size());
    std::optional<unsigned> added{WithTypeArity(*decl.withType, scope, 0)};
    if (!added) {
      messages_.Say(parser::CharBlock{
                        p.tokens->tokens[decl.tokens.first].spelling.begin(),
                        p.tokens->tokens[decl.tokens.last].spelling.end()},
          "the type of %s must be a %s type"_err_en_US, decl.name->name.str(),
          decl.kind == parser::PredOrFunc::Function ? "func" : "pred");
    }
    SymbolKind kind{decl.kind == parser::PredOrFunc::Function
            ? SymbolKind::Function
            : SymbolKind::Predicate};
    parser::CharBlock source{
        p.tokens->tokens[decl.name->tokens.first].spelling.begin(),
        p.tokens->tokens[decl.name->tokens.last].spelling.end()};
    if (Duplicate(symbols, kind, decl.name->name, arity + added.value_or(0),
            &decl, source, p.owner)) {
      continue;
    }
    symbols.Add(kind, decl.name->name, arity + added.value_or(0), &decl,
        source, p.exported, p.owner);
  }
  pending_ = std::move(rest);
}

// The number of arguments of the higher-order type `type` is, through
// equivalences, or nothing if it is not one.
std::optional<unsigned> Resolver::WithTypeArity(
    const parser::Type &type, const Visibility &scope, int depth) {
  if (const auto *higher{std::get_if<const parser::HigherOrderType *>(&type)}) {
    return (*higher)->arguments.size();
  }
  const auto *named{std::get_if<const parser::NamedType *>(&type)};
  if (!named || depth > 50) {
    return std::nullopt;
  }
  for (const Symbol *symbol : Lookup(*(*named)->name, NameUse::Type,
           (*named)->arguments.size(), scope)) {
    const auto *defn{
        std::get_if<const parser::EquivalenceTypeDefn *>(&symbol->declaration)};
    if (!defn) {
      continue;
    }
    // The equivalence's names are in its own module's scope.
    for (const ModuleInterface *module : loader_.modules()) {
      if (module->name == symbol->module) {
        return WithTypeArity((*defn)->type, InterfaceScope(*module), depth + 1);
      }
    }
    return WithTypeArity((*defn)->type, scope, depth + 1);
  }
  return std::nullopt;
}

Visibility Resolver::InterfaceScope(const ModuleInterface &module) {
  Visibility scope{{&Symbols(module), false, true}};
  for (const ModuleInterface *imported : module.imports) {
    scope.push_back({&Symbols(*imported), false, true});
  }
  for (const ModuleInterface *used : module.uses) {
    scope.push_back({&Symbols(*used), true, true});
  }
  return scope;
}

bool Resolver::Accepts(
    const Symbol &symbol, NameUse use, unsigned arity, bool anyArity) {
  auto is{[&](SymbolKind kind) {
    return symbol.kind == kind && (anyArity || symbol.arity == arity);
  }};
  switch (use) {
  case NameUse::Type:
    return is(SymbolKind::Type);
  case NameUse::Inst:
    return is(SymbolKind::Inst);
  case NameUse::Mode:
    return is(SymbolKind::Mode);
  case NameUse::Typeclass:
    return is(SymbolKind::Typeclass);
  case NameUse::Call:
    return is(SymbolKind::Predicate);
  case NameUse::Definition:
    return is(SymbolKind::Predicate) || is(SymbolKind::Function);
  case NameUse::DataTerm:
    // f(X1, ..., Xn) constructs, calls a function of arity n, selects or
    // updates a field, or makes a closure of a predicate or function with
    // more arguments (reference manual, "Higher-order").
    if (anyArity) {
      return symbol.kind != SymbolKind::Type && symbol.kind != SymbolKind::Inst &&
          symbol.kind != SymbolKind::Mode &&
          symbol.kind != SymbolKind::Typeclass;
    }
    switch (symbol.kind) {
    case SymbolKind::Constructor:
    case SymbolKind::Field:
    case SymbolKind::FieldUpdate:
      return symbol.arity == arity;
    case SymbolKind::Function:
    case SymbolKind::Predicate:
      return symbol.arity >= arity;
    default:
      return false;
    }
  }
  return false;
}

llvm::SmallVector<const Symbol *, 4> Resolver::Lookup(
    const parser::SymName &name, NameUse use, unsigned arity,
    const Visibility &scope, bool anyArity, bool qualifiedOnly) const {
  llvm::SmallVector<const Symbol *, 4> found;
  std::string qualifier{llvm::join(name.qualifiers, ".")};
  auto search{[&](const Visible &visible) {
    for (const Symbol *symbol : visible.symbols->Lookup(name.name)) {
      if ((!visible.exportedOnly || symbol->exported) &&
          Accepts(*symbol, use, arity, anyArity)) {
        found.push_back(symbol);
      }
    }
  }};
  for (const Visible &visible : scope) {
    if (!qualifier.empty()) {
      if (Names(visible, qualifier)) {
        search(visible);
      }
    } else if (visible.qualifiedOnly == qualifiedOnly) {
      search(visible);
    }
  }
  if (qualifier.empty() && !qualifiedOnly && use == NameUse::Type) {
    search(Visible{result_.primitive_.get(), false, false});
  }
  return found;
}

void Resolver::Resolve(const parser::SymName &name, NameUse use,
    unsigned arity, const Visibility &scope, const Visibility *outer) {
  llvm::SmallVector<const Symbol *, 4> found{Lookup(name, use, arity, scope)};
  // 'new f'(...) constructs with an existentially typed constructor f/n
  // (reference manual, "Existentially typed data types"), which may be
  // qualified, as m.'new f'.
  const Symbol *plain{nullptr};
  if (found.empty() && use == NameUse::DataTerm && name.name.starts_with("new ")) {
    parser::SymName constructor{name};
    constructor.name = name.name.drop_front(4);
    for (const Symbol *symbol : Lookup(constructor, use, arity, scope)) {
      if (symbol->kind != SymbolKind::Constructor) {
        continue;
      }
      const auto *decl{
          std::get_if<const parser::Constructor *>(&symbol->declaration)};
      if (decl && !(*decl)->existentialVars.empty()) {
        found.push_back(symbol);
      } else {
        plain = symbol;
      }
    }
  }
  if (found.empty() && !(use == NameUse::DataTerm && IsCharConstant(name, arity))) {
    if (plain && plain->owner) {
      // mmc: "undefined symbol `new plain'/1. Invalid use of `new' on a
      // constructor of type `m.plain'/0 which is not existentially typed."
      messages_
          .Say(Source(name), "undefined %s %s/%u"_err_en_US, Describe(use),
              FullName(name), arity)
          .Attach(Source(name),
              "`new' is only for existentially typed constructors, and %s/%u of type %s.%s/%u is not one"_because_en_US,
              plain->name.str(), arity, plain->owner->module.str(),
              plain->owner->name.str(), plain->owner->arity);
    } else {
      Undefined(name, use, arity, scope, outer);
    }
  }
  llvm::ArrayRef<const Symbol *> candidates{
      result_.arena_.SaveArray(llvm::ArrayRef<const Symbol *>{found})};
  name.candidates = candidates;
  if (local_) {
    result_.names_.push_back(ResolvedName{&name, use, arity, candidates});
  }
}

bool IsCharConstant(const parser::SymName &name, unsigned arity) {
  // One code point: one byte that does not continue a UTF-8 sequence.
  return arity == 0 && name.qualifiers.empty() && !name.name.empty() &&
      llvm::count_if(name.name, [](char c) {
        return (static_cast<unsigned char>(c) & 0xc0) != 0x80;
      }) == 1;
}

void Resolver::Undefined(const parser::SymName &name, NameUse use,
    unsigned arity, const Visibility &scope, const Visibility *outer) {
  // mmc: "undefined predicate `p'/1", "undefined symbol `f'/1", "undefined
  // type `t'/0".
  parser::Message &message{messages_.Say(Source(name),
      "undefined %s %s/%u"_err_en_US, Describe(use), FullName(name), arity)};
  std::string qualifier{llvm::join(name.qualifiers, ".")};
  if (!qualifier.empty() &&
      llvm::none_of(scope, [&](const Visible &v) { return Names(v, qualifier); })) {
    bool elsewhere{outer &&
        llvm::any_of(*outer, [&](const Visible &v) { return Names(v, qualifier); })};
    message.Attach(Source(name),
        elsewhere ? "the module %s has not been imported in the interface section"_because_en_US
                  : "the module %s has not been imported"_because_en_US,
        qualifier);
    return;
  }
  if (qualifier.empty() && (name.name == "[|]" || name.name == "[]")) {
    message.Attach(Source(name),
        "the module list has not been imported"_because_en_US);
    return;
  }
  if (qualifier.empty()) {
    auto used{Lookup(name, use, arity, scope, false, true)};
    if (!used.empty()) {
      // mmc: "That symbol is defined in module `string', which does not
      // have an `:- import_module' declaration."
      message.Attach(Source(name),
          "%s is in module %s, which is imported with use_module, so it must be written %s.%s"_because_en_US,
          name.name.str(), used.front()->module.str(),
          used.front()->module.str(), name.name.str());
      return;
    }
  }
  if (outer) {
    if (auto hidden{Lookup(name, use, arity, *outer)}; !hidden.empty()) {
      // mmc: "the type `foo'/0 is not exported, and thus it may not be
      // used in the interface", "(The module `list' has not been imported
      // in the interface.)"
      if (hidden.front()->module == own_->name()) {
        message.Attach(Source(name),
            "%s/%u is not exported, so it may not be used in the interface section"_because_en_US,
            name.name.str(), arity);
      } else {
        message.Attach(Source(name),
            "the module %s has not been imported in the interface section"_because_en_US,
            hidden.front()->module.str());
      }
      return;
    }
  }
  auto others{Lookup(name, use, arity, scope, true)};
  if (!others.empty()) {
    std::string list;
    llvm::raw_string_ostream os{list};
    llvm::interleave(
        others, os, [&](const Symbol *symbol) { os << *symbol; }, ", ");
    message.Attach(Source(name), "there is %s"_because_en_US, list);
  }
}

template <typename T>
void Resolver::ResolveBelow(const T &x, const Visibility &scope, const Visibility *outer) {
  struct Visitor {
    Resolver &resolver;
    const Visibility &scope;
    const Visibility *outer;
    bool Pre(const parser::FactOrRule &) { return false; }
    bool Pre(const parser::DcgRule &) { return false; }
    void Post(const parser::NamedType &type) {
      resolver.Resolve(
          *type.name, NameUse::Type, type.arguments.size(), scope, outer);
    }
    void Post(const parser::NamedInst &inst) {
      resolver.Resolve(
          *inst.name, NameUse::Inst, inst.arguments.size(), scope, outer);
    }
    void Post(const parser::NamedMode &mode) {
      resolver.Resolve(
          *mode.name, NameUse::Mode, mode.arguments.size(), scope, outer);
    }
    void Post(const parser::ClassConstraint &constraint) {
      resolver.Resolve(*constraint.name, NameUse::Typeclass,
          constraint.arguments.size(), scope, outer);
    }
  } visitor{*this, scope, outer};
  parser::Walk(x, visitor);
}

void Resolver::ResolveItem(const parser::Item &item, const Visibility &scope,
    const Visibility *outer, bool local, const parser::TokenSequence &tokens) {
  tokens_ = &tokens;
  local_ = local;
  std::visit(
      common::visitors{
          [&](const parser::ModeDecl *decl) {
            // A mode declaration is for a predicate or function of the
            // module's own.
            Visibility own{{own_, false, false}};
            Resolve(*decl->name, NameUse::Definition, decl->arguments.size(),
                own);
            ResolveBelow(*decl, scope, outer);
          },
          [&](const parser::InstDefn *defn) {
            if (defn->forType) {
              Resolve(*defn->forType->name, NameUse::Type,
                  defn->forType->arity, scope, outer);
            }
            ResolveBelow(*defn, scope, outer);
          },
          [&](const parser::InstanceDecl *decl) {
            Resolve(*decl->name, NameUse::Typeclass, decl->types.size(), scope,
                outer);
            ResolveBelow(*decl, scope, outer);
          },
          [&](const parser::FactOrRule *) {},
          [&](const parser::DcgRule *) {},
          [&](const parser::ImportDecl *) {},
          [&](const auto *x) { ResolveBelow(*x, scope, outer); },
      },
      item);
}

void Resolver::ResolveClause(const ClauseForm &clause, const Visibility &scope) {
  unsigned arity(clause.head.size());
  if (clause.kind == parser::PredOrFunc::Function) {
    --arity;
  }
  Visibility own{{own_, false, false}};
  Resolve(*clause.name, NameUse::Definition, arity, own);
  for (const TypeQualification &qualification : clause.typeQualifications) {
    ResolveBelow(qualification.type, scope);
  }
  ResolveGoal(clause.body, scope);
}

void Resolver::ResolveMethodClause(
    const MethodClause &method, const Visibility &scope) {
  const ClauseForm &clause{*method.clause};
  unsigned arity(clause.head.size());
  SymbolKind kind{SymbolKind::Predicate};
  if (clause.kind == parser::PredOrFunc::Function) {
    --arity;
    kind = SymbolKind::Function;
  }
  // Type checking reports a method that the type class does not have.
  llvm::SmallVector<const Symbol *, 1> found;
  for (const Symbol *typeclass : method.instance->name->candidates) {
    const ModuleSymbols *symbols{typeclass->module == own_->name()
            ? own_
            : result_.Module(typeclass->module)};
    if (!symbols) {
      continue;
    }
    for (const Symbol *symbol : symbols->Lookup(clause.name->name)) {
      if (symbol->owner == typeclass && symbol->kind == kind &&
          symbol->arity == arity) {
        found.push_back(symbol);
      }
    }
  }
  llvm::ArrayRef<const Symbol *> candidates{
      result_.arena_.SaveArray(llvm::ArrayRef<const Symbol *>{found})};
  clause.name->candidates = candidates;
  result_.names_.push_back(
      ResolvedName{clause.name, NameUse::Definition, arity, candidates});
  if (found.empty()) {
    return; // not checked, so neither are its names
  }
  for (const TypeQualification &qualification : clause.typeQualifications) {
    ResolveBelow(qualification.type, scope);
  }
  ResolveGoal(clause.body, scope);
}

void Resolver::ResolveGoal(const Goal &goal, const Visibility &scope) {
  auto each{[&](llvm::ArrayRef<Goal> goals) {
    for (const Goal &g : goals) {
      ResolveGoal(g, scope);
    }
  }};
  std::visit(
      common::visitors{
          [&](const Conj *conj) { each(conj->goals); },
          [&](const Disj *disj) { each(disj->goals); },
          [&](const IfThenElse *ite) {
            ResolveGoal(ite->condition, scope);
            ResolveGoal(ite->then, scope);
            ResolveGoal(ite->otherwise, scope);
          },
          [&](const Not *negation) { ResolveGoal(negation->goal, scope); },
          [&](const Unify *unify) {
            if (unify->isVar) {
              return;
            }
            if (unify->functor.kind == FunctorKind::Name) {
              Resolve(*unify->functor.name, NameUse::DataTerm,
                  unify->arguments.size(), scope);
            } else if (unify->functor.kind == FunctorKind::Cons ||
                unify->functor.kind == FunctorKind::Nil) {
              // List syntax names the list module's constructors, which
              // must be visible (mmc: "undefined symbol `[|]'/2").
              bool cons{unify->functor.kind == FunctorKind::Cons};
              const parser::SymName &name{*result_.arena_.Make<parser::SymName>(
                  parser::SymName{unify->tokens, {}, cons ? "[|]" : "[]"})};
              result_.functors_[unify] = &name;
              Resolve(name, NameUse::DataTerm, cons ? 2 : 0, scope);
            } else if (const Lambda * lambda{unify->functor.lambda}) {
              for (const std::optional<parser::Mode> &mode : lambda->modes) {
                if (mode) {
                  ResolveBelow(*mode, scope);
                }
              }
              ResolveGoal(lambda->body, scope);
            }
          },
          [&](const Call *call) {
            // An event is not a predicate.
            if (!call->event) {
              Resolve(*call->name, NameUse::Call, call->arguments.size(), scope);
            }
          },
          [&](const HOCall *) {},
          [&](const Scope *s) { ResolveGoal(s->goal, scope); },
          [&](const Try *t) {
            ResolveGoal(t->goal, scope);
            ResolveGoal(t->then, scope);
            if (t->otherwise) {
              ResolveGoal(*t->otherwise, scope);
            }
            for (const Catch &c : t->catches) {
              ResolveGoal(c.pattern, scope);
              ResolveGoal(c.goal, scope);
            }
            if (t->catchAny) {
              ResolveGoal(t->catchAny->goal, scope);
            }
          },
          [&](const Atomic *atomic) {
            for (const AtomicAlternative &alternative : atomic->alternatives) {
              ResolveGoal(alternative.goal, scope);
            }
          },
      },
      goal);
}

void Resolver::Run(const parser::ParseTree &tree,
    const parser::TokenSequence &tokens, const SemanticForm &form) {
  result_.primitive_ = std::make_unique<ModuleSymbols>("");
  for (const char *name : primitiveTypes) {
    result_.primitive_->Add(SymbolKind::Type, name, 0, std::monostate{},
        parser::CharBlock{}, true);
  }
  // The module's own symbols, and the modules each section imports.
  std::string name;
  for (const parser::Item &item : tree.items) {
    if (const auto *module{std::get_if<const parser::ModuleDecl *>(&item)}) {
      name = FullName(*(*module)->name);
      break;
    }
  }
  result_.module_ = std::make_unique<ModuleSymbols>(name, &tokens);
  ModuleSymbols &own{*result_.module_};
  own_ = &own;
  Visibility interface{{&own, false, true}}, implementation{{&own, false, false}};
  bool inInterface{false};
  auto add{[&](Visibility &scope, const Visible &visible) {
    if (llvm::none_of(scope, [&](const Visible &v) {
          return v.symbols == visible.symbols &&
              v.qualifiedOnly == visible.qualifiedOnly;
        })) {
      scope.push_back(visible);
    }
  }};
  for (const parser::Item &item : tree.items) {
    if (const auto *section{std::get_if<const parser::SectionDecl *>(&item)}) {
      inInterface = (*section)->section == parser::Section::Interface;
    } else if (const auto *decl{
                   std::get_if<const parser::ImportDecl *>(&item)}) {
      if ((*decl)->kind == parser::ImportKind::IncludeModule) {
        continue;
      }
      bool use{(*decl)->kind == parser::ImportKind::UseModule};
      for (const parser::SymName *module : (*decl)->modules) {
        if (const ModuleInterface * loaded{loader_.Find(FullName(*module))}) {
          Visible visible{&Symbols(*loaded), use, true};
          add(implementation, visible);
          if (inInterface) {
            add(interface, visible);
          }
        }
      }
    } else {
      Declare(own, item, inInterface, tokens);
    }
  }
  if (name != "builtin") {
    if (const ModuleInterface * builtin{loader_.Find("builtin")}) {
      Visible visible{&Symbols(*builtin), false, true};
      add(implementation, visible);
      add(interface, visible);
    }
  }
  DeclareWithTypes(own, implementation);
  // A clause for a predicate or function without a declaration declares
  // it, local to the module, since types are inferred for it.
  for (const ClauseForm *clause : form.clauses) {
    unsigned arity(clause->head.size());
    SymbolKind kind{SymbolKind::Predicate};
    if (clause->kind == parser::PredOrFunc::Function) {
      --arity;
      kind = SymbolKind::Function;
    }
    if (llvm::none_of(own.Lookup(clause->name->name), [&](const Symbol *s) {
          return s->kind == kind && s->arity == arity;
        })) {
      own.Add(kind, clause->name->name, arity, std::monostate{},
          parser::CharBlock{tokens.tokens[clause->name->tokens.first]
                                .spelling.begin(),
              tokens.tokens[clause->name->tokens.last].spelling.end()},
          false);
    }
  }
  // The imported interfaces' names, each in its own scope.
  for (const ModuleInterface *module : loader_.modules()) {
    Visibility scope{InterfaceScope(*module)};
    own_ = &Symbols(*module);
    for (const parser::Item &item : module->items) {
      ResolveItem(item, scope, nullptr, false, *module->tokens);
      if (const auto *decl{std::get_if<const parser::InstanceDecl *>(&item)}) {
        result_.instances_.push_back({*decl, module->tokens, false, true});
      }
    }
  }
  // The module's own names.
  own_ = &own;
  inInterface = false;
  for (const parser::Item &item : tree.items) {
    if (const auto *section{std::get_if<const parser::SectionDecl *>(&item)}) {
      inInterface = (*section)->section == parser::Section::Interface;
    } else {
      ResolveItem(item, inInterface ? interface : implementation,
          inInterface ? &implementation : nullptr, true, tokens);
      if (const auto *decl{std::get_if<const parser::InstanceDecl *>(&item)}) {
        result_.instances_.push_back({*decl, &tokens, true, inInterface});
      } else if (const auto *decl{
                     std::get_if<const parser::TypeclassDecl *>(&item)}) {
        for (const Symbol *symbol : own.Lookup((*decl)->name->name)) {
          if (symbol->kind == SymbolKind::Typeclass &&
              symbol->arity == (*decl)->parameters.size()) {
            result_.typeclasses_.push_back({*decl, symbol, inInterface});
          }
        }
      }
    }
  }
  tokens_ = &tokens;
  local_ = true;
  for (const ClauseForm *clause : form.clauses) {
    ResolveClause(*clause, implementation);
  }
  for (const MethodClause &method : form.methods) {
    ResolveMethodClause(method, implementation);
  }
}

std::unique_ptr<NameResolution> ResolveNames(const parser::ParseTree &tree,
    const parser::TokenSequence &tokens, const SemanticForm &form,
    ModuleLoader &loader, parser::Messages &messages) {
  auto result{std::make_unique<NameResolution>()};
  Resolver{*result, loader, messages}.Run(tree, tokens, form);
  return result;
}

} // namespace merkurius::semantics
