//===-- lib/Semantics/module-loader.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "merkurius/Semantics/module-loader.h"
#include "merkurius/Common/idioms.h"
#include "merkurius/Parser/convert.h"
#include "merkurius/Parser/lexer.h"
#include "merkurius/Parser/source.h"
#include "merkurius/Semantics/desugar.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Path.h"
#include <list>

namespace merkurius::semantics {

using namespace parser::literals;

// An imported module's source, kept for as long as its declarations are.
struct ModuleLoader::File {
  parser::TokenSequence tokens;
  parser::ParseTreeArena arena;
  ModuleInterface interface;
};

namespace {

std::string FullName(const parser::SymName &name) {
  std::string result{llvm::join(name.qualifiers, ".")};
  if (!result.empty()) {
    result += '.';
  }
  return result + name.name.str();
}

parser::CharBlock Source(
    parser::TokenRange range, const parser::TokenSequence &tokens) {
  return {tokens.tokens[range.first].spelling.begin(),
      tokens.tokens[range.last].spelling.end()};
}

} // namespace

std::string SubmoduleName(
    llvm::StringRef parent, const parser::SymName &child) {
  std::string name{FullName(child)};
  if (llvm::StringRef{name}.starts_with((parent + ".").str())) {
    return name;
  }
  // A partly qualified name, b.c for a.b.c, overlaps the parent's last
  // components.
  llvm::StringRef rest{parent};
  while (!rest.empty()) {
    if (llvm::StringRef{name}.starts_with((rest + ".").str()) || name == rest) {
      return (parent.drop_back(rest.size()) + name).str();
    }
    auto dot{rest.find('.')};
    rest = dot == llvm::StringRef::npos ? "" : rest.drop_front(dot + 1);
  }
  return (parent + "." + name).str();
}

void ReportNestedSubmodule(parser::Messages &messages, parser::CharBlock where,
    llvm::StringRef parent, const parser::SymName &child, bool inInterface) {
  std::string name{SubmoduleName(parent, child)};
  // Temporary (nested submodules, docs/INPROGRESS.md): a separate
  // submodule is the same module (reference manual, "Separate
  // submodules").
  messages.Say(where,
      "nested submodules; %s can be a separate submodule, in a file of its own, %s.m, which the %s section of %s includes with `:- include_module %s.`"_todo_en_US,
      name, name, inInterface ? "interface" : "implementation", parent.str(),
      child.name.str());
}

const parser::ParseTree &DropNestedSubmodules(const parser::ParseTree &tree,
    const parser::TokenSequence &tokens, parser::ParseTreeArena &arena,
    parser::Messages &messages) {
  // The modules whose item sequences enclose the item, the innermost last,
  // and the section of each that the item is in.
  std::vector<std::pair<std::string, bool>> enclosing;
  std::optional<std::size_t> first;
  // A submodule declared in one section and defined in another is
  // reported once, where it is declared, as that is where include_module
  // would be (reference manual, "Separate submodules").
  llvm::StringSet<> reported;
  for (std::size_t i{0}; i < tree.items.size(); ++i) {
    const parser::Item &item{tree.items[i]};
    if (const auto *module{std::get_if<const parser::ModuleDecl *>(&item)}) {
      std::string name{FullName(*(*module)->name)};
      if (enclosing.empty()) {
        first = first.value_or(i);
      } else {
        auto &[parent, inInterface]{enclosing.back()};
        name = SubmoduleName(parent, *(*module)->name);
        if (reported.insert(name).second) {
          ReportNestedSubmodule(messages,
              Source((*module)->name->tokens, tokens), parent, *(*module)->name,
              inInterface);
        }
      }
      enclosing.emplace_back(name, false);
    } else if (std::holds_alternative<const parser::EndModuleDecl *>(item)) {
      if (enclosing.size() > 1) {
        enclosing.pop_back();
      }
    } else if (const auto *section{
                   std::get_if<const parser::SectionDecl *>(&item)};
        section && !enclosing.empty()) {
      enclosing.back().second =
          (*section)->section == parser::Section::Interface;
    }
  }
  if (reported.empty()) {
    return tree;
  }
  return *arena.Make<parser::ParseTree>(
      tree.tokens, tree.items.slice(*first, 1));
}

namespace {

std::string Directory(const parser::SourceFile &file) {
  llvm::StringRef parent{llvm::sys::path::parent_path(file.path())};
  return parent.empty() ? "." : parent.str();
}

} // namespace

ModuleLoader::ModuleLoader(parser::AllSources &sources,
    std::string libraryDirectory, parser::Messages &messages, bool sound,
    parser::ReaderOptions readerOptions, bool openTypes)
    : sources_{sources}, libraryDirectory_{std::move(libraryDirectory)},
      messages_{messages}, sound_{sound}, readerOptions_{readerOptions},
      openTypes_{openTypes} {}

ModuleLoader::~ModuleLoader() = default;

// The arity of a clause's head: a state variable !X is two arguments.
static unsigned HeadArity(llvm::ArrayRef<const parser::ModedTerm *> arguments) {
  unsigned arity{0};
  for (const parser::ModedTerm *argument : arguments) {
    const auto *state{std::get_if<const parser::StateVar *>(&argument->term)};
    arity += state && (*state)->kind == parser::StateVarKind::Both ? 2 : 1;
  }
  return arity;
}

// What the item defines, if it is a clause or a C foreign_proc.
static void AddDefinition(
    const parser::Item &item, llvm::StringSet<> &defined) {
  if (const auto *clause{std::get_if<const parser::FactOrRule *>(&item)}) {
    defined.insert(
        DefinitionKey((*clause)->kind == parser::PredOrFunc::Function,
            (*clause)->name->name, HeadArity((*clause)->arguments)));
  } else if (const auto *rule{std::get_if<const parser::DcgRule *>(&item)}) {
    defined.insert(DefinitionKey(
        false, (*rule)->name->name, HeadArity((*rule)->arguments) + 2));
  } else if (const auto *proc{
                 std::get_if<const parser::ForeignProcPragma *>(&item)}) {
    if ((*proc)->language == parser::ForeignLanguage::C) {
      defined.insert(
          DefinitionKey((*proc)->kind == parser::PredOrFunc::Function,
              (*proc)->name->name, (*proc)->arguments.size()));
    }
  } else if (const auto *decl{
                 std::get_if<const parser::ProcessDecl *>(&item)}) {
    // A kind's 'spawn$Kind' and 'spawn_at$Kind', whose clauses the module's own
    // compile adds (ExpandProcesses).
    for (const parser::FactOrRule *spawn :
        {(*decl)->spawnClause, (*decl)->spawnAtClause}) {
      if (spawn) {
        AddDefinition(parser::Item{spawn}, defined);
      }
    }
  }
}

const ModuleInterface *ModuleLoader::Load(
    llvm::StringRef name, parser::CharBlock where, llvm::StringRef directory) {
  if (auto it{loaded_.find(name)}; it != loaded_.end()) {
    return it->second;
  }
  // The importing module's directory first, then the search path, then
  // the library.
  std::list<std::string> path{directory.str()};
  for (const std::string &dir : sources_.searchPath()) {
    path.push_back(dir);
  }
  llvm::SmallString<256> sound;
  if (!libraryDirectory_.empty()) {
    if (sound_) {
      sound = libraryDirectory_;
      llvm::sys::path::append(sound, "sound");
      path.push_back(std::string{sound});
    }
    path.push_back(libraryDirectory_);
  }
  std::string fileName{name.str() + ".m"};
  std::optional<std::string> found{parser::LocateSourceFile(fileName, path)};
  if (!found) {
    messages_.Say(where,
        "cannot find module %s: there is no %s in %s"_err_en_US, name.str(),
        fileName, llvm::join(path, ", "));
    loaded_[name] = nullptr;
    return nullptr;
  }
  // Open types are an extension (design document, "Open terms"); a
  // program's own module named open is not the library's.
  if (name == "open" && !openTypes_ && !libraryDirectory_.empty() &&
      llvm::sys::path::parent_path(*found) == libraryDirectory_) {
    messages_.Say(where,
        "the module open, of open types, needs -fopen-types"_err_en_US);
    loaded_[name] = nullptr;
    return nullptr;
  }
  std::string error;
  llvm::raw_string_ostream errorStream{error};
  parser::SourceFile *source{sources_.OpenPath(*found, errorStream)};
  if (!source) {
    messages_.Say(where, "cannot read module %s: %s"_err_en_US, name.str(),
        error);
    loaded_[name] = nullptr;
    return nullptr;
  }
  auto &file{*files_.emplace_back(std::make_unique<File>())};
  ModuleInterface &interface{file.interface};
  interface.name = name.str();
  interface.file = source;
  interface.tokens = &file.tokens;
  interface.definitionsKnown =
      sound.empty() || llvm::sys::path::parent_path(*found) != sound;
  // solutions' aggregates, which have no clauses: code generation makes
  // their procedures from merk.aggregate (design document, Lowering,
  // "Collecting solutions").
  if (name == "solutions") {
    for (auto [aggregate, arity] : {std::pair{"unsorted_aggregate", 4u},
             {"unsorted_aggregate2", 6u}, {"do_while", 4u}}) {
      interface.defined.insert(DefinitionKey(false, aggregate, arity));
    }
  }
  loaded_[name] = &interface;
  order_.push_back(&interface);
  file.tokens = parser::Lex(*source, parser::LexerOptions{}, messages_);
  // Process declarations' predicates are declared in the interface; their
  // clauses are the module's own.
  const parser::ParseTree &tree{
      ExpandProcesses(parser::ConvertFile(file.tokens, file.arena, messages_,
                          parser::TermReader::defaultMaxDepth, readerOptions_),
          file.tokens, file.arena, messages_, false)};
  interface.tree = &tree;
  // The file must be the module's, and its interface section is the items
  // from :- interface to :- implementation.
  bool inInterface{false}, named{false};
  // The types the interface declares abstractly, and the implementation
  // section's equivalences and other type definitions, by name/arity.
  llvm::StringSet<> abstract;
  std::vector<const parser::EquivalenceTypeDefn *> equivalences;
  std::vector<std::string> implementationTypes;
  for (const parser::Item &item : tree.items) {
    if (const auto *foreign{
            std::get_if<const parser::ForeignDeclPragma *>(&item)};
        foreign && !inInterface && named &&
        (*foreign)->language == parser::ForeignLanguage::C &&
        !(*foreign)->local) {
      interface.sharedForeignDecls.push_back(*foreign);
    }
    if (std::holds_alternative<const parser::ForeignExportPragma *>(item) &&
        named) {
      interface.foreignExports = true;
    }
    if (const auto *decl{std::get_if<const parser::AbstractTypeDecl *>(&item)};
        decl && inInterface) {
      abstract.insert(TypeKey(*(*decl)->name, (*decl)->parameters.size()));
    } else if (const auto *defn{
                   std::get_if<const parser::EquivalenceTypeDefn *>(&item)};
        defn && !inInterface && named) {
      equivalences.push_back(*defn);
      implementationTypes.push_back(
          TypeKey(*(*defn)->name, (*defn)->parameters.size()));
    } else if (const auto *defn{std::get_if<const parser::DuTypeDefn *>(&item)};
        defn && !inInterface && named) {
      implementationTypes.push_back(
          TypeKey(*(*defn)->name, (*defn)->parameters.size()));
    } else if (const auto *decl{
                   std::get_if<const parser::AbstractTypeDecl *>(&item)};
        decl && !inInterface && named) {
      implementationTypes.push_back(
          TypeKey(*(*decl)->name, (*decl)->parameters.size()));
    }
    if (const auto *module{std::get_if<const parser::ModuleDecl *>(&item)}) {
      if (named) {
        // Its items would otherwise be read as the parent's.
        ReportNestedSubmodule(messages_,
            Source((*module)->name->tokens, file.tokens), name,
            *(*module)->name, inInterface);
        break;
      }
      named = true;
      if (FullName(*(*module)->name) != name) {
        messages_.Say(Source((*module)->name->tokens, file.tokens),
            "%s is the source of module %s, not of module %s"_err_en_US,
            *found, FullName(*(*module)->name), name.str());
      }
    } else if (const auto *section{
                   std::get_if<const parser::SectionDecl *>(&item)}) {
      inInterface = (*section)->section == parser::Section::Interface;
    } else if (const auto *decl{
                   std::get_if<const parser::ImportDecl *>(&item)}) {
      if (named && (*decl)->kind == parser::ImportKind::IncludeModule) {
        for (const parser::SymName *child : (*decl)->modules) {
          (inInterface ? interface.interfaceChildren
                       : interface.implementationChildren)
              .insert(SubmoduleName(name, *child));
        }
      }
    } else if (inInterface) {
      interface.items.push_back(item);
    }
    AddDefinition(item, interface.defined);
  }
  if (!named) {
    messages_.Say(where, "%s does not begin with :- module %s"_err_en_US,
        *found, name.str());
  }
  for (const parser::EquivalenceTypeDefn *defn : equivalences) {
    if (abstract.contains(TypeKey(*defn->name, defn->parameters.size()))) {
      interface.hidden.push_back(defn);
    }
  }
  for (const std::string &key : implementationTypes) {
    if (!abstract.contains(key)) {
      interface.implementationTypes.insert(key);
    }
  }
  Imports imports;
  Collect(tree, file.tokens, Directory(*source), true, imports);
  interface.imports = std::move(imports.imports);
  interface.uses = std::move(imports.uses);
  if (!interface.hidden.empty()) {
    Imports all;
    Collect(tree, file.tokens, Directory(*source), false, all);
    for (const auto *list : {&all.imports, &all.uses}) {
      for (const ModuleInterface *module : *list) {
        if (!llvm::is_contained(interface.hiddenImports, module)) {
          interface.hiddenImports.push_back(module);
        }
      }
    }
  }
  // A child's interface may name its ancestors' declarations.
  LoadAncestors(name, where, directory);
  return &interface;
}

std::vector<std::string> AncestorNames(llvm::StringRef name) {
  std::vector<std::string> names;
  for (std::size_t dot{name.find('.')}; dot != llvm::StringRef::npos;
      dot = name.find('.', dot + 1)) {
    names.push_back(name.take_front(dot).str());
  }
  return names;
}

void ModuleLoader::LoadAncestors(
    llvm::StringRef name, parser::CharBlock where, llvm::StringRef directory) {
  for (const std::string &ancestorName : AncestorNames(name)) {
    const ModuleInterface *loaded{Load(ancestorName, where, directory)};
    // The module compiled is its own descendants' ancestor.
    if (!loaded || loaded == &self_ || loaded->ancestor || !loaded->tree) {
      continue;
    }
    auto &ancestor{const_cast<ModuleInterface &>(*loaded)};
    ancestor.ancestor = true;
    bool inInterface{false}, named{false};
    for (const parser::Item &item : ancestor.tree->items) {
      if (std::holds_alternative<const parser::ModuleDecl *>(item)) {
        if (named) {
          break;
        }
        named = true;
      } else if (const auto *section{
                     std::get_if<const parser::SectionDecl *>(&item)}) {
        inInterface = (*section)->section == parser::Section::Interface;
      } else if (named && !inInterface &&
          !std::holds_alternative<const parser::ImportDecl *>(item) &&
          !std::holds_alternative<const parser::EndModuleDecl *>(item)) {
        ancestor.implementationItems.push_back(item);
      }
    }
    Imports imports;
    Collect(*ancestor.tree, *ancestor.tokens, Directory(*ancestor.file), false,
        imports);
    ancestor.implementationImports = std::move(imports.imports);
    ancestor.implementationUses = std::move(imports.uses);
  }
}

std::string TypeKey(const parser::SymName &name, unsigned arity) {
  return (name.name + "/" + llvm::Twine(arity)).str();
}

std::string DefinitionKey(bool function, llvm::StringRef name, unsigned arity) {
  return ((function ? "func " : "pred ") + name + "/" + llvm::Twine(arity))
      .str();
}

ModuleLoader::Imports ModuleLoader::LoadImports(const parser::ParseTree &tree,
    const parser::TokenSequence &tokens, llvm::StringRef directory) {
  for (const parser::Item &item : tree.items) {
    if (const auto *module{std::get_if<const parser::ModuleDecl *>(&item)}) {
      self_.name = FullName(*(*module)->name);
      self_.tokens = &tokens;
      self_.tree = &tree;
      loaded_.try_emplace(self_.name, &self_);
      LoadAncestors(
          self_.name, Source((*module)->name->tokens, tokens), directory);
      break;
    }
  }
  Imports imports;
  Collect(tree, tokens, directory, false, imports);
  return imports;
}

void ModuleLoader::Collect(const parser::ParseTree &tree,
    const parser::TokenSequence &tokens, llvm::StringRef directory,
    bool interfaceOnly, Imports &imports) {
  bool inInterface{false}, named{false};
  for (const parser::Item &item : tree.items) {
    if (std::holds_alternative<const parser::ModuleDecl *>(item)) {
      if (named) {
        break;
      }
      named = true;
    } else if (const auto *section{
                   std::get_if<const parser::SectionDecl *>(&item)}) {
      inInterface = (*section)->section == parser::Section::Interface;
    } else if (const auto *decl{
                   std::get_if<const parser::ImportDecl *>(&item)}) {
      if ((interfaceOnly && !inInterface) ||
          (*decl)->kind == parser::ImportKind::IncludeModule) {
        continue;
      }
      for (const parser::SymName *module : (*decl)->modules) {
        if (const ModuleInterface *
            loaded{Load(FullName(*module), Source(module->tokens, tokens),
                directory)}) {
          auto &list{(*decl)->kind == parser::ImportKind::ImportModule
                  ? imports.imports
                  : imports.uses};
          if (!llvm::is_contained(list, loaded)) {
            list.push_back(loaded);
          }
        }
      }
    }
  }
  // Every module but builtin imports builtin; if it is missing, the error
  // is at the start of the first module that needs it.
  bool isBuiltin{false};
  for (const parser::Item &item : tree.items) {
    if (const auto *module{std::get_if<const parser::ModuleDecl *>(&item)}) {
      isBuiltin = FullName(*(*module)->name) == "builtin";
      break;
    }
  }
  if (!isBuiltin) {
    if (const ModuleInterface *
        builtin{Load("builtin", tokens.tokens.front().spelling, directory)}) {
      if (!llvm::is_contained(imports.imports, builtin)) {
        imports.imports.push_back(builtin);
      }
    }
  }
}

} // namespace merkurius::semantics
