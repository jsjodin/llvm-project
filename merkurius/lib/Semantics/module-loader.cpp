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

std::string Directory(const parser::SourceFile &file) {
  llvm::StringRef parent{llvm::sys::path::parent_path(file.path())};
  return parent.empty() ? "." : parent.str();
}

} // namespace

ModuleLoader::ModuleLoader(parser::AllSources &sources,
    std::string libraryDirectory, parser::Messages &messages, bool sound,
    parser::ReaderOptions readerOptions)
    : sources_{sources}, libraryDirectory_{std::move(libraryDirectory)},
      messages_{messages}, sound_{sound}, readerOptions_{readerOptions} {}

ModuleLoader::~ModuleLoader() = default;

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
  if (!libraryDirectory_.empty()) {
    if (sound_) {
      llvm::SmallString<256> sound{libraryDirectory_};
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
  loaded_[name] = &interface;
  order_.push_back(&interface);
  file.tokens = parser::Lex(*source, parser::LexerOptions{}, messages_);
  // Process declarations' predicates are declared in the interface; their
  // clauses are the module's own.
  const parser::ParseTree &tree{
      ExpandProcesses(parser::ConvertFile(file.tokens, file.arena, messages_,
                          parser::TermReader::defaultMaxDepth, readerOptions_),
          file.tokens, file.arena, messages_, false)};
  // The file must be the module's, and its interface section is the items
  // from :- interface to :- implementation.
  bool inInterface{false}, named{false};
  for (const parser::Item &item : tree.items) {
    if (const auto *module{std::get_if<const parser::ModuleDecl *>(&item)}) {
      if (named) {
        break; // a nested submodule, which is not loaded yet
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
    } else if (inInterface &&
        !std::holds_alternative<const parser::ImportDecl *>(item)) {
      interface.items.push_back(item);
    }
  }
  if (!named) {
    messages_.Say(where, "%s does not begin with :- module %s"_err_en_US,
        *found, name.str());
  }
  Imports imports;
  Collect(tree, file.tokens, Directory(*source), true, imports);
  interface.imports = std::move(imports.imports);
  interface.uses = std::move(imports.uses);
  return &interface;
}

ModuleLoader::Imports ModuleLoader::LoadImports(const parser::ParseTree &tree,
    const parser::TokenSequence &tokens, llvm::StringRef directory) {
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
