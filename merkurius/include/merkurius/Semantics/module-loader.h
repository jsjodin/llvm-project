//===-- include/merkurius/Semantics/module-loader.h -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_SEMANTICS_MODULE_LOADER_H_
#define MERKURIUS_SEMANTICS_MODULE_LOADER_H_

// The module loader (design document, "Front end"): reads the interfaces of
// the modules a module imports, as parse-tree declarations, so that
// imported and local declarations look alike to semantic analysis. Until
// there are interface files, an imported module's interface is read from
// its source, which is found by the module's name, as name.m, in the
// importing module's directory and then the search path.

#include "merkurius/Parser/message.h"
#include "merkurius/Parser/parse-tree.h"
#include "merkurius/Parser/provenance.h"
#include "merkurius/Parser/term-reader.h"
#include "merkurius/Parser/token.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include <memory>
#include <string>
#include <vector>

namespace merkurius::semantics {

// A module's interface: the items of its interface section but its
// imports, and the modules that section imports and uses, which are
// loaded too, since its declarations' names are resolved in its own scope.
struct ModuleInterface {
  std::string name;
  const parser::SourceFile *file;
  const parser::TokenSequence *tokens;
  std::vector<parser::Item> items;
  std::vector<const ModuleInterface *> imports; // import_module
  std::vector<const ModuleInterface *> uses; // use_module
  // The predicates and functions that the module's file defines, by a
  // clause or a C foreign_proc, by their DefinitionKey; known unless the
  // file is a sound stub, which declares what another file defines.
  llvm::StringSet<> defined;
  bool definitionsKnown{true};
  // The implementation section's definitions of the types that the
  // interface declares abstractly and that are equivalences. Importing
  // modules see through them after type checking, so that a value of one
  // has the representation and the type_info of the type it is in every
  // module, as mmc's .int files let them (reference manual, "Abstract
  // types"). The names in them are in the scope of the implementation
  // section: the interface's, those of `hiddenImports`, the modules the
  // implementation section imports, loaded only for them, and those of
  // `implementationTypes`, the types the implementation section defines
  // that the interface does not declare, by name/arity.
  std::vector<const parser::EquivalenceTypeDefn *> hidden;
  std::vector<const ModuleInterface *> hiddenImports;
  llvm::StringSet<> implementationTypes;
};

// How ModuleInterface::defined names a predicate or function: its kind,
// name and arity, a function's without its result, as a Symbol's is.
std::string DefinitionKey(bool function, llvm::StringRef name, unsigned arity);
// How ModuleInterface::implementationTypes names a type: name/arity, the
// name unqualified.
std::string TypeKey(const parser::SymName &, unsigned arity);

class ModuleLoader {
public:
  // Messages about the imported files go to `messages`; the library's
  // directory is searched after the search path of `sources`, and with
  // `sound`, its sound/ directory before it, which has the sound stubs of
  // modules whose manual's declarations work around mmc's unique modes.
  // Each file is read with `readerOptions`, the importing compile's, so
  // that an imported module may use the extensions it may, such as loops.
  // The library's module open, of open types, loads only with `openTypes`
  // (-fopen-types).
  ModuleLoader(parser::AllSources &sources, std::string libraryDirectory,
      parser::Messages &messages, bool sound = false,
      parser::ReaderOptions readerOptions = {}, bool openTypes = false);
  ~ModuleLoader();

  // The interface of the module named `name`, loaded once, with the
  // modules its interface imports; null, with an error at `where`, if its
  // source cannot be found or read. `directory` is the importing module's.
  const ModuleInterface *Load(llvm::StringRef name, parser::CharBlock where,
      llvm::StringRef directory);

  // The modules `tree`'s items import and use, and builtin, which every
  // module imports implicitly (reference manual, "The module system").
  struct Imports {
    std::vector<const ModuleInterface *> imports, uses;
  };
  Imports LoadImports(const parser::ParseTree &tree,
      const parser::TokenSequence &tokens, llvm::StringRef directory);

  // The interface of the module named `name` if it has been loaded; null
  // if it has not, or could not be.
  const ModuleInterface *Find(llvm::StringRef name) const {
    auto it{loaded_.find(name)};
    return it == loaded_.end() ? nullptr : it->second;
  }

  // Every module loaded, in the order loading began.
  llvm::ArrayRef<const ModuleInterface *> modules() const { return order_; }

private:
  struct File;
  void Collect(const parser::ParseTree &, const parser::TokenSequence &,
      llvm::StringRef directory, bool interfaceOnly, Imports &);

  parser::AllSources &sources_;
  std::string libraryDirectory_;
  parser::Messages &messages_;
  bool sound_;
  parser::ReaderOptions readerOptions_;
  bool openTypes_;
  std::vector<std::unique_ptr<File>> files_;
  llvm::StringMap<ModuleInterface *> loaded_;
  std::vector<const ModuleInterface *> order_;
};

} // namespace merkurius::semantics
#endif // MERKURIUS_SEMANTICS_MODULE_LOADER_H_
