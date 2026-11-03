//===-- include/merkurius/Semantics/resolve-names.h -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_SEMANTICS_RESOLVE_NAMES_H_
#define MERKURIUS_SEMANTICS_RESOLVE_NAMES_H_

// Name resolution (design document, "Semantic analysis", pass 1): finds
// the declarations each name in a module can refer to, by the reference
// manual's visibility rules ("The module system"), and records them on the
// name. Overloaded names keep every candidate; type checking chooses.

#include "merkurius/Common/arena.h"
#include "merkurius/Common/enum-class.h"
#include "merkurius/Parser/message.h"
#include "merkurius/Parser/parse-tree.h"
#include "merkurius/Parser/token.h"
#include "merkurius/Semantics/module-loader.h"
#include "merkurius/Semantics/semantic-form.h"
#include "merkurius/Semantics/symbols.h"
#include "llvm/ADT/DenseMap.h"
#include <memory>
#include <vector>

namespace merkurius::semantics {

// What a name is used as: a type, inst, mode or type class; the predicate
// of a call; the function symbol of a data-term, which is a constructor, a
// function, a field or a closure of a predicate or function; or the
// predicate or function that a clause or mode declaration is for.
ENUM_CLASS(NameUse, Type, Inst, Mode, Typeclass, Call, DataTerm, Definition)

struct ResolvedName {
  const parser::SymName *name;
  NameUse use;
  unsigned arity;
  llvm::ArrayRef<const Symbol *> candidates; // empty if it is undefined
};

// A typeclass declaration of the module's own: the abstract one in the
// interface and the definition in the implementation are one symbol.
struct TypeclassItem {
  const parser::TypeclassDecl *decl;
  const Symbol *symbol;
  bool exported; // in the interface section
};

// An instance declaration, of the module or of an interface it loaded.
// Its type class is its name's candidate.
struct InstanceItem {
  const parser::InstanceDecl *decl;
  const parser::TokenSequence *tokens; // of the module that declares it
  bool local; // the module's own
  bool exported; // in an interface section
};

class NameResolution {
public:
  NameResolution();
  ~NameResolution();
  // The module's own symbols.
  const ModuleSymbols &module() const { return *module_; }
  // Loop translation adds the predicates of loops to them.
  ModuleSymbols &module() { return *module_; }
  // The primitive types, which belong to no module.
  const ModuleSymbols &primitive() const { return *primitive_; }
  // The symbols of a module imported, directly or not; null if it was not
  // loaded.
  const ModuleSymbols *Module(llvm::StringRef name) const;
  // The module's names, in the order they were resolved: its declarations'
  // and then its clauses'.
  llvm::ArrayRef<ResolvedName> names() const { return names_; }
  // The module's own typeclass declarations, in order.
  llvm::ArrayRef<TypeclassItem> typeclasses() const { return typeclasses_; }
  // The interfaces loaded, in the order loaded, with their names
  // resolved.
  llvm::ArrayRef<const ModuleInterface *> interfaces() const {
    return interfaces_;
  }
  // The instance declarations of the interfaces loaded, and then the
  // module's own, in order.
  llvm::ArrayRef<InstanceItem> instances() const { return instances_; }
  // The name of a unification's functor: as written, or, for list syntax,
  // '[|]' or '[]'; null for any other functor.
  const parser::SymName *FunctorName(const Unify &) const;
  // The list module's '[|]', for a loop's foreach(X, List), whose List is
  // a list of X's type; null for another iterator.
  const parser::SymName *ListName(const LoopIterator &) const;

private:
  friend class Resolver;
  common::Arena arena_;
  std::unique_ptr<ModuleSymbols> primitive_;
  std::unique_ptr<ModuleSymbols> module_;
  llvm::DenseMap<const ModuleInterface *, std::unique_ptr<ModuleSymbols>>
      imported_;
  std::vector<ResolvedName> names_;
  std::vector<const ModuleInterface *> interfaces_;
  std::vector<TypeclassItem> typeclasses_;
  std::vector<InstanceItem> instances_;
  llvm::DenseMap<const Unify *, const parser::SymName *> functors_;
  llvm::DenseMap<const LoopIterator *, const parser::SymName *> lists_;
};

// A name that is a character constant, such as a or 'é' (reference
// manual, "Character type"; mmc accepts p(a) for p(char)).
bool IsCharConstant(const parser::SymName &, unsigned arity);

// Resolves the names of a module, whose clauses are desugared into `form`,
// and of the interfaces `loader` has loaded for it, with LoadImports. Undefined names are
// reported, and resolve to no candidates.
std::unique_ptr<NameResolution> ResolveNames(const parser::ParseTree &,
    const parser::TokenSequence &, const SemanticForm &, ModuleLoader &,
    parser::Messages &);

} // namespace merkurius::semantics
#endif // MERKURIUS_SEMANTICS_RESOLVE_NAMES_H_
