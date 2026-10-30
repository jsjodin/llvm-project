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

class NameResolution {
public:
  NameResolution();
  ~NameResolution();
  // The module's own symbols.
  const ModuleSymbols &module() const { return *module_; }
  // The module's names, in the order they were resolved: its declarations'
  // and then its clauses'.
  llvm::ArrayRef<ResolvedName> names() const { return names_; }

private:
  friend class Resolver;
  common::Arena arena_;
  std::unique_ptr<ModuleSymbols> primitive_;
  std::unique_ptr<ModuleSymbols> module_;
  llvm::DenseMap<const ModuleInterface *, std::unique_ptr<ModuleSymbols>>
      imported_;
  std::vector<ResolvedName> names_;
};

// Resolves the names of a module, whose clauses are desugared into `form`,
// and of the interfaces `loader` has loaded for it, with LoadImports. Undefined names are
// reported, and resolve to no candidates.
std::unique_ptr<NameResolution> ResolveNames(const parser::ParseTree &,
    const parser::TokenSequence &, const SemanticForm &, ModuleLoader &,
    parser::Messages &);

} // namespace merkurius::semantics
#endif // MERKURIUS_SEMANTICS_RESOLVE_NAMES_H_
