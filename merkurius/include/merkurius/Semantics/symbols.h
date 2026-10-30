//===-- include/merkurius/Semantics/symbols.h -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_SEMANTICS_SYMBOLS_H_
#define MERKURIUS_SEMANTICS_SYMBOLS_H_

// The entities that names resolve to (design document, "Semantic
// analysis"): the types, constructors, fields, predicates, functions,
// insts, modes and type classes a module declares, each identified by its
// module, name and arity.

#include "merkurius/Common/enum-class.h"
#include "merkurius/Parser/char-block.h"
#include "merkurius/Parser/parse-tree.h"
#include "merkurius/Parser/token.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <variant>
#include <vector>

namespace merkurius::semantics {

// A field of a constructor is two functions: Field selects it, f/1, and
// FieldUpdate updates it, 'f :='/2. The primitive types (reference
// manual, "Primitive types") are types of no module.
ENUM_CLASS(SymbolKind, Type, Constructor, Field, FieldUpdate, Predicate,
    Function, Inst, Mode, Typeclass)

// What declares a symbol: for a type, its definition or abstract
// declaration; for a constructor or field, the constructor; for a
// predicate or function, its declaration, or its first clause when it has
// none; nothing for a primitive type.
using Declaration = std::variant<std::monostate, const parser::DuTypeDefn *,
    const parser::EquivalenceTypeDefn *, const parser::AbstractTypeDecl *,
    const parser::ForeignTypePragma *, const parser::Constructor *,
    const parser::PredDecl *, const parser::FactOrRule *,
    const parser::DcgRule *, const parser::InstDefn *,
    const parser::ModeDefn *, const parser::TypeclassDecl *>;

struct Symbol {
  SymbolKind kind;
  llvm::StringRef module; // fully qualified; empty for a primitive type
  llvm::StringRef name;
  unsigned arity; // a function's, without its result
  Declaration declaration;
  parser::CharBlock source; // the name where it is declared
  bool exported; // declared in the interface section
  // A constructor's or field's type, a method's type class.
  const Symbol *owner{nullptr};
  // The tokens of the module's source, which the declaration indexes.
  const parser::TokenSequence *tokens{nullptr};
};

// Prints a symbol as the kind and the qualified name and arity: pred
// list.length/2.
llvm::raw_ostream &operator<<(llvm::raw_ostream &, const Symbol &);

// The symbols one module declares, by name.
class ModuleSymbols {
public:
  explicit ModuleSymbols(
      llvm::StringRef name, const parser::TokenSequence *tokens = nullptr)
      : name_{name.str()}, tokens_{tokens} {}
  ModuleSymbols(const ModuleSymbols &) = delete;

  llvm::StringRef name() const { return name_; }
  // A new symbol, or the one already declared with its kind, name, arity
  // and owner, which becomes exported if this declaration is; a type's
  // definition replaces its abstract declaration as its declaration.
  const Symbol &Add(SymbolKind, llvm::StringRef name, unsigned arity,
      Declaration, parser::CharBlock source, bool exported,
      const Symbol *owner = nullptr);
  // The symbols with this name, of every kind and arity.
  llvm::ArrayRef<const Symbol *> Lookup(llvm::StringRef name) const;
  // Every symbol, in the order declared.
  llvm::ArrayRef<const Symbol *> symbols() const { return order_; }

private:
  std::string name_;
  const parser::TokenSequence *tokens_;
  llvm::BumpPtrAllocator allocator_;
  llvm::StringSaver names_{allocator_};
  std::vector<std::unique_ptr<Symbol>> owned_;
  std::vector<const Symbol *> order_;
  llvm::StringMap<std::vector<const Symbol *>> byName_;
};

} // namespace merkurius::semantics
#endif // MERKURIUS_SEMANTICS_SYMBOLS_H_
