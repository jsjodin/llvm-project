//===-- lib/Semantics/symbols.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "merkurius/Semantics/symbols.h"
#include "llvm/ADT/StringExtras.h"

namespace merkurius::semantics {

const Symbol &ModuleSymbols::Add(SymbolKind kind, llvm::StringRef name,
    unsigned arity, Declaration declaration, parser::CharBlock source,
    bool exported, const Symbol *owner) {
  std::vector<const Symbol *> &same{byName_[name]};
  for (const Symbol *symbol : same) {
    if (symbol->kind == kind && symbol->arity == arity) {
      // A type's abstract declaration in the interface and its definition
      // in the implementation are one symbol.
      // The symbol's declaration is the definition.
      auto &existing{const_cast<Symbol &>(*symbol)};
      existing.exported |= exported;
      if (std::holds_alternative<const parser::AbstractTypeDecl *>(
              existing.declaration)) {
        existing.declaration = declaration;
      }
      return existing;
    }
  }
  auto &symbol{*owned_.emplace_back(std::make_unique<Symbol>(Symbol{kind,
      name_, names_.save(name), arity, declaration, source, exported, owner,
      tokens_}))};
  same.push_back(&symbol);
  order_.push_back(&symbol);
  return symbol;
}

llvm::ArrayRef<const Symbol *> ModuleSymbols::Lookup(
    llvm::StringRef name) const {
  auto it{byName_.find(name)};
  return it == byName_.end() ? llvm::ArrayRef<const Symbol *>{}
                             : llvm::ArrayRef<const Symbol *>{it->second};
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Symbol &symbol) {
  static constexpr const char *kinds[]{"type", "constructor", "field",
      "field update", "pred", "func", "inst", "mode", "typeclass"};
  os << kinds[static_cast<int>(symbol.kind)] << ' ';
  if (!symbol.module.empty()) {
    os << symbol.module << '.';
  }
  // A name that is not an identifier is quoted, as Mercury writes it.
  llvm::StringRef name{symbol.name};
  bool plain{!name.empty() && llvm::isLower(name.front()) &&
      llvm::all_of(name, [](char c) { return llvm::isAlnum(c) || c == '_'; })};
  if (plain) {
    os << name;
  } else {
    os << '\'' << name << '\'';
  }
  return os << '/' << symbol.arity;
}

} // namespace merkurius::semantics
