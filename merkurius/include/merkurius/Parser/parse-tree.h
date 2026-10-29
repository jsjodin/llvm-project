//===-- include/merkurius/Parser/parse-tree.h -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_PARSER_PARSE_TREE_H_
#define MERKURIUS_PARSER_PARSE_TREE_H_

// The parse tree: one type per construct of the reference manual, made by
// the converter from the terms of a file's items. Nodes are immutable and
// live in a ParseTreeArena, so they are trivially destructible and refer to
// each other by pointer. A category with alternatives, such as Item, is a
// std::variant of pointers to its forms.
//
// Every node has a kindName, its first and last token, and a ForEachField
// member that calls f(name, value) for each of its other fields; Walk and
// DumpParseTree use it, so a node type needs nothing else.

#include "merkurius/Common/arena.h"
#include "merkurius/Common/enum-class.h"
#include "merkurius/Parser/term.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <type_traits>
#include <variant>

namespace merkurius::parser {

using ParseTreeArena = common::Arena;

// A name, possibly module-qualified: write, io.write or io__write.
struct SymName {
  static constexpr llvm::StringLiteral kindName{"SymName"};
  TokenRange tokens;
  llvm::ArrayRef<llvm::StringRef> qualifiers; // the outermost module first
  llvm::StringRef name;
  template <typename F> void ForEachField(F &&f) const {
    f("qualifiers", qualifiers);
    f("name", name);
  }
  // Prints io.write, with the qualifiers.
  void PrintInline(llvm::raw_ostream &) const;
};

//===----------------------------------------------------------------------===//
// The module system (reference manual, "The module system")
//===----------------------------------------------------------------------===//

// :- module Name.
struct ModuleDecl {
  static constexpr llvm::StringLiteral kindName{"ModuleDecl"};
  TokenRange tokens;
  const SymName *name;
  template <typename F> void ForEachField(F &&f) const { f("name", name); }
};

// :- end_module Name.
struct EndModuleDecl {
  static constexpr llvm::StringLiteral kindName{"EndModuleDecl"};
  TokenRange tokens;
  const SymName *name;
  template <typename F> void ForEachField(F &&f) const { f("name", name); }
};

ENUM_CLASS(Section, Interface, Implementation)

// :- interface. or :- implementation.
struct SectionDecl {
  static constexpr llvm::StringLiteral kindName{"SectionDecl"};
  TokenRange tokens;
  Section section;
  template <typename F> void ForEachField(F &&f) const {
    f("section", section);
  }
};

ENUM_CLASS(ImportKind, ImportModule, UseModule, IncludeModule)

// :- import_module Modules. and its use_module and include_module forms.
struct ImportDecl {
  static constexpr llvm::StringLiteral kindName{"ImportDecl"};
  TokenRange tokens;
  ImportKind kind;
  llvm::ArrayRef<const SymName *> modules;
  template <typename F> void ForEachField(F &&f) const {
    f("kind", kind);
    f("modules", modules);
  }
};

//===----------------------------------------------------------------------===//
// Items (reference manual, "Items")
//===----------------------------------------------------------------------===//

using Item = std::variant<const ModuleDecl *, const EndModuleDecl *,
    const SectionDecl *, const ImportDecl *>;

// The items of a file that converted without errors, in order.
struct ParseTree {
  static constexpr llvm::StringLiteral kindName{"ParseTree"};
  TokenRange tokens;
  llvm::ArrayRef<Item> items;
  template <typename F> void ForEachField(F &&f) const { f("items", items); }
};

//===----------------------------------------------------------------------===//
// Walking and dumping
//===----------------------------------------------------------------------===//

template <typename T, typename = void> struct IsNode : std::false_type {};
template <typename T>
struct IsNode<T, std::void_t<decltype(T::kindName)>> : std::true_type {};

template <typename T> struct IsVariant : std::false_type {};
template <typename... A>
struct IsVariant<std::variant<A...>> : std::true_type {};

template <typename T> struct IsArrayRef : std::false_type {};
template <typename T> struct IsArrayRef<llvm::ArrayRef<T>> : std::true_type {};

namespace detail {
template <typename V, typename N>
auto CallPre(V &visitor, const N &node, int) -> decltype(visitor.Pre(node)) {
  return visitor.Pre(node);
}
template <typename V, typename N> bool CallPre(V &, const N &, long) {
  return true;
}
template <typename V, typename N>
auto CallPost(V &visitor, const N &node, int) -> decltype(visitor.Post(node)) {
  visitor.Post(node);
}
template <typename V, typename N> void CallPost(V &, const N &, long) {}
} // namespace detail

// Visits `x` and everything below it. For each node, visitor.Pre(node) is
// called first, if the visitor has a Pre for that node type, and the node's
// fields are visited only if it returns true; then visitor.Post(node) is
// called, if there is one.
template <typename V, typename T> void Walk(const T &x, V &visitor) {
  if constexpr (IsNode<T>::value) {
    if (detail::CallPre(visitor, x, 0)) {
      x.ForEachField(
          [&](llvm::StringRef, const auto &field) { Walk(field, visitor); });
      detail::CallPost(visitor, x, 0);
    }
  } else if constexpr (std::is_pointer_v<T>) {
    if (x) {
      Walk(*x, visitor);
    }
  } else if constexpr (IsVariant<T>::value) {
    std::visit([&](const auto &alternative) { Walk(alternative, visitor); }, x);
  } else if constexpr (IsArrayRef<T>::value) {
    for (const auto &element : x) {
      Walk(element, visitor);
    }
  }
}

// Prints the tree, one node or field per line, indented by depth.
void DumpParseTree(llvm::raw_ostream &, const ParseTree &);

} // namespace merkurius::parser
#endif // MERKURIUS_PARSER_PARSE_TREE_H_
