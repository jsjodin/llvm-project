//===-- lib/Parser/parse-tree.cpp -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "merkurius/Parser/parse-tree.h"

namespace merkurius::parser {

void SymName::PrintInline(llvm::raw_ostream &o) const {
  for (llvm::StringRef qualifier : qualifiers) {
    o << qualifier << '.';
  }
  o << name;
}

namespace {

template <typename T, typename = void> struct HasPrintInline : std::false_type {};
template <typename T>
struct HasPrintInline<T,
    std::void_t<decltype(std::declval<const T &>().PrintInline(
        std::declval<llvm::raw_ostream &>()))>> : std::true_type {};

// Each node is a line with its kind, and each of its fields a line below it:
// "name: value" for a value, "name: Kind" for a node, whose fields follow,
// and "name:" for a list, whose elements follow. A node that prints inline,
// such as a SymName, is one line.
class Dumper {
public:
  explicit Dumper(llvm::raw_ostream &o) : o_{o} {}

  template <typename T> void Node(const T &node, llvm::StringRef field = {}) {
    Indent();
    if (!field.empty()) {
      o_ << field << ": ";
    }
    o_ << T::kindName;
    if constexpr (HasPrintInline<T>::value) {
      o_ << ' ';
      node.PrintInline(o_);
      o_ << '\n';
    } else {
      o_ << '\n';
      ++depth_;
      node.ForEachField(
          [&](llvm::StringRef name, const auto &value) { Field(name, value); });
      --depth_;
    }
  }

private:
  template <typename T> void Field(llvm::StringRef name, const T &value) {
    if constexpr (IsNode<T>::value) {
      Node(value, name);
    } else if constexpr (std::is_pointer_v<T>) {
      if (value) {
        Field(name, *value);
      } else {
        Line(name, "none");
      }
    } else if constexpr (IsVariant<T>::value) {
      std::visit(
          [&](const auto &alternative) { Field(name, alternative); }, value);
    } else if constexpr (IsArrayRef<T>::value) {
      Indent();
      o_ << name << ":\n";
      ++depth_;
      for (const auto &element : value) {
        Field({}, element);
      }
      --depth_;
    } else if constexpr (std::is_enum_v<T>) {
      Line(name, EnumToString(value));
    } else if constexpr (std::is_same_v<T, llvm::StringRef>) {
      Line(name, value);
    } else {
      Indent();
      o_ << name << ": " << value << '\n';
    }
  }

  // A value, or a list element when there is no field name.
  void Line(llvm::StringRef name, llvm::StringRef value) {
    Indent();
    if (!name.empty()) {
      o_ << name << ": ";
    }
    o_ << value << '\n';
  }

  void Indent() {
    for (int j{0}; j < depth_; ++j) {
      o_ << "| ";
    }
  }

  llvm::raw_ostream &o_;
  int depth_{0};
};

} // namespace

void DumpParseTree(llvm::raw_ostream &o, const ParseTree &tree) {
  Dumper{o}.Node(tree);
}

} // namespace merkurius::parser
