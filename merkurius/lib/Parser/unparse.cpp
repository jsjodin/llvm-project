//===-- lib/Parser/unparse.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "merkurius/Parser/unparse.h"
#include "merkurius/Parser/message.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <algorithm>
#include <type_traits>

namespace merkurius::parser {

namespace {

class ExactUnparser {
public:
  ExactUnparser(llvm::raw_ostream &o, const TokenSequence &tokens,
      UnparseLayout layout, bool complete, Messages &messages)
      : o_{o}, tokens_{tokens}, layout_{layout}, complete_{complete},
        messages_{messages} {}

  // Prints the tokens of the node's range: its children's through them, and
  // the ones between them itself, checking that its fields hold those.
  template <typename N> void Node(const N &node) {
    Fields fields;
    node.ForEachField(
        [&](llvm::StringRef, const auto &value) { Gather(value, fields); });
    std::stable_sort(fields.children.begin(), fields.children.end(),
        [](const Child &x, const Child &y) {
          return x.tokens.first < y.tokens.first;
        });
    // The tokens between items are only accounted for when none was left
    // out.
    bool check{!std::is_same_v<N, ParseTree> || complete_};
    std::uint32_t next{node.tokens.first};
    const Child *previous{nullptr};
    for (const Child &child : fields.children) {
      if (child.tokens.first > child.tokens.last ||
          child.tokens.first < node.tokens.first ||
          child.tokens.last > node.tokens.last) {
        messages_.Say(Source(child.tokens),
            "internal error: the tokens of this %s are not within those of "
            "its parent %s"_err_en_US,
            child.kind.str(), N::kindName.str());
        check = false;
      } else if (child.tokens.first < next) {
        messages_.Say(Source(child.tokens),
            "internal error: the tokens of this %s overlap those of the %s "
            "before it"_err_en_US,
            child.kind.str(),
            previous ? previous->kind.str() : N::kindName.str());
        check = false;
      } else {
        Gap(next, child.tokens.first, fields, check, N::kindName);
        (this->*child.unparse)(child.node);
        next = child.tokens.last + 1;
        previous = &child;
      }
    }
    Gap(next, node.tokens.last + 1, fields, check, N::kindName);
  }

private:
  struct Child {
    TokenRange tokens;
    llvm::StringRef kind;
    const void *node;
    void (ExactUnparser::*unparse)(const void *);
  };
  // A node's children, and the values of its fields that can hold a token.
  struct Fields {
    llvm::SmallVector<Child, 8> children;
    llvm::SmallVector<llvm::StringRef, 4> strings;
    unsigned numbers{0};
    unsigned enums{0};
  };

  template <typename N> void Unparse(const void *node) {
    Node(*static_cast<const N *>(node));
  }

  template <typename T> void Gather(const T &value, Fields &fields) {
    if constexpr (IsNode<T>::value) {
      fields.children.push_back(
          {value.tokens, T::kindName, &value, &ExactUnparser::Unparse<T>});
    } else if constexpr (std::is_pointer_v<T>) {
      if (value) {
        Gather(*value, fields);
      }
    } else if constexpr (IsOptional<T>::value) {
      if (value) {
        Gather(*value, fields);
      }
    } else if constexpr (IsVariant<T>::value) {
      std::visit([&](const auto &x) { Gather(x, fields); }, value);
    } else if constexpr (IsArrayRef<T>::value) {
      for (const auto &element : value) {
        Gather(element, fields);
      }
    } else if constexpr (std::is_same_v<T, llvm::StringRef>) {
      fields.strings.push_back(value);
    } else if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>) {
      ++fields.numbers;
    } else if constexpr (std::is_enum_v<T>) {
      ++fields.enums;
    }
  }

  // Prints the tokens [from, to) that a node prints itself.
  void Gap(std::uint32_t from, std::uint32_t to, Fields &fields, bool check,
      llvm::StringRef kind) {
    for (std::uint32_t j{from}; j < to; ++j) {
      const Token &token{tokens_.tokens[j]};
      Trivia(token.trivia);
      o_ << token.spelling;
      if (check && !Holds(token, fields)) {
        messages_.Say(token.spelling,
            "internal error: the token '%s' is within this %s, but none of "
            "its fields holds it"_err_en_US,
            token.spelling.str(), kind.str());
      }
    }
  }

  void Trivia(llvm::StringRef trivia) {
    if (layout_ == UnparseLayout::Exact) {
      o_ << trivia;
    } else if (trivia.contains('\n')) {
      o_ << '\n';
    } else if (!trivia.empty()) {
      o_ << ' ';
    }
  }

  // Whether one of the node's fields holds the token, using up that field:
  // a variable is held by a string field equal to its name, and a literal
  // by a number field or any string field, since fields hold literals'
  // values, not their spellings. A string may also be held by an enum
  // field, as "C" is by a ForeignLanguage. Names and punctuation are not
  // checked: they are the node's own syntax.
  static bool Holds(const Token &token, Fields &fields) {
    llvm::StringRef spelling{token.spelling};
    auto take{[&](auto predicate) {
      auto it{llvm::find_if(fields.strings, predicate)};
      if (it == fields.strings.end()) {
        return false;
      }
      fields.strings.erase(it);
      return true;
    }};
    auto equal{[&](llvm::StringRef s) { return s == spelling; }};
    auto any{[](llvm::StringRef) { return true; }};
    switch (token.kind) {
    case TokenKind::Variable:
      return take(equal);
    case TokenKind::Integer:
    case TokenKind::Float:
      if (fields.numbers > 0) {
        --fields.numbers;
        return true;
      }
      return take(equal) || take(any);
    case TokenKind::String:
    case TokenKind::ImplementationDefinedLiteral:
      // Prefer a string that cannot be a variable's name, which a variable
      // later in the node may need.
      if (take(equal) || take([&](llvm::StringRef s) {
            return s.empty() || !(s[0] == '_' || (s[0] >= 'A' && s[0] <= 'Z'));
          }) || take(any)) {
        return true;
      }
      if (token.kind == TokenKind::String && fields.enums > 0) {
        --fields.enums;
        return true;
      }
      return false;
    default:
      return true;
    }
  }

  CharBlock Source(TokenRange range) const {
    std::uint32_t last{std::min<std::uint32_t>(
        range.last, tokens_.tokens.size() - 1)};
    std::uint32_t first{std::min(range.first, last)};
    return CharBlock{tokens_.tokens[first].spelling.begin(),
        tokens_.tokens[last].spelling.end()};
  }

  llvm::raw_ostream &o_;
  const TokenSequence &tokens_;
  UnparseLayout layout_;
  bool complete_;
  Messages &messages_;
};

} // namespace

void UnparseExact(llvm::raw_ostream &o, const ParseTree &tree,
    const TokenSequence &tokens, UnparseLayout layout, bool complete,
    Messages &messages) {
  ExactUnparser{o, tokens, layout, complete, messages}.Node(tree);
}

} // namespace merkurius::parser
