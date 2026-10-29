//===-- include/merkurius/Parser/term-reader.h ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_PARSER_TERM_READER_H_
#define MERKURIUS_PARSER_TERM_READER_H_

// The term reader reads a file's tokens one item at a time: each item is a
// term followed by an end token (reference manual, "Syntax overview"). It
// uses precedence climbing over the builtin operator table and computes the
// values of literals. On a syntax error it reports the token it did not
// expect and resumes after the next end token.

#include "merkurius/Parser/term.h"
#include "merkurius/Parser/token.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include <cstdint>
#include <optional>
#include <string>

namespace merkurius::parser {

class Messages;
struct OperatorDefinition;

// An item as the reader reads it, before conversion.
struct ItemTerm {
  const Term *term; // null after a syntax error
  TokenRange tokens; // from the item's first token to its end token
  unsigned variables; // the number of distinct variables in the item
};

class TermReader {
public:
  // The reader recurses once per level of nesting: brackets, arguments and
  // the operands of operators, so "a, b, c" has three levels. An item that
  // nests more deeply is an error. Each level takes up to about 600 bytes
  // of stack.
  static constexpr unsigned defaultMaxDepth{10000};

  // The terms of each item are made in `arena`, which the caller may reset
  // between items.
  TermReader(const TokenSequence &, TermArena &, Messages &,
      unsigned maxDepth = defaultMaxDepth);

  // Reads the next item, or returns nullopt at the end of the file.
  std::optional<ItemTerm> ReadItem();

private:
  struct Parsed {
    const Term *term; // null after a syntax error
    int priority;
  };

  const Token &Peek(std::size_t ahead = 0) const;
  TokenKind PeekKind(std::size_t ahead = 0) const { return Peek(ahead).kind; }
  std::uint32_t Next() { return at_++; }
  bool AtTerminator(std::size_t ahead = 0) const;
  bool CanStartPrefixOperand() const;
  bool IsInfixOperatorAt(std::size_t ahead) const;
  bool AtBackquoteOperator(std::size_t ahead) const;
  const OperatorDefinition *GetOperator(const Token &) const;
  void SayUnexpected(llvm::StringRef expected);
  template <typename... A> void Say(std::uint32_t first, A &&...);
  void SkipItem();

  Parsed Parse(int minPriority);
  Parsed ParseOperators(int minPriority);
  Parsed ParsePrimary(int minPriority);
  Parsed ParseName(int minPriority);
  Parsed ParseEmptyBracketsCall(llvm::StringRef name, std::uint32_t first);
  Parsed ParseNested(int priority, bool commaIsOperator);
  const Term *ParseArgument();
  const Term *ParseApplications(const Term *, std::uint32_t first);
  bool ParseArguments(TokenKind close, llvm::SmallVectorImpl<const Term *> &,
      const Term **tail = nullptr);
  const Term *ParseBackquoteOperator();
  const Term *MakeLiteral(
      std::uint32_t first, std::uint32_t token, bool negative);
  const Term *MakeInteger(
      std::uint32_t first, std::uint32_t token, bool negative);
  const VariableTerm *MakeVariable(std::uint32_t token);
  const NameTerm *MakeName(std::uint32_t token);
  TokenRange Range(std::uint32_t first) const { return {first, at_ - 1}; }

  const TokenSequence &tokens_;
  TermArena &arena_;
  Messages &messages_;
  const unsigned maxDepth_;
  unsigned depth_{0};
  std::uint32_t at_{0};
  bool failed_{false}; // a syntax error in the current item
  // Within arguments and elements, ',' separates instead of being an
  // operator. argumentStart_ is the first token of the innermost argument,
  // element, parenthesized term or item.
  bool commaIsOperator_{true};
  std::uint32_t argumentStart_{0};
  llvm::StringMap<unsigned> variableNumbers_;
  unsigned variables_{0};
};

// Removes the quotes of a quoted name or string and replaces its escapes
// and doubled quotes; returns other spellings unchanged.
std::string UnquoteSpelling(llvm::StringRef spelling);

} // namespace merkurius::parser
#endif // MERKURIUS_PARSER_TERM_READER_H_
