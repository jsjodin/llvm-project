//===-- include/merkurius/Parser/operators.h --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_PARSER_OPERATORS_H_
#define MERKURIUS_PARSER_OPERATORS_H_

// Mercury's builtin operators (reference manual, "Builtin operators"), with
// the current manual's numbering: a higher priority binds more tightly. In a
// specifier, 'f' is the operator; an 'x' argument must have a strictly higher
// priority than the operator, and a 'y' argument a higher or equal one.

#include "llvm/ADT/StringRef.h"
#include <optional>

namespace merkurius::parser {

enum class Specifier {
  FX, // prefix
  FY,
  FXY, // binary prefix, as in "some [X] Goal"
  XFX, // infix
  XFY,
  YFX,
};

// Priorities of terms. Every operator's priority lies strictly between
// these, so a primary term is a valid argument of any operator.
inline constexpr int lowestPriority{0};
inline constexpr int primaryPriority{1500};
// An item is read at the priority of ':-' and '-->'.
inline constexpr int itemPriority{300};
// Arguments of compound terms and elements of lists and tuples are read
// above the priority of ',' (500), so that a comma separates them.
inline constexpr int argumentPriority{501};
// `name` and `Variable` used as infix operators.
inline constexpr int backquotePriority{1380};
inline constexpr Specifier backquoteSpecifier{Specifier::YFX};

struct OperatorForm {
  Specifier specifier;
  int priority;

  bool IsPrefix() const;
  // The lowest priority allowed for the left argument of an infix operator,
  // or the first argument of a binary prefix operator.
  int LeftArgumentPriority() const;
  // The lowest priority allowed for the right argument of an infix
  // operator, or the (last) argument of a prefix operator.
  int RightArgumentPriority() const;
};

// A name's forms as an operator. No builtin operator is postfix, and none
// has more than one prefix or one infix form.
struct OperatorDefinition {
  llvm::StringRef name;
  std::optional<OperatorForm> prefix; // FX, FY or FXY
  std::optional<OperatorForm> infix; // XFX, XFY or YFX
};

// The builtin operator definition of a name, as written without quotes, or
// null if it is not an operator. A quoted name is the same operator (manual,
// "Tokens": "if a character is an operator, then the result of enclosing
// that character in single quotes is also an operator").
const OperatorDefinition *FindOperator(llvm::StringRef name);

llvm::StringRef GetSpecifierName(Specifier);

} // namespace merkurius::parser
#endif // MERKURIUS_PARSER_OPERATORS_H_
