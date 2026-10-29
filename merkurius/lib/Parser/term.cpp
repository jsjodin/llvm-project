//===-- lib/Parser/term.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "merkurius/Parser/term.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace merkurius::parser {

bool IntegerTerm::InRange() const {
  unsigned bits{size_ == IntegerSize::Bits8 ? 8u
          : size_ == IntegerSize::Bits16    ? 16u
          : size_ == IntegerSize::Bits32    ? 32u
                                            : 64u};
  llvm::APInt value{magnitude()};
  unsigned active{value.getActiveBits()};
  if (isUnsigned_) {
    return active <= bits && (!negative_ || value.isZero());
  }
  // The most negative value is -2^(bits-1).
  return active < bits || (negative_ && active == bits && value.isPowerOf2());
}

CharBlock GetSource(const Term &term, const TokenSequence &tokens) {
  const Token &first{tokens.tokens.at(term.tokens().first)};
  const Token &last{tokens.tokens.at(term.tokens().last)};
  return CharBlock{first.spelling.begin(), last.spelling.end()};
}

namespace {

bool IsAlphanumericName(llvm::StringRef name) {
  if (name.empty() || name[0] < 'a' || name[0] > 'z') {
    return false;
  }
  for (char ch : name) {
    if (!(llvm::isAlnum(ch) || ch == '_')) {
      return false;
    }
  }
  return true;
}

// Writes the characters of a quoted name or string, escaping the quote,
// backslashes and control characters.
void PrintQuoted(llvm::raw_ostream &o, llvm::StringRef text, char quote) {
  o << quote;
  for (char ch : text) {
    switch (ch) {
    case '\\':
      o << "\\\\";
      break;
    case '\n':
      o << "\\n";
      break;
    case '\t':
      o << "\\t";
      break;
    case '\r':
      o << "\\r";
      break;
    default:
      if (ch == quote) {
        o << '\\' << ch;
      } else if (static_cast<unsigned char>(ch) < 0x20 || ch == 0x7f) {
        o << llvm::format("\\x%x\\", static_cast<unsigned char>(ch));
      } else {
        o << ch;
      }
    }
  }
  o << quote;
}

// [] and {} are unquoted only as constants, where they read back as the
// empty list and the empty tuple.
void PrintName(llvm::raw_ostream &o, llvm::StringRef name, bool constant) {
  if (IsAlphanumericName(name) ||
      (constant && (name == "[]" || name == "{}"))) {
    o << name;
  } else {
    PrintQuoted(o, name, '\'');
  }
}

class CanonicalPrinter {
public:
  explicit CanonicalPrinter(llvm::raw_ostream &o) : o_{o} {}

  void Print(const Term &term) {
    switch (term.kind()) {
    case Term::Kind::Variable:
      o_ << llvm::cast<VariableTerm>(term).name();
      return;
    case Term::Kind::Integer:
      PrintInteger(llvm::cast<IntegerTerm>(term));
      return;
    case Term::Kind::Float:
      PrintFloat(llvm::cast<FloatTerm>(term).value());
      return;
    case Term::Kind::String:
      PrintQuoted(o_, llvm::cast<StringTerm>(term).value(), '"');
      return;
    case Term::Kind::Name:
      PrintName(o_, llvm::cast<NameTerm>(term).value(), /*constant=*/true);
      return;
    case Term::Kind::ImplementationDefinedLiteral:
      o_ << '$' << llvm::cast<ImplementationDefinedLiteralTerm>(term).name();
      return;
    case Term::Kind::Compound: {
      const auto &compound{llvm::cast<CompoundTerm>(term)};
      PrintApplication(
          compound.functor()->value(), nullptr, compound.arguments());
      return;
    }
    case Term::Kind::Backquote:
      PrintBackquote(llvm::cast<BackquoteTerm>(term));
      return;
    case Term::Kind::List: {
      const auto &list{llvm::cast<ListTerm>(term)};
      PrintList(list.elements(), list.tail());
      return;
    }
    case Term::Kind::Tuple: {
      llvm::ArrayRef<const Term *> elements{
          llvm::cast<TupleTerm>(term).elements()};
      PrintApplication("{}", nullptr, elements);
      return;
    }
    case Term::Kind::Apply: {
      const auto &apply{llvm::cast<ApplyTerm>(term)};
      PrintApplication("", apply.closure(), apply.arguments());
      return;
    }
    case Term::Kind::Parenthesized:
      Print(*llvm::cast<ParenthesizedTerm>(term).inner());
      return;
    case Term::Kind::Directive:
      o_ << "'%$omp'(";
      Print(*llvm::cast<DirectiveTerm>(term).annotated());
      o_ << ')';
      return;
    }
    llvm_unreachable("unknown term kind");
  }

private:
  // functor(First, Arguments...), where First may be null.
  void PrintApplication(llvm::StringRef functor, const Term *first,
      llvm::ArrayRef<const Term *> arguments) {
    bool constant{!first && arguments.empty()};
    PrintName(o_, functor, constant);
    if (constant) {
      return;
    }
    o_ << '(';
    const char *separator{""};
    if (first) {
      Print(*first);
      separator = ", ";
    }
    for (const Term *argument : arguments) {
      o_ << separator;
      Print(*argument);
      separator = ", ";
    }
    o_ << ')';
  }

  void PrintList(llvm::ArrayRef<const Term *> elements, const Term *tail) {
    if (elements.empty()) {
      if (tail) {
        Print(*tail);
      } else {
        o_ << "[]";
      }
      return;
    }
    o_ << "'[|]'(";
    Print(*elements.front());
    o_ << ", ";
    PrintList(elements.drop_front(), tail);
    o_ << ')';
  }

  // X `Op` Y means Op(X, Y) for a name, ''(Op, X, Y) for a variable, and
  // M.Op(X, Y), that is '.'(M, Op(X, Y)), for a module-qualified name.
  void PrintBackquote(const BackquoteTerm &term) {
    const Term *op{term.op()};
    while (const auto *paren{llvm::dyn_cast<ParenthesizedTerm>(op)}) {
      op = paren->inner();
    }
    const Term *arguments[]{term.left(), term.right()};
    PrintQualifiedApplication(*op, arguments);
  }

  void PrintQualifiedApplication(
      const Term &op, llvm::ArrayRef<const Term *> arguments) {
    if (const auto *name{llvm::dyn_cast<NameTerm>(&op)}) {
      PrintApplication(name->value(), nullptr, arguments);
      return;
    }
    if (const auto *compound{llvm::dyn_cast<CompoundTerm>(&op)}; compound &&
        compound->functor()->value() == "." &&
        compound->arguments().size() == 2) {
      o_ << "'.'(";
      Print(*compound->arguments()[0]);
      o_ << ", ";
      PrintQualifiedApplication(*compound->arguments()[1], arguments);
      o_ << ')';
      return;
    }
    PrintApplication("", &op, arguments);
  }

  void PrintInteger(const IntegerTerm &term) {
    if (term.negative()) {
      o_ << '-';
    }
    llvm::SmallString<32> digits;
    term.magnitude().toStringUnsigned(digits, 10);
    o_ << digits;
    if (term.isUnsigned() || term.size() != IntegerSize::Default) {
      o_ << (term.isUnsigned() ? 'u' : 'i');
      switch (term.size()) {
      case IntegerSize::Default:
        break;
      case IntegerSize::Bits8:
        o_ << 8;
        break;
      case IntegerSize::Bits16:
        o_ << 16;
        break;
      case IntegerSize::Bits32:
        o_ << 32;
        break;
      case IntegerSize::Bits64:
        o_ << 64;
        break;
      }
    }
  }

  // The shortest form that reads back as the same double, always with a
  // fraction or an exponent, so that it reads back as a float.
  void PrintFloat(double value) {
    if (std::isinf(value)) {
      // There is no literal for infinity; this is one that reads as it.
      o_ << (value < 0 ? "-1.0e+309" : "1.0e+309");
      return;
    }
    char buffer[32];
    for (int precision{1}; precision <= 17; ++precision) {
      std::snprintf(buffer, sizeof buffer, "%.*g", precision, value);
      if (std::strtod(buffer, nullptr) == value) {
        break;
      }
    }
    std::string text{buffer};
    if (text.find_first_of(".en") == std::string::npos) { // n: inf and nan
      text += ".0";
    } else if (std::size_t e{text.find('e')};
               e != std::string::npos && text.find('.') == std::string::npos) {
      text.insert(e, ".0");
    }
    o_ << text;
  }

  llvm::raw_ostream &o_;
};

} // namespace

void PrintCanonical(llvm::raw_ostream &o, const Term &term) {
  CanonicalPrinter{o}.Print(term);
}

} // namespace merkurius::parser
