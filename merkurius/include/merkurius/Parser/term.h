//===-- include/merkurius/Parser/term.h -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_PARSER_TERM_H_
#define MERKURIUS_PARSER_TERM_H_

// Terms, as the term reader reads them (reference manual, "Terms"). There
// is one node kind per way a term can be written, so that lists, tuples,
// parentheses and operator notation are kept as written; PrintCanonical
// gives the meaning, in which they are all compound terms. Every node
// records its first and last token. Nodes are allocated in a TermArena,
// which is reset after each item, and are never destroyed individually.

#include "merkurius/Parser/char-block.h"
#include "merkurius/Parser/token.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstdint>
#include <new>
#include <type_traits>

namespace merkurius::parser {

// Indices into TokenSequence::tokens, first and last inclusive.
struct TokenRange {
  std::uint32_t first;
  std::uint32_t last;
};

class Term {
public:
  enum class Kind {
    Variable,
    Integer,
    Float,
    String,
    Name,
    ImplementationDefinedLiteral,
    Compound, // f(Args), or an operator term
    Backquote, // X `Op` Y
    List, // [A, B | T]
    Tuple, // {A, B}
    Apply, // a higher-order term: Closure(Args)
    Parenthesized, // (T)
    Directive, // a %$omp directive and the term after it
  };

  Kind kind() const { return kind_; }
  TokenRange tokens() const { return tokens_; }

protected:
  Term(Kind kind, TokenRange tokens) : kind_{kind}, tokens_{tokens} {}

private:
  Kind kind_;
  TokenRange tokens_;
};

// A variable, numbered from 0 within its item in order of first
// appearance. Each '_' is a distinct variable.
class VariableTerm : public Term {
public:
  VariableTerm(TokenRange tokens, llvm::StringRef name, unsigned number)
      : Term{Kind::Variable, tokens}, name_{name}, number_{number} {}
  llvm::StringRef name() const { return name_; }
  unsigned number() const { return number_; }
  bool IsAnonymous() const { return name_ == "_"; }
  static bool classof(const Term *t) { return t->kind() == Kind::Variable; }

private:
  llvm::StringRef name_;
  unsigned number_;
};

// The size suffix of an integer literal: i8 ... u64, or none.
enum class IntegerSize { Default, Bits8, Bits16, Bits32, Bits64 };

class IntegerTerm : public Term {
public:
  IntegerTerm(TokenRange tokens, std::uint64_t magnitude, bool negative,
      bool isUnsigned, IntegerSize size)
      : Term{Kind::Integer, tokens}, magnitude_{magnitude}, negative_{negative},
        isUnsigned_{isUnsigned}, size_{size} {}
  std::uint64_t magnitude() const { return magnitude_; }
  bool negative() const { return negative_; }
  bool isUnsigned() const { return isUnsigned_; }
  IntegerSize size() const { return size_; }
  static bool classof(const Term *t) { return t->kind() == Kind::Integer; }

private:
  std::uint64_t magnitude_;
  bool negative_;
  bool isUnsigned_;
  IntegerSize size_;
};

class FloatTerm : public Term {
public:
  FloatTerm(TokenRange tokens, double value)
      : Term{Kind::Float, tokens}, value_{value} {}
  double value() const { return value_; }
  static bool classof(const Term *t) { return t->kind() == Kind::Float; }

private:
  double value_;
};

class StringTerm : public Term {
public:
  StringTerm(TokenRange tokens, llvm::StringRef value)
      : Term{Kind::String, tokens}, value_{value} {}
  // The characters of the string, with escapes and doubled quotes replaced.
  llvm::StringRef value() const { return value_; }
  static bool classof(const Term *t) { return t->kind() == Kind::String; }

private:
  llvm::StringRef value_;
};

class NameTerm : public Term {
public:
  NameTerm(TokenRange tokens, llvm::StringRef value)
      : Term{Kind::Name, tokens}, value_{value} {}
  // The name without quotes, with escapes and doubled quotes replaced.
  llvm::StringRef value() const { return value_; }
  static bool classof(const Term *t) { return t->kind() == Kind::Name; }

private:
  llvm::StringRef value_;
};

// '$' and an unquoted name, such as $file or $line.
class ImplementationDefinedLiteralTerm : public Term {
public:
  ImplementationDefinedLiteralTerm(TokenRange tokens, llvm::StringRef name)
      : Term{Kind::ImplementationDefinedLiteral, tokens}, name_{name} {}
  llvm::StringRef name() const { return name_; } // without the '$'
  static bool classof(const Term *t) {
    return t->kind() == Kind::ImplementationDefinedLiteral;
  }

private:
  llvm::StringRef name_;
};

// How a compound term was written.
enum class CompoundForm {
  Canonical, // f(X, Y)
  Prefix, // - X, and the binary prefix form "some [X] G"
  Infix, // X + Y
};

class CompoundTerm : public Term {
public:
  CompoundTerm(TokenRange tokens, const NameTerm *functor,
      llvm::ArrayRef<const Term *> arguments, CompoundForm form)
      : Term{Kind::Compound, tokens}, functor_{functor},
        arguments_{arguments}, form_{form} {}
  const NameTerm *functor() const { return functor_; }
  llvm::ArrayRef<const Term *> arguments() const { return arguments_; }
  CompoundForm form() const { return form_; }
  static bool classof(const Term *t) { return t->kind() == Kind::Compound; }

private:
  const NameTerm *functor_;
  llvm::ArrayRef<const Term *> arguments_;
  CompoundForm form_;
};

// X `Op` Y. Op is a name, a module-qualified name or a variable. With a
// name it means Op(X, Y); with a variable it is a higher-order call.
class BackquoteTerm : public Term {
public:
  BackquoteTerm(
      TokenRange tokens, const Term *left, const Term *op, const Term *right)
      : Term{Kind::Backquote, tokens}, left_{left}, op_{op}, right_{right} {}
  const Term *left() const { return left_; }
  const Term *op() const { return op_; }
  const Term *right() const { return right_; }
  static bool classof(const Term *t) { return t->kind() == Kind::Backquote; }

private:
  const Term *left_, *op_, *right_;
};

// [E1, ..., En | Tail], where the tail is optional; [] has no elements.
class ListTerm : public Term {
public:
  ListTerm(TokenRange tokens, llvm::ArrayRef<const Term *> elements,
      const Term *tail)
      : Term{Kind::List, tokens}, elements_{elements}, tail_{tail} {}
  llvm::ArrayRef<const Term *> elements() const { return elements_; }
  const Term *tail() const { return tail_; } // null if there is none
  static bool classof(const Term *t) { return t->kind() == Kind::List; }

private:
  llvm::ArrayRef<const Term *> elements_;
  const Term *tail_;
};

// {E1, ..., En}; {} has no elements.
class TupleTerm : public Term {
public:
  TupleTerm(TokenRange tokens, llvm::ArrayRef<const Term *> elements)
      : Term{Kind::Tuple, tokens}, elements_{elements} {}
  llvm::ArrayRef<const Term *> elements() const { return elements_; }
  static bool classof(const Term *t) { return t->kind() == Kind::Tuple; }

private:
  llvm::ArrayRef<const Term *> elements_;
};

// A higher-order term, Closure(A1, ..., An), which means ''(Closure, A1,
// ..., An) (reference manual, "Terms").
class ApplyTerm : public Term {
public:
  ApplyTerm(TokenRange tokens, const Term *closure,
      llvm::ArrayRef<const Term *> arguments)
      : Term{Kind::Apply, tokens}, closure_{closure}, arguments_{arguments} {}
  const Term *closure() const { return closure_; }
  llvm::ArrayRef<const Term *> arguments() const { return arguments_; }
  static bool classof(const Term *t) { return t->kind() == Kind::Apply; }

private:
  const Term *closure_;
  llvm::ArrayRef<const Term *> arguments_;
};

class ParenthesizedTerm : public Term {
public:
  ParenthesizedTerm(TokenRange tokens, const Term *inner)
      : Term{Kind::Parenthesized, tokens}, inner_{inner} {}
  const Term *inner() const { return inner_; }
  static bool classof(const Term *t) {
    return t->kind() == Kind::Parenthesized;
  }

private:
  const Term *inner_;
};

// A %$omp directive token and the term that follows it, which the converter
// accepts only where that term is a goal.
class DirectiveTerm : public Term {
public:
  DirectiveTerm(
      TokenRange tokens, std::uint32_t directiveToken, const Term *annotated)
      : Term{Kind::Directive, tokens}, directiveToken_{directiveToken},
        annotated_{annotated} {}
  std::uint32_t directiveToken() const { return directiveToken_; }
  const Term *annotated() const { return annotated_; }
  static bool classof(const Term *t) { return t->kind() == Kind::Directive; }

private:
  std::uint32_t directiveToken_;
  const Term *annotated_;
};

// The arena that holds the terms of one item, and the strings and argument
// arrays they refer to.
class TermArena {
public:
  template <typename T, typename... A> const T *Make(A &&...args) {
    static_assert(std::is_base_of_v<Term, T>);
    static_assert(std::is_trivially_destructible_v<T>);
    return new (allocator_.Allocate<T>()) T(std::forward<A>(args)...);
  }
  llvm::StringRef Save(llvm::StringRef s) {
    if (s.empty()) {
      return {};
    }
    char *p{allocator_.Allocate<char>(s.size())};
    std::copy(s.begin(), s.end(), p);
    return {p, s.size()};
  }
  llvm::ArrayRef<const Term *> Save(llvm::ArrayRef<const Term *> terms) {
    if (terms.empty()) {
      return {};
    }
    const Term **p{allocator_.Allocate<const Term *>(terms.size())};
    std::copy(terms.begin(), terms.end(), p);
    return {p, terms.size()};
  }
  // Frees every term made since the last reset.
  void Reset() { allocator_.Reset(); }

private:
  llvm::BumpPtrAllocator allocator_;
};

// The characters from a term's first token to its last.
CharBlock GetSource(const Term &, const TokenSequence &);

// Prints a term's meaning in canonical notation: every compound term as
// functor(Args), lists as '[|]'(H, T) and [], tuples as {}(Args), higher-order
// terms as ''(Closure, Args), and parentheses removed. Names are quoted when
// they are not alphanumeric. A directive prints as '%$omp'(Term), without its
// text. Two terms mean the same if they print the same.
void PrintCanonical(llvm::raw_ostream &, const Term &);

} // namespace merkurius::parser
#endif // MERKURIUS_PARSER_TERM_H_
