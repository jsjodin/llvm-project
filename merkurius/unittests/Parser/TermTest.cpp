//===-- unittests/Parser/TermTest.cpp ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "merkurius/Parser/term.h"
#include "gtest/gtest.h"
#include "merkurius/Parser/lexer.h"
#include "merkurius/Parser/message.h"
#include "merkurius/Parser/provenance.h"
#include "merkurius/Parser/source.h"
#include "llvm/Support/MemoryBuffer.h"
#include <cmath>
#include <string>

using namespace merkurius::parser;

namespace {

// Builds terms by hand; the token ranges do not matter for printing.
class TermTest : public ::testing::Test {
protected:
  static constexpr TokenRange at{0, 0};

  std::string Canonical(const Term *term) {
    std::string out;
    llvm::raw_string_ostream os{out};
    PrintCanonical(os, *term);
    return out;
  }
  const NameTerm *Name(llvm::StringRef value) {
    return arena.Make<NameTerm>(at, arena.Save(value));
  }
  const VariableTerm *Var(llvm::StringRef name, unsigned number = 0) {
    return arena.Make<VariableTerm>(at, arena.Save(name), number);
  }
  const IntegerTerm *Integer(
      std::uint64_t value, bool negative, bool isUnsigned, IntegerSize size) {
    const std::uint64_t words[]{value};
    return arena.Make<IntegerTerm>(at,
        arena.Save(llvm::ArrayRef<std::uint64_t>(words)), negative, isUnsigned,
        size);
  }
  const IntegerTerm *Int(std::uint64_t value, bool negative = false) {
    return Integer(value, negative, false, IntegerSize::Default);
  }
  const Term *Infix(llvm::StringRef op, const Term *x, const Term *y) {
    const Term *arguments[]{x, y};
    return arena.Make<CompoundTerm>(
        at, Name(op), arena.Save(arguments), CompoundForm::Infix);
  }
  const Term *Call(llvm::StringRef f, llvm::ArrayRef<const Term *> args) {
    return arena.Make<CompoundTerm>(
        at, Name(f), arena.Save(args), CompoundForm::Canonical);
  }

  TermArena arena;
};

TEST_F(TermTest, Atoms) {
  EXPECT_EQ(Canonical(Name("foo")), "foo");
  EXPECT_EQ(Canonical(Name("Foo")), "'Foo'");
  EXPECT_EQ(Canonical(Name("+")), "'+'");
  EXPECT_EQ(Canonical(Name("it's")), "'it\\'s'");
  EXPECT_EQ(Canonical(Name("")), "''");
  EXPECT_EQ(Canonical(Name("[]")), "[]");
  EXPECT_EQ(Canonical(Var("X")), "X");
  EXPECT_EQ(Canonical(Var("_")), "_");
  EXPECT_EQ(Canonical(arena.Make<ImplementationDefinedLiteralTerm>(
                at, arena.Save("line"))),
      "$line");
}

TEST_F(TermTest, Literals) {
  EXPECT_EQ(Canonical(Int(42)), "42");
  EXPECT_EQ(Canonical(Int(5, /*negative=*/true)), "-5");
  EXPECT_EQ(Canonical(Integer(255, false, true, IntegerSize::Bits8)), "255u8");
  EXPECT_EQ(Canonical(Integer(3, false, true, IntegerSize::Default)), "3u");
  EXPECT_EQ(Canonical(Integer(7, false, false, IntegerSize::Bits64)), "7i64");
  EXPECT_EQ(Canonical(arena.Make<FloatTerm>(at, 1.5)), "1.5");
  EXPECT_EQ(Canonical(arena.Make<FloatTerm>(at, 3.0)), "3.0");
  EXPECT_EQ(Canonical(arena.Make<FloatTerm>(at, 0.1)), "0.1");
  EXPECT_EQ(Canonical(arena.Make<FloatTerm>(at, 1.5e10)), "1.5e+10");
  EXPECT_EQ(Canonical(arena.Make<FloatTerm>(at, 1e100)), "1.0e+100");
  EXPECT_EQ(Canonical(arena.Make<StringTerm>(at, arena.Save("a\"b\\c\nd"))),
      "\"a\\\"b\\\\c\\nd\"");
}

TEST_F(TermTest, BigIntegers) {
  // 2^64, which the reader accepts for the converter to reject.
  const std::uint64_t words[]{0, 1};
  EXPECT_EQ(Canonical(arena.Make<IntegerTerm>(at,
                arena.Save(llvm::ArrayRef<std::uint64_t>(words)), false, true,
                IntegerSize::Default)),
      "18446744073709551616u");
  EXPECT_EQ(Canonical(arena.Make<FloatTerm>(at, HUGE_VAL)), "1.0e+309");
  EXPECT_EQ(Canonical(arena.Make<FloatTerm>(at, -HUGE_VAL)), "-1.0e+309");
}

TEST_F(TermTest, InRange) {
  auto inRange{[&](std::uint64_t value, bool negative, bool isUnsigned,
                   IntegerSize size) {
    return Integer(value, negative, isUnsigned, size)->InRange();
  }};
  using IS = IntegerSize;
  EXPECT_TRUE(inRange(127, false, false, IS::Bits8));
  EXPECT_FALSE(inRange(128, false, false, IS::Bits8));
  EXPECT_TRUE(inRange(128, true, false, IS::Bits8));
  EXPECT_FALSE(inRange(129, true, false, IS::Bits8));
  EXPECT_TRUE(inRange(255, false, true, IS::Bits8));
  EXPECT_FALSE(inRange(256, false, true, IS::Bits8));
  EXPECT_TRUE(inRange(32767, false, false, IS::Bits16));
  EXPECT_FALSE(inRange(65536, false, true, IS::Bits16));
  EXPECT_FALSE(inRange(2147483648, false, false, IS::Bits32));
  EXPECT_TRUE(inRange(4294967295, false, true, IS::Bits32));
  EXPECT_TRUE(inRange(9223372036854775807u, false, false, IS::Default));
  EXPECT_FALSE(inRange(9223372036854775808u, false, false, IS::Default));
  EXPECT_TRUE(inRange(9223372036854775808u, true, false, IS::Default));
  EXPECT_TRUE(inRange(18446744073709551615u, false, true, IS::Bits64));
  const std::uint64_t big[]{0, 1};
  EXPECT_FALSE(
      arena
          .Make<IntegerTerm>(at, arena.Save(llvm::ArrayRef<std::uint64_t>(big)),
              false, true, IS::Default)
          ->InRange());
}

TEST_F(TermTest, CompoundTerms) {
  // X + Y * 2 and +(X, *(Y, 2)) mean the same.
  const Term *sum{Infix("+", Var("X"), Infix("*", Var("Y", 1), Int(2)))};
  EXPECT_EQ(Canonical(sum), "'+'(X, '*'(Y, 2))");
  const Term *args[]{Var("X")};
  EXPECT_EQ(Canonical(Call("f", args)), "f(X)");
  // Parentheses are not part of the meaning.
  EXPECT_EQ(
      Canonical(arena.Make<ParenthesizedTerm>(at, sum)), "'+'(X, '*'(Y, 2))");
}

TEST_F(TermTest, ListsAndTuples) {
  const Term *three[]{Int(1), Int(2), Int(3)};
  EXPECT_EQ(Canonical(arena.Make<ListTerm>(at, arena.Save(three), nullptr)),
      "'[|]'(1, '[|]'(2, '[|]'(3, [])))");
  EXPECT_EQ(Canonical(arena.Make<ListTerm>(
                at, arena.Save(llvm::ArrayRef(three).take_front(1)), Var("T"))),
      "'[|]'(1, T)");
  EXPECT_EQ(Canonical(arena.Make<ListTerm>(
                at, llvm::ArrayRef<const Term *>{}, nullptr)),
      "[]");
  EXPECT_EQ(
      Canonical(arena.Make<TupleTerm>(at, arena.Save(three))), "'{}'(1, 2, 3)");
  EXPECT_EQ(
      Canonical(arena.Make<TupleTerm>(at, llvm::ArrayRef<const Term *>{})),
      "{}");
}

TEST_F(TermTest, HigherOrderAndBackquote) {
  const Term *args[]{Var("X", 1), Var("Y", 2)};
  EXPECT_EQ(Canonical(arena.Make<ApplyTerm>(at, Var("P"), arena.Save(args))),
      "''(P, X, Y)");
  // X `max` Y is max(X, Y); X `int.max` Y is int.max(X, Y), that is
  // '.'(int, max(X, Y)); X `F` Y is the higher-order call ''(F, X, Y).
  EXPECT_EQ(
      Canonical(arena.Make<BackquoteTerm>(at, args[0], Name("max"), args[1])),
      "max(X, Y)");
  const Term *qualified{Infix(".", Name("int"), Name("max"))};
  EXPECT_EQ(
      Canonical(arena.Make<BackquoteTerm>(at, args[0], qualified, args[1])),
      "'.'(int, max(X, Y))");
  EXPECT_EQ(
      Canonical(arena.Make<BackquoteTerm>(at, args[0], Var("F"), args[1])),
      "''(F, X, Y)");
}

TEST_F(TermTest, Directive) {
  EXPECT_EQ(Canonical(arena.Make<DirectiveTerm>(at, 0, Name("goal"))),
      "'%$omp'(goal)");
}

TEST_F(TermTest, ArenaReset) {
  const NameTerm *name{Name("first")};
  EXPECT_EQ(name->value(), "first");
  arena.Reset();
  EXPECT_EQ(Canonical(Name("second")), "second");
}

TEST(TermSourceTest, GetSource) {
  AllSources all;
  SourceFile *file{all.AddBuffer(
      "test.m", llvm::MemoryBuffer::getMemBufferCopy("p :- q(X, Y).\n"))};
  Messages messages;
  TokenSequence tokens{Lex(*file, LexerOptions{}, messages)};
  // Tokens: p :- q ( X , Y ) . eof; the term q(X, Y) spans tokens 2 to 7.
  TermArena arena;
  const Term *args[]{arena.Make<VariableTerm>(TokenRange{4, 4}, "X", 0u),
      arena.Make<VariableTerm>(TokenRange{6, 6}, "Y", 1u)};
  const Term *q{arena.Make<CompoundTerm>(TokenRange{2, 7},
      arena.Make<NameTerm>(TokenRange{2, 2}, "q"), arena.Save(args),
      CompoundForm::Canonical)};
  EXPECT_EQ(GetSource(*q, tokens), "q(X, Y)");
}

} // namespace
