//===-- unittests/Parser/TermReaderTest.cpp ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "merkurius/Parser/lexer.h"
#include "merkurius/Parser/message.h"
#include "merkurius/Parser/provenance.h"
#include "merkurius/Parser/source.h"
#include "merkurius/Parser/term-reader.h"
#include "llvm/Support/MemoryBuffer.h"
#include <string>
#include <vector>

using namespace merkurius::parser;

namespace {

struct ReadResult {
  std::vector<std::string> items; // canonical, or "error"
  std::vector<std::string> messages;
  std::vector<unsigned> variables;
};

ReadResult Read(llvm::StringRef text, bool openMP = false) {
  AllSources all;
  SourceFile *file{
      all.AddBuffer("test.m", llvm::MemoryBuffer::getMemBufferCopy(text))};
  Messages messages;
  LexerOptions options;
  options.openMP = openMP;
  TokenSequence tokens{Lex(*file, options, messages)};
  TermArena arena;
  TermReader reader{tokens, arena, messages};
  ReadResult result;
  while (std::optional<ItemTerm> item{reader.ReadItem()}) {
    if (item->term) {
      std::string out;
      llvm::raw_string_ostream os{out};
      PrintCanonical(os, *item->term);
      result.items.push_back(out);
    } else {
      result.items.push_back("error");
    }
    result.variables.push_back(item->variables);
    arena.Reset();
  }
  for (const Message &message : messages.messages()) {
    result.messages.push_back(message.ToString());
  }
  return result;
}

// The canonical form of a single item that reads without messages.
std::string Canonical(llvm::StringRef text) {
  ReadResult result{Read(text)};
  EXPECT_TRUE(result.messages.empty())
      << text.str() << ": " << result.messages.front();
  if (result.items.size() != 1) {
    return "items: " + std::to_string(result.items.size());
  }
  return result.items[0];
}

// The first message of a single item that does not read.
std::string Error(llvm::StringRef text) {
  ReadResult result{Read(text)};
  EXPECT_EQ(result.items.size(), 1u) << text.str();
  if (result.messages.empty()) {
    return "no message: " + (result.items.empty() ? "" : result.items[0]);
  }
  return result.messages.front();
}

TEST(TermReaderTest, Items) {
  EXPECT_EQ(Canonical(":- module m."), "':-'(module(m))");
  EXPECT_EQ(Canonical("p :- q, r."), "':-'(p, ','(q, r))");
  EXPECT_EQ(Canonical("p --> q."), "'-->'(p, q)");
  EXPECT_EQ(Canonical("p."), "p");
  ReadResult two{Read("a. b.\n")};
  EXPECT_EQ(two.items, (std::vector<std::string>{"a", "b"}));
  EXPECT_TRUE(Read("% only a comment\n").items.empty());
}

TEST(TermReaderTest, Associativity) {
  EXPECT_EQ(Canonical("X = a - b - c."), "'='(X, '-'('-'(a, b), c))");
  EXPECT_EQ(Canonical("p :- a, b, c."), "':-'(p, ','(a, ','(b, c)))");
  EXPECT_EQ(Canonical("X = A ^ B ^ C."), "'='(X, '^'(A, '^'(B, C)))");
  EXPECT_EQ(Error("X = a = b."), "expected an operator or '.', but found '='");
  EXPECT_EQ(
      Error("p :- q :- r."), "expected an operator or '.', but found ':-'");
}

TEST(TermReaderTest, Precedence) {
  EXPECT_EQ(Canonical("X = A + B * C."), "'='(X, '+'(A, '*'(B, C)))");
  EXPECT_EQ(Canonical("X = (A + B) * C."), "'='(X, '*'('+'(A, B), C))");
  EXPECT_EQ(Canonical("p :- a ; b, c."), "':-'(p, ';'(a, ','(b, c)))");
  EXPECT_EQ(Canonical("p :- ( if a then b else c )."),
      "':-'(p, else(if(then(a, b)), c))");
  EXPECT_EQ(
      Canonical("X = io.write_string(S)."), "'='(X, '.'(io, write_string(S)))");
  // In arguments ',' separates, so operators that bind less tightly than
  // ',' need no parentheses there.
  EXPECT_EQ(Canonical(":- pred p(int::in, io::di) is det."),
      "':-'(pred(is(p('::'(int, in), '::'(io, di)), det)))");
  EXPECT_EQ(Canonical("X = [A - B, (C, D)]."),
      "'='(X, '[|]'('-'(A, B), '[|]'(','(C, D), [])))");
}

TEST(TermReaderTest, PrefixOperators) {
  EXPECT_EQ(Canonical("p :- \\+ \\+ q."), "':-'(p, '\\\\+'('\\\\+'(q)))");
  EXPECT_EQ(
      Canonical("p :- some [X] q(X)."), "':-'(p, some('[|]'(X, []), q(X)))");
  EXPECT_EQ(Canonical("X = - Y."), "'='(X, '-'(Y))");
  EXPECT_EQ(Canonical("X = -(Y)."), "'='(X, '-'(Y))");
  // '-' is fx, so its operand must bind more tightly than it.
  EXPECT_EQ(Error("X = - - Y."),
      "the prefix operator '-' binds too loosely to be used here without "
      "parentheses");
}

TEST(TermReaderTest, NegativeNumbers) {
  // As in mmc, '-' before a number is its sign, whatever separates them,
  // and it binds more tightly than any operator.
  EXPECT_EQ(Canonical("X = -1."), "'='(X, -1)");
  EXPECT_EQ(Canonical("X = - 1."), "'='(X, -1)");
  EXPECT_EQ(Canonical("X = '-' 1."), "'='(X, -1)");
  EXPECT_EQ(Canonical("X = - /* c */\n 1.5."), "'='(X, -1.5)");
  EXPECT_EQ(Canonical("X = - 1 ^ 2."), "'='(X, '^'(-1, 2))");
  EXPECT_EQ(Canonical("X = - X ^ 2."), "'='(X, '-'('^'(X, 2)))");
  EXPECT_EQ(Canonical("X = a - - 1."), "'='(X, '-'(a, -1))");
  EXPECT_EQ(Canonical("X = - 0xff."), "'='(X, -255)");
  EXPECT_EQ(Canonical("X = -0."), "'='(X, 0)");
  EXPECT_EQ(Canonical("X = -0.0."), "'='(X, 0.0)");
  // Not a sign: an argument list, parentheses, and unsigned literals.
  EXPECT_EQ(Canonical("X = -(1)."), "'='(X, '-'(1))");
  EXPECT_EQ(Canonical("X = - (1)."), "'='(X, '-'(1))");
  EXPECT_EQ(Canonical("X = -1u."), "'='(X, '-'(1u))");
  EXPECT_EQ(Canonical("X = - 1u8."), "'='(X, '-'(1u8))");
  EXPECT_EQ(Canonical("X = Y-1."), "'='(X, '-'(Y, 1))");
  // A negative number cannot be applied; a positive or unsigned one can.
  EXPECT_EQ(Error("X = -1(a)."), "expected an operator or '.', but found '('");
  EXPECT_EQ(Canonical("X = - 1u(a)."), "'='(X, '-'(''(1u, a)))");
}

TEST(TermReaderTest, OperatorsAsConstants) {
  EXPECT_EQ(Canonical("X = f(+)."), "'='(X, f('+'))");
  EXPECT_EQ(Canonical("X = [-, *]."), "'='(X, '[|]'('-', '[|]'('*', [])))");
  EXPECT_EQ(Canonical("X = (:)."), "'='(X, ':')");
  EXPECT_EQ(Canonical("X = (':')."), "'='(X, ':')");
  EXPECT_EQ(Error("X = ':'."),
      "the operator ':' must be in parentheses to be used as an operand");
  EXPECT_EQ(Error("X = mod."),
      "the operator 'mod' must be in parentheses to be used as an operand");
}

TEST(TermReaderTest, Tuples) {
  // As in mmc, the contents are one term; its top-level commas, through
  // parentheses, separate the elements.
  EXPECT_EQ(Canonical("X = {a :- b, c}."), "'='(X, '{}'(':-'(a, ','(b, c))))");
  EXPECT_EQ(Canonical("X = {a ; b, c}."), "'='(X, '{}'(';'(a, ','(b, c))))");
  EXPECT_EQ(Canonical("X = {(a, b)}."), "'='(X, '{}'(a, b))");
  EXPECT_EQ(Canonical("X = {a, (b, c)}."), "'='(X, '{}'(a, b, c))");
  EXPECT_EQ(Canonical("X = {(a, b), c}."), "'='(X, '{}'(','(a, b), c))");
  EXPECT_EQ(Error("X = {a, \\+}."),
      "the operator '\\+' must be in parentheses to be used as an operand");
}

TEST(TermReaderTest, LeadingOperatorConstants) {
  // An infix-only operator may begin a term as a constant before another
  // infix operator; elsewhere it must be parenthesized.
  EXPECT_EQ(Canonical("; -> a."), "'->'(';', a)");
  EXPECT_EQ(Canonical("X = f(; + b, mod = a)."),
      "'='(X, f('+'(';', b), '='(mod, a)))");
  EXPECT_EQ(Canonical("X = (; , a)."), "'='(X, ','(';', a))");
  EXPECT_EQ(Error("a = ; + b."),
      "the operator ';' must be in parentheses to be used as an operand");
  EXPECT_EQ(Canonical("- , a."), "','('-', a)");
  EXPECT_EQ(Canonical("x = (\\+ , a)."), "'='(x, ','('\\\\+', a))");
  EXPECT_EQ(Canonical("; `f` a."), "f(';', a)");
  EXPECT_EQ(Error("- `f` a."), "expected an operator or '.', but found 'f'");
  EXPECT_EQ(Error("- = a."),
      "the operator '-' must be in parentheses to be used as an operand");
}

TEST(TermReaderTest, Compounds) {
  EXPECT_EQ(Canonical("f(a, B, \"s\")."), "f(a, B, \"s\")");
  EXPECT_EQ(Canonical("'+'(1, 2)."), "'+'(1, 2)");
  EXPECT_EQ(Canonical("X = [1, 2 | T]."), "'='(X, '[|]'(1, '[|]'(2, T)))");
  EXPECT_EQ(Canonical("X = []."), "'='(X, [])");
  EXPECT_EQ(Canonical("X = {a, B}."), "'='(X, '{}'(a, B))");
  EXPECT_EQ(Canonical("X = {}."), "'='(X, {})");
  EXPECT_EQ(Canonical("X = P(A, B)."), "'='(X, ''(P, A, B))");
  EXPECT_EQ(Canonical("X = (P)(A)."), "'='(X, ''(P, A))");
  EXPECT_EQ(Canonical("X = P(A)(B)."), "'='(X, ''(''(P, A), B))");
  // Any primary term may be applied; [] and {} then are names.
  EXPECT_EQ(Canonical("X = 1(a)."), "'='(X, ''(1, a))");
  EXPECT_EQ(Canonical("X = \"s\"(a)."), "'='(X, ''(\"s\", a))");
  EXPECT_EQ(Canonical("X = [a](b)."), "'='(X, ''('[|]'(a, []), b))");
  EXPECT_EQ(Canonical("X = {a}(b)."), "'='(X, ''('{}'(a), b))");
  EXPECT_EQ(Canonical("X = $line(a)."), "'='(X, ''($line, a))");
  EXPECT_EQ(Canonical("X = [](a)."), "'='(X, '[]'(a))");
  EXPECT_EQ(Canonical("X = {}(a)."), "'='(X, '{}'(a))");
  // With whitespace before '(', it is an operand, not an argument list.
  EXPECT_EQ(Error("X = f (a)."), "expected an operator or '.', but found '('");
}

TEST(TermReaderTest, Backquotes) {
  EXPECT_EQ(Canonical("X = A `max` B."), "'='(X, max(A, B))");
  EXPECT_EQ(Canonical("X = A `int.max` B."), "'='(X, '.'(int, max(A, B)))");
  EXPECT_EQ(Canonical("X = A `F` B."), "'='(X, ''(F, A, B))");
  EXPECT_EQ(Canonical("X = A `mod` B."), "'='(X, mod(A, B))");
  EXPECT_EQ(Canonical("X = A `f` B `g` C."), "'='(X, g(f(A, B), C))");
  EXPECT_EQ(Canonical("X = A `f` B + C."), "'='(X, '+'(f(A, B), C))");
}

TEST(TermReaderTest, BackquoteAsName) {
  // As in mmc, a backquote where a term begins is a name.
  EXPECT_EQ(Canonical("x = `."), "'='(x, '`')");
  EXPECT_EQ(Canonical("x = f(-, `)."), "'='(x, f('-', '`'))");
  EXPECT_EQ(Canonical("x = - `."), "'='(x, '-'('`'))");
  EXPECT_EQ(Canonical("a `f` `."), "f(a, '`')");
  // A backquote is an operator only before a name and another backquote.
  EXPECT_EQ(Canonical("some [X] ` - 1."), "some('[|]'(X, []), '-'('`', 1))");
  EXPECT_EQ(Canonical("some [X] `f` Y p."), "some(f('[|]'(X, []), Y), p)");
}

TEST(TermReaderTest, Literals) {
  EXPECT_EQ(Canonical("X = 1_000."), "'='(X, 1000)");
  EXPECT_EQ(Canonical("X = 0xff_u8."), "'='(X, 255u8)");
  EXPECT_EQ(Canonical("X = 0b1010."), "'='(X, 10)");
  EXPECT_EQ(Canonical("X = 0o17."), "'='(X, 15)");
  EXPECT_EQ(Canonical("X = 0'a."), "'='(X, 97)");
  EXPECT_EQ(Canonical("X = 7u."), "'='(X, 7u)");
  EXPECT_EQ(Canonical("X = -128i8."), "'='(X, -128i8)");
  EXPECT_EQ(
      Canonical("X = -9223372036854775808."), "'='(X, -9223372036854775808)");
  EXPECT_EQ(
      Canonical("X = 18446744073709551615u."), "'='(X, 18446744073709551615u)");
  EXPECT_EQ(Canonical("X = 1.5e10."), "'='(X, 1.5e+10)");
  EXPECT_EQ(Canonical("X = 1_000.5."), "'='(X, 1000.5)");
  // As in mmc, the reader does not check ranges; the converter does.
  EXPECT_EQ(Canonical("X = 128i8."), "'='(X, 128i8)");
  EXPECT_EQ(
      Canonical("X = 18446744073709551616u."), "'='(X, 18446744073709551616u)");
  EXPECT_EQ(Canonical("X = 0x1_0000_0000_0000_0000_0000_0000."),
      "'='(X, 79228162514264337593543950336)");
  EXPECT_EQ(Canonical("X = 1.0e999."), "'='(X, 1.0e+309)");
}

TEST(TermReaderTest, QuotedText) {
  EXPECT_EQ(UnquoteSpelling("'it''s'"), "it's");
  EXPECT_EQ(UnquoteSpelling(R"('it\'s')"), "it's");
  EXPECT_EQ(UnquoteSpelling(R"("a\nb\t\\")"), "a\nb\t\\");
  EXPECT_EQ(UnquoteSpelling(R"("say ""hi""")"), "say \"hi\"");
  EXPECT_EQ(UnquoteSpelling(R"("\x41\\101\")"), "AA");
  EXPECT_EQ(UnquoteSpelling(R"("é")"), "\xc3\xa9");
  EXPECT_EQ(UnquoteSpelling(R"("\U0001F600")"), "\xf0\x9f\x98\x80");
  EXPECT_EQ(UnquoteSpelling("\"one \\\ntwo\""), "one two");
  EXPECT_EQ(UnquoteSpelling("plain"), "plain");
  EXPECT_EQ(Canonical("X = 'hello world'."), "'='(X, 'hello world')");
  EXPECT_EQ(Canonical("X = 'abc'."), "'='(X, abc)");
}

TEST(TermReaderTest, Variables) {
  ReadResult result{Read("p(X, Y, X, _, _, _Z, _Z).\nq(A).\n")};
  ASSERT_EQ(result.variables.size(), 2u);
  // X, Y, two anonymous variables and _Z; numbering restarts per item.
  EXPECT_EQ(result.variables[0], 5u);
  EXPECT_EQ(result.variables[1], 1u);
}

TEST(TermReaderTest, Directives) {
  ReadResult result{Read("p :-\n"
                         "  %$omp parallel\n"
                         "  ( a & b ), c.\n",
      /*openMP=*/true)};
  EXPECT_TRUE(result.messages.empty());
  ASSERT_EQ(result.items.size(), 1u);
  // The directive annotates one conjunct.
  EXPECT_EQ(result.items[0], "':-'(p, ','('%$omp'('&'(a, b)), c))");
}

TEST(TermReaderTest, Recovery) {
  ReadResult result{Read("a :- = b.\n"
                         "ok.\n"
                         "f(a, .\n"
                         "also_ok.\n"
                         "unterminated(")};
  EXPECT_EQ(result.items,
      (std::vector<std::string>{"error", "ok", "error", "also_ok", "error"}));
  EXPECT_EQ(result.messages,
      (std::vector<std::string>{
          "the operator '=' must be in parentheses to be used as an operand",
          "expected a term, but found '.'",
          "expected a term, but found the end of the file"}));
}

TEST(TermReaderTest, DepthLimit) {
  AllSources all;
  SourceFile *file{all.AddBuffer("test.m",
      llvm::MemoryBuffer::getMemBufferCopy("a :- b, c.\n"
                                           "a :- b, c, d.\n"
                                           "x = f(y).\n"
                                           "x = f(g(y)).\n"
                                           "ok.\n"))};
  Messages messages;
  TokenSequence tokens{Lex(*file, LexerOptions{}, messages)};
  TermArena arena;
  // A level for the item, then one for each operand and argument within
  // it: "a :- b, c" and "x = f(y)" need three.
  TermReader reader{tokens, arena, messages, /*maxDepth=*/3};
  std::vector<bool> read;
  while (std::optional<ItemTerm> item{reader.ReadItem()}) {
    read.push_back(item->term != nullptr);
  }
  EXPECT_EQ(read, (std::vector<bool>{true, false, true, false, true}));
  ASSERT_EQ(messages.messages().size(), 2u);
  EXPECT_EQ(messages.messages().front().ToString(),
      "the term nests more than 3 levels deep");
}

} // namespace
