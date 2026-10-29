//===-- unittests/Parser/ConvertTest.cpp ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "merkurius/Parser/convert.h"
#include "merkurius/Parser/lexer.h"
#include "merkurius/Parser/message.h"
#include "merkurius/Parser/provenance.h"
#include "merkurius/Parser/source.h"
#include "merkurius/Parser/term-combinators.h"
#include "merkurius/Parser/term-reader.h"
#include "llvm/Support/MemoryBuffer.h"
#include <string>
#include <vector>

using namespace merkurius::parser;

namespace {

struct ConvertResult {
  std::string tree; // DumpParseTree's output
  std::vector<std::string> messages;
};

ConvertResult Convert(llvm::StringRef text) {
  AllSources all;
  SourceFile *file{
      all.AddBuffer("test.m", llvm::MemoryBuffer::getMemBufferCopy(text))};
  Messages messages;
  TokenSequence tokens{Lex(*file, LexerOptions{}, messages)};
  ParseTreeArena arena;
  const ParseTree &tree{ConvertFile(tokens, arena, messages)};
  ConvertResult result;
  llvm::raw_string_ostream os{result.tree};
  DumpParseTree(os, tree);
  for (const Message &message : messages.messages()) {
    result.messages.push_back(message.ToString());
  }
  return result;
}

// The dump of the tree without its first two lines, "ParseTree" and
// "| items:", and with each line's two levels of indentation removed.
std::string Items(llvm::StringRef text) {
  ConvertResult result{Convert(text)};
  EXPECT_EQ(result.messages, std::vector<std::string>{}) << text.str();
  llvm::SmallVector<llvm::StringRef, 16> lines;
  llvm::StringRef{result.tree}.split(lines, '\n', -1, false);
  std::string out;
  for (llvm::StringRef line : llvm::ArrayRef(lines).drop_front(2)) {
    line.consume_front("| | ");
    out += line.str() + "\n";
  }
  return out;
}

std::vector<std::string> Messages_(llvm::StringRef text) {
  return Convert(text).messages;
}

TEST(ConvertTest, ModuleSystem) {
  EXPECT_EQ(Items(":- module m.\n"), "ModuleDecl\n| name: SymName m\n");
  EXPECT_EQ(Items(":- end_module m.\n"), "EndModuleDecl\n| name: SymName m\n");
  EXPECT_EQ(Items(":- interface.\n:- implementation.\n"),
      "SectionDecl\n| section: Interface\n"
      "SectionDecl\n| section: Implementation\n");
  EXPECT_EQ(Items(":- import_module io, list.\n"),
      "ImportDecl\n| kind: ImportModule\n| modules:\n"
      "| | SymName io\n| | SymName list\n");
  EXPECT_EQ(Items(":- use_module (a, b), c.\n"),
      "ImportDecl\n| kind: UseModule\n| modules:\n"
      "| | SymName a\n| | SymName b\n| | SymName c\n");
  EXPECT_EQ(Items(":- include_module x.\n"),
      "ImportDecl\n| kind: IncludeModule\n| modules:\n| | SymName x\n");
}

TEST(ConvertTest, QualifiedNames) {
  const char *abc{"ModuleDecl\n| name: SymName a.b.c\n"};
  EXPECT_EQ(Items(":- module a.b.c.\n"), abc);
  EXPECT_EQ(Items(":- module a__b__c.\n"), abc);
  EXPECT_EQ(Items(":- module a.b__c.\n"), abc);
  EXPECT_EQ(Items(":- module 'a'.(b.c).\n"), abc);
  EXPECT_EQ(Items(":- module (a.b.c).\n"), abc);
  // '__' splits only between two nonempty parts.
  EXPECT_EQ(Items(":- module 'a__'.\n"), "ModuleDecl\n| name: SymName a__\n");
  EXPECT_EQ(
      Items(":- module 'a____b'.\n"), "ModuleDecl\n| name: SymName a____b\n");
}

TEST(ConvertTest, Errors) {
  EXPECT_EQ(Messages_(":- module X.\n"),
      std::vector<std::string>{"expected a module name, but found 'X'"});
  // Every bad element of a list is reported.
  EXPECT_EQ(Messages_(":- import_module io, \"s\", 1.\n"),
      (std::vector<std::string>{"expected a module name, but found '\"s\"'",
          "expected a module name, but found '1'"}));
  EXPECT_EQ(Messages_(":- module a.X.\n"),
      std::vector<std::string>{"expected a name, but found 'X'"});
  EXPECT_EQ(Messages_(":- foo.\n"),
      std::vector<std::string>{"expected a declaration, but found 'foo'"});
  EXPECT_EQ(Messages_(":- module(a, b).\n"),
      std::vector<std::string>{
          "expected a declaration, but found 'module(a, b)'"});
  EXPECT_EQ(Messages_(":- pred p(int::in).\n"),
      std::vector<std::string>{
          "the ':- pred' declaration"});
  EXPECT_EQ(Messages_("p :- q.\n"),
      std::vector<std::string>{"clauses"});
  // Long terms and terms over several lines are shortened.
  EXPECT_EQ(Messages_(":- foo(aaaaaaaaaa, bbbbbbbbbb, cccccccccc, dddddddddd).\n"
                      ":- bar(a,\n  b).\n"),
      (std::vector<std::string>{
          "expected a declaration, but found 'foo(aaaaaaaaaa, bbbbbbbbbb, "
          "cccccccccc,...'",
          "expected a declaration, but found 'bar(a,...'"}));
}

TEST(ConvertTest, Recovery) {
  // A bad item is left out, and the items around it are kept.
  ConvertResult result{Convert(":- module m.\n:- foo.\n:- interface.\n"
                               ":- import_module ).\n:- implementation.\n")};
  EXPECT_EQ(result.messages.size(), 2u);
  EXPECT_EQ(result.tree,
      "ParseTree\n| items:\n"
      "| | ModuleDecl\n| | | name: SymName m\n"
      "| | SectionDecl\n| | | section: Interface\n"
      "| | SectionDecl\n| | | section: Implementation\n");
}

// The combinators on their own, with the expected forms they describe.
TEST(ConvertTest, Combinators) {
  constexpr auto ab{anyOf(functor("a"), functor("b"))};
  constexpr auto abc{anyOf(functor("a"), functor("b"), functor("c", ab))};
  std::string text;
  llvm::raw_string_ostream os{text};
  abc.Expected(os);
  EXPECT_EQ(text, "'a', 'b' or 'c'/1");
}

// Walk visits every node, calling Pre and Post where the visitor has them.
TEST(ConvertTest, Walk) {
  AllSources all;
  SourceFile *file{all.AddBuffer("test.m",
      llvm::MemoryBuffer::getMemBufferCopy(
          ":- module m.\n:- import_module a.b, c.\n:- interface.\n"))};
  Messages messages;
  TokenSequence tokens{Lex(*file, LexerOptions{}, messages)};
  ParseTreeArena arena;
  const ParseTree &tree{ConvertFile(tokens, arena, messages)};
  struct Visitor {
    std::vector<std::string> names;
    int sections{0};
    bool Pre(const ModuleDecl &) { return false; } // skips its name
    void Post(const SymName &name) {
      std::string text;
      llvm::raw_string_ostream os{text};
      name.PrintInline(os);
      names.push_back(text);
    }
    void Post(const SectionDecl &) { ++sections; }
  } visitor;
  Walk(tree, visitor);
  EXPECT_EQ(visitor.names, (std::vector<std::string>{"a.b", "c"}));
  EXPECT_EQ(visitor.sections, 1);
}

} // namespace
