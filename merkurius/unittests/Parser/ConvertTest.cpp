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
  EXPECT_EQ(Messages_(":- type t ---> t.\n"),
      std::vector<std::string>{"the ':- type' declaration"});
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

// Items() for a dump with the lines that say a field is empty removed:
// "x: none", and lists with nothing after them.
std::string Compact(llvm::StringRef text) {
  std::string items{Items(text)};
  llvm::SmallVector<llvm::StringRef, 32> lines;
  llvm::StringRef{items}.split(lines, '\n', -1, false);
  std::string out;
  for (std::size_t j{0}; j < lines.size(); ++j) {
    llvm::StringRef line{lines[j]};
    if (line.ends_with(": none")) {
      continue;
    }
    if (line.ends_with(":")) {
      // A list is empty if the next line is no deeper.
      std::size_t depth{line.find_first_not_of("| ")};
      if (j + 1 == lines.size() ||
          lines[j + 1].find_first_not_of("| ") <= depth) {
        continue;
      }
    }
    out += line.str() + "\n";
  }
  return out;
}

TEST(ConvertTest, PredDecls) {
  EXPECT_EQ(Compact(":- pred p(int::in, list(T)::out) is det.\n"),
      "PredDecl\n"
      "| kind: Predicate\n"
      "| purity: Pure\n"
      "| name: SymName p\n"
      "| arguments:\n"
      "| | TypeAndMode\n"
      "| | | type: NamedType\n"
      "| | | | name: SymName int\n"
      "| | | mode: NamedMode\n"
      "| | | | name: SymName in\n"
      "| | TypeAndMode\n"
      "| | | type: NamedType\n"
      "| | | | name: SymName list\n"
      "| | | | arguments:\n"
      "| | | | | TypeVar T\n"
      "| | | mode: NamedMode\n"
      "| | | | name: SymName out\n"
      "| determinism: Det\n");
  EXPECT_EQ(Compact(":- func f = int.\n"),
      "PredDecl\n"
      "| kind: Function\n"
      "| purity: Pure\n"
      "| name: SymName f\n"
      "| result: TypeAndMode\n"
      "| | type: NamedType\n"
      "| | | name: SymName int\n");
  // '=>' binds less tightly than '<=', so the manual's example has
  // parentheses ("Existential class constraints").
  EXPECT_EQ(Compact(":- all [T] some [U] ((impure pred p(T, U) => c(U)) <= "
                    "(d(T), I =< ground)).\n"),
      "PredDecl\n"
      "| kind: Predicate\n"
      "| universalVars:\n"
      "| | T\n"
      "| existentialVars:\n"
      "| | U\n"
      "| purity: Impure\n"
      "| name: SymName p\n"
      "| arguments:\n"
      "| | TypeAndMode\n"
      "| | | type: TypeVar T\n"
      "| | TypeAndMode\n"
      "| | | type: TypeVar U\n"
      "| existentialConstraints:\n"
      "| | ClassConstraint\n"
      "| | | name: SymName c\n"
      "| | | arguments:\n"
      "| | | | TypeVar U\n"
      "| universalConstraints:\n"
      "| | ClassConstraint\n"
      "| | | name: SymName d\n"
      "| | | arguments:\n"
      "| | | | TypeVar T\n"
      "| | InstConstraint\n"
      "| | | var: I\n"
      "| | | inst: BuiltinInst\n"
      "| | | | kind: Ground\n");
  EXPECT_EQ(Compact(":- pred p `with_type` t `with_inst` i.\n"),
      Compact(":- pred p : t `with_inst` i.\n"));
}

TEST(ConvertTest, Types) {
  EXPECT_EQ(Compact(":- pred p({int, T}).\n"),
      "PredDecl\n| kind: Predicate\n| purity: Pure\n| name: SymName p\n"
      "| arguments:\n| | TypeAndMode\n| | | type: TupleType\n"
      "| | | | elements:\n| | | | | NamedType\n| | | | | | name: SymName int\n"
      "| | | | | TypeVar T\n");
  // Higher-order types, with modes and a determinism when they are
  // combined with an inst.
  std::string p{Compact(
      ":- pred p(semipure func(int::in) = (int::out) is semidet).\n")};
  EXPECT_NE(p.find("| | | type: HigherOrderType\n"
                   "| | | | purity: Semipure\n"
                   "| | | | kind: Function\n"),
      std::string::npos)
      << p;
  EXPECT_NE(p.find("| | | | determinism: Semidet\n"), std::string::npos) << p;
  EXPECT_NE(Compact(":- pred p((pred)).\n").find("kind: Predicate"),
      std::string::npos);
  EXPECT_NE(Compact(":- pred p(io.state, io__state).\n")
                .find("SymName io.state\n| | TypeAndMode\n| | | type: "
                      "NamedType\n| | | | name: SymName io.state"),
      std::string::npos);
}

TEST(ConvertTest, InstsAndModes) {
  std::string p{Compact(":- mode p(free >> bound(a ; [ground | I])) is det.\n")};
  EXPECT_EQ(p,
      "ModeDecl\n"
      "| kind: Predicate\n"
      "| name: SymName p\n"
      "| arguments:\n"
      "| | MappingMode\n"
      "| | | initial: BuiltinInst\n"
      "| | | | kind: Free\n"
      "| | | final: BoundInst\n"
      "| | | | unique: false\n"
      "| | | | functors:\n"
      "| | | | | BoundFunctor\n"
      "| | | | | | name: SymName a\n"
      "| | | | | BoundFunctor\n"
      "| | | | | | name: SymName [|]\n"
      "| | | | | | arguments:\n"
      "| | | | | | | BuiltinInst\n"
      "| | | | | | | | kind: Ground\n"
      "| | | | | | | InstVar\n"
      "| | | | | | | | name: I\n"
      "| determinism: Det\n");
  EXPECT_EQ(Compact(":- mode f(in(I =< ground)) = (pred(out) is det).\n"),
      "ModeDecl\n"
      "| kind: Function\n"
      "| name: SymName f\n"
      "| arguments:\n"
      "| | NamedMode\n"
      "| | | name: SymName in\n"
      "| | | arguments:\n"
      "| | | | InstVar\n"
      "| | | | | name: I\n"
      "| | | | | constraint: BuiltinInst\n"
      "| | | | | | kind: Ground\n"
      "| result: HigherOrderInst\n"
      "| | kind: Predicate\n"
      "| | any: false\n"
      "| | arguments:\n"
      "| | | NamedMode\n"
      "| | | | name: SymName out\n"
      "| | determinism: Det\n");
  EXPECT_EQ(Compact(":- (mode p(in(I)) is det) <= I =< ground.\n"),
      "ModeDecl\n| kind: Predicate\n| name: SymName p\n| arguments:\n"
      "| | NamedMode\n| | | name: SymName in\n| | | arguments:\n"
      "| | | | InstVar\n| | | | | name: I\n| determinism: Det\n"
      "| constraints:\n| | InstConstraint\n| | | var: I\n"
      "| | | inst: BuiltinInst\n| | | | kind: Ground\n");
}

// Declarations that mmc accepts convert without messages.
TEST(ConvertTest, AcceptedDeclarations) {
  for (const char *text : {
           ":- pred p1(T::in) is det <= c(T).",
           ":- impure (pred p1(int::in) is det).",
           ":- some [T] pred p1(T::out) is det => c(T).",
           ":- all [T] impure pred p1(T::in) is det.",
           ":- pred p1(int) : fp(int, int).",
           ":- (pred p1(T::in(I)) is det) <= (I =< ground).",
           ":- pred p1((pred)::((pred) is det)) is det.",
           ":- pred p1('{}'::in) is det.",
           ":- pred p1(int::in, int::in) is cc_multi.",
           ":- pred t.p1(int::in) is cc_nondet.",
           ":- func (f1) = int.",
           ":- func f1(int::in) = (int::out) is erroneous.",
           ":- pred p1(int::any_pred(in) is failure) is multi.",
           ":- pred p(io::di, io::uo) is nondet.",
           ":- mode length(in(listskel)) = out is det.",
           ":- mode p(unique >> clobbered, mostly_unique >> mostly_clobbered).",
           ":- mode p(any >> unique_any, mostly_unique_any >> clobbered_any).",
           ":- mode p(free >> mostly_clobbered_any, ground >> not_reached).",
           ":- mode p(free >> bound_unique(f(ground))).",
           ":- mode p(free >> (any_func(in) = out is det)).",
       }) {
    EXPECT_EQ(Messages_(text), std::vector<std::string>{}) << text;
  }
}

TEST(ConvertTest, DeclarationErrors) {
  using V = std::vector<std::string>;
  EXPECT_EQ(Messages_(":- pred p1(int::in, int)."),
      V{"either every argument or none must have a mode"});
  EXPECT_EQ(Messages_(":- func f1(int) = (int::out)."),
      V{"either every argument and the result or none must have a mode"});
  EXPECT_EQ(Messages_(":- func f1(int)."),
      V{"expected a function's arguments and result, as in 'f(int) = int', "
        "but found 'f1(int)'"});
  EXPECT_EQ(Messages_(":- pred p1(int::X)."),
      V{"expected a mode, but found 'X'"});
  EXPECT_EQ(Messages_(":- pred p1(int::in) is det2."),
      V{"expected a determinism category, but found 'det2'"});
  EXPECT_EQ(Messages_(":- pred p1(1::in, \"int\")."),
      (V{"expected a type, but found '1'",
          "expected a type, but found '\"int\"'"}));
  EXPECT_EQ(Messages_(":- mode p(free >> bound(1 ; 2))."),
      (V{"expected a function symbol, but found '1'",
          "expected a function symbol, but found '2'"}));
  EXPECT_EQ(Messages_(":- all [T, 1] pred p(T)."),
      V{"expected a variable, but found '1'"});
  EXPECT_EQ(Messages_(":- all T pred p(T)."),
      V{"expected a list of variables, but found 'T'"});
  EXPECT_EQ(Messages_(":- pred X.p(int)."),
      V{"expected a module name, but found 'X'"});
  EXPECT_EQ(Messages_(":- pred p(T) <= 1."),
      V{"expected a constraint, but found '1'"});
  EXPECT_EQ(Messages_(":- mode p(in) `with_type` t."),
      V{"expected a mode declaration without a type, but found 't'"});
  // mmc too reads the layers in this order only.
  EXPECT_EQ(Messages_(":- some [T] (pred p1(T::out) => c(T)) is det."),
      V{"expected a declaration, but found 'some [T] (pred p1(T::out) => "
        "c(T)) is de...'"});
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
