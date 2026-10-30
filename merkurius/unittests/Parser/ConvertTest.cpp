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

ConvertResult Convert(llvm::StringRef text, bool openMP = false) {
  AllSources all;
  SourceFile *file{
      all.AddBuffer("test.m", llvm::MemoryBuffer::getMemBufferCopy(text))};
  Messages messages;
  TokenSequence tokens{Lex(*file, LexerOptions{openMP}, messages)};
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
  EXPECT_EQ(Messages_(":- pragma mode_check_clauses(p/1).\n"),
      std::vector<std::string>{
          "the ':- pragma mode_check_clauses' declaration"});
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
  EXPECT_EQ(Messages_(":- mode p(free >> bound(X ; f(Y)))."),
      V{"expected a function symbol, but found 'X'"});
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

TEST(ConvertTest, TypeDefinitions) {
  // where applies to all the constructors, not only the last.
  EXPECT_EQ(Items(":- type t ---> a ; b where equality is eq.\n"),
      "DuTypeDefn\n| name: SymName t\n| parameters:\n| supertype: none\n"
      "| constructors:\n| | Constructor\n| | | existentialVars:\n"
      "| | | name: SymName a\n| | | arguments:\n| | | constraints:\n"
      "| | Constructor\n| | | existentialVars:\n| | | name: SymName b\n"
      "| | | arguments:\n| | | constraints:\n| attributes: TypeAttributes\n"
      "| | equality: SymName eq\n| | comparison: none\n| | directArgs:\n"
      "| | abstractNoncanonical: false\n");
  EXPECT_EQ(Compact(":- type fruit ---> apple ; orange.\n"),
      "DuTypeDefn\n| name: SymName fruit\n| constructors:\n"
      "| | Constructor\n| | | name: SymName apple\n"
      "| | Constructor\n| | | name: SymName orange\n");
  EXPECT_EQ(Compact(":- type e ---> e(name :: string, int).\n"),
      "DuTypeDefn\n| name: SymName e\n| constructors:\n"
      "| | Constructor\n| | | name: SymName e\n| | | arguments:\n"
      "| | | | ConstructorArg\n| | | | | field: SymName name\n"
      "| | | | | type: NamedType\n| | | | | | name: SymName string\n"
      "| | | | ConstructorArg\n"
      "| | | | | type: NamedType\n| | | | | | name: SymName int\n");
  // Lists, operators and quoting in braces (reference manual,
  // "Discriminated unions").
  std::string list{Compact(":- type list(T) ---> [] ; [T | list(T)].\n")};
  EXPECT_NE(list.find("| parameters:\n| | T\n"), std::string::npos) << list;
  EXPECT_NE(list.find("name: SymName []\n"), std::string::npos) << list;
  EXPECT_NE(list.find("name: SymName [|]\n"), std::string::npos) << list;
  EXPECT_NE(Compact(":- type pair(A, B) ---> A - B.\n").find("SymName -\n"),
      std::string::npos);
  std::string tricky{Compact(":- type tricky ---> { int ; int } ; { { int } }.\n")};
  EXPECT_NE(tricky.find("| | | name: SymName ;\n"), std::string::npos) << tricky;
  EXPECT_NE(tricky.find("| | | name: SymName {}\n"), std::string::npos) << tricky;
  EXPECT_EQ(Compact(":- type t ---> (a ; b) ; c.\n"),
      Compact(":- type t ---> a ; b ; c.\n"));
  // where binds more tightly than ';', so it is on the last constructor.
  std::string set{Compact(":- type set(T) ---> a ; set(list(T)) where "
                          "equality is eq, comparison is cmp.\n")};
  EXPECT_NE(set.find("| attributes: TypeAttributes\n"
                     "| | equality: SymName eq\n"
                     "| | comparison: SymName cmp\n"),
      std::string::npos)
      << set;
  EXPECT_NE(
      Compact(":- type t ---> f(int) where direct_arg is [f/1, g/0].\n")
          .find("| | directArgs:\n| | | NameAndArity f/1\n"
                "| | | NameAndArity g/0\n"),
      std::string::npos);
  // Existential constructors (reference manual, "Existentially typed data
  // types").
  std::string bar{Compact(":- type bar ---> some [T1, T2] f4(T1, T2) => "
                          "(s(T1), s(T2)) ; some [T] (s(T) => s(T)).\n")};
  EXPECT_NE(bar.find("| | Constructor\n| | | existentialVars:\n| | | | T1\n"
                     "| | | | T2\n| | | name: SymName f4\n"),
      std::string::npos)
      << bar;
  EXPECT_NE(bar.find("| | Constructor\n| | | existentialVars:\n| | | | T\n"
                     "| | | name: SymName s\n"),
      std::string::npos)
      << bar;
  EXPECT_EQ(Compact(":- type citrus =< fruit ---> lemon.\n"),
      "DuTypeDefn\n| name: SymName citrus\n| supertype: NamedType\n"
      "| | name: SymName fruit\n| constructors:\n| | Constructor\n"
      "| | | name: SymName lemon\n");
  EXPECT_EQ(Compact(":- type money == int.\n"),
      "EquivalenceTypeDefn\n| name: SymName money\n| type: NamedType\n"
      "| | name: SymName int\n");
  EXPECT_EQ(Compact(":- type t2(T1, T2).\n"),
      "AbstractTypeDecl\n| name: SymName t2\n| parameters:\n| | T1\n| | T2\n"
      "| solver: false\n");
  EXPECT_EQ(Compact(":- type t where type_is_abstract_noncanonical.\n"),
      "AbstractTypeDecl\n| name: SymName t\n| attributes: TypeAttributes\n"
      "| | abstractNoncanonical: true\n| solver: false\n");
}

TEST(ConvertTest, InstAndModeDefinitions) {
  EXPECT_EQ(Compact(":- inst listskel == bound([] ; [free | listskel]).\n"),
      Compact(":- inst listskel ---> [] ; [free | listskel].\n"));
  EXPECT_EQ(Compact(":- inst maybeskel(I) for maybe/1 ---> no ; yes(I).\n"),
      "InstDefn\n| name: SymName maybeskel\n| parameters:\n| | I\n"
      "| forType: NameAndArity maybe/1\n| inst: BoundInst\n"
      "| | unique: false\n| | functors:\n| | | BoundFunctor\n"
      "| | | | name: SymName no\n| | | BoundFunctor\n"
      "| | | | name: SymName yes\n| | | | arguments:\n| | | | | InstVar\n"
      "| | | | | | name: I\n");
  EXPECT_EQ(Compact(":- inst digit ---> 0 ; 0'1 ; \"two\" ; 3.0.\n"),
      "InstDefn\n| name: SymName digit\n| inst: BoundInst\n"
      "| | unique: false\n| | functors:\n"
      "| | | BoundFunctor\n| | | | name: Literal 0\n"
      "| | | BoundFunctor\n| | | | name: Literal 49\n"
      "| | | BoundFunctor\n| | | | name: Literal \"two\"\n"
      "| | | BoundFunctor\n| | | | name: Literal 3.0\n");
  EXPECT_EQ(Compact(":- mode in(I) == I >> I.\n"),
      "ModeDefn\n| name: SymName in\n| parameters:\n| | I\n"
      "| mode: MappingMode\n| | initial: InstVar\n| | | name: I\n"
      "| | final: InstVar\n| | | name: I\n");
  EXPECT_EQ(Compact(":- mode (+) == in.\n"),
      "ModeDefn\n| name: SymName +\n| mode: NamedMode\n| | name: SymName in\n");
}

TEST(ConvertTest, DefinitionErrors) {
  using V = std::vector<std::string>;
  EXPECT_EQ(Messages_(":- type t(int) ---> a."),
      V{"expected a type parameter, but found 'int'"});
  EXPECT_EQ(Messages_(":- type t(T, T) ---> a."),
      V{"the parameter T appears more than once"});
  EXPECT_EQ(Messages_(":- inst i(I, I) == I."),
      V{"the parameter I appears more than once"});
  EXPECT_EQ(Messages_(":- type t ---> 1 ; \"s\" ; X."),
      (V{"expected a constructor, but found '1'",
          "expected a constructor, but found '\"s\"'",
          "expected a constructor, but found 'X'"}));
  EXPECT_EQ(Messages_(":- type t ---> f(X :: int)."),
      V{"expected a field name, but found 'X'"});
  EXPECT_EQ(Messages_(":- type t ---> a where equality is e, equality is e."),
      V{"the type attribute 'equality' is given more than once"});
  EXPECT_EQ(Messages_(":- type t ---> a where foo."),
      V{"expected 'equality is', 'comparison is', 'direct_arg is' or "
        "'type_is_abstract_noncanonical', but found 'foo'"});
  EXPECT_EQ(Messages_(":- type t ---> f(int) where direct_arg is [f/x]."),
      V{"expected an arity, but found 'x'"});
  EXPECT_EQ(Messages_(":- type t == int where equality is e."),
      V{"an equivalence type cannot have 'where' attributes"});
  EXPECT_EQ(Messages_(":- inst i ---> 256u8 ; -129i8."),
      (V{"the integer '256u8' is too large for its type",
          "the integer '-129i8' is too large for its type"}));
  EXPECT_EQ(Messages_(":- inst i(int) == ground."),
      V{"expected an inst parameter, but found 'int'"});
  EXPECT_EQ(Messages_(":- mode m == X."), V{"expected a mode, but found 'X'"});
}

TEST(ConvertTest, Typeclasses) {
  EXPECT_EQ(Compact(":- typeclass point(T) where [pred coords(T, float), "
                    "mode coords(in, out) is det, func f(T) = T].\n"),
      "TypeclassDecl\n| name: SymName point\n| parameters:\n| | T\n"
      "| abstract: false\n| methods:\n"
      "| | PredDecl\n| | | kind: Predicate\n| | | purity: Pure\n"
      "| | | name: SymName coords\n| | | arguments:\n"
      "| | | | TypeAndMode\n| | | | | type: TypeVar T\n"
      "| | | | TypeAndMode\n| | | | | type: NamedType\n"
      "| | | | | | name: SymName float\n"
      "| | ModeDecl\n| | | kind: Predicate\n| | | name: SymName coords\n"
      "| | | arguments:\n| | | | NamedMode\n| | | | | name: SymName in\n"
      "| | | | NamedMode\n| | | | | name: SymName out\n"
      "| | | determinism: Det\n"
      "| | PredDecl\n| | | kind: Function\n| | | purity: Pure\n"
      "| | | name: SymName f\n| | | arguments:\n| | | | TypeAndMode\n"
      "| | | | | type: TypeVar T\n| | | result: TypeAndMode\n"
      "| | | | type: TypeVar T\n");
  EXPECT_EQ(Compact(":- typeclass a(T1, T2).\n"),
      "TypeclassDecl\n| name: SymName a\n| parameters:\n| | T1\n| | T2\n"
      "| abstract: true\n");
  EXPECT_EQ(Compact(":- typeclass foo(T) where [].\n"),
      "TypeclassDecl\n| name: SymName foo\n| parameters:\n| | T\n"
      "| abstract: false\n");
  // Superclasses and functional dependencies (reference manual,
  // "Functional dependencies").
  EXPECT_EQ(Compact(":- typeclass bar(A, B, C) <= ((A, B -> C), (C -> A), "
                    "ring(A)) where [].\n"),
      "TypeclassDecl\n| name: SymName bar\n| parameters:\n| | A\n| | B\n"
      "| | C\n| constraints:\n"
      "| | FunctionalDependency\n| | | domain:\n| | | | A\n| | | | B\n"
      "| | | range:\n| | | | C\n"
      "| | FunctionalDependency\n| | | domain:\n| | | | C\n"
      "| | | range:\n| | | | A\n"
      "| | ClassConstraint\n| | | name: SymName ring\n| | | arguments:\n"
      "| | | | TypeVar A\n"
      "| abstract: false\n");
}

TEST(ConvertTest, Instances) {
  EXPECT_EQ(Compact(":- instance portrayable(list(T)) <= portrayable(T) "
                    "where [pred(portray/3) is portray_list, "
                    "func(f/1) is io.f].\n"),
      "InstanceDecl\n| name: SymName portrayable\n| types:\n"
      "| | NamedType\n| | | name: SymName list\n| | | arguments:\n"
      "| | | | TypeVar T\n"
      "| constraints:\n| | ClassConstraint\n"
      "| | | name: SymName portrayable\n| | | arguments:\n"
      "| | | | TypeVar T\n"
      "| abstract: false\n| methods:\n"
      "| | MethodByName\n| | | kind: Predicate\n"
      "| | | method: NameAndArity portray/3\n"
      "| | | implementation: SymName portray_list\n"
      "| | MethodByName\n| | | kind: Function\n"
      "| | | method: NameAndArity f/1\n"
      "| | | implementation: SymName io.f\n");
  EXPECT_EQ(Compact(":- instance hashable(int).\n"),
      "InstanceDecl\n| name: SymName hashable\n| types:\n| | NamedType\n"
      "| | | name: SymName int\n| abstract: true\n");
  // Methods defined by clauses.
  EXPECT_EQ(Compact(":- instance c(int) where [p(1), (f(X) = X), "
                    "(q --> [])].\n"),
      "InstanceDecl\n| name: SymName c\n| types:\n| | NamedType\n"
      "| | | name: SymName int\n| abstract: false\n| methods:\n"
      "| | FactOrRule\n| | | kind: Predicate\n| | | name: SymName p\n"
      "| | | arguments:\n| | | | ModedTerm\n| | | | | term: Literal 1\n"
      "| | FactOrRule\n| | | kind: Function\n| | | name: SymName f\n"
      "| | | arguments:\n| | | | ModedTerm\n| | | | | term: Variable X\n"
      "| | | result: ModedTerm\n| | | | term: Variable X\n"
      "| | DcgRule\n| | | name: SymName q\n"
      "| | | body: DcgTerminals\n");
  // A method named by an operator (reference manual, "Instance
  // declarations").
  EXPECT_NE(Compact(":- instance foo(int) where [func(method1/2) is (+)].\n")
                .find("implementation: SymName +\n"),
      std::string::npos);
}

TEST(ConvertTest, TypeclassErrors) {
  using V = std::vector<std::string>;
  EXPECT_EQ(Messages_(":- typeclass c where []."),
      V{"a type class must have at least one parameter"});
  EXPECT_EQ(Messages_(":- typeclass c(T, T) where []."),
      V{"the parameter T appears more than once"});
  EXPECT_EQ(Messages_(":- typeclass c(int) where []."),
      V{"expected a type class parameter, but found 'int'"});
  EXPECT_EQ(Messages_(":- typeclass c(T) where [type t ---> t]."),
      V{"expected a pred, func or mode declaration, but found 'type t ---> "
        "t'"});
  EXPECT_EQ(Messages_(":- typeclass c(T) where [pred p(T) | X]."),
      V{"expected a list of methods, but found '[pred p(T) | X]'"});
  EXPECT_EQ(Messages_(":- typeclass c(T) <= (A -> T) where []."),
      V{"the variable A in a functional dependency is not a parameter of "
        "the type class"});
  EXPECT_EQ(Messages_(":- typeclass c(T) <= (T -> 1) where []."),
      V{"expected a type variable, but found '1'"});
  EXPECT_EQ(Messages_(":- instance c(int) where [pred(p) is q]."),
      V{"expected a name/arity, but found 'p'"});
  EXPECT_EQ(Messages_(":- instance c(int) where [pred(p/1) is X]."),
      V{"expected a predicate or function name, but found 'X'"});
  EXPECT_EQ(Messages_(":- instance c(int) where [pred(p/1) is q, X]."),
      V{"expected a clause head, but found 'X'"});
}

// The kind of the first line in Compact(text) that begins with `field`, at
// any depth: "body: Conjunction" gives "Conjunction", and "left: Variable
// X" gives "Variable".
std::string KindOf(llvm::StringRef text, llvm::StringRef field) {
  std::string compact{Compact(text)};
  llvm::SmallVector<llvm::StringRef, 32> lines;
  llvm::StringRef{compact}.split(lines, '\n', -1, false);
  for (llvm::StringRef line : lines) {
    line = line.ltrim("| ");
    if (line.consume_front(field) && line.consume_front(": ")) {
      return line.split(' ').first.str();
    }
  }
  return "";
}

// The kind of a clause's body goal.
std::string GoalKind(llvm::StringRef goal) {
  return KindOf(("p :- " + goal + ".\n").str(), "body");
}

// The kind of the data-term on the right of X = Term.
std::string DataKind(llvm::StringRef term) {
  return KindOf(("p :- X = (" + term + ").\n").str(), "right");
}

TEST(ConvertTest, Clauses) {
  EXPECT_EQ(Compact("p(a, X).\n"),
      "FactOrRule\n| kind: Predicate\n| name: SymName p\n| arguments:\n"
      "| | ModedTerm\n| | | term: FunctorData\n| | | | name: SymName a\n"
      "| | ModedTerm\n| | | term: Variable X\n");
  EXPECT_EQ(Compact("main(!IO) :- io.write_string(\"x\", !IO).\n"),
      "FactOrRule\n| kind: Predicate\n| name: SymName main\n| arguments:\n"
      "| | ModedTerm\n| | | term: StateVar !IO\n"
      "| body: Call\n| | name: SymName io.write_string\n| | arguments:\n"
      "| | | Literal \"x\"\n| | | StateVar !IO\n");
  EXPECT_EQ(Compact("f(X) = X.\n"),
      "FactOrRule\n| kind: Function\n| name: SymName f\n| arguments:\n"
      "| | ModedTerm\n| | | term: Variable X\n"
      "| result: ModedTerm\n| | term: Variable X\n");
  // Different clauses for different modes.
  EXPECT_EQ(Compact("f(X::in) = (Y::out) :- Y = X.\n"),
      "FactOrRule\n| kind: Function\n| name: SymName f\n| arguments:\n"
      "| | ModedTerm\n| | | term: Variable X\n"
      "| | | mode: NamedMode\n| | | | name: SymName in\n"
      "| result: ModedTerm\n| | term: Variable Y\n"
      "| | mode: NamedMode\n| | | name: SymName out\n"
      "| body: Unification\n| | left: Variable Y\n| | right: Variable X\n");
  EXPECT_EQ(Compact("p --> [a], q.\n"),
      "DcgRule\n| name: SymName p\n| body: DcgConjunction\n"
      "| | parallel: false\n| | goals:\n"
      "| | | DcgTerminals\n| | | | terms:\n"
      "| | | | | FunctorData\n| | | | | | name: SymName a\n"
      "| | | DcgCall\n| | | | name: SymName q\n");
  // A DCG rule's head is not a function head.
  EXPECT_NE(Compact("x = y --> [].\n").find("name: SymName =\n"),
      std::string::npos);
  // As in mmc, X `p` Y and tuples are heads too.
  EXPECT_NE(Compact("X `q` Y :- true.\n").find("name: SymName q\n"),
      std::string::npos);
  EXPECT_NE(Compact("{X} :- true.\n").find("name: SymName {}\n"),
      std::string::npos);
}

TEST(ConvertTest, Goals) {
  EXPECT_EQ(GoalKind("true"), "BuiltinGoal");
  EXPECT_EQ(GoalKind("fail"), "BuiltinGoal");
  EXPECT_EQ(GoalKind("a, b"), "Conjunction");
  EXPECT_EQ(GoalKind("a & b"), "Conjunction");
  EXPECT_EQ(GoalKind("a ; b"), "Disjunction");
  EXPECT_EQ(GoalKind("not a"), "Negation");
  EXPECT_EQ(GoalKind("\\+ a"), "Negation");
  EXPECT_EQ(GoalKind("a => b"), "Implication");
  EXPECT_EQ(GoalKind("a <= b"), "Implication");
  EXPECT_EQ(GoalKind("a <=> b"), "Implication");
  EXPECT_EQ(GoalKind("if a then b else c"), "IfThenElse");
  EXPECT_EQ(GoalKind("a -> b ; c"), "IfThenElse");
  EXPECT_EQ(GoalKind("X = Y"), "Unification");
  EXPECT_EQ(GoalKind("X \\= Y"), "Inequality");
  EXPECT_EQ(GoalKind("call(P, X)"), "Call");
  EXPECT_EQ(GoalKind("X `member` L"), "Call");
  EXPECT_EQ(GoalKind("{a, b}"), "Call");
  EXPECT_EQ(GoalKind("P"), "HigherOrderCall");
  EXPECT_EQ(GoalKind("P(X)"), "HigherOrderCall");
  EXPECT_EQ(GoalKind("X `P` Y"), "HigherOrderCall");
  EXPECT_EQ(GoalKind("some [X, !S] q(X)"), "QuantifiedGoal");
  EXPECT_EQ(GoalKind("all [X] q(X)"), "QuantifiedGoal");
  EXPECT_EQ(GoalKind("impure q"), "PurityGoal");
  EXPECT_EQ(GoalKind("semipure q"), "PurityGoal");
  EXPECT_EQ(GoalKind("promise_pure q"), "PromisePurity");
  EXPECT_EQ(GoalKind("promise_equivalent_solutions [X] q(X)"),
      "PromiseSolutions");
  EXPECT_EQ(GoalKind("promise_equivalent_solution_sets [X] q(X)"),
      "PromiseSolutions");
  EXPECT_EQ(GoalKind("arbitrary [X] q(X)"), "PromiseSolutions");
  EXPECT_EQ(GoalKind("require_det q"), "RequireDeterminism");
  EXPECT_EQ(GoalKind("require_complete_switch [X] q"), "RequireSwitch");
  EXPECT_EQ(GoalKind("require_switch_arms_semidet [X] q"), "RequireSwitch");
  EXPECT_EQ(GoalKind("disable_warnings [singleton_vars] q"),
      "DisableWarnings");
  EXPECT_EQ(GoalKind("trace [io(!IO)] q"), "TraceGoal");
  EXPECT_EQ(GoalKind("try [] q then r"), "TryGoal");
  EXPECT_EQ(GoalKind("event e(X)"), "EventGoal");
  // try(Params, Goal) alone is a call, as in mmc.
  EXPECT_EQ(GoalKind("try [] q"), "Call");
  // Conjunctions flatten through parentheses, and C -> T ; E within a
  // disjunction is an if-then-else.
  EXPECT_EQ(Compact("p :- (a, b), c.\n").find("Conjunction"),
      Compact("p :- (a, b), c.\n").rfind("Conjunction"));
  EXPECT_EQ(Compact("p :- a ; (b -> c ; d).\n"),
      "FactOrRule\n| kind: Predicate\n| name: SymName p\n"
      "| body: Disjunction\n| | goals:\n"
      "| | | Call\n| | | | name: SymName a\n"
      "| | | IfThenElse\n| | | | form: Arrow\n"
      "| | | | condition: Call\n| | | | | name: SymName b\n"
      "| | | | then: Call\n| | | | | name: SymName c\n"
      "| | | | else: Call\n| | | | | name: SymName d\n");
  EXPECT_EQ(Compact("p :- trace [compile_time(flag(\"x\") or grade(debug)), "
                    "run_time(env(\"V\")), io(!IO), state(lvl, !L)] q.\n"),
      "FactOrRule\n| kind: Predicate\n| name: SymName p\n"
      "| body: TraceGoal\n| | params:\n"
      "| | | TraceParam\n| | | | kind: CompileTime\n"
      "| | | | condition: TraceCondition\n| | | | | kind: Or\n"
      "| | | | | operands:\n"
      "| | | | | | TraceCondition\n| | | | | | | kind: Flag\n"
      "| | | | | | | value: x\n"
      "| | | | | | TraceCondition\n| | | | | | | kind: Grade\n"
      "| | | | | | | value: debug\n"
      "| | | TraceParam\n| | | | kind: RunTime\n"
      "| | | | condition: TraceCondition\n| | | | | kind: Env\n"
      "| | | | | value: V\n"
      "| | | TraceParam\n| | | | kind: Io\n| | | | var: StateVar !IO\n"
      "| | | TraceParam\n| | | | kind: State\n"
      "| | | | mutableName: SymName lvl\n| | | | var: StateVar !L\n"
      "| | goal: Call\n| | | name: SymName q\n");
  EXPECT_EQ(Compact("p :- try [io(!IO)] q then r else s "
                    "catch e -> t catch f -> u catch_any E -> v.\n"),
      "FactOrRule\n| kind: Predicate\n| name: SymName p\n"
      "| body: TryGoal\n| | io:\n| | | StateVar !IO\n"
      "| | goal: Call\n| | | name: SymName q\n"
      "| | then: Call\n| | | name: SymName r\n"
      "| | else: Call\n| | | name: SymName s\n"
      "| | catches:\n"
      "| | | CatchClause\n| | | | pattern: FunctorData\n"
      "| | | | | name: SymName e\n"
      "| | | | goal: Call\n| | | | | name: SymName t\n"
      "| | | CatchClause\n| | | | pattern: FunctorData\n"
      "| | | | | name: SymName f\n"
      "| | | | goal: Call\n| | | | | name: SymName u\n"
      "| | catchAny: CatchClause\n| | | pattern: Variable E\n"
      "| | | goal: Call\n| | | | name: SymName v\n");
  EXPECT_EQ(KindOf("p :- require_switch_arms_cc_multi [X] q.\n",
                "armDeterminism"),
      "CcMulti");
}

TEST(ConvertTest, DataTerms) {
  EXPECT_EQ(DataKind("Y"), "Variable");
  EXPECT_EQ(DataKind("!.S"), "StateVar");
  EXPECT_EQ(DataKind("1"), "Literal");
  EXPECT_EQ(DataKind("$line"), "ImplementationLiteral");
  EXPECT_EQ(DataKind("f(Y)"), "FunctorData");
  EXPECT_EQ(DataKind("Y + 1"), "FunctorData");
  EXPECT_EQ(DataKind("apply(F, 1)"), "FunctorData");
  EXPECT_EQ(DataKind("Y `f` Z"), "FunctorData");
  EXPECT_EQ(DataKind("[1, 2 | T]"), "ListData");
  EXPECT_EQ(DataKind("{1, 2}"), "TupleData");
  EXPECT_EQ(DataKind("F(1)"), "ApplyData");
  EXPECT_EQ(DataKind("Y `F` Z"), "ApplyData");
  EXPECT_EQ(DataKind("if a then 1 else 2"), "ConditionalData");
  EXPECT_EQ(DataKind("a -> 1 ; 2"), "ConditionalData");
  EXPECT_EQ(DataKind("A @ f(_)"), "UnifyData");
  EXPECT_EQ(DataKind("T ^ f ^ g(1)"), "FieldSelection");
  EXPECT_EQ(DataKind("T ^ f := 3"), "FieldUpdate");
  EXPECT_EQ(DataKind("Y : int"), "TypeQualified");
  EXPECT_EQ(DataKind("Y `with_type` int"), "TypeQualified");
  EXPECT_EQ(DataKind("coerce(Y)"), "Coerce");
  EXPECT_EQ(DataKind("pred(A::in) is det :- true"), "Lambda");
  EXPECT_EQ(DataKind("func(A) = A + 1"), "Lambda");
  // pred(...) and func(...) alone are ordinary functors.
  EXPECT_EQ(DataKind("pred(A)"), "FunctorData");
  EXPECT_EQ(Compact("p :- X = T ^ f ^ g(1) := 3.\n"),
      "FactOrRule\n| kind: Predicate\n| name: SymName p\n"
      "| body: Unification\n| | left: Variable X\n"
      "| | right: FieldUpdate\n| | | term: Variable T\n| | | fields:\n"
      "| | | | FieldSpec\n| | | | | name: SymName f\n"
      "| | | | FieldSpec\n| | | | | name: SymName g\n"
      "| | | | | arguments:\n| | | | | | Literal 1\n"
      "| | | value: Literal 3\n");
  EXPECT_EQ(Compact("p :- X = (impure pred(!.S::in, !:S::out) is det --> "
                    "[a]).\n"),
      "FactOrRule\n| kind: Predicate\n| name: SymName p\n"
      "| body: Unification\n| | left: Variable X\n"
      "| | right: Lambda\n| | | purity: Impure\n| | | kind: Predicate\n"
      "| | | arguments:\n"
      "| | | | ModedTerm\n| | | | | term: StateVar !.S\n"
      "| | | | | mode: NamedMode\n| | | | | | name: SymName in\n"
      "| | | | ModedTerm\n| | | | | term: StateVar !:S\n"
      "| | | | | mode: NamedMode\n| | | | | | name: SymName out\n"
      "| | | determinism: Det\n"
      "| | | dcgBody: DcgTerminals\n| | | | terms:\n"
      "| | | | | FunctorData\n| | | | | | name: SymName a\n");
  EXPECT_EQ(Compact("p :- X = (func(A) = B is det :- B = A).\n"),
      "FactOrRule\n| kind: Predicate\n| name: SymName p\n"
      "| body: Unification\n| | left: Variable X\n"
      "| | right: Lambda\n| | | purity: Pure\n| | | kind: Function\n"
      "| | | arguments:\n| | | | ModedTerm\n| | | | | term: Variable A\n"
      "| | | result: ModedTerm\n| | | | term: Variable B\n"
      "| | | determinism: Det\n"
      "| | | body: Unification\n| | | | left: Variable B\n"
      "| | | | right: Variable A\n");
}

TEST(ConvertTest, DcgGoals) {
  auto dcgKind{[](llvm::StringRef goal) {
    return KindOf(("p --> " + goal + ".\n").str(), "body");
  }};
  EXPECT_EQ(dcgKind("a, b"), "DcgConjunction");
  EXPECT_EQ(dcgKind("a & b"), "DcgConjunction");
  EXPECT_EQ(dcgKind("a ; b"), "DcgDisjunction");
  EXPECT_EQ(dcgKind("if a then b else c"), "DcgIfThenElse");
  EXPECT_EQ(dcgKind("a -> b ; c"), "DcgIfThenElse");
  EXPECT_EQ(dcgKind("not a"), "DcgNegation");
  EXPECT_EQ(dcgKind("some [X] q(X)"), "DcgQuantified");
  EXPECT_EQ(dcgKind("{ q }"), "DcgBraces");
  EXPECT_EQ(dcgKind("[a, b]"), "DcgTerminals");
  EXPECT_EQ(dcgKind("[]"), "DcgTerminals");
  EXPECT_EQ(dcgKind("=(X)"), "DcgUnify");
  EXPECT_EQ(dcgKind(":=(X)"), "DcgUnify");
  EXPECT_EQ(dcgKind("X =^ f ^ g"), "DcgFieldSelection");
  EXPECT_EQ(dcgKind("^ f := X"), "DcgFieldUpdate");
  EXPECT_EQ(dcgKind("q(X)"), "DcgCall");
  EXPECT_EQ(dcgKind("a => b"), "DcgCall");
  EXPECT_EQ(dcgKind("P"), "DcgHigherOrderCall");
  EXPECT_EQ(dcgKind("P(X)"), "DcgHigherOrderCall");
  // { G1, G2 } holds the conjunction of its goals.
  EXPECT_EQ(Compact("p --> { a, b }.\n"),
      "DcgRule\n| name: SymName p\n| body: DcgBraces\n"
      "| | goal: Conjunction\n| | | parallel: false\n| | | goals:\n"
      "| | | | Call\n| | | | | name: SymName a\n"
      "| | | | Call\n| | | | | name: SymName b\n");
}

TEST(ConvertTest, ClauseErrors) {
  using V = std::vector<std::string>;
  EXPECT_EQ(Messages_("X :- true.\n"),
      V{"expected a clause head, but found 'X'"});
  EXPECT_EQ(Messages_("1.\n"), V{"expected a clause head, but found '1'"});
  EXPECT_EQ(Messages_("p :- 1.\n"), V{"expected a goal, but found '1'"});
  EXPECT_EQ(Messages_("p :- (a -> b).\n"),
      V{"an if-then-else must have an else part"});
  EXPECT_EQ(Messages_("p :- (if a then b).\n"),
      V{"an if-then-else must have an else part"});
  EXPECT_EQ(Messages_("p :- X = (a -> 1).\n"),
      V{"an if-then-else must have an else part"});
  EXPECT_EQ(Messages_("p(X::in, Y).\n"),
      V{"either every argument or none must have a mode"});
  EXPECT_EQ(Messages_("f(X::in) = Y.\n"),
      V{"either every argument and the result or none must have a mode"});
  EXPECT_EQ(Messages_("p :- some X q.\n"),
      V{"expected a list of variables, but found 'X'"});
  EXPECT_EQ(Messages_("p :- some [f(X)] q.\n"),
      V{"expected a variable or a state variable, but found 'f(X)'"});
  EXPECT_EQ(Messages_("p :- q(!f).\n"),
      V{"expected a state variable's name, but found 'f'"});
  EXPECT_EQ(Messages_("p :- require_complete_switch [X, Y] q.\n"),
      V{"expected a list of one variable, but found '[X, Y]'"});
  EXPECT_EQ(Messages_("p :- disable_warnings [a(1)] q.\n"),
      V{"expected a warning name, but found 'a(1)'"});
  EXPECT_EQ(Messages_("p :- trace [foo] q.\n"),
      V{"expected a trace parameter: compile_time(...), run_time(...), "
        "io(!IO) or state(Mutable, !S), but found 'foo'"});
  EXPECT_EQ(Messages_("p :- trace [compile_time(env(\"V\"))] q.\n"),
      V{"expected a compile_time condition: flag(\"Name\"), "
        "tracelevel(Level) or grade(Test), with and, or and not, but found "
        "'env(\"V\")'"});
  EXPECT_EQ(Messages_("p :- trace [run_time(flag(\"x\") and env(\"V\"))] q.\n"),
      V{"expected a run_time condition: env(\"Name\"), with and, or and not, "
        "but found 'flag(\"x\")'"});
  EXPECT_EQ(Messages_("p :- trace [compile_time(flag(x))] q.\n"),
      V{"expected a string, but found 'x'"});
  EXPECT_EQ(Messages_("p :- trace [compile_time(tracelevel(foo))] q.\n"),
      V{"expected 'shallow' or 'deep', but found 'foo'"});
  EXPECT_EQ(Messages_("p :- trace [compile_time(grade(foo))] q.\n"),
      V{"expected a grade test: 'debug', 'ssdebug', 'prof', 'profdeep', "
        "'par', 'trail', 'rbmm', 'llds', 'mlds', 'c', 'csharp' or 'java', "
        "but found 'foo'"});
  EXPECT_EQ(Messages_("p :- trace [io(!A), io(!B)] q.\n"),
      V{"the trace parameter 'io' is given more than once"});
  EXPECT_EQ(Messages_("p :- trace [compile_time(tracelevel(deep) and "
                      "(not grade(c))), state(a, !A), state(b, !B)] q.\n"),
      V{});
  EXPECT_EQ(Messages_("p :- trace [io(IO)] q.\n"),
      V{"expected a state variable !X, but found 'IO'"});
  EXPECT_EQ(Messages_("p :- try [foo] q then r.\n"),
      V{"expected a try parameter io(!S), but found 'foo'"});
  EXPECT_EQ(Messages_("p :- try [] q catch e -> r.\n"),
      V{"a try goal must have a then part: 'try Params Goal then Goal', "
        "then optionally 'else Goal', 'catch Pattern -> Goal' and "
        "'catch_any Var -> Goal'"});
  EXPECT_EQ(Messages_("p :- try [] q then r catch e.\n"),
      V{"expected 'Pattern -> Goal', but found 'e'"});
  EXPECT_EQ(Messages_("p :- event X.\n"),
      V{"expected a predicate call, but found 'X'"});
  EXPECT_EQ(Messages_("p :- atomic([], q).\n"), V{"atomic goals"});
  EXPECT_EQ(Messages_("p :- X = (pred(A) :- true).\n"),
      V{"a lambda expression's head must be 'pred(Args) is Det' or "
        "'func(Args) = Result', with an optional 'is Det'"});
  EXPECT_EQ(Messages_("p :- X = (pred(A) is foo).\n"),
      V{"expected a determinism category, but found 'foo'"});
  EXPECT_EQ(Messages_("p(X) :- X := 1.\n"),
      V{"'X := 1' is a DCG goal, which may appear only in a DCG rule"});
  EXPECT_EQ(Messages_("p(X) :- =(X).\n"),
      V{"'=(X)' is a DCG goal, which may appear only in a DCG rule"});
  EXPECT_EQ(Messages_("p --> X = Y.\n"),
      V{"expected a DCG goal such as '=(Term)' or 'Term =^ Field', but "
        "found 'X = Y'"});
  EXPECT_EQ(Messages_("p --> {}.\n"),
      V{"expected a DCG goal, but found '{}'"});
  EXPECT_EQ(Messages_("p --> [a | T].\n"),
      V{"expected a list without a tail, but found '[a | T]'"});
}

TEST(ConvertTest, ModuleDirectives) {
  EXPECT_EQ(Compact(":- initialise init/2.\n:- finalize fin/0.\n"),
      "InitialiseDecl\n| kind: Initialise\n| pred: NameAndArity init/2\n"
      "InitialiseDecl\n| kind: Finalise\n| pred: NameAndArity fin/0\n");
  EXPECT_EQ(Compact(":- mutable(counter, int, 0, ground, [untrailed, "
                    "attach_to_io_state, foreign_name(\"C\", \"ctr\")]).\n"),
      "MutableDecl\n| name: counter\n| type: NamedType\n"
      "| | name: SymName int\n| initial: Literal 0\n"
      "| inst: BuiltinInst\n| | kind: Ground\n| trailed: false\n"
      "| attachToIoState: true\n| constant: false\n| threadLocal: false\n"
      "| foreignNames:\n| | MutableForeignName\n| | | language: C\n"
      "| | | name: ctr\n");
  // The attribute list may be left out, and the initial value is any
  // data-term.
  EXPECT_EQ(Compact(":- mutable(m, list(int), [1 + 2], ground).\n"),
      "MutableDecl\n| name: m\n| type: NamedType\n| | name: SymName list\n"
      "| | arguments:\n| | | NamedType\n| | | | name: SymName int\n"
      "| initial: ListData\n| | elements:\n| | | FunctorData\n"
      "| | | | name: SymName +\n| | | | arguments:\n| | | | | Literal 1\n"
      "| | | | | Literal 2\n| inst: BuiltinInst\n| | kind: Ground\n"
      "| attachToIoState: false\n| constant: false\n| threadLocal: false\n");
}

TEST(ConvertTest, Promises) {
  EXPECT_EQ(Compact(":- promise all [X] (p(X) => q(X)).\n"),
      "PromiseDecl\n| goal: QuantifiedGoal\n| | quantifier: All\n"
      "| | vars:\n| | | Variable X\n| | goal: Implication\n"
      "| | | kind: Implies\n| | | left: Call\n| | | | name: SymName p\n"
      "| | | | arguments:\n| | | | | Variable X\n| | | right: Call\n"
      "| | | | name: SymName q\n| | | | arguments:\n| | | | | Variable X\n");
  EXPECT_EQ(Compact(":- all [X] promise_exclusive_exhaustive "
                    "(X = a ; X = b).\n"),
      "PromiseDisjunctionDecl\n| kind: ExclusiveExhaustive\n| vars:\n"
      "| | X\n| disjunction: Disjunction\n| | goals:\n"
      "| | | Unification\n| | | | left: Variable X\n"
      "| | | | right: FunctorData\n| | | | | name: SymName a\n"
      "| | | Unification\n| | | | left: Variable X\n"
      "| | | | right: FunctorData\n| | | | | name: SymName b\n");
}

TEST(ConvertTest, SolverTypes) {
  EXPECT_EQ(Compact(":- solver type st(T).\n"),
      "AbstractTypeDecl\n| name: SymName st\n| parameters:\n| | T\n"
      "| solver: true\n");
  EXPECT_EQ(Compact(":- solver type st where representation is int, "
                    "ground is ground, any is ground, constraint_store is "
                    "mutable(cs, int, 0, ground, [untrailed]), equality is "
                    "eq, comparison is cmp.\n"),
      "SolverTypeDefn\n| name: SymName st\n"
      "| representation: NamedType\n| | name: SymName int\n"
      "| ground: BuiltinInst\n| | kind: Ground\n| any: BuiltinInst\n"
      "| | kind: Ground\n"
      "| constraintStore:\n| | MutableDecl\n| | | name: cs\n"
      "| | | type: NamedType\n| | | | name: SymName int\n"
      "| | | initial: Literal 0\n| | | inst: BuiltinInst\n| | | | kind: Ground\n"
      "| | | trailed: false\n| | | attachToIoState: false\n"
      "| | | constant: false\n| | | threadLocal: false\n"
      "| equality: SymName eq\n| comparison: SymName cmp\n");
  // As in mmc, only the representation is required.
  EXPECT_EQ(Compact(":- solver type st where representation is int, "
                    "constraint_store is [mutable(a, int, 0, ground), "
                    "mutable(b, int, 0, ground)].\n")
                .find("constraintStore:\n| | MutableDecl\n| | | name: a\n"),
      Compact(":- solver type st where representation is int, "
              "constraint_store is [mutable(a, int, 0, ground), "
              "mutable(b, int, 0, ground)].\n")
          .find("constraintStore:"));
  EXPECT_NE(Compact(":- solver type st where representation is int.\n")
                .find("SolverTypeDefn"),
      std::string::npos);
}

TEST(ConvertTest, DirectiveErrors) {
  using V = std::vector<std::string>;
  EXPECT_EQ(Messages_(":- initialise i.\n"),
      V{"expected 'Name/Arity', but found 'i'"});
  EXPECT_EQ(Messages_(":- initialize pred(i/2).\n"),
      V{"expected 'Name/Arity', but found 'pred(i/2)'"});
  EXPECT_EQ(Messages_(":- mutable(m, int, 0).\n"),
      V{"a ':- mutable' declaration must have four or five arguments"});
  EXPECT_EQ(Messages_(":- mutable(\"m\", int, 0, ground).\n"),
      V{"expected a mutable name, but found '\"m\"'"});
  EXPECT_EQ(Messages_(":- mutable(m, int, 0, ground, [foo]).\n"),
      V{"expected a mutable attribute: 'trailed', 'untrailed', "
        "'attach_to_io_state', 'constant', 'thread_local' or "
        "'foreign_name(Lang, Name)', but found 'foo'"});
  EXPECT_EQ(Messages_(":- mutable(m, int, 0, ground, [untrailed, "
                      "untrailed]).\n"),
      V{"the attribute 'untrailed' is given more than once"});
  EXPECT_EQ(Messages_(":- mutable(m, int, 0, ground, [trailed, untrailed]).\n"),
      V{"the attributes 'trailed' and 'untrailed' conflict"});
  EXPECT_EQ(Messages_(":- mutable(m, int, 0, ground, [constant, "
                      "attach_to_io_state]).\n"),
      V{"the attributes 'constant' and 'attach_to_io_state' conflict"});
  EXPECT_EQ(Messages_(":- mutable(m, int, 0, ground, [thread_local, "
                      "constant]).\n"),
      V{"the attributes 'thread_local' and 'constant' conflict"});
  EXPECT_EQ(Messages_(":- mutable(m, int, 0, ground, [foreign_name(\"C\", "
                      "\"a\"), foreign_name(c, \"b\")]).\n"),
      V{"the mutable already has a foreign name for this language"});
  EXPECT_EQ(Messages_(":- promise_exhaustive (a ; b).\n"),
      V{"a ':- promise_exhaustive' declaration must have the form ':- all "
        "[Vars] promise_exhaustive Disjunction'"});
  EXPECT_EQ(Messages_(":- all [X] promise_exclusive X = 1.\n"),
      V{"the goal of a ':- promise_exclusive' declaration must be a "
        "disjunction"});
  EXPECT_EQ(Messages_(":- solver st.\n"),
      V{"':- solver' must be followed by a type declaration"});
  EXPECT_EQ(Messages_(":- solver type st == int.\n"),
      V{"a solver type cannot be an equivalence type"});
  EXPECT_EQ(Messages_(":- solver type st ---> st.\n"),
      V{"a solver type cannot have data constructors"});
  EXPECT_EQ(Messages_(":- solver type st where constraint_store is [].\n"),
      V{"a solver type definition must have a 'representation' attribute"});
  EXPECT_EQ(Messages_(":- solver type st where constraint_store is [], "
                      "representation is int.\n"),
      V{"expected a solver type attribute, in the order 'representation "
        "is', 'ground is', 'any is', 'constraint_store is', 'equality is', "
        "'comparison is', but found 'representation is int'"});
  EXPECT_EQ(Messages_(":- solver type st where representation is int, "
                      "constraint_store is [foo].\n"),
      V{"expected a mutable declaration, but found 'foo'"});
}

// The kind of node a pragma converts to.
std::string PragmaKind(llvm::StringRef pragma) {
  std::string items{Items((":- pragma " + pragma + ".\n").str())};
  return items.substr(0, items.find('\n'));
}

TEST(ConvertTest, Pragmas) {
  EXPECT_EQ(PragmaKind("inline(p/1)"), "PredPragma");
  EXPECT_EQ(PragmaKind("no_inline(pred(p/1))"), "PredPragma");
  EXPECT_EQ(PragmaKind("consider_used(func(f/1))"), "PredPragma");
  EXPECT_EQ(PragmaKind("promise_equivalent_clauses(p/1)"), "PredPragma");
  EXPECT_EQ(PragmaKind("check_termination(p/1)"), "PredPragma");
  EXPECT_EQ(PragmaKind("obsolete(p/1, [q/1])"), "ObsoletePragma");
  EXPECT_EQ(PragmaKind("obsolete_proc(p)"), "ObsoletePragma");
  EXPECT_EQ(PragmaKind("type_spec(p/1, T = int)"), "TypeSpecPragma");
  EXPECT_EQ(PragmaKind("memo(p/1)"), "TablingPragma");
  EXPECT_EQ(PragmaKind("loop_check(p(in, out))"), "TablingPragma");
  EXPECT_EQ(PragmaKind("minimal_model(p/1, [allow_reset])"), "TablingPragma");
  EXPECT_EQ(PragmaKind("fact_table(p/2, \"f\")"), "FactTablePragma");
  EXPECT_EQ(PragmaKind("require_feature_set([memo])"),
      "RequireFeatureSetPragma");
  EXPECT_EQ(PragmaKind("source_file(\"x.m\")"), "SourceFilePragma");
  EXPECT_EQ(PragmaKind("foreign_decl(\"C\", \"int x;\")"), "ForeignDeclPragma");
  EXPECT_EQ(PragmaKind("foreign_code(java, include_file(\"x\"))"),
      "ForeignCodePragma");
  EXPECT_EQ(PragmaKind("foreign_proc(\"C#\", p, [], \"\")"),
      "ForeignProcPragma");
  EXPECT_EQ(PragmaKind("foreign_type(\"C\", t, \"long\")"),
      "ForeignTypePragma");
  EXPECT_EQ(PragmaKind("foreign_enum(\"C\", t/0, [a - \"1\"])"),
      "ForeignEnumPragma");
  EXPECT_EQ(PragmaKind("foreign_export_enum(\"C\", t/0)"),
      "ForeignExportEnumPragma");
  EXPECT_EQ(PragmaKind("foreign_export(\"C\", p(in), \"p\")"),
      "ForeignExportPragma");
  EXPECT_EQ(PragmaKind("foreign_import_module(\"C\", m)"),
      "ForeignImportModulePragma");
  EXPECT_EQ(Compact(":- pragma foreign_proc(\"C\", f(X::in) = (Y::out), "
                    "[promise_pure, will_not_call_mercury], \"Y = X;\").\n"),
      "ForeignProcPragma\n| language: C\n| kind: Function\n| name: SymName f\n"
      "| arguments:\n| | ForeignProcArg\n| | | var: Variable X\n"
      "| | | mode: NamedMode\n| | | | name: SymName in\n"
      "| result: ForeignProcArg\n| | var: Variable Y\n"
      "| | mode: NamedMode\n| | | name: SymName out\n"
      "| attributes:\n| | PromisePure\n| | WillNotCallMercury\n"
      "| code: ForeignCode\n| | includeFile: false\n| | text: Y = X;\n");
  EXPECT_EQ(Compact(":- pragma memo(f(in) = out, [allow_reset, "
                    "specified([value, output], hidden_arg_addr), "
                    "size_limit(10)]).\n"),
      "TablingPragma\n| method: Memo\n| pred: ProcSpec\n| | kind: Function\n"
      "| | name: SymName f\n| | modes:\n| | | NamedMode\n"
      "| | | | name: SymName in\n| | result: NamedMode\n"
      "| | | name: SymName out\n| attributes: TablingAttributes\n"
      "| | allowReset: true\n| | statistics: false\n"
      "| | disableWarningIfIgnored: false\n| | fastLoose: false\n"
      "| | specified:\n| | | Value\n| | | Output\n| | hiddenArg: Addr\n"
      "| | sizeLimit: 10\n");
  EXPECT_EQ(Compact(":- pragma foreign_type(java, t(T), \"Object\", "
                    "[stable]) where equality is eq.\n"),
      "ForeignTypePragma\n| language: Java\n| name: SymName t\n"
      "| parameters:\n| | T\n| descriptor: Object\n| assertions:\n"
      "| | Stable\n| attributes: TypeAttributes\n"
      "| | equality: SymName eq\n| | abstractNoncanonical: false\n");
  EXPECT_EQ(Compact(":- pragma foreign_export_enum(\"C\", t/0, "
                    "[prefix(\"T_\"), uppercase], [a - \"A\"]).\n"),
      "ForeignExportEnumPragma\n| language: C\n| type: NameAndArity t/0\n"
      "| prefix: T_\n| uppercase: true\n| overrides:\n"
      "| | ForeignEnumValue\n| | | constructor: SymName a\n"
      "| | | value: A\n");
  EXPECT_EQ(Compact(":- pragma type_spec(p(in, in), (T = int, U = T)).\n"),
      "TypeSpecPragma\n| pred: ProcSpec\n| | kind: Predicate\n"
      "| | name: SymName p\n| | modes:\n| | | NamedMode\n"
      "| | | | name: SymName in\n| | | NamedMode\n| | | | name: SymName in\n"
      "| substitution:\n| | TypeSubstitution\n| | | var: T\n"
      "| | | type: NamedType\n| | | | name: SymName int\n"
      "| | TypeSubstitution\n| | | var: U\n| | | type: TypeVar T\n");
  // Foreign code over several lines is escaped in the dump.
  EXPECT_NE(Compact(":- pragma foreign_code(\"C\", \"a\nb\").\n")
                .find("text: \"a\\nb\"\n"),
      std::string::npos);
}

TEST(ConvertTest, PragmaErrors) {
  using V = std::vector<std::string>;
  auto messages{[](llvm::StringRef pragma) {
    return Messages_((":- pragma " + pragma + ".\n").str());
  }};
  EXPECT_EQ(messages("foo(x)"), V{"'foo' is not a recognized pragma"});
  EXPECT_EQ(messages("require_tail_recursion(p/0)"),
      V{"the ':- pragma require_tail_recursion' declaration"});
  EXPECT_EQ(messages("inline(p/0, f/1)"),
      V{"a ':- pragma inline' declaration must have one argument"});
  EXPECT_EQ(messages("foreign_proc(\"C\", p, \"\")"),
      V{"a ':- pragma foreign_proc' declaration must have four arguments"});
  EXPECT_EQ(messages("foreign_export_enum(\"C\")"),
      V{"a ':- pragma foreign_export_enum' declaration must have two to "
        "four arguments"});
  EXPECT_EQ(messages("inline(p(in))"),
      V{"expected 'Name/Arity', 'pred(Name/Arity)' or 'func(Name/Arity)', "
        "but found 'p(in)'"});
  EXPECT_EQ(messages("obsolete(p/0, [pred(f/1)])"),
      V{"expected a name/arity, but found 'pred(f/1)'"});
  EXPECT_EQ(messages("type_spec(p/0, int = T)"),
      V{"expected a type substitution 'T = Type', but found 'int = T'"});
  EXPECT_EQ(messages("memo(f/1, [statistics, statistics])"),
      V{"the attribute 'statistics' is given more than once"});
  EXPECT_EQ(messages("minimal_model(p/0, [disable_warning_if_ignored])"),
      V{"'disable_warning_if_ignored' is allowed only in a memo pragma"});
  EXPECT_EQ(messages("memo(f/1, [specified([value], foo)])"),
      V{"expected 'hidden_arg_value' or 'hidden_arg_addr', but found 'foo'"});
  EXPECT_EQ(messages("source_file(x)"),
      V{"expected a file name string, but found 'x'"});
  EXPECT_EQ(messages("foreign_proc(\"Fortran\", p, [], \"\")"),
      V{"expected a foreign language: \"C\", \"C#\" or \"Java\", but found "
        "'\"Fortran\"'"});
  EXPECT_EQ(messages("foreign_proc(\"C\", p, [foo], \"\")"),
      V{"expected a foreign_proc attribute, but found 'foo'"});
  EXPECT_EQ(messages("foreign_proc(\"C\", p, [will_not_call_mercury, "
                     "may_call_mercury], \"\")"),
      V{"the attributes 'will_not_call_mercury' and 'may_call_mercury' "
        "conflict"});
  EXPECT_EQ(messages("foreign_proc(\"C\", f(X::in) = Y, [], \"\")"),
      V{"expected 'Var::Mode', but found 'Y'"});
  EXPECT_EQ(messages("foreign_proc(\"C\", f(X::in) = (X::out), [], \"\")"),
      V{"the variable X appears more than once"});
  EXPECT_EQ(messages("foreign_decl(\"C\", foo, \"\")"),
      V{"expected 'local' or 'exported', but found 'foo'"});
  EXPECT_EQ(messages("foreign_export(\"C\", p, \"\")"),
      V{"expected the foreign name, a non-empty string, but found '\"\"'"});
  EXPECT_EQ(messages("foreign_enum(\"C\", t, [a - \"1\"])"),
      V{"expected a type name/arity, but found 't'"});
  EXPECT_EQ(messages("foreign_enum(\"C\", t/0, [a - 1])"),
      V{"expected 'Constructor - \"Value\"', but found 'a - 1'"});
  EXPECT_EQ(messages("foreign_export_enum(\"C\", t/0, [prefix(\"a\"), "
                     "prefix(\"b\")])"),
      V{"the attribute 'prefix' is given more than once"});
  EXPECT_EQ(messages("foreign_type(\"C\", t(T, T), \"long\")"),
      V{"the parameter T appears more than once"});
  EXPECT_EQ(messages("inline(p/0) where equality is eq"),
      V{"only a foreign_type pragma may have 'where' attributes"});
}

// A %$omp directive applies to a goal, and is ignored anywhere else.
TEST(ConvertTest, Directives) {
  using V = std::vector<std::string>;
  auto messages{[](llvm::StringRef text) {
    return Convert(text, /*openMP=*/true).messages;
  }};
  ConvertResult converted{
      Convert("p :-\n  %$omp parallel do\n  q.\n", /*openMP=*/true)};
  EXPECT_EQ(converted.messages, V{});
  EXPECT_NE(converted.tree.find("body: OmpGoal\n"
                                "| | | | directive: OmpDirective\n"
                                "| | | | | id: parallel do\n"),
      std::string::npos)
      << converted.tree;
  EXPECT_EQ(messages("p :-\n  %$omp parallel do schedule(static),\n"
                     "  %$omp&  nowait collapse(2)\n  q.\n"),
      V{});
  EXPECT_EQ(messages("p :-\n  %$omp parallel for\n  q.\n"),
      V{"'parallel for' is C's form of the OpenMP directive; Mercury uses "
        "Fortran's, 'parallel do'"});
  EXPECT_EQ(messages("p :-\n  %$omp do lastprivate\n  q.\n"), V{});
  EXPECT_EQ(messages("p :-\n  %$omp do upper(1, 2)\n  q.\n"),
      V{"expected upper(Bound)"});
  ConvertResult ignored{Convert("%$omp parallel do\np(X) :- X = (\n"
                                "  %$omp parallel do\n  1).\n",
      /*openMP=*/true)};
  EXPECT_EQ(ignored.messages,
      (V{"an OpenMP directive applies only to a goal; it is ignored",
          "an OpenMP directive applies only to a goal; it is ignored"}));
  EXPECT_NE(ignored.tree.find("FactOrRule"), std::string::npos);
  // Without -fopenmp a directive is a comment.
  EXPECT_EQ(Messages_("p :-\n  %$omp parallel do\n  q.\n"), V{});
}

// A directive's variables are numbered as the item's: I and N as in the
// goal, and Z, which only the directive has, after them.
TEST(ConvertTest, DirectiveVariables) {
  AllSources all;
  SourceFile *file{all.AddBuffer("test.m",
      llvm::MemoryBuffer::getMemBufferCopy(
          "p(A, N) :-\n  %$omp do index(I) upper(N) private(Z, A)\n"
          "  q(I, N).\n"))};
  Messages messages;
  TokenSequence tokens{Lex(*file, LexerOptions{true}, messages)};
  ParseTreeArena arena;
  const ParseTree &tree{ConvertFile(tokens, arena, messages)};
  EXPECT_TRUE(messages.messages().empty());
  struct Visitor {
    std::vector<std::string> variables;
    void Post(const Variable &variable) {
      variables.push_back(
          variable.name.str() + "=" + std::to_string(variable.number));
    }
  } visitor;
  Walk(tree, visitor);
  EXPECT_EQ(visitor.variables,
      (std::vector<std::string>{
          "A=0", "N=1", "I=2", "N=1", "Z=3", "A=0", "I=2", "N=1"}));
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
