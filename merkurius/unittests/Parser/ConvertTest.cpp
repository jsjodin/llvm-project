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
  EXPECT_EQ(Messages_(":- initialise p.\n"),
      std::vector<std::string>{"the ':- initialise' declaration"});
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
      "AbstractTypeDecl\n| name: SymName t2\n| parameters:\n| | T1\n| | T2\n");
  EXPECT_EQ(Compact(":- type t where type_is_abstract_noncanonical.\n"),
      "AbstractTypeDecl\n| name: SymName t\n| attributes: TypeAttributes\n"
      "| | abstractNoncanonical: true\n");
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
  EXPECT_EQ(Messages_(":- solver type st."),
      V{"the ':- solver' declaration"});
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
  EXPECT_EQ(Messages_(":- instance c(int) where [p(1), (q :- true)]."),
      (V{"an instance method defined by clauses",
          "an instance method defined by clauses"}));
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
