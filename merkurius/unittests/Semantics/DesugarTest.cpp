//===-- unittests/Semantics/DesugarTest.cpp ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "merkurius/Common/idioms.h"
#include "merkurius/Parser/convert.h"
#include "merkurius/Parser/lexer.h"
#include "merkurius/Parser/message.h"
#include "merkurius/Parser/provenance.h"
#include "merkurius/Parser/source.h"
#include "merkurius/Semantics/desugar.h"
#include "llvm/Support/MemoryBuffer.h"
#include <set>
#include <string>
#include <vector>

using namespace merkurius;
using namespace merkurius::semantics;

namespace {

struct DesugarResult {
  std::string form; // DumpSemanticForm's output
  std::vector<std::string> messages;
  std::vector<std::set<unsigned>> numbers; // each clause's goal numbers
  std::vector<unsigned> goals; // each clause's count of goals
};

void Numbers(const Goal &goal, std::set<unsigned> &numbers) {
  std::visit(
      [&](const auto *g) {
        EXPECT_TRUE(numbers.insert(g->number).second) << g->number;
        using T =
            std::remove_cv_t<std::remove_pointer_t<std::decay_t<decltype(g)>>>;
        if constexpr (std::is_same_v<T, Conj> || std::is_same_v<T, Disj>) {
          for (const Goal &sub : g->goals) {
            Numbers(sub, numbers);
          }
        } else if constexpr (std::is_same_v<T, semantics::IfThenElse>) {
          Numbers(g->condition, numbers);
          Numbers(g->then, numbers);
          Numbers(g->otherwise, numbers);
        } else if constexpr (std::is_same_v<T, Not> ||
            std::is_same_v<T, Scope>) {
          Numbers(g->goal, numbers);
        }
      },
      goal);
}

DesugarResult DesugarText(llvm::StringRef text) {
  parser::AllSources all;
  parser::SourceFile *file{
      all.AddBuffer("test.m", llvm::MemoryBuffer::getMemBufferCopy(text))};
  parser::Messages messages;
  parser::TokenSequence tokens{
      parser::Lex(*file, parser::LexerOptions{}, messages)};
  parser::ParseTreeArena arena;
  const parser::ParseTree &tree{parser::ConvertFile(tokens, arena, messages)};
  SemanticArena semanticArena;
  const SemanticForm &form{Desugar(tree, tokens, semanticArena, messages)};
  DesugarResult result;
  llvm::raw_string_ostream os{result.form};
  DumpSemanticForm(os, form);
  for (const parser::Message &message : messages.messages()) {
    result.messages.push_back(message.ToString());
  }
  for (const ClauseForm *clause : form.clauses) {
    Numbers(clause->body, result.numbers.emplace_back());
    result.goals.push_back(clause->goals);
  }
  return result;
}

std::string Form(llvm::StringRef text) {
  DesugarResult result{DesugarText(text)};
  EXPECT_EQ(result.messages, std::vector<std::string>{}) << text.str();
  return result.form;
}

using V = std::vector<std::string>;

TEST(DesugarTest, Heads) {
  // A repeated head variable and a non-variable argument are unified in
  // the body; a fact's body is the empty conjunction.
  EXPECT_EQ(Form("p(X, X, f(Y)).\n"),
      "Predicate p(X, $1, $2)\n"
      "| Conj #2\n"
      "| | Unify #0 $1 = X\n"
      "| | Unify #1 $2 = f(Y)\n");
  EXPECT_EQ(Form("p.\n"), "Predicate p\n| Conj #0\n");
  EXPECT_EQ(Form("f(X) = X.\n"),
      "Function f(X) = $1\n| Unify #0 $1 = X\n");
  EXPECT_EQ(Form("f(_, _) = 1.\n"),
      "Function f(_$0, _$1) = $2\n| Unify #0 $2 = 1\n");
}

TEST(DesugarTest, Superhomogeneous) {
  // Nested functors become chains through fresh variables, outermost
  // first; a call's arguments are built before it.
  EXPECT_EQ(Form("p :- X = f(g(Y), 1), q(h(X)).\n"),
      "Predicate p\n"
      "| Conj #5\n"
      "| | Unify #2 X = f($1, $3)\n"
      "| | Unify #0 $1 = g(Y)\n"
      "| | Unify #1 $3 = 1\n"
      "| | Unify #3 $4 = h(X)\n"
      "| | Call #4 q($4)\n");
  // The variable side is kept when either side is one; otherwise the left
  // side gets a fresh variable.
  EXPECT_EQ(Form("p :- f(X) = Y, a = b.\n"), 
      "Predicate p\n"
      "| Conj #3\n"
      "| | Unify #0 Y = f(X)\n"
      "| | Unify #1 $2 = a\n"
      "| | Unify #2 $2 = b\n");
  EXPECT_EQ(Form("p :- X = [A, B | T], Y = [], Z = {A, \"s\"}.\n"),
      "Predicate p\n"
      "| Conj #5\n"
      "| | Unify #1 X = [A | $2]\n"
      "| | Unify #0 $2 = [B | T]\n"
      "| | Unify #2 Y = []\n"
      "| | Unify #4 Z = {A, $7}\n"
      "| | Unify #3 $7 = \"s\"\n");
  EXPECT_EQ(Form("p :- X = [A].\n"),
      "Predicate p\n"
      "| Conj #2\n"
      "| | Unify #1 X = [A | $2]\n"
      "| | Unify #0 $2 = []\n");
  EXPECT_EQ(Form("p :- X = Y @ f(Z), W = (X : list(int)), C = coerce(W), "
                 "L = $line.\n"),
      "Predicate p\n"
      "| type: W : list(int)\n"
      "| Conj #5\n"
      "| | Unify #0 X = Y\n"
      "| | Unify #1 X = f(Z)\n"
      "| | Unify #2 W = X\n"
      "| | Unify #3 C = coerce(W)\n"
      "| | Unify #4 L = $line\n");
}

TEST(DesugarTest, HigherOrder) {
  EXPECT_EQ(Form("p :- call(P, X), P(Y), Z = apply(F, X), W = F(1).\n"),
      "Predicate p\n"
      "| Conj #5\n"
      "| | HOCall #0 P(X)\n"
      "| | HOCall #1 P(Y)\n"
      "| | HOCall #2 Z = F(X)\n"
      "| | Unify #3 $6 = 1\n"
      "| | HOCall #4 W = F($6)\n");
  // A qualified call or apply is an ordinary name.
  EXPECT_EQ(Form("p :- m.call(P).\n"), "Predicate p\n| Call #0 m.call(P)\n");
}

TEST(DesugarTest, Goals) {
  // true is the empty conjunction and fail the empty disjunction; a
  // conjunction flattens, but a parallel one keeps its conjuncts.
  EXPECT_EQ(Form("p :- ( if q then true else fail ), (a & (b, c)).\n"),
      "Predicate p\n"
      "| Conj #9\n"
      "| | IfThenElse #3\n"
      "| | | Call #0 q\n"
      "| | | Conj #2\n"
      "| | | Disj #1\n"
      "| | ParConj #8\n"
      "| | | Call #4 a\n"
      "| | | Conj #7\n"
      "| | | | Call #5 b\n"
      "| | | | Call #6 c\n");
  EXPECT_EQ(Form("p :- a ; b ; not c.\n"),
      "Predicate p\n"
      "| Disj #4\n"
      "| | Call #0 a\n"
      "| | Call #1 b\n"
      "| | Not #3\n"
      "| | | Call #2 c\n");
}

TEST(DesugarTest, Negations) {
  // G1 => G2 is not (G1, not G2); G1 <= G2 is G2 => G1; G1 <=> G2 is both.
  EXPECT_EQ(Form("p :- a => b.\n"),
      "Predicate p\n"
      "| Not #4\n"
      "| | Conj #3\n"
      "| | | Call #0 a\n"
      "| | | Not #2\n"
      "| | | | Call #1 b\n");
  EXPECT_EQ(Form("p :- a <= b.\n"),
      "Predicate p\n"
      "| Not #4\n"
      "| | Conj #3\n"
      "| | | Call #0 b\n"
      "| | | Not #2\n"
      "| | | | Call #1 a\n");
  EXPECT_EQ(Form("p :- a <=> b.\n").substr(0, 22), "Predicate p\n| Conj #10");
  // all Vs G is not (some Vs (not G)), and X \\= Y is not (X = Y).
  EXPECT_EQ(Form("p :- all [X] q(X).\n"),
      "Predicate p\n"
      "| Not #3\n"
      "| | Scope #2 Some [X]\n"
      "| | | Not #1\n"
      "| | | | Call #0 q(X)\n");
  EXPECT_EQ(Form("p :- X \\= f(Y).\n"),
      "Predicate p\n| Not #1\n| | Unify #0 X = f(Y)\n");
}

TEST(DesugarTest, Scopes) {
  EXPECT_EQ(Form("p :- some [X] promise_pure (impure q(X)), "
                 "require_complete_switch [Y] r(Y), "
                 "promise_equivalent_solutions [Z] s(Z), "
                 "trace [compile_time(flag(\"f\"))] t, event e(1).\n"),
      "Predicate p\n"
      "| Conj #12\n"
      "| | Scope #3 Some [X]\n"
      "| | | Scope #2 PromisePurity\n"
      "| | | | Scope #1 Purity\n"
      "| | | | | Call #0 q(X)\n"
      "| | Scope #5 RequireSwitch [Y]\n"
      "| | | Call #4 r(Y)\n"
      "| | Scope #7 PromiseSolutions [Z]\n"
      "| | | Call #6 s(Z)\n"
      "| | Scope #9 Trace\n"
      "| | | Call #8 t\n"
      "| | Unify #10 $3 = 1\n"
      "| | Event #11 e($3)\n");
}

TEST(DesugarTest, Numbering) {
  // Goal numbers are dense and distinct within each clause.
  DesugarResult result{DesugarText("p(X, f(X)) :- ( q(g(X)) ; r(X, [1, 2]) ), "
                           "not (s(X) => t(h(X))), all [Y] u(Y, k(1)).\n"
                           "q(a).\n")};
  EXPECT_EQ(result.messages, V{});
  ASSERT_EQ(result.numbers.size(), 2u);
  for (std::size_t j{0}; j < result.numbers.size(); ++j) {
    EXPECT_EQ(result.numbers[j].size(), result.goals[j]);
    EXPECT_EQ(*result.numbers[j].rbegin() + 1, result.goals[j]);
  }
}

TEST(DesugarTest, StateVariables) {
  // The reference manual's examples ("State variables"): !X in a call is
  // !.X, !:X, and each update makes a new version.
  EXPECT_EQ(Form("main(!IO) :- write(1, !IO), X = 2, write(X, !IO).\n"),
      "Predicate main(IO!0, IO!2)\n"
      "| Conj #4\n"
      "| | Unify #0 $1 = 1\n"
      "| | Call #1 write($1, IO!0, IO!1)\n"
      "| | Unify #2 X = 2\n"
      "| | Call #3 write(X, IO!1, IO!2)\n");
  // A body that does not update !X ends by unifying its last version with
  // the first, and the head's !.X and !:X are its first and last.
  EXPECT_EQ(Form("foldl2(_, [], !A, !B).\n"),
      "Predicate foldl2(_$2, $3, A!0, A!1, B!0, B!1)\n"
      "| Conj #3\n"
      "| | Unify #0 $3 = []\n"
      "| | Unify #1 A!1 = A!0\n"
      "| | Unify #2 B!1 = B!0\n");
  EXPECT_EQ(Form("p(!.S, !:S) :- q(!S), r(!.S).\n"),
      "Predicate p(S!0, S!1)\n"
      "| Conj #2\n"
      "| | Call #0 q(S!0, S!1)\n"
      "| | Call #1 r(S!1)\n");
  // !:X twice in one call is one version; !.X there is the one before.
  EXPECT_EQ(Form("p(!S) :- q(!.S, !:S, !:S), !:S = f(!.S).\n"),
      "Predicate p(S!0, S!2)\n"
      "| Conj #2\n"
      "| | Call #0 q(S!0, S!1, S!1)\n"
      "| | Unify #1 S!2 = f(S!1)\n");
  // Branches that leave !X at different versions end at a new one.
  EXPECT_EQ(Form("p(!S) :- ( q(!S) ; r ).\n"),
      "Predicate p(S!0, S!2)\n"
      "| Disj #6\n"
      "| | Conj #4\n"
      "| | | Call #0 q(S!0, S!1)\n"
      "| | | Unify #2 S!2 = S!1\n"
      "| | Conj #5\n"
      "| | | Call #1 r\n"
      "| | | Unify #3 S!2 = S!0\n");
  // The condition's update is seen in the then part, not the else part.
  EXPECT_EQ(Form("p(!S) :- ( if c(!S) then t(!S) else true ).\n"),
      "Predicate p(S!0, S!3)\n"
      "| IfThenElse #5\n"
      "| | Call #0 c(S!0, S!1)\n"
      "| | Conj #4\n"
      "| | | Call #1 t(S!1, S!2)\n"
      "| | | Unify #2 S!3 = S!2\n"
      "| | Unify #3 S!3 = S!0\n");
  // A negated goal's updates are not seen after it.
  EXPECT_EQ(Form("p(!S) :- not q(!S), r(!S).\n"),
      "Predicate p(S!0, S!2)\n"
      "| Conj #3\n"
      "| | Not #1\n"
      "| | | Call #0 q(S!0, S!1)\n"
      "| | Call #2 r(S!0, S!2)\n");
  // some [!X] makes a new state variable, with no version at first, which
  // hides one of the same name.
  EXPECT_EQ(Form("p(!S) :- some [!S] (i(!:S), q(!S)), r(!S).\n"),
      "Predicate p(S!0, S!3)\n"
      "| Conj #5\n"
      "| | Scope #3 Some\n"
      "| | | Conj #2\n"
      "| | | | Call #0 i(S!1)\n"
      "| | | | Call #1 q(S!1, S!2)\n"
      "| | Call #4 r(S!0, S!3)\n");
  // io(!IO) in a trace goal is a new state variable inside it, whose first
  // and last versions are the scope's variables.
  EXPECT_EQ(Form("p :- trace [io(!IO)] w(!IO).\n"),
      "Predicate p\n"
      "| Scope #1 Trace [IO!0, IO!1]\n"
      "| | Call #0 w(IO!0, IO!1)\n");
}

TEST(DesugarTest, StateVariableErrors) {
  auto messages{[](llvm::StringRef text) { return DesugarText(text).messages; }};
  EXPECT_EQ(messages("p :- q(!S).\n"),
      (V{"state variable !.S is not visible in this context",
          "state variable !:S is not visible in this context"}));
  EXPECT_EQ(messages("p(!S) :- !S = 1.\n"),
      V{"!S cannot be a side of a unification; you probably meant !.S or "
        "!:S"});
  EXPECT_EQ(messages("p(!S) :- q(f(!S)).\n"),
      V{"!S may be only an argument of a call or a clause head, since it "
        "stands for two arguments, !.S and !:S"});
  EXPECT_EQ(messages("f = !S.\n"),
      V{"!S cannot be a function's result, since it stands for two "
        "arguments, !.S and !:S"});
  EXPECT_EQ(messages("p(!S) :- some [!.T] q(!S).\n"),
      V{"only a variable or a state variable !X may be quantified by some"});
  EXPECT_EQ(messages("p :- all [!T] q.\n"),
      V{"only a variable may be quantified by all"});
  // As in mmc, reading a state variable before it has a value is only a
  // warning.
  EXPECT_EQ(messages("p(!:S) :- q(!.S).\n"),
      V{"reference to uninitialized state variable !.S"});
}

TEST(DesugarTest, DcgRules) {
  // The manual's DCG-transform ("DCG-goals"): the implicit arguments are
  // the versions of a hidden state variable, printed as dcg!N.
  EXPECT_EQ(Form("digits([D | Ds]) --> [D], { is_digit(D) }, digits(Ds).\n"),
      "Predicate digits($0, dcg!0, dcg!2)\n"
      "| Conj #4\n"
      "| | Unify #0 $0 = [D | Ds]\n"
      "| | Unify #1 dcg!0 = [D | dcg!1]\n"
      "| | Call #2 is_digit(D)\n"
      "| | Call #3 digits(Ds, dcg!1, dcg!2)\n");
  // [] is V_out = V_in.
  EXPECT_EQ(Form("p --> [].\n"),
      "Predicate p(dcg!0, dcg!1)\n| Unify #0 dcg!1 = dcg!0\n");
  // Disjuncts end at one version; a negation leaves V_in unchanged.
  EXPECT_EQ(Form("p --> ( [a] ; [] ), not [b], q.\n"),
      "Predicate p(dcg!0, dcg!4)\n"
      "| Conj #11\n"
      "| | Disj #5\n"
      "| | | Conj #4\n"
      "| | | | Unify #1 dcg!0 = [$2 | dcg!1]\n"
      "| | | | Unify #0 $2 = a\n"
      "| | | | Unify #2 dcg!2 = dcg!1\n"
      "| | | Unify #3 dcg!2 = dcg!0\n"
      "| | Not #9\n"
      "| | | Conj #8\n"
      "| | | | Unify #7 dcg!2 = [$5 | dcg!3]\n"
      "| | | | Unify #6 $5 = b\n"
      "| | Call #10 q(dcg!2, dcg!4)\n");
  // The condition's input match is seen in the then part only.
  EXPECT_EQ(Form("p(X) --> ( if [X] then [] else { X = 0 } ).\n"),
      "Predicate p(X, dcg!0, dcg!2)\n"
      "| IfThenElse #5\n"
      "| | Unify #0 dcg!0 = [X | dcg!1]\n"
      "| | Unify #2 dcg!2 = dcg!1\n"
      "| | Conj #4\n"
      "| | | Unify #1 X = 0\n"
      "| | | Unify #3 dcg!2 = dcg!0\n");
  // =(T) is T = V_in, and :=(T) is V_out = T.
  EXPECT_EQ(Form("p --> =(S), :=(S).\n"),
      "Predicate p(dcg!0, dcg!1)\n"
      "| Conj #2\n"
      "| | Unify #0 dcg!0 = S\n"
      "| | Unify #1 dcg!1 = S\n");
  // A variable as a DCG goal is call(Var); state variables are threaded
  // too.
  EXPECT_EQ(Form("p(P, !A) --> P, call(P, !A), P(1).\n"),
      "Predicate p(P, A!0, A!1, dcg!0, dcg!3)\n"
      "| Conj #4\n"
      "| | HOCall #0 P(dcg!0, dcg!1)\n"
      "| | HOCall #1 P(A!0, A!1, dcg!1, dcg!2)\n"
      "| | Unify #2 $6 = 1\n"
      "| | HOCall #3 P($6, dcg!2, dcg!3)\n");
  EXPECT_EQ(Form("p --> some [X] q(X).\n"),
      "Predicate p(dcg!0, dcg!1)\n"
      "| Scope #1 Some [X]\n"
      "| | Call #0 q(X, dcg!0, dcg!1)\n");
}

TEST(DesugarTest, NotYetImplemented) {
  auto messages{[](llvm::StringRef text) { return DesugarText(text).messages; }};
  EXPECT_EQ(messages("p --> ^ f := 1.\n"),
      V{"desugaring field update"});
  EXPECT_EQ(messages("p --> X =^ f, { q(X) }.\n"),
      V{"desugaring field selection"});
  EXPECT_EQ(messages("p :- X = (func(A) = A).\n"),
      V{"desugaring lambda expressions"});
  EXPECT_EQ(messages("p :- X = Y ^ f.\n"),
      V{"desugaring field selection"});
  EXPECT_EQ(messages("p :- X = (Y ^ f := 1).\n"),
      V{"desugaring field update"});
  EXPECT_EQ(messages("p :- X = (if q then 1 else 2).\n"),
      V{"desugaring conditional expressions"});
  EXPECT_EQ(messages("p :- try [] q then r.\n"),
      V{"desugaring try goals"});
  EXPECT_EQ(messages("p :- atomic [outer(!IO), inner(!S)] q.\n"),
      V{"desugaring atomic goals"});
  EXPECT_EQ(messages("p(!S) :- !S ^ f := 1.\n"),
      V{"desugaring field update"});
  EXPECT_EQ(messages("p(!S) :- promise_equivalent_solutions [!:S] q(!S).\n"),
      V{"desugaring state variables in a scope's list of variables"});
  // A clause with an error is left out; the others are kept.
  DesugarResult result{DesugarText("p :- X = Y ^ f.\nq.\n")};
  EXPECT_EQ(result.form, "Predicate q\n| Conj #0\n");
}

} // namespace
