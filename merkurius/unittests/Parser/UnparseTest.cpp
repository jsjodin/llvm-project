//===-- unittests/Parser/UnparseTest.cpp ------------------------*- C++ -*-===//
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
#include "merkurius/Parser/unparse.h"
#include "llvm/Support/MemoryBuffer.h"
#include <string>
#include <vector>

using namespace merkurius::parser;

namespace {

struct UnparseResult {
  std::string text;
  std::vector<std::string> messages;
};

std::vector<std::string> Strings(const Messages &messages) {
  std::vector<std::string> strings;
  for (const Message &message : messages.messages()) {
    strings.push_back(message.ToString());
  }
  return strings;
}

// Converts `text` and unparses its tree.
UnparseResult Unparse(
    llvm::StringRef text, UnparseLayout layout = UnparseLayout::Exact) {
  AllSources all;
  SourceFile *file{
      all.AddBuffer("test.m", llvm::MemoryBuffer::getMemBufferCopy(text))};
  Messages messages;
  TokenSequence tokens{Lex(*file, LexerOptions{}, messages)};
  ParseTreeArena arena;
  const ParseTree &tree{ConvertFile(tokens, arena, messages)};
  bool complete{!messages.AnyFatalError()};
  UnparseResult result;
  llvm::raw_string_ostream os{result.text};
  UnparseExact(os, tree, tokens, layout, complete, messages);
  result.messages = Strings(messages);
  return result;
}

TEST(UnparseTest, Exact) {
  llvm::StringRef text{"% A comment.\n:- module m.\n:- pred p(int::in).\n"
                       "p(X) :-\n    /* inline */ X > 0,\n#10\n"
                       "    io.write(\"x\" ++ 'y', !IO). % trailing\n"};
  UnparseResult result{Unparse(text)};
  EXPECT_EQ(result.text, text);
  EXPECT_EQ(result.messages, std::vector<std::string>{});
}

TEST(UnparseTest, MinimalLayout) {
  UnparseResult result{Unparse(
      "% A comment.\n:- module m.\n\n\np(X) :-\n    /* c */ X > 0,  "
      "/* c */ X \\= (1).\n#10\nq([A|B], {C}) :- A=B,C=0.",
      UnparseLayout::Minimal)};
  // Layout is kept only where it separates tokens: whether there is any,
  // as before '(' here, and whether it ends a line.
  EXPECT_EQ(result.text,
      "\n:- module m.\np(X) :-\nX > 0, X \\= (1).\nq([A|B], {C}) :- A=B,C=0.");
  EXPECT_EQ(result.messages, std::vector<std::string>{});
}

// Converts `text` and prints it in the canonical layout.
std::string Canonical(
    llvm::StringRef text, bool parenthesize = false, bool openMP = false) {
  AllSources all;
  SourceFile *file{
      all.AddBuffer("test.m", llvm::MemoryBuffer::getMemBufferCopy(text))};
  Messages messages;
  TokenSequence tokens{Lex(*file, LexerOptions{openMP}, messages)};
  ParseTreeArena arena;
  const ParseTree &tree{ConvertFile(tokens, arena, messages)};
  std::string result;
  llvm::raw_string_ostream os{result};
  UnparseCanonical(os, tree, tokens, parenthesize);
  return result;
}

TEST(UnparseTest, CanonicalGoals) {
  EXPECT_EQ(Canonical("p(X):-q(X),r(X)."), "p(X) :-\n    q(X),\n    r(X).\n");
  EXPECT_EQ(Canonical("p(X) :- (if q(X) then r else s), ( t ; u )."),
      "p(X) :-\n"
      "    ( if q(X) then\n        r\n    else\n        s\n    ),\n"
      "    (\n        t\n    ;\n        u\n    ).\n");
  // else if and ; C -> continue their chains.
  EXPECT_EQ(Canonical("p :- (if a then b else if c then d else e)."),
      "p :-\n    ( if a then\n        b\n    else if c then\n        d\n"
      "    else\n        e\n    ).\n");
  EXPECT_EQ(Canonical("p :- (a -> b ; c -> d ; e)."),
      "p :-\n    ( a ->\n        b\n    ; c ->\n        d\n    ;\n        e\n"
      "    ).\n");
  // A scope's goal in parentheses is a block; a condition stays inline.
  EXPECT_EQ(Canonical("p :- promise_pure (a, b), not (c, d)."),
      "p :-\n    promise_pure (\n        a,\n        b\n    ),\n"
      "    not (c, d).\n");
  EXPECT_EQ(Canonical("p --> {q}, [a]."),
      "p -->\n    { q },\n    [a].\n");
}

TEST(UnparseTest, CanonicalTokens) {
  // Spellings and quoting are kept; layout is the canonical one.
  EXPECT_EQ(Canonical("x(0x1F,0'a,'A b',\"s\")."),
      "x(0x1F, 0'a, 'A b', \"s\").\n");
  EXPECT_EQ(Canonical(":- pred  p( int :: in , io.state::di )is det."),
      ":- pred p(int::in, io.state::di) is det.\n");
  // An open stays apart from what precedes it, and an open_ct does not.
  EXPECT_EQ(Canonical("x(- (1), -1, - 1, f(a), X ^ f)."),
      "x(- (1), -1, - 1, f(a), X ^ f).\n");
  EXPECT_EQ(Canonical(":- type t ---> a ; b(int) where equality is eq."),
      ":- type t\n    --->    a\n    ;       b(int)\n    where equality is eq.\n");
  // A long argument list breaks after its commas.
  EXPECT_EQ(Canonical("p :- q(aaaaaaaaaaaaaaaaaaaa, bbbbbbbbbbbbbbbbbbbbbbbbb, "
                      "cccccccccccccccccccccc, ddddddddddd)."),
      "p :-\n    q(\n        aaaaaaaaaaaaaaaaaaaa,\n        bbbbbbbbbbbbbbbbbbbbbbbbb,\n"
      "        cccccccccccccccccccccc,\n        ddddddddddd).\n");
}

TEST(UnparseTest, CanonicalComments) {
  EXPECT_EQ(Canonical("% Head.\n\n\n:- module m. % trailing\n"
                      "p :- % after the neck\n  q, /* inline */ r.\n"
                      "% at the end\n"),
      "% Head.\n\n:- module m. % trailing\np :- % after the neck\n"
      "    q, /* inline */\n    r.\n% at the end\n");
  EXPECT_EQ(Canonical("p :-\n    % before q\n    q,\n\n    r."),
      "p :-\n    % before q\n    q,\n    r.\n");
  // An item with an error is printed as written.
  EXPECT_EQ(Canonical("p :-   q(.\nr:-s."), "p :-   q(.\nr :-\n    s.\n");
}

// A directive is a line of its own, printed as written, before its goal.
TEST(UnparseTest, CanonicalDirectives) {
  EXPECT_EQ(Canonical("p:-\n%$omp parallel do  reduction(+)\nq,r.", false,
                true),
      "p :-\n    %$omp parallel do  reduction(+)\n    q,\n    r.\n");
  EXPECT_EQ(Canonical("p:-q,\n  %$omp critical\n  %$omp&  hint(1)\nr.",
                false, true),
      "p :-\n    q,\n    %$omp critical\n    %$omp&  hint(1)\n    r.\n");
  // The parenthesized form leaves the directive outside the parentheses.
  EXPECT_EQ(Canonical("p:-\n%$omp do\nX = Y + 1.", true, true),
      "p :-\n    %$omp do\n    (X = (Y + 1)).\n");
  // A "%$omp&" line after a blank line is a comment, which stays apart
  // from the directive, so as not to continue it.
  EXPECT_EQ(Canonical("p:-\n%$omp parallel do\n\n%$omp&)\nq.", false, true),
      "p :-\n    %$omp parallel do\n\n    %$omp&)\n    q.\n");
}

TEST(UnparseTest, Parenthesized) {
  EXPECT_EQ(Canonical("p(X) :- X = 1 + 2 * 3, \\+ q(-X).", true),
      "p(X) :-\n    ((X = (1 + (2 * 3))),\n    (\\+ q((-X)))).\n");
}

// An item left out for an error leaves tokens that no node holds; they are
// not reported when the tree is known to be incomplete.
TEST(UnparseTest, Incomplete) {
  UnparseResult result{Unparse(":- module m.\np(X) :- X = .\nq.\n")};
  EXPECT_EQ(result.text, ":- module m.\np(X) :- X = .\nq.\n");
  for (const std::string &message : result.messages) {
    EXPECT_EQ(message.find("internal error"), std::string::npos) << message;
  }
}

// Unparses a tree made by hand for the tokens of `text`.
class HandMadeTree {
public:
  explicit HandMadeTree(llvm::StringRef text) {
    file_ = all_.AddBuffer(
        "test.m", llvm::MemoryBuffer::getMemBufferCopy(text));
    tokens_ = Lex(*file_, LexerOptions{}, messages_);
  }
  const SymName *Name(std::uint32_t first, std::uint32_t last,
      llvm::StringRef name) {
    return arena_.Make<SymName>(
        TokenRange{first, last}, llvm::ArrayRef<llvm::StringRef>{}, name);
  }
  // `:- import_module` with these modules, its tokens [first, last].
  Item Import(std::uint32_t first, std::uint32_t last,
      std::vector<const SymName *> modules) {
    return arena_.Make<ImportDecl>(TokenRange{first, last},
        ImportKind::ImportModule,
        arena_.SaveArray(llvm::ArrayRef<const SymName *>{modules}));
  }
  std::vector<std::string> Messages_(
      std::vector<Item> items, bool complete = true) {
    auto last{static_cast<std::uint32_t>(tokens_.tokens.size() - 1)};
    ParseTree tree{TokenRange{0, last},
        arena_.SaveArray(llvm::ArrayRef<Item>{items})};
    Messages messages;
    std::string text;
    llvm::raw_string_ostream os{text};
    UnparseExact(os, tree, tokens_, UnparseLayout::Exact, complete, messages);
    EXPECT_EQ(text, file_->content());
    return Strings(messages);
  }

private:
  AllSources all_;
  SourceFile *file_;
  Messages messages_;
  TokenSequence tokens_;
  ParseTreeArena arena_;
};

TEST(UnparseTest, Problems) {
  using V = std::vector<std::string>;
  // The tokens: ':-' 0, import_module 1, a 2, ',' 3, b 4, '.' 5, eof 6.
  HandMadeTree tree{":- import_module a, b.\n"};
  EXPECT_EQ(tree.Messages_({tree.Import(
                1, 4, {tree.Name(2, 2, "a"), tree.Name(4, 4, "b")})}),
      V{});
  EXPECT_EQ(tree.Messages_({tree.Import(
                1, 4, {tree.Name(2, 3, "a"), tree.Name(3, 4, "b")})}),
      V{"internal error: the tokens of this SymName overlap those of the "
        "SymName before it"});
  EXPECT_EQ(tree.Messages_({tree.Import(
                1, 4, {tree.Name(2, 2, "a"), tree.Name(4, 5, "b")})}),
      V{"internal error: the tokens of this SymName are not within those of "
        "its parent ImportDecl"});
  EXPECT_EQ(tree.Messages_({tree.Import(1, 4, {tree.Name(2, 2, "a")}),
                tree.Import(3, 4, {tree.Name(4, 4, "b")})}),
      V{"internal error: the tokens of this ImportDecl overlap those of the "
        "ImportDecl before it"});

  // A variable that no node holds: ':-' 0, import_module 1, a 2, ',' 3, X 4.
  HandMadeTree lost{":- import_module a, X.\n"};
  EXPECT_EQ(lost.Messages_({lost.Import(1, 4, {lost.Name(2, 2, "a")})}),
      V{"internal error: the token 'X' is within this ImportDecl, but none "
        "of its fields holds it"});
  // An item that is missing from the tree, unless items were left out.
  HandMadeTree missing{"p(X).\n"};
  EXPECT_EQ(missing.Messages_({}),
      V{"internal error: the token 'X' is within this ParseTree, but none of "
        "its fields holds it"});
  EXPECT_EQ(missing.Messages_({}, false), V{});
}

} // namespace
