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
