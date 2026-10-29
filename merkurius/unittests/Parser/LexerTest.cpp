//===-- unittests/Parser/LexerTest.cpp --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "merkurius/Parser/lexer.h"
#include "gtest/gtest.h"
#include "merkurius/Parser/message.h"
#include "merkurius/Parser/provenance.h"
#include "merkurius/Parser/source.h"
#include "llvm/Support/MemoryBuffer.h"
#include <string>
#include <vector>

using namespace merkurius::parser;

namespace {

struct Lexed {
  SourceFile *file;
  TokenSequence tokens;
  Messages messages;

  // Each token as "kind spelling", the eof token left out.
  std::vector<std::string> Tokens() const {
    std::vector<std::string> result;
    for (const Token &token : tokens.tokens) {
      if (token.kind != TokenKind::Eof) {
        result.push_back(Describe(token));
      }
    }
    return result;
  }
  std::vector<std::string> DirectiveTokens(std::size_t index) const {
    std::vector<std::string> result;
    const Token &directive{tokens.tokens.at(index)};
    for (std::size_t j{0}; j + 1 < directive.directiveTokens; ++j) {
      result.push_back(
          Describe(tokens.directiveTokens[directive.firstDirectiveToken + j]));
    }
    return result;
  }
  std::vector<std::string> MessageTexts() const {
    std::vector<std::string> result;
    for (const Message &message : messages.messages()) {
      result.push_back(message.ToString());
    }
    return result;
  }
  static std::string Describe(const Token &token) {
    return (GetTokenKindName(token.kind) + " " + token.spelling).str();
  }
};

// Lexes `text`, and checks that the tokens give back the text exactly, both
// at the top level and within each directive.
Lexed Lex(AllSources &all, llvm::StringRef text, bool openMP = false) {
  SourceFile *file{
      all.AddBuffer("test.m", llvm::MemoryBuffer::getMemBufferCopy(text))};
  Lexed result{file, {}, {}};
  result.tokens = Lex(*file, LexerOptions{openMP}, result.messages);
  std::string unparsed;
  llvm::raw_string_ostream os{unparsed};
  UnparseTokens(os, result.tokens);
  EXPECT_EQ(unparsed, text);
  EXPECT_EQ(result.tokens.tokens.back().kind, TokenKind::Eof);
  for (const Token &token : result.tokens.tokens) {
    if (token.kind == TokenKind::Directive) {
      std::string inner;
      for (std::size_t j{0}; j < token.directiveTokens; ++j) {
        const Token &t{
            result.tokens.directiveTokens[token.firstDirectiveToken + j]};
        inner += (t.trivia + t.spelling).str();
      }
      EXPECT_EQ(inner, token.spelling);
    }
  }
  return result;
}

using Strings = std::vector<std::string>;

TEST(LexerTest, Clause) {
  AllSources all;
  Lexed lexed{Lex(all, "p(X) :- X = [a, \"s\" | T], q.\n")};
  EXPECT_EQ(lexed.Tokens(),
      (Strings{"name p", "open_ct (", "variable X", "close )", "name :-",
          "variable X", "name =", "open_list [", "name a", "comma ,",
          "string \"s\"", "ht_sep |", "variable T", "close_list ]", "comma ,",
          "name q", "end ."}));
  EXPECT_TRUE(lexed.messages.empty());
  const Token &eof{lexed.tokens.tokens.back()};
  EXPECT_EQ(eof.trivia, "\n");
}

TEST(LexerTest, OpenAndOpenCt) {
  AllSources all;
  EXPECT_EQ(Lex(all, "f(X) - (X) f/*c*/(X)").Tokens(),
      (Strings{"name f", "open_ct (", "variable X", "close )", "name -",
          "open (", "variable X", "close )", "name f", "open (", "variable X",
          "close )"}));
}

TEST(LexerTest, EndAndQualification) {
  AllSources all;
  // As in mmc, "./*" is a graphic name, not an end token and a comment.
  EXPECT_EQ(Lex(all, "a.b. c.%x\nd. /*y*/e./*z*/").Tokens(),
      (Strings{"name a", "name .", "name b", "end .", "name c", "end .",
          "name d", "end .", "name e", "name ./*", "name z", "name */"}));
}

TEST(LexerTest, GraphicNames) {
  AllSources all;
  // As in mmc, a graphic name continues through "/*".
  EXPECT_EQ(Lex(all, "X =.. Y, A<<u, B+/*c*/C, ; ! \\+ -->").Tokens(),
      (Strings{"variable X", "name =..", "variable Y", "comma ,", "variable A",
          "name <<", "name u", "comma ,", "variable B", "name +/*", "name c",
          "name */", "variable C", "comma ,", "name ;", "name !", "name \\+",
          "name -->"}));
}

TEST(LexerTest, Numbers) {
  AllSources all;
  Lexed lexed{Lex(all,
      "0 1_000 0b1010 0o_17 0xff_u8 7i8 42u32 3u 0'a 0'' 0' 0'\xc3\xa5 "
      "1.5 1e10 1.5E-3 1_0.2_5 1.5_e10 1e1_0 1.e5")};
  EXPECT_EQ(lexed.Tokens(),
      (Strings{"integer 0", "integer 1_000", "integer 0b1010", "integer 0o_17",
          "integer 0xff_u8", "integer 7i8", "integer 42u32", "integer 3u",
          "integer 0'a", "integer 0''", "integer 0' ", "integer 0'\xc3\xa5",
          "float 1.5", "float 1e10", "float 1.5E-3", "float 1_0.2_5",
          "float 1.5_e10", "float 1e1_0", "integer 1", "name .", "name e5"}));
  EXPECT_TRUE(lexed.messages.empty());
}

TEST(LexerTest, NumberErrors) {
  AllSources all;
  Lexed lexed{Lex(all, "1_ 0x 0b102 7i7 2_e3 1_.5")};
  EXPECT_EQ(lexed.MessageTexts(),
      (Strings{"an integer literal may not end with an underscore",
          "missing digits after '0x'",
          "invalid digit in a literal with prefix '0b'",
          "invalid integer suffix",
          "an underscore may come before an exponent only after a fraction",
          "an underscore may not come before '.'"}));
}

TEST(LexerTest, StringsAndQuotedNames) {
  AllSources all;
  Lexed lexed{Lex(all,
      "\"a\"\"b\" \"two\nlines\" \"cont\\\ninued\" 'it''s' '' " +
          std::string{R"("\a\b\r\f\t\n\v\\\'\"\x41\\101\\u00e5\U0001F600")"})};
  EXPECT_EQ(lexed.Tokens(),
      (Strings{"string \"a\"\"b\"", "string \"two\nlines\"",
          "string \"cont\\\ninued\"", "name 'it''s'", "name ''",
          R"(string "\a\b\r\f\t\n\v\\\'\"\x41\\101\\u00e5\U0001F600")"}));
  EXPECT_TRUE(lexed.messages.empty());
}

TEST(LexerTest, EscapeErrors) {
  AllSources all;
  Lexed lexed{Lex(all, "\"\\e\" \"\\q\" \"\\x41\" \"\\U00110000\" \"open")};
  EXPECT_EQ(lexed.MessageTexts(),
      (Strings{"'\\e' is not an escape in Mercury 22.01",
          "invalid escape sequence",
          "missing '\\' to end the hexadecimal escape",
          "a Unicode escape may not exceed \\U0010FFFF",
          "unterminated string"}));
}

TEST(LexerTest, VariablesAndLiterals) {
  AllSources all;
  EXPECT_EQ(Lex(all, "_ _x X1 Sum $file $line `with` $").Tokens(),
      (Strings{"variable _", "variable _x", "variable X1", "variable Sum",
          "implementation_defined_literal $file",
          "implementation_defined_literal $line", "backquote `", "name with",
          "backquote `", "name $"}));
}

TEST(LexerTest, Comments) {
  AllSources all;
  // Block comments do not nest: the first */ ends the comment.
  Lexed lexed{Lex(all, "a % one\n/* two /* three */ b */ c")};
  EXPECT_EQ(lexed.Tokens(), (Strings{"name a", "name b", "name */", "name c"}));
  EXPECT_EQ(lexed.tokens.tokens[1].trivia, " % one\n/* two /* three */ ");
  EXPECT_EQ(
      Lex(all, "a /* open").MessageTexts(), (Strings{"unterminated comment"}));
}

TEST(LexerTest, UnexpectedCharacters) {
  AllSources all;
  Lexed lexed{Lex(all, "a \xc3\xa5 # b")};
  EXPECT_EQ(lexed.Tokens(),
      (Strings{"name a", "error \xc3\xa5", "error #", "name b"}));
  EXPECT_EQ(lexed.MessageTexts(),
      (Strings{"unexpected character", "unexpected character"}));
}

TEST(LexerTest, LineNumberDirectives) {
  AllSources all;
  Lexed lexed{Lex(all, "a.\n#100\nb.\r\n#7\r\nc.\n#0\nd #5\n")};
  EXPECT_EQ(lexed.Tokens(),
      (Strings{
          "name a", "end .", "name b", "end .", "name c", "end .", "name d"}));
  EXPECT_EQ(lexed.MessageTexts(),
      (Strings{"line number must be a positive integer"}));
  auto lineOf{[&](std::size_t index) {
    const Token &token{lexed.tokens.tokens[index]};
    return lexed.file
        ->GetSourcePosition(static_cast<std::size_t>(
            token.spelling.begin() - lexed.file->content().begin()))
        .line;
  }};
  EXPECT_EQ(lineOf(0), 1); // a
  EXPECT_EQ(lineOf(2), 100); // b
  EXPECT_EQ(lineOf(4), 7); // c
  EXPECT_EQ(lexed.tokens.tokens[2].trivia, "\n#100\n");
}

TEST(LexerTest, DirectivesNeedOpenMP) {
  AllSources all;
  Lexed lexed{Lex(all, "%$omp parallel do\np.\n")};
  EXPECT_EQ(lexed.Tokens(), (Strings{"name p", "end ."}));
}

TEST(LexerTest, Directive) {
  AllSources all;
  Lexed lexed{Lex(all,
      "p :-\n"
      "    %$omp parallel do reduction(+) % note\n"
      "    %$omp&   private(X)\n"
      "    q.\n",
      /*openMP=*/true)};
  EXPECT_EQ(lexed.Tokens(),
      (Strings{"name p", "name :-",
          "directive %$omp parallel do reduction(+) % note\n"
          "    %$omp&   private(X)",
          "name q", "end ."}));
  EXPECT_EQ(lexed.DirectiveTokens(2),
      (Strings{"name parallel", "name do", "name reduction", "open_ct (",
          "name +", "close )", "name private", "open_ct (", "variable X",
          "close )"}));
  EXPECT_TRUE(lexed.messages.empty());
}

TEST(LexerTest, NotDirectives) {
  AllSources all;
  // Not first on its line, not followed by whitespace, a stray continuation,
  // and inside a string.
  Lexed lexed{Lex(all,
      "a. %$omp parallel\n%$ompx\n%$omp& b\nc = \"\n%$omp parallel\n\".\n",
      /*openMP=*/true)};
  EXPECT_EQ(lexed.Tokens(),
      (Strings{"name a", "end .", "name c",
          "name =", "string \"\n%$omp parallel\n\"", "end ."}));
}

TEST(LexerTest, Empty) {
  AllSources all;
  Lexed lexed{Lex(all, "")};
  ASSERT_EQ(lexed.tokens.tokens.size(), 1u);
  EXPECT_EQ(lexed.tokens.tokens[0].kind, TokenKind::Eof);
}

} // namespace
