//===-- include/merkurius/Parser/token.h ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_PARSER_TOKEN_H_
#define MERKURIUS_PARSER_TOKEN_H_

// Tokens of a Mercury source file. Each token records the trivia before it
// (whitespace, comments and line number directives) and its spelling, both
// as ranges of the file's text. The trivia and spelling of consecutive
// tokens are adjacent, and the eof token holds whatever follows the last
// token, so printing them in order gives back the file byte for byte.

#include "merkurius/Parser/char-block.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <vector>

namespace merkurius::parser {

class SourceFile;

// The token kinds of the reference manual, named as it names them.
enum class TokenKind {
  Name, // unquoted, quoted or graphic, or ';'
  Variable,
  Integer,
  Float,
  String,
  ImplementationDefinedLiteral, // '$' and an unquoted name
  Backquote,
  OpenCt, // '(' not preceded by whitespace
  Open, // '(' preceded by whitespace
  Close,
  OpenList,
  CloseList,
  OpenCurly,
  CloseCurly,
  HtSep, // '|'
  Comma,
  End, // '.' followed by whitespace, a comment or the end of the file
  Directive, // a %$omp directive, holding its own tokens
  Eof,
  Error, // characters that begin no token
};

llvm::StringRef GetTokenKindName(TokenKind);

struct Token {
  TokenKind kind;
  CharBlock trivia;
  CharBlock spelling;
  // For a directive, its tokens in TokenSequence::directiveTokens, which end
  // with an eof token. Their trivia and spellings make up the directive's
  // spelling, the "%$omp" sentinel included.
  std::size_t firstDirectiveToken{0};
  std::size_t directiveTokens{0};
};

struct TokenSequence {
  std::vector<Token> tokens; // ends with an eof token
  std::vector<Token> directiveTokens;
};

// Prints each token's position, kind and spelling, one per line, with a
// directive's tokens indented below it.
void DumpTokens(
    llvm::raw_ostream &, const TokenSequence &, const SourceFile &);
// Prints each token's trivia and spelling: the source file, unchanged.
void UnparseTokens(llvm::raw_ostream &, const TokenSequence &);

} // namespace merkurius::parser
#endif // MERKURIUS_PARSER_TOKEN_H_
