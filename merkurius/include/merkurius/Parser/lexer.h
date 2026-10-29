//===-- include/merkurius/Parser/lexer.h ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_PARSER_LEXER_H_
#define MERKURIUS_PARSER_LEXER_H_

// The lexer reads a source file's text unchanged. Besides the tokens of the
// reference manual, it recognizes %$omp directives (with -fopenmp) and '#'
// line number directives, which it records in the source file's line table.
// Literal values are not computed here; the term reader computes them from
// the spellings, which the lexer has checked.

#include "merkurius/Parser/char-block.h"
#include "merkurius/Parser/token.h"
#include <string>
#include <vector>

namespace merkurius::parser {

class SourceFile;

struct LexerOptions {
  bool openMP{false}; // recognize %$omp directives
};

struct LexerMessage {
  CharBlock at; // within the source file's text
  std::string text;
};

// Lexes a whole file. Errors are appended to `messages`; the tokens still
// cover the whole file, so the round trip holds even for erroneous input.
TokenSequence Lex(SourceFile &, const LexerOptions &,
    std::vector<LexerMessage> &messages);

} // namespace merkurius::parser
#endif // MERKURIUS_PARSER_LEXER_H_
