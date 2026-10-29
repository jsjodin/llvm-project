//===-- include/merkurius/Parser/unparse.h ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_PARSER_UNPARSE_H_
#define MERKURIUS_PARSER_UNPARSE_H_

// The unparser: prints a parse tree back as Mercury source.

#include "merkurius/Parser/parse-tree.h"
#include "merkurius/Parser/token.h"
#include "llvm/Support/raw_ostream.h"

namespace merkurius::parser {

class Messages;

enum class UnparseLayout {
  Exact, // each token's trivia as written: the file, byte for byte
  // Comments and line number directives removed, and each token's trivia
  // replaced by one newline if it held one, one space if it held anything
  // else, and nothing if it was empty.
  Minimal,
};

// Prints the tree using the tokens it recorded. Each node prints the tokens
// of its range, those of its children through them, so the output is the
// file's tokens in order. Reported as errors, the places where the tree
// does not account for its tokens:
//  - a node whose tokens are not within its parent's;
//  - a node whose tokens overlap those of a node before it;
//  - a variable or literal token printed by a node, but held by none of its
//    fields, as when the converter drops part of a goal.
// The tokens between items are checked only if `complete`, that is, when no
// item was left out for an error.
void UnparseExact(llvm::raw_ostream &, const ParseTree &,
    const TokenSequence &, UnparseLayout, bool complete, Messages &);

} // namespace merkurius::parser
#endif // MERKURIUS_PARSER_UNPARSE_H_
