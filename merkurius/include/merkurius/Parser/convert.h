//===-- include/merkurius/Parser/convert.h ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_PARSER_CONVERT_H_
#define MERKURIUS_PARSER_CONVERT_H_

// The converter: reads the items of a file as terms and converts each into
// a parse tree item, following the rules of the reference manual.

#include "merkurius/Parser/parse-tree.h"
#include "merkurius/Parser/token.h"
#include <optional>

namespace merkurius::parser {

class Messages;
class Term;
struct ConvertState;

// Converts one item's term, or reports why it is not an item.
std::optional<Item> ConvertItem(const Term &, ConvertState &);

// Reads and converts every item of a file. An item with an error is
// reported and left out, so the tree is complete only if no error was
// reported.
const ParseTree &ConvertFile(
    const TokenSequence &, ParseTreeArena &, Messages &);

} // namespace merkurius::parser
#endif // MERKURIUS_PARSER_CONVERT_H_
