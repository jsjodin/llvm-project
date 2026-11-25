//===-- include/merkurius/Parser/char-block.h -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Copied from llvm-project flang/include/flang/Parser/char-block.h
// at upstream commit d81bd0216be61d9689fc04e74272b73bc6d8a20b,
// and modified for Merkurius.
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_PARSER_CHAR_BLOCK_H_
#define MERKURIUS_PARSER_CHAR_BLOCK_H_

// Describes a contiguous block of characters; does not own their storage.

#include "llvm/ADT/StringRef.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <string>

namespace llvm {
class raw_ostream;
}

namespace merkurius::parser {

class CharBlock : public llvm::StringRef {
public:
  using llvm::StringRef::StringRef;
  CharBlock(const char *begin, const char *end)
      : llvm::StringRef(begin, end - begin) {}
  CharBlock(const char *begin) : llvm::StringRef(begin, 1) {}

  std::string ToString() const { return str(); }

  // Convert to string, stopping early at any embedded '\0'.
  std::string NULTerminatedToString() const {
    return std::string{begin(), strnlen(begin(), size())};
  }
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const CharBlock &x);

} // namespace merkurius::parser

#endif // MERKURIUS_PARSER_CHAR_BLOCK_H_
