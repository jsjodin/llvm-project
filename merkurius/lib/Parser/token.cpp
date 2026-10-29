//===-- lib/Parser/token.cpp ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "merkurius/Parser/token.h"
#include "merkurius/Parser/source.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"

namespace merkurius::parser {

llvm::StringRef GetTokenKindName(TokenKind kind) {
  switch (kind) {
  case TokenKind::Name:
    return "name";
  case TokenKind::Variable:
    return "variable";
  case TokenKind::Integer:
    return "integer";
  case TokenKind::Float:
    return "float";
  case TokenKind::String:
    return "string";
  case TokenKind::ImplementationDefinedLiteral:
    return "implementation_defined_literal";
  case TokenKind::Backquote:
    return "backquote";
  case TokenKind::OpenCt:
    return "open_ct";
  case TokenKind::Open:
    return "open";
  case TokenKind::Close:
    return "close";
  case TokenKind::OpenList:
    return "open_list";
  case TokenKind::CloseList:
    return "close_list";
  case TokenKind::OpenCurly:
    return "open_curly";
  case TokenKind::CloseCurly:
    return "close_curly";
  case TokenKind::HtSep:
    return "ht_sep";
  case TokenKind::Comma:
    return "comma";
  case TokenKind::End:
    return "end";
  case TokenKind::Directive:
    return "directive";
  case TokenKind::Eof:
    return "eof";
  case TokenKind::Error:
    return "error";
  }
  llvm_unreachable("unknown token kind");
}

// Control characters are escaped so that each token takes one line.
static void PrintSpelling(llvm::raw_ostream &o, CharBlock spelling) {
  for (char ch : spelling) {
    switch (ch) {
    case '\n':
      o << "\\n";
      break;
    case '\r':
      o << "\\r";
      break;
    case '\t':
      o << "\\t";
      break;
    default:
      if (static_cast<unsigned char>(ch) < 0x20 || ch == 0x7f) {
        o << llvm::format("\\x%02x", static_cast<unsigned char>(ch));
      } else {
        o << ch;
      }
    }
  }
}

static void DumpToken(llvm::raw_ostream &o, const Token &token,
    const SourceFile &file, llvm::StringRef indent) {
  SourcePosition pos{file.GetSourcePosition(
      static_cast<std::size_t>(token.spelling.begin() - file.content().begin()))};
  o << indent << pos.line << ':' << pos.column << ' '
    << GetTokenKindName(token.kind);
  if (token.kind != TokenKind::Directive && !token.spelling.empty()) {
    o << ' ';
    PrintSpelling(o, token.spelling);
  }
  o << '\n';
}

void DumpTokens(llvm::raw_ostream &o, const TokenSequence &tokens,
    const SourceFile &file) {
  for (const Token &token : tokens.tokens) {
    DumpToken(o, token, file, "");
    for (std::size_t j{0}; j < token.directiveTokens; ++j) {
      DumpToken(o, tokens.directiveTokens[token.firstDirectiveToken + j], file,
          "  ");
    }
  }
}

void UnparseTokens(llvm::raw_ostream &o, const TokenSequence &tokens) {
  for (const Token &token : tokens.tokens) {
    o << token.trivia << token.spelling;
  }
}

} // namespace merkurius::parser
