//===-- lib/Frontend/FrontendActions.cpp ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "merkurius/Frontend/FrontendActions.h"
#include "merkurius/Driver/Options.h"
#include "merkurius/Frontend/CompilerInstance.h"
#include "merkurius/Parser/lexer.h"
#include "merkurius/Parser/message.h"
#include "merkurius/Parser/provenance.h"
#include "merkurius/Parser/source.h"
#include "merkurius/Parser/term-reader.h"
#include "llvm/ADT/STLExtras.h"
#include <string>
#include <vector>

namespace merkurius::frontend {

bool FrontendAction::run(CompilerInstance &ci, llvm::StringRef input) {
  parser::AllSources &allSources{ci.getAllSources()};
  std::string error;
  llvm::raw_string_ostream errorStream{error};
  parser::SourceFile *sourceFile{input == "-"
          ? allSources.ReadStandardInput(errorStream)
          : allSources.OpenPath(input.str(), errorStream)};
  if (!sourceFile) {
    options::reportError("cannot open input file '" + input + "': " + error);
    return false;
  }
  instance_ = &ci;
  sourceFile_ = sourceFile;
  bool ok{executeAction()};
  sourceFile_ = nullptr;
  instance_ = nullptr;
  return ok;
}

bool PrintPreprocessedInputAction::executeAction() {
  llvm::raw_ostream *os{getInstance().createDefaultOutputFile()};
  if (!os) {
    return false;
  }
  if (getSourceFile().hasByteOrderMark()) {
    *os << "\xef\xbb\xbf";
  }
  *os << getSourceFile().content();
  os->flush();
  return true;
}

bool DebugDumpProvenanceAction::executeAction() {
  llvm::raw_ostream *os{getInstance().createDefaultOutputFile()};
  if (!os) {
    return false;
  }
  getInstance().getAllSources().Dump(*os);
  os->flush();
  return true;
}

bool LexAction::executeAction() {
  CompilerInstance &ci{getInstance()};
  parser::LexerOptions options;
  options.openMP = ci.getLangOpts().openMP;
  parser::Messages messages;
  parser::TokenSequence tokens{parser::Lex(getSourceFile(), options, messages)};
  messages.Emit(llvm::errs(), ci.getAllSources());
  llvm::raw_ostream *os{ci.createDefaultOutputFile()};
  if (!os) {
    return false;
  }
  executeLexAction(*os, tokens);
  os->flush();
  return !messages.AnyFatalError();
}

void DebugDumpTokensAction::executeLexAction(
    llvm::raw_ostream &os, const parser::TokenSequence &tokens) {
  parser::DumpTokens(os, tokens, getSourceFile());
}

void DebugUnparseTokensAction::executeLexAction(
    llvm::raw_ostream &os, const parser::TokenSequence &tokens) {
  parser::UnparseTokens(os, tokens);
}

bool DebugDumpTermsAction::executeAction() {
  CompilerInstance &ci{getInstance()};
  parser::LexerOptions options;
  options.openMP = ci.getLangOpts().openMP;
  parser::Messages messages;
  parser::TokenSequence tokens{parser::Lex(getSourceFile(), options, messages)};
  // The lexer's messages, by location, to find the items they fall in.
  std::vector<const char *> lexerErrors;
  for (const parser::Message &message : messages.messages()) {
    if (const parser::CharBlock * at{message.GetCharBlock()}) {
      lexerErrors.push_back(at->begin());
    }
  }
  llvm::raw_ostream *os{ci.createDefaultOutputFile()};
  if (!os) {
    return false;
  }
  parser::TermArena arena;
  parser::TermReader reader{tokens, arena, messages};
  std::size_t readerMessages{messages.messages().size()};
  while (std::optional<parser::Item> item{reader.ReadItem()}) {
    const parser::Token &first{tokens.tokens[item->tokens.first]};
    const parser::Token &last{tokens.tokens[item->tokens.last]};
    bool lexerError{llvm::any_of(lexerErrors, [&](const char *at) {
      return at >= first.spelling.begin() && at <= last.spelling.end();
    })};
    bool readerError{messages.messages().size() != readerMessages};
    readerMessages = messages.messages().size();
    if (item->term && !lexerError && !readerError) {
      parser::PrintCanonical(*os, *item->term);
    } else {
      *os << "error";
    }
    *os << '\n';
    arena.Reset();
  }
  os->flush();
  messages.Emit(llvm::errs(), ci.getAllSources());
  return !messages.AnyFatalError();
}

} // namespace merkurius::frontend
