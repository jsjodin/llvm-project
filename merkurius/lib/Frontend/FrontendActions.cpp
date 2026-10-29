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
#include <string>

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

} // namespace merkurius::frontend
