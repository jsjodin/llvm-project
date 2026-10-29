//===-- include/merkurius/Frontend/FrontendActions.h ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Frontend actions, layered by how far they run, as Flang's are. Each opens
// its input through the CompilerInstance's AllSources; parsing and analysis
// actions are added with the stages they run.
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_FRONTEND_FRONTENDACTIONS_H_
#define MERKURIUS_FRONTEND_FRONTENDACTIONS_H_

#include "merkurius/Parser/token.h"
#include "llvm/ADT/StringRef.h"

namespace merkurius::parser {
class SourceFile;
} // namespace merkurius::parser

namespace merkurius::frontend {

class CompilerInstance;

class FrontendAction {
public:
  virtual ~FrontendAction() = default;

  // Runs the action on one input file, or on standard input for "-".
  // Returns false after reporting an error.
  bool run(CompilerInstance &ci, llvm::StringRef input);

protected:
  virtual bool executeAction() = 0;

  CompilerInstance &getInstance() { return *instance_; }
  parser::SourceFile &getSourceFile() { return *sourceFile_; }

private:
  CompilerInstance *instance_{nullptr};
  parser::SourceFile *sourceFile_{nullptr};
};

// -E. Prints the source as the lexer reads it, which is the input unchanged.
class PrintPreprocessedInputAction : public FrontendAction {
  bool executeAction() override;
};

// -fdebug-dump-provenance.
class DebugDumpProvenanceAction : public FrontendAction {
  bool executeAction() override;
};

// Actions that run the lexer. Its errors are reported, and the action still
// runs on the tokens, but then fails.
class LexAction : public FrontendAction {
protected:
  bool executeAction() final;
  virtual void executeLexAction(
      llvm::raw_ostream &, const parser::TokenSequence &) = 0;
};

class DebugDumpTokensAction : public LexAction {
  void executeLexAction(
      llvm::raw_ostream &, const parser::TokenSequence &) override;
};

class DebugUnparseTokensAction : public LexAction {
  void executeLexAction(
      llvm::raw_ostream &, const parser::TokenSequence &) override;
};

// Reads the items of the source file and prints their terms.
class DebugDumpTermsAction : public FrontendAction {
  bool executeAction() override;
};

// Converts the source file and prints its parse tree.
class DebugDumpParseTreeNoSemaAction : public FrontendAction {
  bool executeAction() override;
};

} // namespace merkurius::frontend

#endif // MERKURIUS_FRONTEND_FRONTENDACTIONS_H_
