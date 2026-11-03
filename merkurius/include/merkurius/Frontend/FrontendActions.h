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

#include "merkurius/Frontend/FrontendOptions.h"
#include "merkurius/Parser/token.h"
#include "llvm/ADT/StringRef.h"

namespace merkurius::parser {
class SourceFile;
} // namespace merkurius::parser

namespace merkurius::frontend {

class CompilerInstance;

} // namespace merkurius::frontend

namespace llvm {
class Module;
class raw_ostream;
} // namespace llvm

namespace mlir {
class ModuleOp;
} // namespace mlir

namespace merkurius::codegen {
class TargetInfo;
} // namespace merkurius::codegen

namespace merkurius::frontend {

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

// Converts the source file and prints its parse tree back as source, from
// the tokens the tree recorded.
class DebugUnparseExactAction : public FrontendAction {
public:
  explicit DebugUnparseExactAction(bool minimalLayout)
      : minimalLayout_{minimalLayout} {}

private:
  bool executeAction() override;
  bool minimalLayout_;
};

// Converts the source file and prints its parse tree in the canonical
// layout, or with every operator term in parentheses.
class DebugUnparseNoSemaAction : public FrontendAction {
public:
  explicit DebugUnparseNoSemaAction(bool parenthesize)
      : parenthesize_{parenthesize} {}

private:
  bool executeAction() override;
  bool parenthesize_;
};

// Converts the source file, desugars its clauses and prints their semantic
// form.
class DebugDumpSemanticFormAction : public FrontendAction {
  bool executeAction() override;
};

// Loads the modules the source file imports, and lists each module loaded
// with the modules its interface imports and its number of items.
class DebugDumpImportsAction : public FrontendAction {
  bool executeAction() override;
};

// Resolves the names of the source file, and lists the module's symbols
// and, for each name in it, the symbols it may refer to.
class DebugDumpResolvedNamesAction : public FrontendAction {
  bool executeAction() override;
};

// Type checks the source file, and lists, for each clause, the types of
// its variables and the declaration chosen for each overloaded name.
class DebugDumpTypesAction : public FrontendAction {
  bool executeAction() override;
};

// Type and purity checks the source file, translates its loops, and lists
// each clause with loops as translated, followed by its loops' loop and
// body predicates, with their signatures and clauses.
class DebugDumpLoopsAction : public FrontendAction {
  bool executeAction() override;
};

// Type, purity and mode checks the source file, and lists each predicate's
// and function's procedures, with their argument modes and determinisms.
class DebugDumpModesAction : public FrontendAction {
  bool executeAction() override;
};

// Type, purity and mode checks the source file, and lists each checked
// procedure's body as determinism analysis sees it, with its switches.
class DebugDumpDeterminismAction : public FrontendAction {
  bool executeAction() override;
};

// Checks the source file as DebugDumpDeterminismAction does, and lists
// where each checked procedure's type_infos come from.
class DebugDumpPolymorphismAction : public FrontendAction {
  bool executeAction() override;
};

// Checks the source file as DebugDumpPolymorphismAction does, generates its
// procedures' merk operations, and lowers them as far as the action's kind
// says: -emit-merk, -emit-mlir, -emit-llvm, -S or -emit-obj.
class CodeGenAction : public FrontendAction {
public:
  explicit CodeGenAction(ActionKind kind) : kind_{kind} {}

private:
  bool executeAction() override;
  // Lowers the merk module as far as the action's kind says, and writes it.
  bool Lower(mlir::ModuleOp, const codegen::TargetInfo &);
  // Compiles an LLVM IR input as far as the action's kind says (-x ir).
  bool LowerLLVMIR();
  // Embeds the offload objects, optimizes the LLVM module and writes it as
  // the action's kind says.
  bool Emit(llvm::Module &, const codegen::TargetInfo &,
      llvm::raw_ostream &);
  // The output file, named after the source for -emit-obj and -S.
  llvm::raw_ostream *Output();
  ActionKind kind_;
};

} // namespace merkurius::frontend

#endif // MERKURIUS_FRONTEND_FRONTENDACTIONS_H_
