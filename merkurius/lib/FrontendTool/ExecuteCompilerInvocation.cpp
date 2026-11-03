//===-- lib/FrontendTool/ExecuteCompilerInvocation.cpp ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "merkurius/Driver/Options.h"
#include "merkurius/Frontend/CompilerInstance.h"
#include "merkurius/Frontend/FrontendActions.h"
#include "merkurius/FrontendTool/Utils.h"
#include "merkurius/Version.inc"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

namespace merkurius::frontend {

std::unique_ptr<FrontendAction> createFrontendAction(CompilerInstance &ci) {
  switch (ci.getFrontendOpts().programAction) {
  case ActionKind::PrintPreprocessedInput:
    return std::make_unique<PrintPreprocessedInputAction>();
  case ActionKind::DebugDumpProvenance:
    return std::make_unique<DebugDumpProvenanceAction>();
  case ActionKind::DebugDumpTokens:
    return std::make_unique<DebugDumpTokensAction>();
  case ActionKind::DebugUnparseTokens:
    return std::make_unique<DebugUnparseTokensAction>();
  case ActionKind::DebugDumpTerms:
    return std::make_unique<DebugDumpTermsAction>();
  case ActionKind::DebugDumpParseTreeNoSema:
    return std::make_unique<DebugDumpParseTreeNoSemaAction>();
  case ActionKind::DebugUnparseExact:
    return std::make_unique<DebugUnparseExactAction>(false);
  case ActionKind::DebugUnparseExactMinimalLayout:
    return std::make_unique<DebugUnparseExactAction>(true);
  case ActionKind::DebugUnparseNoSema:
    return std::make_unique<DebugUnparseNoSemaAction>(false);
  case ActionKind::DebugUnparseParenthesizedNoSema:
    return std::make_unique<DebugUnparseNoSemaAction>(true);
  case ActionKind::DebugDumpSemanticForm:
    return std::make_unique<DebugDumpSemanticFormAction>();
  case ActionKind::DebugDumpImports:
    return std::make_unique<DebugDumpImportsAction>();
  case ActionKind::DebugDumpResolvedNames:
    return std::make_unique<DebugDumpResolvedNamesAction>();
  case ActionKind::DebugDumpTypes:
    return std::make_unique<DebugDumpTypesAction>();
  case ActionKind::DebugDumpModes:
    return std::make_unique<DebugDumpModesAction>();
  case ActionKind::DebugDumpDeterminism:
    return std::make_unique<DebugDumpDeterminismAction>();
  case ActionKind::DebugDumpPolymorphism:
    return std::make_unique<DebugDumpPolymorphismAction>();
  case ActionKind::EmitMerk:
  case ActionKind::EmitMLIR:
  case ActionKind::EmitLLVM:
  case ActionKind::EmitLLVMBitcode:
  case ActionKind::EmitAssembly:
  case ActionKind::EmitObj:
    return std::make_unique<CodeGenAction>(ci.getFrontendOpts().programAction);
  }
  llvm_unreachable("unknown ActionKind");
}

bool executeCompilerInvocation(CompilerInstance &ci) {
  const FrontendOptions &opts{ci.getFrontendOpts()};
  if (opts.showHelp) {
    options::getDriverOptTable().printHelp(llvm::outs(),
        "merkc -fc1 [options] file...", "Merkurius frontend",
        /*ShowHidden=*/false, /*ShowAllAliases=*/false,
        llvm::opt::Visibility(options::FC1Option));
    return true;
  }
  if (opts.showVersion) {
    llvm::outs() << "merkc -fc1 version " MERKURIUS_VERSION_STRING "\n";
    return true;
  }
  bool ok{true};
  for (const std::string &input : opts.inputs) {
    std::unique_ptr<FrontendAction> action{createFrontendAction(ci)};
    ok &= action->run(ci, input);
  }
  return ok;
}

} // namespace merkurius::frontend
