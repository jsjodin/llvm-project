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
