//===-- include/merkurius/Frontend/CompilerInstance.h -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_FRONTEND_COMPILERINSTANCE_H_
#define MERKURIUS_FRONTEND_COMPILERINSTANCE_H_

#include "merkurius/Frontend/CompilerInvocation.h"
#include "merkurius/Parser/provenance.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

namespace merkurius::frontend {

// Owns what one -fc1 run needs: its invocation, the source files and their
// provenances, and its output. The diagnostics and the MLIR context will
// join as the stages that use them arrive.
class CompilerInstance {
public:
  CompilerInvocation &getInvocation() { return invocation_; }
  FrontendOptions &getFrontendOpts() { return invocation_.getFrontendOpts(); }
  LangOptions &getLangOpts() { return invocation_.getLangOpts(); }
  parser::AllSources &getAllSources() { return allSources_; }

  // Opens the output named by -o, or standard output when there is none.
  // Returns null after reporting an error.
  llvm::raw_ostream *createDefaultOutputFile();

private:
  CompilerInvocation invocation_;
  parser::AllSources allSources_;
  std::unique_ptr<llvm::raw_fd_ostream> output_;
};

} // namespace merkurius::frontend

#endif // MERKURIUS_FRONTEND_COMPILERINSTANCE_H_
