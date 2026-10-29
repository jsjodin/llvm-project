//===-- include/merkurius/Frontend/CompilerInvocation.h ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_FRONTEND_COMPILERINVOCATION_H_
#define MERKURIUS_FRONTEND_COMPILERINVOCATION_H_

#include "merkurius/Frontend/FrontendOptions.h"
#include "llvm/ADT/ArrayRef.h"

namespace merkurius::frontend {

// The options of one -fc1 run, parsed from its command line.
class CompilerInvocation {
public:
  FrontendOptions &getFrontendOpts() { return frontendOpts_; }
  const FrontendOptions &getFrontendOpts() const { return frontendOpts_; }
  LangOptions &getLangOpts() { return langOpts_; }
  const LangOptions &getLangOpts() const { return langOpts_; }

  // Fills `res` from the -fc1 arguments `args`, which exclude "-fc1" itself.
  // Reports errors to standard error and returns false if there were any.
  static bool createFromArgs(
      CompilerInvocation &res, llvm::ArrayRef<const char *> args);

private:
  FrontendOptions frontendOpts_;
  LangOptions langOpts_;
};

} // namespace merkurius::frontend

#endif // MERKURIUS_FRONTEND_COMPILERINVOCATION_H_
