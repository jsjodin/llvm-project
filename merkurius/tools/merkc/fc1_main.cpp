//===-- tools/merkc/fc1_main.cpp - Merkurius frontend -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The entry point of merkc -fc1, which compiles one module.
//
//===----------------------------------------------------------------------===//

#include "merkurius/Frontend/CompilerInstance.h"
#include "merkurius/Frontend/CompilerInvocation.h"
#include "merkurius/FrontendTool/Utils.h"

using namespace merkurius::frontend;

int fc1_main(llvm::ArrayRef<const char *> argv) {
  CompilerInstance ci;
  if (!CompilerInvocation::createFromArgs(ci.getInvocation(), argv)) {
    return 1;
  }
  return executeCompilerInvocation(ci) ? 0 : 1;
}
