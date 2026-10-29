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
#include "llvm/Support/thread.h"
#include <optional>

using namespace merkurius::frontend;

// The term reader and the passes over terms recurse once per level of
// nesting, up to the reader's limit of 10000 levels at about 600 bytes a
// level, so the compiler runs on a thread with a stack large enough for
// that, whatever the main thread's stack is.
static constexpr unsigned compilerStackSize{64 << 20};

int fc1_main(llvm::ArrayRef<const char *> argv) {
  int result{1};
  llvm::thread compiler{
      std::optional<unsigned>{compilerStackSize}, [&]() {
        CompilerInstance ci;
        if (CompilerInvocation::createFromArgs(ci.getInvocation(), argv)) {
          result = executeCompilerInvocation(ci) ? 0 : 1;
        }
      }};
  compiler.join();
  return result;
}
