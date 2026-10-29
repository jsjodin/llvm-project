//===-- include/merkurius/FrontendTool/Utils.h ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_FRONTENDTOOL_UTILS_H_
#define MERKURIUS_FRONTENDTOOL_UTILS_H_

#include <memory>

namespace merkurius::frontend {

class CompilerInstance;
class FrontendAction;

// Creates the action that the invocation's ActionKind selects.
std::unique_ptr<FrontendAction> createFrontendAction(CompilerInstance &ci);

// Handles -help and -version, then runs the selected action on every input.
// Returns false if anything failed.
bool executeCompilerInvocation(CompilerInstance &ci);

} // namespace merkurius::frontend

#endif // MERKURIUS_FRONTENDTOOL_UTILS_H_
