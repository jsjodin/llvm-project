//===-- include/merkurius/CodeGen/Specializer.h -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_CODEGEN_SPECIALIZER_H_
#define MERKURIUS_CODEGEN_SPECIALIZER_H_

// The specializer (docs/INPROGRESS.md, The specializer: known closures and
// dictionaries): a call given a closure known at the call calls a clone of
// its callee, in which calling the closure is a direct call of its entry.

#include "mlir/IR/BuiltinOps.h"

namespace merkurius::codegen {

// Specializes a merk module's procedures that have bodies on the closures
// their calls pass them, and makes each call of a known closure a call of
// its entry. A closure that nothing uses afterwards is removed.
void SpecializeKnownClosures(mlir::ModuleOp module);

} // namespace merkurius::codegen

#endif // MERKURIUS_CODEGEN_SPECIALIZER_H_
