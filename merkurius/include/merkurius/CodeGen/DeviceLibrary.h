//===-- include/merkurius/CodeGen/DeviceLibrary.h ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_CODEGEN_DEVICELIBRARY_H_
#define MERKURIUS_CODEGEN_DEVICELIBRARY_H_

// The library's code on an OpenMP device (docs/INPROGRESS.md, Library code on
// the device): a device's compile copies the procedures its target regions
// reach from the library's modules, as their merk bytecode has them.

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/StringRef.h"

namespace merkurius::codegen {

// On a procedure copied from the library: the procedure that first reached
// it, and where, which an error on the device names, with what reached that
// one.
inline constexpr llvm::StringLiteral kReachedFromAttrName{"merk.reached_from"};
inline constexpr llvm::StringLiteral kReachedAtAttrName{"merk.reached_at"};

// Copies into `module`, a device's merk module, each procedure of another
// module that its target regions reach, from `directory`/<module>.merkbc,
// with what the copies refer to: the procedures they call, transitively, and
// the descriptors and constructors of their types. Without
// `array`, the array module's procedures are left out, as when the module was
// checked against array.m rather than the sound stub that the library's are
// numbered by. A procedure that is still not defined is reported at the
// call, with the chain of calls from the region.
llvm::LogicalResult CopyDeviceLibrary(
    mlir::ModuleOp module, llvm::StringRef directory, bool array);

// Adds to `diagnostic`, on an operation of `proc`, a note for each procedure
// in the chain that reached `proc` if it was copied from the library: a
// merk.proc, or the function lowering made of one.
void NoteReachedFrom(mlir::InFlightDiagnostic &diagnostic, mlir::Operation *proc);

} // namespace merkurius::codegen

#endif // MERKURIUS_CODEGEN_DEVICELIBRARY_H_
