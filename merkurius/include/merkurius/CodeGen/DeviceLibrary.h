//===-- include/merkurius/CodeGen/DeviceLibrary.h ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_CODEGEN_DEVICELIBRARY_H_
#define MERKURIUS_CODEGEN_DEVICELIBRARY_H_

// The library's code on an OpenMP device (design document, Library code on
// the device): a device's compile copies the procedures its target regions
// reach from the library's modules, as their merk bytecode has them.

#include "merkurius/CodeGen/TargetInfo.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/StringRef.h"

namespace merkurius::codegen {

// On a procedure copied from the library: the procedure that first reached
// it, and where, which an error on the device names, with what reached that
// one.
inline constexpr llvm::StringLiteral kReachedFromAttrName{"merk.reached_from"};
inline constexpr llvm::StringLiteral kReachedAtAttrName{"merk.reached_at"};

// On a device's module: the foreign code of each library module that a copy
// came from with a foreign_proc, by the module's name; and what
// CompileDeviceForeignCode compiled for the device, as bitcode, which
// LinkForeignCode links after translation.
inline constexpr llvm::StringLiteral kDeviceForeignCodeAttrName{
    "merk.device_foreign_code"};
inline constexpr llvm::StringLiteral kDeviceForeignBitcodeAttrName{
    "merk.device_foreign_bitcode"};

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

// Compiles for the device, with `clang`, the C functions of the foreign_procs
// that a device's module calls, its own and the library's, and only those, as
// a module's foreign code defines each one only when the device asks for it.
// What one of them needs that is not on a device is reported at the call: a
// function that is neither an intrinsic, the device's runtime, OpenMP's, nor
// in libc or libm for the GPU, which the device's image has.
llvm::LogicalResult CompileDeviceForeignCode(
    mlir::ModuleOp module, const TargetInfo &target, llvm::StringRef clang);

// The guard around a foreign_proc's C function `name` in a module's foreign
// code: defined on the host, and on a device when its compile asks for it.
std::string ForeignFunctionGuard(llvm::StringRef name);

// Adds to `diagnostic`, on an operation of `proc`, a note for each procedure
// in the chain that reached `proc` if it was copied from the library: a
// merk.proc, or the function lowering made of one.
void NoteReachedFrom(mlir::InFlightDiagnostic &diagnostic, mlir::Operation *proc);

} // namespace merkurius::codegen

#endif // MERKURIUS_CODEGEN_DEVICELIBRARY_H_
