//===-- include/merkurius/CodeGen/Lowering.h --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_CODEGEN_LOWERING_H_
#define MERKURIUS_CODEGEN_LOWERING_H_

// Lowering and code generation (design document, "Lowering and code
// generation"): merk to the LLVM dialect, then LLVM IR, then the target's
// assembly or object code. Until terms have layouts, merk is lowered to the
// LLVM dialect directly, as ClangIR's DirectToLLVM lowers CIR; mrt comes
// between them once it does.

#include "merkurius/CodeGen/TargetInfo.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <string>

namespace merkurius::codegen {

// The symbol of the program's entry, a C function of no arguments that the
// runtime's main calls.
inline constexpr llvm::StringLiteral kEntrySymbol{"merk_main"};

// Lowers a merk module to the LLVM dialect, in place. Procedures use LLVM's
// tailcc, so that calls in tail position can be musttail (design document,
// "Calling convention"); merk.entry becomes merk_main.
llvm::LogicalResult LowerToLLVMDialect(mlir::ModuleOp, const TargetInfo &);

// The LLVM IR of a module in the LLVM dialect, with the target's triple and
// data layout; null if it cannot be translated.
std::unique_ptr<llvm::Module> TranslateToLLVMIR(
    mlir::ModuleOp, llvm::LLVMContext &, const TargetInfo &);

// Runs LLVM's pipeline for the target's optimization level, -O0 to -O3.
void OptimizeLLVMIR(llvm::Module &, const TargetInfo &);

// Writes the module's assembly or object code; false after setting `error`
// if the target cannot.
bool EmitCode(llvm::Module &, const TargetInfo &, llvm::CodeGenFileType,
    llvm::raw_pwrite_stream &, std::string &error);

} // namespace merkurius::codegen

#endif // MERKURIUS_CODEGEN_LOWERING_H_
