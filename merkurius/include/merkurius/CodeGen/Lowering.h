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
// generation"): merk to the LLVM dialect, terms through mrt, then LLVM IR,
// then the target's assembly or object code.

#include <cstdint>
#include "merkurius/CodeGen/TargetInfo.h"
#include "merkurius/Dialect/Merk/IR/MerkDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <string>
#include <vector>

namespace merkurius::codegen {

// The symbol of the program's entry, a C function of no arguments that the
// runtime's main calls.
inline constexpr llvm::StringLiteral kEntrySymbol{"merk_main"};

// Leaves out of the solutions of merk.each's and merk.commit's generators
// the values that nothing after them uses.
void PruneSolutions(mlir::ModuleOp);

// Rewrites a merk module's nondeterministic code, in regions, as procedures
// that take continuations (design document, Lowering, "Nondeterministic
// code"). LowerToLLVMDialect does this first.
void OutlineNondeterminism(mlir::ModuleOp);

// Puts a merk.live before each safepoint (Liveness.cpp).
void MarkSafepoints(mlir::ModuleOp);

// Whether a term of a type and functor is a cell on the heap, which can
// have a hole.
using InCell = llvm::function_ref<bool(mlir::Type, merk::FunctorAttr)>;

// Makes a call whose result goes into a cell the procedure returns a tail
// call, through a variant of the callee that fills a hole in the cell
// (design document, Lowering, "Deterministic and semidet code", "Last call
// modulo constructor"). Lowering does this once it knows the types'
// layouts, at every optimization level.
void LastCallModuloConstructor(mlir::ModuleOp, InCell);

// Lowers a merk module to the LLVM dialect, in place. Procedures use LLVM's
// tailcc, so that calls in tail position can be musttail (design document,
// "Calling convention"); merk.entry becomes merk_main.
// A device's compile compiles the foreign code its regions reach with
// `clang`.
llvm::LogicalResult LowerToLLVMDialect(
    mlir::ModuleOp, const TargetInfo &, llvm::StringRef clang = {});

// The LLVM IR of a module in the LLVM dialect, with the target's triple and
// data layout; null if it cannot be translated.
std::unique_ptr<llvm::Module> TranslateToLLVMIR(
    mlir::ModuleOp, llvm::LLVMContext &, const TargetInfo &);

// clang's options for compiling foreign code for the target: its triple,
// CPU and optimization level.
std::vector<std::string> ForeignCodeOptions(const TargetInfo &);

// Compiles C `code`, foreign code, with clang and `options` into LLVM
// `bitcode`; false if it cannot, after setting `error` unless clang ran and
// reported the code's errors itself.
bool CompileForeignCode(llvm::StringRef clang, llvm::StringRef code,
    llvm::ArrayRef<std::string> options, std::string &bitcode,
    std::string &error);

// Compiles the module's foreign code (its merk.foreign_code attribute), if
// it has any, with clang, and links it into its LLVM IR; false after
// setting `error` if it cannot. A device's compile links what lowering
// compiled of it and of the library's, for what its regions reach
// (CompileDeviceForeignCode).
bool LinkForeignCode(llvm::Module &, mlir::ModuleOp, const TargetInfo &,
    llvm::StringRef clang, std::string &error);

// Runs LLVM's pipeline for the target's optimization level, -O0 to -O3.
// A deopt bundle starting with this is of a call in an OpenMP region, which
// is not a safepoint.
inline constexpr std::int64_t kInRegion{-2};

// A module's table of root types, in the section the runtime finds them by.
inline constexpr llvm::StringLiteral kRootTable{"merk.root_types"};

// The order of runtime/roots.cpp's RootKind.
enum RootKind : std::int64_t {
  kRootUnresolved,
  kRootTerm,
  kRootTypeInfo,
  kRootTypeclassInfo,
  kRootRef,
  kRootClosure,
  kRootString,
  kRootArray,
  kRootEnvironment,
  kRootPolymorphicTerm,
  kRootWord,
  kRootBits,
};

// Gives Mercury code statepoints, after the optimizer (Statepoints.cpp).
void RewriteForCollector(llvm::Module &, const TargetInfo &);

void OptimizeLLVMIR(llvm::Module &, const TargetInfo &);

// Adds the check of the stack's limit, which finds an overflow and is how a
// process is preempted, to the optimized procedures and loops that need it
// (design document, "Processes and scheduler"). It comes
// after the optimizer, since a check at a procedure's start stops LLVM from
// turning its recursions into loops, and so that it sees the calls that are
// left.
void AddStackChecks(llvm::Module &);

// Replaces the optimizer's stand-in for the address of the thread's heap,
// which keeps a process that moves to another thread from using the old
// thread's heap, with the thread-local address.
void ExpandHeapAddresses(llvm::Module &);

// Embeds each offload binary in the host module's .llvm.offloading section,
// for the linker wrapper to extract (-fembed-offload-object=, as in Clang
// and Flang), or reports false with the error.
bool EmbedOffloadObjects(llvm::Module &, const std::vector<std::string> &files,
    std::string &error);

// Writes the module's assembly or object code; false after setting `error`
// if the target cannot.
bool EmitCode(llvm::Module &, const TargetInfo &, llvm::CodeGenFileType,
    llvm::raw_pwrite_stream &, std::string &error);

} // namespace merkurius::codegen

#endif // MERKURIUS_CODEGEN_LOWERING_H_
