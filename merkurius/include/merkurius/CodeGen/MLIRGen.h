//===-- include/merkurius/CodeGen/MLIRGen.h ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_CODEGEN_MLIRGEN_H_
#define MERKURIUS_CODEGEN_MLIRGEN_H_

// MLIR generation (design document, "Semantic analysis", "Mapping to
// merk"): the module's procedures as merk operations, read from the
// procedure bodies that determinism analysis made, one procedure at a time.

#include "merkurius/CodeGen/TargetInfo.h"
#include "merkurius/Parser/message.h"
#include "merkurius/Parser/provenance.h"
#include "merkurius/Parser/token.h"
#include "merkurius/Semantics/check-determinism.h"
#include "merkurius/Semantics/check-modes.h"
#include "merkurius/Semantics/polymorphism.h"
#include "merkurius/Semantics/translate-loops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include <string>

namespace merkurius::codegen {

// The symbol of a procedure: its module, then its name, its arity, with an
// `f` before it for a function, and its mode number, joined by `_`, as in
// `hello.main_2_0` or `list.length_f1_0`. A character of a name other than
// a letter, digit or `_` becomes `$` and its two hex digits, so that
// `int.+` is `int.$2b_f2_0`.
std::string MangledName(const semantics::Procedure &);

// The module attribute that holds a module's foreign code: a C file of the
// module's foreign_decl and foreign_code for C, and a function around the
// code of each of its foreign_procs for C, which merk.foreign_call calls.
inline constexpr llvm::StringLiteral kForeignCodeAttrName{"merk.foreign_code"};

// The merk module for `module`'s procedures, with the target recorded on
// it. A construct that code generation does not handle yet is reported as
// not yet implemented, and its procedure left out.
mlir::OwningOpRef<mlir::ModuleOp> GenerateMerk(mlir::MLIRContext &,
    llvm::StringRef module, const parser::ParseTree &,
    const parser::TokenSequence &, const parser::AllSources &,
    const semantics::TypeChecking &, semantics::ModeChecking &,
    const semantics::DeterminismChecking &,
    const semantics::PolymorphismAnalysis &, const semantics::LoopTranslation &,
    const TargetInfo &, parser::Messages &);

} // namespace merkurius::codegen

#endif // MERKURIUS_CODEGEN_MLIRGEN_H_
