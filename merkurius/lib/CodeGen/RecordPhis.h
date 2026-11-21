//===-- lib/CodeGen/RecordPhis.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_LIB_CODEGEN_RECORDPHIS_H
#define MERKURIUS_LIB_CODEGEN_RECORDPHIS_H

#include "llvm/IR/PassManager.h"

namespace merkurius::codegen {

// Splits each phi and select of a flat record, an LLVM struct or array of
// numbers, into one per field, so that a record through a branch costs what
// its fields would. LLVM keeps an aggregate phi whole: on a GPU, where a
// procedure's outputs come back as a struct, the inliner makes one where the
// procedure has several ends, and an if-then-else that gives a record makes
// one or a select. Each field of a whole phi then lives as long as the phi,
// even where nothing reads it, as the outputs a caller ignores with _, and
// InstCombine, GVN and the rest see no field's value through it. SROA does
// this for records in memory, not for SSA values. Once split, the
// shallow-water port with vec3s for its states, increments and eigenvectors
// does the same arithmetic as with three floats each (test/CodeGen/
// record-phis.m).
struct ScalarizeRecordPhisPass
    : public llvm::OptionalPassInfoMixin<ScalarizeRecordPhisPass> {
  llvm::PreservedAnalyses run(
      llvm::Function &function, llvm::FunctionAnalysisManager &);
};

} // namespace merkurius::codegen

#endif // MERKURIUS_LIB_CODEGEN_RECORDPHIS_H
