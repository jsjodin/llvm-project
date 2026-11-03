//===-- lib/CodeGen/Decomposer.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_LIB_CODEGEN_DECOMPOSER_H_
#define MERKURIUS_LIB_CODEGEN_DECOMPOSER_H_

// The decomposition of a compound OpenMP directive, such as parallel do,
// into its leaf constructs, each with the clauses that apply to it, by
// LLVM's ConstructDecompositionT, as Flang and ClangIR decompose theirs
// (design document, "OpenMP directives"). It follows ClangIR's
// CIRGenOpenMPConstructDecomposition.h (THIRD_PARTY.md): a clause is
// converted only as far as the algorithm reads it, and remembers the
// clause it came from, for the leaf's generator to generate.

#include "merkurius/Parser/parse-tree.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Frontend/OpenMP/OMP.h"
#include <optional>

namespace merkurius::codegen::omp {

struct Leaf {
  llvm::omp::Directive id{llvm::omp::Directive::OMPD_unknown};
  // The directive's clauses that apply to the leaf.
  llvm::SmallVector<const parser::OmpClause *> clauses;
  // The clauses the decomposition adds, which the OpenMP specification
  // implies, such as shared for a reduction's variable on the parallel of
  // parallel do; they have no clause of the directive.
  llvm::SmallVector<llvm::omp::Clause> synthesized;
};

// The leaves, outermost first.
using ConstructQueue = llvm::SmallVector<Leaf>;

struct Decomposition {
  ConstructQueue queue;
  // A clause that applies to none of the leaves, when the queue is empty.
  const parser::OmpClause *misplaced{nullptr};
};

// A clause's kind; OMPC_unknown for index and upper, Merkurius's own.
llvm::omp::Clause ClauseId(const parser::OmpClause &);

// Decomposes the directive for `version` of OpenMP. Objects are variables,
// identified by their names, so a clause's variable names the same one as
// the loop's counter, `counter`, which the decomposition must know.
// Merkurius's own clauses take no part: they are not in the queue.
Decomposition Decompose(unsigned version, const parser::OmpDirective &,
    std::optional<llvm::StringRef> counter = std::nullopt);

} // namespace merkurius::codegen::omp
#endif // MERKURIUS_LIB_CODEGEN_DECOMPOSER_H_
