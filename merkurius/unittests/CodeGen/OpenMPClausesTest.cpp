//===-- unittests/CodeGen/OpenMPClausesTest.cpp ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "merkurius/CodeGen/OpenMPClauses.h"
#include "gtest/gtest.h"
#include <string>

using namespace merkurius::codegen::omp;

namespace {

// Whether some OpenMP version allows the clause on the leaf.
bool Allowed(llvm::omp::Directive leaf, llvm::omp::Clause clause) {
  for (llvm::omp::Version version : llvm::omp::getOpenMPVersions()) {
    if (llvm::omp::isAllowedClauseForDirective(leaf, clause, version)) {
      return true;
    }
  }
  return false;
}

std::string Where(const ClauseTable &table, const LeafClauses &leaf,
    llvm::omp::Clause clause) {
  return table.name.str() + ", " +
      llvm::omp::getOpenMPDirectiveName(leaf.leaf).str() + ": " +
      llvm::omp::getOpenMPClauseName(clause).str();
}

// Each leaf's handled and not yet implemented clauses are disjoint, and
// together they are the clauses that OpenMP allows on it.
TEST(OpenMPClausesTest, ListsCoverAllowedClauses) {
  for (const ClauseTable *table : kClauseTables) {
    for (const LeafClauses &leaf : table->leaves) {
      for (unsigned i{0}; i < llvm::omp::Clause_enumSize; ++i) {
        auto clause{static_cast<llvm::omp::Clause>(i)};
        bool handled{llvm::is_contained(leaf.handled, clause)};
        bool notYet{llvm::is_contained(leaf.notYet, clause)};
        EXPECT_FALSE(handled && notYet)
            << "in both lists: " << Where(*table, leaf, clause);
        EXPECT_EQ(handled || notYet, Allowed(leaf.leaf, clause))
            << (handled || notYet ? "not allowed: " : "in neither list: ")
            << Where(*table, leaf, clause);
      }
    }
  }
}

} // namespace
