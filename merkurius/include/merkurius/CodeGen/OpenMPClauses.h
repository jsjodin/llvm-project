//===-- include/merkurius/CodeGen/OpenMPClauses.h ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The clauses that code generation takes on each leaf of a directive's
// decomposition, in each kind of construct it generates, as ClangIR's
// OpenMPClauseEmitter::emitNYI lists them: those it handles, and those not
// yet implemented, which are reported so. Together they are every clause
// that some OpenMP version allows on the leaf, and no clause is in both, as
// OpenMPClausesTest checks, so that a clause that a later OpenMP adds is a
// decision rather than an oversight. A clause that the decomposition adds,
// which the specification implies, is handled if it is shared, since a
// variable is shared already, or in the leaf's handled or implied clauses,
// as the map of a reduction's variable that target implies, which written
// in the directive is not handled yet.

#ifndef MERKURIUS_CODEGEN_OPENMPCLAUSES_H_
#define MERKURIUS_CODEGEN_OPENMPCLAUSES_H_

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Frontend/OpenMP/OMP.h"

namespace merkurius::codegen::omp {

struct LeafClauses {
  llvm::omp::Directive leaf;
  llvm::ArrayRef<llvm::omp::Clause> handled;
  llvm::ArrayRef<llvm::omp::Clause> notYet;
  llvm::ArrayRef<llvm::omp::Clause> implied{};
};

// The leaves of one kind of construct.
struct ClauseTable {
  llvm::StringLiteral name;
  llvm::ArrayRef<LeafClauses> leaves;
};

enum class ClauseStatus { Handled, NotYet, Unlisted };

// A clause of a leaf, written in the directive or, `synthesized`, added by
// the decomposition.
inline ClauseStatus StatusOf(const ClauseTable &table,
    llvm::omp::Directive leaf, llvm::omp::Clause clause, bool synthesized) {
  if (synthesized && clause == llvm::omp::OMPC_shared) {
    return ClauseStatus::Handled;
  }
  for (const LeafClauses &leaves : table.leaves) {
    if (leaves.leaf != leaf) {
      continue;
    }
    if (llvm::is_contained(leaves.handled, clause) ||
        (synthesized && llvm::is_contained(leaves.implied, clause))) {
      return ClauseStatus::Handled;
    }
    if (llvm::is_contained(leaves.notYet, clause)) {
      return ClauseStatus::NotYet;
    }
  }
  return ClauseStatus::Unlisted;
}

namespace clauses {
using namespace llvm::omp;

// The clauses of each leaf that no construct handles yet.
inline constexpr Clause kDistribute[]{OMPC_allocate, OMPC_collapse,
    OMPC_dist_schedule, OMPC_firstprivate, OMPC_lastprivate, OMPC_order};
inline constexpr Clause kParallel[]{OMPC_allocate, OMPC_copyin, OMPC_default,
    OMPC_firstprivate, OMPC_if, OMPC_message, OMPC_num_threads,
    OMPC_ompx_attribute, OMPC_proc_bind, OMPC_severity, OMPC_shared};
inline constexpr Clause kDo[]{OMPC_allocate, OMPC_collapse, OMPC_firstprivate,
    OMPC_lastprivate, OMPC_linear, OMPC_nowait, OMPC_order, OMPC_ordered};
inline constexpr Clause kTargetData[]{OMPC_default, OMPC_if, OMPC_transparent,
    OMPC_use_device_addr, OMPC_use_device_ptr};

inline constexpr Clause kImpliedMap[]{OMPC_map};

// Loops under parallel do and target teams distribute parallel do. private
// names variables, which are local to each iteration or shared, and so is
// checked rather than generated (design document, "Loops", Under a
// parallel directive).
inline constexpr Clause kLoopTargetHandled[]{OMPC_device, OMPC_private};
inline constexpr Clause kLoopTargetNotYet[]{OMPC_allocate, OMPC_default,
    OMPC_defaultmap, OMPC_depend, OMPC_dyn_groupprivate, OMPC_firstprivate,
    OMPC_has_device_addr, OMPC_if, OMPC_in_reduction, OMPC_is_device_ptr,
    OMPC_map, OMPC_nowait, OMPC_ompx_attribute, OMPC_ompx_bare,
    OMPC_ompx_dyn_cgroup_mem, OMPC_replayable, OMPC_thread_limit,
    OMPC_uses_allocators};
inline constexpr Clause kLoopTeamsHandled[]{OMPC_private, OMPC_reduction};
inline constexpr Clause kLoopTeamsNotYet[]{OMPC_allocate, OMPC_default,
    OMPC_dyn_groupprivate, OMPC_firstprivate, OMPC_if, OMPC_num_teams,
    OMPC_ompx_attribute, OMPC_shared, OMPC_thread_limit};
inline constexpr Clause kLoopDistributeHandled[]{OMPC_private};
inline constexpr Clause kLoopParallelHandled[]{OMPC_private, OMPC_reduction};
inline constexpr Clause kLoopDoHandled[]{
    OMPC_private, OMPC_reduction, OMPC_schedule};
inline constexpr LeafClauses kLoopLeaves[]{
    {OMPD_target, kLoopTargetHandled, kLoopTargetNotYet, kImpliedMap},
    {OMPD_teams, kLoopTeamsHandled, kLoopTeamsNotYet},
    {OMPD_distribute, kLoopDistributeHandled, kDistribute},
    {OMPD_parallel, kLoopParallelHandled, kParallel},
    {OMPD_do, kLoopDoHandled, kDo}};

// A teams distribute loop in a GPU process, whose teams leaf may name
// group-private arrays.
inline constexpr Clause kProcessTeamsHandled[]{
    OMPC_dyn_groupprivate, OMPC_private, OMPC_reduction};
inline constexpr Clause kProcessTeamsNotYet[]{OMPC_allocate, OMPC_default,
    OMPC_firstprivate, OMPC_if, OMPC_num_teams, OMPC_ompx_attribute,
    OMPC_shared, OMPC_thread_limit};
inline constexpr LeafClauses kProcessDistributeLeaves[]{
    {OMPD_teams, kProcessTeamsHandled, kProcessTeamsNotYet},
    {OMPD_distribute, kLoopDistributeHandled, kDistribute}};

// target and target teams around a goal.
inline constexpr Clause kTargetHandled[]{OMPC_device};
inline constexpr Clause kTargetNotYet[]{OMPC_allocate, OMPC_default,
    OMPC_defaultmap, OMPC_depend, OMPC_dyn_groupprivate, OMPC_firstprivate,
    OMPC_has_device_addr, OMPC_if, OMPC_in_reduction, OMPC_is_device_ptr,
    OMPC_map, OMPC_nowait, OMPC_ompx_attribute, OMPC_ompx_bare,
    OMPC_ompx_dyn_cgroup_mem, OMPC_private, OMPC_replayable, OMPC_thread_limit,
    OMPC_uses_allocators};
inline constexpr Clause kTeamsHandled[]{OMPC_num_teams};
inline constexpr Clause kTeamsNotYet[]{OMPC_allocate, OMPC_default,
    OMPC_dyn_groupprivate, OMPC_firstprivate, OMPC_if, OMPC_ompx_attribute,
    OMPC_private, OMPC_reduction, OMPC_shared, OMPC_thread_limit};
inline constexpr LeafClauses kTargetLeaves[]{
    {OMPD_target, kTargetHandled, kTargetNotYet},
    {OMPD_teams, kTeamsHandled, kTeamsNotYet}};

// target data around a goal.
inline constexpr Clause kTargetDataHandled[]{OMPC_device, OMPC_map};
inline constexpr LeafClauses kTargetDataLeaves[]{
    {OMPD_target_data, kTargetDataHandled, kTargetData}};
} // namespace clauses

inline constexpr ClauseTable kLoopClauses{
    "parallel loops", clauses::kLoopLeaves};
inline constexpr ClauseTable kProcessDistributeClauses{
    "teams distribute loops in GPU processes",
    clauses::kProcessDistributeLeaves};
inline constexpr ClauseTable kTargetClauses{
    "target regions", clauses::kTargetLeaves};
inline constexpr ClauseTable kTargetDataClauses{
    "target data regions", clauses::kTargetDataLeaves};
inline constexpr const ClauseTable *kClauseTables[]{&kLoopClauses,
    &kProcessDistributeClauses, &kTargetClauses, &kTargetDataClauses};

} // namespace merkurius::codegen::omp
#endif // MERKURIUS_CODEGEN_OPENMPCLAUSES_H_
