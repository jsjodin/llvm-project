//===--- CIRGenOpenMPClause.h - OpenMP clause emitter -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_CODEGEN_CIRGENOPENMPCLAUSE_H
#define LLVM_CLANG_LIB_CIR_CODEGEN_CIRGENOPENMPCLAUSE_H

#include "CIRGenBuilder.h"
#include "CIRGenModule.h"
#include "mlir/Dialect/OpenMP/OpenMPClauseOperands.h"
#include "clang/AST/OpenMPClause.h"
#include "clang/AST/StmtOpenMP.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"

#include <type_traits>

namespace clang::CIRGen {

class CIRGenFunction;

/// Describes a variable referenced by a map clause, along with how the target
/// region body must access it.
struct OMPMapVar {
  const VarDecl *decl = nullptr;
  /// True for a pointer array section such as `map(p[0:n])` where `p` is a
  /// pointer. The map operand (and hence the region block argument) is the
  /// pointed-to data, so the region body needs a local slot holding that
  /// pointer to preserve the usual "load pointer, then index" access pattern.
  bool isPointerSection = false;
};

/// A type-only list of OpenMP clause AST node types.
template <typename... Clauses> struct OpenMPNYIClauseList {};

/// Emits OpenMP clauses for a directive, writing results into the
/// auto-generated ClauseOps from the OMP dialect.
class OpenMPClauseEmitter {
  CIRGenFunction &cgf;
  CIRGenModule &cgm;
  CIRGenBuilderTy &builder;
  mlir::Location loc;
  llvm::ArrayRef<const OMPClause *> clauses;

public:
  OpenMPClauseEmitter(CIRGenFunction &cgf, CIRGenModule &cgm,
                      CIRGenBuilderTy &builder, mlir::Location loc,
                      llvm::ArrayRef<const OMPClause *> clauses)
      : cgf(cgf), cgm(cgm), builder(builder), loc(loc), clauses(clauses) {}

  bool emitProcBind(mlir::omp::ProcBindClauseOps &result) const;

  /// Emit map clauses. The optional \p mapSyms parameter collects the
  /// variables corresponding to each map operand, in operand order.
  bool emitMap(mlir::omp::MapClauseOps &result,
               llvm::SmallVectorImpl<OMPMapVar> *mapSyms = nullptr) const;

  /// Verify the clauses of a directive to make sure all legal cases are either
  /// implemented or give a NYI error. The \p SupportedClauses and \p
  /// NYIClauses type lists must be disjoint and cover all clauses eligible for
  /// the directive being processed.
  template <typename... SupportedClauses, typename... NYIClauses>
  void emitNYI(OpenMPNYIClauseList<NYIClauses...> nyi,
               llvm::omp::Directive directive) const;

private:
  /// True if T is the same type as any of Ts.
  template <typename T, typename... Ts>
  static constexpr bool isAnyOf = (std::is_same_v<T, Ts> || ...);
};

template <typename... SupportedClauses, typename... NYIClauses>
void OpenMPClauseEmitter::emitNYI(OpenMPNYIClauseList<NYIClauses...>,
                                  llvm::omp::Directive directive) const {
  static_assert(
      (!isAnyOf<NYIClauses, SupportedClauses...> && ...),
      "the supported and not-yet-implemented clause lists must be disjoint");

  for (const OMPClause *c : clauses) {
    if (isa<NYIClauses...>(c)) {
      std::string msg =
          (llvm::Twine("OpenMP ") +
           llvm::omp::getOpenMPDirectiveName(directive).upper() + " '" +
           llvm::omp::getOpenMPClauseName(c->getClauseKind()) + "' clause")
              .str();
      cgm.errorNYI(c->getBeginLoc(), msg);
      continue;
    }
    // A directive with no supported clauses (empty SupportedClauses pack) treats
    // every eligible clause as not-yet-implemented, so anything reaching here is
    // an unknown/illegal clause.
    if constexpr (sizeof...(SupportedClauses) > 0)
      if (isa<SupportedClauses...>(c))
        continue;
    llvm_unreachable("unexpected OpenMP clause");
  }
}

} // namespace clang::CIRGen

#endif // LLVM_CLANG_LIB_CIR_CODEGEN_CIRGENOPENMPCLAUSE_H
