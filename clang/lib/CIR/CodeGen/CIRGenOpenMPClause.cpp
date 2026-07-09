//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// OpenMP clause emitter implementation.
//
//===----------------------------------------------------------------------===//

#include "CIRGenOpenMPClause.h"
#include "CIRGenFunction.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "clang/Basic/OpenMPKinds.h"

using namespace clang;
using namespace clang::CIRGen;

static mlir::omp::ClauseMapFlags
mapClauseKindToFlags(OpenMPMapClauseKind kind) {
  switch (kind) {
  case OMPC_MAP_to:
    return mlir::omp::ClauseMapFlags::to;
  case OMPC_MAP_from:
    return mlir::omp::ClauseMapFlags::from;
  case OMPC_MAP_tofrom:
    return mlir::omp::ClauseMapFlags::to | mlir::omp::ClauseMapFlags::from;
  case OMPC_MAP_alloc:
  case OMPC_MAP_release:
    return mlir::omp::ClauseMapFlags::storage;
  case OMPC_MAP_delete:
    return mlir::omp::ClauseMapFlags::del;
  default:
    return mlir::omp::ClauseMapFlags::none;
  }
}

/// Build an omp.map.bounds op describing the full extent of a fixed-size
/// array. Bounds are normalized (zero-based): lower_bound(0),
/// upper_bound(size - 1), extent(size), stride(1) and start_idx(0). The
/// operands are builtin integers, as required by the OpenMP dialect.
static mlir::Value emitWholeArrayBounds(CIRGenBuilderTy &builder,
                                        mlir::Location loc,
                                        cir::ArrayType arrayType) {
  cir::IntType cirI64 = builder.getSInt64Ty();
  mlir::Type i64 = builder.getIntegerType(64);
  auto constant = [&](int64_t value) -> mlir::Value {
    mlir::Value c = builder.getConstInt(loc, cirI64, value);
    return builder.createBuiltinIntCast(loc, c, i64);
  };

  uint64_t size = arrayType.getSize();
  mlir::Value lowerBound = constant(0);
  mlir::Value upperBound = constant(static_cast<int64_t>(size) - 1);
  mlir::Value extent = constant(static_cast<int64_t>(size));
  mlir::Value stride = constant(1);
  mlir::Value startIdx = constant(0);

  return mlir::omp::MapBoundsOp::create(
      builder, loc, builder.getType<mlir::omp::MapBoundsType>(), lowerBound,
      upperBound, extent, stride, /*stride_in_bytes=*/false, startIdx);
}

static mlir::Value emitMapInfoForVar(CIRGenFunction &cgf,
                                     CIRGenBuilderTy &builder,
                                     mlir::Location loc, const VarDecl *vd,
                                     mlir::omp::ClauseMapFlags mapFlags) {
  Address addr = cgf.getAddrOfLocalVar(vd);
  mlir::Value varPtr = addr.getPointer();
  auto varPtrType = mlir::cast<cir::PointerType>(varPtr.getType());
  mlir::Type elementType = varPtrType.getPointee();

  // Cast to generic pointer if needed.
  if (varPtrType.getAddrSpace()) {
    auto genericPtrType =
        cir::PointerType::get(builder.getContext(), elementType);
    varPtr = cir::CastOp::create(builder, loc, genericPtrType,
                                 cir::CastKind::address_space, varPtr);
    varPtrType = genericPtrType;
  }

  // For a fixed-size array, emit bounds describing the whole array so that the
  // runtime maps the entire buffer rather than just the first element. Bounds
  // are host-only metadata used to compute transfer sizes; the device only
  // needs the base pointer, so they are omitted during device compilation.
  llvm::SmallVector<mlir::Value, 1> bounds;
  if (!cgf.getLangOpts().OpenMPIsTargetDevice)
    if (auto arrayType = mlir::dyn_cast<cir::ArrayType>(elementType))
      bounds.push_back(emitWholeArrayBounds(builder, loc, arrayType));

  return mlir::omp::MapInfoOp::create(
      builder, loc,
      /*omp_ptr=*/varPtrType,
      /*var_ptr=*/varPtr,
      /*var_ptr_type=*/mlir::TypeAttr::get(elementType),
      /*map_type=*/builder.getAttr<mlir::omp::ClauseMapFlagsAttr>(mapFlags),
      /*map_capture_type=*/
      builder.getAttr<mlir::omp::VariableCaptureKindAttr>(
          mlir::omp::VariableCaptureKind::ByRef),
      /*var_ptr_ptr=*/mlir::Value{},
      /*var_ptr_ptr_type=*/mlir::TypeAttr{},
      /*members=*/mlir::ValueRange{},
      /*members_index=*/mlir::ArrayAttr{},
      /*bounds=*/bounds,
      /*mapper_id=*/mlir::FlatSymbolRefAttr{},
      /*name=*/builder.getStringAttr(vd->getName()),
      /*partial_map=*/builder.getBoolAttr(false));
}

bool OpenMPClauseEmitter::emitProcBind(
    mlir::omp::ProcBindClauseOps &result) const {
  for (const OMPClause *clause : clauses) {
    const auto *pbc = dyn_cast<OMPProcBindClause>(clause);
    if (!pbc)
      continue;

    llvm::omp::ProcBindKind kind = pbc->getProcBindKind();
    assert(kind != llvm::omp::ProcBindKind::OMP_PROC_BIND_unknown &&
           "unknown proc-bind kind");
    // The 'default' kind has no dialect counterpart; leave the attribute unset.
    if (kind != llvm::omp::ProcBindKind::OMP_PROC_BIND_default)
      result.procBindKind = mlir::omp::ClauseProcBindKindAttr::get(
          builder.getContext(), mlir::omp::convertProcBindKind(kind));
    return true;
  }
  return false;
}

bool OpenMPClauseEmitter::emitMap(
    mlir::omp::MapClauseOps &result,
    llvm::SmallVectorImpl<const VarDecl *> *mapSyms) const {
  bool found = false;
  for (const OMPClause *clause : clauses) {
    const auto *mc = dyn_cast<OMPMapClause>(clause);
    if (!mc)
      continue;

    found = true;

    for (OpenMPMapModifierKind mod : mc->getMapTypeModifiers()) {
      if (mod != OMPC_MAP_MODIFIER_unknown)
        cgm.errorNYI(mc->getBeginLoc(),
                     std::string("OpenMP map modifier '") +
                         getOpenMPSimpleClauseTypeName(
                             llvm::omp::Clause::OMPC_map, mod) +
                         "'");
    }

    if (mc->isImplicit()) {
      cgm.errorNYI(mc->getBeginLoc(), "OpenMP implicit map clause");
      continue;
    }

    mlir::omp::ClauseMapFlags mapFlags = mapClauseKindToFlags(mc->getMapType());

    for (const Expr *varExpr : mc->varlist()) {
      const auto *refExpr = dyn_cast<DeclRefExpr>(varExpr->IgnoreImplicit());
      if (!refExpr) {
        cgm.errorNYI(varExpr->getExprLoc(),
                     "OpenMP map clause with non-DeclRefExpr variable");
        continue;
      }

      const auto *vd = dyn_cast<VarDecl>(refExpr->getDecl());
      if (!vd) {
        cgm.errorNYI(varExpr->getExprLoc(),
                     "OpenMP map clause with non-VarDecl variable");
        continue;
      }

      // Only whole, fixed-size (constant) arrays of non-array element type are
      // supported. Multi-dimensional and variable-length arrays are NYI.
      QualType varType = vd->getType().getCanonicalType();
      if (varType->isArrayType()) {
        const auto *cat = dyn_cast<ConstantArrayType>(varType);
        if (!cat || cat->getElementType()->isArrayType()) {
          cgm.errorNYI(varExpr->getExprLoc(),
                       "OpenMP map clause with non-constant or "
                       "multi-dimensional array");
          continue;
        }
      }

      result.mapVars.push_back(
          emitMapInfoForVar(cgf, builder, loc, vd, mapFlags));
      if (mapSyms)
        mapSyms->push_back(vd);
    }
  }
  return found;
}
