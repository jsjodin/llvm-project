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

/// Widen a CIR integer value to a signed 64-bit CIR integer, inserting an
/// integral cast if necessary.
static mlir::Value widenToCirI64(CIRGenBuilderTy &builder, mlir::Location loc,
                                 mlir::Value value) {
  cir::IntType cirI64 = builder.getSInt64Ty();
  if (value.getType() == cirI64)
    return value;
  return builder.createCast(loc, cir::CastKind::integral, value, cirI64);
}

/// Build an omp.map.bounds op for an array section [lb, lb + extent - 1]. The
/// \p lowerBound and \p extent are signed 64-bit CIR integers; the bounds
/// arithmetic is performed in CIR and the operands are then converted to
/// builtin integers, which the OpenMP dialect requires.
static mlir::Value emitSectionBounds(CIRGenBuilderTy &builder,
                                     mlir::Location loc, mlir::Value lowerBound,
                                     mlir::Value extent) {
  cir::IntType cirI64 = builder.getSInt64Ty();
  mlir::Type i64 = builder.getIntegerType(64);
  mlir::Value cirOne = builder.getConstInt(loc, cirI64, 1);
  // upper_bound = lower_bound + extent - 1
  mlir::Value upperBound = builder.createSub(
      loc, builder.createAdd(loc, lowerBound, extent), cirOne);

  auto toBuiltin = [&](mlir::Value v) {
    return builder.createBuiltinIntCast(loc, v, i64);
  };
  mlir::Value zero =
      toBuiltin(builder.getConstInt(loc, cirI64, 0));
  return mlir::omp::MapBoundsOp::create(
      builder, loc, builder.getType<mlir::omp::MapBoundsType>(),
      toBuiltin(lowerBound), toBuiltin(upperBound), toBuiltin(extent),
      /*stride=*/toBuiltin(cirOne), /*stride_in_bytes=*/false,
      /*start_idx=*/zero);
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

/// Emit an omp.map.info for a pointer array section such as `map(p[lb:len])`
/// where `p` is a pointer variable. The mapped entity is the pointed-to data:
/// `var_ptr` is the loaded pointer value and the map carries bounds describing
/// the section. The region receives the (device) data pointer as the block
/// argument. Returns a null value and emits an NYI diagnostic on unsupported
/// forms.
static mlir::Value emitMapInfoForPointerSection(
    CIRGenFunction &cgf, CIRGenModule &cgm, CIRGenBuilderTy &builder,
    mlir::Location loc, const ArraySectionExpr *section, const VarDecl *vd,
    mlir::omp::ClauseMapFlags mapFlags) {
  // Load the pointer value; the pointed-to data is what gets mapped.
  Address ptrAddr = cgf.getAddrOfLocalVar(vd);
  mlir::Value varPtr =
      cir::LoadOp::create(builder, loc, ptrAddr.getPointer()).getResult();
  auto varPtrType = mlir::cast<cir::PointerType>(varPtr.getType());
  mlir::Type elementType = varPtrType.getPointee();

  // Cast to a generic pointer if the loaded value carries an address space.
  if (varPtrType.getAddrSpace()) {
    auto genericPtrType =
        cir::PointerType::get(builder.getContext(), elementType);
    varPtr = cir::CastOp::create(builder, loc, genericPtrType,
                                 cir::CastKind::address_space, varPtr);
    varPtrType = genericPtrType;
  }

  // Bounds are host-only metadata used to compute the transfer size; the device
  // only needs the base pointer.
  llvm::SmallVector<mlir::Value, 1> bounds;
  if (!cgf.getLangOpts().OpenMPIsTargetDevice) {
    mlir::Value lowerBound;
    if (const Expr *lb = section->getLowerBound())
      lowerBound = widenToCirI64(builder, loc, cgf.emitScalarExpr(lb));
    else
      lowerBound = builder.getConstInt(loc, builder.getSInt64Ty(), 0);

    const Expr *lenExpr = section->getLength();
    if (!lenExpr) {
      cgm.errorNYI(section->getExprLoc(),
                   "OpenMP pointer map array section without explicit length");
      return {};
    }
    mlir::Value extent =
        widenToCirI64(builder, loc, cgf.emitScalarExpr(lenExpr));
    bounds.push_back(emitSectionBounds(builder, loc, lowerBound, extent));
  }

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
    llvm::SmallVectorImpl<OMPMapVar> *mapSyms) const {
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
      const Expr *stripped = varExpr->IgnoreParenImpCasts();

      // A pointer array section, e.g. map(p[0:n]) with `p` a pointer, maps the
      // pointed-to data described by the section bounds.
      if (const auto *section = dyn_cast<ArraySectionExpr>(stripped)) {
        const auto *baseRef = dyn_cast<DeclRefExpr>(
            section->getBase()->IgnoreParenImpCasts());
        const auto *vd =
            baseRef ? dyn_cast<VarDecl>(baseRef->getDecl()) : nullptr;
        if (!vd || !vd->getType().getCanonicalType()->isPointerType()) {
          cgm.errorNYI(varExpr->getExprLoc(),
                       "OpenMP map clause array section on non-pointer base");
          continue;
        }
        if (section->getStride()) {
          cgm.errorNYI(varExpr->getExprLoc(),
                       "OpenMP map clause array section with stride");
          continue;
        }
        mlir::Value mapInfo = emitMapInfoForPointerSection(
            cgf, cgm, builder, loc, section, vd, mapFlags);
        if (!mapInfo)
          continue;
        result.mapVars.push_back(mapInfo);
        if (mapSyms)
          mapSyms->push_back({vd, /*isPointerSection=*/true});
        continue;
      }

      const auto *refExpr = dyn_cast<DeclRefExpr>(stripped);
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
        mapSyms->push_back({vd, /*isPointerSection=*/false});
    }
  }
  return found;
}
