//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Emit OpenMP Stmt nodes as CIR code.
//
//===----------------------------------------------------------------------===//

#include "CIRGenBuilder.h"
#include "CIRGenFunction.h"
#include "CIRGenOpenMPClause.h"
#include "CIRGenOpenMPConstructDecomposition.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "clang/AST/OpenMPClause.h"
#include "clang/AST/StmtOpenMP.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Frontend/OpenMP/OMP.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"
using namespace clang;
using namespace clang::CIRGen;

mlir::LogicalResult
CIRGenFunction::emitOMPScopeDirective(const OMPScopeDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPScopeDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPErrorDirective(const OMPErrorDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPErrorDirective");
  return mlir::failure();
}

/// Returns the clauses the OpenMP spec assigns to \p leaf within \p s. For a
/// leaf directive this is just its own clauses; combined/composite directives
/// carry the union of their leaves' clauses, so we run the shared construct
/// decomposition (tomp::ConstructDecompositionT, as Flang does) to distribute
/// them.
///
/// The decomposition can synthesize clauses that were not written by the user
/// and have no Clang AST node (e.g. a `shared` on `parallel` derived from a
/// `lastprivate`). Those cannot go through the AST-based emitters, so we report
/// any synthesized clause landing on the lowered leaf as not-yet-implemented
/// rather than dropping a spec-required clause. This is currently unreachable:
/// CIR only lowers the leaf `parallel`/`target` directives, which trigger no
/// synthesis.
static llvm::SmallVector<const OMPClause *>
getLeafClauses(CIRGenFunction &cgf, const OMPExecutableDirective &s,
               llvm::omp::Directive leaf) {
  unsigned version = cgf.getContext().getLangOpts().OpenMP;
  llvm::SmallVector<omp::LeafWithClauses> leaves = omp::decompose(version, s);

  llvm::SmallVector<const OMPClause *> result;
  for (const omp::LeafWithClauses &l : leaves) {
    if (l.id != leaf)
      continue;
    for (llvm::omp::Clause synth : l.synthesized)
      cgf.getCIRGenModule().errorNYI(
          s.getSourceRange(),
          (llvm::Twine("OpenMP synthesized '") +
           llvm::omp::getOpenMPClauseName(synth) +
           "' clause from construct decomposition")
              .str());
    llvm::append_range(result, l.clauses);
  }
  return result;
}

template <typename DirectiveTy>
static mlir::LogicalResult
emitParallelOp(CIRGenFunction &cgf, const DirectiveTy &s, mlir::Location begin,
               mlir::Location end,
               llvm::function_ref<mlir::LogicalResult()> emitBody) {
  CIRGenBuilderTy &builder = cgf.getBuilder();
  CIRGenModule &cgm = cgf.getCIRGenModule();

  llvm::SmallVector<const OMPClause *> clauses =
      getLeafClauses(cgf, s, llvm::omp::OMPD_parallel);

  mlir::omp::ParallelOperands clauseOps;
  OpenMPClauseEmitter ce(cgf, cgm, builder, begin, clauses);
  ce.emitProcBind(clauseOps);
  ce.emitNYI</*supported=*/OMPProcBindClause>(
      /*nyi=*/OpenMPNYIClauseList<
          OMPAllocateClause, OMPCopyinClause, OMPDefaultClause,
          OMPFirstprivateClause, OMPIfClause, OMPNumThreadsClause,
          OMPPrivateClause, OMPReductionClause, OMPSharedClause>{},
      llvm::omp::Directive::OMPD_parallel);

  auto parallelOp = mlir::omp::ParallelOp::create(builder, begin, clauseOps);

  mlir::Block &block = parallelOp.getRegion().emplaceBlock();
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(&block);

  CIRGenFunction::LexicalScope ls{cgf, begin, builder.getInsertionBlock()};

  if (s.hasCancel())
    cgm.errorNYI(s.getBeginLoc(), "OpenMP Parallel with Cancel");
  if (s.getTaskReductionRefExpr())
    cgm.errorNYI(s.getBeginLoc(), "OpenMP Parallel with Task Reduction");

  mlir::LogicalResult res = emitBody();
  mlir::omp::TerminatorOp::create(builder, end);
  return res;
}

template <typename DirectiveTy>
static mlir::LogicalResult
emitTeamsOp(CIRGenFunction &cgf, const DirectiveTy &s, mlir::Location begin,
            mlir::Location end,
            llvm::function_ref<mlir::LogicalResult()> emitBody) {
  CIRGenBuilderTy &builder = cgf.getBuilder();
  CIRGenModule &cgm = cgf.getCIRGenModule();

  llvm::SmallVector<const OMPClause *> clauses =
      getLeafClauses(cgf, s, llvm::omp::OMPD_teams);

  // No teams clauses are emittable yet, so report each eligible clause as NYI
  // rather than silently dropping it.
  mlir::omp::TeamsOperands clauseOps;
  OpenMPClauseEmitter ce(cgf, cgm, builder, begin, clauses);
  ce.emitNYI</*supported=*/>(
      /*nyi=*/OpenMPNYIClauseList<
          OMPAllocateClause, OMPDefaultClause, OMPDynGroupprivateClause,
          OMPFirstprivateClause, OMPIfClause, OMPNumTeamsClause,
          OMPPrivateClause, OMPReductionClause, OMPSharedClause,
          OMPThreadLimitClause, OMPXAttributeClause>{},
      llvm::omp::Directive::OMPD_teams);

  auto teamsOp = mlir::omp::TeamsOp::create(builder, begin, clauseOps);

  mlir::Block &block = teamsOp.getRegion().emplaceBlock();
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(&block);

  CIRGenFunction::LexicalScope ls{cgf, begin, builder.getInsertionBlock()};

  mlir::LogicalResult res = emitBody();
  mlir::omp::TerminatorOp::create(builder, end);
  return res;
}

mlir::LogicalResult
CIRGenFunction::emitOMPParallelDirective(const OMPParallelDirective &s) {
  mlir::Location begin = getLoc(s.getBeginLoc());
  mlir::Location end = getLoc(s.getEndLoc());

  return emitParallelOp(*this, s, begin, end, [&]() -> mlir::LogicalResult {
    // Don't lower the captured statement directly since this will be
    // special-cased depending on the kind of OpenMP directive that is the
    // parent, also the non-OpenMP context captured statements lowering does
    // not apply directly.
    const CapturedStmt *cs = s.getCapturedStmt(llvm::omp::OMPD_parallel);
    return emitStmt(cs->getCapturedStmt(), /*useCurrentScope=*/true);
  });
}

/// Ensure a CIR value has the given CIR integer type, inserting an integral
/// cast if necessary. Loads through CIR pointers first.
static mlir::Value ensureCIRIntType(CIRGenBuilderTy &builder,
                                    mlir::Location loc, mlir::Value cirValue,
                                    cir::IntType targetCIRType) {
  if (mlir::isa<cir::PointerType>(cirValue.getType()))
    cirValue = cir::LoadOp::create(builder, loc, cirValue).getResult();

  if (cirValue.getType() == targetCIRType)
    return cirValue;

  return builder.createCast(loc, cir::CastKind::integral, cirValue,
                            targetCIRType);
}

/// Convert a CIR integer value to a standard MLIR integer type suitable for
/// use as an omp.loop_nest operand.
static mlir::Value cirIntToStdInt(CIRGenBuilderTy &builder, mlir::Location loc,
                                  mlir::Value cirValue) {
  auto cirIntType = mlir::cast<cir::IntType>(cirValue.getType());
  mlir::Type stdIntType = builder.getIntegerType(cirIntType.getWidth());
  return builder.createBuiltinIntCast(loc, cirValue, stdIntType);
}

/// Emits the Sema-generated pre-init statements for an OpenMP loop directive.
/// For DeclStmts, emits each VarDecl directly so that OMPCapturedExprDecls
/// are not skipped.
static mlir::LogicalResult doEmitPreinits(CIRGenFunction &cgf,
                                          const Stmt *preInits) {
  if (!preInits)
    return mlir::success();

  llvm::SmallVector<const Stmt *> stmts;
  if (const auto *compound = dyn_cast<CompoundStmt>(preInits))
    llvm::append_range(stmts, compound->body());
  else
    stmts.push_back(preInits);

  for (const Stmt *stmt : stmts) {
    if (const auto *declStmt = dyn_cast<DeclStmt>(stmt)) {
      for (const Decl *d : declStmt->decls())
        cgf.emitVarDecl(cast<VarDecl>(*d));
    } else {
      if (cgf.emitStmt(stmt, /*useCurrentScope=*/true).failed())
        return mlir::failure();
    }
  }
  return mlir::success();
}

/// Emit an omp.loop_nest for the worksharing loop `forStmt`. The bounds
/// (lb/ub/step) are builtin integers. The induction variable block argument is
/// converted back to the loop variable's CIR integer type and stored into its
/// alloca before the loop body is lowered. This mirrors Flang's genLoopNestOp,
/// which builds the loop nest and lowers the body directly rather than reusing
/// the regular loop emitter.
static mlir::LogicalResult
emitOMPLoopNest(CIRGenFunction &cgf, const ForStmt &forStmt, mlir::Value lb,
                mlir::Value ub, mlir::Value step, bool inclusive,
                const VarDecl *inductionVar) {
  CIRGenBuilderTy &builder = cgf.getBuilder();
  mlir::Location loc = cgf.getLoc(forStmt.getSourceRange());

  auto loopNestOp = mlir::omp::LoopNestOp::create(
      builder, loc, /*collapse_num_loops=*/1, lb, ub, step,
      /*loop_inclusive=*/inclusive, /*tile_sizes=*/nullptr);
  mlir::Block *block = new mlir::Block();
  loopNestOp.getRegion().push_back(block);
  block->addArgument(lb.getType(), loc);

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(block);

  // Store the induction variable block argument into the loop variable alloca,
  // converting back from the builtin integer to the CIR integer type.
  mlir::Value iv = block->getArgument(0);
  Address inductionAddr = cgf.getAddrOfLocalVar(inductionVar);
  mlir::Value civVal =
      builder.createBuiltinIntCast(loc, iv, inductionAddr.getElementType());
  builder.createStore(loc, civVal, inductionAddr);

  mlir::LogicalResult bodyRes = mlir::success();
  if (forStmt.getBody())
    if (cgf.emitStmt(forStmt.getBody(), /*useCurrentScope=*/true).failed())
      bodyRes = mlir::failure();

  mlir::omp::YieldOp::create(builder, cgf.getLoc(forStmt.getEndLoc()));
  return bodyRes;
}

/// The iteration space of a canonical OpenMP loop, ready to be handed to
/// omp.loop_nest. The bounds are builtin integers and the for-init (induction
/// variable alloca) has already been emitted into the enclosing region.
namespace {
struct LoweredOMPLoop {
  const ForStmt *forStmt = nullptr;
  const VarDecl *inductionVar = nullptr;
  mlir::Value lowerBound;
  mlir::Value upperBound;
  mlir::Value step;
  bool inclusive = false;
};
} // namespace

/// Emits the pre-inits and for-init of an OMPLoopDirective and computes the
/// canonical loop bounds for omp.loop_nest. The induction variable alloca is
/// emitted here (before any loop-wrapper op) so it stays visible to the loop
/// body. Returns failure for loop shapes that are not yet supported.
static mlir::LogicalResult emitLoweredOMPLoop(CIRGenFunction &cgf,
                                              const OMPLoopDirective &s,
                                              LoweredOMPLoop &out) {
  CIRGenBuilderTy &builder = cgf.getBuilder();
  mlir::Location loc = cgf.getLoc(s.getBeginLoc());

  if (doEmitPreinits(cgf, s.getPreInits()).failed())
    return mlir::failure();

  const CapturedStmt *capturedStmt = s.getInnermostCapturedStmt();
  const auto *forStmt = cast<ForStmt>(capturedStmt->getCapturedStmt());

  // omp.loop_nest takes the original iteration space and stores its block
  // argument directly into the user's loop variable.
  mlir::Value lowerBound;
  mlir::Value upperBound;
  mlir::Value step;
  bool inclusive = false;

  const auto *declStmt = dyn_cast_or_null<DeclStmt>(forStmt->getInit());
  const auto *varDecl =
      declStmt ? dyn_cast<VarDecl>(declStmt->getSingleDecl()) : nullptr;
  if (!varDecl)
    return mlir::failure();

  QualType loopVarQType = varDecl->getType();
  auto cirIntType = mlir::cast<cir::IntType>(cgf.convertType(loopVarQType));

  if (!varDecl->hasInit())
    return mlir::failure();
  {
    mlir::Value v = cgf.emitScalarExpr(varDecl->getInit());
    lowerBound = ensureCIRIntType(builder, loc, v, cirIntType);
  }

  {
    const auto *condBinOp = dyn_cast_or_null<BinaryOperator>(forStmt->getCond());
    if (!condBinOp)
      return mlir::failure();
    BinaryOperatorKind op = condBinOp->getOpcode();
    const Expr *boundExpr = nullptr;
    if (op == BO_LT || op == BO_LE) {
      boundExpr = condBinOp->getRHS();
      inclusive = (op == BO_LE);
    } else if (op == BO_GT || op == BO_GE) {
      boundExpr = condBinOp->getLHS();
      inclusive = (op == BO_GE);
    } else {
      return mlir::failure();
    }
    mlir::Value v = cgf.emitScalarExpr(boundExpr);
    upperBound = ensureCIRIntType(builder, loc, v, cirIntType);
  }

  if (const auto *unary = dyn_cast_or_null<UnaryOperator>(forStmt->getInc())) {
    step =
        builder.getConstInt(loc, cirIntType, unary->isIncrementOp() ? 1 : -1);
  } else if (const auto *binOp =
                 dyn_cast_or_null<BinaryOperator>(forStmt->getInc())) {
    const Expr *stepExpr = nullptr;
    if (binOp->isCompoundAssignmentOp()) {
      stepExpr = binOp->getRHS();
    } else if (binOp->isAssignmentOp()) {
      if (auto *sub =
              dyn_cast<BinaryOperator>(binOp->getRHS()->IgnoreImpCasts())) {
        const Expr *lhs = sub->getLHS()->IgnoreImpCasts();
        const Expr *rhs = sub->getRHS()->IgnoreImpCasts();
        if (auto *lhsRef = dyn_cast<DeclRefExpr>(lhs))
          stepExpr = (lhsRef->getDecl() == varDecl) ? rhs : lhs;
        else if (auto *rhsRef = dyn_cast<DeclRefExpr>(rhs))
          stepExpr = (rhsRef->getDecl() == varDecl) ? lhs : rhs;
      }
    }
    if (stepExpr) {
      mlir::Value v = cgf.emitScalarExpr(stepExpr);
      step = ensureCIRIntType(builder, loc, v, cirIntType);
    }
  }
  if (!step)
    step = builder.getConstInt(loc, cirIntType, 1);

  // The induction variable alloca must be visible in the loop-wrapper region
  // created by the caller, so emit the init before that op is created.
  if (forStmt->getInit())
    if (cgf.emitStmt(forStmt->getInit(), /*useCurrentScope=*/true).failed())
      return mlir::failure();

  // omp.loop_nest requires IntLikeType operands, not CIR integer types.
  out.forStmt = forStmt;
  out.inductionVar = varDecl;
  out.lowerBound = cirIntToStdInt(builder, loc, lowerBound);
  out.upperBound = cirIntToStdInt(builder, loc, upperBound);
  out.step = cirIntToStdInt(builder, loc, step);
  out.inclusive = inclusive;
  return mlir::success();
}

/// Emits the omp.loop_nest for \p loop inside the wrapper region the caller has
/// already entered (e.g. an omp.wsloop or omp.distribute block).
static mlir::LogicalResult emitOMPLoopNestBody(CIRGenFunction &cgf,
                                               const LoweredOMPLoop &loop) {
  return emitOMPLoopNest(cgf, *loop.forStmt, loop.lowerBound, loop.upperBound,
                         loop.step, loop.inclusive, loop.inductionVar);
}

/// Lowers an OMPLoopDirective into an omp.wsloop + omp.loop_nest.
/// The original loop bounds are passed directly to omp.loop_nest, which
/// handles work distribution. The induction variable alloca is emitted before
/// the wsloop region so that the loop body can reference it.
static mlir::LogicalResult emitOMPWorksharingLoop(CIRGenFunction &cgf,
                                                  const OMPLoopDirective &s) {
  CIRGenBuilderTy &builder = cgf.getBuilder();
  CIRGenModule &cgm = cgf.getCIRGenModule();
  mlir::Location loc = cgf.getLoc(s.getBeginLoc());

  // The worksharing loop emits no clauses yet, so route the `for`-leaf clauses
  // through the decomposition and report each eligible clause as NYI rather
  // than silently dropping it.
  llvm::SmallVector<const OMPClause *> clauses =
      getLeafClauses(cgf, s, llvm::omp::OMPD_for);
  OpenMPClauseEmitter ce(cgf, cgm, builder, loc, clauses);
  ce.emitNYI</*supported=*/>(
      /*nyi=*/OpenMPNYIClauseList<
          OMPAllocateClause, OMPCollapseClause, OMPFirstprivateClause,
          OMPLastprivateClause, OMPLinearClause, OMPNowaitClause,
          OMPOrderClause, OMPOrderedClause, OMPPrivateClause,
          OMPReductionClause, OMPScheduleClause>{},
      llvm::omp::Directive::OMPD_for);

  LoweredOMPLoop loop;
  if (emitLoweredOMPLoop(cgf, s, loop).failed())
    return mlir::failure();

  llvm::SmallVector<mlir::Type> retTy;
  llvm::SmallVector<mlir::Value> operands;
  auto wsloopOp = mlir::omp::WsloopOp::create(builder, loc, retTy, operands);
  mlir::Block *innerBlock = new mlir::Block();
  wsloopOp.getRegion().push_back(innerBlock);

  // Emit the omp.loop_nest directly inside the wsloop region. The for-init was
  // already emitted above so the induction variable alloca lives outside the
  // loop region.
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(innerBlock);
  return emitOMPLoopNestBody(cgf, loop);
}

/// Lowers an OMPLoopDirective into an omp.distribute + omp.loop_nest. The
/// distribute construct partitions loop iterations across the teams of the
/// enclosing teams region; it shares the canonical loop lowering with the
/// worksharing loop and only differs in the wrapper op and clause set.
static mlir::LogicalResult emitOMPDistributeLoop(CIRGenFunction &cgf,
                                                 const OMPLoopDirective &s) {
  CIRGenBuilderTy &builder = cgf.getBuilder();
  CIRGenModule &cgm = cgf.getCIRGenModule();
  mlir::Location loc = cgf.getLoc(s.getBeginLoc());

  // No distribute clauses are emittable yet, so report each eligible
  // `distribute`-leaf clause as NYI rather than silently dropping it.
  llvm::SmallVector<const OMPClause *> clauses =
      getLeafClauses(cgf, s, llvm::omp::OMPD_distribute);
  OpenMPClauseEmitter ce(cgf, cgm, builder, loc, clauses);
  ce.emitNYI</*supported=*/>(
      /*nyi=*/OpenMPNYIClauseList<
          OMPAllocateClause, OMPCollapseClause, OMPDistScheduleClause,
          OMPFirstprivateClause, OMPLastprivateClause, OMPOrderClause,
          OMPPrivateClause>{},
      llvm::omp::Directive::OMPD_distribute);

  LoweredOMPLoop loop;
  if (emitLoweredOMPLoop(cgf, s, loop).failed())
    return mlir::failure();

  mlir::omp::DistributeOperands clauseOps;
  auto distributeOp = mlir::omp::DistributeOp::create(builder, loc, clauseOps);
  mlir::Block *innerBlock = new mlir::Block();
  distributeOp.getRegion().push_back(innerBlock);

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(innerBlock);
  return emitOMPLoopNestBody(cgf, loop);
}

mlir::LogicalResult
CIRGenFunction::emitOMPTaskwaitDirective(const OMPTaskwaitDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPTaskwaitDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPTaskyieldDirective(const OMPTaskyieldDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPTaskyieldDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPBarrierDirective(const OMPBarrierDirective &s) {
  mlir::omp::BarrierOp::create(builder, getLoc(s.getBeginLoc()));
  assert(s.clauses().empty() && "omp barrier doesn't support clauses");
  return mlir::success();
}
mlir::LogicalResult
CIRGenFunction::emitOMPMetaDirective(const OMPMetaDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPMetaDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPCanonicalLoop(const OMPCanonicalLoop &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPCanonicalLoop");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPSimdDirective(const OMPSimdDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPSimdDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPTileDirective(const OMPTileDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPTileDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPUnrollDirective(const OMPUnrollDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPUnrollDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPFuseDirective(const OMPFuseDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPFuseDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPForDirective(const OMPForDirective &s) {
  return emitOMPWorksharingLoop(*this, s);
}
mlir::LogicalResult
CIRGenFunction::emitOMPForSimdDirective(const OMPForSimdDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPForSimdDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPSectionsDirective(const OMPSectionsDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPSectionsDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPSectionDirective(const OMPSectionDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPSectionDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPSingleDirective(const OMPSingleDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPSingleDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPMasterDirective(const OMPMasterDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPMasterDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPCriticalDirective(const OMPCriticalDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPCriticalDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPParallelForDirective(const OMPParallelForDirective &s) {
  mlir::Location begin = getLoc(s.getBeginLoc());
  mlir::Location end = getLoc(s.getEndLoc());

  // `parallel for` decomposes into a `parallel` leaf wrapping a `for` leaf, so
  // emit an omp.parallel whose region holds the worksharing loop.
  return emitParallelOp(*this, s, begin, end, [&]() -> mlir::LogicalResult {
    return emitOMPWorksharingLoop(*this, s);
  });
}
mlir::LogicalResult CIRGenFunction::emitOMPParallelForSimdDirective(
    const OMPParallelForSimdDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPParallelForSimdDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPParallelMasterDirective(
    const OMPParallelMasterDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPParallelMasterDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPParallelSectionsDirective(
    const OMPParallelSectionsDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPParallelSectionsDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPTaskDirective(const OMPTaskDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPTaskDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPTaskgroupDirective(const OMPTaskgroupDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPTaskgroupDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPFlushDirective(const OMPFlushDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPFlushDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPDepobjDirective(const OMPDepobjDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPDepobjDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPScanDirective(const OMPScanDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPScanDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPOrderedDirective(const OMPOrderedDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPOrderedDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPAtomicDirective(const OMPAtomicDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPAtomicDirective");
  return mlir::failure();
}

/// Check for unsupported implicit captures in a target region.
static void
emitOMPTargetImplicitCaptures(CIRGenFunction &cgf,
                              const OMPExecutableDirective &s,
                              llvm::ArrayRef<const VarDecl *> mapSyms) {
  const CapturedStmt *cs = s.getCapturedStmt(llvm::omp::OMPD_target);
  for (const auto &capture : cs->captures()) {
    if (capture.capturesThis()) {
      cgf.getCIRGenModule().errorNYI(s.getBeginLoc(),
                                     "OpenMP target capture of 'this' pointer");
      continue;
    }
    if (capture.capturesVariableByCopy()) {
      cgf.getCIRGenModule().errorNYI(s.getBeginLoc(),
                                     "OpenMP target capture by copy");
      continue;
    }
    if (capture.capturesVariableArrayType()) {
      cgf.getCIRGenModule().errorNYI(
          s.getBeginLoc(),
          "OpenMP target capture of variable-length array type");
      continue;
    }
    if (capture.capturesVariable()) {
      const VarDecl *vd = capture.getCapturedVar();
      if (llvm::is_contained(mapSyms, vd))
        continue;

      cgf.getCIRGenModule().errorNYI(s.getBeginLoc(),
                                     "OpenMP target implicit by-ref capture");
    }
  }
}

template <typename DirectiveTy>
static mlir::LogicalResult
emitTargetOp(CIRGenFunction &cgf, const DirectiveTy &s, mlir::Location begin,
             mlir::Location end,
             llvm::function_ref<mlir::LogicalResult()> emitBody) {
  CIRGenBuilderTy &builder = cgf.getBuilder();
  CIRGenModule &cgm = cgf.getCIRGenModule();

  llvm::SmallVector<const OMPClause *> clauses =
      getLeafClauses(cgf, s, llvm::omp::OMPD_target);

  mlir::omp::TargetExtOperands clauseOps;
  llvm::SmallVector<const VarDecl *> mapSyms;

  OpenMPClauseEmitter ce(cgf, cgm, builder, begin, clauses);
  ce.emitMap(clauseOps, &mapSyms);
  ce.emitNYI</*supported=*/OMPMapClause>(
      /*nyi=*/OpenMPNYIClauseList<
          OMPAllocateClause, OMPDefaultClause, OMPDefaultmapClause,
          OMPDependClause, OMPDeviceClause, OMPFirstprivateClause,
          OMPHasDeviceAddrClause, OMPIfClause, OMPInReductionClause,
          OMPIsDevicePtrClause, OMPNowaitClause, OMPPrivateClause,
          OMPThreadLimitClause, OMPUsesAllocatorsClause, OMPXBareClause>{},
      llvm::omp::Directive::OMPD_target);

  emitOMPTargetImplicitCaptures(cgf, s, mapSyms);

  // Use generic for now.
  clauseOps.kernelType = mlir::omp::TargetExecModeAttr::get(
      &cgf.getMLIRContext(), mlir::omp::TargetExecMode::generic);

  auto targetOp = mlir::omp::TargetOp::create(builder, begin, clauseOps);

  mlir::Block &block = targetOp.getRegion().emplaceBlock();
  for (mlir::Value mapVar : clauseOps.mapVars)
    block.addArgument(mapVar.getType(), begin);

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(&block);

  CIRGenFunction::LexicalScope ls{cgf, begin, builder.getInsertionBlock()};

  llvm::SmallVector<std::pair<const VarDecl *, Address>> savedAddrs;
  for (auto [idx, vd] : llvm::enumerate(mapSyms)) {
    Address origAddr = cgf.getAddrOfLocalVar(vd);
    savedAddrs.push_back({vd, origAddr});
    mlir::Value blockArg = block.getArgument(idx);
    cgf.replaceAddrOfLocalVar(vd, Address(blockArg, origAddr.getAlignment()));
  }

  mlir::LogicalResult res = emitBody();
  mlir::omp::TerminatorOp::create(builder, end);

  for (auto &[vd, addr] : savedAddrs)
    cgf.replaceAddrOfLocalVar(vd, addr);

  return res;
}

mlir::LogicalResult
CIRGenFunction::emitOMPTargetDirective(const OMPTargetDirective &s) {
  mlir::Location begin = getLoc(s.getBeginLoc());
  mlir::Location end = getLoc(s.getEndLoc());

  return emitTargetOp(*this, s, begin, end, [&]() -> mlir::LogicalResult {
    const CapturedStmt *cs = s.getCapturedStmt(llvm::omp::OMPD_target);
    return emitStmt(cs->getCapturedStmt(), /*useCurrentScope=*/true);
  });
}
mlir::LogicalResult
CIRGenFunction::emitOMPTeamsDirective(const OMPTeamsDirective &s) {
  mlir::Location begin = getLoc(s.getBeginLoc());
  mlir::Location end = getLoc(s.getEndLoc());

  return emitTeamsOp(*this, s, begin, end, [&]() -> mlir::LogicalResult {
    const CapturedStmt *cs = s.getCapturedStmt(llvm::omp::OMPD_teams);
    return emitStmt(cs->getCapturedStmt(), /*useCurrentScope=*/true);
  });
}
mlir::LogicalResult CIRGenFunction::emitOMPCancellationPointDirective(
    const OMPCancellationPointDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPCancellationPointDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPCancelDirective(const OMPCancelDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPCancelDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPTargetDataDirective(const OMPTargetDataDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPTargetDataDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPTargetEnterDataDirective(
    const OMPTargetEnterDataDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPTargetEnterDataDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPTargetExitDataDirective(
    const OMPTargetExitDataDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPTargetExitDataDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPTargetParallelDirective(
    const OMPTargetParallelDirective &s) {
  mlir::Location begin = getLoc(s.getBeginLoc());
  mlir::Location end = getLoc(s.getEndLoc());

  return emitTargetOp(*this, s, begin, end, [&]() -> mlir::LogicalResult {
    return emitParallelOp(*this, s, begin, end, [&]() -> mlir::LogicalResult {
      const CapturedStmt *cs = s.getCapturedStmt(llvm::omp::OMPD_parallel);
      return emitStmt(cs->getCapturedStmt(), /*useCurrentScope=*/true);
    });
  });
}
mlir::LogicalResult CIRGenFunction::emitOMPTargetParallelForDirective(
    const OMPTargetParallelForDirective &s) {
  mlir::Location begin = getLoc(s.getBeginLoc());
  mlir::Location end = getLoc(s.getEndLoc());

  // `target parallel for` decomposes into `target`, `parallel` and `for`
  // leaves, so nest the worksharing loop inside an omp.parallel inside an
  // omp.target.
  return emitTargetOp(*this, s, begin, end, [&]() -> mlir::LogicalResult {
    return emitParallelOp(*this, s, begin, end, [&]() -> mlir::LogicalResult {
      return emitOMPWorksharingLoop(*this, s);
    });
  });
}
mlir::LogicalResult
CIRGenFunction::emitOMPTaskLoopDirective(const OMPTaskLoopDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPTaskLoopDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPTaskLoopSimdDirective(
    const OMPTaskLoopSimdDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPTaskLoopSimdDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPMaskedTaskLoopDirective(
    const OMPMaskedTaskLoopDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPMaskedTaskLoopDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPMaskedTaskLoopSimdDirective(
    const OMPMaskedTaskLoopSimdDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPMaskedTaskLoopSimdDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPMasterTaskLoopDirective(
    const OMPMasterTaskLoopDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPMasterTaskLoopDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPMasterTaskLoopSimdDirective(
    const OMPMasterTaskLoopSimdDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPMasterTaskLoopSimdDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPParallelGenericLoopDirective(
    const OMPParallelGenericLoopDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPParallelGenericLoopDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPParallelMaskedDirective(
    const OMPParallelMaskedDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPParallelMaskedDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPParallelMaskedTaskLoopDirective(
    const OMPParallelMaskedTaskLoopDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPParallelMaskedTaskLoopDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPParallelMaskedTaskLoopSimdDirective(
    const OMPParallelMaskedTaskLoopSimdDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPParallelMaskedTaskLoopSimdDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPParallelMasterTaskLoopDirective(
    const OMPParallelMasterTaskLoopDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPParallelMasterTaskLoopDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPParallelMasterTaskLoopSimdDirective(
    const OMPParallelMasterTaskLoopSimdDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPParallelMasterTaskLoopSimdDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPDistributeDirective(const OMPDistributeDirective &s) {
  return emitOMPDistributeLoop(*this, s);
}
mlir::LogicalResult CIRGenFunction::emitOMPDistributeParallelForDirective(
    const OMPDistributeParallelForDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPDistributeParallelForDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPDistributeParallelForSimdDirective(
    const OMPDistributeParallelForSimdDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPDistributeParallelForSimdDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPDistributeSimdDirective(
    const OMPDistributeSimdDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPDistributeSimdDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPTargetParallelGenericLoopDirective(
    const OMPTargetParallelGenericLoopDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPTargetParallelGenericLoopDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPTargetParallelForSimdDirective(
    const OMPTargetParallelForSimdDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPTargetParallelForSimdDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPTargetSimdDirective(const OMPTargetSimdDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPTargetSimdDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPTargetTeamsGenericLoopDirective(
    const OMPTargetTeamsGenericLoopDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPTargetTeamsGenericLoopDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPTargetUpdateDirective(
    const OMPTargetUpdateDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPTargetUpdateDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPTeamsDistributeDirective(
    const OMPTeamsDistributeDirective &s) {
  mlir::Location begin = getLoc(s.getBeginLoc());
  mlir::Location end = getLoc(s.getEndLoc());

  // `teams distribute` decomposes into a `teams` leaf wrapping a `distribute`
  // leaf, so nest an omp.distribute inside an omp.teams.
  return emitTeamsOp(*this, s, begin, end, [&]() -> mlir::LogicalResult {
    return emitOMPDistributeLoop(*this, s);
  });
}
mlir::LogicalResult CIRGenFunction::emitOMPTeamsDistributeSimdDirective(
    const OMPTeamsDistributeSimdDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPTeamsDistributeSimdDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPTeamsDistributeParallelForSimdDirective(
    const OMPTeamsDistributeParallelForSimdDirective &s) {
  getCIRGenModule().errorNYI(
      s.getSourceRange(), "OpenMP OMPTeamsDistributeParallelForSimdDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPTeamsDistributeParallelForDirective(
    const OMPTeamsDistributeParallelForDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPTeamsDistributeParallelForDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPTeamsGenericLoopDirective(
    const OMPTeamsGenericLoopDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPTeamsGenericLoopDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPTargetTeamsDirective(const OMPTargetTeamsDirective &s) {
  mlir::Location begin = getLoc(s.getBeginLoc());
  mlir::Location end = getLoc(s.getEndLoc());

  // `target teams` decomposes into a `target` leaf wrapping a `teams` leaf, so
  // nest an omp.teams inside an omp.target.
  return emitTargetOp(*this, s, begin, end, [&]() -> mlir::LogicalResult {
    return emitTeamsOp(*this, s, begin, end, [&]() -> mlir::LogicalResult {
      const CapturedStmt *cs = s.getCapturedStmt(llvm::omp::OMPD_teams);
      return emitStmt(cs->getCapturedStmt(), /*useCurrentScope=*/true);
    });
  });
}
mlir::LogicalResult CIRGenFunction::emitOMPTargetTeamsDistributeDirective(
    const OMPTargetTeamsDistributeDirective &s) {
  mlir::Location begin = getLoc(s.getBeginLoc());
  mlir::Location end = getLoc(s.getEndLoc());

  // `target teams distribute` decomposes into `target`, `teams` and
  // `distribute` leaves, so nest an omp.distribute inside an omp.teams inside
  // an omp.target.
  return emitTargetOp(*this, s, begin, end, [&]() -> mlir::LogicalResult {
    return emitTeamsOp(*this, s, begin, end, [&]() -> mlir::LogicalResult {
      return emitOMPDistributeLoop(*this, s);
    });
  });
}
mlir::LogicalResult
CIRGenFunction::emitOMPTargetTeamsDistributeParallelForDirective(
    const OMPTargetTeamsDistributeParallelForDirective &s) {
  getCIRGenModule().errorNYI(
      s.getSourceRange(),
      "OpenMP OMPTargetTeamsDistributeParallelForDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPTargetTeamsDistributeParallelForSimdDirective(
    const OMPTargetTeamsDistributeParallelForSimdDirective &s) {
  getCIRGenModule().errorNYI(
      s.getSourceRange(),
      "OpenMP OMPTargetTeamsDistributeParallelForSimdDirective");
  return mlir::failure();
}
mlir::LogicalResult CIRGenFunction::emitOMPTargetTeamsDistributeSimdDirective(
    const OMPTargetTeamsDistributeSimdDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPTargetTeamsDistributeSimdDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPInteropDirective(const OMPInteropDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPInteropDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPDispatchDirective(const OMPDispatchDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPDispatchDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPGenericLoopDirective(const OMPGenericLoopDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPGenericLoopDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPReverseDirective(const OMPReverseDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPReverseDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPSplitDirective(const OMPSplitDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPSplitDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPInterchangeDirective(const OMPInterchangeDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(),
                             "OpenMP OMPInterchangeDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPAssumeDirective(const OMPAssumeDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPAssumeDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPMaskedDirective(const OMPMaskedDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPMaskedDirective");
  return mlir::failure();
}
mlir::LogicalResult
CIRGenFunction::emitOMPStripeDirective(const OMPStripeDirective &s) {
  getCIRGenModule().errorNYI(s.getSourceRange(), "OpenMP OMPStripeDirective");
  return mlir::failure();
}
