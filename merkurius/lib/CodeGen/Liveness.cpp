//===-- lib/CodeGen/Liveness.cpp - What is live at a safepoint ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Puts a merk.live before each operation at which a collection can happen,
// listing the values the collector must find there with their merk types,
// which lowering loses (docs/INPROGRESS.md, Statepoints and stack maps).
//
//===----------------------------------------------------------------------===//

#include "merkurius/CodeGen/Lowering.h"
#include "merkurius/Dialect/Merk/IR/MerkDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Regex.h"

namespace merkurius::codegen {
namespace {

using Live = llvm::SmallSetVector<mlir::Value, 8>;

// A value the collector follows.
bool IsRoot(mlir::Type type) {
  return mlir::isa<merk::TermType, merk::WordType, merk::StringType,
      merk::ArrayType, merk::ClosureType, merk::TypeInfoType,
      merk::TypeclassInfoType, merk::RefType>(type);
}

bool IsPolymorphic(mlir::Type type) {
  if (mlir::isa<merk::WordType>(type)) {
    return true;
  }
  auto term{mlir::dyn_cast<merk::TermType>(type)};
  return term && llvm::Regex{"[(, ]T[0-9]+[,)]"}.match(term.getName());
}

bool IsTypeWord(mlir::Type type) {
  return mlir::isa<merk::TypeInfoType, merk::TypeclassInfoType>(type);
}

// An operation that makes a cell, which can start a collection. What it stores
// is live while it does.
bool Allocates(mlir::Operation *op) {
  return mlir::isa<merk::ConstructOp, merk::ClosureOp, merk::TypeInfoOp,
      merk::TypeclassInfoOp>(op);
}

// An operation that runs Mercury code, which can collect.
bool Calls(mlir::Operation *op) {
  return mlir::isa<merk::CallOp, merk::CallClosureOp, merk::CallMethodOp,
      merk::CallContOp, merk::UnifyWordOp, merk::CompareWordOp>(op);
}

void Union(Live &into, const Live &from) {
  into.insert(from.begin(), from.end());
}

class Marker {
public:
  explicit Marker(merk::ProcOp proc) {
    for (mlir::BlockArgument argument : proc.getArguments()) {
      if (IsTypeWord(argument.getType())) {
        typeWords_.push_back(argument);
      }
    }
  }

  void Scan(mlir::Block &block, Live &live);

private:
  void ScanRegions(mlir::Operation *op, Live &live);
  void Mark(mlir::Operation *op, const Live &live);

  llvm::SmallVector<mlir::Value> typeWords_;
};

// A value defined outside `op` that something inside it uses.
void UsesInside(mlir::Operation *op, Live &uses) {
  for (mlir::Region &region : op->getRegions()) {
    region.walk([&](mlir::Operation *inner) {
      for (mlir::Value operand : inner->getOperands()) {
        if (!op->isProperAncestor(operand.getParentBlock()->getParentOp()) &&
            operand.getParentBlock()->getParentOp() != op) {
          uses.insert(operand);
        }
      }
    });
  }
}

void Marker::Mark(mlir::Operation *op, const Live &live) {
  llvm::SmallVector<mlir::Value> values;
  llvm::SmallVector<mlir::Attribute> types;
  bool polymorphic{false};
  mlir::Builder builder{op->getContext()};
  llvm::SmallSetVector<mlir::Value, 8> all{live.begin(), live.end()};
  if (Allocates(op)) {
    all.insert(op->operand_begin(), op->operand_end());
  }
  for (mlir::Value value : all) {
    polymorphic = polymorphic || IsPolymorphic(value.getType());
  }
  // Temporary (chunk 3, docs/INPROGRESS.md, Statepoints and stack maps, piece
  // 5): a !merk.word does not say which type_info is its own, so all of the
  // procedure's stay live, as much as the collector could need.
  if (polymorphic) {
    all.insert(typeWords_.begin(), typeWords_.end());
  }
  for (mlir::Value value : all) {
    if (IsRoot(value.getType())) {
      values.push_back(value);
      types.push_back(mlir::TypeAttr::get(value.getType()));
    }
  }
  mlir::OpBuilder before{op};
  merk::LiveOp::create(
      before, op->getLoc(), values, builder.getArrayAttr(types));
}

void Marker::ScanRegions(mlir::Operation *op, Live &live) {
  Live after{live};
  if (auto ite{mlir::dyn_cast<merk::IteOp>(op)}) {
    Live thenLive{after};
    Scan(ite.getThenRegion().front(), thenLive);
    Live elseLive{after};
    Scan(ite.getElseRegion().front(), elseLive);
    live = after;
    Union(live, thenLive);
    Union(live, elseLive);
    Scan(ite.getCondition().front(), live);
    return;
  }
  if (mlir::isa<merk::SwitchOp>(op)) {
    for (mlir::Region &region : op->getRegions()) {
      Live caseLive{after};
      Scan(region.front(), caseLive);
      Union(live, caseLive);
    }
    return;
  }
  // Any other: what any of its regions uses is live in all of them, which is
  // more than need be, but only for constructs that outlining has mostly
  // taken away. OpenMP's regions have no safepoints (D2).
  Live uses;
  UsesInside(op, uses);
  Union(live, uses);
  if (mlir::isa<mlir::omp::OpenMPDialect>(op->getDialect())) {
    return;
  }
  for (mlir::Region &region : op->getRegions()) {
    for (mlir::Block &block : region) {
      Live inner{after};
      Union(inner, uses);
      Scan(block, inner);
    }
  }
}

void Marker::Scan(mlir::Block &block, Live &live) {
  for (mlir::Operation &op :
      llvm::make_early_inc_range(llvm::reverse(block))) {
    if (mlir::isa<merk::LiveOp>(op)) {
      continue;
    }
    for (mlir::Value result : op.getResults()) {
      live.remove(result);
    }
    if (op.getNumRegions() != 0) {
      ScanRegions(&op, live);
    }
    if (Calls(&op) || Allocates(&op)) {
      Mark(&op, live);
    }
    live.insert(op.operand_begin(), op.operand_end());
  }
  for (mlir::BlockArgument argument : block.getArguments()) {
    live.remove(argument);
  }
}

} // namespace

void MarkSafepoints(mlir::ModuleOp module) {
  for (auto proc : module.getOps<merk::ProcOp>()) {
    if (proc.isExternal()) {
      continue;
    }
    Marker marker{proc};
    Live live;
    marker.Scan(proc.getBody().front(), live);
  }
}

} // namespace merkurius::codegen
