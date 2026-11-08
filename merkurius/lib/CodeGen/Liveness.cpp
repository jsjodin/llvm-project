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
#include "mlir/IR/Dominance.h"

namespace merkurius::codegen {
namespace {

using Live = llvm::SmallSetVector<mlir::Value, 8>;

// A value the collector follows.
bool IsRoot(mlir::Type type) {
  return mlir::isa<merk::TermType, merk::WordType, merk::StringType,
      merk::ArrayType, merk::ClosureType, merk::TypeInfoType,
      merk::TypeclassInfoType, merk::RefType>(type);
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
  explicit Marker(merk::ProcOp proc) : dominance_{proc} {
    llvm::SmallVector<merk::RootTypeOp> notes;
    proc.walk([&](merk::RootTypeOp note) { notes.push_back(note); });
    for (merk::RootTypeOp note : notes) {
      infos_[note.getValue()].assign(
          note.getInfos().begin(), note.getInfos().end());
      note.erase();
    }
  }

  void Scan(mlir::Block &block, Live &live);

private:
  void ScanRegions(mlir::Operation *op, Live &live);
  void Mark(mlir::Operation *op, Live &live);
  void MarkRegionCalls(mlir::Operation *op);
  // Whether the type_infos are defined before `op`. A value can be made before
  // the code that finds its type_info.
  bool Available(llvm::ArrayRef<mlir::Value> infos, mlir::Operation *op);

  // The type_infos of each polymorphic value that has them.
  llvm::DenseMap<mlir::Value, llvm::SmallVector<mlir::Value, 2>> infos_;
  mlir::DominanceInfo dominance_;
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

bool Marker::Available(llvm::ArrayRef<mlir::Value> infos, mlir::Operation *op) {
  return llvm::all_of(infos, [&](mlir::Value info) {
    return dominance_.properlyDominates(info, op);
  });
}

void Marker::Mark(mlir::Operation *op, Live &live) {
  llvm::SmallVector<mlir::Value> values;
  llvm::SmallVector<mlir::Attribute> types;
  llvm::SmallVector<std::int64_t> counts;
  mlir::Builder builder{op->getContext()};
  llvm::SmallSetVector<mlir::Value, 8> all{live.begin(), live.end()};
  if (Allocates(op)) {
    all.insert(op->operand_begin(), op->operand_end());
  }
  // A call that is given an environment runs while it is in use, though it may
  // not be used after.
  if (Calls(op)) {
    for (mlir::Value operand : op->getOperands()) {
      if (mlir::isa<merk::EnvType>(operand.getType())) {
        all.insert(operand);
      }
    }
  }
  // What a root's type_infos are needs them live here, and so before here.
  llvm::SmallVector<mlir::Value> roots{all.begin(), all.end()};
  for (mlir::Value value : roots) {
    auto it{infos_.find(value)};
    if (it != infos_.end() && Available(it->second, op)) {
      all.insert(it->second.begin(), it->second.end());
      live.insert(it->second.begin(), it->second.end());
    }
  }
  for (mlir::Value value : all) {
    if (IsRoot(value.getType())) {
      values.push_back(value);
      types.push_back(mlir::TypeAttr::get(value.getType()));
      auto it{infos_.find(value)};
      if (it != infos_.end() && Available(it->second, op)) {
        values.append(it->second.begin(), it->second.end());
        counts.push_back(it->second.size());
      } else {
        counts.push_back(0);
      }
    } else if (auto env{value.getDefiningOp<merk::EnvNewOp>()}) {
      // One of this frame, given by its layout. One that is an argument is
      // in the frame of a caller, which lists it.
      if (llvm::any_of(env.getLayout().getAsValueRange<mlir::TypeAttr>(),
              IsRoot)) {
        llvm::SmallVector<mlir::Type> layout{
            env.getLayout().getAsValueRange<mlir::TypeAttr>()};
        values.push_back(value);
        types.push_back(mlir::TypeAttr::get(
            mlir::TupleType::get(op->getContext(), layout)));
        counts.push_back(0);
      }
    }
  }
  mlir::OpBuilder before{op};
  auto marker{merk::LiveOp::create(
      before, op->getLoc(), values, builder.getArrayAttr(types))};
  marker->setAttr("info_counts", builder.getI64ArrayAttr(counts));
}

// Nothing collects during a region. Its calls say so, as the optimizer can
// inline the outlined region into a function that has safepoints.
void Marker::MarkRegionCalls(mlir::Operation *op) {
  op->walk([](mlir::Operation *inner) {
    if (Calls(inner)) {
      mlir::OpBuilder before{inner};
      auto live{merk::LiveOp::create(before, inner->getLoc(), mlir::ValueRange{},
          before.getArrayAttr({}))};
      live->setAttr("in_region", before.getUnitAttr());
    }
  });
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
  // A target data region stays in its function, and the host runs it.
  if (mlir::isa<mlir::omp::OpenMPDialect>(op->getDialect()) &&
      !mlir::isa<mlir::omp::TargetDataOp>(op)) {
    MarkRegionCalls(op);
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
