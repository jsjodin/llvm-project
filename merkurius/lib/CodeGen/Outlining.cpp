//===-- lib/CodeGen/Outlining.cpp - Outline nondeterministic code ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Rewrites the merk dialect's nondeterministic code, in regions, as the
// continuation passing code that lowering takes (design document, Lowering,
// "Nondeterministic code"): a nondet or multi procedure becomes a semidet
// one with no outputs that takes a continuation and its environment after
// its inputs, and gives each solution to the continuation; it succeeds if
// the continuation asks it to stop. The rest of the code after a nondet
// call or merk.each becomes a continuation procedure, whose environment
// holds the values it uses, in the frame of the code it came from; the
// generator of merk.commit becomes a procedure that stores its first
// solution in its environment and stops.
//
//===----------------------------------------------------------------------===//

#include "merkurius/CodeGen/Lowering.h"
#include "merkurius/Dialect/Merk/IR/MerkDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringMap.h"

namespace merkurius::codegen {
namespace {

// Where a solution goes: to a continuation and its environment, or, for
// merk.commit's, into the slots of an environment from `first` on.
struct Sink {
  mlir::Value cont;
  mlir::Value env;
  unsigned first{0};
  mlir::ArrayAttr layout;
};

class Outliner {
public:
  explicit Outliner(mlir::ModuleOp module)
      : module_{module}, builder_{module.getContext()} {}
  void Run();

private:
  void Rewrite(mlir::Region &, const Sink *succeed, const Sink *solution);
  void RewriteCall(merk::NondetCallOp, const Sink *succeed, const Sink *solution);
  void RewriteEach(merk::EachOp, const Sink *succeed, const Sink *solution);
  void RewriteCommit(merk::CommitOp);
  void Deliver(mlir::Operation *end, mlir::ValueRange values, const Sink &);
  merk::ProcOp Outline(mlir::Region &, llvm::ArrayRef<mlir::Type> slots,
      llvm::SmallVectorImpl<mlir::Value> &captures, mlir::ArrayAttr &layout);
  mlir::Value MakeEnv(mlir::Location, mlir::ValueRange captures,
      mlir::ArrayAttr layout);
  const Sink *PassedOn(mlir::Region &continuation, const Sink *succeed,
      const Sink *solution);

  mlir::ModuleOp module_;
  mlir::OpBuilder builder_;
  // The procedure being rewritten, which the procedures outlined from it
  // follow and are named after.
  merk::ProcOp proc_;
  llvm::StringMap<unsigned> outlined_;
};

// The merk.solution operations that give the solutions of a generator: not
// those of a merk.commit in it, but those of a merk.each's continuation.
void Solutions(mlir::Region &region, llvm::SmallVectorImpl<merk::SolutionOp> &out) {
  for (mlir::Block &block : region) {
    for (mlir::Operation &op : block) {
      if (auto solution{mlir::dyn_cast<merk::SolutionOp>(op)}) {
        out.push_back(solution);
      } else if (auto each{mlir::dyn_cast<merk::EachOp>(op)}) {
        Solutions(each.getContinuation(), out);
      } else if (!mlir::isa<merk::CommitOp>(op)) {
        for (mlir::Region &inner : op.getRegions()) {
          Solutions(inner, out);
        }
      }
    }
  }
}

// Leaves out of a generator's solutions the values that nothing uses: the
// arguments of merk.each's continuation that it does not use, and the
// results of merk.commit that nothing uses, which code generation gives for
// every variable the goal binds.
void Prune(mlir::Operation *op, mlir::Region &generator,
    llvm::ArrayRef<bool> dead) {
  llvm::SmallVector<merk::SolutionOp> solutions;
  Solutions(generator, solutions);
  for (merk::SolutionOp solution : solutions) {
    for (unsigned i = static_cast<unsigned>(dead.size()); i-- > 0;) {
      if (dead[i]) {
        solution.getValuesMutable().erase(i);
      }
    }
  }
}

void PruneUnused(mlir::ModuleOp module) {
  llvm::SmallVector<mlir::Operation *> ops;
  module.walk([&](mlir::Operation *op) {
    if (mlir::isa<merk::EachOp, merk::CommitOp>(op)) {
      ops.push_back(op); // the inner ones first
    }
  });
  mlir::OpBuilder builder{module.getContext()};
  for (mlir::Operation *op : ops) {
    if (auto each{mlir::dyn_cast<merk::EachOp>(op)}) {
      mlir::Block &continuation{each.getContinuation().front()};
      llvm::SmallVector<bool> dead;
      for (mlir::BlockArgument argument : continuation.getArguments()) {
        dead.push_back(argument.use_empty());
      }
      Prune(each, each.getGenerator(), dead);
      for (unsigned i = continuation.getNumArguments(); i-- > 0;) {
        if (dead[i]) {
          continuation.eraseArgument(i);
        }
      }
      continue;
    }
    auto commit{mlir::cast<merk::CommitOp>(op)};
    llvm::SmallVector<bool> dead;
    llvm::SmallVector<mlir::Type> types;
    llvm::SmallVector<mlir::Value> kept;
    for (mlir::OpResult result : commit.getResults()) {
      dead.push_back(result.use_empty());
      if (!result.use_empty()) {
        types.push_back(result.getType());
        kept.push_back(result);
      }
    }
    if (kept.size() == commit.getNumResults()) {
      continue;
    }
    Prune(commit, commit.getGenerator(), dead);
    builder.setInsertionPoint(commit);
    auto pruned{merk::CommitOp::create(builder, commit.getLoc(), types)};
    pruned.getGenerator().takeBody(commit.getGenerator());
    for (auto [old, result] : llvm::zip_equal(kept, pruned.getResults())) {
      old.replaceAllUsesWith(result);
    }
    commit.erase();
  }
}

bool IsNondet(merk::ProcOp proc) {
  return proc.getDeterminism() == merk::Determinism::Nondet ||
      proc.getDeterminism() == merk::Determinism::Multi;
}

} // namespace

void Outliner::Run() {
  PruneUnused(module_);
  mlir::MLIRContext *context{module_.getContext()};
  mlir::Type cont{merk::ContType::get(context)};
  mlir::Type env{merk::EnvType::get(context)};
  // Every module rewrites the types of the nondet procedures it declares
  // as of those it defines, so that their calls agree.
  llvm::DenseMap<mlir::Operation *, Sink> sinks;
  for (auto proc : module_.getOps<merk::ProcOp>()) {
    if (!IsNondet(proc)) {
      continue;
    }
    llvm::SmallVector<mlir::Type> inputs{proc.getFunctionType().getInputs()};
    inputs.append({cont, env});
    proc.setFunctionType(mlir::FunctionType::get(context, inputs, {}));
    proc.setDeterminism(merk::Determinism::Semidet);
    if (!proc.getBody().empty()) {
      mlir::Block &entry{proc.getBody().front()};
      Sink sink;
      sink.cont = entry.addArgument(cont, proc.getLoc());
      sink.env = entry.addArgument(env, proc.getLoc());
      sinks[proc] = sink;
    }
  }
  for (auto proc : llvm::to_vector(module_.getOps<merk::ProcOp>())) {
    proc_ = proc;
    auto it{sinks.find(proc)};
    Rewrite(proc.getBody(), it == sinks.end() ? nullptr : &it->second, nullptr);
  }
}

void Outliner::Rewrite(
    mlir::Region &region, const Sink *succeed, const Sink *solution) {
  for (mlir::Block &block : region) {
    for (mlir::Operation &op : llvm::make_early_inc_range(block)) {
      if (auto end{mlir::dyn_cast<merk::SucceedOp>(op)}) {
        if (succeed) {
          Deliver(end, end.getOutputs(), *succeed);
        }
      } else if (auto end{mlir::dyn_cast<merk::SolutionOp>(op)}) {
        Deliver(end, end.getValues(), *solution);
      } else if (auto call{mlir::dyn_cast<merk::NondetCallOp>(op)}) {
        RewriteCall(call, succeed, solution);
      } else if (auto each{mlir::dyn_cast<merk::EachOp>(op)}) {
        RewriteEach(each, succeed, solution);
      } else if (auto commit{mlir::dyn_cast<merk::CommitOp>(op)}) {
        RewriteCommit(commit);
      } else {
        for (mlir::Region &inner : op.getRegions()) {
          Rewrite(inner, succeed, solution);
        }
      }
    }
  }
}

// The sink that a continuation that only gives its arguments, in order, as
// the solution passes them on to, which the call can then be given in its
// place.
const Sink *Outliner::PassedOn(mlir::Region &continuation, const Sink *succeed,
    const Sink *solution) {
  mlir::Block &block{continuation.front()};
  if (!llvm::hasSingleElement(block)) {
    return nullptr;
  }
  const Sink *sink{nullptr};
  mlir::OperandRange values{block.front().getOperands()};
  if (mlir::isa<merk::SucceedOp>(block.front())) {
    sink = succeed;
  } else if (mlir::isa<merk::SolutionOp>(block.front())) {
    sink = solution;
  }
  if (!sink || !sink->cont ||
      !llvm::equal(values, block.getArguments(),
          [](mlir::Value v, mlir::BlockArgument a) { return v == a; })) {
    return nullptr;
  }
  return sink;
}

void Outliner::RewriteCall(
    merk::NondetCallOp call, const Sink *succeed, const Sink *solution) {
  mlir::Location loc{call.getLoc()};
  mlir::Value cont, env;
  if (const Sink *sink{PassedOn(call.getContinuation(), succeed, solution)}) {
    cont = sink->cont;
    env = sink->env;
  } else {
    Rewrite(call.getContinuation(), succeed, solution);
    llvm::SmallVector<mlir::Value> captures;
    mlir::ArrayAttr layout;
    merk::ProcOp proc{Outline(call.getContinuation(), {}, captures, layout)};
    builder_.setInsertionPoint(call);
    env = MakeEnv(loc, captures, layout);
    cont = merk::ContRefOp::create(builder_, loc,
        merk::ContType::get(builder_.getContext()), proc.getSymName());
  }
  builder_.setInsertionPoint(call);
  llvm::SmallVector<mlir::Value> inputs{call.getInputs()};
  inputs.append({cont, env});
  merk::CallOp::create(builder_, loc, mlir::TypeRange{}, call.getCallee(), inputs);
  merk::SucceedOp::create(builder_, loc, mlir::ValueRange{});
  call.erase();
}

void Outliner::RewriteEach(
    merk::EachOp each, const Sink *succeed, const Sink *solution) {
  mlir::Location loc{each.getLoc()};
  Sink inner;
  if (const Sink *sink{PassedOn(each.getContinuation(), succeed, solution)}) {
    inner = *sink;
  } else {
    Rewrite(each.getContinuation(), succeed, solution);
    llvm::SmallVector<mlir::Value> captures;
    mlir::ArrayAttr layout;
    merk::ProcOp proc{Outline(each.getContinuation(), {}, captures, layout)};
    builder_.setInsertionPoint(each);
    inner.env = MakeEnv(loc, captures, layout);
    inner.cont = merk::ContRefOp::create(builder_, loc,
        merk::ContType::get(builder_.getContext()), proc.getSymName());
  }
  // The generator runs in the place of merk.each, giving its solutions to
  // the continuation.
  Rewrite(each.getGenerator(), succeed, &inner);
  mlir::Block *block{each->getBlock()};
  block->getOperations().splice(each->getIterator(),
      each.getGenerator().front().getOperations());
  each.erase();
}

void Outliner::RewriteCommit(merk::CommitOp commit) {
  mlir::Location loc{commit.getLoc()};
  llvm::SmallVector<mlir::Value> captures;
  mlir::ArrayAttr layout;
  llvm::SmallVector<mlir::Type> results{commit.getResultTypes()};
  merk::ProcOp proc{Outline(commit.getGenerator(), results, captures, layout)};
  Sink store;
  store.env = proc.getBody().front().getArgument(0);
  store.first = static_cast<unsigned>(captures.size());
  store.layout = layout;
  merk::ProcOp outer{proc_};
  proc_ = proc;
  Rewrite(proc.getBody(), nullptr, &store);
  proc_ = outer;
  builder_.setInsertionPoint(commit);
  mlir::Value env{MakeEnv(loc, captures, layout)};
  merk::CallOp::create(builder_, loc, mlir::TypeRange{}, proc.getSymName(),
      mlir::ValueRange{env});
  for (auto [i, result] : llvm::enumerate(commit.getResults())) {
    result.replaceAllUsesWith(merk::EnvGetOp::create(builder_, loc,
        result.getType(), env,
        builder_.getI32IntegerAttr(static_cast<int32_t>(store.first + i)),
        layout));
  }
  commit.erase();
}

void Outliner::Deliver(
    mlir::Operation *end, mlir::ValueRange values, const Sink &sink) {
  mlir::Location loc{end->getLoc()};
  builder_.setInsertionPoint(end);
  if (sink.cont) {
    merk::CallContOp::create(builder_, loc, sink.cont, sink.env, values);
  } else {
    for (auto [i, value] : llvm::enumerate(values)) {
      merk::EnvSetOp::create(builder_, loc, sink.env,
          builder_.getI32IntegerAttr(static_cast<int32_t>(sink.first + i)),
          value, sink.layout);
    }
  }
  // It stops if the continuation asks it to; the stored solution is the
  // first, after which it stops.
  merk::SucceedOp::create(builder_, loc, mlir::ValueRange{});
  end->erase();
}

// A procedure of the region's code, which takes an environment of the
// values it uses from outside it, then of slots of the types `slots`, and
// then the region's arguments.
merk::ProcOp Outliner::Outline(mlir::Region &region,
    llvm::ArrayRef<mlir::Type> slots, llvm::SmallVectorImpl<mlir::Value> &captures,
    mlir::ArrayAttr &layout) {
  mlir::MLIRContext *context{module_.getContext()};
  mlir::Location loc{region.getParentOp()->getLoc()};
  llvm::SetVector<mlir::Value> used;
  mlir::getUsedValuesDefinedAbove(region, region, used);
  captures.assign(used.begin(), used.end());
  llvm::SmallVector<mlir::Attribute> types;
  for (mlir::Value capture : captures) {
    types.push_back(mlir::TypeAttr::get(capture.getType()));
  }
  for (mlir::Type slot : slots) {
    types.push_back(mlir::TypeAttr::get(slot));
  }
  layout = mlir::ArrayAttr::get(context, types);
  mlir::Type env{merk::EnvType::get(context)};
  llvm::SmallVector<mlir::Type> inputs{env};
  for (mlir::Type type : region.front().getArgumentTypes()) {
    inputs.push_back(type);
  }
  std::string name{(proc_.getSymName() + ".cont." +
      llvm::Twine(outlined_[proc_.getSymName()]++)).str()};
  builder_.setInsertionPointAfter(proc_);
  auto proc{merk::ProcOp::create(builder_, loc, name,
      mlir::FunctionType::get(context, inputs, {}), merk::Determinism::Semidet)};
  proc.setPrivate();
  proc.getBody().takeBody(region);
  mlir::Block &entry{proc.getBody().front()};
  mlir::Value envArgument{entry.insertArgument(0u, env, loc)};
  builder_.setInsertionPointToStart(&entry);
  for (auto [i, capture] : llvm::enumerate(captures)) {
    mlir::Value value{merk::EnvGetOp::create(builder_, loc, capture.getType(),
        envArgument, builder_.getI32IntegerAttr(static_cast<int32_t>(i)),
        layout)};
    capture.replaceUsesWithIf(value, [&](mlir::OpOperand &use) {
      return proc->isProperAncestor(use.getOwner());
    });
  }
  return proc;
}

mlir::Value Outliner::MakeEnv(
    mlir::Location loc, mlir::ValueRange captures, mlir::ArrayAttr layout) {
  return merk::EnvNewOp::create(builder_, loc,
      merk::EnvType::get(module_.getContext()), captures, layout);
}

void PruneSolutions(mlir::ModuleOp module) { PruneUnused(module); }

void OutlineNondeterminism(mlir::ModuleOp module) { Outliner{module}.Run(); }

} // namespace merkurius::codegen
