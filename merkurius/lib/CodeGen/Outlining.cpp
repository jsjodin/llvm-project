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
// solution in its environment and stops; the then part of merk.soft_ite
// becomes a continuation that records in its environment that the condition
// has a solution, which the else part is tried after only if it has none.
// The step of merk.aggregate becomes a continuation that keeps the
// accumulators in its environment, in the frame of the code it came from,
// and asks to stop when the step says not to go on; its generator becomes a
// procedure that gives its solutions to it.
//
//===----------------------------------------------------------------------===//

#include "merkurius/CodeGen/Lowering.h"
#include "merkurius/Dialect/Merk/IR/MerkDialect.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
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
  void RewriteClosureCall(
      merk::NondetCallClosureOp, const Sink *succeed, const Sink *solution);
  void RewriteMethodCall(
      merk::NondetCallMethodOp, const Sink *succeed, const Sink *solution);
  Sink Continuation(mlir::Operation *call, mlir::Region &continuation,
      const Sink *succeed, const Sink *solution);
  void RewriteEach(merk::EachOp, const Sink *succeed, const Sink *solution);
  void RewriteSoftIte(merk::SoftIteOp, const Sink *succeed, const Sink *solution);
  void RewriteCommit(merk::CommitOp);
  void RewriteAggregate(merk::AggregateOp);
  void Deliver(mlir::Operation *end, mlir::ValueRange values, const Sink &);
  merk::ProcOp Outline(mlir::Region &, llvm::ArrayRef<mlir::Type> slots,
      llvm::SmallVectorImpl<mlir::Value> &captures, mlir::ArrayAttr &layout,
      mlir::ValueRange slotValues = {});
  merk::ProcOp OutlineWith(
      mlir::Region &, mlir::ValueRange captures, mlir::ArrayAttr layout);
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
// those of a merk.commit or merk.aggregate in it, but those of a merk.each's continuation, and
// those of a merk.soft_ite's then and else parts rather than its condition.
void Solutions(mlir::Region &region, llvm::SmallVectorImpl<merk::SolutionOp> &out) {
  for (mlir::Block &block : region) {
    for (mlir::Operation &op : block) {
      if (auto solution{mlir::dyn_cast<merk::SolutionOp>(op)}) {
        out.push_back(solution);
      } else if (auto each{mlir::dyn_cast<merk::EachOp>(op)}) {
        Solutions(each.getContinuation(), out);
      } else if (auto ite{mlir::dyn_cast<merk::SoftIteOp>(op)}) {
        Solutions(ite.getThenRegion(), out);
        Solutions(ite.getElseRegion(), out);
      } else if (!mlir::isa<merk::CommitOp, merk::AggregateOp>(op)) {
        for (mlir::Region &inner : op.getRegions()) {
          Solutions(inner, out);
        }
      }
    }
  }
}

// Leaves out of a generator's solutions the values that nothing uses: the
// arguments of merk.each's continuation, of merk.soft_ite's then part or of
// merk.aggregate's step, for a solution, that it does not use, and the
// results of merk.commit that nothing uses, which
// code generation gives for every variable the goal binds.
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

// Whether nothing uses `value`, forgetting a note of its type.
bool Unused(mlir::Value value) {
  if (!llvm::all_of(value.getUsers(), [](mlir::Operation *user) {
        return mlir::isa<merk::RootTypeOp>(user);
      })) {
    return false;
  }
  for (mlir::Operation *user : llvm::make_early_inc_range(value.getUsers())) {
    user->erase();
  }
  return true;
}

// The note of a value's type, if it has one or is the same value at another
// type, across a conversion to or from a word.
merk::RootTypeOp NoteOf(mlir::Value value) {
  llvm::SetVector<mlir::Value> seen;
  seen.insert(value);
  for (std::size_t i = 0; i < seen.size(); ++i) {
    mlir::Value at{seen[i]};
    if (mlir::Operation *def{at.getDefiningOp()};
        def && mlir::isa<merk::ToWordOp, merk::FromWordOp>(def)) {
      seen.insert(def->getOperand(0));
    }
    for (mlir::Operation *user : at.getUsers()) {
      if (auto note{mlir::dyn_cast<merk::RootTypeOp>(user)}) {
        return note;
      }
      if (mlir::isa<merk::ToWordOp, merk::FromWordOp>(user)) {
        seen.insert(user->getResult(0));
      }
    }
  }
  return nullptr;
}

// Adds to `captures` the type_infos of each of them and of the values that
// will be in the environment's slots, available at `at`, so that the outlined
// code and a walk of the environment can say what they are.
void CaptureTypeInfos(llvm::SmallVectorImpl<mlir::Value> &captures,
    mlir::Operation *at, mlir::ValueRange slots = {}) {
  mlir::DominanceInfo dominance{at->getParentOfType<merk::ProcOp>()};
  llvm::SetVector<mlir::Value> all{captures.begin(), captures.end()};
  llvm::SmallVector<mlir::Value> described{captures.begin(), captures.end()};
  described.append(slots.begin(), slots.end());
  for (mlir::Value value : described) {
    merk::RootTypeOp note{NoteOf(value)};
    if (note &&
        llvm::all_of(note.getInfos(), [&](mlir::Value info) {
          return dominance.properlyDominates(info, at);
        })) {
      all.insert(note.getInfos().begin(), note.getInfos().end());
    }
  }
  captures.assign(all.begin(), all.end());
}

// Records on an environment where the type_infos of its slots are: for each
// slot after the captures, the captures' positions.
void DescribeSlots(mlir::Value env, mlir::ValueRange captures,
    mlir::ValueRange slots) {
  mlir::Builder builder{env.getContext()};
  llvm::SmallVector<mlir::Attribute> described;
  for (mlir::Value slot : slots) {
    llvm::SmallVector<std::int64_t> places;
    if (merk::RootTypeOp note{NoteOf(slot)}) {
      for (mlir::Value info : note.getInfos()) {
        auto found{llvm::find(captures, info)};
        if (found == captures.end()) {
          places.clear();
          break;
        }
        places.push_back(found - captures.begin());
      }
    }
    described.push_back(builder.getI64ArrayAttr(places));
  }
  env.getDefiningOp()->setAttr("slot_infos", builder.getArrayAttr(described));
}

void PruneUnused(mlir::ModuleOp module) {
  llvm::SmallVector<mlir::Operation *> ops;
  module.walk([&](mlir::Operation *op) {
    if (mlir::isa<merk::EachOp, merk::SoftIteOp, merk::CommitOp,
            merk::AggregateOp>(op)) {
      ops.push_back(op); // the inner ones first
    }
  });
  mlir::OpBuilder builder{module.getContext()};
  for (mlir::Operation *op : ops) {
    auto each{mlir::dyn_cast<merk::EachOp>(op)};
    auto ite{mlir::dyn_cast<merk::SoftIteOp>(op)};
    if (each || ite) {
      mlir::Block &continuation{
          each ? each.getContinuation().front() : ite.getThenRegion().front()};
      llvm::SmallVector<bool> dead;
      for (mlir::BlockArgument argument : continuation.getArguments()) {
        dead.push_back(Unused(argument));
      }
      Prune(op, each ? each.getGenerator() : ite.getCondition(), dead);
      for (unsigned i = continuation.getNumArguments(); i-- > 0;) {
        if (dead[i]) {
          continuation.eraseArgument(i);
        }
      }
      continue;
    }
    if (auto aggregate{mlir::dyn_cast<merk::AggregateOp>(op)}) {
      mlir::Block &step{aggregate.getStep().front()};
      unsigned given{step.getNumArguments() - aggregate.getNumResults()};
      llvm::SmallVector<bool> dead;
      for (unsigned i{0}; i < given; ++i) {
        dead.push_back(Unused(step.getArgument(i)));
      }
      Prune(op, aggregate.getGenerator(), dead);
      for (unsigned i = given; i-- > 0;) {
        if (dead[i]) {
          step.eraseArgument(i);
        }
      }
      continue;
    }
    auto commit{mlir::cast<merk::CommitOp>(op)};
    llvm::SmallVector<bool> dead;
    llvm::SmallVector<mlir::Type> types;
    llvm::SmallVector<mlir::Value> kept;
    for (mlir::OpResult result : commit.getResults()) {
      dead.push_back(Unused(result));
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
      } else if (auto call{mlir::dyn_cast<merk::NondetCallClosureOp>(op)}) {
        RewriteClosureCall(call, succeed, solution);
      } else if (auto call{mlir::dyn_cast<merk::NondetCallMethodOp>(op)}) {
        RewriteMethodCall(call, succeed, solution);
      } else if (auto each{mlir::dyn_cast<merk::EachOp>(op)}) {
        RewriteEach(each, succeed, solution);
      } else if (auto ite{mlir::dyn_cast<merk::SoftIteOp>(op)}) {
        RewriteSoftIte(ite, succeed, solution);
      } else if (auto commit{mlir::dyn_cast<merk::CommitOp>(op)}) {
        RewriteCommit(commit);
      } else if (auto aggregate{mlir::dyn_cast<merk::AggregateOp>(op)}) {
        RewriteAggregate(aggregate);
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
  mlir::Operation *only{nullptr};
  for (mlir::Operation &op : block) {
    if (mlir::isa<merk::RootTypeOp>(op)) {
      continue;
    }
    if (only) {
      return nullptr;
    }
    only = &op;
  }
  const Sink *sink{nullptr};
  mlir::OperandRange values{only->getOperands()};
  if (mlir::isa<merk::SucceedOp>(only)) {
    sink = succeed;
  } else if (mlir::isa<merk::SolutionOp>(only)) {
    sink = solution;
  }
  if (!sink || !sink->cont ||
      !llvm::equal(values, block.getArguments(),
          [](mlir::Value v, mlir::BlockArgument a) { return v == a; })) {
    return nullptr;
  }
  return sink;
}

// The continuation and environment that a call is given for the rest of
// the code: its caller's, if the rest only gives its arguments as the
// solution, and otherwise the rest outlined, with an environment of what it
// uses in the caller's frame.
Sink Outliner::Continuation(mlir::Operation *call, mlir::Region &continuation,
    const Sink *succeed, const Sink *solution) {
  if (const Sink *sink{PassedOn(continuation, succeed, solution)}) {
    return *sink;
  }
  mlir::Location loc{call->getLoc()};
  Rewrite(continuation, succeed, solution);
  llvm::SmallVector<mlir::Value> captures;
  mlir::ArrayAttr layout;
  merk::ProcOp proc{Outline(continuation, {}, captures, layout)};
  builder_.setInsertionPoint(call);
  Sink sink;
  sink.env = MakeEnv(loc, captures, layout);
  sink.cont = merk::ContRefOp::create(builder_, loc,
      merk::ContType::get(builder_.getContext()), proc.getSymName());
  return sink;
}

void Outliner::RewriteCall(
    merk::NondetCallOp call, const Sink *succeed, const Sink *solution) {
  mlir::Location loc{call.getLoc()};
  Sink sink{Continuation(call, call.getContinuation(), succeed, solution)};
  builder_.setInsertionPoint(call);
  llvm::SmallVector<mlir::Value> inputs{call.getInputs()};
  inputs.append({sink.cont, sink.env});
  merk::CallOp::create(builder_, loc, mlir::TypeRange{}, call.getCallee(), inputs);
  merk::SucceedOp::create(builder_, loc, mlir::ValueRange{});
  call.erase();
}

// A semidet call of the closure's entry, which outlining has rewritten as
// it does every nondet procedure.
void Outliner::RewriteClosureCall(merk::NondetCallClosureOp call,
    const Sink *succeed, const Sink *solution) {
  mlir::Location loc{call.getLoc()};
  Sink sink{Continuation(call, call.getContinuation(), succeed, solution)};
  builder_.setInsertionPoint(call);
  llvm::SmallVector<mlir::Value> inputs{call.getInputs()};
  inputs.append({sink.cont, sink.env});
  merk::CallClosureOp::create(builder_, loc, mlir::TypeRange{},
      call.getClosure(), inputs, merk::Determinism::Semidet);
  merk::SucceedOp::create(builder_, loc, mlir::ValueRange{});
  call.erase();
}

// A semidet call of the instance's dispatch procedure, which outlining has
// rewritten as it does every nondet procedure.
void Outliner::RewriteMethodCall(merk::NondetCallMethodOp call,
    const Sink *succeed, const Sink *solution) {
  mlir::Location loc{call.getLoc()};
  Sink sink{Continuation(call, call.getContinuation(), succeed, solution)};
  builder_.setInsertionPoint(call);
  llvm::SmallVector<mlir::Value> inputs{call.getInputs()};
  inputs.append({sink.cont, sink.env});
  merk::CallMethodOp::create(builder_, loc, mlir::TypeRange{}, call.getInfo(),
      call.getMethod(), inputs, merk::Determinism::Semidet);
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

// The then part becomes a continuation whose environment has a slot after
// the values it uses, which it sets on entry: that the condition has a
// solution. The condition runs in the place of merk.soft_ite, giving its
// solutions to it, as the first branch of a disjunction whose second, run
// when the condition has no more, is the else part if the slot is not set.
void Outliner::RewriteSoftIte(
    merk::SoftIteOp ite, const Sink *succeed, const Sink *solution) {
  mlir::Location loc{ite.getLoc()};
  mlir::Type i1{builder_.getI1Type()};
  Rewrite(ite.getThenRegion(), succeed, solution);
  llvm::SmallVector<mlir::Value> captures;
  mlir::ArrayAttr layout;
  merk::ProcOp proc{Outline(ite.getThenRegion(), {i1}, captures, layout)};
  mlir::IntegerAttr found{
      builder_.getI32IntegerAttr(static_cast<int32_t>(captures.size()))};
  mlir::Block &entry{proc.getBody().front()};
  builder_.setInsertionPointToStart(&entry);
  merk::EnvSetOp::create(builder_, loc, entry.getArgument(0), found,
      mlir::arith::ConstantIntOp::create(builder_, loc, i1, 1), layout);
  builder_.setInsertionPoint(ite);
  Sink inner;
  inner.env = MakeEnv(loc, captures, layout);
  merk::EnvSetOp::create(builder_, loc, inner.env, found,
      mlir::arith::ConstantIntOp::create(builder_, loc, i1, 0), layout);
  inner.cont = merk::ContRefOp::create(builder_, loc,
      merk::ContType::get(builder_.getContext()), proc.getSymName());
  Rewrite(ite.getCondition(), succeed, &inner);
  Rewrite(ite.getElseRegion(), succeed, solution);
  builder_.setInsertionPoint(ite);
  auto disj{merk::DisjOp::create(builder_, loc, 2)};
  mlir::Block &first{disj.getBranches()[0].emplaceBlock()};
  first.getOperations().splice(
      first.end(), ite.getCondition().front().getOperations());
  mlir::Block &second{disj.getBranches()[1].emplaceBlock()};
  builder_.setInsertionPointToStart(&second);
  mlir::Value isSet{merk::EnvGetOp::create(builder_, loc, i1, inner.env, found,
      layout)};
  merk::GuardOp::create(builder_, loc,
      mlir::arith::XOrIOp::create(builder_, loc, isSet,
          mlir::arith::ConstantIntOp::create(builder_, loc, i1, 1)));
  second.getOperations().splice(
      second.end(), ite.getElseRegion().front().getOperations());
  ite.erase();
}

void Outliner::RewriteCommit(merk::CommitOp commit) {
  mlir::Location loc{commit.getLoc()};
  llvm::SmallVector<mlir::Value> captures;
  mlir::ArrayAttr layout;
  llvm::SmallVector<mlir::Type> results{commit.getResultTypes()};
  merk::ProcOp proc{Outline(
      commit.getGenerator(), results, captures, layout, commit.getResults())};
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
  DescribeSlots(env, captures, commit.getResults());
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

// The index of an environment's slot.
static mlir::IntegerAttr Slot(mlir::OpBuilder &builder, std::size_t index) {
  return builder.getI32IntegerAttr(static_cast<int32_t>(index));
}

// The generator and the step share an environment: the values either uses
// from outside it, then a slot for each accumulator. The step becomes a
// continuation that takes a solution's values, reads the accumulators from
// their slots, and writes the new ones back; it fails, to have the
// generator go on, unless the step said not to. The generator becomes a
// procedure that gives its solutions to it, which fails when it has no more
// and succeeds when the step stopped it, and is called in the place of
// merk.aggregate in the condition of an if-then-else, so that either way the
// code after it reads the accumulators from their slots.
void Outliner::RewriteAggregate(merk::AggregateOp aggregate) {
  mlir::MLIRContext *context{module_.getContext()};
  mlir::Location loc{aggregate.getLoc()};
  mlir::Type i1{builder_.getI1Type()};
  Rewrite(aggregate.getStep(), nullptr, nullptr);
  llvm::SetVector<mlir::Value> used;
  mlir::getUsedValuesDefinedAbove(aggregate->getRegions(), used);
  llvm::SmallVector<mlir::Value> captures{used.begin(), used.end()};
  CaptureTypeInfos(captures, aggregate, aggregate.getInitial());
  llvm::SmallVector<mlir::Attribute> types;
  for (mlir::Value capture : captures) {
    types.push_back(mlir::TypeAttr::get(capture.getType()));
  }
  for (mlir::Type type : aggregate.getResultTypes()) {
    types.push_back(mlir::TypeAttr::get(type));
  }
  mlir::ArrayAttr layout{mlir::ArrayAttr::get(context, types)};
  // The accumulators' slots follow the captures'.
  std::size_t base{captures.size()};
  // The step: its accumulator arguments become reads of their slots.
  unsigned count{aggregate.getNumResults()};
  merk::ProcOp step{OutlineWith(aggregate.getStep(), captures, layout)};
  mlir::Block &entry{step.getBody().front()};
  mlir::Value env{entry.getArgument(0)};
  unsigned first{entry.getNumArguments() - count};
  for (unsigned i{0}; i < count; ++i) {
    mlir::BlockArgument argument{entry.getArgument(first + i)};
    argument.replaceAllUsesWith(merk::EnvGetOp::create(
        builder_, loc, argument.getType(), env, Slot(builder_, base + i), layout));
  }
  entry.eraseArguments(first, count);
  step.setFunctionType(mlir::FunctionType::get(context,
      entry.getArgumentTypes(), {}));
  auto yield{mlir::cast<merk::YieldOp>(entry.getTerminator())};
  builder_.setInsertionPoint(yield);
  for (auto [i, value] : llvm::enumerate(yield.getValues().drop_front())) {
    merk::EnvSetOp::create(builder_, loc, env, Slot(builder_, base + i), value, layout);
  }
  merk::GuardOp::create(builder_, loc,
      mlir::arith::XOrIOp::create(builder_, loc, yield.getValues().front(),
          mlir::arith::ConstantIntOp::create(builder_, loc, i1, 1)));
  merk::SucceedOp::create(builder_, loc, mlir::ValueRange{});
  yield.erase();
  // The generator, which gives its solutions to the step.
  merk::ProcOp generator{
      OutlineWith(aggregate.getGenerator(), captures, layout)};
  mlir::Block &start{generator.getBody().front()};
  builder_.setInsertionPointToStart(&start);
  Sink sink;
  sink.env = start.getArgument(0);
  sink.cont = merk::ContRefOp::create(builder_, loc,
      merk::ContType::get(context), step.getSymName());
  merk::ProcOp outer{proc_};
  proc_ = generator;
  Rewrite(generator.getBody(), nullptr, &sink);
  proc_ = outer;
  builder_.setInsertionPoint(aggregate);
  mlir::Value shared{MakeEnv(loc, captures, layout)};
  DescribeSlots(shared, captures, aggregate.getInitial());
  for (auto [i, value] : llvm::enumerate(aggregate.getInitial())) {
    merk::EnvSetOp::create(builder_, loc, shared, Slot(builder_, base + i), value, layout);
  }
  auto ite{merk::IteOp::create(builder_, loc, mlir::TypeRange{})};
  for (mlir::Region &region : ite->getRegions()) {
    builder_.setInsertionPointToStart(&region.emplaceBlock());
    merk::YieldOp::create(builder_, loc, mlir::ValueRange{});
  }
  builder_.setInsertionPointToStart(&ite.getCondition().front());
  merk::CallOp::create(builder_, loc, mlir::TypeRange{},
      generator.getSymName(), mlir::ValueRange{shared});
  builder_.setInsertionPointAfter(ite);
  for (auto [i, result] : llvm::enumerate(aggregate.getResults())) {
    result.replaceAllUsesWith(merk::EnvGetOp::create(
        builder_, loc, result.getType(), shared, Slot(builder_, base + i), layout));
  }
  aggregate.erase();
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
    mlir::ArrayAttr &layout, mlir::ValueRange slotValues) {
  mlir::MLIRContext *context{module_.getContext()};
  mlir::Location loc{region.getParentOp()->getLoc()};
  llvm::SetVector<mlir::Value> used;
  mlir::getUsedValuesDefinedAbove(region, region, used);
  captures.assign(used.begin(), used.end());
  CaptureTypeInfos(captures, region.getParentOp(), slotValues);
  llvm::SmallVector<mlir::Attribute> types;
  for (mlir::Value capture : captures) {
    types.push_back(mlir::TypeAttr::get(capture.getType()));
  }
  for (mlir::Type slot : slots) {
    types.push_back(mlir::TypeAttr::get(slot));
  }
  layout = mlir::ArrayAttr::get(context, types);
  return OutlineWith(region, captures, layout);
}

// A procedure of the region's code, which takes an environment of the
// layout, whose first slots are the captures, and then the region's
// arguments.
merk::ProcOp Outliner::OutlineWith(
    mlir::Region &region, mlir::ValueRange captures, mlir::ArrayAttr layout) {
  mlir::MLIRContext *context{module_.getContext()};
  mlir::Location loc{region.getParentOp()->getLoc()};
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
  llvm::SmallVector<mlir::Value> values;
  for (auto [i, capture] : llvm::enumerate(captures)) {
    mlir::Value value{merk::EnvGetOp::create(builder_, loc, capture.getType(),
        envArgument, builder_.getI32IntegerAttr(static_cast<int32_t>(i)),
        layout)};
    capture.replaceUsesWithIf(value, [&](mlir::OpOperand &use) {
      return proc->isProperAncestor(use.getOwner());
    });
    values.push_back(value);
  }
  // What the captures' types are, now that they are this procedure's.
  for (auto [capture, value] : llvm::zip_equal(captures, values)) {
    merk::RootTypeOp note{NoteOf(capture)};
    if (!note) {
      continue;
    }
    llvm::SmallVector<mlir::Value> infos;
    for (mlir::Value info : note.getInfos()) {
      auto found{llvm::find(captures, info)};
      if (found == captures.end()) {
        infos.clear();
        break;
      }
      infos.push_back(values[found - captures.begin()]);
    }
    if (!infos.empty()) {
      merk::RootTypeOp::create(builder_, loc, value, infos);
    }
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
