//===-- lib/CodeGen/Frees.cpp - Freeing a GPU process's values ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Puts a merk.free where each cell, array and string that a GPU process's
// code makes or receives is last used (docs/INPROGRESS.md, Memory that a GPU
// process gives back, f3). GPU code makes only flat values, so the values of
// a procedure fall into classes that live and die together, found by
// union-find: a value and what it is made of, a field and the cell it came
// from, an array and the arrays that updating it in place gives, a call's
// result and the arguments the callee may return in it. A class is freed
// only when everything in it is the procedure's own: nothing in it is an
// argument, the result of a call the pass knows nothing of, or returned, or
// given to a foreign call that may keep it. A send copies its message, and
// a loop across the process's teams ends before the process goes on, so
// neither keeps what it is given.
//
// What is freed is each block that the class allocated: a construction, an
// array's, a received message, which the runtime keeps as one group of
// blocks, and a call's result that the callee's summary says is a new
// block. Each is freed once on each path that made it, after the class's
// last use, in each branch of an if-then-else or a switch where the last use
// is one, and never between a tail call and its return, where the process's
// next state is called: a class whose last use is there is left to the
// process's end, as are the blocks of a branch that the code after it
// cannot name. A block that is stored in one cell and used nowhere else, as
// a received message is in its `message(...)`, or that is a branch's result
// alone, is freed through what holds it: the field is taken where the
// functor is the one it was stored with, and freed before its cell.
//
// The summaries of the module's procedures, for their callers, are found by
// iterating to a fixed point: for each output, which inputs it may be or
// hold, whether it is always the same block as one input, and whether it is
// a new block; for each input, whether the procedure keeps it.
//
//===----------------------------------------------------------------------===//

#include "merkurius/CodeGen/Lowering.h"
#include "merkurius/Dialect/Merk/IR/MerkDialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include <optional>

namespace merkurius::codegen {
namespace {

// A value with a block of its own: a term, which may be a cell, an array or a
// string.
bool IsHeap(mlir::Type type) {
  return mlir::isa<merk::TermType, merk::ArrayType, merk::StringType>(type);
}

// A value that may hold a block: one of those, or a word.
bool MayHold(mlir::Type type) {
  return IsHeap(type) || mlir::isa<merk::WordType>(type);
}

// Where a value that may hold a block comes from.
enum class Origin {
  // A new block: a construction, an array's, a received message, a fresh
  // call result, or the result of a branch whose every part gives a new
  // block or a constant.
  Alloc,
  // The same block as another value, or one of several.
  Alias,
  // A field of a cell, or an element of an array.
  Part,
  // A constant or a static cell, which nothing frees.
  Static,
  // An argument of the procedure.
  Arg,
  // Anything else, which the pass knows nothing of.
  Unknown,
  // A call's result whose summary the iteration has not found yet.
  Pending,
};

struct ResultSummary {
  bool determined{false};
  // What it may hold is not known.
  bool unknown{false};
  // A new block, of the caller's from then on.
  bool fresh{false};
  // The input that it always is, if any.
  int mustArg{-1};
  // The inputs that it may be or hold.
  llvm::SmallVector<unsigned> args;

  bool operator==(const ResultSummary &other) const {
    return determined == other.determined && unknown == other.unknown &&
        fresh == other.fresh && mustArg == other.mustArg && args == other.args;
  }
};

struct Summary {
  llvm::SmallVector<ResultSummary> results;
  // Whether the procedure keeps the input past its return, other than in its
  // results.
  llvm::SmallVector<bool> kept;

  bool operator==(const Summary &other) const {
    return results == other.results && kept == other.kept;
  }
};

// One step from a cell to a block it holds: the field `index` of a cell of
// `functor`, whose fields are of `fields`' types.
struct Step {
  merk::FunctorAttr functor;
  unsigned index;
  llvm::SmallVector<mlir::Type> fields;

  bool operator==(const Step &other) const {
    return functor == other.functor && index == other.index;
  }
};
using Path = llvm::SmallVector<Step, 2>;
using Paths = llvm::SmallVector<Path>;

// For each functor that a value may have, the paths to the blocks it holds
// that are freed through it; none if what it may be is not known.
using Functors = llvm::SmallVector<std::pair<merk::FunctorAttr, Paths>>;
using Shape = std::optional<Functors>;

// A point where a class is freed: after `after`, or at the start of `block`.
struct Point {
  mlir::Block *block;
  mlir::Operation *after;
};

bool IsLoopCall(llvm::StringRef callee) {
  return callee == "merk_device_process_loop" ||
      callee == "merk_device_process_distribute" ||
      callee == "merk_device_team_loop";
}

// Whether `op`, followed only by the terminators of its regions, ends the
// procedure: a call there is a tail call. A condition's yield goes on to
// the then part.
bool InTailPosition(mlir::Operation *op) {
  mlir::Operation *next{op->getNextNode()};
  if (!next || !next->hasTrait<mlir::OpTrait::IsTerminator>()) {
    return false;
  }
  if (mlir::isa<merk::SucceedOp>(next)) {
    return true;
  }
  mlir::Operation *parent{op->getParentOp()};
  if (!mlir::isa<merk::YieldOp>(next) ||
      !mlir::isa<merk::IteOp, merk::SwitchOp>(parent)) {
    return false;
  }
  if (auto ite{mlir::dyn_cast<merk::IteOp>(parent)};
      ite && op->getParentRegion() == &ite.getCondition()) {
    return false;
  }
  return InTailPosition(parent);
}

bool ContainsCall(mlir::Operation *op) {
  return op->walk([](merk::CallOp) { return mlir::WalkResult::interrupt(); })
      .wasInterrupted();
}

// What a part of an if-then-else or a switch gives its op's results: its
// yield, or null if it fails, as a merk.fail ends it. `other` is set when it
// ends otherwise, as nondeterministic code may.
merk::YieldOp YieldOf(mlir::Region *branch, bool &other) {
  mlir::Operation *terminator{branch->front().getTerminator()};
  if (auto yield{mlir::dyn_cast_or_null<merk::YieldOp>(terminator)}) {
    return yield;
  }
  if (!mlir::isa_and_present<merk::FailOp>(terminator)) {
    other = true;
  }
  return {};
}

// One procedure's classes, and where they are freed.
class Analysis {
public:
  Analysis(merk::ProcOp proc, const llvm::StringMap<Summary> &summaries,
      const llvm::StringSet<> &noTag)
      : proc_{proc}, summaries_{summaries}, noTag_{noTag} {}

  void Run();
  Summary Summarize();
  void InsertFrees();

private:
  mlir::Value Find(mlir::Value);
  void Join(mlir::Value, mlir::Value);
  void Track(mlir::Value, Origin, mlir::Value id = {});
  bool Tracked(mlir::Value value) const { return origin_.count(value); }
  Origin OriginOf(mlir::Value value) const { return origin_.lookup(value); }
  mlir::Value IdOf(mlir::Value value) const { return id_.lookup(value); }
  void Escape(mlir::Value);
  void EscapeOperands(mlir::Operation *);
  void UnknownResults(mlir::Operation *);

  void VisitBlock(mlir::Block &);
  void Visit(mlir::Operation *);
  void VisitBranches(mlir::Operation *, llvm::ArrayRef<mlir::Region *>);
  void VisitCall(merk::CallOp);
  void VisitForeignCall(merk::ForeignCallOp);
  void VisitStore(mlir::LLVM::StoreOp);

  bool HiddenContained(mlir::Value) const;
  Shape ShapeOf(mlir::Value) const;
  Paths PathsOf(mlir::Value) const;

  void NoteTouches(mlir::Operation *);
  bool Touches(mlir::Operation *op, mlir::Value cls) const {
    auto it{touches_.find(op)};
    return it != touches_.end() && it->second.contains(cls);
  }
  void Place(mlir::Value cls, mlir::Region &, llvm::SmallVector<Point> &);
  bool Visible(mlir::Value, const Point &) const;
  void EmitFree(mlir::OpBuilder &, mlir::Location, mlir::Value, const Path &,
      unsigned step);

  merk::ProcOp proc_;
  const llvm::StringMap<Summary> &summaries_;
  // The types whose one constructor has one field, which is the term itself
  // (Lowering.cpp, Layouts, TermLayout::Kind::NoTag): no cell.
  const llvm::StringSet<> &noTag_;
  // The values that may hold a block, in the order they are defined.
  llvm::SmallVector<mlir::Value> values_;
  llvm::DenseMap<mlir::Value, Origin> origin_;
  // The value whose block a value always is: itself for a new block.
  llvm::DenseMap<mlir::Value, mlir::Value> id_;
  llvm::DenseMap<mlir::Value, mlir::Value> parent_;
  llvm::SmallVector<mlir::Value> escaping_;
  llvm::SmallVector<mlir::Value> returned_;
  // The values that a loop's call uses, by their words in its buffer.
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<mlir::Value>> loopUses_;
  // The buffers of the words of a loop's values.
  llvm::DenseSet<mlir::Value> buffers_;
  // The words of messages that the runtime copied into blocks of their own.
  llvm::DenseSet<mlir::Value> received_;
  // A new block that a branch yields, and the result that is its block after
  // the branch.
  llvm::DenseMap<mlir::Value, mlir::Value> yieldedInto_;
  // Each operation's classes, its own and its regions' operations'.
  llvm::DenseMap<mlir::Operation *, llvm::DenseSet<mlir::Value>> touches_;
};

mlir::Value Analysis::Find(mlir::Value value) {
  mlir::Value root{value};
  while (true) {
    auto it{parent_.find(root)};
    if (it == parent_.end() || it->second == root) {
      break;
    }
    root = it->second;
  }
  // Shortens the chain.
  while (value != root) {
    mlir::Value next{parent_[value]};
    parent_[value] = root;
    value = next;
  }
  return root;
}

void Analysis::Join(mlir::Value a, mlir::Value b) {
  if (!Tracked(a) || !Tracked(b)) {
    return;
  }
  mlir::Value ra{Find(a)}, rb{Find(b)};
  if (ra != rb) {
    parent_[rb] = ra;
  }
}

void Analysis::Track(mlir::Value value, Origin origin, mlir::Value id) {
  if (!Tracked(value)) {
    values_.push_back(value);
    parent_[value] = value;
  }
  origin_[value] = origin;
  if (id) {
    id_[value] = id;
  }
}

void Analysis::Escape(mlir::Value value) {
  if (Tracked(value)) {
    escaping_.push_back(value);
  }
}

void Analysis::EscapeOperands(mlir::Operation *op) {
  for (mlir::Value operand : op->getOperands()) {
    Escape(operand);
  }
}

void Analysis::UnknownResults(mlir::Operation *op) {
  for (mlir::Value result : op->getResults()) {
    if (MayHold(result.getType())) {
      Track(result, Origin::Unknown);
    }
  }
}

void Analysis::Run() {
  if (proc_.isExternal()) {
    return;
  }
  mlir::Block &entry{proc_.getBody().front()};
  for (mlir::BlockArgument argument : entry.getArguments()) {
    if (MayHold(argument.getType())) {
      Track(argument, Origin::Arg, argument);
    }
  }
  VisitBlock(entry);
}

void Analysis::VisitBlock(mlir::Block &block) {
  for (mlir::Operation &op : block) {
    Visit(&op);
  }
}

// The results of an if-then-else or a switch, from what each part yields.
// A result is a new block when each part yields a new block that it made, or
// a constant; it is the same block as a value from before when each part
// yields that value; else one of several. Parts that yield what a call whose
// summary is pending gives are left out, as the iteration's guess.
void Analysis::VisitBranches(
    mlir::Operation *op, llvm::ArrayRef<mlir::Region *> branches) {
  llvm::DenseSet<mlir::Value> hidden;
  for (auto [index, result] : llvm::enumerate(op->getResults())) {
    if (!MayHold(result.getType())) {
      continue;
    }
    bool fresh{true}, same{true}, any{false}, allStatic{true}, other{false};
    mlir::Value outer;
    llvm::SmallVector<mlir::Value> made;
    for (mlir::Region *branch : branches) {
      merk::YieldOp yield{YieldOf(branch, other)};
      if (!yield) {
        continue;
      }
      mlir::Value yielded{yield.getValues()[index]};
      if (!Tracked(yielded)) {
        fresh = same = false;
        continue;
      }
      Origin origin{OriginOf(yielded)};
      if (origin == Origin::Pending) {
        continue;
      }
      any = true;
      if (origin == Origin::Static) {
        same = false;
        continue;
      }
      allStatic = false;
      mlir::Value id{IdOf(yielded)};
      if (id && branch->isAncestor(id.getParentRegion())) {
        // A block that the part made: a new block of the result's, unless
        // another result has it already.
        if (OriginOf(id) == Origin::Alloc && IsHeap(id.getType()) &&
            !hidden.contains(id)) {
          made.push_back(id);
        } else {
          fresh = false;
        }
        same = false;
      } else {
        fresh = false;
        if (!id || (outer && outer != id)) {
          same = false;
        }
        outer = id;
      }
    }
    if (other) {
      Track(result, Origin::Unknown);
    } else if (!any) {
      Track(result, Origin::Pending);
    } else if (allStatic) {
      Track(result, Origin::Static);
    } else if (fresh && IsHeap(result.getType())) {
      Track(result, Origin::Alloc, result);
      for (mlir::Value block : made) {
        hidden.insert(block);
        yieldedInto_[block] = result;
      }
    } else if (same && outer) {
      Track(result, Origin::Alias, outer);
    } else {
      Track(result, Origin::Alias);
    }
    for (mlir::Region *branch : branches) {
      if (merk::YieldOp yield{YieldOf(branch, other)}) {
        Join(result, yield.getValues()[index]);
      }
    }
  }
}

void Analysis::VisitCall(merk::CallOp call) {
  auto it{summaries_.find(call.getCallee())};
  if (it == summaries_.end()) {
    EscapeOperands(call);
    UnknownResults(call);
    return;
  }
  const Summary &summary{it->second};
  for (auto [index, input] : llvm::enumerate(call.getInputs())) {
    if (index < summary.kept.size() && summary.kept[index]) {
      Escape(input);
    }
  }
  for (auto [index, output] : llvm::enumerate(call.getOutputs())) {
    if (!MayHold(output.getType())) {
      continue;
    }
    const ResultSummary &result{summary.results[index]};
    if (!result.determined) {
      Track(output, Origin::Pending);
      continue;
    }
    if (result.unknown) {
      Track(output, Origin::Unknown);
      continue;
    }
    if (result.fresh && IsHeap(output.getType())) {
      Track(output, Origin::Alloc, output);
    } else if (result.mustArg >= 0 &&
        Tracked(call.getInputs()[result.mustArg])) {
      mlir::Value input{call.getInputs()[result.mustArg]};
      Track(output,
          OriginOf(input) == Origin::Static ? Origin::Static : Origin::Alias,
          IdOf(input));
    } else {
      Track(output, Origin::Alias);
    }
    for (unsigned arg : result.args) {
      Join(output, call.getInputs()[arg]);
    }
  }
}

void Analysis::VisitForeignCall(merk::ForeignCallOp call) {
  llvm::StringRef callee{call.getCallee()};
  // A send copies its message, or stops the program for a team mailbox
  // (runtime/device/device.cpp, TeamBlockFromProcess).
  if (callee == "merk_device_send" || callee == "merk_device_send_term" ||
      callee == "merk_device_send_array" || IsLoopCall(callee)) {
    return;
  }
  if (callee == "merk_device_receive_term" ||
      callee == "merk_device_receive_array") {
    received_.insert(call.getOutputs()[1]);
    return;
  }
  if (callee == "merk_device_loop_values" ||
      callee == "merk_device_team_values") {
    buffers_.insert(call.getOutputs()[0]);
    return;
  }
  EscapeOperands(call);
  UnknownResults(call);
}

// A value's word stored in a loop's buffer is used until the loop's call
// that follows, which returns when the loop has ended
// (MLIRGenProcesses.cpp, ProcessLoopCall).
void Analysis::VisitStore(mlir::LLVM::StoreOp store) {
  mlir::Value value{store.getValue()};
  Escape(store.getAddr());
  if (!Tracked(value)) {
    return;
  }
  auto gep{store.getAddr().getDefiningOp<mlir::LLVM::GEPOp>()};
  if (gep && buffers_.contains(gep.getBase())) {
    for (mlir::Operation *next{store->getNextNode()}; next;
        next = next->getNextNode()) {
      auto call{mlir::dyn_cast<merk::ForeignCallOp>(next)};
      if (call && IsLoopCall(call.getCallee())) {
        loopUses_[call].push_back(value);
        return;
      }
    }
  }
  Escape(value);
}

void Analysis::Visit(mlir::Operation *op) {
  if (auto ite{mlir::dyn_cast<merk::IteOp>(op)}) {
    VisitBlock(ite.getCondition().front());
    bool other{false};
    merk::YieldOp yield{YieldOf(&ite.getCondition(), other)};
    mlir::Block &then{ite.getThenRegion().front()};
    for (auto [index, argument] : llvm::enumerate(then.getArguments())) {
      if (!MayHold(argument.getType())) {
        continue;
      }
      mlir::Value value{yield ? yield.getValues()[index] : mlir::Value{}};
      if (value && Tracked(value)) {
        Track(argument,
            OriginOf(value) == Origin::Static ? Origin::Static : Origin::Alias);
        Join(argument, value);
      } else {
        Track(argument, Origin::Unknown);
      }
    }
    VisitBlock(then);
    VisitBlock(ite.getElseRegion().front());
    VisitBranches(ite, {&ite.getThenRegion(), &ite.getElseRegion()});
    return;
  }
  if (auto switchOp{mlir::dyn_cast<merk::SwitchOp>(op)}) {
    llvm::SmallVector<mlir::Region *> branches;
    for (mlir::Region &region : switchOp.getCaseRegions()) {
      VisitBlock(region.front());
      branches.push_back(&region);
    }
    VisitBranches(switchOp, branches);
    return;
  }
  if (auto construct{mlir::dyn_cast<merk::ConstructOp>(op)}) {
    mlir::Value term{construct.getTerm()};
    if (!mlir::isa<merk::TermType>(term.getType())) {
      // A record by value holds no block.
      EscapeOperands(op);
      return;
    }
    if (noTag_.contains(mlir::cast<merk::TermType>(term.getType()).getName())) {
      // The same block as its field.
      mlir::Value field{construct.getArguments().front()};
      if (Tracked(field)) {
        Origin origin{OriginOf(field)};
        Track(term,
            origin == Origin::Static || origin == Origin::Pending ||
                    origin == Origin::Unknown
                ? origin
                : Origin::Alias,
            IdOf(field));
        Join(term, field);
      } else if (MayHold(field.getType())) {
        Track(term, Origin::Unknown);
      } else {
        Track(term, Origin::Static);
      }
      return;
    }
    Track(term,
        construct.getArguments().empty() ? Origin::Static : Origin::Alloc,
        construct.getArguments().empty() ? mlir::Value{} : term);
    for (mlir::Value argument : construct.getArguments()) {
      Join(term, argument);
    }
    return;
  }
  if (auto deconstruct{mlir::dyn_cast<merk::DeconstructOp>(op)}) {
    mlir::Value term{deconstruct.getTerm()};
    for (mlir::Value field : deconstruct.getFields()) {
      if (!MayHold(field.getType())) {
        continue;
      }
      auto type{mlir::dyn_cast<merk::TermType>(term.getType())};
      if (Tracked(term) && type && noTag_.contains(type.getName())) {
        // The term itself.
        Origin origin{OriginOf(term)};
        Track(field,
            origin == Origin::Alloc || origin == Origin::Arg ? Origin::Alias
                                                             : origin,
            IdOf(term));
        Join(field, term);
      } else if (Tracked(term) && type) {
        Track(field, Origin::Part);
        Join(field, term);
      } else {
        Track(field, Origin::Unknown);
      }
    }
    return;
  }
  if (mlir::isa<merk::HasFunctorOp, merk::CheckIndexOp, merk::ArraySizeOp,
          merk::ArrayWordsOp, merk::GuardOp, merk::ThrowIfOp,
          merk::StringCompareOp, merk::StringEqualOp, merk::YieldOp>(op)) {
    return;
  }
  if (auto get{mlir::dyn_cast<merk::ArrayGetOp>(op)}) {
    mlir::Value element{get->getResult(0)};
    if (MayHold(element.getType())) {
      Track(element, Origin::Part);
      Join(element, get->getOperand(0));
    }
    return;
  }
  if (mlir::isa<merk::ArrayNewOp, merk::ArrayCopyOp, merk::ArrayResizeOp>(op)) {
    // A new block, which holds the elements it is given; a copy's or a
    // resize's array is only read.
    mlir::Value array{op->getResult(0)};
    Track(array, Origin::Alloc, array);
    bool copies{!mlir::isa<merk::ArrayNewOp>(op)};
    for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
      if (!(copies && index == 0)) {
        Join(array, operand);
      }
    }
    return;
  }
  if (mlir::isa<merk::ArraySetOp, merk::ArrayFillOp>(op)) {
    mlir::Value array{op->getOperand(0)}, result{op->getResult(0)};
    if (!Tracked(array)) {
      Track(result, Origin::Unknown);
      return;
    }
    Track(result,
        OriginOf(array) == Origin::Static ? Origin::Static : Origin::Alias,
        IdOf(array));
    for (mlir::Value operand : op->getOperands()) {
      Join(result, operand);
    }
    return;
  }
  if (auto toWord{mlir::dyn_cast<merk::ToWordOp>(op)}) {
    mlir::Value value{toWord->getOperand(0)}, word{toWord->getResult(0)};
    if (Tracked(value)) {
      Origin origin{OriginOf(value)};
      Track(word,
          origin == Origin::Static || origin == Origin::Pending ? origin
                                                                : Origin::Alias,
          IdOf(value));
      Join(word, value);
    }
    return;
  }
  if (auto fromWord{mlir::dyn_cast<merk::FromWordOp>(op)}) {
    mlir::Value word{fromWord->getOperand(0)}, value{fromWord->getResult(0)};
    if (received_.contains(word)) {
      // The message's own blocks, if this is the only use of its word.
      if (IsHeap(value.getType()) && word.hasOneUse()) {
        Track(value, Origin::Alloc, value);
      } else if (MayHold(value.getType())) {
        Track(value, Origin::Unknown);
      }
      return;
    }
    if (Tracked(word)) {
      Origin origin{OriginOf(word)};
      // An integer of the word's bits holds the block too.
      Track(value,
          origin == Origin::Static || origin == Origin::Pending ? origin
                                                                : Origin::Alias,
          IdOf(word));
      Join(value, word);
    } else if (MayHold(value.getType())) {
      Track(value, Origin::Unknown);
    }
    return;
  }
  if (auto constant{mlir::dyn_cast<merk::ConstantOp>(op)}) {
    if (MayHold(constant.getResult().getType())) {
      Track(constant.getResult(), Origin::Static);
    }
    return;
  }
  if (auto call{mlir::dyn_cast<merk::CallOp>(op)}) {
    VisitCall(call);
    return;
  }
  if (auto call{mlir::dyn_cast<merk::ForeignCallOp>(op)}) {
    VisitForeignCall(call);
    return;
  }
  if (auto store{mlir::dyn_cast<mlir::LLVM::StoreOp>(op)}) {
    VisitStore(store);
    return;
  }
  if (auto succeed{mlir::dyn_cast<merk::SucceedOp>(op)}) {
    for (mlir::Value output : succeed.getOutputs()) {
      if (Tracked(output)) {
        returned_.push_back(output);
      }
    }
    return;
  }
  // Anything else keeps what it is given, as far as the pass knows: what it
  // yields from its regions too.
  for (mlir::Region &region : op->getRegions()) {
    for (mlir::Block &block : region) {
      VisitBlock(block);
      for (mlir::BlockArgument argument : block.getArguments()) {
        if (MayHold(argument.getType())) {
          Track(argument, Origin::Unknown);
        }
      }
      if (mlir::Operation * terminator{block.getTerminator()}) {
        EscapeOperands(terminator);
      }
    }
  }
  EscapeOperands(op);
  UnknownResults(op);
}

Summary Analysis::Summarize() {
  Summary summary;
  mlir::FunctionType type{proc_.getFunctionType()};
  summary.kept.assign(type.getNumInputs(), false);
  summary.results.resize(type.getNumResults());
  if (proc_.isExternal()) {
    return summary;
  }
  llvm::DenseSet<mlir::Value> escaping, unknown;
  for (mlir::Value value : escaping_) {
    escaping.insert(Find(value));
  }
  for (mlir::Value value : values_) {
    if (OriginOf(value) == Origin::Unknown) {
      unknown.insert(Find(value));
    }
  }
  mlir::Block &entry{proc_.getBody().front()};
  for (auto [index, argument] : llvm::enumerate(entry.getArguments())) {
    if (Tracked(argument)) {
      summary.kept[index] = escaping.contains(Find(argument));
    }
  }
  llvm::SmallVector<merk::SucceedOp> succeeds;
  proc_.walk([&](merk::SucceedOp op) { succeeds.push_back(op); });
  for (unsigned index{0}; index < type.getNumResults(); ++index) {
    ResultSummary &result{summary.results[index]};
    if (!MayHold(type.getResult(index))) {
      result.determined = true;
      continue;
    }
    if (succeeds.size() != 1) {
      // Not det code: nothing is known.
      result.determined = result.unknown = true;
      continue;
    }
    mlir::Value output{succeeds.front().getOutputs()[index]};
    if (!Tracked(output)) {
      result.determined = result.unknown = true;
      continue;
    }
    Origin origin{OriginOf(output)};
    if (origin == Origin::Pending) {
      continue;
    }
    result.determined = true;
    mlir::Value cls{Find(output)};
    if (unknown.contains(cls) || escaping.contains(cls)) {
      result.unknown = true;
      continue;
    }
    for (auto [arg, argument] : llvm::enumerate(entry.getArguments())) {
      if (Tracked(argument) && Find(argument) == cls) {
        result.args.push_back(arg);
      }
    }
    mlir::Value id{IdOf(output)};
    if (auto argument{mlir::dyn_cast_or_null<mlir::BlockArgument>(id)};
        argument && argument.getOwner() == &entry) {
      result.mustArg = static_cast<int>(argument.getArgNumber());
    }
    // A new block of its own, which no other output may be or hold.
    bool alone{true};
    for (auto [other, value] : llvm::enumerate(succeeds.front().getOutputs())) {
      if (other != index && Tracked(value) && Find(value) == cls) {
        alone = false;
      }
    }
    result.fresh = alone && id && OriginOf(id) == Origin::Alloc &&
        IsHeap(id.getType()) && !HiddenContained(id) && !yieldedInto_.count(id);
  }
  return summary;
}

// A new block that is stored in one cell and used nowhere else, which is
// freed through it.
bool Analysis::HiddenContained(mlir::Value value) const {
  if (OriginOf(value) != Origin::Alloc || IdOf(value) != value ||
      !value.hasOneUse()) {
    return false;
  }
  auto construct{mlir::dyn_cast<merk::ConstructOp>(*value.getUsers().begin())};
  if (!construct) {
    return false;
  }
  auto type{mlir::dyn_cast<merk::TermType>(construct.getTerm().getType())};
  return type && !noTag_.contains(type.getName());
}

Shape Analysis::ShapeOf(mlir::Value value) const {
  if (auto construct{value.getDefiningOp<merk::ConstructOp>()}) {
    Paths paths;
    bool hiddenTypes{llvm::any_of(construct.getArguments(), [](mlir::Value a) {
      return mlir::isa<merk::TypeInfoType, merk::TypeclassInfoType>(
          a.getType());
    })};
    if (!hiddenTypes) {
      Step step{construct.getFunctor(), 0,
          llvm::to_vector(construct.getArguments().getTypes())};
      for (auto [index, argument] : llvm::enumerate(construct.getArguments())) {
        if (!HiddenContained(argument)) {
          continue;
        }
        step.index = static_cast<unsigned>(index);
        for (const Path &inner : PathsOf(argument)) {
          Path path{step};
          path.append(inner.begin(), inner.end());
          paths.push_back(std::move(path));
        }
        paths.push_back(Path{step});
      }
    }
    Functors functors;
    functors.push_back({construct.getFunctor(), std::move(paths)});
    return functors;
  }
  if (!mlir::isa<merk::TermType>(value.getType())) {
    // An array or a string holds no block that is freed through it.
    return Functors{};
  }
  mlir::Operation *op{value.getDefiningOp()};
  if (!op || !mlir::isa<merk::IteOp, merk::SwitchOp>(op) ||
      OriginOf(value) != Origin::Alloc || IdOf(value) != value) {
    return std::nullopt;
  }
  // A branch's result: the union of what each part yields, as long as the
  // parts that may have the same functor free the same paths through it.
  unsigned index{mlir::cast<mlir::OpResult>(value).getResultNumber()};
  Functors shape;
  llvm::SmallVector<mlir::Region *> branches;
  if (auto ite{mlir::dyn_cast<merk::IteOp>(op)}) {
    branches = {&ite.getThenRegion(), &ite.getElseRegion()};
  } else {
    for (mlir::Region &region :
        mlir::cast<merk::SwitchOp>(op).getCaseRegions()) {
      branches.push_back(&region);
    }
  }
  for (mlir::Region *branch : branches) {
    bool other{false};
    merk::YieldOp yield{YieldOf(branch, other)};
    if (other) {
      return std::nullopt;
    }
    if (!yield) {
      continue;
    }
    mlir::Value yielded{yield.getValues()[index]};
    mlir::Value id{
        OriginOf(yielded) == Origin::Static ? yielded : IdOf(yielded)};
    if (!id) {
      return std::nullopt;
    }
    Shape part{ShapeOf(id)};
    if (!part) {
      return std::nullopt;
    }
    for (auto &[functor, paths] : *part) {
      auto same{llvm::find_if(
          shape, [&](const auto &entry) { return entry.first == functor; })};
      if (same == shape.end()) {
        shape.push_back({functor, paths});
      } else if (same->second != paths) {
        // Parts disagree on what a cell of this functor holds: nothing is
        // freed through it.
        same->second.clear();
      }
    }
  }
  return shape;
}

Paths Analysis::PathsOf(mlir::Value value) const {
  Paths paths;
  if (Shape shape{ShapeOf(value)}) {
    for (auto &[functor, functorPaths] : *shape) {
      paths.append(functorPaths.begin(), functorPaths.end());
    }
  }
  return paths;
}

void Analysis::NoteTouches(mlir::Operation *op) {
  llvm::DenseSet<mlir::Value> classes;
  for (mlir::Value value : op->getOperands()) {
    if (Tracked(value)) {
      classes.insert(Find(value));
    }
  }
  for (mlir::Value value : op->getResults()) {
    if (Tracked(value)) {
      classes.insert(Find(value));
    }
  }
  if (auto it{loopUses_.find(op)}; it != loopUses_.end()) {
    for (mlir::Value value : it->second) {
      classes.insert(Find(value));
    }
  }
  for (mlir::Region &region : op->getRegions()) {
    for (mlir::Block &block : region) {
      for (mlir::BlockArgument argument : block.getArguments()) {
        if (Tracked(argument)) {
          classes.insert(Find(argument));
        }
      }
      for (mlir::Operation &inner : block) {
        NoteTouches(&inner);
        const llvm::DenseSet<mlir::Value> &nested{touches_[&inner]};
        classes.insert(nested.begin(), nested.end());
      }
    }
  }
  touches_[op] = std::move(classes);
}

// The points where a class is freed in a region: after its last use there,
// or in each part of the if-then-else or switch that is.
void Analysis::Place(
    mlir::Value cls, mlir::Region &region, llvm::SmallVector<Point> &points) {
  mlir::Block &block{region.front()};
  mlir::Operation *last{nullptr};
  for (mlir::Operation &op : block.without_terminator()) {
    if (Touches(&op, cls)) {
      last = &op;
    }
  }
  if (!last) {
    points.push_back({&block, nullptr});
    return;
  }
  llvm::SmallVector<mlir::Region *> branches;
  if (auto ite{mlir::dyn_cast<merk::IteOp>(last)}) {
    branches = {&ite.getThenRegion(), &ite.getElseRegion()};
  } else if (auto switchOp{mlir::dyn_cast<merk::SwitchOp>(last)}) {
    for (mlir::Region &branch : switchOp.getCaseRegions()) {
      branches.push_back(&branch);
    }
  }
  bool yields{llvm::any_of(last->getResults(), [&](mlir::Value result) {
    return Tracked(result) && Find(result) == cls;
  })};
  if (!branches.empty() && !yields) {
    for (mlir::Region *branch : branches) {
      bool used{llvm::any_of(branch->front().without_terminator(),
          [&](mlir::Operation &op) { return Touches(&op, cls); })};
      if (used) {
        Place(cls, *branch, points);
      } else {
        points.push_back({&branch->front(), nullptr});
      }
    }
    return;
  }
  // Not between a tail call and its return.
  if (InTailPosition(last) && ContainsCall(last)) {
    return;
  }
  points.push_back({&block, last});
}

// Whether a value is defined on every path to the point, where it can be
// named.
bool Analysis::Visible(mlir::Value value, const Point &point) const {
  mlir::Block *block{point.block};
  mlir::Operation *before{point.after};
  bool inclusive{true};
  while (block) {
    if (auto argument{mlir::dyn_cast<mlir::BlockArgument>(value)}) {
      if (argument.getOwner() == block) {
        return true;
      }
    } else if (value.getDefiningOp()->getBlock() == block) {
      mlir::Operation *def{value.getDefiningOp()};
      if (!before) {
        return false;
      }
      return def == before ? inclusive : def->isBeforeInBlock(before);
    }
    before = block->getParentOp();
    inclusive = false;
    if (!before || mlir::isa<merk::ProcOp>(before)) {
      return false;
    }
    block = before->getBlock();
  }
  return false;
}

// Frees the block at the end of the path's steps from `step` on, from a
// cell that has the step's functor.
void Analysis::EmitFree(mlir::OpBuilder &builder, mlir::Location loc,
    mlir::Value value, const Path &path, unsigned step) {
  if (step == path.size()) {
    merk::FreeOp::create(builder, loc, value);
    return;
  }
  const Step &at{path[step]};
  auto ite{merk::IteOp::create(builder, loc, mlir::TypeRange{})};
  mlir::OpBuilder::InsertionGuard guard{builder};
  builder.setInsertionPointToStart(&ite.getCondition().emplaceBlock());
  mlir::Value has{merk::HasFunctorOp::create(builder, loc, value, at.functor)};
  merk::GuardOp::create(builder, loc, has);
  auto fields{
      merk::DeconstructOp::create(builder, loc, at.fields, value, at.functor)};
  merk::YieldOp::create(
      builder, loc, mlir::ValueRange{fields.getFields()[at.index]});
  mlir::Block &then{ite.getThenRegion().emplaceBlock()};
  mlir::Value field{then.addArgument(at.fields[at.index], loc)};
  builder.setInsertionPointToStart(&then);
  EmitFree(builder, loc, field, path, step + 1);
  merk::YieldOp::create(builder, loc, mlir::ValueRange{});
  builder.setInsertionPointToStart(&ite.getElseRegion().emplaceBlock());
  merk::YieldOp::create(builder, loc, mlir::ValueRange{});
}

void Analysis::InsertFrees() {
  if (proc_.isExternal()) {
    return;
  }
  // Classes that may be freed: nothing in them is an argument's, unknown,
  // pending, returned or kept.
  llvm::DenseSet<mlir::Value> kept;
  for (mlir::Value value : escaping_) {
    kept.insert(Find(value));
  }
  for (mlir::Value value : returned_) {
    kept.insert(Find(value));
  }
  for (mlir::Value value : values_) {
    Origin origin{OriginOf(value)};
    if (origin == Origin::Arg || origin == Origin::Unknown ||
        origin == Origin::Pending) {
      kept.insert(Find(value));
    }
  }
  // Each class's blocks that are freed by their own values, in order.
  llvm::SmallVector<mlir::Value> classes;
  llvm::DenseMap<mlir::Value, llvm::SmallVector<mlir::Value>> blocks;
  for (mlir::Value value : values_) {
    if (OriginOf(value) != Origin::Alloc || IdOf(value) != value ||
        !IsHeap(value.getType()) || HiddenContained(value) ||
        yieldedInto_.count(value)) {
      continue;
    }
    mlir::Value cls{Find(value)};
    if (kept.contains(cls)) {
      continue;
    }
    auto [it, added]{blocks.try_emplace(cls)};
    if (added) {
      classes.push_back(cls);
    }
    it->second.push_back(value);
  }
  if (classes.empty()) {
    return;
  }
  NoteTouches(proc_);
  mlir::OpBuilder builder{proc_.getContext()};
  for (mlir::Value cls : classes) {
    llvm::SmallVector<Point> points;
    Place(cls, proc_.getBody(), points);
    for (const Point &point : points) {
      if (point.after) {
        builder.setInsertionPointAfter(point.after);
      } else {
        builder.setInsertionPointToStart(point.block);
      }
      mlir::Location loc{point.after ? point.after->getLoc() : proc_.getLoc()};
      for (mlir::Value value : blocks[cls]) {
        if (!Visible(value, point)) {
          continue;
        }
        // The deepest blocks first, then the cells that hold them.
        Paths paths{PathsOf(value)};
        llvm::stable_sort(paths,
            [](const Path &a, const Path &b) { return a.size() > b.size(); });
        for (const Path &path : paths) {
          EmitFree(builder, loc, value, path, 0);
        }
        merk::FreeOp::create(builder, loc, value);
      }
    }
  }
}

} // namespace

void FreeAtLastUse(mlir::ModuleOp module) {
  llvm::SmallVector<merk::ProcOp> procs{
      llvm::to_vector(module.getOps<merk::ProcOp>())};
  llvm::StringSet<> noTag;
  for (auto def : module.getOps<merk::TypeDefOp>()) {
    auto constructors{def.getConstructors()};
    if (constructors.size() == 1 &&
        mlir::cast<merk::ConstructorAttr>(constructors[0]).getFields().size() ==
            1) {
      noTag.insert(def.getName());
    }
  }
  llvm::StringMap<Summary> summaries;
  for (merk::ProcOp proc : procs) {
    if (proc.isExternal()) {
      continue;
    }
    Summary &summary{summaries[proc.getSymName()]};
    summary.kept.assign(proc.getFunctionType().getNumInputs(), false);
    summary.results.resize(proc.getFunctionType().getNumResults());
  }
  // To a fixed point, from the guess that every output is pending and no
  // input kept; or, if it is not reached, knowing nothing.
  constexpr unsigned kMostRounds{64};
  bool changed{true};
  for (unsigned round{0}; changed && round < kMostRounds; ++round) {
    changed = false;
    for (merk::ProcOp proc : procs) {
      if (proc.isExternal()) {
        continue;
      }
      Analysis analysis{proc, summaries, noTag};
      analysis.Run();
      Summary summary{analysis.Summarize()};
      Summary &old{summaries[proc.getSymName()]};
      if (!(summary == old)) {
        old = std::move(summary);
        changed = true;
      }
    }
  }
  for (auto &entry : summaries) {
    Summary &summary{entry.second};
    for (ResultSummary &result : summary.results) {
      if (changed || !result.determined) {
        result = ResultSummary{};
        result.determined = result.unknown = true;
      }
    }
    if (changed) {
      summary.kept.assign(summary.kept.size(), true);
    }
  }
  for (merk::ProcOp proc : procs) {
    Analysis analysis{proc, summaries, noTag};
    analysis.Run();
    analysis.InsertFrees();
  }
}

} // namespace merkurius::codegen
