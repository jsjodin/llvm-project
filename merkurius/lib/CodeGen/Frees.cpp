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
// argument but an owned one, the result of a call the pass knows nothing
// of, or given to a foreign call that may keep it; one that is returned is
// kept on the paths that return it. A send copies its message, and a loop
// across the process's teams ends before the process goes on, so neither
// keeps what it is given.
//
// An input that the procedure takes by mode di (merk.clobbered) is owned
// when every call of the procedure, all of which the module holds, gives it
// a whole block of the caller's own: never a part of another, as an array
// that a received message holds is of the message's group. Its caller does
// not use it again, so the procedure takes it, always: it frees it as one
// of its own blocks on the paths that do not give it back, and its callers
// never do. A tracking pass that makes the next pass's array and gives it
// to a tail call of itself frees each. A result that the callee gives back
// in owned inputs' place, one of them or a new block, is the caller's,
// which frees it rather than the inputs; a procedure that gives it back
// itself gives it back in place of its own inputs that the callee took. A
// resize that clobbers its array (merk.clobbers, as array.resize's
// array_di) frees the old one right after it, when it is a whole block of
// the procedure's own.
//
// A send of an array or a term that is the last use of its class moves the
// message's arrays rather than copying them (merk.foreign_call's `moves`):
// when every array of the class that would be freed after it is one that
// the message holds, by its construction, the send is marked so and those
// arrays are not freed, here or through the cells that hold them. The
// cells are still copied and freed (docs/INPROGRESS.md, Messages between
// GPU processes on one GPU, step 2).
//
// What is freed is each block that the class allocated: a construction, an
// array's, a received message, which the runtime keeps as one group of
// blocks, and a call's result that the callee's summary says is a new
// block. Each is freed once on each path that made it, after its last use
// there, found by where values come from rather than by the class, as the
// class joins every path's values: a part, a construction or an update
// holds what its operands hold, a branch's result what its yields hold, a
// call's result the inputs its callee's summary names. A path that gives
// the block on, as the procedure's result or a branch's that a later
// operation uses, does not free it. Nor does one where its last use is
// between a tail call that may come back to the procedure and its return,
// where the process's next state or a loop's next round is called: it is
// left to the process's end, as are the blocks of a branch that the code
// after it cannot name. A tail call of a procedure that does not call back
// is no loop, and costs one frame when its caller frees after it. A block
// that is stored in one cell and used nowhere else, as a received message
// is in its `message(...)`, or that is a branch's result alone, is freed
// through what holds it: the field is taken where the functor is the one
// it was stored with, and freed before its cell; so is one that a callee's
// new result holds, as the callee's summary says.
//
// The summaries of the module's procedures, for their callers, are found by
// iterating to a fixed point: for each output, which inputs it may be or
// hold, whether it is always the same block as one input, whether it is a
// new block, and whether it is a whole one; for each input, whether the
// procedure keeps it, and whether it takes it, as an owned input. Owned
// inputs depend on the summaries, and the summaries on them: both are
// found again until no more inputs stop being owned.
//
//===----------------------------------------------------------------------===//

#include "merkurius/CodeGen/Lowering.h"
#include "merkurius/CodeGen/MLIRGen.h"
#include "merkurius/Dialect/Merk/IR/MerkDialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
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
  // A whole block, never a part of one: a new or a static one, or one of
  // `classArgs` as given, when those are whole blocks. Known even when what
  // it holds is not, as an array's elements may be parts of a message.
  bool whole{true};
  // The inputs in its class, which it may be or hold, even when it is
  // unknown.
  llvm::SmallVector<unsigned> classArgs;
  // For a new block, the blocks it holds that are freed through it, as a
  // received message is through its message(...) cell, if known.
  Shape shape;

  // Nothing known: it may be or hold any of the `inputs`.
  void MakeUnknown(unsigned inputs) {
    determined = unknown = true;
    whole = false;
    classArgs.clear();
    for (unsigned input{0}; input < inputs; ++input) {
      classArgs.push_back(input);
    }
  }

  bool operator==(const ResultSummary &other) const {
    return determined == other.determined && unknown == other.unknown &&
        fresh == other.fresh && mustArg == other.mustArg &&
        args == other.args && whole == other.whole &&
        classArgs == other.classArgs && shape == other.shape;
  }
};

struct Summary {
  llvm::SmallVector<ResultSummary, 0> results;
  // Whether the procedure keeps the input past its return, other than in its
  // results.
  llvm::SmallVector<bool> kept;
  // Whether the procedure takes the input, an owned argument (Owned) that no
  // result may be or hold: it frees it, or gives it to what does.
  llvm::SmallVector<bool> consumed;

  bool operator==(const Summary &other) const {
    return results == other.results && kept == other.kept &&
        consumed == other.consumed;
  }
};

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

// Whether the cast is of an array to the pointer that only sends of it as
// the whole message are given.
bool SendsArray(mlir::UnrealizedConversionCastOp cast) {
  if (cast->getNumOperands() != 1 || cast->getNumResults() != 1 ||
      !mlir::isa<merk::ArrayType>(cast->getOperand(0).getType())) {
    return false;
  }
  return llvm::all_of(cast->getResult(0).getUsers(), [](mlir::Operation *user) {
    auto call{mlir::dyn_cast<merk::ForeignCallOp>(user)};
    return call && call.getCallee() == "merk_device_send_array";
  });
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

// The strongly connected components of the module's call graph, by the
// names of its procedures: a call whose callee is in its caller's component
// may come back to the caller.
using Components = llvm::StringMap<unsigned>;

// Tarjan's algorithm over the procedures' merk.call edges.
class ComponentFinder {
public:
  explicit ComponentFinder(llvm::ArrayRef<merk::ProcOp> procs) {
    for (merk::ProcOp proc : procs) {
      llvm::SmallVector<llvm::StringRef> &callees{callees_[proc.getSymName()]};
      if (!proc.isExternal()) {
        proc.walk(
            [&](merk::CallOp call) { callees.push_back(call.getCallee()); });
      }
    }
  }

  Components Find() {
    for (auto &entry : callees_) {
      if (!index_.contains(entry.getKey())) {
        Visit(entry.getKey());
      }
    }
    return std::move(components_);
  }

private:
  void Visit(llvm::StringRef name) {
    unsigned index{next_++};
    index_[name] = index;
    low_[name] = index;
    stack_.push_back(name);
    onStack_.insert(name);
    auto it{callees_.find(name)};
    if (it != callees_.end()) {
      for (llvm::StringRef callee : it->second) {
        if (!callees_.contains(callee)) {
          continue;
        }
        if (!index_.contains(callee)) {
          Visit(callee);
          low_[name] = std::min(low_[name], low_[callee]);
        } else if (onStack_.contains(callee)) {
          low_[name] = std::min(low_[name], index_[callee]);
        }
      }
    }
    if (low_[name] != index) {
      return;
    }
    unsigned component{count_++};
    llvm::StringRef member;
    do {
      member = stack_.pop_back_val();
      onStack_.erase(member);
      components_[member] = component;
    } while (member != name);
  }

  llvm::StringMap<llvm::SmallVector<llvm::StringRef>> callees_;
  llvm::StringMap<unsigned> index_;
  llvm::StringMap<unsigned> low_;
  llvm::SmallVector<llvm::StringRef> stack_;
  llvm::StringSet<> onStack_;
  Components components_;
  unsigned next_{0};
  unsigned count_{0};
};

// Whether `op` holds a call that may come back to the procedure `caller`.
bool ContainsCallBack(
    mlir::Operation *op, llvm::StringRef caller, const Components &components) {
  auto own{components.find(caller)};
  return op
      ->walk([&](merk::CallOp call) {
        auto callee{components.find(call.getCallee())};
        if (own == components.end() || callee == components.end() ||
            callee->second == own->second) {
          return mlir::WalkResult::interrupt();
        }
        return mlir::WalkResult::advance();
      })
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

// For each procedure, by name, which of its inputs are owned: clobbered, as
// of mode di (merk.clobbered), so that its caller does not use them again,
// and given a block of the caller's own by each of its calls, which the
// module holds all of. The procedure frees such an input as one of its own
// blocks, and its callers do not.
using Owned = llvm::StringMap<llvm::SmallVector<bool>>;

// One procedure's classes, and where they are freed.
class Analysis {
public:
  Analysis(merk::ProcOp proc, const llvm::StringMap<Summary> &summaries,
      const llvm::StringSet<> &noTag, const Owned &owned)
      : proc_{proc}, summaries_{summaries}, noTag_{noTag}, owned_{owned} {}

  void Run();
  Summary Summarize();
  void InsertFrees(const Components &);
  bool IsOwned(unsigned index) const;
  // Whether the value is always a whole block, never a part of one: owned
  // arguments are, and with `anyArgument` every argument as given.
  bool Whole(mlir::Value, bool anyArgument) const;
  // The resizes that clobber a whole block of the procedure's own, which is
  // freed after them unless its class frees it.
  llvm::SmallVector<merk::ArrayResizeOp> FreedResizes() const;
  // Whole(value, false), collecting the blocks of the procedure's own that
  // the value may be.
  bool WholeBlocks(mlir::Value, llvm::SmallVectorImpl<mlir::Value> &) const;
  // Whether a call that takes a value of the class may come before the
  // point, on its path: the class's blocks are not the procedure's there.
  bool ConsumedBefore(mlir::Value cls, const Point &point);
  // Whether a resize that frees the block may come before the point.
  bool ResizedBefore(mlir::Value block, const Point &point) const;
  // Whether one of the operations may come before the point, on its path.
  static bool AnyBefore(
      const llvm::SmallPtrSetImpl<mlir::Operation *> &, const Point &);
  // An argument of the procedure that it owns.
  bool IsOwnedArgument(mlir::Value, mlir::Block &entry) const;
  // Clears each callee's owned input that one of the procedure's calls gives
  // neither a block of the procedure's own nor one of its owned arguments;
  // whether it cleared any.
  bool CheckOwnedInputs(Owned &) const;

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
  // For each result of the call, the owned input of its callee that it
  // gives back in its place, or -1: a whole value whose blocks, which may
  // be several, as a branch's, are the procedure's own.
  llvm::SmallVector<llvm::SmallVector<unsigned, 2>> HandOver(
      merk::CallOp, const Summary &) const;
  void VisitStore(mlir::LLVM::StoreOp);

  bool HiddenContained(mlir::Value) const;
  Shape ShapeOf(mlir::Value) const;
  Paths PathsOf(mlir::Value) const;

  class Holds;
  // Where a class's blocks are freed: after their last use on each path,
  // where what the path gives on cannot hold them, so that a class that is
  // returned on some paths is kept only on those. `given` is which of the
  // region's yielded values its op's caller gives on, or null for all, as
  // the procedure's own return does.
  void PlaceAlongPaths(Holds &, mlir::Region &, llvm::SmallVector<Point> &,
      const llvm::SmallBitVector *given);
  bool Visible(mlir::Value, const Point &) const;
  void EmitFree(mlir::OpBuilder &, mlir::Location, mlir::Value, const Path &,
      unsigned step);
  merk::ForeignCallOp SendAt(const Point &, mlir::Value cls);
  llvm::DenseSet<mlir::Value> HeldBy(mlir::Value message) const;
  bool Moves(merk::ForeignCallOp send, mlir::Value cls,
      llvm::ArrayRef<mlir::Value> freed) const;

  merk::ProcOp proc_;
  const llvm::StringMap<Summary> &summaries_;
  // The types whose one constructor has one field, which is the term itself
  // (Lowering.cpp, Layouts, TermLayout::Kind::NoTag): no cell.
  const llvm::StringSet<> &noTag_;
  const Owned &owned_;
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
  // The calls that take a value, an owned input of their callee.
  llvm::SmallVector<std::pair<mlir::Operation *, mlir::Value>> consumed_;
  // The blocks that clobbering resizes free, by the resize.
  llvm::SmallVector<std::pair<mlir::Operation *, mlir::Value>> resizedBlocks_;
  // The call graph's components, while frees are inserted.
  const Components *components_{nullptr};
  // Those results: blocks of the procedure's own, but not new ones for its
  // callers, as they may be the blocks given.
  llvm::DenseSet<mlir::Value> successors_;
  // Each such result, with the input it replaces.
  llvm::SmallVector<std::pair<mlir::Value, mlir::Value>> replaces_;
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
  llvm::SmallVector<llvm::SmallVector<unsigned, 2>> successors{
      HandOver(call, summary)};
  for (auto [index, input] : llvm::enumerate(call.getInputs())) {
    // What the callee keeps or takes is the callee's from then on, but for
    // what it gives back as a result's.
    if (llvm::any_of(successors, [index](const auto &args) {
          return llvm::is_contained(args, static_cast<unsigned>(index));
        })) {
      continue;
    }
    if (index < summary.kept.size() && summary.kept[index]) {
      Escape(input);
    } else if (index < summary.consumed.size() && summary.consumed[index] &&
        Tracked(input)) {
      // Given away here, on this path only: the class is not freed after
      // the call, but may be on the procedure's other paths.
      consumed_.push_back({call, input});
    }
  }
  for (auto [index, output] : llvm::enumerate(call.getOutputs())) {
    if (!MayHold(output.getType())) {
      continue;
    }
    if (!successors[index].empty()) {
      // The inputs' blocks are the callee's, which gives back this one in
      // their place: one of them, or a new one after freeing them. The
      // inputs are taken here, on this path, and dead after the call, so
      // the result starts a class of its own: what becomes of it, as a
      // later call's taking it, is not the inputs' on the procedure's other
      // paths.
      for (unsigned arg : successors[index]) {
        consumed_.push_back({call, call.getInputs()[arg]});
        replaces_.push_back({output, call.getInputs()[arg]});
      }
      successors_.insert(output);
      Track(output, Origin::Alloc, output);
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

llvm::SmallVector<llvm::SmallVector<unsigned, 2>> Analysis::HandOver(
    merk::CallOp call, const Summary &summary) const {
  llvm::SmallVector<llvm::SmallVector<unsigned, 2>> successors(
      call.getNumResults());
  auto owned{owned_.find(call.getCallee())};
  if (owned == owned_.end()) {
    return successors;
  }
  for (auto [index, output] : llvm::enumerate(call.getOutputs())) {
    const ResultSummary &result{summary.results[index]};
    // What the result holds may be unknown, as its elements may be parts of
    // messages, but not which block it is: one of the owned inputs it may
    // be, as a procedure that swaps two arrays gives back either.
    if (!IsHeap(output.getType()) || !result.determined || !result.whole ||
        result.classArgs.empty()) {
      continue;
    }
    bool given{llvm::all_of(result.classArgs, [&](unsigned arg) {
      llvm::SmallVector<mlir::Value> blocks;
      return arg < owned->second.size() && owned->second[arg] &&
          WholeBlocks(call.getInputs()[arg], blocks) &&
          arg < summary.kept.size() && !summary.kept[arg];
    })};
    if (!given ||
        (result.mustArg >= 0 &&
            !llvm::is_contained(
                result.classArgs, static_cast<unsigned>(result.mustArg))) ||
        llvm::any_of(result.args, [&](unsigned arg) {
          return !llvm::is_contained(result.classArgs, arg);
        })) {
      continue;
    }
    // No other result may hold them.
    bool alone{llvm::none_of(llvm::enumerate(summary.results), [&](auto other) {
      return other.index() != index &&
          (!other.value().determined ||
              llvm::any_of(other.value().classArgs, [&](unsigned arg) {
                return llvm::is_contained(result.classArgs, arg);
              }));
    })};
    if (alone) {
      successors[index].assign(
          result.classArgs.begin(), result.classArgs.end());
    }
  }
  return successors;
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
  if (auto cast{mlir::dyn_cast<mlir::UnrealizedConversionCastOp>(op)};
      cast && SendsArray(cast)) {
    // The pointer that a send of an array as the whole message is given,
    // the array's block (MLIRGenOpenMP.cpp, SendInTarget): the send, which
    // copies it, is a use of the array.
    mlir::Value array{cast->getOperand(0)}, pointer{cast->getResult(0)};
    if (Tracked(array)) {
      Origin origin{OriginOf(array)};
      Track(pointer,
          origin == Origin::Static || origin == Origin::Pending ? origin
                                                                : Origin::Alias,
          IdOf(array));
      Join(pointer, array);
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
  summary.consumed.assign(type.getNumInputs(), false);
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
      result.MakeUnknown(type.getNumInputs());
      continue;
    }
    mlir::Value output{succeeds.front().getOutputs()[index]};
    if (!Tracked(output)) {
      result.MakeUnknown(type.getNumInputs());
      continue;
    }
    Origin origin{OriginOf(output)};
    if (origin == Origin::Pending) {
      continue;
    }
    result.determined = true;
    mlir::Value cls{Find(output)};
    result.whole = Whole(output, /*anyArgument=*/true);
    // The classes that the output's stands for: its own, and those of the
    // inputs that a callee gave back a block of it in place of. A caller
    // hands over to it what it gave in their place.
    llvm::DenseSet<mlir::Value> behind{cls};
    for (bool grew{true}; grew;) {
      grew = false;
      for (auto [successor, input] : replaces_) {
        if (behind.contains(Find(successor)) && Tracked(input)) {
          grew |= behind.insert(Find(input)).second;
        }
      }
    }
    bool replacesArgs{false};
    for (auto [arg, argument] : llvm::enumerate(entry.getArguments())) {
      if (Tracked(argument) && behind.contains(Find(argument))) {
        result.classArgs.push_back(arg);
        replacesArgs |= Find(argument) != cls;
      }
    }
    if (unknown.contains(cls) || escaping.contains(cls)) {
      result.unknown = true;
      continue;
    }
    for (auto [arg, argument] : llvm::enumerate(entry.getArguments())) {
      if (Tracked(argument) && behind.contains(Find(argument))) {
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
    // A block given back in place of the procedure's own new blocks, and
    // none of its inputs', is a new one for its callers.
    result.fresh = alone && id && OriginOf(id) == Origin::Alloc &&
        IsHeap(id.getType()) && !HiddenContained(id) &&
        !yieldedInto_.count(id) && !(successors_.contains(id) && replacesArgs);
    if (result.fresh) {
      result.shape = ShapeOf(id);
    }
  }
  // An owned input is always taken: the procedure frees it on the paths
  // that do not give it back, and on those that do, its caller has it as
  // the result's (HandOver), never as the input's, which the procedure may
  // have freed on another path or after a resize.
  for (unsigned index{0}; index < type.getNumInputs(); ++index) {
    summary.consumed[index] = IsOwned(index);
  }
  return summary;
}

bool Analysis::IsOwned(unsigned index) const {
  merk::ProcOp proc{proc_};
  auto it{owned_.find(proc.getSymName())};
  return it != owned_.end() && index < it->second.size() && it->second[index];
}

bool Analysis::Whole(mlir::Value value, bool anyArgument) const {
  if (!Tracked(value)) {
    return false;
  }
  // A call's result, whatever it holds, as its callee's summary says.
  if (auto call{value.getDefiningOp<merk::CallOp>()}) {
    unsigned index{mlir::cast<mlir::OpResult>(value).getResultNumber()};
    auto it{summaries_.find(call.getCallee())};
    if (it == summaries_.end() || index >= it->second.results.size()) {
      return false;
    }
    // One that the iteration has not found yet is taken as whole, as
    // recursion through it only gives what its other paths give: the
    // greatest fixed point. After the iteration every result is found, or
    // unknown.
    const ResultSummary &result{it->second.results[index]};
    if (!result.determined) {
      return true;
    }
    return result.whole && llvm::all_of(result.classArgs, [&](unsigned arg) {
      return Whole(call.getInputs()[arg], anyArgument);
    });
  }
  mlir::Operation *op{value.getDefiningOp()};
  // An update in place gives the same block.
  if (mlir::isa_and_present<merk::ArraySetOp, merk::ArrayFillOp>(op)) {
    return Whole(op->getOperand(0), anyArgument);
  }
  llvm::SmallVector<mlir::Region *> branches;
  if (auto ite{mlir::dyn_cast_or_null<merk::IteOp>(op)}) {
    branches = {&ite.getThenRegion(), &ite.getElseRegion()};
  } else if (auto switchOp{mlir::dyn_cast_or_null<merk::SwitchOp>(op)}) {
    for (mlir::Region &branch : switchOp.getCaseRegions()) {
      branches.push_back(&branch);
    }
  }
  if (!branches.empty()) {
    // Each part's, but those that fail.
    unsigned index{mlir::cast<mlir::OpResult>(value).getResultNumber()};
    return llvm::all_of(branches, [&](mlir::Region *branch) {
      bool other{false};
      merk::YieldOp yield{YieldOf(branch, other)};
      return !other && (!yield || Whole(yield.getValues()[index], anyArgument));
    });
  }
  switch (OriginOf(value)) {
  case Origin::Alloc:
  case Origin::Static:
    return true;
  case Origin::Arg: {
    auto argument{mlir::cast<mlir::BlockArgument>(value)};
    return anyArgument || IsOwned(argument.getArgNumber());
  }
  case Origin::Alias: {
    mlir::Value id{IdOf(value)};
    return id && id != value && Whole(id, anyArgument);
  }
  default:
    return false;
  }
}

llvm::SmallVector<merk::ArrayResizeOp> Analysis::FreedResizes() const {
  llvm::SmallVector<merk::ArrayResizeOp> resizes;
  merk::ProcOp proc{proc_};
  proc.walk([&](merk::ArrayResizeOp resize) {
    mlir::Value array{resize.getArray()};
    if (resize->hasAttr("merk.clobbers") && Tracked(array) &&
        Whole(array, /*anyArgument=*/false)) {
      resizes.push_back(resize);
    }
  });
  return resizes;
}

bool Analysis::WholeBlocks(
    mlir::Value value, llvm::SmallVectorImpl<mlir::Value> &blocks) const {
  if (!Tracked(value)) {
    return false;
  }
  // A block of the procedure's own that the value may be.
  if ((OriginOf(value) == Origin::Alloc && IdOf(value) == value) ||
      IsOwnedArgument(value, merk::ProcOp{proc_}.getBody().front())) {
    blocks.push_back(value);
  }
  if (auto call{value.getDefiningOp<merk::CallOp>()}) {
    unsigned index{mlir::cast<mlir::OpResult>(value).getResultNumber()};
    auto it{summaries_.find(call.getCallee())};
    if (it == summaries_.end() || index >= it->second.results.size()) {
      return false;
    }
    const ResultSummary &result{it->second.results[index]};
    if (!result.determined) {
      return true;
    }
    return result.whole && llvm::all_of(result.classArgs, [&](unsigned arg) {
      return WholeBlocks(call.getInputs()[arg], blocks);
    });
  }
  mlir::Operation *op{value.getDefiningOp()};
  if (mlir::isa_and_present<merk::ArraySetOp, merk::ArrayFillOp>(op)) {
    return WholeBlocks(op->getOperand(0), blocks);
  }
  llvm::SmallVector<mlir::Region *> branches;
  if (auto ite{mlir::dyn_cast_or_null<merk::IteOp>(op)}) {
    branches = {&ite.getThenRegion(), &ite.getElseRegion()};
  } else if (auto switchOp{mlir::dyn_cast_or_null<merk::SwitchOp>(op)}) {
    for (mlir::Region &branch : switchOp.getCaseRegions()) {
      branches.push_back(&branch);
    }
  }
  if (!branches.empty()) {
    unsigned index{mlir::cast<mlir::OpResult>(value).getResultNumber()};
    return llvm::all_of(branches, [&](mlir::Region *branch) {
      bool other{false};
      merk::YieldOp yield{YieldOf(branch, other)};
      return !other &&
          (!yield || WholeBlocks(yield.getValues()[index], blocks));
    });
  }
  switch (OriginOf(value)) {
  case Origin::Alloc:
  case Origin::Static:
    return true;
  case Origin::Arg:
    return IsOwnedArgument(value, merk::ProcOp{proc_}.getBody().front());
  case Origin::Alias: {
    mlir::Value id{IdOf(value)};
    return id && id != value && WholeBlocks(id, blocks);
  }
  default:
    return false;
  }
}

bool Analysis::ConsumedBefore(mlir::Value cls, const Point &point) {
  llvm::SmallPtrSet<mlir::Operation *, 8> takers;
  for (auto [call, value] : consumed_) {
    if (Find(value) == cls) {
      takers.insert(call);
    }
  }
  return AnyBefore(takers, point);
}

bool Analysis::ResizedBefore(mlir::Value block, const Point &point) const {
  llvm::SmallPtrSet<mlir::Operation *, 8> resizes;
  for (auto [resize, old] : resizedBlocks_) {
    if (old == block) {
      resizes.insert(resize);
    }
  }
  return AnyBefore(resizes, point);
}

bool Analysis::AnyBefore(
    const llvm::SmallPtrSetImpl<mlir::Operation *> &ops, const Point &point) {
  if (ops.empty()) {
    return false;
  }
  // The operations before the point, in its block and in those that hold it,
  // with what they hold: one in an earlier branch may have been on the
  // path.
  mlir::Block *block{point.block};
  mlir::Operation *last{point.after};
  while (block) {
    if (last) {
      for (mlir::Operation *op{last}; op; op = op->getPrevNode()) {
        bool takes{false};
        op->walk([&](mlir::Operation *inner) {
          if (ops.contains(inner)) {
            takes = true;
          }
        });
        if (takes) {
          return true;
        }
      }
    }
    mlir::Operation *parent{block->getParentOp()};
    if (!parent || mlir::isa<merk::ProcOp>(parent)) {
      break;
    }
    block = parent->getBlock();
    last = parent->getPrevNode();
  }
  return false;
}

bool Analysis::IsOwnedArgument(mlir::Value value, mlir::Block &entry) const {
  auto argument{mlir::dyn_cast<mlir::BlockArgument>(value)};
  return argument && argument.getOwner() == &entry &&
      IsOwned(argument.getArgNumber());
}

bool Analysis::CheckOwnedInputs(Owned &owned) const {
  bool changed{false};
  merk::ProcOp proc{proc_};
  proc.walk([&](merk::CallOp call) {
    auto it{owned.find(call.getCallee())};
    if (it == owned.end()) {
      return;
    }
    for (auto [index, input] : llvm::enumerate(call.getInputs())) {
      if (index >= it->second.size() || !it->second[index]) {
        continue;
      }
      // A whole block of the procedure's own, not a part of another, as an
      // array that a received message holds is of its message's group:
      // freeing it would free the group. Uniqueness has made sure that
      // nothing else of the procedure's that is still used holds it.
      if (!Whole(input, /*anyArgument=*/false)) {
        it->second[index] = false;
        changed = true;
      }
    }
  });
  return changed;
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
  if (auto call{value.getDefiningOp<merk::CallOp>()};
      call && OriginOf(value) == Origin::Alloc && IdOf(value) == value) {
    // A new block of the callee's, which says what it holds.
    auto it{summaries_.find(call.getCallee())};
    unsigned index{mlir::cast<mlir::OpResult>(value).getResultNumber()};
    if (it == summaries_.end() || index >= it->second.results.size() ||
        !it->second.results[index].fresh) {
      return std::nullopt;
    }
    return it->second.results[index].shape;
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

// Whether values may be or hold some of a class's blocks, by where they
// come from, along the paths that the class joins: a part, a construction
// or an update holds what its operands hold; a branch's result what its
// yields do; a call's result what the callee's summary says of its inputs.
// An argument of the procedure was made before its blocks were, and a new
// block is none of them; nor is a number or a record by value, but an
// integer of a block's word that the class has. An array's element is the
// class's if the array is, as an update in place changes the array that
// other values name too. Anything else may.
class Analysis::Holds {
public:
  Holds(Analysis &analysis, mlir::Value cls, llvm::ArrayRef<mlir::Value> blocks)
      : analysis_{analysis}, cls_{cls}, blocks_{blocks.begin(), blocks.end()} {}

  bool Value(mlir::Value value) {
    if (blocks_.contains(value)) {
      return true;
    }
    if (mlir::isa<mlir::IntegerType, mlir::FloatType, mlir::IndexType,
            merk::RecordType>(value.getType())) {
      return analysis_.Tracked(value) && analysis_.Find(value) == cls_;
    }
    auto [it, added]{memo_.try_emplace(value, true)};
    if (!added) {
      return it->second;
    }
    bool holds{Compute(value)};
    memo_[value] = holds;
    return holds;
  }

  bool Is(mlir::Value value) const { return blocks_.contains(value); }

  // Whether the operation uses a value that may be or hold the blocks, or is
  // one, or holds such a use. A loop's call uses the words stored for it.
  bool Uses(mlir::Operation *op) {
    return op
        ->walk([&](mlir::Operation *inner) {
          bool uses{llvm::any_of(inner->getOperands(),
              [&](mlir::Value operand) { return Value(operand); })};
          if (auto it{analysis_.loopUses_.find(inner)};
              it != analysis_.loopUses_.end()) {
            uses |= llvm::any_of(
                it->second, [&](mlir::Value stored) { return Value(stored); });
          }
          bool defines{llvm::any_of(inner->getResults(),
              [&](mlir::Value result) { return blocks_.contains(result); })};
          return uses || defines ? mlir::WalkResult::interrupt()
                                 : mlir::WalkResult::advance();
        })
        .wasInterrupted();
  }

private:
  bool Compute(mlir::Value value) {
    if (auto argument{mlir::dyn_cast<mlir::BlockArgument>(value)}) {
      mlir::Operation *parent{argument.getOwner()->getParentOp()};
      if (mlir::isa<merk::ProcOp>(parent)) {
        return false;
      }
      auto ite{mlir::dyn_cast<merk::IteOp>(parent)};
      if (!ite || argument.getParentRegion() != &ite.getThenRegion()) {
        return true;
      }
      bool other{false};
      merk::YieldOp yield{YieldOf(&ite.getCondition(), other)};
      return other || !yield ||
          Value(yield.getValues()[argument.getArgNumber()]);
    }
    mlir::Operation *op{value.getDefiningOp()};
    unsigned index{mlir::cast<mlir::OpResult>(value).getResultNumber()};
    if (auto get{mlir::dyn_cast<merk::ArrayGetOp>(op)}) {
      mlir::Value array{get->getOperand(0)};
      return !analysis_.Tracked(array) || analysis_.Find(array) == cls_;
    }
    if (auto call{mlir::dyn_cast<merk::CallOp>(op)}) {
      auto it{analysis_.summaries_.find(call.getCallee())};
      if (it == analysis_.summaries_.end() ||
          index >= it->second.results.size()) {
        return true;
      }
      const ResultSummary &result{it->second.results[index]};
      if (!result.determined) {
        return true;
      }
      return llvm::any_of(result.classArgs,
          [&](unsigned arg) { return Value(call.getInputs()[arg]); });
    }
    llvm::SmallVector<mlir::Region *> branches;
    if (auto ite{mlir::dyn_cast<merk::IteOp>(op)}) {
      branches = {&ite.getThenRegion(), &ite.getElseRegion()};
    } else if (auto switchOp{mlir::dyn_cast<merk::SwitchOp>(op)}) {
      for (mlir::Region &branch : switchOp.getCaseRegions()) {
        branches.push_back(&branch);
      }
    }
    if (!branches.empty()) {
      return llvm::any_of(branches, [&](mlir::Region *branch) {
        bool other{false};
        merk::YieldOp yield{YieldOf(branch, other)};
        return other || (yield && Value(yield.getValues()[index]));
      });
    }
    // Calls that the summaries do not cover, and operations with regions,
    // may give back anything.
    if (op->getNumRegions() != 0 ||
        mlir::isa<merk::ForeignCallOp, merk::CallClosureOp, merk::CallMethodOp>(
            op)) {
      return true;
    }
    return llvm::any_of(
        op->getOperands(), [&](mlir::Value operand) { return Value(operand); });
  }

  Analysis &analysis_;
  mlir::Value cls_;
  llvm::DenseSet<mlir::Value> blocks_;
  llvm::DenseMap<mlir::Value, bool> memo_;
};

void Analysis::PlaceAlongPaths(Holds &holds, mlir::Region &region,
    llvm::SmallVector<Point> &points, const llvm::SmallBitVector *passed) {
  mlir::Block &block{region.front()};
  mlir::Operation *last{nullptr};
  for (mlir::Operation &op : block.without_terminator()) {
    if (holds.Uses(&op)) {
      last = &op;
    }
  }
  // What the region gives on that may hold the blocks: a branch's yielded
  // value whose result nothing uses gives nothing.
  mlir::Operation *terminator{block.getTerminator()};
  llvm::SmallVector<mlir::Value> given;
  if (terminator) {
    for (auto [index, operand] : llvm::enumerate(terminator->getOperands())) {
      if ((!passed || (index < passed->size() && passed->test(index))) &&
          holds.Value(operand)) {
        given.push_back(operand);
      }
    }
  }
  if (!last) {
    if (given.empty()) {
      points.push_back({&block, nullptr});
    }
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
  bool fromLast{llvm::all_of(
      given, [&](mlir::Value value) { return value.getDefiningOp() == last; })};
  if (!given.empty()) {
    // Given on: the branches of `last` that do not give the blocks free
    // them, if all that is given comes from it.
    if (!fromLast) {
      return;
    }
    llvm::SmallBitVector results(last->getNumResults());
    for (mlir::Value value : given) {
      results.set(mlir::cast<mlir::OpResult>(value).getResultNumber());
    }
    for (mlir::Region *branch : branches) {
      PlaceAlongPaths(holds, *branch, points, &results);
    }
    return;
  }
  // Nothing after `last` uses what it gives that may hold the blocks, so
  // its branches free them, unless it gives one of them, a branch's block
  // that only its result names.
  bool gives{llvm::any_of(last->getResults(),
      [&](mlir::Value result) { return holds.Is(result); })};
  if (!branches.empty() && !gives) {
    llvm::SmallBitVector none(last->getNumResults());
    for (mlir::Region *branch : branches) {
      PlaceAlongPaths(holds, *branch, points, &none);
    }
    return;
  }
  if (InTailPosition(last) &&
      ContainsCallBack(last, merk::ProcOp{proc_}.getSymName(), *components_)) {
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

// Whether a path ends in an array, which a move gives the receiver.
bool EndsInArray(const Path &path) {
  const Step &last{path.back()};
  return mlir::isa<merk::ArrayType>(last.fields[last.index]);
}

// The send of an array or a term after which the class is freed, whose
// message is in the class, if the point is one.
merk::ForeignCallOp Analysis::SendAt(const Point &point, mlir::Value cls) {
  auto send{mlir::dyn_cast_or_null<merk::ForeignCallOp>(point.after)};
  if (!send ||
      (send.getCallee() != "merk_device_send_array" &&
          send.getCallee() != "merk_device_send_term") ||
      send.getInputs().size() < 3) {
    return {};
  }
  mlir::Value message{send.getInputs()[2]};
  if (!Tracked(message) || Find(message) != cls) {
    return {};
  }
  return send;
}

// The blocks that the message holds for certain, by their values: its own,
// and those its cells were constructed with, through their fields.
llvm::DenseSet<mlir::Value> Analysis::HeldBy(mlir::Value message) const {
  llvm::DenseSet<mlir::Value> held;
  llvm::SmallVector<mlir::Value> work;
  if (mlir::Value id{IdOf(message)}) {
    work.push_back(id);
  }
  while (!work.empty()) {
    mlir::Value block{work.pop_back_val()};
    if (!held.insert(block).second) {
      continue;
    }
    auto construct{block.getDefiningOp<merk::ConstructOp>()};
    if (!construct) {
      continue;
    }
    for (mlir::Value field : construct.getArguments()) {
      if (mlir::Value id{Tracked(field) ? IdOf(field) : mlir::Value{}}) {
        work.push_back(id);
      }
    }
  }
  return held;
}

// Whether the send moves its message's arrays: every array of the class
// that is freed after it, by its own value or through a cell, is one that
// the message holds, and there is one. The send copies an array that is
// not a block of its own, and gives the receiver one that is, so none is
// freed twice and none is left.
bool Analysis::Moves(merk::ForeignCallOp send, mlir::Value cls,
    llvm::ArrayRef<mlir::Value> freed) const {
  llvm::DenseSet<mlir::Value> held{HeldBy(send.getInputs()[2])};
  bool any{false};
  for (mlir::Value value : freed) {
    bool holds{held.contains(value)};
    if (mlir::isa<merk::ArrayType>(value.getType())) {
      if (!holds) {
        return false;
      }
      any = true;
    }
    for (const Path &path : PathsOf(value)) {
      if (EndsInArray(path)) {
        if (!holds) {
          return false;
        }
        any = true;
      }
    }
  }
  return any;
}

void Analysis::InsertFrees(const Components &components) {
  if (proc_.isExternal()) {
    return;
  }
  components_ = &components;
  // Classes that may be freed: nothing in them is an argument's but an
  // owned one, unknown, pending, returned or kept.
  mlir::Block &entry{proc_.getBody().front()};
  // A class that is only returned is kept on the paths that return its
  // blocks: on the others, they are freed after their last use there.
  llvm::DenseSet<mlir::Value> kept;
  llvm::DenseSet<mlir::Value> returned;
  for (mlir::Value value : escaping_) {
    kept.insert(Find(value));
  }
  for (mlir::Value value : returned_) {
    returned.insert(Find(value));
  }
  for (mlir::Value value : values_) {
    Origin origin{OriginOf(value)};
    if ((origin == Origin::Arg && !IsOwnedArgument(value, entry)) ||
        origin == Origin::Unknown || origin == Origin::Pending) {
      kept.insert(Find(value));
    }
  }
  // Each class's blocks that are freed by their own values, in order.
  llvm::SmallVector<mlir::Value> classes;
  llvm::DenseMap<mlir::Value, llvm::SmallVector<mlir::Value>> blocks;
  llvm::DenseSet<mlir::Value> partly;
  for (mlir::Value value : values_) {
    bool block{
        OriginOf(value) == Origin::Alloc || IsOwnedArgument(value, entry)};
    if (!block || IdOf(value) != value || !IsHeap(value.getType()) ||
        HiddenContained(value) || yieldedInto_.count(value)) {
      continue;
    }
    mlir::Value cls{Find(value)};
    if (kept.contains(cls)) {
      continue;
    }
    if (returned.contains(cls)) {
      partly.insert(cls);
    }
    auto [it, added]{blocks.try_emplace(cls)};
    if (added) {
      classes.push_back(cls);
    }
    it->second.push_back(value);
  }
  // A resize that clobbers its array leaves the old one dead: when it is a
  // whole block of the procedure's own, it is freed after the resize, and
  // its class does not free it again on the paths through the resize.
  for (merk::ArrayResizeOp resize : FreedResizes()) {
    llvm::SmallVector<mlir::Value> old;
    WholeBlocks(resize.getArray(), old);
    for (mlir::Value block : old) {
      resizedBlocks_.push_back({resize, block});
    }
    mlir::OpBuilder builder{resize->getContext()};
    builder.setInsertionPointAfter(resize);
    merk::FreeOp::create(builder, resize.getLoc(), resize.getArray());
  }
  if (classes.empty()) {
    return;
  }
  mlir::OpBuilder builder{proc_.getContext()};
  llvm::SmallBitVector none(proc_.getFunctionType().getNumResults());
  for (mlir::Value cls : classes) {
    // Each block where it is dead on each path, by where it comes from; the
    // blocks that are dead at the same point are freed there together.
    llvm::SmallVector<std::pair<Point, llvm::SmallVector<mlir::Value>>> groups;
    for (mlir::Value value : blocks[cls]) {
      llvm::SmallVector<Point> points;
      Holds holds{*this, cls, value};
      PlaceAlongPaths(holds, proc_.getBody(), points,
          partly.contains(cls) ? nullptr : &none);
      for (const Point &point : points) {
        auto group{llvm::find_if(groups, [&](const auto &entry) {
          return entry.first.block == point.block &&
              entry.first.after == point.after;
        })};
        if (group == groups.end()) {
          groups.push_back({point, {value}});
        } else {
          group->second.push_back(value);
        }
      }
    }
    for (const auto &[point, dead] : groups) {
      if (ConsumedBefore(cls, point)) {
        continue;
      }
      if (point.after) {
        builder.setInsertionPointAfter(point.after);
      } else {
        builder.setInsertionPointToStart(point.block);
      }
      mlir::Location loc{point.after ? point.after->getLoc() : proc_.getLoc()};
      llvm::SmallVector<mlir::Value> freed;
      for (mlir::Value value : dead) {
        if (Visible(value, point) && !ResizedBefore(value, point)) {
          freed.push_back(value);
        }
      }
      bool moves{false};
      if (merk::ForeignCallOp send{SendAt(point, cls)};
          send && Moves(send, cls, freed)) {
        send.setMoves(true);
        moves = true;
      }
      for (mlir::Value value : freed) {
        if (moves && mlir::isa<merk::ArrayType>(value.getType())) {
          continue;
        }
        // The deepest blocks first, then the cells that hold them.
        Paths paths{PathsOf(value)};
        llvm::stable_sort(paths,
            [](const Path &a, const Path &b) { return a.size() > b.size(); });
        for (const Path &path : paths) {
          if (!(moves && EndsInArray(path))) {
            EmitFree(builder, loc, value, path, 0);
          }
        }
        merk::FreeOp::create(builder, loc, value);
      }
    }
  }
}

// The inputs that may be owned: of a heap type and clobbered, of a
// procedure that the module calls, only by merk.call, so that the module
// holds all of its calls. An exported procedure's are all here too: a
// device's compile copies what it calls of other modules (CopyDeviceLibrary)
// and calls nothing it has not compiled (Lowerer::CheckDeviceCalls). One
// that the module never calls, or a process's root, may be called by the
// runtime, which gives it no block of its own.
Owned OwnedCandidates(
    mlir::ModuleOp module, llvm::ArrayRef<merk::ProcOp> procs) {
  Owned owned;
  for (merk::ProcOp proc : procs) {
    if (proc.isExternal() || proc->hasAttr(kProcessRootAttrName)) {
      continue;
    }
    std::optional<mlir::SymbolTable::UseRange> uses{
        mlir::SymbolTable::getSymbolUses(proc, module)};
    if (!uses || uses->empty() ||
        llvm::any_of(*uses, [](const mlir::SymbolTable::SymbolUse &use) {
          return !mlir::isa<merk::CallOp>(use.getUser());
        })) {
      continue;
    }
    mlir::FunctionType type{proc.getFunctionType()};
    llvm::SmallVector<bool> inputs(type.getNumInputs(), false);
    bool any{false};
    for (unsigned index{0}; index < type.getNumInputs(); ++index) {
      inputs[index] = IsHeap(type.getInput(index)) &&
          proc.getArgAttr(index, "merk.clobbered");
      any |= inputs[index];
    }
    if (any) {
      owned[proc.getSymName()] = std::move(inputs);
    }
  }
  return owned;
}

} // namespace

// The summaries of the module's procedures, to a fixed point, from the
// guess that every output is pending and no input kept, with `owned` the
// owned inputs; whether the fixed point was reached. If it was not, nothing
// is known: every result is unknown and every input kept.
bool SummarizeAll(llvm::ArrayRef<merk::ProcOp> procs,
    const llvm::StringSet<> &noTag, const Owned &owned,
    llvm::StringMap<Summary> &summaries) {
  summaries.clear();
  for (merk::ProcOp proc : procs) {
    if (proc.isExternal()) {
      continue;
    }
    Summary &summary{summaries[proc.getSymName()]};
    summary.kept.assign(proc.getFunctionType().getNumInputs(), false);
    summary.consumed.assign(proc.getFunctionType().getNumInputs(), false);
    summary.results.resize(proc.getFunctionType().getNumResults());
  }
  constexpr unsigned kMostRounds{64};
  bool changed{true};
  for (unsigned round{0}; changed && round < kMostRounds; ++round) {
    changed = false;
    for (merk::ProcOp proc : procs) {
      if (proc.isExternal()) {
        continue;
      }
      Analysis analysis{proc, summaries, noTag, owned};
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
        result.MakeUnknown(summary.kept.size());
      }
    }
    if (changed) {
      summary.kept.assign(summary.kept.size(), true);
    }
  }
  return !changed;
}

// Clears the owned inputs that some call does not give a whole block of its
// caller's own, to a fixed point, as an owned argument passed on is one only
// while it stays owned; whether it cleared any.
bool ClearOwnedInputs(llvm::ArrayRef<merk::ProcOp> procs,
    const llvm::StringSet<> &noTag, const llvm::StringMap<Summary> &summaries,
    Owned &owned) {
  bool any{false};
  for (bool cleared{true}; cleared;) {
    cleared = false;
    for (merk::ProcOp proc : procs) {
      if (proc.isExternal()) {
        continue;
      }
      Analysis analysis{proc, summaries, noTag, owned};
      analysis.Run();
      cleared |= analysis.CheckOwnedInputs(owned);
    }
    any |= cleared;
  }
  return any;
}

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
  // The summaries depend on which inputs are owned, and which are owned on
  // the summaries; owned inputs only become fewer, from the candidates.
  Owned owned{OwnedCandidates(module, procs)};
  llvm::StringMap<Summary> summaries;
  while (true) {
    if (!SummarizeAll(procs, noTag, owned, summaries)) {
      // Knowing nothing, every input is kept, and none is owned.
      owned.clear();
      break;
    }
    if (!ClearOwnedInputs(procs, noTag, summaries, owned)) {
      break;
    }
  }
  Components components{ComponentFinder{procs}.Find()};
  for (merk::ProcOp proc : procs) {
    Analysis analysis{proc, summaries, noTag, owned};
    analysis.Run();
    analysis.InsertFrees(components);
  }
}

} // namespace merkurius::codegen
