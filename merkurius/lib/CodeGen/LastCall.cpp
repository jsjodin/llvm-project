//===-- lib/CodeGen/LastCall.cpp - Last call modulo constructor -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Makes a recursive call whose results go into cells that the procedure
// returns a tail call (design document, Lowering, "Deterministic and semidet
// code", "Last call modulo constructor"). The procedure makes the cells
// first, with holes where the call's results go, and calls a variant of the
// callee that stores those outputs in the holes instead of returning them. In the variant the same
// call becomes a tail call, so that append, map and the like run in
// constant stack.
//
// A variant `<proc>.lcmc` takes the procedure's inputs and then the
// addresses of the holes for its outputs that go into cells, and returns its
// other outputs, such as an accumulator's final value. A callee gets one
// when a call of it is the last call modulo constructor of a procedure in
// its strongly connected component of the call graph; its outputs that go
// into cells at such calls are the ones it stores.
//
//===----------------------------------------------------------------------===//

#include "merkurius/CodeGen/Lowering.h"
#include "merkurius/Dialect/Merk/IR/MerkDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"

namespace merkurius::codegen {
namespace {

std::string VariantName(llvm::StringRef proc) {
  return (proc + ".lcmc").str();
}

// Whether a procedure can have a variant: a det or semidet one with a body
// and outputs.
bool Eligible(merk::ProcOp proc) {
  merk::Determinism determinism{proc.getDeterminism()};
  return !proc.isExternal() &&
      (determinism == merk::Determinism::Det ||
          determinism == merk::Determinism::Semidet) &&
      !proc.getResultTypes().empty();
}

// The regions of an if-then-else or a switch whose yields are its results.
llvm::SmallVector<mlir::Region *> ResultRegions(mlir::Operation *op) {
  llvm::SmallVector<mlir::Region *> regions;
  if (auto ite{mlir::dyn_cast<merk::IteOp>(op)}) {
    regions.push_back(&ite.getThenRegion());
    regions.push_back(&ite.getElseRegion());
  } else {
    for (mlir::Region &region : op->getRegions()) {
      regions.push_back(&region);
    }
  }
  return regions;
}

// The yield that ends a region, if it succeeds.
merk::YieldOp YieldOf(mlir::Region *region) {
  if (region->empty() || region->front().empty()) {
    return nullptr;
  }
  return mlir::dyn_cast<merk::YieldOp>(region->front().back());
}

// An if-then-else or switch like `op` without its results `drop`, whose
// regions it takes; their yields no longer have those values.
void DropResults(mlir::Operation *op, llvm::ArrayRef<unsigned> drop) {
  llvm::SmallVector<mlir::Type> types;
  for (auto [i, type] : llvm::enumerate(op->getResultTypes())) {
    if (!llvm::is_contained(drop, i)) {
      types.push_back(type);
    }
  }
  mlir::OpBuilder builder{op};
  mlir::Operation *created;
  if (mlir::isa<merk::IteOp>(op)) {
    created = merk::IteOp::create(builder, op->getLoc(), types);
  } else {
    auto switchOp{mlir::cast<merk::SwitchOp>(op)};
    created = merk::SwitchOp::create(builder, op->getLoc(), types,
        switchOp.getValue(), switchOp.getCases(), switchOp.getCanFail());
  }
  created->setDiscardableAttrs(op->getDiscardableAttrDictionary());
  for (unsigned i = 0; i < op->getNumRegions(); ++i) {
    created->getRegion(i).takeBody(op->getRegion(i));
  }
  for (unsigned i = 0, j = 0; i < op->getNumResults(); ++i) {
    if (!llvm::is_contained(drop, i)) {
      op->getResult(i).replaceAllUsesWith(created->getResult(j++));
    }
  }
  op->erase();
}

// Erases the operands of `op` at `positions`.
void EraseOperands(mlir::Operation *op, llvm::ArrayRef<unsigned> positions) {
  llvm::SmallVector<unsigned> sorted{positions};
  llvm::sort(sorted, std::greater<unsigned>{});
  for (unsigned position : sorted) {
    op->eraseOperand(position);
  }
}

// Whether everything between `call` and `end` but `except` is free of
// effects, so that the call can go last.
bool Movable(merk::CallOp call, mlir::Operation *end,
    llvm::ArrayRef<mlir::Operation *> except) {
  for (mlir::Operation *op{call->getNextNode()}; op != end;
      op = op->getNextNode()) {
    if (!llvm::is_contained(except, op) && !mlir::isMemoryEffectFree(op)) {
      return false;
    }
  }
  return true;
}

// Where a procedure's outputs come from: operands `positions` of `end`, a
// merk.succeed or the yield of a branch.
struct Leaf {
  mlir::Operation *end;
  llvm::SmallVector<unsigned> positions;

  mlir::Value Output(unsigned k) const {
    return end->getOperand(positions[k]);
  }
};

// What becomes of a result of a call: an argument of a construction that
// makes output `k`, or output `k` itself.
struct Use {
  enum class Kind { Construct, Direct } kind;
  unsigned k;
  merk::ConstructOp construct;
  unsigned argument{0};
};

// A call each of whose results becomes part of an output, which can be
// made last.
struct Site {
  merk::CallOp call;
  llvm::SmallVector<Use> uses;
};

class LastCall {
public:
  LastCall(mlir::ModuleOp module, InCell inCell)
      : module_{module}, inCell_{inCell} {}
  void Run();

private:
  using Refs = llvm::SmallVector<unsigned>;
  // The positions of a procedure's outputs, and where they come from.
  void Leaves(const Leaf &, llvm::SmallVectorImpl<Leaf> &) const;
  mlir::Operation *Branching(const Leaf &) const;
  std::optional<Use> UseOf(mlir::OpOperand &, const Leaf &) const;
  std::optional<Site> SiteOf(const Leaf &,
      llvm::function_ref<bool(const Site &)> accept) const;
  bool Valid(const Site &, const Refs *stored) const;
  bool Reaches(llvm::StringRef from, llvm::StringRef to) const;
  void Rewrite(Site, const Leaf &, const Refs *stored,
      llvm::ArrayRef<mlir::Value> refs);
  void Store(const Leaf &, const Refs &stored, llvm::ArrayRef<mlir::Value> refs);
  void Variant(merk::ProcOp, const Refs &stored);
  bool Need(llvm::StringRef caller, merk::ProcOp);

  mlir::ModuleOp module_;
  InCell inCell_;
  llvm::StringMap<merk::ProcOp> eligible_;
  llvm::StringMap<llvm::StringSet<>> callees_;
  // The procedures that have variants, and which of their outputs the
  // variants store through references, in order.
  llvm::StringMap<Refs> stored_;
};

// The if-then-else or switch whose results are all of the leaf's outputs,
// distinct, which nothing else uses, if there is one.
mlir::Operation *LastCall::Branching(const Leaf &leaf) const {
  mlir::Operation *branching{nullptr};
  for (unsigned k = 0; k < leaf.positions.size(); ++k) {
    mlir::Value value{leaf.Output(k)};
    mlir::Operation *def{value.getDefiningOp()};
    if (!def || !mlir::isa<merk::IteOp, merk::SwitchOp>(def) ||
        def->getBlock() != leaf.end->getBlock() || !value.hasOneUse() ||
        (branching && def != branching)) {
      return nullptr;
    }
    branching = def;
  }
  return branching;
}

// The leaves where the outputs at `leaf` are made: through the branches of
// if-then-elses and switches that make them all.
void LastCall::Leaves(const Leaf &leaf, llvm::SmallVectorImpl<Leaf> &leaves)
    const {
  mlir::Operation *branching{Branching(leaf)};
  if (!branching) {
    leaves.push_back(leaf);
    return;
  }
  for (mlir::Region *region : ResultRegions(branching)) {
    if (merk::YieldOp yield{YieldOf(region)}) {
      Leaf inner{yield, {}};
      for (unsigned k = 0; k < leaf.positions.size(); ++k) {
        inner.positions.push_back(
            mlir::cast<mlir::OpResult>(leaf.Output(k)).getResultNumber());
      }
      Leaves(inner, leaves);
    }
  }
}

// What becomes of a call's result, through the use `use`: an argument of a
// construction of a functor in a cell that makes an output, and nothing
// else, or an output.
std::optional<Use> LastCall::UseOf(mlir::OpOperand &use, const Leaf &leaf)
    const {
  mlir::Operation *user{use.getOwner()};
  if (user == leaf.end) {
    for (auto [k, position] : llvm::enumerate(leaf.positions)) {
      if (position == use.getOperandNumber()) {
        return Use{Use::Kind::Direct, static_cast<unsigned>(k), nullptr};
      }
    }
    return std::nullopt;
  }
  auto construct{mlir::dyn_cast<merk::ConstructOp>(user)};
  if (!construct || construct->getBlock() != leaf.end->getBlock() ||
      !construct.getTerm().hasOneUse() ||
      !mlir::isa<merk::TermType>(use.get().getType()) ||
      !inCell_(construct.getTerm().getType(), construct.getFunctor())) {
    return std::nullopt;
  }
  mlir::OpOperand &output{*construct.getTerm().use_begin()};
  if (output.getOwner() != leaf.end) {
    return std::nullopt;
  }
  for (auto [k, position] : llvm::enumerate(leaf.positions)) {
    if (position == output.getOperandNumber()) {
      return Use{Use::Kind::Construct, static_cast<unsigned>(k), construct,
          use.getOperandNumber()};
    }
  }
  return std::nullopt;
}

// A call at the leaf that `accept`s, each of whose results has one use,
// that makes an output, and no two of them in one construction.
std::optional<Site> LastCall::SiteOf(const Leaf &leaf,
    llvm::function_ref<bool(const Site &)> accept) const {
  // The calls that make outputs or arguments of constructions that do.
  llvm::SmallVector<mlir::Value> values;
  for (unsigned k = 0; k < leaf.positions.size(); ++k) {
    mlir::Value value{leaf.Output(k)};
    values.push_back(value);
    if (auto construct{value.getDefiningOp<merk::ConstructOp>()}) {
      values.append(
          construct.getArguments().begin(), construct.getArguments().end());
    }
  }
  llvm::SmallVector<merk::CallOp> candidates;
  for (mlir::Value value : values) {
    auto call{value.getDefiningOp<merk::CallOp>()};
    if (call && call->getBlock() == leaf.end->getBlock() &&
        !llvm::is_contained(candidates, call)) {
      candidates.push_back(call);
    }
  }
  for (merk::CallOp call : candidates) {
    Site site{call, {}};
    llvm::SmallVector<mlir::Operation *> constructs;
    bool ok{true};
    for (mlir::Value result : call->getResults()) {
      std::optional<Use> use;
      if (result.hasOneUse()) {
        use = UseOf(*result.use_begin(), leaf);
      }
      if (!use ||
          (use->construct && llvm::is_contained(constructs, use->construct))) {
        ok = false;
        break;
      }
      if (use->construct) {
        constructs.push_back(use->construct);
      }
      site.uses.push_back(*use);
    }
    if (ok && Movable(call, leaf.end, constructs) && accept(site)) {
      return site;
    }
  }
  return std::nullopt;
}

// Whether a site can call its callee's variant: the results that go into
// cells are those the variant stores, and the others become outputs that
// the procedure returns. In a variant, `stored` are the outputs it stores:
// a result that becomes one of them is stored by the callee's variant
// through the same reference. A site whose results all become outputs is
// worth it only in a variant.
bool LastCall::Valid(const Site &site, const Refs *stored) const {
  merk::CallOp call{site.call};
  auto it{stored_.find(call.getCallee())};
  if (it == stored_.end()) {
    return false;
  }
  const Refs &callee{it->second};
  bool construct{false};
  for (auto [i, use] : llvm::enumerate(site.uses)) {
    bool byRef{llvm::is_contained(callee, i)};
    bool outputByRef{stored && llvm::is_contained(*stored, use.k)};
    if (use.kind == Use::Kind::Construct) {
      construct = true;
      if (!byRef) {
        return false;
      }
    } else if (byRef != outputByRef) {
      return false;
    }
  }
  return construct || stored;
}

// Whether `from` calls `to`, directly or through other procedures.
bool LastCall::Reaches(llvm::StringRef from, llvm::StringRef to) const {
  llvm::StringSet<> visited;
  llvm::SmallVector<llvm::StringRef> work{from};
  while (!work.empty()) {
    llvm::StringRef proc{work.pop_back_val()};
    auto it{callees_.find(proc)};
    if (it == callees_.end()) {
      continue;
    }
    for (const auto &callee : it->second) {
      if (callee.getKey() == to) {
        return true;
      }
      if (visited.insert(callee.getKey()).second) {
        work.push_back(callee.getKey());
      }
    }
  }
  return false;
}

// Makes the cells before the call, with holes for the call's results, and
// calls the callee's variant last to fill them. In a variant, which stores
// its outputs `stored` through `refs`, those outputs are stored first.
void LastCall::Rewrite(Site site, const Leaf &leaf, const Refs *stored,
    llvm::ArrayRef<mlir::Value> refs) {
  mlir::Location loc{site.call.getLoc()};
  mlir::OpBuilder builder{leaf.end};
  const Refs &callee{stored_.find(site.call.getCallee())->second};
  // The variant's stored outputs that the call does not make.
  if (stored) {
    for (auto [r, k] : llvm::enumerate(*stored)) {
      bool made{llvm::any_of(
          site.uses, [&, k = k](const Use &use) { return use.k == k; })};
      if (!made) {
        merk::RefStoreOp::create(builder, loc, leaf.Output(k), refs[r]);
      }
    }
  }
  llvm::SmallVector<mlir::Value> inputs{site.call.getInputs()};
  llvm::SmallVector<mlir::Type> results;
  for (auto [i, use] : llvm::enumerate(site.uses)) {
    mlir::Value result{site.call->getResult(i)};
    if (!llvm::is_contained(callee, i)) {
      results.push_back(result.getType());
      continue;
    }
    if (use.kind == Use::Kind::Direct) {
      inputs.push_back(refs[llvm::find(*stored, use.k) - stored->begin()]);
      continue;
    }
    mlir::Value hole{merk::HoleOp::create(builder, loc, result.getType())};
    use.construct->setOperand(use.argument, hole);
    use.construct->moveBefore(leaf.end);
    if (stored && llvm::is_contained(*stored, use.k)) {
      merk::RefStoreOp::create(builder, loc, use.construct.getTerm(),
          refs[llvm::find(*stored, use.k) - stored->begin()]);
    }
    inputs.push_back(merk::FieldRefOp::create(builder, loc,
        use.construct.getTerm(), use.construct.getFunctor(), use.argument));
  }
  auto call{merk::CallOp::create(builder, loc, results,
      VariantName(site.call.getCallee()), inputs)};
  for (unsigned i = 0, j = 0; i < site.uses.size(); ++i) {
    if (!llvm::is_contained(callee, i)) {
      site.call->getResult(i).replaceAllUsesWith(call->getResult(j++));
    }
  }
  if (stored) {
    llvm::SmallVector<unsigned> positions;
    for (unsigned k : *stored) {
      positions.push_back(leaf.positions[k]);
    }
    EraseOperands(leaf.end, positions);
  }
  site.call.erase();
}

// In a variant, which stores its outputs `stored` through `refs`: stores
// them where they are made, and leaves them out of what is returned. A call
// of a procedure with a variant that makes them calls the variant last
// instead.
void LastCall::Store(const Leaf &leaf, const Refs &stored,
    llvm::ArrayRef<mlir::Value> refs) {
  llvm::SmallVector<unsigned> positions;
  for (unsigned k : stored) {
    positions.push_back(leaf.positions[k]);
  }
  if (mlir::Operation *branching{Branching(leaf)}) {
    llvm::SmallVector<unsigned> results;
    for (unsigned k = 0; k < leaf.positions.size(); ++k) {
      results.push_back(
          mlir::cast<mlir::OpResult>(leaf.Output(k)).getResultNumber());
    }
    for (mlir::Region *region : ResultRegions(branching)) {
      if (merk::YieldOp yield{YieldOf(region)}) {
        Store(Leaf{yield, results}, stored, refs);
      }
    }
    llvm::SmallVector<unsigned> drop;
    for (unsigned k : stored) {
      drop.push_back(results[k]);
    }
    EraseOperands(leaf.end, positions);
    DropResults(branching, drop);
    return;
  }
  if (std::optional<Site> site{SiteOf(leaf,
          [&](const Site &site) { return Valid(site, &stored); })}) {
    Rewrite(*site, leaf, &stored, refs);
    return;
  }
  mlir::OpBuilder builder{leaf.end};
  for (auto [r, k] : llvm::enumerate(stored)) {
    merk::RefStoreOp::create(builder, leaf.end->getLoc(), leaf.Output(k), refs[r]);
  }
  EraseOperands(leaf.end, positions);
}

// The merk.succeeds that end a procedure's body, with its `outputs` outputs.
llvm::SmallVector<Leaf> Ends(merk::ProcOp proc, std::size_t outputs) {
  llvm::SmallVector<Leaf> ends;
  proc.walk([&](merk::SucceedOp succeed) {
    if (succeed->getParentOp() == proc &&
        succeed.getOutputs().size() == outputs) {
      Leaf leaf{succeed, {}};
      for (unsigned k = 0; k < outputs; ++k) {
        leaf.positions.push_back(k);
      }
      ends.push_back(leaf);
    }
  });
  return ends;
}

// The variant of a procedure, a copy of it that takes the addresses of
// holes for its outputs `stored` after its inputs, stores those outputs
// there and returns the others.
void LastCall::Variant(merk::ProcOp proc, const Refs &stored) {
  mlir::MLIRContext *context{proc.getContext()};
  auto variant{mlir::cast<merk::ProcOp>(proc->clone())};
  variant.setSymName(VariantName(proc.getSymName()));
  mlir::SymbolTable::setSymbolVisibility(
      variant, mlir::SymbolTable::Visibility::Private);
  llvm::SmallVector<mlir::Type> inputs{proc.getArgumentTypes()};
  llvm::SmallVector<mlir::Type> results;
  for (auto [k, type] : llvm::enumerate(proc.getResultTypes())) {
    if (!llvm::is_contained(stored, k)) {
      results.push_back(type);
    }
  }
  auto refType{merk::RefType::get(context)};
  inputs.append(stored.size(), refType);
  variant.setFunctionType(mlir::FunctionType::get(context, inputs, results));
  if (mlir::ArrayAttr argAttrs{variant.getArgAttrsAttr()}) {
    llvm::SmallVector<mlir::Attribute> attrs{argAttrs.begin(), argAttrs.end()};
    attrs.append(stored.size(), mlir::DictionaryAttr::get(context));
    variant.setArgAttrsAttr(mlir::ArrayAttr::get(context, attrs));
  }
  variant.removeResAttrsAttr();
  llvm::SmallVector<mlir::Value> refs;
  for (std::size_t r = 0; r < stored.size(); ++r) {
    refs.push_back(
        variant.getBody().front().addArgument(refType, proc.getLoc()));
  }
  module_.getBody()->getOperations().insert(
      std::next(proc->getIterator()), variant);
  for (const Leaf &end : Ends(variant, proc.getResultTypes().size())) {
    Store(end, stored, refs);
  }
}

// Finds the callees that need variants for a procedure: those of its last
// calls modulo constructor in its strongly connected component, whose
// results that go into cells their variants store; and, if the procedure
// has a variant, those of its last calls that make outputs its variant
// stores, which the callees' variants store too, so that those calls are
// tail calls in the variant. Whether that found any more.
bool LastCall::Need(llvm::StringRef caller, merk::ProcOp proc) {
  Refs own;
  if (auto it{stored_.find(caller)}; it != stored_.end()) {
    own = it->second;
  }
  bool more{false};
  llvm::SmallVector<Leaf> leaves;
  for (const Leaf &end : Ends(proc, proc.getResultTypes().size())) {
    Leaves(end, leaves);
  }
  for (const Leaf &leaf : leaves) {
    std::optional<Site> site{SiteOf(leaf, [&](const Site &site) {
      merk::CallOp call{site.call};
      llvm::StringRef callee{call.getCallee()};
      return eligible_.contains(callee) && Reaches(callee, caller) &&
          llvm::any_of(site.uses, [&](const Use &use) {
            return use.kind == Use::Kind::Construct ||
                llvm::is_contained(own, use.k);
          });
    })};
    if (!site) {
      continue;
    }
    Refs &stored{stored_[site->call.getCallee()]};
    for (auto [i, use] : llvm::enumerate(site->uses)) {
      if ((use.kind == Use::Kind::Construct ||
              llvm::is_contained(own, use.k)) &&
          !llvm::is_contained(stored, i)) {
        stored.push_back(i);
        more = true;
      }
    }
    llvm::sort(stored);
  }
  return more;
}

void LastCall::Run() {
  for (auto proc : module_.getOps<merk::ProcOp>()) {
    if (proc.isExternal()) {
      continue;
    }
    if (Eligible(proc)) {
      eligible_[proc.getSymName()] = proc;
    }
    llvm::StringSet<> &callees{callees_[proc.getSymName()]};
    proc.walk([&](merk::CallOp call) { callees.insert(call.getCallee()); });
  }
  // The callees that need variants, and which of their outputs the variants
  // store, until no more do.
  for (bool more{true}; more;) {
    more = false;
    for (auto &entry : eligible_) {
      more = Need(entry.getKey(), entry.getValue()) || more;
    }
  }
  if (stored_.empty()) {
    return;
  }
  // Each variant is made from its procedure before that changes.
  llvm::SmallVector<merk::ProcOp> originals;
  for (auto &entry : eligible_) {
    originals.push_back(entry.getValue());
  }
  for (auto &entry : stored_) {
    Variant(eligible_.lookup(entry.getKey()), entry.getValue());
  }
  for (merk::ProcOp proc : originals) {
    llvm::SmallVector<Leaf> leaves;
    for (const Leaf &end : Ends(proc, proc.getResultTypes().size())) {
      Leaves(end, leaves);
    }
    for (const Leaf &leaf : leaves) {
      if (std::optional<Site> site{SiteOf(leaf, [&](const Site &site) {
            return Valid(site, nullptr);
          })}) {
        Rewrite(*site, leaf, nullptr, {});
      }
    }
  }
}

} // namespace

void LastCallModuloConstructor(mlir::ModuleOp module, InCell inCell) {
  LastCall{module, inCell}.Run();
}

} // namespace merkurius::codegen
