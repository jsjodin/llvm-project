//===-- lib/CodeGen/Specializer.cpp -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The specializer for known closures (docs/INPROGRESS.md, The specializer:
// known closures and dictionaries).
//
// A closure is known where a merk.closure makes it, and in a clone made for
// it. A call that passes known closures to a procedure with a body calls a
// clone of it that takes, after each such closure, the values the closure
// captured; in the clone, merk.closure_word of the closure is that value, and
// merk.call_closure of it is a call of its entry, which is cloned in turn.
// A clone keeps the closure itself as a parameter, so that a use that still
// needs it, such as putting it in a term, has it without making it again.
// Once nothing is left to clone, a clone's parameters that nothing uses are
// removed from it and its calls, and then the closures that nothing uses.

#include "merkurius/CodeGen/Specializer.h"
#include "merkurius/Dialect/Merk/IR/MerkDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringMap.h"
#include <optional>
#include <string>
#include <vector>

namespace merkurius::codegen {
namespace {

// How deep a known closure's captured values are followed: a closure that
// captured a known closure is two levels. A deeper one is passed as a value,
// as mmc's --higher-order-arg-limit has it.
constexpr unsigned kMaxDepth{3};
// The clones one procedure gets, after which its calls stay as they are, so
// that patterns that grow at each call, as in polymorphic recursion, end.
constexpr unsigned kMaxClones{16};

// A known closure: its entry, what it says of each captured value (none on
// a device), and those values; in a clone, also the type_info parameter of
// each captured word that has one.
struct Known {
  mlir::FlatSymbolRefAttr entry;
  mlir::ArrayAttr captures;
  llvm::SmallVector<mlir::Value> captured;
  llvm::SmallVector<mlir::Value> infos;
};

// What a call knows of an argument: nothing (no entry), or a closure of an
// entry, with what it knows of each captured value.
struct Shape {
  mlir::FlatSymbolRefAttr entry;
  mlir::ArrayAttr captures;
  std::vector<Shape> captured;

  bool known() const { return static_cast<bool>(entry); }
  void AppendKey(std::string &key) const {
    if (!known()) {
      key += '_';
      return;
    }
    key += entry.getValue();
    key += '(';
    for (const Shape &shape : captured) {
      shape.AppendKey(key);
      key += ',';
    }
    key += ')';
  }
};

class Specializer {
public:
  explicit Specializer(mlir::ModuleOp module)
      : module_{module}, symbols_{module} {}
  void Run();

private:
  std::optional<Known> KnownClosure(mlir::Value) const;
  Shape ShapeOf(mlir::Value, unsigned depth) const;
  void Flatten(mlir::Value, const Shape &, mlir::OpBuilder &,
      llvm::SmallVectorImpl<mlir::Value> &) const;
  bool Fold(merk::ProcOp);
  void SpecializeCalls(merk::ProcOp);
  merk::ProcOp CloneFor(merk::ProcOp callee, llvm::ArrayRef<Shape> shapes);
  void Bind(mlir::Value parameter, const Shape &,
      llvm::ArrayRef<mlir::BlockArgument> &holes, mlir::OpBuilder &);
  bool RemoveUnusedParameters(merk::ProcOp clone);
  void Unused(mlir::Operation *op);
  bool RemoveDeadValues();
  void RemoveUselessClones();

  mlir::ModuleOp module_;
  mlir::SymbolTable symbols_;
  // A clone's parameter that is a known closure, or a word that holds one.
  llvm::DenseMap<mlir::Value, Known> parameters_;
  // The clone for a callee and what its call knows, by key.
  llvm::StringMap<mlir::StringAttr> clones_;
  // The procedure each clone was made from, and how many clones each has.
  llvm::StringMap<std::string> origins_;
  llvm::StringMap<unsigned> counts_;
  // The procedure each clone is a copy of.
  llvm::DenseMap<mlir::Operation *, merk::ProcOp> sources_;
  llvm::SmallVector<merk::ProcOp> made_;
  llvm::SmallVector<merk::ProcOp> pending_;
  // What the specializer took an operand away from, which may be dead now.
  llvm::SetVector<mlir::Operation *> unused_;
};

// The known closure a value is, through the words that hold it.
std::optional<Known> Specializer::KnownClosure(mlir::Value value) const {
  while (value) {
    if (auto found{parameters_.find(value)}; found != parameters_.end()) {
      return found->second;
    }
    mlir::Operation *def{value.getDefiningOp()};
    if (auto closure{mlir::dyn_cast_or_null<merk::ClosureOp>(def)}) {
      Known known{closure.getEntryAttr(), closure.getCapturesAttr(), {}, {}};
      known.captured.append(
          closure.getCaptured().begin(), closure.getCaptured().end());
      return known;
    }
    if (auto toWord{mlir::dyn_cast_or_null<merk::ToWordOp>(def)}) {
      value = toWord.getValue();
    } else if (auto fromWord{mlir::dyn_cast_or_null<merk::FromWordOp>(def)}) {
      value = fromWord.getWord();
    } else {
      return std::nullopt;
    }
  }
  return std::nullopt;
}

Shape Specializer::ShapeOf(mlir::Value value, unsigned depth) const {
  if (depth >= kMaxDepth) {
    return {};
  }
  std::optional<Known> known{KnownClosure(value)};
  if (!known) {
    return {};
  }
  Shape shape{known->entry, known->captures, {}};
  for (mlir::Value captured : known->captured) {
    shape.captured.push_back(ShapeOf(captured, depth + 1));
  }
  return shape;
}

// The type of a clone's parameter for a captured value: a type_info or a
// typeclass_info as itself, so that the collector knows what it is, and
// anything else as the word the closure holds.
static mlir::Type HoleType(mlir::ArrayAttr captures, std::size_t i,
    mlir::MLIRContext *context) {
  auto kind{captures ? mlir::dyn_cast<mlir::StringAttr>(captures[i])
                     : mlir::StringAttr{}};
  if (kind && kind.getValue() == "type_info") {
    return merk::TypeInfoType::get(context);
  }
  if (kind && kind.getValue() == "typeclass_info") {
    return merk::TypeclassInfoType::get(context);
  }
  return merk::WordType::get(context);
}

// Whether a closure's `capture` describes a type, as a type variable's place
// or a constructor applied to such descriptions, rather than being a
// type_info or a typeclass_info.
static bool Describable(mlir::Attribute capture) {
  if (auto place{mlir::dyn_cast<mlir::DenseI64ArrayAttr>(capture)}) {
    return place.size() == 2;
  }
  auto parts{mlir::dyn_cast<mlir::ArrayAttr>(capture)};
  if (!parts || parts.empty() || !mlir::isa<mlir::StringAttr>(parts[0])) {
    return false;
  }
  return llvm::all_of(parts.getValue().drop_front(), Describable);
}

static bool Describable(mlir::ArrayAttr captures, std::size_t i) {
  return captures && Describable(captures[i]);
}

// The types of the parameters a clone takes after a known argument: for each
// captured value, its own, then the type_info of a word whose type the closure
// describes, which the collector needs, and then what a known captured
// closure captured.
static void HoleTypes(const Shape &shape, mlir::MLIRContext *context,
    llvm::SmallVectorImpl<mlir::Type> &types) {
  for (auto [i, inner] : llvm::enumerate(shape.captured)) {
    types.push_back(HoleType(shape.captures, i, context));
    if (Describable(shape.captures, i)) {
      types.push_back(merk::TypeInfoType::get(context));
    }
    if (inner.known()) {
      HoleTypes(inner, context, types);
    }
  }
}

// The type_info that a closure's `capture` describes (Describable), made by
// `builder` from the closure's captured words.
static mlir::Value DescribedInfo(mlir::Attribute capture,
    llvm::ArrayRef<mlir::Value> captured, mlir::OpBuilder &builder,
    mlir::Location loc) {
  mlir::MLIRContext *context{builder.getContext()};
  auto infoType{merk::TypeInfoType::get(context)};
  // A type variable: the type_info at a place among the captured values, or
  // a word of the typeclass_info there.
  if (auto place{mlir::dyn_cast<mlir::DenseI64ArrayAttr>(capture)}) {
    if (place.size() != 2 ||
        static_cast<std::size_t>(place[0]) >= captured.size()) {
      return {};
    }
    mlir::Value word{captured[place[0]]};
    if (place[1] == 0) {
      return merk::FromWordOp::create(builder, loc, infoType, word);
    }
    mlir::Value dictionary{merk::FromWordOp::create(
        builder, loc, merk::TypeclassInfoType::get(context), word)};
    return merk::TypeclassInfoWordOp::create(
        builder, loc, infoType, dictionary, place[1]);
  }
  auto parts{mlir::dyn_cast<mlir::ArrayAttr>(capture)};
  if (!parts || parts.empty()) {
    return {};
  }
  auto name{mlir::dyn_cast<mlir::StringAttr>(parts[0])};
  if (!name) {
    return {};
  }
  llvm::SmallVector<mlir::Value> arguments;
  for (mlir::Attribute part : parts.getValue().drop_front()) {
    mlir::Value argument{DescribedInfo(part, captured, builder, loc)};
    if (!argument) {
      return {};
    }
    arguments.push_back(argument);
  }
  return merk::TypeInfoOp::create(builder, loc, infoType, name, arguments);
}

// The values a clone takes after a known argument: what it captured, each
// followed by what a known captured closure captured, at the types HoleType
// gives, converted by `builder`.
void Specializer::Flatten(mlir::Value value, const Shape &shape,
    mlir::OpBuilder &builder, llvm::SmallVectorImpl<mlir::Value> &values) const {
  std::optional<Known> known{KnownClosure(value)};
  for (auto [i, captured, inner] :
      llvm::enumerate(known->captured, shape.captured)) {
    mlir::Type type{HoleType(shape.captures, i, builder.getContext())};
    values.push_back(type == captured.getType()
            ? captured
            : merk::FromWordOp::create(builder, captured.getLoc(), type, captured)
                  .getResult());
    if (Describable(shape.captures, i)) {
      values.push_back(i < known->infos.size() && known->infos[i]
              ? known->infos[i]
              : DescribedInfo(shape.captures[i], known->captured, builder,
                    captured.getLoc()));
    }
    if (inner.known()) {
      Flatten(captured, inner, builder, values);
    }
  }
}

// Records that a clone's parameter is a known closure whose captured values
// are the parameters `holes` begins with, and takes those from `holes`. A
// type_info or a typeclass_info is used as the word the closure held, made by
// `builder` at the start, and a word whose type the closure describes gets a
// note (merk.root_type) of the type_info after it, as MLIR generation gives
// a polymorphic value one, for the collector.
void Specializer::Bind(mlir::Value parameter, const Shape &shape,
    llvm::ArrayRef<mlir::BlockArgument> &holes, mlir::OpBuilder &builder) {
  Known known{shape.entry, shape.captures, {}, {}};
  for (auto [i, inner] : llvm::enumerate(shape.captured)) {
    mlir::BlockArgument hole{holes.front()};
    holes = holes.drop_front();
    if (mlir::isa<merk::WordType>(hole.getType())) {
      known.captured.push_back(hole);
    } else {
      known.captured.push_back(merk::ToWordOp::create(builder, hole.getLoc(),
          merk::WordType::get(builder.getContext()), hole));
    }
    if (Describable(shape.captures, i)) {
      mlir::BlockArgument info{holes.front()};
      holes = holes.drop_front();
      merk::RootTypeOp::create(builder, hole.getLoc(), hole, info);
      known.infos.resize(i + 1);
      known.infos[i] = info;
    }
    if (inner.known()) {
      Bind(hole, inner, holes, builder);
    }
  }
  parameters_[parameter] = std::move(known);
}

// Makes the procedure's uses of known closures direct: a captured value read
// from one is the value, and a call of one calls its entry. Whether it
// changed anything.
bool Specializer::Fold(merk::ProcOp proc) {
  bool changed{false};
  llvm::SmallVector<mlir::Operation *> ops;
  proc.walk([&](mlir::Operation *op) {
    if (mlir::isa<merk::ClosureWordOp, merk::CallClosureOp,
            merk::NondetCallClosureOp>(op)) {
      ops.push_back(op);
    }
  });
  for (mlir::Operation *op : ops) {
    if (auto read{mlir::dyn_cast<merk::ClosureWordOp>(op)}) {
      std::optional<Known> known{KnownClosure(read.getClosure())};
      if (known && read.getIndex() < known->captured.size()) {
        read.getWord().replaceAllUsesWith(known->captured[read.getIndex()]);
        Unused(read);
        read.erase();
        changed = true;
      }
      continue;
    }
    mlir::Value closure{op->getOperand(0)};
    std::optional<Known> known{KnownClosure(closure)};
    if (!known) {
      continue;
    }
    auto entry{symbols_.lookup<merk::ProcOp>(known->entry.getAttr())};
    if (!entry) {
      continue;
    }
    llvm::SmallVector<mlir::Value> inputs{closure};
    inputs.append(std::next(op->operand_begin()), op->operand_end());
    mlir::FunctionType type{entry.getFunctionType()};
    if (mlir::ValueRange{inputs}.getTypes() != type.getInputs()) {
      continue;
    }
    bool nondet{entry.getDeterminism() == merk::Determinism::Nondet ||
        entry.getDeterminism() == merk::Determinism::Multi};
    mlir::OpBuilder builder{op};
    if (auto call{mlir::dyn_cast<merk::CallClosureOp>(op)}) {
      if (nondet || call.getOutputs().getTypes() != type.getResults()) {
        continue;
      }
      auto direct{merk::CallOp::create(builder, op->getLoc(),
          call.getOutputs().getTypes(), known->entry.getValue(), inputs)};
      call.replaceAllUsesWith(direct.getOutputs());
      Unused(call);
      call.erase();
    } else {
      auto nondetCall{mlir::cast<merk::NondetCallClosureOp>(op)};
      if (!nondet ||
          nondetCall.getContinuation().front().getArgumentTypes() !=
              type.getResults()) {
        continue;
      }
      auto direct{merk::NondetCallOp::create(
          builder, op->getLoc(), known->entry.getValue(), inputs)};
      direct.getContinuation().takeBody(nondetCall.getContinuation());
      Unused(nondetCall);
      nondetCall.erase();
    }
    changed = true;
  }
  return changed;
}

// Whether a procedure may be cloned: it has a body, and neither foreign code,
// which may not be duplicated, nor a target region, whose kernel would be.
static bool Clonable(merk::ProcOp proc) {
  if (proc.isExternal()) {
    return false;
  }
  mlir::WalkResult found{proc.walk([](mlir::Operation *op) {
    return mlir::isa<merk::ForeignCallOp, mlir::omp::TargetOp>(op)
        ? mlir::WalkResult::interrupt()
        : mlir::WalkResult::advance();
  })};
  return !found.wasInterrupted();
}

// Makes each call in the procedure that passes a known closure to a
// procedure with a body call a clone of it.
void Specializer::SpecializeCalls(merk::ProcOp proc) {
  llvm::SmallVector<mlir::Operation *> calls;
  proc.walk([&](mlir::Operation *op) {
    if (mlir::isa<merk::CallOp, merk::NondetCallOp>(op)) {
      calls.push_back(op);
    }
  });
  for (mlir::Operation *call : calls) {
    auto symbol{call->getAttrOfType<mlir::FlatSymbolRefAttr>("callee")};
    auto callee{symbols_.lookup<merk::ProcOp>(symbol.getAttr())};
    if (!callee || !Clonable(callee)) {
      continue;
    }
    llvm::SmallVector<Shape> shapes;
    bool any{false};
    for (mlir::Value input : call->getOperands()) {
      shapes.push_back(ShapeOf(input, 0));
      any = any || shapes.back().known();
    }
    if (!any) {
      continue;
    }
    merk::ProcOp clone{CloneFor(callee, shapes)};
    if (!clone) {
      continue;
    }
    llvm::SmallVector<mlir::Value> inputs;
    mlir::OpBuilder builder{call};
    for (auto [input, shape] : llvm::zip_equal(call->getOperands(), shapes)) {
      inputs.push_back(input);
      if (shape.known()) {
        Flatten(input, shape, builder, inputs);
      }
    }
    call->setOperands(inputs);
    call->setAttr("callee", mlir::FlatSymbolRefAttr::get(clone.getSymNameAttr()));
  }
}

// The clone of `callee` for a call whose inputs have those shapes, made if
// there is none yet; null once the procedure has had its clones.
merk::ProcOp Specializer::CloneFor(
    merk::ProcOp callee, llvm::ArrayRef<Shape> shapes) {
  std::string key{callee.getSymName().str()};
  key += '|';
  for (const Shape &shape : shapes) {
    shape.AppendKey(key);
    key += ';';
  }
  if (auto found{clones_.find(key)}; found != clones_.end()) {
    return symbols_.lookup<merk::ProcOp>(found->second);
  }
  std::string origin{callee.getSymName().str()};
  if (auto found{origins_.find(origin)}; found != origins_.end()) {
    origin = found->second;
  }
  unsigned &count{counts_[origin]};
  if (count >= kMaxClones) {
    return {};
  }
  ++count;
  mlir::MLIRContext *context{callee.getContext()};
  auto clone{mlir::cast<merk::ProcOp>(callee->clone())};
  std::string name;
  for (unsigned n = count;; ++n) {
    name = origin + "$spec" + std::to_string(n);
    if (!symbols_.lookup(name)) {
      break;
    }
  }
  clone.setSymName(name);
  mlir::SymbolTable::setSymbolVisibility(
      clone, mlir::SymbolTable::Visibility::Private);
  symbols_.insert(clone, std::next(callee->getIterator()));
  clones_[key] = clone.getSymNameAttr();
  origins_[name] = origin;
  sources_[clone] = callee;
  // After each known argument, its captured values, from the last argument
  // to the first so that the earlier places stay.
  mlir::Block &entry{clone.getBody().front()};
  mlir::ArrayAttr argAttrs{clone.getArgAttrsAttr()};
  llvm::SmallVector<mlir::Attribute> attrs;
  if (argAttrs) {
    attrs.append(argAttrs.begin(), argAttrs.end());
  }
  for (std::size_t i = shapes.size(); i-- > 0;) {
    if (!shapes[i].known()) {
      continue;
    }
    llvm::SmallVector<mlir::Type> types;
    HoleTypes(shapes[i], context, types);
    llvm::SmallVector<mlir::BlockArgument> holes;
    for (auto [k, type] : llvm::enumerate(types)) {
      holes.push_back(entry.insertArgument(
          static_cast<unsigned>(i + 1 + k), type, clone.getLoc()));
    }
    if (argAttrs) {
      attrs.insert(attrs.begin() + i + 1, types.size(),
          mlir::DictionaryAttr::get(context));
    }
    llvm::ArrayRef<mlir::BlockArgument> rest{holes};
    mlir::OpBuilder builder{&entry, entry.begin()};
    Bind(entry.getArgument(static_cast<unsigned>(i)), shapes[i], rest, builder);
  }
  clone.setFunctionType(mlir::FunctionType::get(
      context, entry.getArgumentTypes(), callee.getResultTypes()));
  if (argAttrs) {
    clone.setArgAttrsAttr(mlir::ArrayAttr::get(context, attrs));
  }
  made_.push_back(clone);
  pending_.push_back(clone);
  return clone;
}

// Whether nothing uses a clone's parameter but its calls of itself, which
// pass it on in the same place, as a fold passes its closure.
static bool OnlyPassedOn(
    mlir::BlockArgument parameter, merk::ProcOp clone, unsigned place) {
  for (mlir::OpOperand &use : parameter.getUses()) {
    mlir::Operation *user{use.getOwner()};
    if (auto note{mlir::dyn_cast<merk::RootTypeOp>(user)};
        note && note.getValue() == parameter) {
      continue; // Removed with it.
    }
    auto callee{user->getAttrOfType<mlir::FlatSymbolRefAttr>("callee")};
    if (!mlir::isa<merk::CallOp, merk::NondetCallOp>(user) || !callee ||
        callee.getAttr() != clone.getSymNameAttr() ||
        use.getOperandNumber() != place) {
      return false;
    }
  }
  return true;
}

// Removes the clone's parameters that nothing uses, or only passes on to
// itself, from it and its calls. Whether it removed any.
bool Specializer::RemoveUnusedParameters(merk::ProcOp clone) {
  std::optional<mlir::SymbolTable::UseRange> uses{
      mlir::SymbolTable::getSymbolUses(clone, module_)};
  if (!uses) {
    return false;
  }
  llvm::SmallVector<mlir::Operation *> calls;
  for (const mlir::SymbolTable::SymbolUse &use : *uses) {
    if (!mlir::isa<merk::CallOp, merk::NondetCallOp>(use.getUser())) {
      return false;
    }
    calls.push_back(use.getUser());
  }
  mlir::Block &entry{clone.getBody().front()};
  mlir::ArrayAttr argAttrs{clone.getArgAttrsAttr()};
  llvm::SmallVector<mlir::Attribute> attrs;
  if (argAttrs) {
    attrs.append(argAttrs.begin(), argAttrs.end());
  }
  bool changed{false};
  for (unsigned i = entry.getNumArguments(); i-- > 0;) {
    if (!OnlyPassedOn(entry.getArgument(i), clone, i)) {
      continue;
    }
    for (mlir::Operation *call : calls) {
      if (mlir::Operation *def{call->getOperand(i).getDefiningOp()}) {
        unused_.insert(def);
      }
      call->eraseOperand(i);
    }
    for (mlir::Operation *user :
        llvm::make_early_inc_range(entry.getArgument(i).getUsers())) {
      Unused(user);
      user->erase(); // Its merk.root_type.
    }
    entry.eraseArgument(i);
    if (argAttrs) {
      attrs.erase(attrs.begin() + i);
    }
    changed = true;
  }
  if (changed) {
    clone.setFunctionType(mlir::FunctionType::get(clone.getContext(),
        entry.getArgumentTypes(), clone.getResultTypes()));
    if (argAttrs) {
      clone.setArgAttrsAttr(mlir::ArrayAttr::get(clone.getContext(), attrs));
    }
  }
  return changed;
}

// Records that `op`, about to be erased, no longer uses its operands.
void Specializer::Unused(mlir::Operation *op) {
  for (mlir::Value operand : op->getOperands()) {
    if (mlir::Operation *def{operand.getDefiningOp()}) {
      unused_.insert(def);
    }
  }
}

// Removes what the specializer left unused: closures, the words and captured
// values of them, and the type_infos that said their types, and in turn what
// they used. Only those: MLIR generation leaves conversions that nothing
// uses as notes of a word's type, which the collector reads. Whether it
// removed any.
bool Specializer::RemoveDeadValues() {
  bool changed{false};
  while (!unused_.empty()) {
    mlir::Operation *op{unused_.pop_back_val()};
    if (!op->use_empty() ||
        !mlir::isa<merk::ClosureOp, merk::ClosureWordOp, merk::ToWordOp,
            merk::FromWordOp, merk::TypeInfoOp, merk::TypeclassInfoWordOp>(op)) {
      continue;
    }
    Unused(op);
    unused_.remove(op);
    op->erase();
    changed = true;
  }
  return changed;
}

void Specializer::Run() {
  for (auto proc : module_.getOps<merk::ProcOp>()) {
    if (!proc.isExternal()) {
      pending_.push_back(proc);
    }
  }
  while (!pending_.empty()) {
    merk::ProcOp proc{pending_.pop_back_val()};
    while (Fold(proc)) {
    }
    SpecializeCalls(proc);
  }
  parameters_.clear();
  bool changed{true};
  while (changed) {
    changed = RemoveDeadValues();
    for (merk::ProcOp clone : made_) {
      changed = RemoveUnusedParameters(clone) || changed;
    }
  }
  RemoveUselessClones();
}

// Calls the procedure a clone was copied from in place of the clone, where
// nothing was made direct in it and it is the same procedure, as when the
// callee only put the closure into another closure.
void Specializer::RemoveUselessClones() {
  for (merk::ProcOp clone : llvm::reverse(made_)) {
    merk::ProcOp source{sources_.lookup(clone)};
    if (!source || source.getFunctionType() != clone.getFunctionType() ||
        !mlir::OperationEquivalence::isRegionEquivalentTo(&source.getBody(),
            &clone.getBody(), mlir::OperationEquivalence::IgnoreLocations)) {
      continue;
    }
    if (mlir::failed(mlir::SymbolTable::replaceAllSymbolUses(
            clone, source.getSymNameAttr(), module_))) {
      continue;
    }
    symbols_.erase(clone);
  }
}

} // namespace

void SpecializeKnownClosures(mlir::ModuleOp module) {
  Specializer{module}.Run();
}

} // namespace merkurius::codegen
