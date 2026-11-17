//===-- lib/CodeGen/MLIRGenAtomic.cpp - OpenMP atomic ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The atomic directive on an update of an array's element, Fortran's
// x = x op expr with x an element (docs/INPROGRESS.md, Atomic updates of
// array elements):
//
//   %$omp atomic
//   !A ^ elem(I) := !.A ^ elem(I) + E
//
//   %$omp atomic capture
//   ( V = !.A ^ elem(I), !A ^ elem(I) := V + 1 )
//
// The directive's goal is recognised in its flattened form, the calls of
// array.elem or lookup, of the operator and of array.'elem :=' or set, and
// generated as Flang generates the statement: the operand E before the
// atomic operation, then omp.atomic.update on the element's address, whose
// region applies the operator to the element's old value, within
// omp.atomic.capture with omp.atomic.read for a capture.
//
//===----------------------------------------------------------------------===//

#include "MLIRGenImpl.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

namespace merkurius::codegen::mlirgen {

namespace {

// A call among the directive's goals, with its arguments, a function's
// result last, split into its inputs and outputs.
struct AtomicCall {
  const Body *goal{nullptr};
  const Procedure *callee{nullptr};
  llvm::SmallVector<BodyVar> in;
  llvm::SmallVector<BodyVar> out;
};

// A read of the element: of the array before the update, its old value,
// or after it, its new value.
struct AtomicRead {
  const Body *goal;
  BodyVar array, index, value;
};

} // namespace

static bool IsReadName(llvm::StringRef name) {
  return llvm::is_contained(
      {"lookup", "unsafe_lookup", "elem", "unsafe_elem"}, name);
}

static bool IsWriteName(llvm::StringRef name) {
  return llvm::is_contained(
      {"set", "unsafe_set", "elem :=", "unsafe_elem :="}, name);
}

// The operators of OpenMP's atomic update that the int and float modules
// have as functions, built in.
static bool IsAtomicOperator(llvm::StringRef module, llvm::StringRef name) {
  if (module == "float") {
    return llvm::is_contained({"+", "-", "*"}, name);
  }
  return module == "int" &&
      llvm::is_contained({"+", "-", "*", "/\\", "\\/", "xor"}, name);
}

static void Leaves(const Body &goal, llvm::SmallVectorImpl<const Body *> &leaves) {
  if (goal.kind == BodyKind::Conj) {
    for (const Body *conjunct : goal.goals) {
      Leaves(*conjunct, leaves);
    }
  } else {
    leaves.push_back(&goal);
  }
}

// The variables a unification or a call names.
static void VarsOf(const Body &goal, llvm::SmallVectorImpl<BodyVar> &vars) {
  vars.append(goal.arguments.begin(), goal.arguments.end());
  if (goal.kind == BodyKind::Unify) {
    vars.push_back(goal.left);
    if (goal.isVar) {
      vars.push_back(goal.var);
    }
  }
}

// An index given as a constant, by merk.constant or arith.constant.
static std::optional<std::int64_t> ConstantIndex(mlir::Value index) {
  if (auto constant{index.getDefiningOp<merk::ConstantOp>()}) {
    if (auto value{mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue())}) {
      return value.getInt();
    }
    return std::nullopt;
  }
  return mlir::getConstantIntValue(index);
}

static bool SameIndex(mlir::Value a, mlir::Value b) {
  if (a == b) {
    return true;
  }
  std::optional<std::int64_t> x{ConstantIndex(a)}, y{ConstantIndex(b)};
  return x && y && *x == *y;
}

void MerkGenerator::OmpAtomic(const Body &goal, const ProcedureBody &body) {
  using namespace parser::literals;
  const parser::OmpDirective &directive{*goal.directive};
  parser::CharBlock at{Where(goal.tokens)};
  mlir::Location loc{Location(goal.tokens)};
  bool capture{false};
  for (const parser::OmpClause &clause : directive.clauses) {
    llvm::omp::Clause id{omp::ClauseId(clause)};
    if (id == llvm::omp::OMPC_capture) {
      capture = true;
    } else if (id != llvm::omp::OMPC_update) {
      NotYet(at,
          "the " + llvm::omp::getOpenMPClauseName(id).str() +
              " clause of the atomic directive");
      return;
    }
  }
  static constexpr char kUpdateForm[]{
      "an atomic update must be !A ^ elem(I) := !.A ^ elem(I) op E, with op "
      "one of int's or float's +, -, * or int's /\\, \\/ and xor, and E "
      "not using the element"};
  static constexpr char kCaptureForm[]{
      "an atomic capture must be an atomic update and a read of its "
      "element, V = !.A ^ elem(I) before it or V = !.A ^ elem(I) after it"};
  auto form{[&]() {
    Error(at, "%s"_err_en_US, capture ? kCaptureForm : kUpdateForm);
  }};

  llvm::SmallVector<const Body *> leaves;
  Leaves(*goal.goals.front(), leaves);
  llvm::SmallVector<const Body *> others;
  llvm::SmallVector<AtomicCall> writes;
  llvm::SmallVector<AtomicCall> reads;
  llvm::DenseMap<BodyVar, AtomicCall> results;
  for (const Body *leaf : leaves) {
    bool call{leaf->kind == BodyKind::Call ||
        (leaf->kind == BodyKind::Unify && leaf->unify == UnifyKind::Call)};
    if (!call && leaf->kind != BodyKind::Unify) {
      form();
      return;
    }
    if (!call || !leaf->modes || !leaf->modes->callee) {
      others.push_back(leaf);
      continue;
    }
    AtomicCall c{leaf, leaf->modes->callee};
    llvm::SmallVector<BodyVar> arguments(
        leaf->arguments.begin(), leaf->arguments.end());
    if (leaf->kind == BodyKind::Unify) {
      arguments.push_back(leaf->left);
    }
    for (auto [argument, mode] : llvm::zip(arguments, c.callee->modes)) {
      (IsOutput(mode) ? c.out : c.in).push_back(argument);
    }
    const semantics::Symbol &symbol{*c.callee->symbol};
    if (symbol.module == "array" && IsWriteName(symbol.name) &&
        c.in.size() == 3 && c.out.size() == 1) {
      writes.push_back(c);
    } else if (symbol.module == "array" && IsReadName(symbol.name) &&
        c.in.size() == 2 && c.out.size() == 1) {
      reads.push_back(c);
    } else {
      if (c.out.size() == 1) {
        results[c.out[0]] = c;
      }
      others.push_back(leaf);
    }
  }
  if (writes.size() != 1) {
    form();
    return;
  }
  // set(I, X, !A), set(A, I, X) = A1 and 'elem :='(I, A, X) = A1.
  const AtomicCall &write{writes.front()};
  llvm::StringRef writeName{write.callee->symbol->name};
  bool function{
      write.callee->symbol->kind == semantics::SymbolKind::Function};
  BodyVar index, before, element;
  if (writeName.ends_with("elem :=")) {
    std::tie(index, before, element) =
        std::tuple{write.in[0], write.in[1], write.in[2]};
  } else if (function) {
    std::tie(before, index, element) =
        std::tuple{write.in[0], write.in[1], write.in[2]};
  } else {
    std::tie(index, element, before) =
        std::tuple{write.in[0], write.in[1], write.in[2]};
  }
  BodyVar after{write.out[0]};
  // The operator that makes the element written.
  auto op{results.find(element)};
  if (op == results.end() || op->second.in.size() != 2) {
    form();
    return;
  }
  const AtomicCall &update{op->second};
  const semantics::Symbol &opSymbol{*update.callee->symbol};
  if (opSymbol.kind != semantics::SymbolKind::Function ||
      !IsAtomicOperator(opSymbol.module, opSymbol.name)) {
    form();
    return;
  }
  llvm::erase(others, update.goal);
  // The reads: lookup(A, I) and elem(I, A).
  llvm::SmallVector<AtomicRead> old, fresh;
  for (const AtomicCall &read : reads) {
    bool elem{read.callee->symbol->name.ends_with("elem")};
    AtomicRead r{read.goal, read.in[elem ? 1 : 0], read.in[elem ? 0 : 1],
        read.out[0]};
    if (r.array == before) {
      old.push_back(r);
    } else if (r.array == after) {
      fresh.push_back(r);
    } else {
      form();
      return;
    }
  }
  // The operator applies to one old value and to E.
  auto isOld{[&](BodyVar var) {
    return llvm::any_of(old, [&](const AtomicRead &r) { return r.value == var; });
  }};
  bool firstOld{isOld(update.in[0])}, secondOld{isOld(update.in[1])};
  if (firstOld == secondOld) {
    form();
    return;
  }
  BodyVar oldVar{update.in[firstOld ? 0 : 1]};
  BodyVar operand{update.in[firstOld ? 1 : 0]};
  // What it captures: the one read's value, or the value written, that the
  // goal gives to what follows it.
  std::optional<BodyVar> captured;
  bool capturesNew{false};
  unsigned capturing{0};
  auto nonlocal{[&](BodyVar var) {
    return llvm::is_contained(goal.nonlocals, var);
  }};
  for (const AtomicRead &r : old) {
    if (nonlocal(r.value)) {
      ++capturing;
      captured = r.value;
    } else if (r.value != oldVar) {
      form();
      return;
    }
  }
  for (const AtomicRead &r : fresh) {
    if (!nonlocal(r.value)) {
      form();
      return;
    }
    ++capturing;
    captured = r.value;
    capturesNew = true;
  }
  if (nonlocal(element)) {
    ++capturing;
    captured = element;
    capturesNew = true;
  }
  if (capture ? capturing != 1 : capturing != 0 || old.size() != 1) {
    form();
    return;
  }
  // Nothing else uses the element's values, nor the array, but to give
  // the array updated.
  llvm::DenseSet<BodyVar> atomic{element, before};
  for (const AtomicRead &r : old) {
    atomic.insert(r.value);
  }
  for (const AtomicRead &r : fresh) {
    atomic.insert(r.value);
  }
  for (const Body *other : others) {
    llvm::SmallVector<BodyVar> vars;
    VarsOf(*other, vars);
    if (llvm::any_of(vars, [&](BodyVar var) { return atomic.contains(var); })) {
      form();
      return;
    }
  }
  if (atomic.contains(operand) || atomic.contains(index)) {
    form();
    return;
  }

  // E and the index, then the atomic operation.
  for (const Body *other : others) {
    Goal(*other, body);
  }
  if (failed_ || Terminated()) {
    return;
  }
  mlir::Value array{ValueOf(before)}, at_{ValueOf(index)};
  if (!array || !at_ || !ValueOf(operand)) {
    NotYet(at, "this atomic update");
    return;
  }
  for (const AtomicRead &r : old) {
    if (!SameIndex(ValueOf(r.index), at_)) {
      form();
      return;
    }
  }
  for (const AtomicRead &r : fresh) {
    if (!SameIndex(ValueOf(r.index), at_)) {
      form();
      return;
    }
  }
  mlir::Type type{mlir::cast<merk::ArrayType>(array.getType()).getElement()};
  if (!(type.isInteger(64) && opSymbol.module == "int") &&
      !(type.isF64() && opSymbol.module == "float")) {
    NotYet(at, "atomic updates of arrays other than of ints and floats");
    return;
  }
  if (!writeName.starts_with("unsafe")) {
    merk::CheckIndexOp::create(builder_, loc, array, at_, "array.set");
  }
  auto ptr{mlir::LLVM::LLVMPointerType::get(&context_)};
  mlir::Value address{
      merk::ArrayElementRefOp::create(builder_, loc, ptr, array, at_)};
  mlir::Value slot;
  if (capture) {
    slot = Slot(type, loc);
    auto op{mlir::omp::AtomicCaptureOp::create(builder_, loc,
        mlir::IntegerAttr{}, mlir::omp::ClauseMemoryOrderKindAttr{},
        /*fail_only=*/nullptr)};
    builder_.createBlock(&op.getRegion());
  }
  auto read{[&]() {
    mlir::omp::AtomicReadOp::create(builder_, loc, address, slot,
        mlir::TypeAttr::get(type), mlir::IntegerAttr{},
        mlir::omp::ClauseMemoryOrderKindAttr{});
  }};
  if (capture && !capturesNew) {
    read();
  }
  // The atomic control of Flang's defaults (flang/lib/Lower/OpenMP/
  // Atomic.cpp, from -fatomic-ignore-denormal-mode and the rest).
  auto atomicUpdate{mlir::omp::AtomicUpdateOp::create(builder_, loc, address,
      mlir::omp::AtomicControlAttr::get(&context_, false, false, false),
      mlir::IntegerAttr{}, mlir::omp::ClauseMemoryOrderKindAttr{})};
  {
    mlir::OpBuilder::InsertionGuard guard{builder_};
    mlir::Block *block{
        builder_.createBlock(&atomicUpdate.getRegion(), {}, {type}, {loc})};
    values_[oldVar] = block->getArgument(0);
    llvm::SmallVector<BodyVar> arguments(update.in.begin(), update.in.end());
    arguments.push_back(element);
    Call(*update.goal, *update.callee, arguments, body);
    mlir::Value updated{ValueOf(element)};
    if (failed_ || !updated) {
      NotYet(at, "this atomic update");
      return;
    }
    mlir::omp::YieldOp::create(builder_, loc, updated);
  }
  values_.erase(oldVar);
  values_.erase(element);
  if (capture && capturesNew) {
    read();
  }
  if (capture) {
    mlir::omp::TerminatorOp::create(builder_, loc);
    builder_.setInsertionPointAfter(builder_.getInsertionBlock()->getParentOp());
    values_[*captured] = mlir::LLVM::LoadOp::create(builder_, loc, type, slot);
  }
  // The array after it is the one before, updated in place.
  values_[after] = array;
}

} // namespace merkurius::codegen::mlirgen
