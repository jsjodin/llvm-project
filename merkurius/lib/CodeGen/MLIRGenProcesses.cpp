//===-- lib/CodeGen/MLIRGenProcesses.cpp - GPU processes' kernels ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Process kinds that run on a GPU (docs/INPROGRESS.md, GPU processes without
// omp target, step 4b, and GPU processes that run loops, one kernel for a
// GPU's processes). The device's compile of them (-fprocess-kernels) makes a
// start for each, which calls the kind's predicate, and
// merk_process_dispatch, which starts a kind by its number. The image's one
// kernel, merk_place, comes from its start object (process-start.cpp): each
// of its waves runs a process, which it starts by merk_process_dispatch. The
// host's compile embeds the image, one for each GPU, and registers it with
// the runtime at start-up, with the kinds' names in the order that numbers
// them.
//
//===----------------------------------------------------------------------===//

#include "MLIRGenImpl.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/BinaryFormat/MsgPackDocument.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Support/MemoryBuffer.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace merkurius::codegen {

using namespace parser::literals;
using semantics::Procedure;

namespace mlirgen {

namespace {
// A mailbox handle's cell, as the kernel takes it: its five words
// (FlatCell).
constexpr unsigned kMailboxWords{5};
// The words that a loop across a GPU process's teams gives its body, in the
// process's control block: runtime/place.h's kProcessLoopWords.
constexpr unsigned kProcessLoopWords{32};
// The most reductions of such a loop: runtime/place.h's
// kProcessLoopReductions.
constexpr unsigned kProcessLoopReductions{8};
} // namespace

void MerkGenerator::CollectGpuKinds() {
  for (const parser::Item &item : tree_.items) {
    if (const auto *decl{std::get_if<const parser::ProcessDecl *>(&item)};
        decl && (*decl)->onGpu) {
      gpuKinds_.push_back(*decl);
    }
  }
}

// The procedure of a kind's predicate: its arguments, a mailbox for each
// message type, and the io state.
const Procedure *MerkGenerator::KindProcedure(const parser::ProcessDecl &decl) {
  std::size_t arity{decl.arguments.size() + decl.receives.size() + 2};
  for (const Procedure &procedure : modes_.procedures()) {
    const semantics::Symbol &symbol{*procedure.symbol};
    if (symbol.module == moduleName_ &&
        symbol.kind == semantics::SymbolKind::Predicate &&
        symbol.name == decl.name && symbol.arity == arity) {
      return &procedure;
    }
  }
  return nullptr;
}

namespace {
// The word of `words`, a generic pointer, at `index`.
mlir::Value LoadWord(mlir::OpBuilder &builder, mlir::Location loc,
    mlir::Value words, std::int32_t index) {
  auto i64{builder.getI64Type()};
  mlir::Value address{mlir::LLVM::GEPOp::create(builder, loc, words.getType(),
      i64, words, llvm::ArrayRef<mlir::LLVM::GEPArg>{index})};
  return mlir::LLVM::LoadOp::create(builder, loc, i64, address);
}

// Stores `word`, an i64, at `index` of `words`, a generic pointer.
void StoreWord(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value words,
    std::int32_t index, mlir::Value word) {
  auto i64{builder.getI64Type()};
  mlir::Value address{mlir::LLVM::GEPOp::create(builder, loc, words.getType(),
      i64, words, llvm::ArrayRef<mlir::LLVM::GEPArg>{index})};
  mlir::LLVM::StoreOp::create(builder, loc, word, address);
}

// A value as the i64 of its word.
mlir::Value WordBits(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value value) {
  return merk::FromWordOp::create(builder, loc, builder.getI64Type(),
      merk::ToWordOp::create(
          builder, loc, merk::WordType::get(builder.getContext()), value));
}

// The value of the type whose word's bits are the i64 `bits`.
mlir::Value FromWordBits(mlir::OpBuilder &builder, mlir::Location loc,
    mlir::Type type, mlir::Value bits) {
  return merk::FromWordOp::create(builder, loc, type,
      merk::ToWordOp::create(
          builder, loc, merk::WordType::get(builder.getContext()), bits));
}
} // namespace

void MerkGenerator::GenerateProcessStarts() {
  if (!target_.options().processKernels) {
    return;
  }
  llvm::SmallVector<merk::ProcOp> starts;
  for (const parser::ProcessDecl *decl : gpuKinds_) {
    starts.push_back(GenerateProcessStart(*decl));
  }
  // merk_process_dispatch(kind, words): the start of the kind with the
  // number, by the kinds' order in the module, which the host's runtime
  // finds by their names (RegisterProcessImages). Every image has it, as its
  // kernel calls it (process-start.cpp), even one of a module without kinds
  // that run on a GPU, which the host's compile leaves out.
  mlir::Location loc{builder_.getUnknownLoc()};
  auto i64{builder_.getI64Type()};
  mlir::OpBuilder::InsertionGuard guard{builder_};
  builder_.setInsertionPointToEnd(module_.getBody());
  auto dispatch{merk::ProcOp::create(builder_, loc, "merk_process_dispatch",
      builder_.getFunctionType({i64, i64}, {}), merk::Determinism::Det)};
  dispatch->setAttr(kProcessRootAttrName, builder_.getUnitAttr());
  mlir::Block *entry{dispatch.addEntryBlock()};
  builder_.setInsertionPointToStart(entry);
  if (llvm::none_of(starts, [](merk::ProcOp start) { return bool(start); })) {
    merk::SucceedOp::create(builder_, loc, mlir::ValueRange{});
    return;
  }
  llvm::SmallVector<mlir::Attribute> cases;
  llvm::SmallVector<merk::ProcOp> called;
  for (auto [number, start] : llvm::enumerate(starts)) {
    if (start) {
      cases.push_back(
          builder_.getArrayAttr({builder_.getI64IntegerAttr(number)}));
      called.push_back(start);
    }
  }
  auto op{merk::SwitchOp::create(builder_, loc, mlir::TypeRange{},
      entry->getArgument(0), builder_.getArrayAttr(cases), /*canFail=*/false)};
  for (auto [region, start] : llvm::zip_equal(op.getCaseRegions(), called)) {
    builder_.setInsertionPointToStart(&region.emplaceBlock());
    merk::CallOp::create(builder_, loc, mlir::TypeRange{}, start.getSymName(),
        mlir::ValueRange{entry->getArgument(1)});
    merk::YieldOp::create(builder_, loc, mlir::ValueRange{});
  }
  builder_.setInsertionPointAfter(op);
  merk::SucceedOp::create(builder_, loc, mlir::ValueRange{});
}

// The start of a kind, merk_process_<kind>(words), which takes the address
// of the process's argument words, the kind's arguments and then its
// mailbox's handle, and calls the kind's predicate. In the first version,
// the arguments are integers, floats, characters and mailboxes, a word
// each but a mailbox, whose handle is five words, and the kind is not
// polymorphic (decision D3). The start puts a mailbox's handle in a cell on
// its stack: the predicate returns before the start does, and a handle that
// it sends is copied. Null for a kind it cannot start, which it reported.
merk::ProcOp MerkGenerator::GenerateProcessStart(
    const parser::ProcessDecl &decl) {
  parser::CharBlock at{Where(decl.tokens)};
  mlir::Location loc{Location(at)};
  const Procedure *procedure{KindProcedure(decl)};
  // A kind whose predicate code generation left out has been reported.
  merk::ProcOp kind{procedure ? procs_.lookup(procedure) : nullptr};
  if (!kind || kind.isExternal()) {
    return nullptr;
  }
  const semantics::Signature *signature{types_.SignatureOf(*procedure->symbol)};
  if (!signature) {
    return nullptr;
  }
  if (signature->typeVars != 0) {
    messages_.Say(at, "code generation for %s"_todo_en_US,
        "a process kind that runs on a GPU and is polymorphic");
    return nullptr;
  }
  // The kind's arguments and its mailbox; the io state has no value.
  llvm::SmallVector<mlir::Type> values;
  bool ok{true};
  for (const semantics::Type *type :
      signature->arguments.take_front(decl.arguments.size() + 1)) {
    std::optional<Represented> represented{Represent(type)};
    mlir::Type merkType{represented ? represented->type : mlir::Type{}};
    auto term{mlir::dyn_cast_if_present<merk::TermType>(merkType)};
    if (!(represented && !represented->dummy &&
            mlir::isa<mlir::IntegerType, mlir::FloatType>(merkType)) &&
        !(term && term.getName().starts_with("process.mailbox("))) {
      messages_.Say(at, "code generation for %s"_todo_en_US,
          "a process kind that runs on a GPU and takes a value of type " +
              semantics::TypeToString(*type) +
              ": its start takes integers, floats, characters and "
              "mailboxes");
      ok = false;
      continue;
    }
    values.push_back(merkType);
  }
  if (!ok) {
    return nullptr;
  }
  auto i64{builder_.getI64Type()};
  mlir::OpBuilder::InsertionGuard guard{builder_};
  builder_.setInsertionPointToEnd(module_.getBody());
  auto start{
      merk::ProcOp::create(builder_, loc, ("merk_process_" + decl.name).str(),
          builder_.getFunctionType({i64}, {}), merk::Determinism::Det)};
  start.setPrivate();
  mlir::Block *entry{start.addEntryBlock()};
  builder_.setInsertionPointToStart(entry);
  auto generic{mlir::LLVM::LLVMPointerType::get(
      &context_, target_.addressSpaces().generic)};
  auto stack{mlir::LLVM::LLVMPointerType::get(
      &context_, target_.addressSpaces().private_)};
  mlir::Value words{mlir::LLVM::IntToPtrOp::create(
      builder_, loc, generic, entry->getArgument(0))};
  std::int32_t next{0};
  llvm::SmallVector<mlir::Value> arguments;
  for (mlir::Type type : values) {
    if (auto integer{mlir::dyn_cast<mlir::IntegerType>(type)}) {
      mlir::Value value{LoadWord(builder_, loc, words, next++)};
      if (integer.getWidth() < 64) {
        value = mlir::arith::TruncIOp::create(builder_, loc, integer, value);
      }
      arguments.push_back(value);
      continue;
    }
    if (auto floating{mlir::dyn_cast<mlir::FloatType>(type)}) {
      // The float's bits, in the low bits of its word.
      mlir::Value value{LoadWord(builder_, loc, words, next++)};
      auto bits{builder_.getIntegerType(floating.getWidth())};
      if (floating.getWidth() < 64) {
        value = mlir::arith::TruncIOp::create(builder_, loc, bits, value);
      }
      arguments.push_back(
          mlir::arith::BitcastOp::create(builder_, loc, floating, value));
      continue;
    }
    auto cellType{mlir::LLVM::LLVMArrayType::get(i64, kMailboxWords)};
    mlir::Value cell{mlir::LLVM::AllocaOp::create(
        builder_, loc, stack, cellType, IntegerConstant(i64, 1, loc))};
    if (stack != generic) {
      cell = mlir::LLVM::AddrSpaceCastOp::create(builder_, loc, generic, cell);
    }
    for (std::int32_t i{0}; i < static_cast<std::int32_t>(kMailboxWords); ++i) {
      mlir::Value address{mlir::LLVM::GEPOp::create(builder_, loc, generic, i64,
          cell, llvm::ArrayRef<mlir::LLVM::GEPArg>{i})};
      mlir::LLVM::StoreOp::create(
          builder_, loc, LoadWord(builder_, loc, words, next++), address);
    }
    arguments.push_back(
        mlir::UnrealizedConversionCastOp::create(builder_, loc, type, cell)
            .getResult(0));
  }
  merk::CallOp::create(
      builder_, loc, mlir::TypeRange{}, kind.getSymName(), arguments);
  merk::SucceedOp::create(builder_, loc, mlir::ValueRange{});
  return start;
}

// A loop under distribute parallel do in a GPU process's code
// (docs/INPROGRESS.md, GPU processes that run loops): the process's wave
// computes the bounds and the trip count, puts the body procedure's inputs,
// but the counter, in the words of its control block, a word each, and calls
// merk_device_process_loop with the loop's number, which starts its worker
// teams on the iterations and returns when they have run them all. Each
// worker thread runs its iterations through merk_process_loop
// (GenerateProcessLoops). By the directive's promise, an array that the body
// updates in place gives each iteration its own elements, so after the loop
// it is the array given, as for a parallel do.
void MerkGenerator::ProcessLoop(
    const ParallelLoop &loop, const ProcedureBody &body, mlir::Location loc) {
  for (const Body *g : loop.goals) {
    if (g != loop.trip && g != loop.call) {
      Goal(*g, body);
    }
  }
  if (failed_) {
    return;
  }
  BodyVar big{loop.trip->goals.front()->arguments[0]};
  BodyVar small{loop.trip->goals.front()->arguments[1]};
  std::int64_t step{loop.translated->step};
  mlir::Value lower{ValueOf(step > 0 ? small : big)};
  mlir::Value upper{ValueOf(step > 0 ? big : small)};
  llvm::SmallVector<const semantics::Type *> types;
  for (BodyVar argument : loop.bodyCall->arguments) {
    types.push_back(loop.loopBody.vars[argument].type);
  }
  merk::ProcOp proc{ProcFor(loop.bodyProcedure, types, loop.at)};
  if (!proc) {
    return;
  }
  if (const auto *hidden{
          polymorphism_.lookup(determinism_.BodyOf(loop.bodyProcedure))};
      hidden && !hidden->hidden.empty()) {
    NotYet(loop.at,
        "loops across a GPU process's teams in polymorphic "
        "procedures");
    return;
  }
  // The body procedure's inputs: the counter, and the values the loop
  // procedure was called with for its other arguments.
  ProcessLoopCase entry{proc.getSymName().str(), {}, {}, 0, {}};
  llvm::SmallVector<mlir::Value> words;
  // The reductions' values before the loop, in the order of their inputs.
  llvm::SmallVector<mlir::Value> before;
  llvm::SmallVector<const LoopReduction *> reduced;
  bool counted{false};
  for (auto [argument, mode] :
      llvm::zip(loop.bodyCall->arguments, loop.bodyProcedure.modes)) {
    std::optional<Represented> represented{
        Represent(loop.loopBody.vars[argument].type)};
    if (!represented || represented->dummy || IsOutput(mode)) {
      continue;
    }
    auto position{llvm::find(loop.loopBody.head, argument)};
    if (position == loop.loopBody.head.end()) {
      NotYet(loop.at,
          "parallel do loops whose body takes a value the loop computes");
      return;
    }
    std::size_t k(position - loop.loopBody.head.begin());
    const LoopReduction *reduction{nullptr};
    for (const LoopReduction &r : loop.reductions) {
      if (r.value->in == k) {
        reduction = &r;
      }
    }
    if (k == 0) {
      entry.counter = static_cast<unsigned>(entry.inputs.size());
      counted = true;
    } else if (reduction) {
      entry.reductions.push_back({static_cast<unsigned>(entry.inputs.size()),
          reduction->output, reduction->op, reduction->type});
      before.push_back(ValueOf(loop.call->arguments[k]));
      reduced.push_back(reduction);
    } else {
      words.push_back(ValueOf(loop.call->arguments[k]));
    }
    entry.inputs.push_back(represented->type);
  }
  if (entry.reductions.size() != loop.reductions.size()) {
    NotYet(loop.at,
        "reductions in a loop across a GPU process's teams whose body does "
        "not take the reduction's value");
    return;
  }
  if (entry.reductions.size() > kProcessLoopReductions) {
    NotYet(loop.at,
        "loops across a GPU process's teams with more than 8 reductions");
    return;
  }
  if (!counted || !entry.inputs[entry.counter].isInteger(64)) {
    NotYet(loop.at,
        "loops across a GPU process's teams whose body does not "
        "take the counter as an int");
    return;
  }
  if (words.size() > kProcessLoopWords) {
    NotYet(loop.at,
        "loops across a GPU process's teams whose body takes "
        "more than 32 values besides the counter");
    return;
  }
  for (mlir::Type result : proc.getFunctionType().getResults()) {
    entry.results.push_back(result);
  }
  auto i64{builder_.getI64Type()};
  auto generic{mlir::LLVM::LLVMPointerType::get(
      &context_, target_.addressSpaces().generic)};
  if (!words.empty()) {
    mlir::Value values{
        merk::ForeignCallOp::create(builder_, loc, mlir::TypeRange{generic},
            "merk_device_loop_values", mlir::ValueRange{}, /*can_fail=*/false)
            .getResult(0)};
    for (auto [index, value] : llvm::enumerate(words)) {
      mlir::Value word{merk::FromWordOp::create(builder_, loc, i64,
          merk::ToWordOp::create(
              builder_, loc, merk::WordType::get(&context_), value))};
      mlir::Value address{
          mlir::LLVM::GEPOp::create(builder_, loc, generic, i64, values,
              llvm::ArrayRef<mlir::LLVM::GEPArg>{
                  static_cast<std::int32_t>(index)})};
      mlir::LLVM::StoreOp::create(builder_, loc, word, address);
    }
  }
  // The reductions' values before the loop, which the runtime combines with
  // the worker teams' partial results into their results.
  mlir::Value results;
  if (!before.empty()) {
    results =
        merk::ForeignCallOp::create(builder_, loc, mlir::TypeRange{generic},
            "merk_device_loop_results", mlir::ValueRange{}, /*can_fail=*/false)
            .getResult(0);
    for (auto [index, value] : llvm::enumerate(before)) {
      StoreWord(builder_, loc, results, static_cast<std::int32_t>(index),
          WordBits(builder_, loc, value));
    }
  }
  // The trip count: from lower to upper by the step, inclusive, or none.
  mlir::Value distance{step > 0
          ? mlir::arith::SubIOp::create(builder_, loc, upper, lower)
          : mlir::arith::SubIOp::create(builder_, loc, lower, upper)};
  mlir::Value zero{IntegerConstant(i64, 0, loc)};
  mlir::Value trips{mlir::arith::AddIOp::create(builder_, loc,
      mlir::arith::DivSIOp::create(builder_, loc, distance,
          IntegerConstant(i64, step > 0 ? step : -step, loc)),
      IntegerConstant(i64, 1, loc))};
  mlir::Value none{mlir::arith::CmpIOp::create(
      builder_, loc, mlir::arith::CmpIPredicate::slt, distance, zero)};
  mlir::Value count{
      mlir::arith::SelectOp::create(builder_, loc, none, zero, trips)};
  merk::ForeignCallOp::create(builder_, loc, mlir::TypeRange{},
      "merk_device_process_loop",
      mlir::ValueRange{
          IntegerConstant(
              i64, static_cast<std::int64_t>(processLoops_.size()), loc),
          lower, IntegerConstant(i64, step, loc), count,
          IntegerConstant(i64, static_cast<std::int64_t>(before.size()), loc)},
      /*can_fail=*/false);
  processLoops_.push_back(std::move(entry));
  for (const semantics::TranslatedLoop::Threaded &value :
      loop.translated->threaded) {
    values_[loop.call->arguments[value.after]] =
        ValueOf(loop.call->arguments[value.in]);
  }
  for (auto [index, reduction] : llvm::enumerate(reduced)) {
    values_[loop.call->arguments[reduction->value->after]] =
        FromWordBits(builder_, loc, reduction->type,
            LoadWord(builder_, loc, results, static_cast<std::int32_t>(index)));
  }
}

// The procedures that the device's runtime calls for the loops that GPU
// processes run across their teams (ProcessLoop), each a switch over the
// module's loops by their number. Every image of GPU processes has them, as
// the runtime calls them, even with no loops.
// - merk_process_loop(loop, start, rounds, stride, count, first, step,
//   values, partials), which each thread of a process's worker teams calls
//   once for each loop, for its share of the iterations: the loop's rounds
//   (ProcessLoopRounds), with the values from the control block's words, at
//   `values`, and the thread's partial results, at `partials`, which it
//   stores back after the last round.
// - merk_process_loop_start(loop, partials): the reductions' identities.
// - merk_process_loop_reduce(loop, into, from): the partial results at
//   `from` combined into those at `into`, by each reduction's operator.
void MerkGenerator::GenerateProcessLoops() {
  if (!target_.options().processKernels) {
    return;
  }
  mlir::Location loc{builder_.getUnknownLoc()};
  auto i64{builder_.getI64Type()};
  auto generic{mlir::LLVM::LLVMPointerType::get(
      &context_, target_.addressSpaces().generic)};
  mlir::OpBuilder::InsertionGuard guard{builder_};
  for (unsigned which{0}; which < 3; ++which) {
    builder_.setInsertionPointToEnd(module_.getBody());
    llvm::StringRef name{which == 0 ? "merk_process_loop"
            : which == 1            ? "merk_process_loop_start"
                                    : "merk_process_loop_reduce"};
    // The partial results come as pointers, not words, so that, inlined
    // into the runtime's combine, they can stay in registers.
    llvm::SmallVector<mlir::Type> parameters{i64};
    parameters.append(which == 0 ? 8 : which,
        which == 0 ? mlir::Type{i64} : mlir::Type{generic});
    auto proc{merk::ProcOp::create(builder_, loc, name,
        builder_.getFunctionType(parameters, {}), merk::Determinism::Det)};
    proc->setAttr(kProcessRootAttrName, builder_.getUnitAttr());
    mlir::Block *entry{proc.addEntryBlock()};
    builder_.setInsertionPointToStart(entry);
    if (processLoops_.empty()) {
      merk::SucceedOp::create(builder_, loc, mlir::ValueRange{});
      continue;
    }
    llvm::SmallVector<mlir::Attribute> cases;
    for (std::size_t number{0}; number < processLoops_.size(); ++number) {
      cases.push_back(builder_.getArrayAttr(
          {builder_.getI64IntegerAttr(static_cast<std::int64_t>(number))}));
    }
    auto op{merk::SwitchOp::create(builder_, loc, mlir::TypeRange{},
        entry->getArgument(0), builder_.getArrayAttr(cases),
        /*canFail=*/false)};
    for (auto [region, loop] :
        llvm::zip_equal(op.getCaseRegions(), processLoops_)) {
      builder_.setInsertionPointToStart(&region.emplaceBlock());
      if (which == 0) {
        ProcessLoopShare(loop, entry, generic, loc);
      } else if (which == 1) {
        mlir::Value partials{entry->getArgument(1)};
        for (auto [index, reduction] : llvm::enumerate(loop.reductions)) {
          StoreWord(builder_, loc, partials, static_cast<std::int32_t>(index),
              WordBits(builder_, loc,
                  ReductionIdentity(reduction.op, reduction.type, loc)));
        }
      } else {
        mlir::Value into{entry->getArgument(1)};
        mlir::Value from{entry->getArgument(2)};
        for (auto [index, reduction] : llvm::enumerate(loop.reductions)) {
          auto at{static_cast<std::int32_t>(index)};
          mlir::Value a{FromWordBits(builder_, loc, reduction.type,
              LoadWord(builder_, loc, into, at))};
          mlir::Value b{FromWordBits(builder_, loc, reduction.type,
              LoadWord(builder_, loc, from, at))};
          StoreWord(builder_, loc, into, at,
              WordBits(
                  builder_, loc, ReductionCombine(reduction.op, a, b, loc)));
        }
      }
      merk::YieldOp::create(builder_, loc, mlir::ValueRange{});
    }
    builder_.setInsertionPointAfter(op);
    merk::SucceedOp::create(builder_, loc, mlir::ValueRange{});
  }
}

// A thread's share of the loop, in merk_process_loop's case for it, whose
// arguments `entry` has (GenerateProcessLoops): the loop's words and the
// thread's partial results loaded once, its rounds, and the partial results
// stored back.
void MerkGenerator::ProcessLoopShare(const ProcessLoopCase &loop,
    mlir::Block *entry, mlir::Type generic, mlir::Location loc) {
  mlir::Value values{mlir::LLVM::IntToPtrOp::create(
      builder_, loc, generic, entry->getArgument(7))};
  mlir::Value partials{mlir::LLVM::IntToPtrOp::create(
      builder_, loc, generic, entry->getArgument(8))};
  // The rounds' arguments: the rounds, the first iteration, the stride, the
  // count, the first counter and the step, then the body's inputs but the
  // counter and the reductions' values, then those.
  llvm::SmallVector<mlir::Value> arguments{entry->getArgument(2),
      entry->getArgument(1), entry->getArgument(3), entry->getArgument(4),
      entry->getArgument(5), entry->getArgument(6)};
  std::int32_t next{0};
  for (auto [index, type] : llvm::enumerate(loop.inputs)) {
    if (index == loop.counter || IsReductionInput(loop, index)) {
      continue;
    }
    arguments.push_back(FromWordBits(
        builder_, loc, type, LoadWord(builder_, loc, values, next++)));
  }
  for (auto [index, reduction] : llvm::enumerate(loop.reductions)) {
    arguments.push_back(FromWordBits(builder_, loc, reduction.type,
        LoadWord(builder_, loc, partials, static_cast<std::int32_t>(index))));
  }
  merk::ProcOp rounds{ProcessLoopRounds(loop, loc)};
  auto call{merk::CallOp::create(builder_, loc,
      rounds.getFunctionType().getResults(), rounds.getSymName(), arguments)};
  for (auto [index, result] : llvm::enumerate(call.getResults())) {
    StoreWord(builder_, loc, partials, static_cast<std::int32_t>(index),
        WordBits(builder_, loc, result));
  }
}

// Whether the body's input at `index` is a reduction's value.
bool MerkGenerator::IsReductionInput(
    const ProcessLoopCase &loop, std::size_t index) {
  return llvm::any_of(loop.reductions,
      [&](const ProcessLoopReduction &r) { return r.input == index; });
}

// The rounds of a thread's share of the loop, a procedure of their own that
// calls itself for the next round, which becomes a loop:
//
//   rounds(Rounds, I, Stride, Count, First, Step, Values..., Partials...) =
//     if Rounds > 0 then
//       Next = (if I < Count then the body's reductions' next values, with
//         the counter First + I * Step, else Partials),
//       a barrier across the team,
//       rounds(Rounds - 1, I + Stride, ..., Next...)
//     else Partials
//
// Every thread of a team goes round as often, so all reach each barrier.
merk::ProcOp MerkGenerator::ProcessLoopRounds(
    const ProcessLoopCase &loop, mlir::Location loc) {
  auto i64{builder_.getI64Type()};
  llvm::SmallVector<mlir::Type> parameters(6, i64);
  llvm::SmallVector<mlir::Type> results;
  for (auto [index, type] : llvm::enumerate(loop.inputs)) {
    if (index != loop.counter && !IsReductionInput(loop, index)) {
      parameters.push_back(type);
    }
  }
  for (const ProcessLoopReduction &reduction : loop.reductions) {
    parameters.push_back(reduction.type);
    results.push_back(reduction.type);
  }
  // A loop of the kind's code and of its variant for its words calls the
  // same body, with the same rounds.
  std::string name{loop.body + "$rounds"};
  if (auto made{module_.lookupSymbol<merk::ProcOp>(name)}) {
    return made;
  }
  mlir::OpBuilder::InsertionGuard guard{builder_};
  builder_.setInsertionPointToEnd(module_.getBody());
  auto proc{merk::ProcOp::create(builder_, loc, name,
      builder_.getFunctionType(parameters, results), merk::Determinism::Det)};
  proc.setPrivate();
  mlir::Block *entry{proc.addEntryBlock()};
  builder_.setInsertionPointToStart(entry);
  mlir::Value remaining{entry->getArgument(0)};
  mlir::Value i{entry->getArgument(1)};
  mlir::Value stride{entry->getArgument(2)};
  mlir::Value count{entry->getArgument(3)};
  mlir::Value first{entry->getArgument(4)};
  mlir::Value step{entry->getArgument(5)};
  std::size_t words{parameters.size() - 6 - loop.reductions.size()};
  auto values{entry->getArguments().slice(6, words)};
  auto partials{entry->getArguments().slice(6 + words)};
  mlir::Value zero{IntegerConstant(i64, 0, loc)};
  auto more{merk::IteOp::create(builder_, loc, results)};
  builder_.setInsertionPointToStart(&more.getCondition().emplaceBlock());
  merk::GuardOp::create(builder_, loc,
      mlir::arith::CmpIOp::create(
          builder_, loc, mlir::arith::CmpIPredicate::sgt, remaining, zero));
  merk::YieldOp::create(builder_, loc, mlir::ValueRange{});
  builder_.setInsertionPointToStart(&more.getThenRegion().emplaceBlock());
  // This round's iteration, if the thread has one.
  auto iteration{merk::IteOp::create(builder_, loc, results)};
  {
    mlir::OpBuilder::InsertionGuard roundGuard{builder_};
    builder_.setInsertionPointToStart(&iteration.getCondition().emplaceBlock());
    merk::GuardOp::create(builder_, loc,
        mlir::arith::CmpIOp::create(
            builder_, loc, mlir::arith::CmpIPredicate::slt, i, count));
    merk::YieldOp::create(builder_, loc, mlir::ValueRange{});
    builder_.setInsertionPointToStart(
        &iteration.getThenRegion().emplaceBlock());
    llvm::SmallVector<mlir::Value> inputs;
    std::size_t nextValue{0};
    for (auto [index, type] : llvm::enumerate(loop.inputs)) {
      if (index == loop.counter) {
        inputs.push_back(mlir::arith::AddIOp::create(builder_, loc, first,
            mlir::arith::MulIOp::create(builder_, loc, i, step)));
        continue;
      }
      auto reduction{llvm::find_if(loop.reductions,
          [&](const ProcessLoopReduction &r) { return r.input == index; })};
      inputs.push_back(reduction != loop.reductions.end()
              ? partials[reduction - loop.reductions.begin()]
              : values[nextValue++]);
    }
    auto body{
        merk::CallOp::create(builder_, loc, loop.results, loop.body, inputs)};
    llvm::SmallVector<mlir::Value> next;
    for (const ProcessLoopReduction &reduction : loop.reductions) {
      next.push_back(body->getResult(reduction.output));
    }
    merk::YieldOp::create(builder_, loc, next);
    builder_.setInsertionPointToStart(
        &iteration.getElseRegion().emplaceBlock());
    merk::YieldOp::create(builder_, loc, partials);
  }
  merk::ForeignCallOp::create(builder_, loc, mlir::TypeRange{},
      "merk_device_sync_threads", mlir::ValueRange{}, /*can_fail=*/false);
  llvm::SmallVector<mlir::Value> arguments{
      mlir::arith::SubIOp::create(
          builder_, loc, remaining, IntegerConstant(i64, 1, loc)),
      mlir::arith::AddIOp::create(builder_, loc, i, stride), stride, count,
      first, step};
  arguments.append(values.begin(), values.end());
  arguments.append(
      iteration.getResults().begin(), iteration.getResults().end());
  auto again{merk::CallOp::create(builder_, loc, results, name, arguments)};
  merk::YieldOp::create(builder_, loc, again.getResults());
  builder_.setInsertionPointToStart(&more.getElseRegion().emplaceBlock());
  merk::YieldOp::create(builder_, loc, partials);
  builder_.setInsertionPointAfter(more);
  merk::SucceedOp::create(builder_, loc, more.getResults());
  return proc;
}

// Whether the kind's code, or a procedure of the module that it calls, has a
// loop under distribute parallel do, as check-gpu-code follows them.
bool MerkGenerator::KindRunsLoops(const parser::ProcessDecl &decl) {
  const Procedure *procedure{KindProcedure(decl)};
  if (!procedure) {
    return false;
  }
  llvm::DenseSet<const ProcedureBody *> visited;
  llvm::SmallVector<const ProcedureBody *> pending;
  llvm::SmallVector<const Body *> goals;
  if (const ProcedureBody *body{determinism_.BodyOf(*procedure)}) {
    pending.push_back(body);
  }
  while (!pending.empty()) {
    const ProcedureBody *body{pending.pop_back_val()};
    if (!visited.insert(body).second || !body->body) {
      continue;
    }
    goals.assign({body->body});
    while (!goals.empty()) {
      const Body *goal{goals.pop_back_val()};
      if (goal->kind == BodyKind::Scope && goal->directive &&
          goal->directive->id ==
              llvm::omp::Directive::OMPD_distribute_parallel_do) {
        return true;
      }
      if ((goal->kind == BodyKind::Call ||
              (goal->kind == BodyKind::Unify &&
                  goal->unify == semantics::UnifyKind::Call)) &&
          goal->modes && goal->modes->callee &&
          goal->modes->callee->symbol->module == moduleName_) {
        if (const ProcedureBody *called{
                determinism_.BodyOf(*goal->modes->callee)}) {
          pending.push_back(called);
        }
      }
      if (goal->lambda) {
        goals.push_back(goal->lambda);
      }
      for (const semantics::SwitchCase &arm : goal->cases) {
        goals.push_back(arm.goal);
      }
      goals.append(goal->goals.begin(), goal->goals.end());
    }
  }
  return false;
}

// A global of the module with `bytes`, constant and internal.
static mlir::LLVM::GlobalOp BytesGlobal(mlir::OpBuilder &builder,
    mlir::Location loc, llvm::StringRef name, llvm::StringRef bytes,
    std::uint64_t alignment) {
  auto type{mlir::LLVM::LLVMArrayType::get(
      builder.getI8Type(), static_cast<unsigned>(bytes.size()))};
  auto global{mlir::LLVM::GlobalOp::create(builder, loc, type,
      /*isConstant=*/true, mlir::LLVM::Linkage::Internal, name,
      builder.getStringAttr(bytes))};
  global.setAlignment(alignment);
  return global;
}

// The bytes of stack a lane that the calls of the image's kernel take at
// most, outside any recursion, from its AMDGPU metadata, if its stack is
// not fixed, which the runtime checks against the stack that it launches the
// kernel with (runtime/gpu-processes.cpp); or 0 for a kernel whose stack
// size is fixed, which the plugin launches with its own. An image without
// the metadata, which the runtime cannot load either, has 0 too.
static std::uint64_t StackFrames(llvm::StringRef image) {
  llvm::Expected<std::unique_ptr<llvm::object::ObjectFile>> object{
      llvm::object::ObjectFile::createObjectFile(
          llvm::MemoryBufferRef{image, "image"})};
  if (!object) {
    llvm::consumeError(object.takeError());
    return 0;
  }
  auto *elf{llvm::dyn_cast<llvm::object::ELF64LEObjectFile>(object->get())};
  if (!elf) {
    return 0;
  }
  const auto &file{elf->getELFFile()};
  auto headers{file.program_headers()};
  if (!headers) {
    llvm::consumeError(headers.takeError());
    return 0;
  }
  for (const auto &header : *headers) {
    if (header.p_type != llvm::ELF::PT_NOTE) {
      continue;
    }
    llvm::Error error{llvm::Error::success()};
    for (const auto &note : file.notes(header, error)) {
      if (note.getName() != "AMDGPU" ||
          note.getType() != llvm::ELF::NT_AMDGPU_METADATA) {
        continue;
      }
      auto desc{note.getDesc(header.p_align)};
      llvm::msgpack::Document document;
      if (!document.readFromBlob(
              llvm::StringRef{
                  reinterpret_cast<const char *>(desc.data()), desc.size()},
              /*Multi=*/false)) {
        continue;
      }
      llvm::msgpack::DocNode &kernels{
          document.getRoot().getMap(true)["amdhsa.kernels"]};
      if (!kernels.isArray()) {
        continue;
      }
      for (llvm::msgpack::DocNode &kernel : kernels.getArray()) {
        if (!kernel.isMap()) {
          continue;
        }
        llvm::msgpack::MapDocNode &fields{kernel.getMap()};
        llvm::msgpack::DocNode &name{fields[".name"]};
        if (!name.isString() || name.getString() != "merk_place") {
          continue;
        }
        llvm::msgpack::DocNode &dynamic{fields[".uses_dynamic_stack"]};
        llvm::msgpack::DocNode &frames{fields[".private_segment_fixed_size"]};
        if (dynamic.getKind() != llvm::msgpack::Type::Boolean ||
            !dynamic.getBool() ||
            frames.getKind() != llvm::msgpack::Type::UInt) {
          llvm::consumeError(std::move(error));
          return 0;
        }
        std::uint64_t bytes{frames.getUInt()};
        llvm::consumeError(std::move(error));
        return bytes;
      }
    }
    llvm::consumeError(std::move(error));
  }
  return 0;
}

// The host's compile of a module with kinds that run on a GPU embeds the
// image of their kernels for each GPU that -fembed-process-image= names,
// as gpu:file, and registers each at start-up, with the module's name, the
// GPU's and its kinds' names, in their order, separated by spaces, and the
// stack its kernel's calls take (StackFrames), in a
// procedure that a constructor calls, as message types are registered. The
// runtime loads an image when a process is first spawned on a GPU of its kind
// (step 4c). False after reporting an image that cannot be read.
bool MerkGenerator::RegisterProcessImages() {
  if (gpuKinds_.empty() || target_.options().IsDevice() ||
      target_.options().processImages.empty()) {
    return true;
  }
  mlir::Location loc{builder_.getUnknownLoc()};
  struct Image {
    std::string gpu;
    std::unique_ptr<llvm::MemoryBuffer> bytes;
  };
  llvm::SmallVector<Image> images;
  for (llvm::StringRef option : target_.options().processImages) {
    auto [gpu, file]{option.split(':')};
    if (gpu.empty() || file.empty()) {
      mlir::emitError(loc) << "-fembed-process-image=" << option
                           << ": expected the GPU and the image's file, as "
                              "gfx1030:file";
      return false;
    }
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> bytes{
        llvm::MemoryBuffer::getFile(file, /*IsText=*/false,
            /*RequiresNullTerminator=*/false)};
    if (!bytes) {
      mlir::emitError(loc) << "cannot read the GPU processes' image " << file
                           << ": " << bytes.getError().message();
      return false;
    }
    images.push_back({gpu.str(), std::move(*bytes)});
  }
  mlir::OpBuilder::InsertionGuard guard{builder_};
  builder_.setInsertionPointToEnd(module_.getBody());
  std::string prefix{moduleName_.str() + ".merk_process_image"};
  // C strings, and the image aligned as an ELF file's headers must be.
  mlir::LLVM::GlobalOp name{BytesGlobal(builder_, loc, prefix + ".module",
      (moduleName_ + llvm::StringRef("\0", 1)).str(), 1)};
  std::string kindNames;
  for (const parser::ProcessDecl *decl : gpuKinds_) {
    kindNames += (kindNames.empty() ? "" : " ") + decl->name.str();
  }
  mlir::LLVM::GlobalOp kinds{BytesGlobal(
      builder_, loc, prefix + ".kinds", kindNames + std::string(1, '\0'), 1)};
  // For each kind, in the same order, 1 if it runs loops across teams, so
  // that its spawn gives it worker teams, else 0.
  std::string loopFlags;
  for (const parser::ProcessDecl *decl : gpuKinds_) {
    loopFlags += KindRunsLoops(*decl) ? '1' : '0';
  }
  mlir::LLVM::GlobalOp loops{BytesGlobal(
      builder_, loc, prefix + ".loops", loopFlags + std::string(1, '\0'), 1)};
  llvm::SmallVector<std::pair<mlir::LLVM::GlobalOp, mlir::LLVM::GlobalOp>>
      globals;
  for (const Image &image : images) {
    std::string stem{prefix + "." + image.gpu};
    globals.push_back({BytesGlobal(builder_, loc, stem + ".gpu",
                           image.gpu + std::string(1, '\0'), 1),
        BytesGlobal(builder_, loc, stem, image.bytes->getBuffer(), 16)});
  }
  std::string procName{moduleName_.str() + ".merk_register_process_images"};
  auto proc{merk::ProcOp::create(builder_, loc, procName,
      builder_.getFunctionType({}, {}), merk::Determinism::Det)};
  proc.setPrivate();
  builder_.setInsertionPointToStart(proc.addEntryBlock());
  auto i64{builder_.getI64Type()};
  for (auto [image, globalPair] : llvm::zip_equal(images, globals)) {
    auto [gpu, bytes]{globalPair};
    merk::ForeignCallOp::create(builder_, loc, mlir::TypeRange{},
        "merk_register_process_image",
        mlir::ValueRange{mlir::LLVM::AddressOfOp::create(builder_, loc, name),
            mlir::LLVM::AddressOfOp::create(builder_, loc, gpu),
            mlir::LLVM::AddressOfOp::create(builder_, loc, bytes),
            IntegerConstant(i64,
                static_cast<std::int64_t>(image.bytes->getBufferSize()), loc),
            mlir::LLVM::AddressOfOp::create(builder_, loc, kinds),
            IntegerConstant(i64,
                static_cast<std::int64_t>(
                    StackFrames(image.bytes->getBuffer())),
                loc),
            mlir::LLVM::AddressOfOp::create(builder_, loc, loops)},
        /*can_fail=*/false);
  }
  merk::SucceedOp::create(builder_, loc, mlir::ValueRange{});
  builder_.setInsertionPointToEnd(module_.getBody());
  merk::ConstructorOp::create(builder_, loc, procName);
  return true;
}

} // namespace mlirgen
} // namespace merkurius::codegen
