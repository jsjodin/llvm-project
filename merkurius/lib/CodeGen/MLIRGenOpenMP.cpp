//===-- lib/CodeGen/MLIRGenOpenMP.cpp - OpenMP ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// OpenMP directives: parallel loops and their reductions, target data and
// target regions.
//
//===----------------------------------------------------------------------===//

#include "MLIRGenImpl.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/Support/xxhash.h"

namespace merkurius::codegen::mlirgen {

mlir::Value MerkGenerator::Slot(mlir::Type type, mlir::Location loc) {
  auto proc{builder_.getInsertionBlock()->getParent()
                ->getParentOfType<merk::ProcOp>()};
  mlir::OpBuilder::InsertionGuard guard{builder_};
  // In a target region, in the region: otherwise it would be another of the
  // kernel's arguments, which the host and device compiles must agree on.
  mlir::omp::TargetOp target{EnclosingTarget()};
  builder_.setInsertionPointToStart(
      target ? &target.getRegion().front() : &proc.getBody().front());
  unsigned addressSpace{target_.allocaAddressSpace()};
  mlir::Value one{mlir::LLVM::ConstantOp::create(
      builder_, loc, builder_.getI64Type(), builder_.getI64IntegerAttr(1))};
  mlir::Value slot{mlir::LLVM::AllocaOp::create(builder_, loc,
      mlir::LLVM::LLVMPointerType::get(&context_, addressSpace), type, one)};
  if (addressSpace != 0) {
    // On a GPU the stack is an address space of its own.
    slot = mlir::LLVM::AddrSpaceCastOp::create(
        builder_, loc, mlir::LLVM::LLVMPointerType::get(&context_), slot);
  }
  return slot;
}

// The scalars, integers and floats, arrays of them, and terms whose cells
// hold only scalars can cross into a target region so far; a region gives
// only scalars and arrays of them so far.
static bool Scalar(mlir::Type type) {
  return mlir::isa<mlir::IntegerType, mlir::FloatType>(type);
}

// A pointer the region passes on without reading it, such as a type_info
// of the host's that a send gives back to the host.
static bool Passed(mlir::Type type) {
  return mlir::isa<mlir::LLVM::LLVMPointerType>(type);
}

static bool Givable(mlir::Type type) {
  if (auto array{mlir::dyn_cast<merk::ArrayType>(type)}) {
    return Scalar(array.getElement());
  }
  return Scalar(type);
}

// A mailbox handle is process.mailbox's cell of five words, the node's id
// in two, the place and slot, the generation and the secret
// (library/process.m): the type is abstract outside the library, so its
// layout is known here, as an array's is.
static constexpr unsigned mailboxWords{5};

std::optional<unsigned> MerkGenerator::FlatCell(mlir::Type type) {
  auto term{mlir::dyn_cast<merk::TermType>(type)};
  if (!term) {
    return std::nullopt;
  }
  if (term.getName().starts_with("process.mailbox(")) {
    return mailboxWords;
  }
  for (auto def : module_.getOps<merk::TypeDefOp>()) {
    if (def.getName() != term.getName()) {
      continue;
    }
    // One constructor of one argument is the argument, with no cell.
    if (def.getConstructors().size() != 1) {
      return std::nullopt;
    }
    auto constructor{mlir::cast<merk::ConstructorAttr>(def.getConstructors()[0])};
    llvm::ArrayRef<mlir::Type> fields{constructor.getFields()};
    if (constructor.getHidden() != 0 || fields.size() < 2 ||
        !llvm::all_of(fields, Scalar)) {
      return std::nullopt;
    }
    return static_cast<unsigned>(fields.size());
  }
  return std::nullopt;
}

bool MerkGenerator::Mappable(mlir::Type type) {
  return Givable(type) || Passed(type) || FlatCell(type);
}

mlir::omp::TargetOp MerkGenerator::EnclosingTarget() {
  mlir::Operation *op{builder_.getInsertionBlock()->getParentOp()};
  if (auto target{mlir::dyn_cast<mlir::omp::TargetOp>(op)}) {
    return target;
  }
  return op->getParentOfType<mlir::omp::TargetOp>();
}

// Whether a message of the type is one word that the device sends as it
// is: an integer, a float, a character, or a constant of a type of
// constants only (docs/INPROGRESS.md, The device's side of the first
// slice, Decision 3).
bool MerkGenerator::WordMessage(mlir::Type type) {
  if (Scalar(type)) {
    return true;
  }
  auto term{mlir::dyn_cast<merk::TermType>(type)};
  if (!term) {
    return false;
  }
  for (auto def : module_.getOps<merk::TypeDefOp>()) {
    if (def.getName() == term.getName()) {
      return llvm::all_of(def.getConstructors(), [](mlir::Attribute a) {
        return mlir::cast<merk::ConstructorAttr>(a).getFields().empty();
      });
    }
  }
  return false;
}

bool MerkGenerator::ArrayMessage(mlir::Type type) {
  auto array{mlir::dyn_cast<merk::ArrayType>(type)};
  return array && WordMessage(array.getElement());
}

// Whether a type has no type variables, so that its type_info is known at
// compile time.
static bool Ground(const semantics::Type &type) {
  if (type.kind == semantics::TypeKind::Variable ||
      type.kind == semantics::TypeKind::Existential) {
    return false;
  }
  if (type.result && !Ground(*type.result)) {
    return false;
  }
  return llvm::all_of(type.arguments,
      [](const semantics::Type *argument) { return Ground(*argument); });
}

// Whether a type, or one of its arguments, is a higher-order type.
static bool HasHigherOrder(const semantics::Type &type) {
  if (type.kind == semantics::TypeKind::HigherOrder) {
    return true;
  }
  return llvm::any_of(type.arguments, [](const semantics::Type *argument) {
    return HasHigherOrder(*argument);
  });
}

// docs/INPROGRESS.md, GPU processes without omp target, step 4e. A closure
// in a constructor's field is found when the runtime copies it.
bool MerkGenerator::CopiedMessage(
    const semantics::Type &type, mlir::Type represented) {
  return Ground(type) && !HasHigherOrder(type) && !WordMessage(represented) &&
      !ArrayMessage(represented);
}

mlir::Value MerkGenerator::MessageTypeInfo(
    const semantics::Type *type, mlir::Location loc, parser::CharBlock at) {
  mlir::Value info{TypeInfoOfType(type, loc, at)};
  if (!info) {
    return {};
  }
  return mlir::UnrealizedConversionCastOp::create(
      builder_, loc, mlir::LLVM::LLVMPointerType::get(&context_), info)
      .getResult(0);
}

std::optional<std::uint64_t> MerkGenerator::MessageKey(
    const semantics::Type &type, mlir::Type represented) {
  if (!Ground(type) ||
      (!WordMessage(represented) && !ArrayMessage(represented) &&
          !CopiedMessage(type, represented))) {
    return std::nullopt;
  }
  std::uint64_t key{llvm::xxh3_64bits(semantics::TypeToString(type))};
  messageTypes_.try_emplace(key, &type);
  return key;
}

// process.send in a target region, or in any procedure of the device's
// compile, all of which run on the device. It is a call of
// merk_device_send with the handle's cell, the message type's key and the
// message as a word, or of merk_device_send_array with the array's pointer
// if it is an array of one-word elements. The key is a hash of the type's name, which the
// host's compile registers with the host's type_info at start-up
// (RegisterMessageTypes), since the host encodes the message by it if the
// mailbox is another node's; so a procedure that a region calls sends as
// the region does (docs/INPROGRESS.md, The device's side of the first
// slice, step 4). Outside a region, a send that the device does not have
// is left to the call of process.send (false), as ReceiveInTarget leaves a
// receive.
bool MerkGenerator::SendInTarget(const Body &goal,
    llvm::ArrayRef<BodyVar> arguments, const ProcedureBody &body,
    bool inRegion) {
  parser::CharBlock at{Where(goal.tokens)};
  mlir::Location loc{Location(goal.tokens)};
  const semantics::Type *type{body.vars[arguments[1]].type};
  mlir::Value mailbox{ValueOf(arguments[0])};
  mlir::Value message{ValueOf(arguments[1])};
  if (!mailbox || !message) {
    failed_ = true;
    return true;
  }
  std::optional<std::uint64_t> key{MessageKey(*type, message.getType())};
  if (!key) {
    if (!inRegion) {
      return false;
    }
    NotYet(at,
        "sending messages of type " + semantics::TypeToString(*type) +
            " in target regions");
    return true;
  }
  bool array{ArrayMessage(message.getType())};
  mlir::Value word{array ? ArrayPointer(message, loc)
                         : mlir::Value{merk::ToWordOp::create(builder_, loc,
                               merk::WordType::get(&context_), message)}};
  llvm::SmallVector<mlir::Value> operands{mailbox,
      IntegerConstant(
          builder_.getI64Type(), static_cast<std::int64_t>(*key), loc),
      word};
  bool copied{CopiedMessage(*type, message.getType())};
  if (copied) {
    mlir::Value info{MessageTypeInfo(type, loc, at)};
    if (!info) {
      failed_ = true;
      return true;
    }
    operands.push_back(info);
  }
  merk::ForeignCallOp::create(builder_, loc, mlir::TypeRange{},
      array        ? "merk_device_send_array"
          : copied ? "merk_device_send_term"
                   : "merk_device_send",
      operands, /*can_fail=*/false);
  return true;
}

// The host's compile of a module with target regions, or with process kinds
// that run on a GPU, registers the key of each message type that its device
// code may send with the type's type_info, in a procedure that runs at
// start-up: every one-word, ground message type of the module's sends,
// since the host's compile does not know which procedures run on the
// device. The runtime checks that no two types have the same key.
void MerkGenerator::RegisterMessageTypes() {
  if (messageTypes_.empty() || target_.options().IsDevice()) {
    return;
  }
  bool device{!gpuKinds_.empty()};
  module_.walk([&](mlir::omp::TargetOp) { device = true; });
  if (!device) {
    return;
  }
  mlir::Location loc{builder_.getUnknownLoc()};
  mlir::OpBuilder::InsertionGuard guard{builder_};
  builder_.setInsertionPointToEnd(module_.getBody());
  std::string name{moduleName_.str() + ".merk_register_message_types"};
  auto proc{merk::ProcOp::create(builder_, loc, name,
      builder_.getFunctionType({}, {}), merk::Determinism::Det)};
  proc.setPrivate();
  builder_.setInsertionPointToStart(proc.addEntryBlock());
  // In the order of the keys, so that the module is the same each time.
  llvm::SmallVector<std::pair<std::uint64_t, const semantics::Type *>> types{
      messageTypes_.begin(), messageTypes_.end()};
  llvm::sort(types, llvm::less_first{});
  for (auto [key, type] : types) {
    mlir::Value info{TypeInfoOfType(type, loc, {})};
    if (!info) {
      failed_ = true;
      return;
    }
    merk::ForeignCallOp::create(builder_, loc, mlir::TypeRange{},
        "merk_register_message_type",
        mlir::ValueRange{
            IntegerConstant(builder_.getI64Type(),
                static_cast<std::int64_t>(key), loc),
            mlir::UnrealizedConversionCastOp::create(builder_, loc,
                mlir::LLVM::LLVMPointerType::get(&context_), info)
                .getResult(0)},
        /*can_fail=*/false);
  }
  merk::SucceedOp::create(builder_, loc, mlir::ValueRange{});
  builder_.setInsertionPointToEnd(module_.getBody());
  merk::ConstructorOp::create(builder_, loc, name);
}

// A process.timeout's milliseconds, -1 for infinity, as library/process.m's
// milliseconds has them.
mlir::Value MerkGenerator::TimeoutMilliseconds(
    mlir::Value timeout, mlir::Location loc) {
  auto i64{builder_.getI64Type()};
  auto cases{builder_.getArrayAttr(
      {builder_.getArrayAttr(
           {merk::FunctorAttr::get(&context_, "infinity", 0)}),
          builder_.getArrayAttr(
              {merk::FunctorAttr::get(&context_, "after", 1)})})};
  auto switchOp{merk::SwitchOp::create(builder_, loc, mlir::TypeRange{i64},
      timeout, cases, /*canFail=*/false)};
  mlir::OpBuilder::InsertionGuard guard{builder_};
  builder_.setInsertionPointToStart(&switchOp->getRegion(0).emplaceBlock());
  merk::YieldOp::create(builder_, loc, IntegerConstant(i64, -1, loc));
  builder_.setInsertionPointToStart(&switchOp->getRegion(1).emplaceBlock());
  auto deconstruct{merk::DeconstructOp::create(builder_, loc,
      mlir::TypeRange{i64}, timeout,
      merk::FunctorAttr::get(&context_, "after", 1))};
  mlir::Value milliseconds{mlir::arith::MaxSIOp::create(builder_, loc,
      deconstruct.getResult(0), IntegerConstant(i64, 0, loc))};
  merk::YieldOp::create(builder_, loc, milliseconds);
  return switchOp->getResult(0);
}

// receive(Mailbox, any, Timeout, Result, !IO) in a target region, or in
// any procedure of the device's compile, all of which run on the device:
// the only receive the device has so far (docs/INPROGRESS.md, The device's
// side of the first slice, Decision 4). It is a call of merk_device_receive
// with the handle's cell and the timeout's milliseconds, or of
// merk_device_receive_array for an array of one-word elements, which
// succeeds with the mailbox's oldest message as a word, or fails once the
// timeout passes. Unlike a send, it needs nothing of the host's, so it is the same
// in a procedure that a region calls. Outside a region, a receive that the
// device does not have is left to the call of process.receive, which the
// device's compile rejects only if a region calls it (false); the procedure
// may run only on the host, such as a polymorphic one that main calls.
bool MerkGenerator::ReceiveInTarget(const Body &goal,
    llvm::ArrayRef<BodyVar> arguments, const ProcedureBody &body,
    bool inRegion) {
  parser::CharBlock at{Where(goal.tokens)};
  mlir::Location loc{Location(goal.tokens)};
  mlir::Value mailbox{ValueOf(arguments[0])};
  mlir::Value selector{ValueOf(arguments[1])};
  mlir::Value timeout{ValueOf(arguments[2])};
  if (!mailbox || !selector || !timeout) {
    failed_ = true;
    return true;
  }
  // any's closure captures only the message type's type_info.
  auto closure{selector.getDefiningOp<merk::ClosureOp>()};
  if (!closure || !closure.getEntry().starts_with("process.any_2_0$closure")) {
    if (!inRegion) {
      return false;
    }
    NotYet(at, "receiving in target regions with a selector other than any");
    return true;
  }
  BodyVar result{arguments[3]};
  const semantics::Type *receivedType{body.vars[result].type};
  const semantics::Type *messageType{receivedType->arguments.front()};
  mlir::Type message{Represent(messageType)->type};
  bool array{ArrayMessage(message)};
  bool copied{CopiedMessage(*messageType, message)};
  if (!WordMessage(message) && !array && !copied) {
    if (!inRegion) {
      return false;
    }
    NotYet(at,
        "receiving messages of type " + semantics::TypeToString(*messageType) +
            " in target regions");
    return true;
  }
  llvm::SmallVector<mlir::Value> operands{
      mailbox, TimeoutMilliseconds(timeout, loc)};
  if (copied) {
    mlir::Value info{MessageTypeInfo(messageType, loc, at)};
    if (!info) {
      failed_ = true;
      return true;
    }
    operands.push_back(info);
  }
  auto word{merk::WordType::get(&context_)};
  auto received{merk::ForeignCallOp::create(builder_, loc,
      mlir::TypeRange{builder_.getI1Type(), word},
      array        ? "merk_device_receive_array"
          : copied ? "merk_device_receive_term"
                   : "merk_device_receive",
      operands, /*can_fail=*/true)};
  auto termType{mlir::cast<merk::TermType>(Represent(receivedType)->type)};
  auto ite{merk::IteOp::create(builder_, loc, mlir::TypeRange{termType})};
  {
    mlir::OpBuilder::InsertionGuard guard{builder_};
    builder_.setInsertionPointToStart(&ite.getCondition().emplaceBlock());
    merk::GuardOp::create(builder_, loc, received.getResult(0));
    merk::YieldOp::create(builder_, loc, mlir::ValueRange{});
    builder_.setInsertionPointToStart(&ite.getThenRegion().emplaceBlock());
    mlir::Value value{merk::FromWordOp::create(
        builder_, loc, message, received.getResult(1))};
    merk::YieldOp::create(builder_, loc,
        mlir::ValueRange{merk::ConstructOp::create(builder_, loc, termType,
            merk::FunctorAttr::get(&context_, "message", 1),
            mlir::ValueRange{value})});
    builder_.setInsertionPointToStart(&ite.getElseRegion().emplaceBlock());
    merk::YieldOp::create(builder_, loc,
        mlir::ValueRange{merk::ConstructOp::create(builder_, loc, termType,
            merk::FunctorAttr::get(&context_, "timeout", 0),
            mlir::ValueRange{})});
  }
  values_[result] = ite.getResult(0);
  // any's closure, which the device cannot make, is not called.
  if (closure->use_empty()) {
    values_.erase(arguments[1]);
    closure->erase();
  }
  return true;
}

// process.team_mailbox(T, Box, !IO) in a target region, or in any
// procedure of the device's compile, all of which run on the device: a call
// of merk_device_team_mailbox with the team and the message type's key,
// which gives the handle's cell in the device's arena (docs/INPROGRESS.md,
// Processes as teams, step 2). Messages of one word, and arrays of one-word
// elements, which go by their pointer (step 3); other types are left to the
// call outside a region, as SendInTarget leaves a send.
bool MerkGenerator::TeamMailboxOnDevice(const Body &goal,
    llvm::ArrayRef<BodyVar> arguments, const ProcedureBody &body,
    bool inRegion) {
  parser::CharBlock at{Where(goal.tokens)};
  mlir::Location loc{Location(goal.tokens)};
  mlir::Value team{ValueOf(arguments[0])};
  if (!team) {
    failed_ = true;
    return true;
  }
  const semantics::Type *mailboxType{body.vars[arguments[1]].type};
  const semantics::Type *messageType{mailboxType->arguments.front()};
  mlir::Type message{Represent(messageType)->type};
  // Temporary (docs/INPROGRESS.md, GPU processes without omp target, step
  // 4e): a team mailbox of messages that are copied by their type_info
  // would share them by their pointer, as arrays, but is not tested yet.
  std::optional<std::uint64_t> key{CopiedMessage(*messageType, message)
          ? std::nullopt
          : MessageKey(*messageType, message)};
  if (!key) {
    if (!inRegion) {
      return false;
    }
    NotYet(at,
        "team mailboxes for messages of type " +
            semantics::TypeToString(*messageType) + " on a device");
    return true;
  }
  auto ptr{mlir::LLVM::LLVMPointerType::get(&context_)};
  auto handle{merk::ForeignCallOp::create(builder_, loc, mlir::TypeRange{ptr},
      "merk_device_team_mailbox",
      mlir::ValueRange{team,
          IntegerConstant(builder_.getI64Type(),
              static_cast<std::int64_t>(*key), loc)},
      /*can_fail=*/false)};
  values_[arguments[1]] = mlir::UnrealizedConversionCastOp::create(builder_,
      loc, Represent(mailboxType)->type, handle.getResult(0))
                              .getResult(0);
  return true;
}

// The call with which a target teams region's teams say so at their start
// (TargetTeams), or null for another region.
static merk::ForeignCallOp TeamsEnter(mlir::omp::TargetOp target) {
  merk::ForeignCallOp enter;
  target.walk([&](merk::ForeignCallOp call) {
    if (call.getCallee() == "merk_device_teams_enter") {
      enter = call;
    }
  });
  return enter;
}

// The library's process.team_barrier, as the device's compile copies it in
// (CopyDeviceLibrary), whose C waits for the region's other teams.
static constexpr llvm::StringLiteral kTeamBarrier{"process.team_barrier_2_0"};

// Whether `op`, or code of the module that it refers to by a symbol, such as
// a procedure it calls or a closure's entry, waits for the other teams of
// its region: it calls merk_device_team_mailbox, whose messages come from
// them, or is the barrier. `seen` holds the symbols' operations looked at
// already.
static bool WaitsForTeams(mlir::Operation *op, mlir::ModuleOp module,
    llvm::DenseSet<mlir::Operation *> &seen) {
  mlir::WalkResult result{op->walk([&](mlir::Operation *inner) {
    auto call{mlir::dyn_cast<merk::ForeignCallOp>(inner)};
    if (call && call.getCallee() == "merk_device_team_mailbox") {
      return mlir::WalkResult::interrupt();
    }
    auto proc{mlir::dyn_cast<merk::ProcOp>(inner)};
    if (proc && proc.getSymName() == kTeamBarrier) {
      return mlir::WalkResult::interrupt();
    }
    for (mlir::NamedAttribute attribute : inner->getAttrs()) {
      mlir::Operation *symbol{nullptr};
      if (auto ref{mlir::dyn_cast<mlir::FlatSymbolRefAttr>(
              attribute.getValue())}) {
        symbol = module.lookupSymbol(ref);
      } else if (auto name{mlir::dyn_cast<mlir::StringAttr>(
                     attribute.getValue())}) {
        symbol = module.lookupSymbol(name);
      }
      if (symbol && seen.insert(symbol).second &&
          WaitsForTeams(symbol, module, seen)) {
        return mlir::WalkResult::interrupt();
      }
    }
    return mlir::WalkResult::advance();
  })};
  return result.wasInterrupted();
}

void MerkGenerator::MarkRegionsForTeamMailboxes() {
  if (!target_.options().openMPIsTargetDevice) {
    return;
  }
  bool teamMailboxes{module_.lookupSymbol(kTeamBarrier) != nullptr};
  module_.walk([&](merk::ForeignCallOp call) {
    if (call.getCallee() == "merk_device_team_mailbox") {
      teamMailboxes = true;
    }
  });
  if (!teamMailboxes) {
    return;
  }
  llvm::SmallVector<mlir::omp::TargetOp> regions;
  llvm::SmallVector<merk::ForeignCallOp> waits;
  module_.walk([&](mlir::omp::TargetOp target) {
    merk::ForeignCallOp enter{TeamsEnter(target)};
    if (!enter) {
      regions.push_back(target);
      return;
    }
    llvm::DenseSet<mlir::Operation *> seen;
    if (WaitsForTeams(target, module_, seen)) {
      waits.push_back(enter);
    }
  });
  // The teams of a region with team mailboxes or the barrier wait until all
  // have started, as a team that is not running would deadlock those
  // waiting for it (docs/INPROGRESS.md, Processes as teams, step 4).
  for (merk::ForeignCallOp enter : waits) {
    mlir::OpBuilder::InsertionGuard guard{builder_};
    builder_.setInsertionPointAfter(enter);
    merk::ForeignCallOp::create(builder_, enter.getLoc(), mlir::TypeRange{},
        "merk_device_teams_wait_all", mlir::ValueRange{}, /*can_fail=*/false);
  }
  for (mlir::omp::TargetOp target : regions) {
    // Where the region reads its inputs (Target): in an SPMD kernel, in its
    // teams and parallel, which every thread runs.
    mlir::Block *start{&target.getRegion().front()};
    if (auto teams{mlir::dyn_cast<mlir::omp::TeamsOp>(start->front())}) {
      start = &teams.getRegion().front();
      if (auto parallel{mlir::dyn_cast<mlir::omp::ParallelOp>(start->front())}) {
        start = &parallel.getRegion().front();
      }
    }
    mlir::OpBuilder::InsertionGuard guard{builder_};
    builder_.setInsertionPointToStart(start);
    merk::ForeignCallOp::create(builder_, target.getLoc(), mlir::TypeRange{},
        "merk_device_no_teams", mlir::ValueRange{}, /*can_fail=*/false);
  }
}

// As MapArray maps an array: by its pointer, with the cell's words as the
// section, which only the host compile gives.
mlir::Value MerkGenerator::MapCell(
    mlir::Value term, unsigned words, llvm::StringRef name, mlir::Location loc) {
  auto ptr{mlir::LLVM::LLVMPointerType::get(&context_)};
  auto i64{builder_.getI64Type()};
  llvm::SmallVector<mlir::Value, 1> bounds;
  if (!target_.options().openMPIsTargetDevice) {
    bounds.push_back(mlir::omp::MapBoundsOp::create(builder_, loc,
        builder_.getType<mlir::omp::MapBoundsType>(),
        IntegerConstant(i64, 0, loc), IntegerConstant(i64, words - 1, loc),
        IntegerConstant(i64, words, loc),
        /*stride=*/IntegerConstant(i64, 1, loc), /*strideInBytes=*/false,
        /*startIdx=*/IntegerConstant(i64, 0, loc)));
  }
  mlir::Value pointer{
      mlir::UnrealizedConversionCastOp::create(builder_, loc, ptr, term)
          .getResult(0)};
  return mlir::omp::MapInfoOp::create(builder_, loc, ptr, pointer,
      mlir::TypeAttr::get(i64),
      builder_.getAttr<mlir::omp::ClauseMapFlagsAttr>(
          mlir::omp::ClauseMapFlags::to),
      builder_.getAttr<mlir::omp::VariableCaptureKindAttr>(
          mlir::omp::VariableCaptureKind::ByRef),
      /*varPtrPtr=*/mlir::Value{}, /*varPtrPtrType=*/mlir::TypeAttr{},
      /*members=*/mlir::ValueRange{}, /*membersIndex=*/mlir::ArrayAttr{},
      bounds, /*mapperId=*/mlir::FlatSymbolRefAttr{},
      builder_.getStringAttr(name), builder_.getBoolAttr(false));
}

// The kernel of a target region with the leaves `queue`, target first, and
// whether the host computes its loop's bounds and step, as host_eval
// values, by Flang's patterns (KernelTypeVisitor and HostEvalVisitor in
// flang/lib/Lower/OpenMP/OpenMP.cpp): after target, teams distribute
// parallel do and parallel do are SPMD, with the bounds on the host;
// teams distribute is generic, with the bounds on the host too, from which
// the host computes the number of teams; anything else is generic. The
// simd and loop forms of Flang's patterns are not implemented yet, and
// neither is a target region whose goal is a single directive, which
// would continue the pattern as a combined directive does.
struct Kernel {
  mlir::omp::TargetExecMode mode{mlir::omp::TargetExecMode::generic};
  bool hostEvalBounds{false};
};
static bool IsDirectives(llvm::ArrayRef<llvm::omp::Directive> leaves,
    std::initializer_list<llvm::omp::Directive> pattern) {
  return llvm::equal(leaves, pattern);
}

static Kernel KernelOf(const omp::ConstructQueue &queue) {
  using llvm::omp::Directive;
  llvm::SmallVector<Directive> leaves;
  for (const omp::Leaf &leaf : queue) {
    leaves.push_back(leaf.id);
  }
  if (IsDirectives(leaves,
          {Directive::OMPD_target, Directive::OMPD_teams,
              Directive::OMPD_distribute, Directive::OMPD_parallel,
              Directive::OMPD_do}) ||
      IsDirectives(leaves,
          {Directive::OMPD_target, Directive::OMPD_parallel,
              Directive::OMPD_do})) {
    return {mlir::omp::TargetExecMode::spmd, true};
  }
  if (IsDirectives(leaves,
          {Directive::OMPD_target, Directive::OMPD_teams,
              Directive::OMPD_distribute})) {
    return {mlir::omp::TargetExecMode::generic, true};
  }
  return {};
}

// The goals of a conjunction, in order, flattened.
static void CollectConjuncts(const Body &goal, std::vector<const Body *> &goals) {
  if (goal.kind == BodyKind::Conj) {
    for (const Body *conjunct : goal.goals) {
      CollectConjuncts(*conjunct, goals);
    }
  } else {
    goals.push_back(&goal);
  }
}

// The last call in a goal, in the order of its text, of a predicate.
static const Body *LastCallOf(const Body &goal, const semantics::Symbol *callee) {
  const Body *found{nullptr};
  if (goal.kind == BodyKind::Call && goal.modes && goal.modes->callee &&
      goal.modes->callee->symbol == callee) {
    found = &goal;
  }
  for (const Body *inner : goal.goals) {
    if (const Body *last{LastCallOf(*inner, callee)}) {
      found = last;
    }
  }
  return found;
}

// A reduction is on the leaves the decomposition puts it on, do and teams, as
// Flang puts it, and the map of its variable that target implies is the
// reduction's.
static bool IsOmpLoopClauseHandled(
    llvm::omp::Directive leaf, llvm::omp::Clause id, bool synthesized) {
  if (synthesized) {
    return id == llvm::omp::OMPC_map && leaf == llvm::omp::OMPD_target;
  }
  if (id == llvm::omp::OMPC_device) {
    return leaf == llvm::omp::OMPD_target;
  }
  return id == llvm::omp::OMPC_reduction &&
      (leaf == llvm::omp::OMPD_do || leaf == llvm::omp::OMPD_teams);
}

static bool IsTargetClauseHandled(
    llvm::omp::Directive, llvm::omp::Clause id, bool) {
  return id == llvm::omp::OMPC_device;
}

static bool IsTeamsClauseHandled(
    llvm::omp::Directive, llvm::omp::Clause id, bool) {
  return id == llvm::omp::OMPC_num_teams;
}

static bool IsTargetDataClauseHandled(
    llvm::omp::Directive, llvm::omp::Clause id, bool) {
  return id == llvm::omp::OMPC_map || id == llvm::omp::OMPC_device;
}

static bool IsNamedType(const semantics::Type &type, llvm::StringRef module,
    llvm::StringRef name) {
  return type.kind == semantics::TypeKind::Named &&
      type.constructor->module == module && type.constructor->name == name;
}

// The type of a value a loop carries, at the call of its loop procedure.
static const semantics::Type *ThreadedType(const ProcedureBody &body,
    const Body &call, const semantics::TranslatedLoop::Threaded &value) {
  return body.vars[call.arguments[value.in]].type;
}

static bool IsReduction(llvm::ArrayRef<LoopReduction> reductions,
    const semantics::TranslatedLoop::Threaded &value) {
  return llvm::any_of(
      reductions, [&](const LoopReduction &r) { return r.value == &value; });
}

static bool HasLeaf(const omp::ConstructQueue &queue, llvm::omp::Directive id) {
  return llvm::any_of(
      queue, [&](const omp::Leaf &leaf) { return leaf.id == id; });
}

// The operator of a reduction clause by Mercury's name: +, *, min and
// max, of ints and floats, and /\, \/ and xor, of ints.
static std::optional<ReductionOp> ReductionOpOf(llvm::StringRef name) {
  return llvm::StringSwitch<std::optional<ReductionOp>>(name)
      .Case("+", ReductionOp::Add)
      .Case("*", ReductionOp::Multiply)
      .Case("max", ReductionOp::Max)
      .Case("min", ReductionOp::Min)
      .Case("/\\", ReductionOp::And)
      .Case("\\/", ReductionOp::Or)
      .Case("xor", ReductionOp::Xor)
      .Default(std::nullopt);
}

// The array that an array a target region gives was, before the region
// updated it in place, if the region's own operations show it.
static mlir::Value UpdatedFrom(mlir::Value array) {
  while (mlir::Operation *op = array.getDefiningOp()) {
    if (!mlir::isa<merk::ArraySetOp, merk::ArrayFillOp>(op)) {
      break;
    }
    array = op->getOperand(0);
  }
  return array;
}

// The goal under the directive is the loop's translation: the goals of its
// bounds, the computation of its trip count, and the call of its loop
// predicate (design document, "From a nest to omp.loop_nest"). The bounds'
// goals are generated as they are; the trip count, and the loop predicate,
// are not needed: omp.loop_nest runs the iterations from Lo to Hi, by the
// step, inclusive, and each calls the body procedure with the iteration's
// counter, the threaded arrays as they were before the loop and the
// shared variables. By the directive's promise, an array that the body
// updates in place gives each iteration its own elements, so after the
// loop it is the array given (design document, "Loops", Under a parallel
// directive). The directive is decomposed into its leaves, each with its
// own clauses, as Flang decomposes it: parallel and do, or target, teams,
// distribute, parallel and do, where the loop is in a target region whose
// kernel is SPMD, given its bounds and step by the host as host_eval
// values.
void MerkGenerator::OmpLoop(const Body &goal, const ProcedureBody &body) {
  using namespace parser::literals;
  const parser::OmpDirective &directive{*goal.directive};
  parser::CharBlock at{Where(goal.tokens)};
  std::string name{parser::EnumToString(directive.id)};
  bool offloaded{directive.id ==
      llvm::omp::Directive::OMPD_target_teams_distribute_parallel_do};
  // A loop that a GPU process runs across its teams (docs/INPROGRESS.md, GPU
  // processes that run loops), which check-gpu-code allows only in a GPU
  // kind's code: on the host, as in a kind spawned there, a parallel do.
  bool processLoop{
      directive.id == llvm::omp::Directive::OMPD_distribute_parallel_do};
  if (directive.id != llvm::omp::Directive::OMPD_parallel_do && !offloaded &&
      !processLoop) {
    NotYet(at, "the " + name + " directive");
    return;
  }
  if (offloaded &&
      builder_.getInsertionBlock()
          ->getParent()
          ->getParentOfType<mlir::omp::TargetOp>()) {
    NotYet(at, "target regions in target regions");
    return;
  }
  std::vector<const Body *> goals;
  CollectConjuncts(*goal.goals.front(), goals);
  const Body *call{nullptr};
  const semantics::TranslatedLoop *loop{nullptr};
  for (const Body *g : goals) {
    if (g->kind == BodyKind::Call && g->modes && g->modes->callee) {
      if (const auto *found{loops_.Of(*g->modes->callee->symbol)}) {
        call = g;
        loop = found;
      }
    }
  }
  if (!loop) {
    Error(at, "the %s directive must be before a loop"_err_en_US, name);
    return;
  }
  std::optional<omp::ConstructQueue> queue{Decompose(directive, at,
      loop->counter.empty() ? std::nullopt
                            : std::optional<llvm::StringRef>{loop->counter})};
  if (!queue) {
    return;
  }
  for (const omp::Leaf &leaf : *queue) {
    if (!LeafClauses(leaf, at, IsOmpLoopClauseHandled)) {
      return;
    }
  }
  // Its form: a for, and fromtos that do not end it; a loop has at most
  // one for.
  bool counted{false};
  for (const semantics::LoopIterator &iterator : loop->loop->iterators) {
    llvm::StringRef kind;
    switch (iterator.kind) {
    case parser::LoopIteratorKind::For:
      counted = true;
      break;
    case parser::LoopIteratorKind::Foreach:
      kind = "foreach";
      break;
    case parser::LoopIteratorKind::Count:
      kind = "count";
      break;
    case parser::LoopIteratorKind::Fromto:
      if (iterator.ends) {
        kind = "a fromto that ends the loop";
      }
      break;
    }
    if (!kind.empty()) {
      Error(Where(iterator.tokens),
          "a parallel do loop is counted by its for, so it cannot have %s"_err_en_US,
          kind);
      return;
    }
  }
  if (!counted) {
    Error(at, "a parallel do loop must have a for iterator"_err_en_US);
    return;
  }
  const Body *trip{nullptr};
  for (const Body *g : goals) {
    if (g->kind == BodyKind::IfThenElse &&
        g->goals.front()->kind == BodyKind::Call && g->goals.front()->call &&
        g->goals.front()->call->number == *loop->tripTest) {
      trip = g;
    }
  }
  const Procedure &loopProcedure{*call->modes->callee};
  const ProcedureBody *loopBody{determinism_.BodyOf(loopProcedure)};
  // The loop procedure's call of the body procedure.
  const Body *bodyCall{loopBody && loopBody->body
          ? LastCallOf(*loopBody->body, loop->bodyPredicate)
          : nullptr};
  if (!trip || !bodyCall) {
    NotYet(at, "parallel do loops whose translation mode analysis changed");
    return;
  }
  // The checks of the parallel loop annotations (design document): the
  // body is det and pure, and threads only arrays, until reductions.
  const Procedure &bodyProcedure{*bodyCall->modes->callee};
  std::optional<parser::Determinism> determinism{
      determinism_.DeterminismOf(bodyProcedure)};
  if (determinism != parser::Determinism::Det) {
    Error(at,
        "the body of a parallel do loop must be det, but it is %s"_err_en_US,
        determinism ? semantics::DeterminismName(*determinism) : "unknown");
    return;
  }
  const semantics::Signature *signature{
      types_.SignatureOf(*loop->bodyPredicate)};
  if (signature && signature->purity != parser::Purity::Pure) {
    Error(at, "the body of a parallel do loop must be pure"_err_en_US);
    return;
  }
  // The reductions: each value the loop carries that a reduction clause
  // names, by its name, or the one scalar it carries, for a clause that
  // names none (design document, "Loops", Reductions).
  llvm::SmallVector<LoopReduction> reductions;
  for (const parser::OmpClause &clause : directive.clauses) {
    const auto *reduction{
        std::get_if<const parser::OmpReductionClause *>(&clause)};
    if (!reduction) {
      continue;
    }
    if ((*reduction)->modifier) {
      NotYet(at, "reduction modifiers");
      return;
    }
    // An operator is any function the programmer promises is associative;
    // so far, those of int and float that Flang has as intrinsic ones.
    const parser::SymName &opName{*(*reduction)->op};
    std::optional<ReductionOp> op{ReductionOpOf(opName.name)};
    if (!op || opName.qualifiers.size() > 1) {
      NotYet(at,
          "reductions by operators other than +, *, min, max, /\\, \\/ and "
          "xor");
      return;
    }
    llvm::SmallVector<const semantics::TranslatedLoop::Threaded *> named;
    if ((*reduction)->variables.empty()) {
      for (const semantics::TranslatedLoop::Threaded &value : loop->threaded) {
        if (!IsNamedType(*ThreadedType(body, *call, value), "array", "array")) {
          named.push_back(&value);
        }
      }
      if (named.size() != 1) {
        Error(at,
            "a reduction clause without variables names the one scalar the loop carries from one iteration to the next, but it carries %s"_err_en_US,
            named.empty() ? "none" : "several");
        return;
      }
    }
    for (const parser::Variable *variable : (*reduction)->variables) {
      auto value{llvm::find_if(
          loop->threaded, [&](const semantics::TranslatedLoop::Threaded &v) {
            return v.name == variable->name;
          })};
      if (value == loop->threaded.end()) {
        Error(at,
            "%s is not a value the loop carries from one iteration to the next, a fromto's Last or a state variable"_err_en_US,
            variable->name);
        return;
      }
      named.push_back(&*value);
    }
    for (const semantics::TranslatedLoop::Threaded *value : named) {
      if (IsReduction(reductions, *value)) {
        Error(
            at, "%s is named in two reduction clauses"_err_en_US, value->name);
        return;
      }
      // The primitive int and float.
      bool real{IsNamedType(*ThreadedType(body, *call, *value), "", "float")};
      if (!real &&
          !IsNamedType(*ThreadedType(body, *call, *value), "", "int")) {
        NotYet(at,
            "reductions of type " +
                semantics::TypeToString(
                    *body.vars[call->arguments[value->in]].type));
        return;
      }
      if (real && *op != ReductionOp::Add && *op != ReductionOp::Multiply &&
          *op != ReductionOp::Max && *op != ReductionOp::Min) {
        Error(at,
            "the reduction operator %s is not defined on floats"_err_en_US,
            opName.name);
        return;
      }
      // A qualified operator is the type's own, int.max of an int.
      if (!opName.qualifiers.empty() &&
          opName.qualifiers.front() != (real ? "float" : "int")) {
        Error(at,
            real
                ? "the reduction operator %s is not defined on floats"_err_en_US
                : "the reduction operator %s is not defined on ints"_err_en_US,
            (opName.qualifiers.front() + "." + opName.name).str());
        return;
      }
      reductions.push_back({value, *op,
          Represent(body.vars[call->arguments[value->in]].type)->type});
    }
  }
  for (const semantics::TranslatedLoop::Threaded &value : loop->threaded) {
    if (IsNamedType(*ThreadedType(body, *call, value), "io", "state")) {
      Error(at,
          "a parallel do loop must not thread the io state, since its iterations run in any order"_err_en_US);
      return;
    }
    if (!IsReduction(reductions, value) &&
        !IsNamedType(*ThreadedType(body, *call, value), "array", "array")) {
      Error(value.fromto ? Where(value.fromto->tokens) : at,
          "a parallel do loop can carry %s from one iteration to the next only if it is named in a reduction clause or is an array updated in place"_err_en_US,
          value.fromto ? "a fromto's value" : "a state variable");
      return;
    }
  }
  // A reduction's next value is the body procedure's output that the loop
  // procedure passes on, to its own call, as the value's In.
  const Body *recursion{LastCallOf(*loopBody->body, loop->loopPredicate)};
  for (LoopReduction &reduction : reductions) {
    std::optional<unsigned> output;
    unsigned outputs{0};
    for (auto [argument, mode] :
        llvm::zip(bodyCall->arguments, bodyProcedure.modes)) {
      if (Represent(loopBody->vars[argument].type)->dummy || !IsOutput(mode)) {
        continue;
      }
      if (recursion && argument == recursion->arguments[reduction.value->in]) {
        output = outputs;
      }
      ++outputs;
    }
    if (!output) {
      NotYet(at, "reductions whose next value the loop's body does not give");
      return;
    }
    reduction.output = *output;
  }
  if (processLoop && target_.options().processKernels) {
    ProcessLoop({goals, trip, call, loop, *loopBody, bodyCall, bodyProcedure,
                    reductions, false, false, false, at},
        body, Location(goal.tokens));
    return;
  }
  // The leaves around the loop's wrappers.
  bool teams{HasLeaf(*queue, llvm::omp::Directive::OMPD_teams)};
  bool distribute{
      !processLoop && HasLeaf(*queue, llvm::omp::Directive::OMPD_distribute)};
  // Each reduction's variable is a slot of the procedure, holding First, or
  // the state variable's value before the loop, which the partial results
  // are combined with; after the loop, Last, or the state variable's value
  // after it, is what the slot holds.
  mlir::Location loc{Location(goal.tokens)};
  Kernel kernel{offloaded ? KernelOf(*queue) : Kernel{}};
  bool evaluated{
      kernel.hostEvalBounds && !target_.options().openMPIsTargetDevice};
  // The loop's goals, before the region too, for First, which is one of
  // them, and for the host_eval values.
  if (!reductions.empty() || evaluated) {
    for (const Body *g : goals) {
      if (g != trip && g != call) {
        Goal(*g, body);
      }
    }
    if (failed_) {
      return;
    }
  }
  llvm::SmallVector<UpdatedSlot> updated;
  // On a device, a loop that is not a kernel of its own starts from one thread
  // of a region, whose stack the team's other threads cannot read.
  bool sharedSlots{!offloaded && target_.options().openMPIsTargetDevice};
  llvm::SmallVector<mlir::Value> shared;
  for (LoopReduction &reduction : reductions) {
    reduction.slot =
        sharedSlots ? SharedSlots(1, loc) : Slot(reduction.type, loc);
    if (sharedSlots) {
      shared.push_back(reduction.slot);
    }
    mlir::LLVM::StoreOp::create(builder_, loc,
        ValueOf(call->arguments[reduction.value->in]), reduction.slot);
    updated.push_back({reduction.slot, reduction.type, reduction.value->name});
  }
  ParallelLoop parallelLoop{goals, trip, call, loop, *loopBody, bodyCall,
      bodyProcedure, reductions, teams, distribute, offloaded, at};
  if (!offloaded) {
    GenerateParallelLoop(parallelLoop, body, loc);
    if (failed_) {
      return;
    }
    if (sharedSlots) {
      ShareCapturedScalars(
          mlir::cast<mlir::omp::ParallelOp>(builder_.getInsertionBlock()->back()),
          loc);
    }
    LoadReductions(reductions, *call, loc);
    for (mlir::Value slots : llvm::reverse(shared)) {
      FreeShared(slots, 1, loc);
    }
    return;
  }
  llvm::SmallVector<mlir::Value, 3> hostEval;
  if (evaluated) {
    BodyVar big{trip->goals.front()->arguments[0]};
    BodyVar small{trip->goals.front()->arguments[1]};
    hostEval = {ValueOf(loop->step > 0 ? small : big),
        ValueOf(loop->step > 0 ? big : small),
        IntegerConstant(builder_.getI64Type(), loop->step, loc)};
  }
  std::optional<mlir::Value> device{IntegerClause(queue->front(),
      llvm::omp::OMPC_device, "device number", body, at, loc)};
  if (!device) {
    return;
  }
  Target(
      goal, body, [&]() { GenerateParallelLoop(parallelLoop, body, loc); },
      /*combined=*/true, kernel.mode, hostEval, updated, *device);
  if (!failed_) {
    LoadReductions(reductions, *call, loc);
  }
}

// Offloaded, the kernel is as Flang makes it (KernelOf): for target teams
// distribute parallel do, SPMD, and the host computes the bounds and the
// step before the target region and gives them to it as host_eval
// values, from which it computes the trip count and so the number of
// teams and threads, and the nest runs over them. The bounds'
// goals are in the region too, on both sides, and the nest is first made
// over their values, so that the host and the device map the same values,
// a bound that is a variable included. Target then gives the host's nest
// the host_eval values; the device's stays over those, as Flang's does,
// since the device has no host_eval.
// Words that the threads of a device's team can all reach, as Clang's
// globalization has for a variable that a parallel region shares: the device
// runtime's __kmpc_alloc_shared, freed in the reverse order.
mlir::Value MerkGenerator::SharedSlots(std::size_t count, mlir::Location loc) {
  return merk::ForeignCallOp::create(builder_, loc,
      mlir::TypeRange{mlir::LLVM::LLVMPointerType::get(&context_)},
      "merk_device_shared_alloc",
      mlir::ValueRange{IntegerConstant(builder_.getI64Type(),
          static_cast<std::int64_t>(count * sizeof(std::uint64_t)), loc)},
      /*can_fail=*/false)
      .getResult(0);
}

void MerkGenerator::FreeShared(
    mlir::Value slots, std::size_t count, mlir::Location loc) {
  merk::ForeignCallOp::create(builder_, loc, mlir::TypeRange{},
      "merk_device_shared_free",
      mlir::ValueRange{slots,
          IntegerConstant(builder_.getI64Type(),
              static_cast<std::int64_t>(count * sizeof(std::uint64_t)), loc)},
      /*can_fail=*/false);
}

// A scalar that a parallel region on a device uses from outside it: the
// outlining would pass it in a slot of the starting thread's stack, which the
// team's other threads cannot read. So it goes in shared memory, stored before
// the region and loaded at its start. A pointer is passed as it is.
void MerkGenerator::ShareCapturedScalars(
    mlir::omp::ParallelOp parallel, mlir::Location loc) {
  llvm::SetVector<mlir::Value> used;
  mlir::getUsedValuesDefinedAbove(parallel.getRegion(), used);
  llvm::SmallVector<mlir::Value> scalars;
  for (mlir::Value value : used) {
    if (mlir::isa<mlir::IntegerType, mlir::FloatType>(value.getType())) {
      scalars.push_back(value);
    }
  }
  if (scalars.empty()) {
    return;
  }
  mlir::OpBuilder::InsertionGuard guard{builder_};
  builder_.setInsertionPoint(parallel);
  mlir::Value slots{SharedSlots(scalars.size(), loc)};
  mlir::Type pointer{mlir::LLVM::LLVMPointerType::get(&context_)};
  mlir::Type byte{builder_.getI8Type()};
  auto at{[&](std::size_t index) -> mlir::Value {
    return mlir::LLVM::GEPOp::create(builder_, loc, pointer, byte, slots,
        llvm::ArrayRef<mlir::LLVM::GEPArg>{static_cast<std::int32_t>(
            index * sizeof(std::uint64_t))});
  }};
  for (auto [index, value] : llvm::enumerate(scalars)) {
    mlir::LLVM::StoreOp::create(builder_, loc, value, at(index));
  }
  mlir::Block &entry{parallel.getRegion().front()};
  builder_.setInsertionPointToStart(&entry);
  for (auto [index, value] : llvm::enumerate(scalars)) {
    mlir::Value loaded{mlir::LLVM::LoadOp::create(
        builder_, loc, value.getType(), at(index))};
    mlir::replaceAllUsesInRegionWith(value, loaded, parallel.getRegion());
  }
  builder_.setInsertionPointAfter(parallel);
  FreeShared(slots, scalars.size(), loc);
}

void MerkGenerator::GenerateParallelLoop(const ParallelLoop &loop,
    const ProcedureBody &body, mlir::Location loc) {
  mlir::OpBuilder::InsertionGuard guard{builder_};
  // Teams and parallel end with terminators, added after what is in them,
  // since a goal is not generated after one.
  llvm::SmallVector<mlir::Block *, 2> terminated;
  // The leaves, each in the one before, as Flang marks them: a leaf
  // around the parallel, a combined one; with distribute, the parallel,
  // distribute and do are the parts of a composite one. The bounds'
  // goals are in the parallel, as Flang places them: on an AMD GPU, a
  // generic kernel traps when the parallel in its teams uses a value
  // computed outside the parallel.
  // A reduction is on teams and on the do, each with its own copy of the
  // variable, the teams' copy the do's original, as in Flang: by value,
  // by its slot's pointer, which teams has from the target region, mapped
  // tofrom.
  llvm::SmallVector<mlir::Value> originals;
  for (const LoopReduction &reduction : loop.reductions) {
    originals.push_back(reduction.slot);
  }
  llvm::SmallVector<mlir::Type> pointers(
      loop.reductions.size(), mlir::LLVM::LLVMPointerType::get(&context_));
  llvm::SmallVector<mlir::Location> locs(loop.reductions.size(), loc);
  if (loop.teams) {
    mlir::omp::TeamsOperands operands;
    AddReductions(operands, loop.reductions, originals, loc);
    auto op{mlir::omp::TeamsOp::create(builder_, loc, operands)};
    op.setCombined(true);
    terminated.push_back(
        builder_.createBlock(&op.getRegion(), {}, pointers, locs));
    originals.assign(op.getRegion().args_begin(), op.getRegion().args_end());
  }
  // A parallel region on the host holds up no worker, and its team's size is
  // claimed from the cores that are free (design document, "OpenMP
  // cooperation"). In a target region or on a device, the process has moved
  // already, or is not there.
  mlir::Operation *around{builder_.getInsertionBlock()->getParentOp()};
  bool claims{!target_.options().openMPIsTargetDevice && !loop.teams &&
      !mlir::isa<mlir::omp::TargetOp>(around) &&
      !around->getParentOfType<mlir::omp::TargetOp>()};
  mlir::omp::ParallelOperands parallelOperands;
  if (claims) {
    parallelOperands.numThreadsVars.push_back(
        merk::ForeignCallOp::create(builder_, loc,
            mlir::TypeRange{builder_.getI32Type()}, "merk_parallel_enter",
            mlir::ValueRange{}, /*can_fail=*/false)
            .getResult(0));
  }
  auto parallel{
      mlir::omp::ParallelOp::create(builder_, loc, parallelOperands)};
  if (loop.distribute) {
    parallel.setComposite(true);
  } else {
    parallel.setCombined(true);
  }
  terminated.push_back(builder_.createBlock(&parallel.getRegion()));
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
  mlir::Value lower{ValueOf(loop.translated->step > 0 ? small : big)};
  mlir::Value upper{ValueOf(loop.translated->step > 0 ? big : small)};
  mlir::Type i64{builder_.getI64Type()};
  mlir::Value step{IntegerConstant(i64, loop.translated->step, loc)};

  llvm::SmallVector<const semantics::Type *> types;
  for (BodyVar argument : loop.bodyCall->arguments) {
    types.push_back(loop.loopBody.vars[argument].type);
  }
  merk::ProcOp proc{ProcFor(loop.bodyProcedure, types, loop.at)};
  if (!proc) {
    return;
  }
  {
    mlir::OpBuilder::InsertionGuard guard{builder_};
    if (loop.distribute) {
      auto op{mlir::omp::DistributeOp::create(
          builder_, loc, mlir::omp::DistributeOperands{})};
      op.setComposite(true);
      builder_.createBlock(&op.getRegion());
    }
    mlir::omp::WsloopOperands operands;
    AddReductions(operands, loop.reductions, originals, loc);
    auto wsloop{mlir::omp::WsloopOp::create(builder_, loc, operands)};
    if (loop.distribute) {
      wsloop.setComposite(true);
    }
    mlir::Block *privates{
        builder_.createBlock(&wsloop.getRegion(), {}, pointers, locs)};
    mlir::omp::LoopNestOperands bounds;
    bounds.loopLowerBounds.push_back(lower);
    bounds.loopUpperBounds.push_back(upper);
    bounds.loopSteps.push_back(step);
    bounds.loopInclusive = builder_.getUnitAttr();
    bounds.collapseNumLoops = builder_.getI64IntegerAttr(1);
    auto nest{mlir::omp::LoopNestOp::create(builder_, loc, bounds)};
    mlir::Block *iteration{
        builder_.createBlock(&nest.getRegion(), {}, {i64}, {loc})};
    // The body procedure's inputs: its hidden arguments, which are the
    // procedure's own, for its type variables and constraints have the same
    // numbers, the counter, and the values the loop procedure was called
    // with for its other arguments.
    llvm::SmallVector<mlir::Value> inputs;
    if (const auto *hidden{
            polymorphism_.lookup(determinism_.BodyOf(loop.bodyProcedure))};
        hidden && !hidden->hidden.empty()) {
      // Temporary (specialization, design document, "What polymorphism
      // costs"): a target region would have to map them.
      if (loop.offloaded) {
        NotYet(loop.at, "target regions in polymorphic procedures");
        return;
      }
      for (const semantics::HiddenArgument &argument : hidden->hidden) {
        if (argument.output) {
          continue;
        }
        inputs.push_back(argument.typeInfo
                ? TypeInfoOfType(argument.typeInfo->type, loc, loop.at,
                      /*freeIsVoid=*/true)
                : dictionaries_.lookup(argument.dictionary->parameter));
      }
    }
    for (auto [argument, mode] :
        llvm::zip(loop.bodyCall->arguments, loop.bodyProcedure.modes)) {
      if (Represent(loop.loopBody.vars[argument].type)->dummy ||
          IsOutput(mode)) {
        continue;
      }
      auto position{llvm::find(loop.loopBody.head, argument)};
      if (position == loop.loopBody.head.end()) {
        NotYet(loop.at,
            "parallel do loops whose body takes a value the loop computes");
        return;
      }
      std::size_t k(position - loop.loopBody.head.begin());
      // A reduction's In is the do's copy of its variable.
      const auto *reduction{llvm::find_if(loop.reductions,
          [&](const LoopReduction &r) { return r.value->in == k; })};
      if (reduction != loop.reductions.end()) {
        inputs.push_back(
            mlir::LLVM::LoadOp::create(builder_, loc, reduction->type,
                privates->getArgument(reduction - loop.reductions.begin())));
        continue;
      }
      inputs.push_back(k == 0 ? iteration->getArgument(0)
                              : ValueOf(loop.call->arguments[k]));
    }
    auto bodyOp{merk::CallOp::create(builder_, loc,
        proc.getFunctionType().getResults(), proc.getSymName(), inputs)};
    for (auto [i, reduction] : llvm::enumerate(loop.reductions)) {
      mlir::LLVM::StoreOp::create(builder_, loc,
          bodyOp->getResult(reduction.output), privates->getArgument(i));
    }
    mlir::omp::YieldOp::create(builder_, loc);
  }
  for (mlir::Block *block : terminated) {
    builder_.setInsertionPointToEnd(block);
    mlir::omp::TerminatorOp::create(builder_, loc);
  }
  if (claims) {
    builder_.setInsertionPointAfter(parallel);
    merk::ForeignCallOp::create(builder_, loc, mlir::TypeRange{},
        "merk_parallel_exit", mlir::ValueRange{}, /*can_fail=*/false);
  }
  for (const semantics::TranslatedLoop::Threaded &value :
      loop.translated->threaded) {
    if (!IsReduction(loop.reductions, value)) {
      values_[loop.call->arguments[value.after]] =
          ValueOf(loop.call->arguments[value.in]);
    }
  }
}

template <typename Operands>
void MerkGenerator::AddReductions(Operands &operands,
    llvm::ArrayRef<LoopReduction> reductions, mlir::ValueRange vars,
    mlir::Location loc) {
  for (auto [reduction, var] : llvm::zip(reductions, vars)) {
    operands.reductionVars.push_back(var);
    operands.reductionByref.push_back(false);
    operands.reductionSyms.push_back(mlir::SymbolRefAttr::get(
        DeclareReduction(reduction.op, reduction.type, loc)));
  }
}

// After the loop, each reduction's Last is what its slot holds.
void MerkGenerator::LoadReductions(llvm::ArrayRef<LoopReduction> reductions,
    const Body &call, mlir::Location loc) {
  for (const LoopReduction &reduction : reductions) {
    values_[call.arguments[reduction.value->after]] =
        mlir::LLVM::LoadOp::create(
            builder_, loc, reduction.type, reduction.slot);
  }
}

// Its identity and combiner are Flang's (getReductionInitValue and
// createScalarCombiner), for by-value integers and floats.
mlir::omp::DeclareReductionOp MerkGenerator::DeclareReduction(
    ReductionOp op, mlir::Type type, mlir::Location loc) {
  bool real{mlir::isa<mlir::FloatType>(type)};
  llvm::StringRef prefix;
  switch (op) {
  case ReductionOp::Add:
    prefix = "add_reduction";
    break;
  case ReductionOp::Multiply:
    prefix = "multiply_reduction";
    break;
  case ReductionOp::Max:
    prefix = "max";
    break;
  case ReductionOp::Min:
    prefix = "min";
    break;
  case ReductionOp::And:
    prefix = "iand";
    break;
  case ReductionOp::Or:
    prefix = "ior";
    break;
  case ReductionOp::Xor:
    prefix = "ieor";
    break;
  }
  std::string name{(prefix + (real ? "_f" : "_i") +
      llvm::Twine{type.getIntOrFloatBitWidth()})
          .str()};
  if (auto found{module_.lookupSymbol<mlir::omp::DeclareReductionOp>(name)}) {
    return found;
  }
  // At the module's start, after those made before it.
  mlir::OpBuilder::InsertionGuard guard{builder_};
  builder_.setInsertionPointToStart(module_.getBody());
  for (mlir::Operation &op : *module_.getBody()) {
    if (!mlir::isa<mlir::omp::DeclareReductionOp>(op)) {
      break;
    }
    builder_.setInsertionPointAfter(&op);
  }
  auto decl{mlir::omp::DeclareReductionOp::create(builder_, loc, name,
      /*sym_visibility=*/nullptr, type, /*byref_element_type=*/nullptr)};
  builder_.createBlock(&decl.getInitializerRegion(), {}, {type}, {loc});
  mlir::omp::YieldOp::create(builder_, loc, ReductionIdentity(op, type, loc));
  mlir::Block *combiner{builder_.createBlock(
      &decl.getReductionRegion(), {}, {type, type}, {loc, loc})};
  mlir::omp::YieldOp::create(builder_, loc,
      ReductionCombine(
          op, combiner->getArgument(0), combiner->getArgument(1), loc));
  return decl;
}

// The identity of the reduction's operator on the type, Flang's
// (getReductionInitValue).
mlir::Value MerkGenerator::ReductionIdentity(
    ReductionOp op, mlir::Type type, mlir::Location loc) {
  bool real{mlir::isa<mlir::FloatType>(type)};
  mlir::Value identity;
  if (real) {
    const llvm::fltSemantics &semantics{
        mlir::cast<mlir::FloatType>(type).getFloatSemantics()};
    llvm::APFloat value{
        op == ReductionOp::Max ? llvm::APFloat::getLargest(semantics, true)
            : op == ReductionOp::Min
            ? llvm::APFloat::getLargest(semantics, false)
            : llvm::APFloat{semantics, op == ReductionOp::Multiply ? 1u : 0u}};
    identity = mlir::arith::ConstantOp::create(
        builder_, loc, type, builder_.getFloatAttr(type, value));
  } else {
    unsigned bits{type.getIntOrFloatBitWidth()};
    llvm::APInt value{op == ReductionOp::Max ? llvm::APInt::getSignedMinValue(bits)
            : op == ReductionOp::Min        ? llvm::APInt::getSignedMaxValue(bits)
            : op == ReductionOp::And        ? llvm::APInt::getAllOnes(bits)
            : op == ReductionOp::Multiply   ? llvm::APInt{bits, 1}
                                            : llvm::APInt{bits, 0}};
    identity = mlir::arith::ConstantOp::create(
        builder_, loc, type, builder_.getIntegerAttr(type, value));
  }
  return identity;
}

// Two values combined by the reduction's operator, Flang's
// (createScalarCombiner).
mlir::Value MerkGenerator::ReductionCombine(
    ReductionOp op, mlir::Value a, mlir::Value b, mlir::Location loc) {
  bool real{mlir::isa<mlir::FloatType>(a.getType())};
  mlir::Value combined;
  switch (op) {
  case ReductionOp::Add:
    combined = real ? mlir::arith::AddFOp::create(builder_, loc, a, b).getResult()
                    : mlir::arith::AddIOp::create(builder_, loc, a, b).getResult();
    break;
  case ReductionOp::Multiply:
    combined = real ? mlir::arith::MulFOp::create(builder_, loc, a, b).getResult()
                    : mlir::arith::MulIOp::create(builder_, loc, a, b).getResult();
    break;
  case ReductionOp::Max:
    combined = real
        ? mlir::arith::MaxNumFOp::create(builder_, loc, a, b).getResult()
        : mlir::arith::MaxSIOp::create(builder_, loc, a, b).getResult();
    break;
  case ReductionOp::Min:
    combined = real
        ? mlir::arith::MinNumFOp::create(builder_, loc, a, b).getResult()
        : mlir::arith::MinSIOp::create(builder_, loc, a, b).getResult();
    break;
  case ReductionOp::And:
    combined = mlir::arith::AndIOp::create(builder_, loc, a, b);
    break;
  case ReductionOp::Or:
    combined = mlir::arith::OrIOp::create(builder_, loc, a, b);
    break;
  case ReductionOp::Xor:
    combined = mlir::arith::XOrIOp::create(builder_, loc, a, b);
    break;
  }
  return combined;
}

std::optional<omp::ConstructQueue> MerkGenerator::Decompose(
    const parser::OmpDirective &directive, parser::CharBlock at,
    std::optional<llvm::StringRef> counter) {
  using namespace parser::literals;
  // Merkurius's own clauses, which the decomposition leaves out.
  for (const parser::OmpClause &clause : directive.clauses) {
    if (std::holds_alternative<const parser::OmpIndexClause *>(clause) ||
        std::holds_alternative<const parser::OmpUpperClause *>(clause)) {
      NotYet(at,
          std::holds_alternative<const parser::OmpIndexClause *>(clause)
              ? "the index clause"
              : "the upper clause");
      return std::nullopt;
    }
  }
  omp::Decomposition decomposition{omp::Decompose(
      target_.options().openMPVersion, directive, counter)};
  if (decomposition.queue.empty()) {
    std::string clause{decomposition.misplaced
            ? llvm::omp::getOpenMPClauseName(
                  omp::ClauseId(*decomposition.misplaced))
                  .str()
            : "a"};
    Error(at, "%s"_err_en_US,
        "the " + clause + " clause is not allowed on the " +
            std::string{parser::EnumToString(directive.id)} + " directive");
    return std::nullopt;
  }
  return std::move(decomposition.queue);
}

bool MerkGenerator::LeafClauses(const omp::Leaf &leaf, parser::CharBlock at,
    llvm::function_ref<bool(llvm::omp::Directive, llvm::omp::Clause, bool)>
        handled) {
  std::string directive{parser::EnumToString(leaf.id)};
  for (const parser::OmpClause *clause : leaf.clauses) {
    llvm::omp::Clause id{omp::ClauseId(*clause)};
    if (handled && handled(leaf.id, id, false)) {
      continue;
    }
    NotYet(at,
        "the " + llvm::omp::getOpenMPClauseName(id).str() + " clause of the " +
            directive + " directive");
    return false;
  }
  for (llvm::omp::Clause id : leaf.synthesized) {
    if (id != llvm::omp::OMPC_shared && !(handled && handled(leaf.id, id, true))) {
      NotYet(at,
          "the " + llvm::omp::getOpenMPClauseName(id).str() +
              " clause that the " + directive + " directive implies");
      return false;
    }
  }
  return true;
}

void MerkGenerator::Omp(const Body &goal, const ProcedureBody &body) {
  const parser::OmpDirective &directive{*goal.directive};
  parser::CharBlock at{Where(goal.tokens)};
  // GPU processes' kernels are compiled without OpenMP: of the directives,
  // they run only the loops across their teams (-fprocess-loops), and
  // check-gpu-code reports any other in what the kinds reach. The module's
  // other procedures, which the kernels' compile leaves out of the image
  // (Lowering's FilterKernelProcs), run the goal as it is.
  if (target_.options().processKernels &&
      directive.id != llvm::omp::Directive::OMPD_distribute_parallel_do) {
    Goal(*goal.goals.front(), body);
    return;
  }
  if (llvm::any_of(llvm::omp::getLeafConstructsOrSelf(directive.id),
          [](llvm::omp::Directive leaf) {
            return llvm::omp::getDirectiveAssociation(leaf) ==
                llvm::omp::Association::LoopNest;
          })) {
    OmpLoop(goal, body);
    return;
  }
  if (directive.id == llvm::omp::Directive::OMPD_target_data) {
    std::optional<omp::ConstructQueue> queue{Decompose(directive, at)};
    if (queue && LeafClauses(queue->front(), at, IsTargetDataClauseHandled)) {
      TargetData(goal, body, queue->front());
    }
    return;
  }
  bool teams{directive.id == llvm::omp::Directive::OMPD_target_teams};
  if (directive.id != llvm::omp::Directive::OMPD_target && !teams) {
    NotYet(at,
        "the " + std::string{parser::EnumToString(directive.id)} +
            " directive");
    return;
  }
  std::optional<omp::ConstructQueue> queue{Decompose(directive, at)};
  if (!queue || !LeafClauses(queue->front(), at, IsTargetClauseHandled) ||
      (teams && !LeafClauses(queue->back(), at, IsTeamsClauseHandled))) {
    return;
  }
  if (builder_.getInsertionBlock()->getParent()
          ->getParentOfType<mlir::omp::TargetOp>()) {
    NotYet(at, "target regions in target regions");
    return;
  }
  std::optional<mlir::Value> device{IntegerClause(queue->front(),
      llvm::omp::OMPC_device, "device number", body, at, Location(goal.tokens))};
  if (!device) {
    return;
  }
  if (teams) {
    TargetTeams(goal, body, *queue, *device);
    return;
  }
  Target(goal, body, [&]() { Goal(*goal.goals.front(), body); },
      /*combined=*/false, KernelOf(*queue).mode, {}, {}, *device);
}

// target teams, as Flang lowers it with no loop: a generic kernel whose
// omp.teams runs the goal on each team's initial thread, while the team's
// other threads wait in the state machine (design document, Who
// communicates). Its num_teams is a host_eval value on the host, from which
// the host launches the teams, and on the device the value mapped in, read
// before omp.teams. The region maps it in both compiles, so that the
// kernel's arguments agree. The number is checked when the region starts,
// as one that is not positive, or does not fit in OpenMP's 32 bits, would
// ask libomptarget for its default or for billions of teams.
void MerkGenerator::TargetTeams(const Body &goal, const ProcedureBody &body,
    const omp::ConstructQueue &queue, mlir::Value device) {
  parser::CharBlock at{Where(goal.tokens)};
  mlir::Location loc{Location(goal.tokens)};
  std::optional<mlir::Value> number{IntegerClause(queue.back(),
      llvm::omp::OMPC_num_teams, "number of teams", body, at, loc)};
  if (!number) {
    return;
  }
  mlir::Value count;
  if (*number) {
    if (std::optional<std::int64_t> constant{
            mlir::getConstantIntValue(*number)};
        constant && (*constant <= 0 || *constant > INT32_MAX)) {
      Error(at, "the number of teams must be from 1 to %s"_err_en_US,
          std::to_string(INT32_MAX));
      return;
    }
    merk::ForeignCallOp::create(builder_, loc, mlir::TypeRange{},
        "merk_check_teams", mlir::ValueRange{*number}, /*can_fail=*/false);
    count = mlir::arith::TruncIOp::create(
        builder_, loc, builder_.getI32Type(), *number);
  }
  // The region's id and the node's, which its teams' mailboxes are found by
  // and named by (docs/INPROGRESS.md, Processes as teams, step 2).
  auto i64{builder_.getI64Type()};
  auto begin{merk::ForeignCallOp::create(builder_, loc,
      mlir::TypeRange{i64, i64, i64}, "merk_teams_begin", mlir::ValueRange{},
      /*can_fail=*/false)};
  mlir::Value id{begin.getResult(0)};
  bool host{!target_.options().openMPIsTargetDevice};
  mlir::omp::TeamsOp teamsOp;
  Target(
      goal, body,
      [&]() {
        mlir::omp::TeamsOperands operands;
        if (count) {
          operands.numTeamsUpperVars.push_back(count);
        }
        teamsOp = mlir::omp::TeamsOp::create(builder_, loc, operands);
        mlir::OpBuilder::InsertionGuard guard{builder_};
        builder_.setInsertionPointToStart(
            &teamsOp.getRegion().emplaceBlock());
        auto teams{merk::ForeignCallOp::create(builder_, loc,
            mlir::TypeRange{builder_.getI32Type()}, "omp_get_num_teams",
            mlir::ValueRange{}, /*can_fail=*/false)};
        merk::ForeignCallOp::create(builder_, loc, mlir::TypeRange{},
            "merk_device_teams_enter",
            mlir::ValueRange{id, begin.getResult(1), begin.getResult(2),
                teams.getResult(0)},
            /*can_fail=*/false);
        Goal(*goal.goals.front(), body);
        if (!failed_ && !Terminated()) {
          merk::ForeignCallOp::create(builder_, loc, mlir::TypeRange{},
              "merk_device_teams_leave", mlir::ValueRange{},
              /*can_fail=*/false);
          mlir::omp::TerminatorOp::create(builder_, loc);
        }
      },
      /*combined=*/true, KernelOf(queue).mode,
      host && count ? mlir::ValueRange{count} : mlir::ValueRange{}, {},
      device);
  if (failed_) {
    return;
  }
  // The host launches by the host_eval value, the block's first argument.
  if (host && count && teamsOp) {
    teamsOp.getNumTeamsUpperVarsMutable().assign(
        teamsOp->getParentOfType<mlir::omp::TargetOp>()
            .getRegion()
            .front()
            .getArgument(0));
  }
  merk::ForeignCallOp::create(builder_, loc, mlir::TypeRange{},
      "merk_teams_end", mlir::ValueRange{id}, /*can_fail=*/false);
}

// A clause of one integer, as Flang's ClauseProcessor makes device(N) and
// num_teams(N): the number is a non-negative integer, or an integer
// variable bound before the directive. Other terms need code in a clause,
// which nothing has yet.
std::optional<mlir::Value> MerkGenerator::IntegerClause(const omp::Leaf &leaf,
    llvm::omp::Clause id, llvm::StringRef what, const ProcedureBody &body,
    parser::CharBlock at, mlir::Location loc) {
  std::string name{llvm::omp::getOpenMPClauseName(id).str()};
  const parser::OmpGenericClause *clause{nullptr};
  for (const parser::OmpClause *c : leaf.clauses) {
    const auto *generic{std::get_if<const parser::OmpGenericClause *>(c)};
    if (!generic || (*generic)->id != id) {
      continue;
    }
    if (clause) {
      Error(at, "a directive can have only one %s clause"_err_en_US, name);
      return std::nullopt;
    }
    clause = *generic;
  }
  if (!clause) {
    return mlir::Value{};
  }
  if (!clause->modifiers.empty()) {
    NotYet(at, "modifiers of the " + name + " clause");
    return std::nullopt;
  }
  if (clause->arguments.size() != 1) {
    Error(at, "%s"_err_en_US, "the " + name + " clause takes one " + what.str());
    return std::nullopt;
  }
  const parser::DataTerm &number{clause->arguments.front()};
  if (const auto *literal{std::get_if<const parser::Literal *>(&number)}) {
    std::int64_t value;
    if ((*literal)->kind == parser::LiteralKind::Integer &&
        !(*literal)->text.getAsInteger(10, value)) {
      return IntegerConstant(builder_.getI64Type(), value, loc);
    }
  } else if (const auto *variable{
                 std::get_if<const parser::Variable *>(&number)}) {
    for (const auto &[var, value] : values_) {
      if (body.vars[var].name != (*variable)->name) {
        continue;
      }
      if (!llvm::is_contained({"int", "uint", "int8", "int16", "int32", "int64",
                                  "uint8", "uint16", "uint32", "uint64"},
              semantics::TypeToString(*body.vars[var].type))) {
        Error(at, "%s"_err_en_US,
            "the " + what.str() + " " + (*variable)->name.str() +
                " must be an integer");
        return std::nullopt;
      }
      return value;
    }
    Error(at, "%s"_err_en_US,
        "the " + what.str() + " " + (*variable)->name.str() +
            " is not bound before the directive");
    return std::nullopt;
  }
  NotYet(at, "a " + what.str() + " that is not an integer or a variable");
  return std::nullopt;
}

// An array is mapped as ClangIR maps a pointer's section, p[0:n]: by its own
// pointer, with bounds over its size word and its elements, which only the
// host needs, to copy them.
mlir::Value MerkGenerator::MapArray(mlir::Value array,
    mlir::omp::ClauseMapFlags flags, llvm::StringRef name, mlir::Location loc) {
  auto ptr{mlir::LLVM::LLVMPointerType::get(&context_)};
  auto i64{builder_.getI64Type()};
  llvm::SmallVector<mlir::Value, 1> bounds;
  if (!target_.options().openMPIsTargetDevice) {
    // Words 0 to the size.
    mlir::Value upper{merk::ArraySizeOp::create(builder_, loc, i64, array)};
    mlir::Value extent{
        mlir::arith::AddIOp::create(
            builder_, loc, upper, IntegerConstant(i64, 1, loc))};
    bounds.push_back(mlir::omp::MapBoundsOp::create(builder_, loc,
        builder_.getType<mlir::omp::MapBoundsType>(),
        IntegerConstant(i64, 0, loc), upper, extent,
        /*stride=*/IntegerConstant(i64, 1, loc), /*strideInBytes=*/false,
        /*startIdx=*/IntegerConstant(i64, 0, loc)));
  }
  mlir::Value pointer{
      mlir::UnrealizedConversionCastOp::create(builder_, loc, ptr, array)
          .getResult(0)};
  return mlir::omp::MapInfoOp::create(builder_, loc, ptr, pointer,
      mlir::TypeAttr::get(i64),
      builder_.getAttr<mlir::omp::ClauseMapFlagsAttr>(flags),
      builder_.getAttr<mlir::omp::VariableCaptureKindAttr>(
          mlir::omp::VariableCaptureKind::ByRef),
      /*varPtrPtr=*/mlir::Value{}, /*varPtrPtrType=*/mlir::TypeAttr{},
      /*members=*/mlir::ValueRange{}, /*membersIndex=*/mlir::ArrayAttr{},
      bounds, /*mapperId=*/mlir::FlatSymbolRefAttr{},
      builder_.getStringAttr(name), builder_.getBoolAttr(false));
}

// Whether a variable named `text` is a version of the state variable !`name`,
// the variable of that name that has a value.
static bool IsStateVariableVersion(llvm::StringRef text, llvm::StringRef name) {
  unsigned version;
  return text.consume_front("STATE_VARIABLE_") && text.consume_front(name) &&
      text.consume_front("_") && !text.getAsInteger(10, version);
}

// A structured data region: the arrays it maps stay on the device for its
// goal, so that the map of a target region in it finds them there and copies
// nothing. As Flang lowers !$omp target data, omp.target_data holds the goal,
// and what the goal binds leaves the region through slots.
//
// A map names a variable, or a state variable !A, whose current version is
// mapped. The directive promises that the host does not use a mapped array's
// elements in the goal, which the compiler cannot see through calls. It does
// check at run time that !A leaves as the array that came in: at the end of
// the region, the device's elements are copied to that one.
void MerkGenerator::TargetData(const Body &goal, const ProcedureBody &body,
    const omp::Leaf &leaf) {
  using namespace parser::literals;
  parser::CharBlock at{Where(goal.tokens)};
  mlir::Location loc{Location(goal.tokens)};
  if (builder_.getInsertionBlock()->getParent()
          ->getParentOfType<mlir::omp::TargetOp>()) {
    NotYet(at, "target data in target regions");
    return;
  }
  // Failing would leave the region without unmapping its arrays.
  if (auto it{body.goals.find(goal.goals.front())};
      it != body.goals.end() && it->second != parser::Determinism::Det &&
      it->second != parser::Determinism::Erroneous) {
    Error(at,
        "the goal of a target data directive must be det, but it is %s"_err_en_US,
        semantics::DeterminismName(it->second));
    return;
  }
  struct Mapped {
    llvm::StringRef name;
    BodyVar var;
    bool state;
    bool copiedBack;
  };
  llvm::SmallVector<Mapped> mapped;
  llvm::SmallVector<mlir::Value> maps;
  Values before{values_};
  for (const parser::OmpClause *clause : leaf.clauses) {
    const auto *map{std::get_if<const parser::OmpMapClause *>(clause)};
    if (!map) {
      continue;
    }
    if (!(*map)->modifiers.empty()) {
      NotYet(at, "map modifiers on the target data directive");
      return;
    }
    mlir::omp::ClauseMapFlags flags{
        mlir::omp::ClauseMapFlags::to | mlir::omp::ClauseMapFlags::from};
    switch ((*map)->type.value_or(parser::OmpMapType::Tofrom)) {
    case parser::OmpMapType::Tofrom:
      break;
    case parser::OmpMapType::To:
      flags = mlir::omp::ClauseMapFlags::to;
      break;
    case parser::OmpMapType::From:
      flags = mlir::omp::ClauseMapFlags::from;
      break;
    case parser::OmpMapType::Alloc:
      flags = mlir::omp::ClauseMapFlags::storage;
      break;
    default:
      NotYet(at, "this map type on the target data directive");
      return;
    }
    for (const parser::OmpMapObject *object : (*map)->objects) {
      if (object->lower) {
        NotYet(at, "array sections on the target data directive");
        return;
      }
      llvm::StringRef name{object->variable->name};
      std::optional<BodyVar> found;
      bool state{false};
      for (BodyVar var : goal.nonlocals) {
        if (!before.lookup(var)) {
          continue;
        }
        if (body.vars[var].name == name) {
          found = var;
        } else if (IsStateVariableVersion(body.vars[var].name, name)) {
          found = var;
          state = true;
        }
      }
      if (!found) {
        Error(at,
            "%s is not a variable or state variable that the goal of the target data directive uses"_err_en_US,
            name);
        return;
      }
      if (llvm::any_of(mapped, [&](const Mapped &m) { return m.var == *found; })) {
        Error(at, "%s is mapped twice"_err_en_US, name);
        return;
      }
      mlir::Value value{before.lookup(*found)};
      if (!mlir::isa<merk::ArrayType>(value.getType())) {
        NotYet(at,
            "mapping values of type " +
                semantics::TypeToString(*body.vars[*found].type) +
                " with the target data directive");
        return;
      }
      maps.push_back(MapArray(value, flags, name, loc));
      mapped.push_back({name, *found, state,
          mlir::omp::bitEnumContainsAll(flags, mlir::omp::ClauseMapFlags::from)});
    }
  }
  if (maps.empty()) {
    Error(at, "a target data directive must have a map clause"_err_en_US);
    return;
  }
  std::optional<mlir::Value> device{IntegerClause(
      leaf, llvm::omp::OMPC_device, "device number", body, at, loc)};
  if (!device) {
    return;
  }
  mlir::omp::TargetDataOperands operands;
  operands.device = *device;
  operands.mapVars = maps;
  auto region{mlir::omp::TargetDataOp::create(builder_, loc, operands)};
  mlir::Block *block{new mlir::Block};
  region.getRegion().push_back(block);
  // What the goal binds and is used after it leaves in a word's slot, since
  // a value in the region does not dominate what follows it.
  llvm::SmallVector<std::tuple<BodyVar, mlir::Value, mlir::Type>> outputs;
  {
    mlir::OpBuilder::InsertionGuard guard{builder_};
    builder_.setInsertionPointToStart(block);
    Goal(*goal.goals.front(), body);
    if (failed_) {
      values_ = before;
      return;
    }
    for (BodyVar var : BoundSince(values_, before)) {
      mlir::Value value{values_.lookup(var)};
      if (value && llvm::is_contained(goal.nonlocals, var)) {
        mlir::Value slot{Slot(builder_.getI64Type(), loc)};
        mlir::LLVM::StoreOp::create(builder_, loc,
            Coerce(value, builder_.getI64Type(), loc), slot);
        outputs.emplace_back(var, slot, value.getType());
      }
    }
    mlir::omp::TerminatorOp::create(builder_, loc);
  }
  values_ = before;
  for (auto [var, slot, type] : outputs) {
    values_[var] = Coerce(mlir::LLVM::LoadOp::create(
                              builder_, loc, builder_.getI64Type(), slot),
        type, loc);
  }
  // A state variable's array must leave as the array that came in, and the
  // device's updates of it must come back.
  for (const Mapped &m : mapped) {
    if (!m.state) {
      continue;
    }
    for (auto &[var, slot, type] : outputs) {
      if (!IsStateVariableVersion(body.vars[var].name, m.name)) {
        continue;
      }
      if (!m.copiedBack) {
        Error(at,
            "the goal of the target data directive updates %s, whose map type does not copy it back; use tofrom or from"_err_en_US,
            m.name);
        return;
      }
      mlir::Value in{Coerce(before.lookup(m.var), builder_.getI64Type(), loc)};
      mlir::Value out{Coerce(values_.lookup(var), builder_.getI64Type(), loc)};
      merk::ThrowIfOp::create(builder_, loc,
          mlir::arith::CmpIOp::create(builder_, loc,
              mlir::arith::CmpIPredicate::ne, in, out),
          "Software Error: target data: the array " + m.name.str() +
              " was replaced in the region, not updated in place");
    }
  }
}

// The variable whose value `input` is, if it is one's: one of `preferred`, if
// several have the value, as an array updated in place does.
std::optional<BodyVar> MerkGenerator::VarWithValue(const Values &values,
    llvm::ArrayRef<BodyVar> preferred, mlir::Value input) {
  std::optional<BodyVar> found;
  for (const auto &[var, value] : values) {
    if (value != input) {
      continue;
    }
    if (llvm::is_contained(preferred, var)) {
      return var;
    }
    found = var;
  }
  return found;
}

const MerkGenerator::UpdatedSlot *MerkGenerator::FindUpdatedSlot(
    llvm::ArrayRef<UpdatedSlot> slots, mlir::Value slot) {
  const UpdatedSlot *found{llvm::find_if(
      slots, [&](const UpdatedSlot &u) { return u.slot == slot; })};
  return found == slots.end() ? nullptr : found;
}

// As Flang maps a scalar: the map entry of a slot, by its pointer.
mlir::Value MerkGenerator::MapSlot(mlir::Value slot, mlir::Type type,
    mlir::omp::ClauseMapFlags flags, mlir::omp::VariableCaptureKind capture,
    llvm::StringRef name, mlir::Location loc) {
  return mlir::omp::MapInfoOp::create(builder_, loc,
      mlir::LLVM::LLVMPointerType::get(&context_), slot,
      mlir::TypeAttr::get(type),
      builder_.getAttr<mlir::omp::ClauseMapFlagsAttr>(flags),
      builder_.getAttr<mlir::omp::VariableCaptureKindAttr>(capture),
      /*varPtrPtr=*/mlir::Value{}, /*varPtrPtrType=*/mlir::TypeAttr{},
      /*members=*/mlir::ValueRange{}, /*membersIndex=*/mlir::ArrayAttr{},
      /*bounds=*/mlir::ValueRange{}, /*mapperId=*/mlir::FlatSymbolRefAttr{},
      builder_.getStringAttr(name), builder_.getBoolAttr(false));
}

mlir::Value MerkGenerator::ArrayPointer(mlir::Value array, mlir::Location loc) {
  return mlir::UnrealizedConversionCastOp::create(
      builder_, loc, mlir::LLVM::LLVMPointerType::get(&context_), array)
      .getResult(0);
}

void MerkGenerator::Target(const Body &goal, const ProcedureBody &body,
    llvm::function_ref<void()> generate, bool combined,
    mlir::omp::TargetExecMode mode, mlir::ValueRange hostEval,
    llvm::ArrayRef<UpdatedSlot> slots, mlir::Value device) {
  parser::CharBlock at{Where(goal.tokens)};
  mlir::Location loc{Location(goal.tokens)};
  Values before{values_};
  // What `generate` builds, in a target region whose map entries are added
  // after it: they are the values it uses from outside and those it binds.
  // The host_eval arguments come before the map entries' arguments.
  mlir::omp::TargetExtOperands operands;
  operands.kernelType = mlir::omp::TargetExecModeAttr::get(&context_, mode);
  operands.device = device;
  operands.hostEvalVars.append(hostEval.begin(), hostEval.end());
  // A region on a GPU whose processes' kernel holds every unit would never
  // start (gpu-processes.cpp): the device's number, or -1 for the default.
  merk::ForeignCallOp::create(builder_, loc, mlir::TypeRange{},
      "merk_check_region_units",
      mlir::ValueRange{
          device ? device : IntegerConstant(builder_.getI64Type(), -1, loc)},
      /*can_fail=*/false);
  // The host counts the running regions, and gives each a slot, whose
  // chunks on the device hold the cells the region makes, freed as it ends
  // (docs/INPROGRESS.md, The device's side of the first slice, step 6).
  auto regionSlot{merk::ForeignCallOp::create(builder_, loc,
      mlir::TypeRange{builder_.getI64Type()}, "merk_region_enter",
      mlir::ValueRange{}, /*can_fail=*/false)
          .getResult(0)};
  auto targetOp{mlir::omp::TargetOp::create(builder_, loc, operands)};
  targetOp.setCombined(combined);
  mlir::Region &region{targetOp.getRegion()};
  mlir::Block *block{&region.emplaceBlock()};
  for (mlir::Value value : hostEval) {
    block->addArgument(value.getType(), loc);
  }
  {
    mlir::OpBuilder::InsertionGuard guard{builder_};
    builder_.setInsertionPointToStart(block);
    generate();
  }
  if (failed_) {
    values_ = before;
    return;
  }
  {
    // Each team says which slot is the region's before it makes a cell: in
    // omp.teams, and in a parallel region nested in it, where the ops of
    // an SPMD kernel are, after the inputs that the region reads.
    mlir::OpBuilder::InsertionGuard guard{builder_};
    mlir::Block *enter{block};
    if (auto teams{enter->empty()
                ? nullptr
                : mlir::dyn_cast<mlir::omp::TeamsOp>(enter->front())}) {
      enter = &teams.getRegion().front();
      if (auto parallel{enter->empty()
                  ? nullptr
                  : mlir::dyn_cast<mlir::omp::ParallelOp>(enter->front())}) {
        enter = &parallel.getRegion().front();
      }
    }
    builder_.setInsertionPointToStart(enter);
    merk::ForeignCallOp::create(builder_, loc, mlir::TypeRange{},
        "merk_device_region_enter", mlir::ValueRange{regionSlot},
        /*can_fail=*/false);
  }
  builder_.setInsertionPoint(targetOp);
  llvm::SetVector<mlir::Value> inputs;
  mlir::getUsedValuesDefinedAbove(region, inputs);
  // What it binds that is used after it: those of its nonlocals it binds.
  llvm::SmallVector<BodyVar> outputs;
  for (BodyVar var : BoundSince(values_, before)) {
    if (values_.lookup(var) && llvm::is_contained(goal.nonlocals, var)) {
      outputs.push_back(var);
    }
  }
  // A teams region with no loop runs its goal in every team, which would
  // each bind the output.
  bool teamsGoal{mode == mlir::omp::TargetExecMode::generic &&
      !block->empty() && mlir::isa<mlir::omp::TeamsOp>(block->front())};
  if (teamsGoal && !outputs.empty()) {
    Error(at,
        "a target teams region must not bind %s, which is used after it, "
        "since every team would bind it"_err_en_US,
        body.vars[outputs.front()].name);
    values_ = before;
    return;
  }
  for (mlir::Value input : inputs) {
    if (!FindUpdatedSlot(slots, input) && !Mappable(input.getType())) {
      std::optional<BodyVar> var{VarWithValue(before, goal.nonlocals, input)};
      NotYet(at,
          var ? "values of type " +
                  semantics::TypeToString(*body.vars[*var].type) +
                  " in target regions"
              : std::string{"values that are not scalars in target regions"});
      values_ = before;
      return;
    }
  }
  // An array the region gives may be one it was given, updated in place:
  // it is mapped back, and after the region is the given array. When the
  // region's operations do not show which, as when it calls a procedure
  // that updates it, the region finds which given array of its type it is,
  // by its pointer, and gives its number. If it is none of them, the region
  // made it, and gives its pointer too: on a GPU that of a copy on the
  // host, since the device's arena ends with the region
  // (merk_device_give_array).
  llvm::DenseMap<BodyVar, mlir::Value> updated;
  llvm::DenseMap<BodyVar, llvm::SmallVector<mlir::Value>> candidates;
  llvm::DenseMap<BodyVar, mlir::Type> arrayTypes;
  // A constant of a type of constants only is one word, and leaves as one.
  llvm::DenseMap<BodyVar, mlir::Type> constants;
  for (BodyVar var : outputs) {
    mlir::Value value{values_.lookup(var)};
    if (mlir::isa<merk::TermType>(value.getType()) &&
        WordMessage(value.getType())) {
      constants[var] = value.getType();
      continue;
    }
    if (!Givable(value.getType())) {
      NotYet(at,
          "values of type " + semantics::TypeToString(*body.vars[var].type) +
              " in target regions");
      values_ = before;
      return;
    }
    if (mlir::isa<merk::ArrayType>(value.getType())) {
      mlir::Value from{UpdatedFrom(value)};
      if (inputs.contains(from)) {
        updated[var] = from;
        continue;
      }
      arrayTypes[var] = value.getType();
      llvm::SmallVector<mlir::Value> &same{candidates[var]};
      for (mlir::Value input : inputs) {
        if (input.getType() == value.getType()) {
          same.push_back(input);
        }
      }
    }
  }
  llvm::erase_if(outputs, [&](BodyVar var) { return updated.contains(var); });
  // What is mapped back: what an output is, or may be.
  llvm::SmallPtrSet<mlir::Value, 4> mappedBack;
  for (const auto &[var, from] : updated) {
    mappedBack.insert(from);
  }
  for (const auto &[var, same] : candidates) {
    mappedBack.insert(same.begin(), same.end());
  }
  // As Flang maps a scalar: an input is copied in from its slot, an output
  // is stored in its slot, from which it is read after the region.
  auto ptr{mlir::LLVM::LLVMPointerType::get(&context_)};
  llvm::SmallVector<mlir::Value> maps;
  for (mlir::Value input : inputs) {
    if (const UpdatedSlot * u{FindUpdatedSlot(slots, input)}) {
      maps.push_back(MapSlot(input, u->type,
          mlir::omp::ClauseMapFlags::to | mlir::omp::ClauseMapFlags::from,
          mlir::omp::VariableCaptureKind::ByRef, u->name, loc));
      continue;
    }
    std::optional<BodyVar> var{VarWithValue(before, goal.nonlocals, input)};
    llvm::StringRef name{var      ? llvm::StringRef{body.vars[*var].name}
            : input == regionSlot ? "region_slot"
                                  : ""};

    if (mlir::isa<merk::ArrayType>(input.getType())) {
      maps.push_back(MapArray(input,
          mappedBack.contains(input)
              ? mlir::omp::ClauseMapFlags::to | mlir::omp::ClauseMapFlags::from
              : mlir::omp::ClauseMapFlags::to,
          name, loc));
      continue;
    }
    if (std::optional<unsigned> words{FlatCell(input.getType())}) {
      maps.push_back(MapCell(input, *words, name, loc));
      continue;
    }
    mlir::Value slot{Slot(input.getType(), loc)};
    mlir::LLVM::StoreOp::create(builder_, loc, input, slot);
    // Named after the variable whose value it is, if it is one's.
    maps.push_back(MapSlot(slot, input.getType(), mlir::omp::ClauseMapFlags::to,
        mlir::omp::VariableCaptureKind::ByCopy, name, loc));
  }
  llvm::SmallVector<mlir::Value> outputSlots;
  llvm::SmallVector<mlir::Type> outputTypes;
  for (BodyVar var : outputs) {
    // An array whose given array the region finds gives its number.
    mlir::Type type{candidates.contains(var) || constants.contains(var)
            ? builder_.getI64Type()
            : values_.lookup(var).getType()};
    outputTypes.push_back(type);
    outputSlots.push_back(Slot(type, loc));
    maps.push_back(
        MapSlot(outputSlots.back(), type, mlir::omp::ClauseMapFlags::from,
            mlir::omp::VariableCaptureKind::ByRef, body.vars[var].name, loc));
  }
  llvm::DenseMap<BodyVar, mlir::Value> pointerSlots;
  for (BodyVar var : outputs) {
    if (candidates.contains(var)) {
      pointerSlots[var] = Slot(ptr, loc);
      maps.push_back(
          MapSlot(pointerSlots[var], ptr, mlir::omp::ClauseMapFlags::from,
              mlir::omp::VariableCaptureKind::ByRef, body.vars[var].name, loc));
    }
  }
  targetOp.getMapVarsMutable().assign(maps);
  // The region reads its inputs from their slots and stores its outputs;
  // an output may be an input's value, as in Y = X. A region that is
  // teams around a parallel region reads them in the parallel, as Flang
  // does, for the reason OmpLoop computes its bounds there. One whose teams
  // run a goal reads them before omp.teams, whose number of teams may be
  // one of them.
  mlir::Block *reads{block};
  if (auto teams{mlir::dyn_cast<mlir::omp::TeamsOp>(block->front())};
      teams && outputs.empty() && !teamsGoal) {
    reads = &teams.getRegion().front();
    if (auto parallel{mlir::dyn_cast<mlir::omp::ParallelOp>(reads->front())}) {
      reads = &parallel.getRegion().front();
    }
  }
  builder_.setInsertionPointToStart(reads);
  llvm::DenseMap<mlir::Value, mlir::Value> loaded;
  for (mlir::Value input : inputs) {
    // An array's argument is its pointer on the device, and so is a
    // term's and an updated slot's.
    mlir::Value argument{block->addArgument(ptr, loc)};
    mlir::Value value{FindUpdatedSlot(slots, input) ? argument
            : mlir::isa<merk::ArrayType>(input.getType()) ||
                FlatCell(input.getType())
            ? mlir::UnrealizedConversionCastOp::create(
                  builder_, loc, input.getType(), argument)
                  .getResult(0)
            : mlir::Value{mlir::LLVM::LoadOp::create(
                  builder_, loc, input.getType(), argument)}};
    mlir::replaceAllUsesInRegionWith(input, value, targetOp.getRegion());
    loaded[input] = value;
  }
  llvm::SmallVector<mlir::Value> outputArgs;
  for (size_t i = 0; i < outputs.size(); ++i) {
    outputArgs.push_back(block->addArgument(ptr, loc));
  }
  llvm::DenseMap<BodyVar, mlir::Value> pointerArgs;
  for (BodyVar var : outputs) {
    if (candidates.contains(var)) {
      pointerArgs[var] = block->addArgument(ptr, loc);
    }
  }
  // The maps are decided: the host's nest runs over the host_eval values,
  // the block's first arguments.
  if (!hostEval.empty()) {
    region.walk([&](mlir::omp::LoopNestOp nest) {
      nest.getLoopLowerBoundsMutable().assign(block->getArgument(0));
      nest.getLoopUpperBoundsMutable().assign(block->getArgument(1));
      nest.getLoopStepsMutable().assign(block->getArgument(2));
    });
  }
  // Where the goal ended: the last block of the region.
  mlir::Block *end{&targetOp.getRegion().back()};
  builder_.setInsertionPointToEnd(end);
  if (!Terminated()) {
    for (auto [var, arg] : llvm::zip(outputs, outputArgs)) {
      mlir::Value value{values_.lookup(var)};
      if (auto same{candidates.find(var)}; same != candidates.end()) {
        // The number of the given array whose pointer it is, or -1.
        auto i64{builder_.getI64Type()};
        mlir::Value number{IntegerConstant(i64, -1, loc)};
        for (auto [i, input] : llvm::enumerate(same->second)) {
          mlir::Value is{mlir::LLVM::ICmpOp::create(builder_, loc,
              mlir::LLVM::ICmpPredicate::eq, ArrayPointer(value, loc),
              ArrayPointer(loaded.lookup(input), loc))};
          number = mlir::arith::SelectOp::create(
              builder_, loc, is, IntegerConstant(i64, i, loc), number);
        }
        mlir::LLVM::StoreOp::create(builder_, loc,
            merk::ForeignCallOp::create(builder_, loc, mlir::TypeRange{ptr},
                "merk_device_give_array",
                mlir::ValueRange{ArrayPointer(value, loc), number},
                /*can_fail=*/false)
                .getResult(0),
            pointerArgs.lookup(var));
        value = number;
      }
      if (loaded.lookup(value)) {
        value = loaded.lookup(value);
      }
      if (constants.contains(var)) {
        value = Coerce(value, builder_.getI64Type(), loc);
      }
      mlir::LLVM::StoreOp::create(builder_, loc, value, arg);
    }
    mlir::omp::TerminatorOp::create(builder_, loc);
  }
  builder_.setInsertionPointAfter(targetOp);
  merk::ForeignCallOp::create(builder_, loc, mlir::TypeRange{},
      "merk_region_exit", mlir::ValueRange{regionSlot},
      /*can_fail=*/false);
  values_ = before;
  for (auto [var, type, slot] : llvm::zip(outputs, outputTypes, outputSlots)) {
    values_[var] = mlir::LLVM::LoadOp::create(builder_, loc, type, slot);
    if (auto constant{constants.find(var)}; constant != constants.end()) {
      values_[var] = Coerce(values_[var], constant->second, loc);
    }
    if (auto same{candidates.find(var)}; same != candidates.end()) {
      // Chosen among the pointers, which the arrays become.
      mlir::Value number{values_[var]};
      mlir::Value chosen{mlir::LLVM::LoadOp::create(
          builder_, loc, ptr, pointerSlots.lookup(var))};
      for (auto [i, input] : llvm::enumerate(same->second)) {
        mlir::Value is{mlir::arith::CmpIOp::create(builder_, loc,
            mlir::arith::CmpIPredicate::eq, number,
            IntegerConstant(type, i, loc))};
        chosen = mlir::LLVM::SelectOp::create(
            builder_, loc, is, ArrayPointer(input, loc), chosen);
      }
      values_[var] = mlir::UnrealizedConversionCastOp::create(
          builder_, loc, arrayTypes.lookup(var), chosen)
                         .getResult(0);
    }
  }
  for (auto [var, from] : updated) {
    values_[var] = from;
  }
}

} // namespace merkurius::codegen::mlirgen
