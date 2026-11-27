//===-- lib/CodeGen/MLIRGenCalls.cpp - Calls ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Calls of procedures, with their hidden arguments, the word copies of
// procedures, closures, lambda expressions and higher-order calls.
//
//===----------------------------------------------------------------------===//

#include "MLIRGenImpl.h"

namespace merkurius::codegen::mlirgen {

// Whether another module's predicate or function has a definition, a clause
// or a C foreign_proc, in its module's file, so that its procedure is there
// to call; true where that is not known, and for this module's and type
// class methods.
bool MerkGenerator::Defined(const Procedure &procedure) const {
  const semantics::Symbol &symbol{*procedure.symbol};
  bool function{symbol.kind == semantics::SymbolKind::Function};
  if (symbol.module == moduleName_ ||
      (!function && symbol.kind != semantics::SymbolKind::Predicate) ||
      (symbol.owner &&
          symbol.owner->kind == semantics::SymbolKind::Typeclass)) {
    return true;
  }
  for (const semantics::ModuleInterface *module : resolution_.interfaces()) {
    if (module->name == symbol.module) {
      return !module->definitionsKnown ||
          module->defined.contains(
              semantics::DefinitionKey(function, symbol.name, symbol.arity));
    }
  }
  return true;
}

merk::ProcOp MerkGenerator::ProcFor(const Procedure &procedure,
    llvm::ArrayRef<const semantics::Type *> types, parser::CharBlock at) {
  if (merk::ProcOp proc{procs_.lookup(&procedure)}) {
    return proc;
  }
  if (!Defined(procedure)) {
    const semantics::Symbol &symbol{*procedure.symbol};
    NotYet(at,
        ("calls of " + symbol.module + "." + symbol.name + "/" +
            llvm::Twine(symbol.arity) +
            ", which its module declares but does not define yet")
            .str());
    return nullptr;
  }
  std::optional<parser::Determinism> determinism{
      determinism_.DeterminismOf(procedure)};
  std::optional<merk::Determinism> merkDeterminism;
  if (determinism) {
    merkDeterminism = ClosureDeterminism(*determinism);
  }
  if (!merkDeterminism) {
    NotYet(at,
        determinism
            ? std::string{semantics::DeterminismName(*determinism)} +
                " procedures"
            : std::string{"procedures of unknown determinism"});
    return nullptr;
  }
  std::optional<llvm::SmallVector<mlir::Type>> hidden{
      HiddenInputs(procedure, at)};
  if (!hidden) {
    return nullptr;
  }
  merk::ProcOp proc{NewProc(MangledName(procedure), at, *hidden, procedure.modes,
      types, *merkDeterminism, HiddenOutputs(procedure))};
  if (proc) {
    procs_[&procedure] = proc;
    MarkInlining(proc, procedure);
  }
  return proc;
}

merk::ProcOp MerkGenerator::NewProc(llvm::StringRef name, parser::CharBlock at,
    llvm::ArrayRef<mlir::Type> hidden, llvm::ArrayRef<semantics::ArgMode> modes,
    llvm::ArrayRef<const semantics::Type *> types,
    merk::Determinism determinism, llvm::ArrayRef<mlir::Type> hiddenOutputs) {
  llvm::SmallVector<mlir::Type> inputs{hidden};
  llvm::SmallVector<mlir::Type> outputs{hiddenOutputs};
  // The inputs that the procedure clobbers, which FreeAtLastUse may free.
  llvm::SmallVector<unsigned> clobbered;
  for (auto [mode, type] : llvm::zip(modes, types)) {
    std::optional<Represented> represented{Represent(type)};
    if (!represented) {
      NotYet(at, "values of type " + semantics::TypeToString(*type));
      return nullptr;
    }
    // A closed(T) that may be unbound passes as its open word.
    mlir::Type passed{OpenParameter(mode, type)
            ? mlir::Type{merk::WordType::get(&context_)}
            : represented->type};
    if (!represented->dummy && !IsUnused(mode)) {
      if (IsOutput(mode)) {
        outputs.push_back(passed);
      } else {
        if (IsClobbered(mode)) {
          clobbered.push_back(inputs.size());
        }
        inputs.push_back(passed);
      }
    }
  }
  mlir::OpBuilder::InsertionGuard guard{builder_};
  builder_.setInsertionPointToEnd(module_.getBody());
  auto proc{merk::ProcOp::create(builder_, Location(at), name,
      builder_.getFunctionType(inputs, outputs), determinism)};
  proc.setPrivate();
  for (unsigned index : clobbered) {
    proc.setArgAttr(index, "merk.clobbered", builder_.getUnitAttr());
  }
  return proc;
}

void MerkGenerator::Call(const Body &goal, const Procedure &callee,
    llvm::ArrayRef<BodyVar> arguments, const ProcedureBody &body) {
  if (callee.modes.size() != arguments.size()) {
    NotYet(Where(goal.tokens), "calls with hidden arguments");
    return;
  }
  if (goal.modes && goal.modes->implied) {
    // An implied mode: the call outputs to fresh values the arguments that
    // are already bound, and each is then tested against its own value.
    // The value stays for the call to read, as A = A mod B also passes A
    // as an input; the call reads its inputs before it sets its outputs.
    // A partially instantiated term has no value, only bound parts to test.
    llvm::SmallVector<std::pair<BodyVar, mlir::Value>> bound;
    llvm::SmallVector<std::pair<BodyVar, const Body *>> partials;
    for (auto [argument, mode] : llvm::zip(arguments, callee.modes)) {
      if (const Body *partial{PartialTerm(body, argument)};
          partial && IsOutput(mode) && !values_.lookup(argument)) {
        partials.push_back({argument, partial});
      } else if (IsOutput(mode) && ValueOf(argument) &&
          !Represent(body.vars[argument].type)->dummy &&
          !llvm::is_contained(llvm::make_first_range(bound), argument)) {
        bound.push_back({argument, values_.lookup(argument)});
      }
    }
    Called(goal, callee, arguments, body);
    for (auto [argument, value] : bound) {
      mlir::Value output{ValueOf(argument)};
      values_[argument] = value;
      if (failed_ || Terminated()) {
        continue;
      }
      Test(goal, value, output, body.vars[argument].type);
    }
    for (auto [argument, partial] : partials) {
      if (!failed_ && !Terminated()) {
        TestPartial(goal, ValueOf(argument), *partial, body);
      }
    }
    return;
  }
  Called(goal, callee, arguments, body);
  FormProcessOperation(goal, callee, arguments, body);
}

// process.any's procedure, a plain receive's matcher that takes every
// message (Lowering.cpp, LowerSelect).
constexpr llvm::StringLiteral kAnyMatcher{"process.any_2_0"};

namespace {
// A value as it was before Coerce made a word of it, or a value of another
// type from that word.
mlir::Value Uncoerced(mlir::Value value) {
  while (mlir::Operation * op{value.getDefiningOp()}) {
    if (auto fromWord{mlir::dyn_cast<merk::FromWordOp>(op)}) {
      value = fromWord.getWord();
    } else if (auto toWord{mlir::dyn_cast<merk::ToWordOp>(op)}) {
      value = toWord.getValue();
    } else {
      break;
    }
  }
  return value;
}

// Whether an operation is part of a receive goal's selector: a call of
// process.selecting_matching or process.new_selector, or a closure.
bool IsSelectorPart(mlir::Operation *op) {
  if (auto link{mlir::dyn_cast_or_null<merk::CallOp>(op)}) {
    return link.getCallee().starts_with("process.selecting_matching") ||
        link.getCallee().starts_with("process.new_selector");
  }
  return mlir::isa_and_nonnull<merk::ClosureOp>(op);
}
} // namespace

// Desugaring makes a receive goal a selector of a case for each mailbox,
// the last added outermost, each process.selecting_matching(Mailbox,
// Matcher, Rest) down to process.new_selector, then a call of
// process.receive_goal (desugar.cpp, DesugarReceive). Here, that call, just
// generated, becomes a merk.receive of the cases, first written first, each
// with its mailbox, its messages' type_info, and its matcher's procedure
// with the values the closure captures, in the types the procedure takes.
// The selector and the closures go: lowering makes them again where it
// calls the library. Its result has the goal's type, not the library's
// receive_result(T0), from which lowering knows its selectors', which the
// collector needs. A call of receive_goal that is not of that shape, all
// in one block, stays a call. A call of send/4 or receive/5 is its
// operation too (FormSend, FormPlainReceive), as are those of the
// mailboxes and monitors (FormMailboxOperation).
void MerkGenerator::FormProcessOperation(const Body &goal,
    const Procedure &callee, llvm::ArrayRef<BodyVar> arguments,
    const ProcedureBody &body) {
  const semantics::Symbol &symbol{*callee.symbol};
  if (failed_ || Terminated() || symbol.module != "process") {
    return;
  }
  if (symbol.name == "send" && symbol.arity == 4) {
    FormSend();
    return;
  }
  if ((symbol.arity == 3 &&
          llvm::is_contained({"new_mailbox", "close_mailbox", "stop",
                                 "demonitor"},
              symbol.name)) ||
      (symbol.arity == 6 && symbol.name == "monitor")) {
    FormMailboxOperation(symbol.name);
    return;
  }
  if (symbol.name == "receive" && symbol.arity == 6) {
    FormPlainReceive(arguments, body);
    return;
  }
  if (symbol.name != "receive_goal" || symbol.arity != 5) {
    return;
  }
  // A call that stays is the library's, which a GPU cannot run, as its
  // selector is closures; the device's compile rejects it where the device
  // runs it, as other host-only library code.
  (void)FormReceiveOp(goal, arguments, body);
}

// FormReceive's merk.receive: false if the call is not of desugaring's
// shape. In a device's compile, the operation records each case's ring kind
// (device-ring.h), as its sends choose it (SendInTarget), in
// merk.ring_kinds, for lowering (Lowering.cpp, LowerSelect); or, if the
// device cannot make the receive, why in merk.not_on_device, which lowering
// reports where the device runs it (RejectOnDevice), as the procedure may
// run only on the host. In any compile, merk.arms_type is the type of what
// the matchers give, if it is known.
bool MerkGenerator::FormReceiveOp(const Body &goal,
    llvm::ArrayRef<BodyVar> arguments, const ProcedureBody &body) {
  mlir::Value result{values_.lookup(arguments[2])};
  if (!result) {
    return false;
  }
  auto call{Uncoerced(result).getDefiningOp<merk::CallOp>()};
  if (!call || !call.getCallee().starts_with("process.receive_goal") ||
      call.getInputs().size() != 3 || call->getNumResults() != 1) {
    return false;
  }
  mlir::Block *block{call->getBlock()};
  struct Case {
    mlir::Value mailbox;
    mlir::Value info;
    merk::ClosureOp closure;
    merk::ProcOp matcher;
  };
  llvm::SmallVector<Case> cases;
  mlir::Value selector{call.getInputs()[1]};
  for (mlir::Value at{selector};;) {
    auto link{Uncoerced(at).getDefiningOp<merk::CallOp>()};
    if (!link || link->getBlock() != block) {
      return false;
    }
    if (link.getCallee().starts_with("process.new_selector") &&
        link.getInputs().size() == 1) {
      break;
    }
    if (!link.getCallee().starts_with("process.selecting_matching") ||
        link.getInputs().size() != 5) {
      return false;
    }
    auto closure{
        Uncoerced(link.getInputs()[3]).getDefiningOp<merk::ClosureOp>()};
    if (!closure || closure->getBlock() != block) {
      return false;
    }
    auto lambda{entryLambdas_.find(closure.getEntry())};
    if (lambda == entryLambdas_.end()) {
      return false;
    }
    auto matcher{module_.lookupSymbol<merk::ProcOp>(lambda->second)};
    if (!matcher ||
        matcher.getFunctionType().getNumInputs() !=
            closure.getCaptured().size() + 1) {
      return false;
    }
    cases.push_back(
        {link.getInputs()[2], link.getInputs()[0], closure, matcher});
    at = link.getInputs()[4];
  }
  if (cases.empty()) {
    return false;
  }
  std::reverse(cases.begin(), cases.end());
  // On a GPU, each case's ring kind; the runtime takes a receive of a few
  // cases (device.cpp, kMostSelectCases).
  llvm::SmallVector<std::int64_t> kinds;
  std::string notOnDevice;
  if (target_.options().IsDevice()) {
    constexpr std::size_t kMostCases{8};
    if (cases.size() > kMostCases) {
      notOnDevice = "selects with more than " + std::to_string(kMostCases) +
          " cases in one receive goal";
    }
    for (Case &c : cases) {
      mlir::Type message{c.matcher.getFunctionType().getInputs().back()};
      if (WordMessage(message)) {
        kinds.push_back(0);
      } else if (ArrayMessage(message)) {
        kinds.push_back(1);
      } else if (Uncoerced(c.info).getDefiningOp<merk::TypeInfoOp>()) {
        // A copied term, of a type whose type_info is known at compile
        // time, as a send's must be (CopiedMessage).
        kinds.push_back(2);
      } else {
        notOnDevice = "receives from a mailbox of a polymorphic type in a "
                      "receive goal";
      }
    }
  }
  mlir::Location loc{call.getLoc()};
  mlir::OpBuilder::InsertionGuard guard{builder_};
  builder_.setInsertionPoint(call);
  llvm::SmallVector<mlir::Value> mailboxes, infos, captured;
  llvm::SmallVector<mlir::Attribute> matchers, captures;
  llvm::SmallVector<std::int64_t> counts;
  bool layouts{true};
  for (Case &c : cases) {
    // The mailbox as the goal had it, not as the library's call takes it.
    mlir::Value mailbox{Uncoerced(c.mailbox)};
    mailboxes.push_back(
        mlir::isa<merk::TermType>(mailbox.getType()) ? mailbox : c.mailbox);
    infos.push_back(c.info);
    mlir::FunctionType type{c.matcher.getFunctionType()};
    for (auto [i, word] : llvm::enumerate(c.closure.getCaptured())) {
      captured.push_back(Coerce(word, type.getInput(i), loc));
    }
    matchers.push_back(
        mlir::FlatSymbolRefAttr::get(&context_, c.matcher.getSymName()));
    counts.push_back(static_cast<std::int64_t>(c.closure.getCaptured().size()));
    if (std::optional<mlir::ArrayAttr> layout{c.closure.getCaptures()}) {
      captures.push_back(*layout);
    } else {
      layouts = false;
    }
  }
  mlir::Type resultType{call.getResult(0).getType()};
  if (std::optional<Represented> represented{
          Represent(body.vars[arguments[2]].type)};
      represented && mlir::isa<merk::TermType>(represented->type)) {
    resultType = represented->type;
  }
  auto receive{merk::ReceiveOp::create(builder_, loc, resultType,
      call.getInputs()[2], call.getInputs()[0], mailboxes, infos, captured,
      builder_.getArrayAttr(matchers), builder_.getDenseI64ArrayAttr(counts),
      layouts ? builder_.getArrayAttr(captures) : mlir::ArrayAttr{})};
  // The type of what a matcher gives, receive_result(X)'s X, which the
  // receive makes of the matcher's word (Lowering.cpp,
  // ExpandReceiveAsSelect).
  const semantics::Type *received{body.vars[arguments[2]].type};
  std::optional<Represented> arms;
  if (received->kind == semantics::TypeKind::Named &&
      received->arguments.size() == 1) {
    arms = Represent(received->arguments.front());
  }
  if (arms) {
    receive->setAttr("merk.arms_type", mlir::TypeAttr::get(arms->type));
  }
  if (!notOnDevice.empty()) {
    receive->setAttr("merk.not_on_device", builder_.getStringAttr(notOnDevice));
  } else if (!kinds.empty()) {
    if (arms) {
      receive->setAttr("merk.ring_kinds", builder_.getDenseI64ArrayAttr(kinds));
    } else {
      receive->setAttr("merk.not_on_device",
          builder_.getStringAttr("receives a result of an unknown type in a "
                                 "receive goal"));
    }
  }
  ReplaceReceiveCall(call, receive.getResult(), loc);
  return true;
}

// A receive's call, replaced by its merk.receive's result: the call, the
// selector's calls and closures that nothing else uses, and the coercions
// between them go. They have no effects. A coercion that is a variable's
// value, as a mailbox's may be, stays, unless it is of the selector or a
// closure.
void MerkGenerator::ReplaceReceiveCall(
    merk::CallOp call, mlir::Value result, mlir::Location loc) {
  mlir::Block *block{call->getBlock()};
  mlir::Value old{call.getResult(0)};
  mlir::Value got{Coerce(result, old.getType(), loc)};
  old.replaceAllUsesWith(got);
  for (auto &[var, value] : values_) {
    if (value == old) {
      value = got;
    }
  }
  llvm::DenseSet<mlir::Value> held;
  for (const auto &[var, value] : values_) {
    held.insert(value);
  }
  llvm::SmallVector<mlir::Operation *> work{call};
  llvm::DenseSet<mlir::Value> erased;
  llvm::DenseSet<mlir::Operation *> erasedOps;
  while (!work.empty()) {
    mlir::Operation *op{work.pop_back_val()};
    if (erasedOps.contains(op) || !op->use_empty()) {
      continue;
    }
    llvm::SmallVector<mlir::Value> operands{op->getOperands()};
    erased.insert(op->result_begin(), op->result_end());
    erasedOps.insert(op);
    op->erase();
    for (mlir::Value operand : operands) {
      mlir::Operation *from{operand.getDefiningOp()};
      if (!from || from->getBlock() != block) {
        continue;
      }
      if (IsSelectorPart(from) ||
          (mlir::isa<merk::ToWordOp, merk::FromWordOp>(from) &&
              (!held.contains(operand) ||
                  IsSelectorPart(Uncoerced(operand).getDefiningOp())))) {
        work.push_back(from);
      }
    }
  }
  // No variable keeps a value of what went.
  llvm::SmallVector<BodyVar> gone;
  for (const auto &[var, value] : values_) {
    if (erased.contains(value)) {
      gone.push_back(var);
    }
  }
  for (BodyVar var : gone) {
    values_.erase(var);
  }
}

// The call of `callee`, a procedure of no more than `inputs` inputs, just
// generated: the last operation before the insertion point but the
// coercions of its results; or null if that is not such a call.
merk::CallOp MerkGenerator::JustCalled(
    llvm::StringRef callee, unsigned inputs) {
  mlir::Block *block{builder_.getInsertionBlock()};
  if (!block) {
    return {};
  }
  for (auto at{builder_.getInsertionPoint()}; at != block->begin();) {
    mlir::Operation *op{&*--at};
    if (mlir::isa<merk::ToWordOp, merk::FromWordOp>(op)) {
      continue;
    }
    auto call{mlir::dyn_cast<merk::CallOp>(op)};
    if (!call || call.getCallee() != callee ||
        call.getInputs().size() != inputs) {
      return {};
    }
    return call;
  }
  return {};
}

namespace {
// A mailbox as its own type, before Coerce made a word of it, if that is a
// term.
mlir::Value MailboxTerm(mlir::Value mailbox) {
  mlir::Value term{Uncoerced(mailbox)};
  return mlir::isa<merk::TermType>(term.getType()) ? term : mailbox;
}

// Every use of a call's result, and every variable that has it, given
// another value instead.
void ReplaceResult(mlir::Value from, mlir::Value to,
    llvm::DenseMap<BodyVar, mlir::Value> &values) {
  from.replaceAllUsesWith(to);
  for (auto &[var, value] : values) {
    if (value == from) {
      value = to;
    }
  }
}

// Marks a process operation that a device cannot make, saying why
// (Lowering.cpp, RejectDeviceMessages).
void NotOnDevice(mlir::Operation *op, llvm::StringRef why) {
  op->setAttr(
      "merk.not_on_device", mlir::StringAttr::get(op->getContext(), why));
}
} // namespace

// A call of process.send/4, just generated, that SendInTarget did not make
// the device runtime's (docs/INPROGRESS.md, Concurrency in merk, step 2):
// a merk.send of the message, to the mailbox, with the message type's
// type_info, which the call takes first. On a device it says why the
// device cannot make it, in merk.not_on_device, which lowering reports
// where the device runs it (Lowering.cpp, RejectDeviceMessages).
void MerkGenerator::FormSend() {
  merk::CallOp call{JustCalled("process.send_4_0", 3)};
  if (!call || call->getNumResults() != 0) {
    return;
  }
  mlir::OpBuilder::InsertionGuard guard{builder_};
  builder_.setInsertionPoint(call);
  auto send{merk::SendOp::create(builder_, call.getLoc(),
      Uncoerced(call.getInputs()[2]), MailboxTerm(call.getInputs()[1]),
      call.getInputs()[0], /*message_type=*/nullptr, /*ring_kind=*/nullptr)};
  if (target_.options().IsDevice()) {
    NotOnDevice(send, "sends messages of a type that a device does not send");
  }
  call->erase();
}

// A call of new_mailbox/3, close_mailbox/3, stop/3, monitor/6 or
// demonitor/3, just generated, as its merk operation (docs/INPROGRESS.md,
// Concurrency in merk, step 2 (b)), with the type_infos the call takes, so
// that lowering can make the same call again where it must. On a device,
// which has none of them, each says so (merk.not_on_device).
void MerkGenerator::FormMailboxOperation(llvm::StringRef name) {
  mlir::OpBuilder::InsertionGuard guard{builder_};
  if (name == "new_mailbox") {
    merk::CallOp call{JustCalled("process.new_mailbox_3_0", 1)};
    if (!call || call->getNumResults() != 1) {
      return;
    }
    builder_.setInsertionPoint(call);
    auto op{merk::NewMailboxOp::create(builder_, call.getLoc(),
        call.getResult(0).getType(), call.getInputs()[0])};
    ReplaceResult(call.getResult(0), op.getMailbox(), values_);
    call->erase();
    if (target_.options().IsDevice()) {
      NotOnDevice(op, "makes a mailbox");
    }
    return;
  }
  if (name == "close_mailbox" || name == "stop") {
    merk::CallOp call{JustCalled(
        name == "stop" ? "process.stop_3_0" : "process.close_mailbox_3_0", 2)};
    if (!call || call->getNumResults() != 0) {
      return;
    }
    builder_.setInsertionPoint(call);
    mlir::Value mailbox{MailboxTerm(call.getInputs()[1])};
    mlir::Operation *formed;
    if (name == "stop") {
      formed = merk::StopOp::create(
          builder_, call.getLoc(), mailbox, call.getInputs()[0]);
    } else {
      formed = merk::CloseMailboxOp::create(
          builder_, call.getLoc(), mailbox, call.getInputs()[0]);
    }
    call->erase();
    if (target_.options().IsDevice()) {
      NotOnDevice(formed, name == "stop" ? "stops a mailbox's owner"
                                         : "closes a mailbox");
    }
    return;
  }
  if (name == "monitor") {
    merk::CallOp call{JustCalled("process.monitor_6_0", 5)};
    if (!call || call->getNumResults() != 1 ||
        !mlir::isa<merk::ClosureType>(call.getInputs()[4].getType())) {
      return;
    }
    builder_.setInsertionPoint(call);
    auto op{merk::MonitorOp::create(builder_, call.getLoc(),
        call.getResult(0).getType(), MailboxTerm(call.getInputs()[2]),
        MailboxTerm(call.getInputs()[3]), call.getInputs()[4],
        call.getInputs()[0], call.getInputs()[1])};
    ReplaceResult(call.getResult(0), op.getRef(), values_);
    call->erase();
    if (target_.options().IsDevice()) {
      NotOnDevice(op, "monitors a mailbox");
    }
    return;
  }
  if (name == "demonitor") {
    merk::CallOp call{JustCalled("process.demonitor_3_0", 1)};
    if (!call || call->getNumResults() != 0) {
      return;
    }
    builder_.setInsertionPoint(call);
    auto op{merk::DemonitorOp::create(
        builder_, call.getLoc(), call.getInputs()[0])};
    call->erase();
    if (target_.options().IsDevice()) {
      NotOnDevice(op, "removes a monitor");
    }
  }
}

// A call of process.receive/5, just generated, that ReceiveInTarget did
// not make the device's (docs/INPROGRESS.md, Concurrency in merk, step 1
// (d)): a plain merk.receive of one case. A selector that is a closure of
// a known procedure, a named predicate or a lambda expression, as `any`
// is, makes that procedure the matcher, with the closure's captured
// values; any other selector is a value, and the case calls the closure.
// On a device, only receive/5 with `any` is received so far, which
// ReceiveInTarget makes; any other says why, in merk.not_on_device.
void MerkGenerator::FormPlainReceive(
    llvm::ArrayRef<BodyVar> arguments, const ProcedureBody &body) {
  mlir::Value result{values_.lookup(arguments[3])};
  if (!result) {
    return;
  }
  auto call{Uncoerced(result).getDefiningOp<merk::CallOp>()};
  if (!call || !call.getCallee().starts_with("process.receive_6_0") ||
      call.getInputs().size() != 5 || call->getNumResults() != 1) {
    return;
  }
  mlir::Location loc{call.getLoc()};
  mlir::OpBuilder::InsertionGuard guard{builder_};
  builder_.setInsertionPoint(call);
  mlir::Value selector{call.getInputs()[3]};
  llvm::SmallVector<mlir::Value> captured;
  mlir::Attribute matcher{mlir::UnitAttr::get(&context_)};
  auto closure{Uncoerced(selector).getDefiningOp<merk::ClosureOp>()};
  merk::ProcOp known;
  if (closure && closure->getBlock() == call->getBlock()) {
    auto lambda{entryLambdas_.find(closure.getEntry())};
    if (lambda != entryLambdas_.end()) {
      known = module_.lookupSymbol<merk::ProcOp>(lambda->second);
    }
    if (known &&
        (known.getDeterminism() != merk::Determinism::Semidet ||
            known.getFunctionType().getNumInputs() !=
                closure.getCaptured().size() + 1 ||
            known.getFunctionType().getNumResults() != 1)) {
      known = nullptr;
    }
  }
  if (known) {
    mlir::FunctionType type{known.getFunctionType()};
    for (auto [i, word] : llvm::enumerate(closure.getCaptured())) {
      captured.push_back(Coerce(word, type.getInput(i), loc));
    }
    matcher = mlir::FlatSymbolRefAttr::get(&context_, known.getSymName());
  } else {
    captured.push_back(selector);
  }
  mlir::Value mailbox{Uncoerced(call.getInputs()[2])};
  if (!mlir::isa<merk::TermType>(mailbox.getType())) {
    mailbox = call.getInputs()[2];
  }
  mlir::Type resultType{call.getResult(0).getType()};
  const semantics::Type *received{body.vars[arguments[3]].type};
  if (std::optional<Represented> represented{Represent(received)};
      represented && mlir::isa<merk::TermType>(represented->type)) {
    resultType = represented->type;
  }
  auto receive{merk::ReceiveOp::create(builder_, loc, resultType,
      call.getInputs()[4], call.getInputs()[1], mlir::ValueRange{mailbox},
      mlir::ValueRange{call.getInputs()[0]}, captured,
      builder_.getArrayAttr({matcher}),
      builder_.getDenseI64ArrayAttr(
          {static_cast<std::int64_t>(captured.size())}),
      known && closure.getCaptures()
          ? builder_.getArrayAttr({*closure.getCaptures()})
          : mlir::ArrayAttr{},
      /*plain=*/true)};
  if (received->kind == semantics::TypeKind::Named &&
      received->arguments.size() == 1) {
    if (std::optional<Represented> message{
            Represent(received->arguments.front())}) {
      receive->setAttr("merk.arms_type", mlir::TypeAttr::get(message->type));
    }
  }
  if (target_.options().IsDevice()) {
    bool any{known && known.getSymName() == kAnyMatcher};
    receive->setAttr("merk.not_on_device",
        builder_.getStringAttr(any ? "receives messages of a type that a "
                                     "device does not receive"
                                   : "receives with a selector other than "
                                     "any"));
  }
  ReplaceReceiveCall(call, receive.getResult(), loc);
}

void MerkGenerator::PassTypeInfo(const semantics::TypeInfo &info,
    llvm::SmallVectorImpl<mlir::Value> &inputs, mlir::Location loc,
    parser::CharBlock at) {
  mlir::Value value{TypeInfoValue(info, loc, at)};
  inputs.push_back(value);
  failed_ = failed_ || !value;
}

void MerkGenerator::PassDictionary(const semantics::Dictionary &dictionary,
    llvm::SmallVectorImpl<mlir::Value> &inputs, mlir::Location loc,
    parser::CharBlock at) {
  mlir::Value value{DictionaryValue(dictionary, loc, at)};
  inputs.push_back(value);
  failed_ = failed_ || !value;
}

void MerkGenerator::Called(const Body &goal, const Procedure &callee,
    llvm::ArrayRef<BodyVar> arguments, const ProcedureBody &body) {
  if (ReportOpenMessages(callee, goal, arguments, body)) {
    return;
  }
  // open's predicates call the runtime's walkers; unify_oc unifies, and so
  // walks closed(U) too.
  if (callee.symbol->module == "open" &&
      ReportOpenShapes(
          goal, arguments, body, callee.symbol->name == "unify_oc")) {
    return;
  }
  if (Builtin(goal, callee, arguments, body)) {
    return;
  }
  parser::CharBlock at{Where(goal.tokens)};
  mlir::Location loc{Location(goal.tokens)};
  std::optional<llvm::SmallVector<mlir::Type>> hidden{HiddenInputs(callee, at)};
  if (!hidden) {
    failed_ = true;
    return;
  }
  bool isMethod{callee.symbol->owner &&
      callee.symbol->owner->kind == semantics::SymbolKind::Typeclass};
  llvm::SmallVector<mlir::Type> hiddenOutputs{HiddenOutputs(callee)};
  if (isMethod && !hiddenOutputs.empty()) {
    NotYet(at, "methods with existential types");
    return;
  }
  bool polymorphic{!hidden->empty() || !hiddenOutputs.empty()};
  const semantics::PassedInfos *passed{nullptr};
  if (polymorphic) {
    const semantics::ProcedurePolymorphism *polymorphism{
        polymorphism_.lookup(&body)};
    if (polymorphism) {
      if (auto it{polymorphism->goals.find(&goal)};
          it != polymorphism->goals.end()) {
        passed = &it->second;
      }
    }
    if (!passed ||
        passed->typeInfos.size() + passed->dictionaries.size() !=
            hidden->size() + hiddenOutputs.size()) {
      NotYet(at, "this call's type_infos");
      return;
    }
  }
  // A method of an instance known here, whose body this module has, is called
  // directly: no typeclass_info is made, and the values keep their
  // representations, as a device needs (design document, "What polymorphism
  // costs").
  const semantics::Dictionary *known{nullptr};
  const ProcedureBody *direct{nullptr};
  if (isMethod) {
    const semantics::Dictionary *own{passed->dictionaries.front()};
    if ((own->kind == semantics::DictionaryKind::Constant ||
            own->kind == semantics::DictionaryKind::Instance) &&
        own->instance) {
      known = own;
      direct = MethodBody(*own->instance, callee);
    }
  }
  // A polymorphic callee takes its declared types, whatever the caller's are.
  llvm::SmallVector<const semantics::Type *> types;
  if (direct) {
    llvm::ArrayRef<const semantics::Type *> declared{
        direct->method->method->signature.arguments};
    types.append(declared.begin(), declared.end());
  } else if (polymorphic) {
    llvm::ArrayRef<const semantics::Type *> declared{
        types_.SignatureOf(*callee.symbol)->arguments};
    types.append(declared.begin(), declared.end());
  } else {
    for (BodyVar argument : arguments) {
      types.push_back(body.vars[argument].type);
    }
  }
  // In a body that gives words, a call whose outputs are the body's calls the
  // callee's word copy, which gives them as words too, so that it is a tail
  // call when it comes last.
  llvm::SmallVector<BodyVar> outputVars;
  for (auto [argument, mode] : llvm::zip(arguments, callee.modes)) {
    if (IsOutput(mode)) {
      outputVars.push_back(argument);
    }
  }
  // Arguments passed as closed(T)'s open words (OpenParameter).
  bool openClosed{false};
  for (auto [mode, type] : llvm::zip(callee.modes, types)) {
    openClosed |= OpenParameter(mode, type);
  }
  for (auto [argument, mode, type] :
      llvm::zip(arguments, callee.modes, types)) {
    if (!IsClosedType(type) && ClosedOpen(body, argument) &&
        OpenParameter(mode, body.vars[argument].type)) {
      // Temporary (docs/INPROGRESS.md, Open types: what is left): code at a
      // type variable reads closed(U) through its type_info, as U's word, which
      // an unbound one has not.
      NotYet(at,
          "passing a closed(T) value that may be unbound where the "
          "callee has a type variable");
      return;
    }
  }
  if (openClosed && isMethod) {
    // Temporary (docs/INPROGRESS.md, Open types: what is left).
    NotYet(at, "methods with closed(T) arguments that may be unbound");
    return;
  }
  const ProcedureBody *copyable{nullptr};
  if (wordOutputs_ && !isMethod && hiddenOutputs.empty() && !openClosed &&
      !(goal.modes && goal.modes->implied) &&
      llvm::equal(outputVars, *wordOutputs_)) {
    copyable = CopyableBody(callee);
  }
  merk::ProcOp proc;
  if (direct) {
    proc = MethodProc(*direct);
    if (!proc) {
      failed_ = true;
      return;
    }
  } else if (copyable) {
    proc = WordProcFor(*copyable);
    if (!proc) {
      return;
    }
  } else if (!isMethod) {
    proc = ProcFor(callee, types, at);
    if (!proc) {
      return;
    }
  }
  llvm::SmallVector<mlir::Value> inputs;
  mlir::Value dictionary;
  if (direct) {
    // The instance's hidden arguments come first.
    for (const semantics::TypeInfo *info : known->typeInfos) {
      PassTypeInfo(*info, inputs, loc, at);
    }
    for (const semantics::Dictionary *premise : known->premises) {
      PassDictionary(*premise, inputs, loc, at);
    }
  }
  // What this goal gives, in the order of the callee's first results.
  llvm::SmallVector<const void *> gives;
  std::size_t givenTypeInfos{0}; // of which the first are type_infos
  if (passed) {
    for (const semantics::TypeInfo *info : passed->typeInfos) {
      if (info->kind == semantics::TypeInfoKind::Output && info->goal == &goal) {
        NoteGivenType(*info);
        gives.push_back(info);
        ++givenTypeInfos;
      } else {
        PassTypeInfo(*info, inputs, loc, at);
      }
    }
    for (auto [i, given] : llvm::enumerate(passed->dictionaries)) {
      if (given->kind == semantics::DictionaryKind::Output &&
          given->goal == &goal) {
        NoteGivenTypes(*given);
        gives.push_back(given);
      } else if (!isMethod || i != 0) {
        PassDictionary(*given, inputs, loc, at);
      } else if (!direct) {
        // A method's type class's comes first, and chooses the procedure.
        dictionary = DictionaryValue(*given, loc, at);
        failed_ = failed_ || !dictionary;
      }
    }
  }
  if (failed_) {
    return;
  }
  // None for a value with no bits at a type variable's place, whose word
  // comes back unused.
  llvm::SmallVector<std::optional<BodyVar>> outputs;
  llvm::DenseSet<BodyVar> openOutputs;
  for (auto [argument, mode, type] : llvm::zip(arguments, callee.modes, types)) {
    // By the callee's type: at a type variable, a closed(U) is U's word, as
    // the code there reads it through closed(U)'s type_info.
    bool open{OpenParameter(mode, type)};
    std::optional<Represented> actual{Represent(body.vars[argument].type)};
    if (!actual) {
      NotYet(at, "values of type " +
              semantics::TypeToString(*body.vars[argument].type));
      return;
    }
    bool dummy{actual->dummy};
    if (Represent(type)->dummy || IsUnused(mode)) {
      continue;
    }
    if (IsOutput(mode)) {
      outputs.push_back(dummy ? std::nullopt : std::make_optional(argument));
      if (open) {
        openOutputs.insert(argument);
      }
    } else if (dummy) {
      inputs.push_back(merk::ToWordOp::create(builder_, loc,
          merk::WordType::get(&context_),
          IntegerConstant(builder_.getI64Type(), 0, loc)));
    } else if (IsClosedType(body.vars[argument].type)) {
      mlir::Value value{
          ClosedAs(ValueOf(argument), open, body.vars[argument].type, loc, at)};
      if (!value) {
        failed_ = true;
        return;
      }
      inputs.push_back(Coerce(value,
          open ? mlir::Type{merk::WordType::get(&context_)}
               : Represent(type)->type,
          loc));
    } else {
      inputs.push_back(Coerce(ValueOf(argument), Represent(type)->type, loc));
    }
  }
  if (copyable) {
    llvm::ArrayRef<mlir::Type> types{proc.getFunctionType().getInputs()};
    if (types.size() != inputs.size()) {
      NotYet(at, "this call of a word copy");
      return;
    }
    for (auto [input, type] : llvm::zip(inputs, types)) {
      input = Coerce(input, type, loc);
    }
  }
  mlir::Operation *call;
  if (isMethod && !direct) {
    std::optional<parser::Determinism> determinism{
        determinism_.DeterminismOf(callee)};
    std::optional<merk::Determinism> merkDeterminism;
    if (determinism) {
      merkDeterminism = ClosureDeterminism(*determinism);
    }
    if (!merkDeterminism) {
      NotYet(at, "methods of this determinism");
      return;
    }
    llvm::SmallVector<const Procedure *> slots{
        MethodSlots(*callee.symbol->owner)};
    llvm::SmallVector<mlir::Type> resultTypes;
    for (auto [mode, type] : llvm::zip(callee.modes, types)) {
      std::optional<Represented> represented{Represent(type)};
      if (represented && !represented->dummy && IsOutput(mode)) {
        resultTypes.push_back(represented->type);
      }
    }
    std::int64_t slot{llvm::find(slots, &callee) - slots.begin()};
    if (*merkDeterminism == merk::Determinism::Nondet ||
        *merkDeterminism == merk::Determinism::Multi) {
      // The rest of the code is the call's continuation, as for a nondet
      // procedure.
      auto nondet{merk::NondetCallMethodOp::create(
          builder_, loc, dictionary, slot, inputs)};
      mlir::Block &continuation{nondet.getContinuation().emplaceBlock()};
      for (mlir::Type type : resultTypes) {
        continuation.addArgument(type, loc);
      }
      builder_.setInsertionPointToStart(&continuation);
      call = nondet;
    } else {
      call = merk::CallMethodOp::create(builder_, loc, resultTypes, dictionary,
          slot, inputs, *merkDeterminism);
    }
  } else if (Nondeterministic(callee)) {
    // The rest of the code is the call's continuation, which takes the
    // callee's outputs.
    auto nondet{merk::NondetCallOp::create(
        builder_, loc, proc.getSymName(), inputs)};
    mlir::Block &continuation{nondet.getContinuation().emplaceBlock()};
    for (mlir::Type type : proc.getFunctionType().getResults()) {
      continuation.addArgument(type, loc);
    }
    builder_.setInsertionPointToStart(&continuation);
    call = nondet;
  } else {
    call = merk::CallOp::create(builder_, loc,
        proc.getFunctionType().getResults(), proc.getSymName(), inputs);
  }
  mlir::ValueRange results{call->getResults()};
  if (auto nondet{mlir::dyn_cast<merk::NondetCallOp>(call)}) {
    results = nondet.getContinuation().front().getArguments();
  } else if (auto nondet{mlir::dyn_cast<merk::NondetCallMethodOp>(call)}) {
    results = nondet.getContinuation().front().getArguments();
  }
  // A call that never succeeds ends the branch it is in, which then need not
  // bind what the other branches bind.
  if (std::optional<parser::Determinism> determinism{
          determinism_.DeterminismOf(callee)};
      determinism && (*determinism == parser::Determinism::Erroneous ||
                         *determinism == parser::Determinism::Failure)) {
    merk::FailOp::create(builder_, loc);
    return;
  }
  if (copyable) {
    for (auto [var, word] : llvm::zip_equal(outputVars, results)) {
      SetFromWord(var, word, body, loc);
    }
    return;
  }
  // The results go on with the outputs.
  std::size_t given{0};
  for (auto [node, result] : llvm::zip(gives, results)) {
    if (given++ < givenTypeInfos) {
      GiveTypeInfo(*static_cast<const semantics::TypeInfo *>(node), result, loc);
    } else {
      GiveDictionary(
          *static_cast<const semantics::Dictionary *>(node), result, loc);
    }
    Give(node, result);
  }
  for (auto [var, result] :
      llvm::zip(outputs, results.drop_front(gives.size()))) {
    if (!var) {
      continue;
    }
    values_[*var] = openOutputs.contains(*var)
        ? result
        : Coerce(result, Represent(body.vars[*var].type)->type, loc);
  }
}

const ProcedureBody *MerkGenerator::CopyableBody(const Procedure &procedure) {
  auto it{bodies_.find(&procedure)};
  if (it == bodies_.end() || !HiddenOutputs(procedure).empty()) {
    return nullptr;
  }
  return it->second;
}

merk::ProcOp MerkGenerator::WordProcFor(const ProcedureBody &body) {
  const Procedure &procedure{*body.procedure};
  if (merk::ProcOp proc{wordProcs_.lookup(&procedure)}) {
    return proc;
  }
  parser::CharBlock at{procedure.symbol->source};
  std::optional<parser::Determinism> determinism{
      determinism_.DeterminismOf(procedure)};
  std::optional<merk::Determinism> merkDeterminism;
  if (determinism) {
    merkDeterminism = ClosureDeterminism(*determinism);
  }
  if (!merkDeterminism) {
    NotYet(at, "word copies of procedures of this determinism");
    return nullptr;
  }
  std::optional<llvm::SmallVector<mlir::Type>> inputs{
      HiddenInputs(procedure, at)};
  if (!inputs) {
    return nullptr;
  }
  unsigned outputs{0};
  for (auto [var, mode] : llvm::zip(body.head, procedure.modes)) {
    const semantics::Type *type{body.vars[var].type};
    std::optional<Represented> represented{Represent(type)};
    if (!represented) {
      NotYet(at, "values of type " + semantics::TypeToString(*type));
      return nullptr;
    }
    if (IsOutput(mode)) {
      ++outputs;
    } else if (!represented->dummy && !IsUnused(mode)) {
      inputs->push_back(represented->type);
    }
  }
  mlir::OpBuilder::InsertionGuard guard{builder_};
  builder_.setInsertionPointToEnd(module_.getBody());
  auto proc{merk::ProcOp::create(builder_, Location(at),
      MangledName(procedure) + "$words",
      builder_.getFunctionType(*inputs,
          llvm::SmallVector<mlir::Type>(outputs, merk::WordType::get(&context_))),
      *merkDeterminism)};
  proc.setPrivate();
  MarkInlining(proc, procedure);
  wordProcs_[&procedure] = proc;
  pendingCopies_.push_back(&body);
  return proc;
}

void MerkGenerator::EraseUnwordedOutputs(mlir::Region &region) {
  llvm::SmallVector<merk::FromWordOp> unused;
  region.walk([&](merk::FromWordOp op) {
    auto argument{mlir::dyn_cast<mlir::BlockArgument>(op.getWord())};
    if (op->use_empty() && argument &&
        mlir::isa<merk::EachOp, merk::NondetCallOp, merk::NondetCallClosureOp,
            merk::NondetCallMethodOp>(argument.getOwner()->getParentOp())) {
      unused.push_back(op);
    }
  });
  for (merk::FromWordOp op : unused) {
    op->erase();
  }
}

llvm::SmallVector<mlir::Value> MerkGenerator::WordResults(
    llvm::ArrayRef<BodyVar> outputs, const ProcedureBody &body,
    mlir::Location loc) {
  auto word{merk::WordType::get(&context_)};
  llvm::SmallVector<mlir::Value> results;
  for (BodyVar var : outputs) {
    if (Represent(body.vars[var].type)->dummy) {
      mlir::Value given{values_.lookup(kDummyWordVar + var)};
      results.push_back(given ? given : ZeroWord(loc));
    } else if (mlir::Value value{ValueOf(var)}) {
      results.push_back(Coerce(value, word, loc));
    } else {
      results.push_back(nullptr);
    }
  }
  return results;
}

void MerkGenerator::SetFromWord(BodyVar var, mlir::Value word,
    const ProcedureBody &body, mlir::Location loc) {
  Represented represented{*Represent(body.vars[var].type)};
  if (!represented.dummy) {
    values_[var] = Coerce(word, represented.type, loc);
  } else if (wordOutputs_) {
    values_[kDummyWordVar + var] = word;
  }
}

merk::ProcOp MerkGenerator::ClosureEntry(merk::ProcOp proc, unsigned captured,
    llvm::ArrayRef<ClosureArgument> arguments, parser::CharBlock at) {
  std::string name{proc.getSymName().str() + "$closure" + std::to_string(captured)};
  if (merk::ProcOp entry{entries_.lookup(name)}) {
    return entry;
  }
  mlir::Location loc{Location(at)};
  auto word{merk::WordType::get(&context_)};
  llvm::SmallVector<mlir::Type> inputs{merk::ClosureType::get(&context_)};
  unsigned outputs{0};
  for (const ClosureArgument &argument : arguments) {
    if (argument.output) {
      ++outputs;
    } else {
      inputs.push_back(word);
    }
  }
  llvm::ArrayRef<mlir::Type> types{proc.getFunctionType().getInputs()};
  unsigned passed{captured};
  for (const ClosureArgument &argument : arguments) {
    passed += !argument.output && !argument.dummy;
  }
  if (passed != types.size()) {
    NotYet(at, "this closure's arguments");
    return nullptr;
  }
  mlir::OpBuilder::InsertionGuard guard{builder_};
  builder_.setInsertionPointToEnd(module_.getBody());
  auto entry{merk::ProcOp::create(builder_, loc, name,
      builder_.getFunctionType(inputs, llvm::SmallVector<mlir::Type>(outputs, word)),
      proc.getDeterminism())};
  entry.setPrivate();
  entries_[name] = entry;
  entryLambdas_[name] = proc.getSymName().str();
  mlir::Block *block{entry.addEntryBlock()};
  builder_.setInsertionPointToStart(block);
  mlir::Value closure{block->getArgument(0)};
  llvm::SmallVector<mlir::Value> values;
  for (unsigned i{0}; i < captured; ++i) {
    values.push_back(Coerce(
        merk::ClosureWordOp::create(builder_, loc, word, closure, i), types[i], loc));
  }
  unsigned input{1};
  for (const ClosureArgument &argument : arguments) {
    if (argument.output) {
      continue;
    }
    mlir::Value value{block->getArgument(input++)};
    if (!argument.dummy) {
      values.push_back(Coerce(value, types[values.size()], loc));
    }
  }
  // A nondet or multi procedure's solutions are the entry's, in words.
  mlir::ValueRange given;
  if (proc.getDeterminism() == merk::Determinism::Nondet ||
      proc.getDeterminism() == merk::Determinism::Multi) {
    auto call{merk::NondetCallOp::create(builder_, loc, proc.getSymName(), values)};
    mlir::Block &continuation{call.getContinuation().emplaceBlock()};
    for (mlir::Type type : proc.getFunctionType().getResults()) {
      continuation.addArgument(type, loc);
    }
    builder_.setInsertionPointToStart(&continuation);
    given = continuation.getArguments();
  } else {
    given = merk::CallOp::create(builder_, loc,
        proc.getFunctionType().getResults(), proc.getSymName(), values)
                  .getResults();
  }
  llvm::SmallVector<mlir::Value> results;
  unsigned result{0};
  for (const ClosureArgument &argument : arguments) {
    if (argument.output) {
      results.push_back(argument.dummy
              ? ZeroWord(loc)
              : Coerce(given[result++], word, loc));
    }
  }
  merk::SucceedOp::create(builder_, loc, results);
  return entry;
}

merk::ProcOp MerkGenerator::BuiltinEntry(const Body &goal,
    const Procedure &callee, llvm::ArrayRef<const semantics::Type *> types,
    unsigned curried, bool &failed) {
  // Code generation builds in only the library's procedures.
  if (callee.symbol->module == moduleName_) {
    return nullptr;
  }
  std::string name{"merk.builtin." + MangledName(callee) + "$closure" +
      std::to_string(curried)};
  if (merk::ProcOp entry{entries_.lookup(name)}) {
    return entry;
  }
  std::optional<parser::Determinism> determinism{
      determinism_.DeterminismOf(callee)};
  std::optional<merk::Determinism> merkDeterminism;
  if (determinism) {
    merkDeterminism = MerkDeterminism(*determinism);
  }
  if (!merkDeterminism) {
    return nullptr;
  }
  parser::CharBlock at{Where(goal.tokens)};
  mlir::Location loc{Location(goal.tokens)};
  auto word{merk::WordType::get(&context_)};
  llvm::SmallVector<mlir::Type> inputs{merk::ClosureType::get(&context_)};
  unsigned outputs{0};
  for (auto [i, mode] : llvm::enumerate(callee.modes)) {
    if (IsOutput(mode)) {
      ++outputs;
    } else if (i >= curried) {
      inputs.push_back(word);
    }
  }
  // A call of it in a body of its own, whose variables are its arguments.
  ProcedureBody own;
  own.procedure = &callee;
  llvm::SmallVector<BodyVar> vars;
  for (const semantics::Type *type : types) {
    vars.push_back(static_cast<BodyVar>(own.vars.size()));
    own.vars.push_back({"", type});
  }
  Body call{BodyKind::Call, goal.tokens, nullptr};
  mlir::OpBuilder::InsertionGuard guard{builder_};
  builder_.setInsertionPointToEnd(module_.getBody());
  auto entry{merk::ProcOp::create(builder_, loc, name,
      builder_.getFunctionType(inputs, llvm::SmallVector<mlir::Type>(outputs, word)),
      *merkDeterminism)};
  entry.setPrivate();
  mlir::Block *block{entry.addEntryBlock()};
  builder_.setInsertionPointToStart(block);
  Values saved;
  saved.swap(values_);
  llvm::DenseMap<unsigned, mlir::Value> savedInfos;
  savedInfos.swap(typeInfos_);
  std::optional<llvm::SmallVector<BodyVar>> savedWords{std::move(wordOutputs_)};
  wordOutputs_.reset();
  mlir::Value closure{block->getArgument(0)};
  // A polymorphic procedure's closure captures a type_info for each of its
  // type variables first, as its calls pass them, then the curried arguments.
  unsigned captured{0};
  bool constrained{false};
  if (const semantics::Signature *signature{
          types_.SignatureOf(*callee.symbol)}) {
    constrained = !signature->constraints.empty();
    for (unsigned var{0}; var < signature->typeVars - signature->existentials;
        ++var) {
      typeInfos_[var] =
          Coerce(merk::ClosureWordOp::create(builder_, loc,
                     merk::WordType::get(&context_), closure, captured++),
              merk::TypeInfoType::get(&context_), loc);
    }
  }
  if (constrained) {
    NotYet(at, "closures of built-in procedures with class constraints");
  }
  unsigned input{1};
  bool represented{true};
  for (auto [i, mode, type] : llvm::enumerate(callee.modes, types)) {
    std::optional<Represented> value{Represent(type)};
    represented = represented && value;
    if (!represented || IsOutput(mode)) {
      continue;
    }
    mlir::Value word{i < curried
            ? (value->dummy ? mlir::Value{}
                            : mlir::Value{merk::ClosureWordOp::create(builder_,
                                  loc, merk::WordType::get(&context_), closure,
                                  captured++)})
            : block->getArgument(input++)};
    if (!value->dummy) {
      values_[vars[i]] = Coerce(word, value->type, loc);
    }
  }
  bool handled{!constrained && represented && Builtin(call, callee, vars, own)};
  if (handled && !failed_ && !Terminated()) {
    llvm::SmallVector<mlir::Value> results;
    for (auto [i, mode, type] : llvm::enumerate(callee.modes, types)) {
      if (!IsOutput(mode)) {
        continue;
      }
      if (Represent(type)->dummy) {
        results.push_back(ZeroWord(loc));
      } else if (mlir::Value value{ValueOf(vars[i])}) {
        results.push_back(Coerce(value, word, loc));
      } else {
        NotYet(at, "this closure's outputs");
        break;
      }
    }
    if (!failed_) {
      merk::SucceedOp::create(builder_, loc, results);
    }
  }
  values_.swap(saved);
  typeInfos_.swap(savedInfos);
  wordOutputs_ = std::move(savedWords);
  if (!handled || failed_) {
    failed = failed_;
    entry.erase();
    return nullptr;
  }
  entries_[name] = entry;
  return entry;
}

// The variables a goal names, at any depth: the fields its kind has, since
// the others are 0, the first variable's number.
static void CollectVars(const Body &goal, llvm::DenseSet<BodyVar> &vars) {
  switch (goal.kind) {
  case BodyKind::Unify:
    vars.insert(goal.left);
    if (goal.isVar) {
      vars.insert(goal.var);
    }
    break;
  case BodyKind::HOCall:
    vars.insert(goal.closure);
    break;
  case BodyKind::Switch:
    vars.insert(goal.switchVar);
    break;
  default:
    break;
  }
  vars.insert(goal.arguments.begin(), goal.arguments.end());
  vars.insert(goal.lambdaHead.begin(), goal.lambdaHead.end());
  vars.insert(goal.nonlocals.begin(), goal.nonlocals.end());
  vars.insert(goal.promised.begin(), goal.promised.end());
  if (goal.switchOn) {
    vars.insert(*goal.switchOn);
  }
  for (const auto &[var, inst] : goal.entry) {
    vars.insert(var);
  }
  if (goal.lambda) {
    CollectVars(*goal.lambda, vars);
  }
  for (const Body *inner : goal.goals) {
    CollectVars(*inner, vars);
  }
  for (const semantics::SwitchCase &each : goal.cases) {
    CollectVars(*each.goal, vars);
  }
}

void MerkGenerator::LambdaClosure(const Body &goal, const ProcedureBody &body) {
  parser::CharBlock at{Where(goal.tokens)};
  llvm::ArrayRef<semantics::ArgMode> modes{goal.modes->lambda};
  std::optional<merk::Determinism> determinism{
      ClosureDeterminism(goal.modes->determinism)};
  if (!determinism) {
    NotYet(at,
        std::string{semantics::DeterminismName(goal.modes->determinism)} +
            " lambda expressions");
    return;
  }
  if (modes.size() != goal.lambdaHead.size()) {
    NotYet(at, "this lambda expression's modes");
    return;
  }
  llvm::SmallVector<BodyVar> outputs, inputs;
  llvm::SmallVector<ClosureArgument> arguments;
  llvm::SmallVector<mlir::Type> inputTypes;
  for (auto [var, mode] : llvm::zip(goal.lambdaHead, modes)) {
    const semantics::Type *type{body.vars[var].type};
    std::optional<Represented> represented{Represent(type)};
    if (!represented) {
      NotYet(at, "values of type " + semantics::TypeToString(*type));
      return;
    }
    if (IsOutput(mode)) {
      outputs.push_back(var);
      arguments.push_back({true, false});
    } else {
      bool unpassed{represented->dummy || IsUnused(mode)};
      arguments.push_back({false, unpassed});
      if (!unpassed) {
        inputs.push_back(var);
        inputTypes.push_back(represented->type);
      }
    }
  }
  bool committed{goal.modes->determinism == parser::Determinism::CcMulti ||
      goal.modes->determinism == parser::Determinism::CcNondet};
  mlir::Value closure{OutlineClosure(goal, *goal.lambda, goal.lambdaHead,
      arguments, inputs, inputTypes, static_cast<unsigned>(outputs.size()),
      *determinism, committed, outputs, body,
      [&](mlir::Location loc) { return WordResults(outputs, body, loc); })};
  if (closure) {
    values_[goal.left] = closure;
  }
}

mlir::Value MerkGenerator::OutlineClosure(const Body &key, const Body &inner,
    llvm::ArrayRef<BodyVar> head, llvm::ArrayRef<ClosureArgument> arguments,
    llvm::ArrayRef<BodyVar> inputVars, llvm::ArrayRef<mlir::Type> inputTypes,
    unsigned outputs, merk::Determinism determinism, bool committed,
    std::optional<llvm::SmallVector<BodyVar>> wordOutputs,
    const ProcedureBody &body,
    llvm::function_ref<llvm::SmallVector<mlir::Value>(mlir::Location)>
        results) {
  parser::CharBlock at{Where(key.tokens)};
  mlir::Location loc{Location(key.tokens)};
  // What it captures, each in order: the body's type_infos, its
  // typeclass_infos, what goals gave, and the variables bound before it
  // that it uses.
  llvm::DenseSet<BodyVar> used;
  CollectVars(inner, used);
  llvm::SmallVector<unsigned> infoKeys;
  for (const auto &[var, value] : typeInfos_) {
    if (value) {
      infoKeys.push_back(var);
    }
  }
  llvm::sort(infoKeys);
  llvm::SmallVector<unsigned> dictionaryKeys;
  for (const auto &[constraint, value] : dictionaries_) {
    if (value) {
      dictionaryKeys.push_back(constraint);
    }
  }
  llvm::sort(dictionaryKeys);
  llvm::SmallVector<BodyVar> captured;
  for (const auto &[var, value] : values_) {
    if (value && !IsDummyWord(var) &&
        (var >= kFirstGivenVar ||
            (used.contains(var) && !llvm::is_contained(head, var)))) {
      captured.push_back(var);
    }
  }
  llvm::sort(captured);
  llvm::SmallVector<mlir::Value> capturedValues;
  for (unsigned var : infoKeys) {
    capturedValues.push_back(typeInfos_.lookup(var));
  }
  for (unsigned constraint : dictionaryKeys) {
    capturedValues.push_back(dictionaries_.lookup(constraint));
  }
  for (BodyVar var : captured) {
    capturedValues.push_back(values_.lookup(var));
  }
  llvm::SmallVector<mlir::Type> inputs{
      llvm::map_range(capturedValues, [](mlir::Value v) { return v.getType(); })};
  inputs.append(inputTypes.begin(), inputTypes.end());
  auto word{merk::WordType::get(&context_)};
  // A word copy's body has the same lambda expressions as the procedure's,
  // which capture values of the same types: one procedure serves both.
  merk::ProcOp proc{lambdaProcs_.lookup(&key)};
  if (!proc ||
      proc.getFunctionType().getInputs() != llvm::ArrayRef<mlir::Type>{inputs}) {
    {
      mlir::OpBuilder::InsertionGuard guard{builder_};
      builder_.setInsertionPointToEnd(module_.getBody());
      proc = merk::ProcOp::create(builder_, loc,
          procName_ + "$lambda" + std::to_string(lambdas_++),
          builder_.getFunctionType(
              inputs, llvm::SmallVector<mlir::Type>(outputs, word)),
          determinism);
      proc.setPrivate();
    }
    // Its body, with the captured values and its inputs as the procedure's.
    Values savedValues;
    savedValues.swap(values_);
    llvm::DenseMap<unsigned, mlir::Value> savedInfos;
    savedInfos.swap(typeInfos_);
    llvm::DenseMap<unsigned, mlir::Value> savedGiven;
    savedGiven.swap(givenInfos_);
    llvm::DenseMap<unsigned, mlir::Value> savedDictionaries;
    savedDictionaries.swap(dictionaries_);
    std::optional<llvm::SmallVector<BodyVar>> savedWords{std::move(wordOutputs_)};
    std::string savedName{procName_};
    {
      mlir::OpBuilder::InsertionGuard guard{builder_};
      mlir::Block *entry{proc.addEntryBlock()};
      builder_.setInsertionPointToStart(entry);
      unsigned input{0};
      for (unsigned var : infoKeys) {
        typeInfos_[var] = entry->getArgument(input++);
      }
      for (unsigned constraint : dictionaryKeys) {
        dictionaries_[constraint] = entry->getArgument(input++);
      }
      for (BodyVar var : captured) {
        values_[var] = entry->getArgument(input++);
      }
      // The type_infos in what goals gave that it captures.
      for (const auto &[variable, place] : givenTypes_) {
        auto given{givenVars_.find(place.first)};
        mlir::Value value{given == givenVars_.end()
                ? mlir::Value{}
                : values_.lookup(given->second)};
        if (value) {
          givenInfos_[variable] = place.second == 0
              ? value
              : merk::TypeclassInfoWordOp::create(builder_, loc,
                    merk::TypeInfoType::get(&context_), value, place.second);
        }
      }
      for (BodyVar var : inputVars) {
        values_[var] = entry->getArgument(input++);
      }
      wordOutputs_ = std::move(wordOutputs);
      procName_ = proc.getSymName().str();
      const Body *outerLambda{committedLambda_};
      committedLambda_ = committed ? &inner : nullptr;
      Goal(inner, body);
      committedLambda_ = outerLambda;
      if (!failed_ && !Terminated()) {
        llvm::SmallVector<mlir::Value> words{results(loc)};
        if (llvm::is_contained(words, mlir::Value{})) {
          NotYet(at, "this lambda expression's outputs");
        } else {
          merk::SucceedOp::create(builder_, loc, words);
          EraseUnwordedOutputs(proc.getBody());
        }
      }
    }
    values_.swap(savedValues);
    typeInfos_.swap(savedInfos);
    givenInfos_ = std::move(savedGiven);
    dictionaries_.swap(savedDictionaries);
    wordOutputs_ = std::move(savedWords);
    procName_ = savedName;
    if (failed_) {
      proc.getBody().dropAllReferences();
      proc.getBody().getBlocks().clear();
      return {};
    }
    lambdaProcs_[&key] = proc;
  }
  merk::ProcOp entry{ClosureEntry(
      proc, static_cast<unsigned>(capturedValues.size()), arguments, at)};
  if (!entry) {
    return {};
  }
  CapturePlaces places;
  llvm::SmallVector<mlir::Attribute> captures;
  for (auto [i, var] : llvm::enumerate(infoKeys)) {
    places[var] = {static_cast<unsigned>(i), 0};
    captures.push_back(builder_.getStringAttr("type_info"));
  }
  captures.append(
      dictionaryKeys.size(), builder_.getStringAttr("typeclass_info"));
  // A type variable a goal gave has its type_info in what the goal gave.
  for (const auto &[variable, place] : givenTypes_) {
    auto given{givenVars_.find(place.first)};
    if (given == givenVars_.end()) {
      continue;
    }
    auto found{llvm::find(captured, given->second)};
    if (found != captured.end()) {
      places[variable] = {static_cast<unsigned>(infoKeys.size() +
                              dictionaryKeys.size() + (found - captured.begin())),
          place.second};
    }
  }
  for (BodyVar var : captured) {
    if (var < kFirstGivenVar) {
      mlir::Attribute capture{CaptureTypeAttr(*body.vars[var].type, places, at)};
      if (!capture) {
        return {};
      }
      captures.push_back(capture);
    } else {
      mlir::Type given{values_.lookup(var).getType()};
      if (!mlir::isa<merk::TypeInfoType, merk::TypeclassInfoType>(given)) {
        NotYet(at, "a closure that captures what a goal gave, which is not a "
                   "type_info or a typeclass_info");
        return {};
      }
      captures.push_back(builder_.getStringAttr(
          mlir::isa<merk::TypeInfoType>(given) ? "type_info"
                                               : "typeclass_info"));
    }
  }
  llvm::SmallVector<mlir::Value> words;
  for (mlir::Value value : capturedValues) {
    words.push_back(Coerce(value, word, loc));
  }
  return merk::ClosureOp::create(builder_, loc,
      merk::ClosureType::get(&context_),
      mlir::FlatSymbolRefAttr::get(&context_, entry.getSymName()), words,
      CapturesAttr(captures));
}

namespace {
// The goals of a conjunction, with those of the conjunctions in it.
void Conjuncts(const Body &goal, llvm::SmallVectorImpl<const Body *> &out) {
  if (goal.kind != BodyKind::Conj) {
    out.push_back(&goal);
    return;
  }
  for (const Body *conjunct : goal.goals) {
    Conjuncts(*conjunct, out);
  }
}

// The goal of a try goal, if `goal` is its scope or a conjunction of only it.
const Body *TryGoalOf(const Body &goal) {
  llvm::SmallVector<const Body *> goals;
  Conjuncts(goal, goals);
  return goals.size() == 1 && goals.front()->kind == BodyKind::Scope &&
          goals.front()->scope == semantics::ScopeKind::TryGoal
      ? goals.front()
      : nullptr;
}

// The procedure of exception.try or try_io for a goal of this determinism:
// their modes are in that order (library/exception.m).
const Procedure *TryProcedure(llvm::ArrayRef<const Procedure *> procedures,
    bool io, parser::Determinism determinism) {
  using parser::Determinism;
  std::size_t index;
  switch (determinism) {
  case Determinism::Det:
  case Determinism::Erroneous:
    index = 0;
    break;
  case Determinism::Semidet:
  case Determinism::Failure:
    index = io ? 2 : 1;
    break;
  case Determinism::CcMulti:
    index = io ? 1 : 2;
    break;
  case Determinism::CcNondet:
    index = io ? 2 : 3;
    break;
  default:
    return nullptr;
  }
  return index < procedures.size() ? procedures[index] : nullptr;
}

// The determinism of the closure that try or try_io calls for a goal.
parser::Determinism TryClosureDeterminism(parser::Determinism determinism) {
  switch (determinism) {
  case parser::Determinism::Erroneous:
    return parser::Determinism::Det;
  case parser::Determinism::Failure:
    return parser::Determinism::Semidet;
  default:
    return determinism;
  }
}
} // namespace

void MerkGenerator::Try(const Body &goal, const ProcedureBody &body) {
  parser::CharBlock at{Where(goal.tokens)};
  mlir::Location loc{Location(goal.tokens)};
  BodyVar resultVar{goal.tryVars.front()};
  bool io{goal.tryVars.size() > 1};
  if (goal.tryVars.size() > 3) {
    NotYet(at, "try goals with more than one io parameter");
    return;
  }
  // The expansion: magic_exception_result(Result), then a switch on Result
  // with the cases exception and succeeded.
  llvm::SmallVector<const Body *> top;
  Conjuncts(*goal.goals.front(), top);
  const Body *switchGoal{nullptr};
  for (const Body *g : top) {
    if (g->kind == BodyKind::Switch && g->switchVar == resultVar) {
      switchGoal = g;
    }
  }
  const semantics::SwitchCase *exceptionCase{nullptr}, *succeededCase{nullptr};
  if (switchGoal) {
    for (const semantics::SwitchCase &c : switchGoal->cases) {
      if (c.functor.name == "exception") {
        exceptionCase = &c;
      } else if (c.functor.name == "succeeded") {
        succeededCase = &c;
      }
    }
  }
  if (!exceptionCase || !succeededCase) {
    NotYet(at, "this try goal's form");
    return;
  }
  // In the succeeded case, after the tests of Result: the goal, then the
  // then part; or an if-then-else of the goal, the then and else parts.
  llvm::SmallVector<const Body *> succeeded;
  Conjuncts(*succeededCase->goal, succeeded);
  const Body *tryGoal{nullptr};
  const Body *elseGoal{nullptr};
  llvm::SmallVector<const Body *> thenGoals;
  for (auto [i, g] : llvm::enumerate(succeeded)) {
    if (const Body *own{TryGoalOf(*g)}) {
      tryGoal = own;
      thenGoals.append(succeeded.begin() + i + 1, succeeded.end());
      break;
    }
    if (g->kind == BodyKind::IfThenElse) {
      if (const Body *own{TryGoalOf(*g->goals[0])}) {
        tryGoal = own;
        thenGoals.push_back(g->goals[1]);
        elseGoal = g->goals[2];
        thenGoals.append(succeeded.begin() + i + 1, succeeded.end());
        break;
      }
    }
  }
  if (!tryGoal) {
    NotYet(at, "this try goal's form");
    return;
  }
  auto determinismIt{body.goals.find(tryGoal)};
  if (determinismIt == body.goals.end()) {
    NotYet(at, "this try goal's determinism");
    return;
  }
  parser::Determinism goalDeterminism{
      TryClosureDeterminism(determinismIt->second)};
  std::optional<merk::Determinism> closureDeterminism{
      ClosureDeterminism(goalDeterminism)};
  // What the goal binds that the goals after it use, but its io states.
  llvm::SmallVector<BodyVar> outputs;
  llvm::SmallVector<const semantics::Type *> outputTypes;
  for (BodyVar var : tryGoal->nonlocals) {
    std::optional<Represented> represented{Represent(body.vars[var].type)};
    if (values_.lookup(var) || llvm::is_contained(goal.tryVars, var) ||
        (represented && represented->dummy)) {
      continue;
    }
    if (!represented) {
      NotYet(at, "values of type " +
              semantics::TypeToString(*body.vars[var].type));
      return;
    }
    outputs.push_back(var);
    outputTypes.push_back(body.vars[var].type);
  }
  const semantics::Type *tuple{types_.types().Tuple(outputTypes)};
  std::optional<Represented> tupleRepresented{Represent(tuple)};
  const semantics::Symbol *resultType{
      body.vars[resultVar].type->kind == semantics::TypeKind::Named
          ? body.vars[resultVar].type->constructor
          : nullptr};
  const semantics::ModuleSymbols *exception{resolution_.Module("exception")};
  const semantics::Symbol *callee{nullptr};
  if (exception) {
    for (const semantics::Symbol *symbol :
        exception->Lookup(io ? "try_io" : "try")) {
      if (symbol->kind == semantics::SymbolKind::Predicate) {
        callee = symbol;
      }
    }
  }
  const Procedure *procedure{callee
          ? TryProcedure(modes_.ProceduresOf(*callee), io, goalDeterminism)
          : nullptr};
  if (!closureDeterminism || !tupleRepresented || !resultType || !procedure) {
    NotYet(at, "this try goal");
    return;
  }
  // The closure: it gives the tuple, and the io state after the goal.
  llvm::SmallVector<ClosureArgument> arguments{{true, false}};
  if (io) {
    arguments.append({{false, true}, {true, false}});
  }
  auto word{merk::WordType::get(&context_)};
  bool committed{goalDeterminism == parser::Determinism::CcMulti ||
      goalDeterminism == parser::Determinism::CcNondet};
  mlir::Value closure{OutlineClosure(*tryGoal, *tryGoal, {}, arguments, {},
      {}, io ? 2u : 1u, *closureDeterminism, committed, std::nullopt, body,
      [&](mlir::Location at) {
        llvm::SmallVector<mlir::Value> words;
        llvm::SmallVector<mlir::Value> elements;
        for (BodyVar var : outputs) {
          elements.push_back(ValueOf(var));
        }
        if (llvm::is_contained(elements, mlir::Value{})) {
          words.push_back(mlir::Value{});
          return words;
        }
        if (tupleRepresented->dummy) {
          words.push_back(merk::ToWordOp::create(builder_, at, word,
              IntegerConstant(builder_.getI64Type(), 0, at)));
        } else {
          words.push_back(Coerce(
              merk::ConstructOp::create(builder_, at, tupleRepresented->type,
                  merk::FunctorAttr::get(&context_, "{}",
                      static_cast<unsigned>(elements.size())),
                  elements),
              word, at));
        }
        if (io) {
          llvm::SmallVector<mlir::Value> state{
              WordResults({goal.tryVars[2]}, body, at)};
          words.append(state.begin(), state.end());
        }
        return words;
      })};
  if (!closure) {
    failed_ = true;
    return;
  }
  // try(Closure, Result) or try_io(Closure, Result, IO0, IO), whose io
  // states have no bits.
  merk::ProcOp proc{
      ProcFor(*procedure, types_.SignatureOf(*callee)->arguments, at)};
  mlir::Value info{TypeInfoOfType(tuple, loc, at)};
  if (!proc || !info) {
    failed_ = true;
    return;
  }
  mlir::FunctionType function{proc.getFunctionType()};
  llvm::SmallVector<mlir::Value> inputs{info, closure};
  if (function.getNumInputs() != inputs.size() ||
      function.getNumResults() != 1) {
    NotYet(at, "this try goal's call of exception." +
            std::string{io ? "try_io" : "try"});
    return;
  }
  for (auto [input, declared] : llvm::zip(inputs, function.getInputs())) {
    input = Coerce(input, declared, loc);
  }
  auto call{merk::CallOp::create(
      builder_, loc, function.getResults(), proc.getSymName(), inputs)};
  const semantics::Type *realType{types_.types().Named(*resultType, {tuple})};
  std::optional<Represented> realRepresented{Represent(realType)};
  std::optional<Represented> magicRepresented{
      Represent(body.vars[resultVar].type)};
  if (!realRepresented || !magicRepresented) {
    NotYet(at, "this try goal's result");
    return;
  }
  // The results' type_infos, whose type constructors' descriptors the
  // collector needs to describe them as roots.
  if (!TypeInfoOfType(realType, loc, at) ||
      !TypeInfoOfType(body.vars[resultVar].type, loc, at)) {
    failed_ = true;
    return;
  }
  mlir::Value result{Coerce(call.getResult(0), realRepresented->type, loc)};
  // The switch on the result: exception, succeeded, and failed when the
  // goal can fail.
  llvm::SmallVector<mlir::Attribute> cases{
      builder_.getArrayAttr({merk::FunctorAttr::get(&context_, "exception", 1)}),
      builder_.getArrayAttr(
          {merk::FunctorAttr::get(&context_, "succeeded", 1)})};
  bool canFail{goalDeterminism == parser::Determinism::Semidet ||
      goalDeterminism == parser::Determinism::CcNondet};
  if (canFail) {
    if (!elseGoal) {
      NotYet(at, "this try goal's form");
      return;
    }
    cases.push_back(
        builder_.getArrayAttr({merk::FunctorAttr::get(&context_, "failed", 0)}));
  }
  bool nondeterministic{Nondeterministic(*exceptionCase->goal, body) ||
      (elseGoal && Nondeterministic(*elseGoal, body))};
  for (const Body *g : thenGoals) {
    nondeterministic = nondeterministic || Nondeterministic(*g, body);
  }
  Values before{values_};
  merk::EachOp each;
  if (nondeterministic) {
    each = BeginEach(loc);
  }
  auto op{merk::SwitchOp::create(builder_, loc, mlir::TypeRange{}, result,
      builder_.getArrayAttr(cases), false)};
  llvm::SmallVector<Branch> branches;
  {
    mlir::OpBuilder::InsertionGuard guard{builder_};
    auto regions{op.getCaseRegions()};
    // exception(Univ): the catches, which take Result apart, and which may
    // rethrow it.
    values_ = before;
    builder_.setInsertionPointToStart(&regions[0].emplaceBlock());
    values_[resultVar] = Coerce(result, magicRepresented->type, loc);
    Goal(*exceptionCase->goal, body);
    branches.push_back({builder_.getInsertionBlock(), values_, Terminated()});
    // succeeded(Tuple): the goal's bindings, then the then part.
    values_ = before;
    builder_.setInsertionPointToStart(&regions[1].emplaceBlock());
    // A goal that never succeeds: the case is never taken, and need not bind
    // what the catches bind.
    if (determinismIt->second == parser::Determinism::Erroneous ||
        determinismIt->second == parser::Determinism::Failure) {
      merk::FailOp::create(builder_, loc);
    } else if (!outputs.empty()) {
      auto value{merk::DeconstructOp::create(builder_, loc,
          mlir::TypeRange{tupleRepresented->type}, result,
          merk::FunctorAttr::get(&context_, "succeeded", 1))};
      llvm::SmallVector<mlir::Type> fields;
      for (const semantics::Type *type : outputTypes) {
        fields.push_back(Represent(type)->type);
      }
      auto elements{merk::DeconstructOp::create(builder_, loc, fields,
          value.getFields()[0],
          merk::FunctorAttr::get(
              &context_, "{}", static_cast<unsigned>(outputs.size())))};
      for (auto [var, element] : llvm::zip(outputs, elements.getFields())) {
        values_[var] = element;
      }
    }
    for (const Body *g : thenGoals) {
      if (failed_ || Terminated()) {
        break;
      }
      Goal(*g, body);
    }
    branches.push_back({builder_.getInsertionBlock(), values_, Terminated()});
    // failed: the else part.
    if (canFail) {
      values_ = before;
      builder_.setInsertionPointToStart(&regions[2].emplaceBlock());
      Goal(*elseGoal, body);
      branches.push_back({builder_.getInsertionBlock(), values_, Terminated()});
    }
  }
  if (failed_) {
    values_ = before;
    return;
  }
  Join(op, branches, before, loc, each);
}

void MerkGenerator::NameClosure(const Body &goal, const ProcedureBody &body) {
  const Procedure &callee{*goal.modes->callee};
  parser::CharBlock at{Where(goal.tokens)};
  mlir::Location loc{Location(goal.tokens)};
  if (callee.symbol->owner &&
      callee.symbol->owner->kind == semantics::SymbolKind::Typeclass) {
    NotYet(at, "closures of methods");
    return;
  }
  // mmc reports such a closure as an error ("Where mmc and the manual
  // disagree").
  if (!HiddenOutputs(callee).empty()) {
    NotYet(at, "closures of existentially typed procedures");
    return;
  }
  std::optional<llvm::SmallVector<mlir::Type>> hidden{HiddenInputs(callee, at)};
  if (!hidden) {
    failed_ = true;
    return;
  }
  // A polymorphic callee takes its declared types; a closure's arguments
  // after the curried ones have its type's.
  const semantics::Type *closure{body.vars[goal.left].type};
  std::size_t curried{goal.arguments.size()};
  llvm::SmallVector<const semantics::Type *> types;
  if (!hidden->empty()) {
    llvm::ArrayRef<const semantics::Type *> declared{
        types_.SignatureOf(*callee.symbol)->arguments};
    types.append(declared.begin(), declared.end());
  } else {
    for (BodyVar argument : goal.arguments) {
      types.push_back(body.vars[argument].type);
    }
    types.append(closure->arguments.begin(), closure->arguments.end());
    if (closure->result) {
      types.push_back(closure->result);
    }
  }
  if (types.size() != callee.modes.size()) {
    NotYet(at, "this closure's arguments");
    return;
  }
  // What it captures: what the call in its body would pass, as polymorphism
  // analysis says, and the curried arguments.
  llvm::SmallVector<mlir::Value> captured;
  if (!hidden->empty()) {
    const semantics::PassedInfos *passed{nullptr};
    if (current_) {
      if (auto it{current_->goals.find(&goal)}; it != current_->goals.end()) {
        passed = &it->second;
      }
    }
    if (!passed ||
        passed->typeInfos.size() + passed->dictionaries.size() != hidden->size()) {
      NotYet(at, "this closure's type_infos");
      return;
    }
    for (const semantics::TypeInfo *info : passed->typeInfos) {
      PassTypeInfo(*info, captured, loc, at);
    }
    for (const semantics::Dictionary *dictionary : passed->dictionaries) {
      PassDictionary(*dictionary, captured, loc, at);
    }
    if (failed_) {
      return;
    }
  }
  // The declared types' variables are the hidden arguments', in their order.
  CapturePlaces places;
  llvm::SmallVector<mlir::Attribute> captures;
  if (!hidden->empty()) {
    const semantics::PassedInfos &passed{current_->goals.find(&goal)->second};
    const semantics::Signature &signature{*types_.SignatureOf(*callee.symbol)};
    llvm::SmallVector<bool> constrained{
        ConstrainedVariables(signature.typeVars, signature.constraints)};
    unsigned infos{0};
    for (unsigned var{0}; var < signature.typeVars; ++var) {
      if (!constrained[var]) {
        places[var] = {infos++, 0};
      }
    }
    for (unsigned var{0}; var < signature.typeVars; ++var) {
      if (constrained[var]) {
        if (auto place{ConstrainedPlace(var, signature.constraints, infos)}) {
          places[var] = *place;
        }
      }
    }
    captures.append(
        passed.typeInfos.size(), builder_.getStringAttr("type_info"));
    captures.append(
        passed.dictionaries.size(), builder_.getStringAttr("typeclass_info"));
  }
  auto word{merk::WordType::get(&context_)};
  for (auto [argument, type, mode] :
      llvm::zip(goal.arguments, types, callee.modes)) {
    std::optional<Represented> declared{Represent(type)};
    std::optional<Represented> actual{Represent(body.vars[argument].type)};
    if (!declared || !actual) {
      NotYet(at, "values of type " + semantics::TypeToString(*type));
      return;
    }
    if (!declared->dummy && !IsUnused(mode)) {
      mlir::Attribute capture{CaptureTypeAttr(*type, places, at)};
      if (!capture) {
        return;
      }
      captures.push_back(capture);
      captured.push_back(actual->dummy
              ? ZeroWord(loc)
              : Coerce(Coerce(ValueOf(argument), declared->type, loc), word, loc));
    }
  }
  for (mlir::Value &value : captured) {
    value = Coerce(value, word, loc);
  }
  bool failed{false};
  merk::ProcOp entry{BuiltinEntry(
      goal, callee, types, static_cast<unsigned>(curried), failed)};
  if (failed) {
    return;
  }
  if (!entry) {
    // A procedure of this module is called through its word copy.
    const ProcedureBody *copyable{CopyableBody(callee)};
    merk::ProcOp proc{copyable ? WordProcFor(*copyable) : ProcFor(callee, types, at)};
    if (!proc) {
      failed_ = true;
      return;
    }
    llvm::SmallVector<ClosureArgument> arguments;
    for (auto [mode, type] : llvm::drop_begin(
             llvm::zip(callee.modes, types), static_cast<unsigned>(curried))) {
      bool dummy{Represent(type)->dummy || IsUnused(mode)};
      bool output{IsOutput(mode)};
      arguments.push_back({output, dummy && !(output && copyable)});
    }
    entry = ClosureEntry(proc, static_cast<unsigned>(captured.size()), arguments, at);
    if (!entry) {
      return;
    }
  }
  values_[goal.left] = merk::ClosureOp::create(builder_, loc,
      merk::ClosureType::get(&context_),
      mlir::FlatSymbolRefAttr::get(&context_, entry.getSymName()), captured,
      CapturesAttr(captures));
}

void MerkGenerator::HigherOrderCall(const Body &goal, const ProcedureBody &body) {
  parser::CharBlock at{Where(goal.tokens)};
  mlir::Location loc{Location(goal.tokens)};
  mlir::Value closure{ValueOf(goal.closure)};
  llvm::ArrayRef<semantics::ArgMode> modes{goal.modes->lambda};
  if (!closure || modes.size() != goal.arguments.size()) {
    NotYet(at, "this higher-order call");
    return;
  }
  // Its implied mode tests whole values, so a partial term's bound parts
  // would go unchecked.
  for (BodyVar argument : goal.arguments) {
    if (PartialTerm(body, argument) && !values_.lookup(argument)) {
      NotYet(at, "this higher-order call with a partially instantiated term");
      return;
    }
  }
  std::optional<merk::Determinism> determinism{
      ClosureDeterminism(goal.modes->determinism)};
  if (!determinism) {
    NotYet(at,
        std::string{semantics::DeterminismName(goal.modes->determinism)} +
            " closures");
    return;
  }
  for (auto [argument, mode] : llvm::zip(goal.arguments, modes)) {
    if (OpenParameter(mode, body.vars[argument].type)) {
      // Temporary (docs/INPROGRESS.md, Open types: what is left): a closure's
      // entry would need to know which of its parameters take open words.
      NotYet(at,
          "higher-order calls with closed(T) arguments that may be "
          "unbound");
      return;
    }
  }
  // Every argument is a word, one of a type with no values the word 0.
  auto word{merk::WordType::get(&context_)};
  llvm::SmallVector<mlir::Value> inputs;
  llvm::SmallVector<BodyVar> outputs;
  llvm::SmallVector<std::pair<BodyVar, mlir::Value>> bound;
  for (auto [argument, mode] : llvm::zip(goal.arguments, modes)) {
    const semantics::Type *type{body.vars[argument].type};
    std::optional<Represented> represented{Represent(type)};
    if (!represented) {
      NotYet(at, "values of type " + semantics::TypeToString(*type));
      return;
    }
    if (IsOutput(mode)) {
      outputs.push_back(argument);
      // One bound already is tested against the call's.
      if (mlir::Value value{ValueOf(argument)}; value && !represented->dummy) {
        bound.push_back({argument, value});
      }
    } else if (!represented->dummy && !IsUnused(mode) && IsClosedType(type)) {
      // A closure's entry takes T's word.
      mlir::Value value{ClosedAs(ValueOf(argument), false, type, loc, at)};
      if (!value) {
        failed_ = true;
        return;
      }
      inputs.push_back(Coerce(value, word, loc));
    } else {
      inputs.push_back(represented->dummy || IsUnused(mode)
              ? ZeroWord(loc)
              : Coerce(ValueOf(argument), word, loc));
    }
  }
  // A nondet or multi closure runs the rest of the code, in its
  // continuation, for each of its solutions.
  mlir::ValueRange results;
  if (*determinism == merk::Determinism::Nondet ||
      *determinism == merk::Determinism::Multi) {
    auto call{merk::NondetCallClosureOp::create(builder_, loc, closure, inputs)};
    mlir::Block &continuation{call.getContinuation().emplaceBlock()};
    for (std::size_t i{0}; i < outputs.size(); ++i) {
      continuation.addArgument(word, loc);
    }
    builder_.setInsertionPointToStart(&continuation);
    results = continuation.getArguments();
  } else {
    auto call{merk::CallClosureOp::create(builder_, loc,
        llvm::SmallVector<mlir::Type>(outputs.size(), word), closure, inputs,
        *determinism)};
    results = call.getResults();
  }
  if (goal.modes->determinism == parser::Determinism::Erroneous ||
      goal.modes->determinism == parser::Determinism::Failure) {
    merk::FailOp::create(builder_, loc);
    return;
  }
  for (auto [var, result] : llvm::zip_equal(outputs, results)) {
    SetFromWord(var, result, body, loc);
  }
  for (auto [var, value] : bound) {
    mlir::Value output{ValueOf(var)};
    values_[var] = value;
    Test(goal, value, output, body.vars[var].type);
  }
}

} // namespace merkurius::codegen::mlirgen
