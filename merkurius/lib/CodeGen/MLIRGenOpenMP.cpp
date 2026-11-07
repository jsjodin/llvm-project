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

namespace merkurius::codegen::mlirgen {

mlir::Value MerkGenerator::Slot(mlir::Type type, mlir::Location loc) {
  auto proc{builder_.getInsertionBlock()->getParent()
                ->getParentOfType<merk::ProcOp>()};
  mlir::OpBuilder::InsertionGuard guard{builder_};
  builder_.setInsertionPointToStart(&proc.getBody().front());
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

// Only the scalars, integers and floats, and arrays of them can cross into
// a target region so far.
static bool Scalar(mlir::Type type) {
  return mlir::isa<mlir::IntegerType, mlir::FloatType>(type);
}

static bool Mappable(mlir::Type type) {
  if (auto array{mlir::dyn_cast<merk::ArrayType>(type)}) {
    return Scalar(array.getElement());
  }
  return Scalar(type);
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
  return id == llvm::omp::OMPC_reduction &&
      (leaf == llvm::omp::OMPD_do || leaf == llvm::omp::OMPD_teams);
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
  if (directive.id != llvm::omp::Directive::OMPD_parallel_do && !offloaded) {
    NotYet(at, "the " + name + " directive");
    return;
  }
  if (builder_.getInsertionBlock()
          ->getParent()
          ->getParentOfType<mlir::omp::TargetOp>()) {
    NotYet(at,
        offloaded ? std::string{"target regions in target regions"}
                  : name + " in target regions");
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
  // The leaves around the loop's wrappers.
  bool teams{HasLeaf(*queue, llvm::omp::Directive::OMPD_teams)};
  bool distribute{HasLeaf(*queue, llvm::omp::Directive::OMPD_distribute)};
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
  for (LoopReduction &reduction : reductions) {
    reduction.slot = Slot(reduction.type, loc);
    mlir::LLVM::StoreOp::create(builder_, loc,
        ValueOf(call->arguments[reduction.value->in]), reduction.slot);
    updated.push_back({reduction.slot, reduction.type, reduction.value->name});
  }
  ParallelLoop parallelLoop{goals, trip, call, loop, *loopBody, bodyCall,
      bodyProcedure, reductions, teams, distribute, offloaded, at};
  if (!offloaded) {
    GenerateParallelLoop(parallelLoop, body, {}, loc);
    if (!failed_) {
      LoadReductions(reductions, *call, loc);
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
  Target(
      goal, body,
      [&](mlir::ValueRange evaluated) {
        GenerateParallelLoop(parallelLoop, body, evaluated, loc);
      },
      /*combined=*/true, kernel.mode, hostEval, updated);
  if (!failed_) {
    LoadReductions(reductions, *call, loc);
  }
}

// Offloaded, the kernel is as Flang makes it (KernelOf): for target teams
// distribute parallel do, SPMD, and the host computes the bounds and the
// step before the target region and gives them to it as host_eval
// values, from which it computes the trip count and so the number of
// teams and threads, and the nest runs over them. The bounds'
// goals are in the region too, on both sides, so that the host and the
// device map the same values; the device's nest runs over those, as
// Flang's does, since the device has no host_eval.
void MerkGenerator::GenerateParallelLoop(const ParallelLoop &loop,
    const ProcedureBody &body, mlir::ValueRange hostEval, mlir::Location loc) {
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
  auto parallel{mlir::omp::ParallelOp::create(
      builder_, loc, mlir::omp::ParallelOperands{})};
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
  if (!hostEval.empty()) {
    lower = hostEval[0];
    upper = hostEval[1];
    step = hostEval[2];
  }

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
  mlir::omp::YieldOp::create(builder_, loc, identity);
  mlir::Block *combiner{builder_.createBlock(
      &decl.getReductionRegion(), {}, {type, type}, {loc, loc})};
  mlir::Value a{combiner->getArgument(0)};
  mlir::Value b{combiner->getArgument(1)};
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
  mlir::omp::YieldOp::create(builder_, loc, combined);
  return decl;
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
    if (queue && LeafClauses(queue->front(), at,
            [](llvm::omp::Directive, llvm::omp::Clause id, bool) {
              return id == llvm::omp::OMPC_map;
            })) {
      TargetData(goal, body, queue->front());
    }
    return;
  }
  if (directive.id != llvm::omp::Directive::OMPD_target) {
    NotYet(at,
        "the " + std::string{parser::EnumToString(directive.id)} +
            " directive");
    return;
  }
  std::optional<omp::ConstructQueue> queue{Decompose(directive, at)};
  if (!queue || !LeafClauses(queue->front(), at)) {
    return;
  }
  if (builder_.getInsertionBlock()->getParent()
          ->getParentOfType<mlir::omp::TargetOp>()) {
    NotYet(at, "target regions in target regions");
    return;
  }
  Target(goal, body, [&](mlir::ValueRange) { Goal(*goal.goals.front(), body); },
      /*combined=*/false, KernelOf(*queue).mode);
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
  mlir::omp::TargetDataOperands operands;
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
    llvm::function_ref<void(mlir::ValueRange hostEval)> generate, bool combined,
    mlir::omp::TargetExecMode mode, mlir::ValueRange hostEval,
    llvm::ArrayRef<UpdatedSlot> slots) {
  parser::CharBlock at{Where(goal.tokens)};
  mlir::Location loc{Location(goal.tokens)};
  Values before{values_};
  // What `generate` builds, in a target region whose map entries are added
  // after it: they are the values it uses from outside and those it binds.
  // The host_eval arguments come before the map entries' arguments.
  mlir::omp::TargetExtOperands operands;
  operands.kernelType = mlir::omp::TargetExecModeAttr::get(&context_, mode);
  operands.hostEvalVars.append(hostEval.begin(), hostEval.end());
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
    generate(block->getArguments());
  }
  if (failed_) {
    values_ = before;
    return;
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
  // The device has no runtime to allocate or fill an array.
  mlir::Operation *allocates{nullptr};
  region.walk([&](mlir::Operation *op) {
    if (mlir::isa<merk::ArrayNewOp, merk::ArrayCopyOp, merk::ArrayResizeOp,
            merk::ArrayFillOp>(op)) {
      allocates = op;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  if (allocates) {
    NotYet(at,
        mlir::isa<merk::ArrayFillOp>(allocates)
            ? "filling arrays in target regions"
            : "making arrays in target regions");
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
  // An array the region gives is one it was given, updated in place, since
  // the device cannot allocate one: it is mapped back, and after the
  // region is the given array. When the region's operations do not show
  // which, as when it calls a procedure that updates it, the region finds
  // which given array of its type it is, by its pointer, and gives its
  // number, and its pointer too: when the region runs on the host, a
  // procedure it calls may have allocated it.
  llvm::DenseMap<BodyVar, mlir::Value> updated;
  llvm::DenseMap<BodyVar, llvm::SmallVector<mlir::Value>> candidates;
  llvm::DenseMap<BodyVar, mlir::Type> arrayTypes;
  for (BodyVar var : outputs) {
    mlir::Value value{values_.lookup(var)};
    if (!Mappable(value.getType())) {
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
    llvm::StringRef name{var ? llvm::StringRef{body.vars[*var].name} : ""};
    if (mlir::isa<merk::ArrayType>(input.getType())) {
      maps.push_back(MapArray(input,
          mappedBack.contains(input)
              ? mlir::omp::ClauseMapFlags::to | mlir::omp::ClauseMapFlags::from
              : mlir::omp::ClauseMapFlags::to,
          name, loc));
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
    mlir::Type type{candidates.contains(var) ? builder_.getI64Type()
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
  // does, for the reason OmpLoop computes its bounds there.
  mlir::Block *reads{block};
  if (auto teams{mlir::dyn_cast<mlir::omp::TeamsOp>(block->front())};
      teams && outputs.empty()) {
    reads = &teams.getRegion().front();
    if (auto parallel{mlir::dyn_cast<mlir::omp::ParallelOp>(reads->front())}) {
      reads = &parallel.getRegion().front();
    }
  }
  builder_.setInsertionPointToStart(reads);
  llvm::DenseMap<mlir::Value, mlir::Value> loaded;
  for (mlir::Value input : inputs) {
    // An array's argument is its pointer on the device, and so is an
    // updated slot's.
    mlir::Value argument{block->addArgument(ptr, loc)};
    mlir::Value value{FindUpdatedSlot(slots, input) ? argument
            : mlir::isa<merk::ArrayType>(input.getType())
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
  // Where the goal ended: the last block of the region.
  mlir::Block *end{&targetOp.getRegion().back()};
  builder_.setInsertionPointToEnd(end);
  if (!Terminated()) {
    for (auto [var, arg] : llvm::zip(outputs, outputArgs)) {
      mlir::Value value{values_.lookup(var)};
      if (auto same{candidates.find(var)}; same != candidates.end()) {
        // The number of the given array whose pointer it is, or -1.
        auto i64{builder_.getI64Type()};
        mlir::LLVM::StoreOp::create(
            builder_, loc, ArrayPointer(value, loc), pointerArgs.lookup(var));
        mlir::Value number{IntegerConstant(i64, -1, loc)};
        for (auto [i, input] : llvm::enumerate(same->second)) {
          mlir::Value is{mlir::LLVM::ICmpOp::create(builder_, loc,
              mlir::LLVM::ICmpPredicate::eq, ArrayPointer(value, loc),
              ArrayPointer(loaded.lookup(input), loc))};
          number = mlir::arith::SelectOp::create(
              builder_, loc, is, IntegerConstant(i64, i, loc), number);
        }
        value = number;
      }
      mlir::LLVM::StoreOp::create(builder_, loc,
          loaded.lookup(value) ? loaded.lookup(value) : value, arg);
    }
    mlir::omp::TerminatorOp::create(builder_, loc);
  }
  builder_.setInsertionPointAfter(targetOp);
  values_ = before;
  for (auto [var, type, slot] : llvm::zip(outputs, outputTypes, outputSlots)) {
    values_[var] = mlir::LLVM::LoadOp::create(builder_, loc, type, slot);
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
