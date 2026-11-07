//===-- lib/CodeGen/MLIRGenControl.cpp - Control flow ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Conjunctions, if-then-elses, disjunctions, commits, switches, negations
// and trace goals.
//
//===----------------------------------------------------------------------===//

#include "MLIRGenImpl.h"

namespace merkurius::codegen::mlirgen {

void MerkGenerator::Test(const Body &goal, mlir::Value x, mlir::Value y,
    const semantics::Type *type) {
  llvm::StringSet<> seen;
  if (!x || !y || !Comparable(x.getType(), seen)) {
    NotYet(Where(goal.tokens), "this unification");
    return;
  }
  equalAt_ = Where(goal.tokens);
  mlir::Location loc{Location(goal.tokens)};
  if (HasGivenType(*type)) {
    mlir::Value info{GivenTypeInfo(goal, type, loc)};
    if (!info) {
      return;
    }
    auto word{merk::WordType::get(&context_)};
    merk::GuardOp::create(builder_, loc,
        merk::UnifyWordOp::create(builder_, loc, builder_.getI1Type(), info,
            Coerce(x, word, loc), Coerce(y, word, loc)));
    return;
  }
  Equal(x, y, type, loc);
}

mlir::Value MerkGenerator::GivenTypeInfo(
    const Body &goal, const semantics::Type *type, mlir::Location loc) {
  parser::CharBlock at{Where(goal.tokens)};
  if (current_) {
    if (auto it{current_->goals.find(&goal)};
        it != current_->goals.end() && !it->second.typeInfos.empty()) {
      return TypeInfoValue(*it->second.typeInfos.front(), loc, at);
    }
  }
  NotYet(at, "a type_info for type " + semantics::TypeToString(*type));
  return nullptr;
}

std::vector<BodyVar> MerkGenerator::BoundSince(
    const Values &now, const Values &before) {
  std::vector<BodyVar> bound;
  for (const auto &[var, value] : now) {
    if (!before.count(var)) {
      bound.push_back(var);
    }
  }
  llvm::sort(bound);
  return bound;
}

void MerkGenerator::Join(mlir::Operation *op, llvm::ArrayRef<Branch> branches,
    const Values &before, mlir::Location loc, merk::EachOp each) {
  // The variables that every branch that succeeds binds; those of a branch
  // only are its own.
  std::optional<std::vector<BodyVar>> results;
  for (const Branch &branch : branches) {
    if (branch.fails) {
      continue;
    }
    std::vector<BodyVar> bound{BoundSince(branch.values, before)};
    if (!results) {
      results = std::move(bound);
    } else {
      std::vector<BodyVar> both;
      std::set_intersection(results->begin(), results->end(), bound.begin(),
          bound.end(), std::back_inserter(both));
      results = std::move(both);
    }
  }
  if (!results) {
    results.emplace();
  }
  // The word of a variable of a type with no values that one branch's call
  // gave, in a body that gives words, leaves with the word 0 of the others,
  // so that the call stays a tail call.
  std::vector<BodyVar> words;
  for (const Branch &branch : branches) {
    if (branch.fails) {
      continue;
    }
    for (BodyVar var : BoundSince(branch.values, before)) {
      if (IsDummyWord(var) && !llvm::is_contained(*results, var)) {
        words.push_back(var);
      }
    }
  }
  results->insert(results->end(), words.begin(), words.end());
  llvm::sort(*results);
  results->erase(std::unique(results->begin(), results->end()), results->end());
  // In a body that gives words, merk.each's continuation takes the body's
  // outputs as words, so that a branch whose call gives them as words gives
  // them on unchanged, and the continuation passes on a solution that it
  // only gives as the body's (EraseUnwordedOutputs).
  auto word{merk::WordType::get(&context_)};
  llvm::SmallVector<bool> asWords;
  for (BodyVar var : *results) {
    asWords.push_back(each && wordOutputs_ && llvm::is_contained(*wordOutputs_, var));
  }
  llvm::SmallVector<mlir::Type> types;
  for (const Branch &branch : branches) {
    if (branch.fails) {
      continue;
    }
    mlir::OpBuilder::InsertionGuard guard{builder_};
    builder_.setInsertionPointToEnd(branch.block);
    llvm::SmallVector<mlir::Value> values;
    for (BodyVar var : *results) {
      mlir::Value value{branch.values.lookup(var)};
      values.push_back(value || !IsDummyWord(var) ? value : ZeroWord(loc));
    }
    if (types.empty() && !values.empty()) {
      for (mlir::Value value : values) {
        types.push_back(value.getType());
      }
    }
    if (each) {
      for (auto [value, asWord] : llvm::zip(values, asWords)) {
        if (asWord) {
          value = Coerce(value, word, loc);
        }
      }
      merk::SolutionOp::create(builder_, loc, values);
    } else {
      merk::YieldOp::create(builder_, loc, values);
    }
  }
  if (each) {
    // Every branch ends, and the rest of the code is the continuation.
    if (!op->hasTrait<mlir::OpTrait::IsTerminator>()) {
      builder_.setInsertionPointAfter(op);
      merk::FailOp::create(builder_, loc);
    }
    mlir::Block &continuation{each.getContinuation().emplaceBlock()};
    values_ = before;
    builder_.setInsertionPointToStart(&continuation);
    for (auto [var, type, asWord] : llvm::zip(*results, types, asWords)) {
      values_[var] = asWord
          ? Coerce(continuation.addArgument(word, loc), type, loc)
          : mlir::Value{continuation.addArgument(type, loc)};
    }
    return;
  }
  // The operation again, with the results.
  mlir::OperationState state{op->getLoc(), op->getName()};
  state.addOperands(op->getOperands());
  state.addAttributes(op->getAttrs());
  state.addTypes(types);
  for (mlir::Region &region : op->getRegions()) {
    state.addRegion()->takeBody(region);
  }
  builder_.setInsertionPoint(op);
  mlir::Operation *joined{builder_.create(state)};
  op->erase();
  builder_.setInsertionPointAfter(joined);
  values_ = before;
  for (auto [var, result] : llvm::zip_equal(*results, joined->getResults())) {
    values_[var] = result;
  }
  if (llvm::all_of(branches, [](const Branch &b) { return b.fails; })) {
    // No branch succeeds, so neither does the goal.
    merk::FailOp::create(builder_, loc);
  }
}

void MerkGenerator::IfThenElse(const Body &goal, const ProcedureBody &body) {
  if (Nondeterministic(*goal.goals[0], body)) {
    SoftIte(goal, body);
    return;
  }
  Ite(goal.tokens, [&] { Goal(*goal.goals[0], body); }, goal.goals[1],
      [&] { Goal(*goal.goals[2], body); }, body, goal.goals[2]);
}

// A disjunction with at most one solution as the goals around it see it,
// or in a context where only its first solution is wanted: the branches in
// turn, as if-then-elses, each tried when the one before it fails. One
// that may have to give its next solution, when a goal after it fails, is
// nondeterministic.
void MerkGenerator::Disjunction(const Body &goal, const ProcedureBody &body) {
  if (!body.goals.count(&goal)) {
    NotYet(Where(goal.tokens), "disjunctions of unknown determinism");
    return;
  }
  // One in merk.commit, whose branches have more than one solution, gives
  // its first: the first branch's first, or else the next branch's.
  if (Nondeterministic(goal, body) || committing_ == &goal) {
    NondetDisjunction(goal, body);
    return;
  }
  DisjunctionFrom(goal, 0, body);
}

void MerkGenerator::NondetDisjunction(const Body &goal, const ProcedureBody &body) {
  mlir::Location loc{Location(goal.tokens)};
  Values before{values_};
  merk::EachOp each{BeginEach(loc)};
  auto disj{merk::DisjOp::create(
      builder_, loc, static_cast<unsigned>(goal.goals.size()))};
  llvm::SmallVector<Branch> branches;
  {
    mlir::OpBuilder::InsertionGuard guard{builder_};
    for (auto [disjunct, region] : llvm::zip(goal.goals, disj.getBranches())) {
      values_ = before;
      builder_.setInsertionPointToStart(&region.emplaceBlock());
      Goal(*disjunct, body);
      branches.push_back({builder_.getInsertionBlock(), values_, Terminated()});
    }
  }
  if (failed_) {
    values_ = before;
    return;
  }
  Join(disj, branches, before, loc, each);
}

// The if-then-else has the then part's solutions for each of the
// condition's, so it has more than one too, and its then and else parts give
// theirs to the continuation that they share. In merk.commit, it gives the
// first of them.
void MerkGenerator::SoftIte(const Body &goal, const ProcedureBody &body) {
  mlir::Location loc{Location(goal.tokens)};
  Values before{values_};
  merk::EachOp each{BeginEach(loc)};
  auto ite{merk::SoftIteOp::create(builder_, loc)};
  llvm::SmallVector<Branch, 2> branches;
  {
    mlir::OpBuilder::InsertionGuard guard{builder_};
    // Each solution of the condition gives what it binds to the then part,
    // which PruneSolutions leaves out of if the then part does not use it.
    builder_.setInsertionPointToStart(&ite.getCondition().emplaceBlock());
    // The condition's solutions are the then part's, so no call in it is a
    // tail call that a word copy would keep.
    std::optional<llvm::SmallVector<BodyVar>> words{std::move(wordOutputs_)};
    wordOutputs_.reset();
    Goal(*goal.goals[0], body);
    wordOutputs_ = std::move(words);
    std::vector<BodyVar> bound;
    llvm::SmallVector<mlir::Value> given;
    for (BodyVar var : BoundSince(values_, before)) {
      if (mlir::Value value{values_.lookup(var)}) {
        bound.push_back(var);
        given.push_back(value);
      }
    }
    bool conditionFails{Terminated()};
    if (!conditionFails) {
      merk::SolutionOp::create(builder_, loc, given);
    }
    mlir::Block &then{ite.getThenRegion().emplaceBlock()};
    values_ = before;
    for (auto [var, value] : llvm::zip(bound, given)) {
      values_[var] = then.addArgument(value.getType(), loc);
    }
    builder_.setInsertionPointToStart(&then);
    if (conditionFails) {
      merk::FailOp::create(builder_, loc);
    } else {
      Goal(*goal.goals[1], body);
    }
    branches.push_back({builder_.getInsertionBlock(), values_, Terminated()});
    values_ = before;
    builder_.setInsertionPointToStart(&ite.getElseRegion().emplaceBlock());
    Goal(*goal.goals[2], body);
    branches.push_back({builder_.getInsertionBlock(), values_, Terminated()});
  }
  if (failed_) {
    values_ = before;
    return;
  }
  Join(ite, branches, before, loc, each);
}

merk::EachOp MerkGenerator::BeginEach(mlir::Location loc) {
  auto each{merk::EachOp::create(builder_, loc)};
  builder_.setInsertionPointToStart(&each.getGenerator().emplaceBlock());
  return each;
}

bool MerkGenerator::Nondeterministic(const Procedure &procedure) {
  std::optional<parser::Determinism> determinism{
      determinism_.DeterminismOf(procedure)};
  return determinism && (*determinism == parser::Determinism::Nondet ||
                            *determinism == parser::Determinism::Multi);
}

bool MerkGenerator::Nondeterministic(const Body &goal, const ProcedureBody &body) {
  auto it{body.goals.find(&goal)};
  return it != body.goals.end() &&
      (it->second == parser::Determinism::Nondet ||
          it->second == parser::Determinism::Multi);
}

// A goal with no outputs, or in a context where only its first solution is
// wanted, as a committed choice, has at most one solution as the goals
// around it see it, though a call in it, or the goals of a conjunction, or
// the branches of a disjunction, an if-then-else or a switch, may have
// more. The parts of a negation are such goals of their own.
bool MerkGenerator::NeedsCommit(const Body &goal, const ProcedureBody &body) {
  // The body of a committed choice procedure or lambda expression, whose
  // first solution is the procedure's, though the goals in it may want all
  // of a goal's.
  bool committedProc{&goal == body.body && body.determinism &&
      (*body.determinism == parser::Determinism::CcMulti ||
          *body.determinism == parser::Determinism::CcNondet)};
  if ((committedProc || &goal == committedLambda_) &&
      Nondeterministic(goal, body)) {
    return true;
  }
  if (!body.goals.count(&goal) || Nondeterministic(goal, body)) {
    return false;
  }
  auto anyNondet{[&](auto &&goals) {
    return llvm::any_of(goals, [&](const Body *g) { return Nondeterministic(*g, body); });
  }};
  switch (goal.kind) {
  case BodyKind::Call:
    return goal.modes && goal.modes->callee &&
        Nondeterministic(*goal.modes->callee);
  case BodyKind::Unify:
    return goal.unify == UnifyKind::Call && goal.modes && goal.modes->callee &&
        Nondeterministic(*goal.modes->callee);
  case BodyKind::HOCall:
    return goal.modes &&
        (goal.modes->determinism == parser::Determinism::Nondet ||
            goal.modes->determinism == parser::Determinism::Multi);
  case BodyKind::Conj:
  case BodyKind::Scope:
  case BodyKind::Disj:
    return anyNondet(goal.goals);
  case BodyKind::IfThenElse:
    return anyNondet(goal.goals);
  case BodyKind::Switch:
    return llvm::any_of(goal.cases, [&](const semantics::SwitchCase &c) {
      return Nondeterministic(*c.goal, body);
    });
  default:
    return false;
  }
}

// The goal in merk.commit's generator, whose first solution gives the values
// of the variables it binds.
void MerkGenerator::Commit(const Body &goal, const ProcedureBody &body) {
  mlir::Location loc{Location(goal.tokens)};
  Values before{values_};
  auto commit{merk::CommitOp::create(builder_, loc, mlir::TypeRange{})};
  std::vector<BodyVar> bound;
  llvm::SmallVector<mlir::Type> types;
  bool solves{false};
  {
    mlir::OpBuilder::InsertionGuard guard{builder_};
    builder_.setInsertionPointToStart(&commit.getGenerator().emplaceBlock());
    const Body *outer{committing_};
    committing_ = &goal;
    // Its solution is the commit's, not the body's, so no call in it is a
    // tail call that a word copy would keep.
    std::optional<llvm::SmallVector<BodyVar>> words{std::move(wordOutputs_)};
    wordOutputs_.reset();
    Goal(goal, body);
    wordOutputs_ = std::move(words);
    committing_ = outer;
    if (failed_) {
      values_ = before;
      return;
    }
    if (!Terminated()) {
      solves = true;
      llvm::SmallVector<mlir::Value> values;
      for (BodyVar var : BoundSince(values_, before)) {
        if (mlir::Value value{values_.lookup(var)}) {
          bound.push_back(var);
          values.push_back(value);
          types.push_back(value.getType());
        }
      }
      merk::SolutionOp::create(builder_, loc, values);
    }
  }
  // The operation again, with the results.
  builder_.setInsertionPoint(commit);
  auto joined{merk::CommitOp::create(builder_, loc, types)};
  joined.getGenerator().takeBody(commit.getGenerator());
  commit.erase();
  builder_.setInsertionPointAfter(joined);
  values_ = before;
  for (auto [var, result] : llvm::zip_equal(bound, joined.getResults())) {
    values_[var] = result;
  }
  if (!solves) {
    merk::FailOp::create(builder_, loc);
  }
}

void MerkGenerator::DisjunctionFrom(
    const Body &goal, std::size_t next, const ProcedureBody &body) {
  if (next + 1 == goal.goals.size()) {
    Goal(*goal.goals[next], body);
    return;
  }
  Ite(goal.tokens, [&] { Goal(*goal.goals[next], body); }, nullptr,
      [&] { DisjunctionFrom(goal, next + 1, body); }, body);
}

// if condition then `then` (or true) else otherwise.
void MerkGenerator::Ite(semantics::TokenRange tokens,
    llvm::function_ref<void()> condition, const Body *thenPart,
    llvm::function_ref<void()> otherwise, const ProcedureBody &body,
    const Body *otherwiseGoal) {
  mlir::Location loc{Location(tokens)};
  Values before{values_};
  merk::EachOp each;
  if ((thenPart && Nondeterministic(*thenPart, body)) ||
      (otherwiseGoal && Nondeterministic(*otherwiseGoal, body))) {
    each = BeginEach(loc);
  }
  auto ite{merk::IteOp::create(builder_, loc, mlir::TypeRange{})};
  llvm::SmallVector<Branch, 2> branches;
  {
    mlir::OpBuilder::InsertionGuard guard{builder_};
    // The condition yields what it binds, to the then part.
    builder_.setInsertionPointToStart(&ite.getCondition().emplaceBlock());
    condition();
    std::vector<BodyVar> bound{BoundSince(values_, before)};
    llvm::SmallVector<mlir::Value> yielded;
    for (BodyVar var : bound) {
      yielded.push_back(values_.lookup(var));
    }
    merk::YieldOp conditionYield;
    bool conditionFails{Terminated()};
    if (!conditionFails) {
      conditionYield = merk::YieldOp::create(builder_, loc, yielded);
    }
    mlir::Block &then{ite.getThenRegion().emplaceBlock()};
    values_ = before;
    for (auto [var, value] : llvm::zip(bound, yielded)) {
      values_[var] = then.addArgument(value.getType(), loc);
    }
    builder_.setInsertionPointToStart(&then);
    if (conditionFails) {
      // A condition that cannot succeed, such as a unification mode
      // analysis replaced with fail: the then part is never reached, so
      // only the else part binds the variables after the if-then-else.
      merk::FailOp::create(builder_, loc);
    } else if (thenPart) {
      Goal(*thenPart, body);
    }
    branches.push_back({builder_.getInsertionBlock(), values_, Terminated()});
    mlir::Block &otherwiseBlock{ite.getElseRegion().emplaceBlock()};
    values_ = before;
    builder_.setInsertionPointToStart(&otherwiseBlock);
    otherwise();
    branches.push_back({builder_.getInsertionBlock(), values_, Terminated()});
    if (failed_) {
      values_ = before;
      return;
    }
    Join(ite, branches, before, loc, each);
    // The condition passes the then part only the values it uses: those
    // the condition binds for itself are local to it.
    for (unsigned i = then.getNumArguments(); i-- > 0;) {
      if (then.getArgument(i).use_empty()) {
        then.eraseArgument(i);
        if (conditionYield) {
          conditionYield.getValuesMutable().erase(i);
        }
      }
    }
  }
  if (each) {
    builder_.setInsertionPointToStart(&each.getContinuation().front());
  }
}

// A trace goal (reference manual, "Trace goals") is left out unless its
// compile_time condition holds, and is run only if its run_time condition
// holds. It binds nothing outside it, and its io state is of a type with
// no values, so it has nothing to pass. One that has more than one
// solution is in merk.commit (NeedsCommit).
void MerkGenerator::Trace(const Body &goal, const ProcedureBody &body) {
  const parser::TraceCondition *runTime{nullptr};
  for (const parser::TraceParam *param : goal.trace->params) {
    if (param->kind == parser::TraceParamKind::CompileTime &&
        !CompileTimeHolds(*param->condition)) {
      return;
    }
    if (param->kind == parser::TraceParamKind::RunTime) {
      runTime = param->condition;
    }
  }
  if (!runTime) {
    Goal(*goal.goals.front(), body);
    return;
  }
  mlir::Location loc{Location(goal.tokens)};
  Ite(
      goal.tokens,
      [&] { merk::GuardOp::create(builder_, loc, RunTimeHolds(*runTime, loc)); },
      goal.goals.front(), [] {}, body);
}

// flag(Name) holds with --trace-flag=Name. There is no execution tracing,
// so neither tracelevel holds, and of the grade tests only c and mlds do,
// as in mmc's default grade, hlc.gc: the foreign language is C, and the
// code is high level. The others are of what Merkurius does not have,
// though trail will hold once there is a trail.
bool MerkGenerator::CompileTimeHolds(const parser::TraceCondition &condition) {
  using Kind = parser::TraceConditionKind;
  switch (condition.kind) {
  case Kind::And:
    return llvm::all_of(condition.operands,
        [&](const parser::TraceCondition *c) { return CompileTimeHolds(*c); });
  case Kind::Or:
    return llvm::any_of(condition.operands,
        [&](const parser::TraceCondition *c) { return CompileTimeHolds(*c); });
  case Kind::Not:
    return !CompileTimeHolds(*condition.operands.front());
  case Kind::Flag:
    return llvm::is_contained(traceFlags_, *condition.value);
  case Kind::TraceLevel:
    return false;
  case Kind::Grade:
    return *condition.value == "c" || *condition.value == "mlds";
  case Kind::Env:
    break;
  }
  return false;
}

// env(Name) holds if the environment variable existed when the program
// started, which the runtime saved (runtime/trace.cpp).
mlir::Value MerkGenerator::RunTimeHolds(
    const parser::TraceCondition &condition, mlir::Location loc) {
  using Kind = parser::TraceConditionKind;
  if (condition.kind == Kind::And || condition.kind == Kind::Or) {
    // Each operand in turn: they have no effects to order, but the IR
    // should not depend on the C++ compiler's order of arguments.
    mlir::Value left{RunTimeHolds(*condition.operands[0], loc)};
    mlir::Value right{RunTimeHolds(*condition.operands[1], loc)};
    if (condition.kind == Kind::And) {
      return mlir::arith::AndIOp::create(builder_, loc, left, right);
    }
    return mlir::arith::OrIOp::create(builder_, loc, left, right);
  }
  switch (condition.kind) {
  case Kind::Not:
    return mlir::arith::XOrIOp::create(builder_, loc,
        RunTimeHolds(*condition.operands.front(), loc),
        mlir::arith::ConstantIntOp::create(
            builder_, loc, builder_.getI1Type(), 1));
  default:
    break;
  }
  auto string{merk::StringType::get(&context_)};
  mlir::Value name{merk::ConstantOp::create(builder_, loc, string,
      mlir::StringAttr::get(*condition.value, string))};
  return merk::ForeignCallOp::create(builder_, loc,
      mlir::TypeRange{builder_.getI1Type()}, "merk_env_at_start",
      mlir::ValueRange{name}, /*can_fail=*/true)
      .getOutputs()
      .front();
}

// The constant of a switch's case: an integer literal, or a character's
// code point.
static std::optional<llvm::APInt> CaseConstant(const semantics::InstFunctor &functor,
    const semantics::Type &type, unsigned bits) {
  if (functor.arity != 0) {
    return std::nullopt;
  }
  if (functor.literal) {
    return IntegerValue(functor.name, bits);
  }
  if (type.kind == semantics::TypeKind::Named &&
      type.constructor->name == "character") {
    if (std::optional<llvm::UTF32> codePoint{OnlyCodePoint(functor.name)}) {
      return llvm::APInt{bits, *codePoint};
    }
  }
  return std::nullopt;
}

// The constant of a case of a switch on a string or a float, or null for
// one that is not a literal of the type.
static mlir::TypedAttr TestedConstant(
    const semantics::InstFunctor &functor, mlir::Type type) {
  if (!functor.literal || functor.arity != 0) {
    return {};
  }
  if (mlir::isa<merk::StringType>(type)) {
    if (!functor.name.starts_with("\"")) {
      return {};
    }
    return mlir::StringAttr::get(parser::UnquoteSpelling(functor.name), type);
  }
  auto f{mlir::dyn_cast<mlir::FloatType>(type)};
  if (!f) {
    return {};
  }
  std::string digits{functor.name};
  llvm::erase(digits, '_');
  llvm::APFloat v{f.getFloatSemantics()};
  auto status{v.convertFromString(digits, llvm::APFloat::rmNearestTiesToEven)};
  if (!status) {
    llvm::consumeError(status.takeError());
    return {};
  }
  return mlir::FloatAttr::get(f, v);
}

mlir::Value MerkGenerator::IsOneOf(mlir::Value value,
    llvm::ArrayRef<mlir::TypedAttr> constants, mlir::Location loc) {
  mlir::Value test;
  for (mlir::TypedAttr attr : constants) {
    mlir::Value constant{
        merk::ConstantOp::create(builder_, loc, attr.getType(), attr)};
    mlir::Value equal;
    if (mlir::isa<merk::StringType>(value.getType())) {
      equal = merk::StringEqualOp::create(
          builder_, loc, builder_.getI1Type(), value, constant);
    } else {
      // As ==, so that 0.0 is -0.0's case, and a NaN is in none.
      equal = mlir::arith::CmpFOp::create(builder_, loc,
          mlir::arith::CmpFPredicate::OEQ, value, constant);
    }
    test = test ? mlir::arith::OrIOp::create(builder_, loc, test, equal)
                : equal;
  }
  return test;
}

// A string or a float has too many values for a jump table, so its cases
// are if-then-elses, each testing the value against the case's constants;
// a value that no case has fails, unless the switch is complete, when the
// last case is what is left.
void MerkGenerator::TestCases(const Body &goal, std::size_t next,
    mlir::Value value,
    llvm::ArrayRef<llvm::SmallVector<mlir::TypedAttr>> constants,
    const ProcedureBody &body) {
  mlir::Location loc{Location(goal.tokens)};
  if (next == goal.cases.size()) {
    merk::FailOp::create(builder_, loc);
    return;
  }
  const semantics::SwitchCase &c{goal.cases[next]};
  if (next + 1 == goal.cases.size() && !goal.canFail) {
    Goal(*c.goal, body);
    return;
  }
  Ite(
      goal.tokens,
      [&] {
        merk::GuardOp::create(
            builder_, loc, IsOneOf(value, constants[next], loc));
      },
      c.goal,
      [&] { TestCases(goal, next + 1, value, constants, body); }, body);
}

void MerkGenerator::Switch(const Body &goal, const ProcedureBody &body) {
  mlir::Location loc{Location(goal.tokens)};
  const semantics::Type *type{body.vars[goal.switchVar].type};
  mlir::Value value{ValueOf(goal.switchVar)};
  auto integer{value ? mlir::dyn_cast<mlir::IntegerType>(value.getType())
                     : mlir::IntegerType{}};
  bool term{value && mlir::isa<merk::TermType>(value.getType())};
  if (value && mlir::isa<merk::StringType, mlir::FloatType>(value.getType())) {
    llvm::SmallVector<llvm::SmallVector<mlir::TypedAttr>> constants;
    for (const semantics::SwitchCase &c : goal.cases) {
      llvm::SmallVector<mlir::TypedAttr> &own{constants.emplace_back()};
      std::vector<semantics::InstFunctor> functors{c.functor};
      functors.insert(functors.end(), c.others.begin(), c.others.end());
      for (const semantics::InstFunctor &functor : functors) {
        mlir::TypedAttr constant{TestedConstant(functor, value.getType())};
        if (!constant) {
          NotYet(Where(goal.tokens),
              "switches on values of type " + semantics::TypeToString(*type) +
                  " with a case that is not a literal");
          return;
        }
        own.push_back(constant);
      }
    }
    TestCases(goal, 0, value, constants, body);
    return;
  }
  if (!integer && !term) {
    NotYet(Where(goal.tokens),
        "switches on values of type " + semantics::TypeToString(*type));
    return;
  }
  llvm::SmallVector<mlir::Attribute> cases;
  for (const semantics::SwitchCase &c : goal.cases) {
    llvm::SmallVector<mlir::Attribute> constants;
    std::vector<semantics::InstFunctor> functors{c.functor};
    functors.insert(functors.end(), c.others.begin(), c.others.end());
    for (const semantics::InstFunctor &functor : functors) {
      if (term) {
        constants.push_back(
            merk::FunctorAttr::get(&context_, functor.name, functor.arity));
        continue;
      }
      std::optional<llvm::APInt> v{
          CaseConstant(functor, *type, integer.getWidth())};
      if (!v) {
        NotYet(Where(goal.tokens),
            "switches on values of type " + semantics::TypeToString(*type));
        return;
      }
      constants.push_back(builder_.getIntegerAttr(integer, *v));
    }
    cases.push_back(builder_.getArrayAttr(constants));
  }
  Values before{values_};
  merk::EachOp each;
  if (llvm::any_of(goal.cases, [&](const semantics::SwitchCase &c) {
        return Nondeterministic(*c.goal, body);
      })) {
    each = BeginEach(loc);
  }
  auto op{merk::SwitchOp::create(builder_, loc, mlir::TypeRange{}, value,
      builder_.getArrayAttr(cases), goal.canFail)};
  llvm::SmallVector<Branch> branches;
  {
    mlir::OpBuilder::InsertionGuard guard{builder_};
    for (auto [c, region] : llvm::zip(goal.cases, op.getCaseRegions())) {
      mlir::Block &block{region.emplaceBlock()};
      values_ = before;
      builder_.setInsertionPointToStart(&block);
      Goal(*c.goal, body);
      branches.push_back({builder_.getInsertionBlock(), values_, Terminated()});
    }
  }
  if (failed_) {
    values_ = before;
    return;
  }
  Join(op, branches, before, loc, each);
}

void MerkGenerator::Not(const Body &goal, const ProcedureBody &body) {
  mlir::Location loc{Location(goal.tokens)};
  Values before{values_};
  auto op{merk::NotOp::create(builder_, loc)};
  {
    mlir::OpBuilder::InsertionGuard guard{builder_};
    builder_.setInsertionPointToStart(&op.getGoal().emplaceBlock());
    Goal(*goal.goals.front(), body);
    if (!Terminated()) {
      merk::YieldOp::create(builder_, loc, mlir::ValueRange{});
    }
  }
  values_ = before;
}

} // namespace merkurius::codegen::mlirgen
