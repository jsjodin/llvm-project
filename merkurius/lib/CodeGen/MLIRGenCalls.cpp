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

merk::ProcOp MerkGenerator::ProcFor(const Procedure &procedure,
    llvm::ArrayRef<const semantics::Type *> types, parser::CharBlock at) {
  if (merk::ProcOp proc{procs_.lookup(&procedure)}) {
    return proc;
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
  }
  return proc;
}

merk::ProcOp MerkGenerator::NewProc(llvm::StringRef name, parser::CharBlock at,
    llvm::ArrayRef<mlir::Type> hidden, llvm::ArrayRef<semantics::ArgMode> modes,
    llvm::ArrayRef<const semantics::Type *> types,
    merk::Determinism determinism, llvm::ArrayRef<mlir::Type> hiddenOutputs) {
  llvm::SmallVector<mlir::Type> inputs{hidden};
  llvm::SmallVector<mlir::Type> outputs{hiddenOutputs};
  for (auto [mode, type] : llvm::zip(modes, types)) {
    std::optional<Represented> represented{Represent(type)};
    if (!represented) {
      NotYet(at, "values of type " + semantics::TypeToString(*type));
      return nullptr;
    }
    if (!represented->dummy && !IsUnused(mode)) {
      (IsOutput(mode) ? outputs : inputs).push_back(represented->type);
    }
  }
  mlir::OpBuilder::InsertionGuard guard{builder_};
  builder_.setInsertionPointToEnd(module_.getBody());
  auto proc{merk::ProcOp::create(builder_, Location(at), name,
      builder_.getFunctionType(inputs, outputs), determinism)};
  proc.setPrivate();
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
    llvm::SmallVector<std::pair<BodyVar, mlir::Value>> bound;
    for (auto [argument, mode] : llvm::zip(arguments, callee.modes)) {
      if (IsOutput(mode) && ValueOf(argument) &&
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
    return;
  }
  Called(goal, callee, arguments, body);
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
  const ProcedureBody *copyable{nullptr};
  if (wordOutputs_ && !isMethod && hiddenOutputs.empty() &&
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
  if (passed) {
    for (const semantics::TypeInfo *info : passed->typeInfos) {
      if (info->kind == semantics::TypeInfoKind::Output && info->goal == &goal) {
        gives.push_back(info);
      } else {
        PassTypeInfo(*info, inputs, loc, at);
      }
    }
    for (auto [i, given] : llvm::enumerate(passed->dictionaries)) {
      if (given->kind == semantics::DictionaryKind::Output &&
          given->goal == &goal) {
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
  for (auto [argument, mode, type] : llvm::zip(arguments, callee.modes, types)) {
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
    } else if (dummy) {
      inputs.push_back(merk::ToWordOp::create(builder_, loc,
          merk::WordType::get(&context_),
          IntegerConstant(builder_.getI64Type(), 0, loc)));
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
  for (auto [node, result] : llvm::zip(gives, results)) {
    Give(node, result);
  }
  for (auto [var, result] :
      llvm::zip(outputs, results.drop_front(gives.size()))) {
    if (!var) {
      continue;
    }
    values_[*var] =
        Coerce(result, Represent(body.vars[*var].type)->type, loc);
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
  std::optional<llvm::SmallVector<BodyVar>> savedWords{std::move(wordOutputs_)};
  wordOutputs_.reset();
  mlir::Value closure{block->getArgument(0)};
  unsigned captured{0};
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
  bool handled{represented && Builtin(call, callee, vars, own)};
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
  mlir::Location loc{Location(goal.tokens)};
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
  // What it captures, each in order: the body's type_infos, its
  // typeclass_infos, what goals gave, and the variables bound before it
  // that it uses.
  llvm::DenseSet<BodyVar> used;
  CollectVars(*goal.lambda, used);
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
            (used.contains(var) && !llvm::is_contained(goal.lambdaHead, var)))) {
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
  llvm::SmallVector<BodyVar> outputs;
  llvm::SmallVector<ClosureArgument> arguments;
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
        inputs.push_back(represented->type);
      }
    }
  }
  auto word{merk::WordType::get(&context_)};
  // A word copy's body has the same lambda expressions as the procedure's,
  // which capture values of the same types: one procedure serves both.
  merk::ProcOp proc{lambdaProcs_.lookup(&goal)};
  if (!proc ||
      proc.getFunctionType().getInputs() != llvm::ArrayRef<mlir::Type>{inputs}) {
    {
      mlir::OpBuilder::InsertionGuard guard{builder_};
      builder_.setInsertionPointToEnd(module_.getBody());
      proc = merk::ProcOp::create(builder_, loc,
          procName_ + "$lambda" + std::to_string(lambdas_++),
          builder_.getFunctionType(
              inputs, llvm::SmallVector<mlir::Type>(outputs.size(), word)),
          *determinism);
      proc.setPrivate();
    }
    // Its body, with the captured values and its inputs as the procedure's.
    Values savedValues;
    savedValues.swap(values_);
    llvm::DenseMap<unsigned, mlir::Value> savedInfos;
    savedInfos.swap(typeInfos_);
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
      for (auto [var, argument] : llvm::zip(goal.lambdaHead, arguments)) {
        if (!argument.output && !argument.dummy) {
          values_[var] = entry->getArgument(input++);
        }
      }
      wordOutputs_ = outputs;
      procName_ = proc.getSymName().str();
      const Body *outerLambda{committedLambda_};
      committedLambda_ = goal.modes->determinism == parser::Determinism::CcMulti ||
              goal.modes->determinism == parser::Determinism::CcNondet
          ? goal.lambda
          : nullptr;
      Goal(*goal.lambda, body);
      committedLambda_ = outerLambda;
      if (!failed_ && !Terminated()) {
        llvm::SmallVector<mlir::Value> results{WordResults(outputs, body, loc)};
        if (llvm::is_contained(results, mlir::Value{})) {
          NotYet(at, "this lambda expression's outputs");
        } else {
          merk::SucceedOp::create(builder_, loc, results);
          EraseUnwordedOutputs(proc.getBody());
        }
      }
    }
    values_.swap(savedValues);
    typeInfos_.swap(savedInfos);
    dictionaries_.swap(savedDictionaries);
    wordOutputs_ = std::move(savedWords);
    procName_ = savedName;
    if (failed_) {
      proc.getBody().dropAllReferences();
      proc.getBody().getBlocks().clear();
      return;
    }
    lambdaProcs_[&goal] = proc;
  }
  merk::ProcOp entry{ClosureEntry(
      proc, static_cast<unsigned>(capturedValues.size()), arguments, at)};
  if (!entry) {
    return;
  }
  CapturePlaces places;
  llvm::SmallVector<mlir::Attribute> captures;
  for (auto [i, var] : llvm::enumerate(infoKeys)) {
    places[var] = {static_cast<unsigned>(i), 0};
    captures.push_back(builder_.getStringAttr("type_info"));
  }
  captures.append(
      dictionaryKeys.size(), builder_.getStringAttr("typeclass_info"));
  for (BodyVar var : captured) {
    if (var < kFirstGivenVar) {
      mlir::Attribute capture{CaptureTypeAttr(*body.vars[var].type, places, at)};
      if (!capture) {
        return;
      }
      captures.push_back(capture);
    } else {
      mlir::Type given{values_.lookup(var).getType()};
      captures.push_back(builder_.getStringAttr(
          mlir::isa<merk::TypeInfoType>(given)            ? "type_info"
              : mlir::isa<merk::TypeclassInfoType>(given) ? "typeclass_info"
                                                          : "unknown"));
    }
  }
  llvm::SmallVector<mlir::Value> words;
  for (mlir::Value value : capturedValues) {
    words.push_back(Coerce(value, word, loc));
  }
  values_[goal.left] = merk::ClosureOp::create(builder_, loc,
      merk::ClosureType::get(&context_),
      mlir::FlatSymbolRefAttr::get(&context_, entry.getSymName()), words,
      CapturesAttr(captures));
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
  std::optional<merk::Determinism> determinism{
      ClosureDeterminism(goal.modes->determinism)};
  if (!determinism) {
    NotYet(at,
        std::string{semantics::DeterminismName(goal.modes->determinism)} +
            " closures");
    return;
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
