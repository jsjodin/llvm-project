//===-- lib/CodeGen/MLIRGenUnify.cpp - Unification ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unifications of terms, and the unify and compare procedures of types and
// arrays.
//
//===----------------------------------------------------------------------===//

#include "MLIRGenImpl.h"

namespace merkurius::codegen::mlirgen {

void MerkGenerator::TermUnify(const Body &goal, const ProcedureBody &body) {
  mlir::Location loc{Location(goal.tokens)};
  parser::CharBlock at{Where(goal.tokens)};
  const semantics::Type *type{body.vars[goal.left].type};
  auto term{mlir::cast<merk::TermType>(Represent(type)->type)};
  unsigned arity{static_cast<unsigned>(goal.arguments.size())};
  auto functor{merk::FunctorAttr::get(&context_, goal.functor.name, arity)};
  // An existentially typed constructor's cell has its hidden fields first,
  // and its arguments' types are its own, which the values' are coerced to.
  HiddenFields hidden;
  std::optional<llvm::SmallVector<mlir::Type>> cell;
  if (const semantics::ConstructorTypes *constructor{
          ConstructorOf(type, goal.functor.name, arity)}) {
    cell = CellFields(type, *constructor, hidden);
    if (!cell) {
      NotYet(at, "this unification");
      return;
    }
  }
  // What polymorphism analysis has a construction pass, and a deconstruction
  // give, by the cell's hidden fields.
  const semantics::PassedInfos *passed{nullptr};
  if (!hidden.types.empty() && current_) {
    if (auto it{current_->goals.find(&goal)}; it != current_->goals.end()) {
      passed = &it->second;
    }
  }
  if (passed &&
      passed->typeInfos.size() + passed->dictionaries.size() !=
          hidden.types.size()) {
    NotYet(at, "this unification's type_infos");
    return;
  }
  if (goal.unify == UnifyKind::Construct) {
    llvm::SmallVector<mlir::Value> arguments;
    if (!hidden.types.empty()) {
      if (!passed) {
        NotYet(at, "this unification's type_infos");
        return;
      }
      for (const semantics::TypeInfo *info : passed->typeInfos) {
        arguments.push_back(TypeInfoValue(*info, loc, at));
      }
      for (const semantics::Dictionary *dictionary : passed->dictionaries) {
        arguments.push_back(DictionaryValue(*dictionary, loc, at));
      }
      if (llvm::is_contained(arguments, mlir::Value{})) {
        failed_ = true;
        return;
      }
    }
    for (BodyVar argument : goal.arguments) {
      mlir::Value value{ValueOf(argument)};
      if (!value) {
        // A term with free arguments. As in mmc it comes before the call of
        // an implied mode, which binds the free parts; it needs no cell.
        std::optional<Represented> represented{
            Represent(body.vars[argument].type)};
        if (!hidden.types.empty() || !represented || represented->dummy) {
          NotYet(at, "this unification");
          return;
        }
        partialTerms_[{&body, goal.left}] = &goal;
        return;
      }
      if (!hidden.types.empty()) {
        value = Coerce(value, (*cell)[arguments.size()], loc);
      }
      // An ordinary term holds a closed(T) as T's word, so one that may be
      // unbound here would need the term to hold its open word.
      if (IsClosedType(body.vars[argument].type)) {
        for (const auto &[var, inst] : goal.entry) {
          if (var == argument && inst &&
              modes_.ops().HasAny(inst, body.vars[argument].type)) {
            NotYet(at,
                "terms of type " +
                    semantics::TypeToString(*body.vars[goal.left].type) +
                    " holding closed(T) values that may be unbound");
            return;
          }
        }
        value = ClosedAs(value, false, body.vars[argument].type, loc, at);
      }
      arguments.push_back(value);
    }
    values_[goal.left] =
        merk::ConstructOp::create(builder_, loc, term, functor, arguments);
    return;
  }
  mlir::Value value{ValueOf(goal.left)};
  if (!goal.switchTest &&
      (goal.canFail || goal.unify == UnifyKind::Test)) {
    merk::GuardOp::create(builder_, loc,
        merk::HasFunctorOp::create(builder_, loc, value, functor));
  }
  if (goal.arguments.empty()) {
    return;
  }
  // The arguments are new variables: mmc's form tests any that was bound
  // with a unification after it.
  llvm::SmallVector<mlir::Type> fields{hidden.types};
  for (BodyVar argument : goal.arguments) {
    std::optional<Represented> represented{
        Represent(body.vars[argument].type)};
    if (goal.unify == UnifyKind::Test || !represented || represented->dummy ||
        values_.count(argument)) {
      NotYet(at, "this unification");
      return;
    }
    fields.push_back(represented->type);
  }
  if (!hidden.types.empty()) {
    fields = *cell;
  }
  auto deconstruct{
      merk::DeconstructOp::create(builder_, loc, fields, value, functor)};
  mlir::ValueRange results{deconstruct.getFields()};
  // The type_infos and typeclass_infos it gives, which a deconstruction whose
  // arguments nothing uses gives none of.
  if (passed) {
    unsigned field{0};
    for (const semantics::TypeInfo *info : passed->typeInfos) {
      NoteGivenType(*info);
      GiveTypeInfo(*info, results[field], loc);
      Give(info, results[field++]);
    }
    for (const semantics::Dictionary *dictionary : passed->dictionaries) {
      NoteGivenTypes(*dictionary);
      GiveDictionary(*dictionary, results[field], loc);
      Give(dictionary, results[field++]);
    }
  }
  for (auto [argument, field] :
      llvm::zip(goal.arguments, results.drop_front(hidden.types.size()))) {
    values_[argument] = hidden.types.empty()
        ? field
        : Coerce(field, Represent(body.vars[argument].type)->type, loc);
  }
}

void MerkGenerator::TestPartial(const Body &goal, mlir::Value value,
    const Body &partial, const ProcedureBody &body) {
  parser::CharBlock at{Where(goal.tokens)};
  mlir::Location loc{Location(goal.tokens)};
  const semantics::Type *type{body.vars[partial.left].type};
  auto functor{merk::FunctorAttr::get(&context_, partial.functor.name,
      static_cast<unsigned>(partial.arguments.size()))};
  const std::vector<semantics::ConstructorTypes> *constructors{
      modes_.ops().ConstructorsOf(type)};
  if (!constructors || constructors->size() > 1) {
    merk::GuardOp::create(builder_, loc,
        merk::HasFunctorOp::create(builder_, loc, value, functor));
  }
  llvm::SmallVector<mlir::Type> fields;
  for (BodyVar argument : partial.arguments) {
    std::optional<Represented> represented{
        Represent(body.vars[argument].type)};
    if (!represented || represented->dummy) {
      NotYet(at, "this unification");
      return;
    }
    fields.push_back(represented->type);
  }
  auto deconstruct{
      merk::DeconstructOp::create(builder_, loc, fields, value, functor)};
  for (auto [argument, field] :
      llvm::zip(partial.arguments, deconstruct.getFields())) {
    if (failed_ || Terminated()) {
      return;
    }
    if (mlir::Value bound{ValueOf(argument)}) {
      Test(goal, bound, field, body.vars[argument].type);
    } else if (const Body *inner{PartialTerm(body, argument)}) {
      TestPartial(goal, field, *inner, body);
    }
  }
}

bool MerkGenerator::Comparable(mlir::Type type, llvm::StringSet<> &seen) {
  if (mlir::isa<mlir::IntegerType, mlir::FloatType, merk::StringType,
          merk::WordType, merk::ClosureType>(type)) {
    return true;
  }
  if (auto array{mlir::dyn_cast<merk::ArrayType>(type)}) {
    return Comparable(array.getElement(), seen);
  }
  auto term{mlir::dyn_cast<merk::TermType>(type)};
  if (!term) {
    return false;
  }
  if (!seen.insert(term.getName()).second || IsAbstract(term)) {
    return true;
  }
  const semantics::Type *semantic{terms_.lookup(term.getName())};
  for (const semantics::ConstructorTypes &constructor :
      *modes_.ops().ConstructorsOf(semantic)) {
    for (const semantics::Type *argument : constructor.arguments) {
      if (!Comparable(Represent(argument)->type, seen)) {
        return false;
      }
    }
  }
  return true;
}

bool MerkGenerator::IsAbstract(mlir::Type type) {
  auto term{mlir::dyn_cast<merk::TermType>(type)};
  if (!term) {
    return false;
  }
  const semantics::Type *semantic{terms_.lookup(term.getName())};
  return semantic && !modes_.ops().ConstructorsOf(semantic);
}

merk::ProcOp MerkGenerator::UnifyProc(merk::TermType term) {
  if (merk::ProcOp proc{unifyProcs_.lookup(term.getName())}) {
    return proc;
  }
  mlir::Location loc{builder_.getUnknownLoc()};
  const semantics::Type *type{terms_.lookup(term.getName())};
  const std::vector<semantics::ConstructorTypes> &constructors{
      *modes_.ops().ConstructorsOf(type)};
  mlir::OpBuilder::InsertionGuard guard{builder_};
  builder_.setInsertionPointToEnd(module_.getBody());
  // It takes a type_info for each type variable of its type, first.
  llvm::SmallVector<unsigned> variables{VariablesOf(*type)};
  llvm::SmallVector<mlir::Type> inputs(
      variables.size(), merk::TypeInfoType::get(&context_));
  inputs.append({term, term});
  auto proc{merk::ProcOp::create(builder_, loc,
      ("unify." + term.getName()).str(), builder_.getFunctionType(inputs, {}),
      merk::Determinism::Semidet)};
  proc.setPrivate();
  // Recorded first, for the calls of it on its terms' arguments.
  unifyProcs_[term.getName()] = proc;
  mlir::Block *entry{proc.addEntryBlock()};
  builder_.setInsertionPointToStart(entry);
  auto saved{std::move(typeInfos_)};
  typeInfos_.clear();
  for (auto [i, var] : llvm::enumerate(variables)) {
    typeInfos_[var] = entry->getArgument(i);
  }
  mlir::Value x{entry->getArgument(variables.size())};
  mlir::Value y{entry->getArgument(variables.size() + 1)};
  NoteRoot(x, type);
  NoteRoot(y, type);
  // A user-defined equality: its predicate's; or else the comparison's (=).
  if (const semantics::UserEquality *user{UserEqualityOf(type)}) {
    mlir::Value order;
    if (user->equality) {
      CallUserPredicate(*user->equality, false, type, x, y, loc, order);
    } else if (user->comparison &&
        CallUserPredicate(*user->comparison, true, type, x, y, loc, order)) {
      merk::GuardOp::create(builder_, loc, IsComparison(order, "=", loc));
    }
    merk::SucceedOp::create(builder_, loc, mlir::ValueRange{});
    typeInfos_ = std::move(saved);
    return proc;
  }
  llvm::SmallVector<mlir::Attribute> cases;
  for (const semantics::ConstructorTypes &constructor : constructors) {
    cases.push_back(builder_.getArrayAttr({merk::FunctorAttr::get(&context_,
        constructor.functor.name, constructor.functor.arity)}));
  }
  auto op{merk::SwitchOp::create(builder_, loc, mlir::TypeRange{}, x,
      builder_.getArrayAttr(cases), /*canFail=*/false)};
  for (auto [constructor, region, c] :
      llvm::zip(constructors, op.getCaseRegions(), cases)) {
    builder_.setInsertionPointToStart(&region.emplaceBlock());
    auto functor{mlir::cast<merk::FunctorAttr>(
        mlir::cast<mlir::ArrayAttr>(c)[0])};
    merk::GuardOp::create(builder_, loc,
        merk::HasFunctorOp::create(builder_, loc, y, functor));
    if (!constructor.arguments.empty()) {
      HiddenFields hidden;
      llvm::SmallVector<mlir::Type> fields{
          *CellFields(type, constructor, hidden)};
      auto xs{merk::DeconstructOp::create(builder_, loc, fields, x, functor)};
      auto ys{merk::DeconstructOp::create(builder_, loc, fields, y, functor)};
      // As mmc's, terms of an existentially typed constructor are equal only
      // if their types are the same.
      for (auto [k, place] : llvm::enumerate(hidden.places)) {
        mlir::Value a{HiddenTypeInfo(xs.getFields(), place, loc)};
        mlir::Value b{HiddenTypeInfo(ys.getFields(), place, loc)};
        merk::GuardOp::create(builder_, loc,
            mlir::arith::CmpIOp::create(builder_, loc,
                mlir::arith::CmpIPredicate::eq,
                merk::CompareTypeInfoOp::create(
                    builder_, loc, builder_.getI32Type(), a, b),
                IntegerConstant(builder_.getI32Type(), 0, loc)));
        typeInfos_[ExistentialBase(*type) + static_cast<unsigned>(k)] = a;
      }
      std::size_t first{hidden.types.size()};
      for (auto [a, b, argument] :
          llvm::zip(xs.getFields().drop_front(first),
              ys.getFields().drop_front(first),
              ArgumentsOf(type, constructor))) {
        NoteRoot(a, argument);
        NoteRoot(b, argument);
        Equal(a, b, argument, loc);
      }
    }
    merk::YieldOp::create(builder_, loc, mlir::ValueRange{});
  }
  builder_.setInsertionPointAfter(op);
  merk::SucceedOp::create(builder_, loc, mlir::ValueRange{});
  typeInfos_ = std::move(saved);
  return proc;
}

const semantics::UserEquality *MerkGenerator::UserEqualityOf(
    const semantics::Type *type) {
  if (!type || type->kind != semantics::TypeKind::Named) {
    return nullptr;
  }
  return types_.UserEqualityOf(*type->constructor);
}

// Binds the type variables of `pattern` to make it `type`, as `bindings`
// numbers them; whether it can.
static bool MatchType(const semantics::Type *pattern,
    const semantics::Type *type,
    llvm::SmallVectorImpl<const semantics::Type *> &bindings) {
  if (pattern->kind == semantics::TypeKind::Variable) {
    if (bindings.size() <= pattern->var) {
      bindings.resize(pattern->var + 1, nullptr);
    }
    const semantics::Type *&bound{bindings[pattern->var]};
    if (!bound) {
      bound = type;
    }
    return bound == type;
  }
  if (type->kind != pattern->kind ||
      type->constructor != pattern->constructor ||
      type->arguments.size() != pattern->arguments.size() ||
      !type->result != !pattern->result) {
    return false;
  }
  for (auto [p, t] : llvm::zip(pattern->arguments, type->arguments)) {
    if (!MatchType(p, t, bindings)) {
      return false;
    }
  }
  return !pattern->result || MatchType(pattern->result, type->result, bindings);
}

bool MerkGenerator::CallUserPredicate(const semantics::Symbol &predicate,
    bool comparison, const semantics::Type *type, mlir::Value x, mlir::Value y,
    mlir::Location loc, mlir::Value &result) {
  parser::CharBlock at{predicate.source};
  const semantics::Signature *signature{types_.SignatureOf(predicate)};
  if (!signature || !signature->constraints.empty() ||
      signature->existentials != 0) {
    NotYet(at, "user-defined equality or comparison predicates with type "
               "class constraints or existential types");
    failed_ = true;
    return false;
  }
  // Its mode for this call: of inputs, after the comparison's output.
  const Procedure *procedure{nullptr};
  for (const Procedure *candidate : modes_.ProceduresOf(predicate)) {
    bool fits{candidate->modes.size() == (comparison ? 3u : 2u)};
    for (auto [i, mode] : llvm::enumerate(candidate->modes)) {
      bool output{comparison && i == 0};
      fits = fits &&
          (output ? IsOutput(mode)
                  : !modes_.ops().IsFree(mode.initial) &&
                      !modes_.ops().IsFree(mode.final));
    }
    if (fits) {
      procedure = candidate;
      break;
    }
  }
  if (!procedure) {
    NotYet(at, comparison
            ? "user-defined comparison predicates without a mode that takes "
              "the values in and gives the result out"
            : "user-defined equality predicates without a mode that takes "
              "both values in");
    failed_ = true;
    return false;
  }
  // The type_infos of its type variables, which the type's instance binds.
  llvm::SmallVector<const semantics::Type *> bindings;
  std::size_t first{comparison ? 1u : 0u};
  for (std::size_t i{first}; i < signature->arguments.size(); ++i) {
    if (!MatchType(signature->arguments[i], type, bindings)) {
      NotYet(at, "this user-defined equality or comparison predicate's types");
      failed_ = true;
      return false;
    }
  }
  bindings.resize(signature->typeVars, nullptr);
  merk::ProcOp proc{ProcFor(*procedure, signature->arguments, at)};
  if (!proc) {
    failed_ = true;
    return false;
  }
  llvm::SmallVector<mlir::Value> inputs;
  for (const semantics::Type *bound : bindings) {
    mlir::Value info{bound ? TypeInfoOfType(bound, loc, at)
                           : TypeInfoOfType(types_.types().Variable(0), loc, at,
                                 /*freeIsVoid=*/true)};
    if (!info) {
      failed_ = true;
      return false;
    }
    inputs.push_back(info);
  }
  inputs.append({x, y});
  mlir::FunctionType function{proc.getFunctionType()};
  if (function.getNumInputs() != inputs.size()) {
    NotYet(at, "this user-defined equality or comparison predicate's "
               "type_infos");
    failed_ = true;
    return false;
  }
  for (auto [input, declared] : llvm::zip(inputs, function.getInputs())) {
    input = Coerce(input, declared, loc);
  }
  mlir::Type order{*ComparisonType()};
  if (Nondeterministic(*procedure)) {
    // Its call has no outputs but the comparison's, which is det: its first
    // solution, as mmc prunes it.
    auto commit{merk::CommitOp::create(builder_, loc,
        comparison ? mlir::TypeRange{order} : mlir::TypeRange{})};
    mlir::OpBuilder::InsertionGuard guard{builder_};
    builder_.setInsertionPointToStart(&commit.getGenerator().emplaceBlock());
    auto nondet{merk::NondetCallOp::create(
        builder_, loc, proc.getSymName(), inputs)};
    mlir::Block &continuation{nondet.getContinuation().emplaceBlock()};
    for (mlir::Type t : function.getResults()) {
      continuation.addArgument(t, loc);
    }
    builder_.setInsertionPointToStart(&continuation);
    llvm::SmallVector<mlir::Value> solution;
    if (comparison) {
      solution.push_back(Coerce(continuation.getArgument(0), order, loc));
    }
    merk::SolutionOp::create(builder_, loc, solution);
    if (comparison) {
      result = commit.getResult(0);
    }
    return true;
  }
  auto call{merk::CallOp::create(
      builder_, loc, function.getResults(), proc.getSymName(), inputs)};
  if (comparison) {
    if (call.getNumResults() != 1) {
      NotYet(at, "this user-defined comparison predicate's results");
      failed_ = true;
      return false;
    }
    result = Coerce(call.getResult(0), order, loc);
  }
  return true;
}

llvm::SmallVector<unsigned> MerkGenerator::VariablesOf(
    const semantics::Type &type) {
  llvm::SmallVector<unsigned> variables;
  semantics::ForEachVariable(type, [&](unsigned var) {
    if (!llvm::is_contained(variables, var)) {
      variables.push_back(var);
    }
  });
  return variables;
}

mlir::Value MerkGenerator::ArrayElement(mlir::Value array, mlir::Value index,
    mlir::Value elementTypeInfo, mlir::Location loc) {
  return merk::ArrayGetOp::create(builder_, loc,
      mlir::cast<merk::ArrayType>(array.getType()).getElement(), array, index,
      elementTypeInfo);
}

mlir::Value MerkGenerator::ArrayWithElement(mlir::Value array,
    mlir::Value index, mlir::Value element, mlir::Value elementTypeInfo,
    mlir::Location loc) {
  return merk::ArraySetOp::create(
      builder_, loc, array.getType(), array, index, element, elementTypeInfo)
      .getResult();
}

mlir::Value MerkGenerator::ArraySize(mlir::Value array, mlir::Location loc) {
  return merk::ArraySizeOp::create(builder_, loc, builder_.getI64Type(), array);
}

// As an unsigned comparison, a negative index is out of range too.
mlir::Value MerkGenerator::InArrayBounds(
    mlir::Value array, mlir::Value index, mlir::Location loc) {
  return CompareInts(
      mlir::arith::CmpIPredicate::ult, index, ArraySize(array, loc), loc);
}

mlir::Value MerkGenerator::CompareInts(mlir::arith::CmpIPredicate predicate,
    mlir::Value a, mlir::Value b, mlir::Location loc) {
  return mlir::arith::CmpIOp::create(builder_, loc, predicate, a, b);
}

bool MerkGenerator::GiveOutputs(
    llvm::ArrayRef<BodyVar> out, std::initializer_list<mlir::Value> values) {
  for (auto [var, value] : llvm::zip(out, values)) {
    values_[var] = value;
  }
  return true;
}

bool HasShape(llvm::ArrayRef<mlir::Value> in, llvm::ArrayRef<BodyVar> out,
    unsigned inputs, unsigned outputs) {
  return in.size() == inputs && out.size() == outputs;
}

// An array's elements are words whatever its type, so one procedure serves.
// The arrays that the generated array procedures take, for the collector:
// the type_info is of their elements.
void MerkGenerator::NoteArrayArguments(mlir::Block *entry, mlir::Location loc) {
  for (unsigned i : {1u, 2u}) {
    merk::RootTypeOp::create(
        builder_, loc, entry->getArgument(i), entry->getArgument(0));
  }
}

merk::ProcOp MerkGenerator::UnifyArrayProc() {
  if (unifyArray_) {
    return unifyArray_;
  }
  mlir::Location loc{builder_.getUnknownLoc()};
  auto info{merk::TypeInfoType::get(&context_)};
  auto word{merk::WordType::get(&context_)};
  auto array{merk::ArrayType::get(&context_, word)};
  auto i64{builder_.getI64Type()};
  mlir::OpBuilder::InsertionGuard guard{builder_};
  builder_.setInsertionPointToEnd(module_.getBody());
  // From an index to the size: each element's test, then the rest's, as a
  // tail call.
  auto from{merk::ProcOp::create(builder_, loc, "unify.array.from",
      builder_.getFunctionType({info, array, array, i64, i64}, {}),
      merk::Determinism::Semidet)};
  from.setPrivate();
  {
    mlir::Block *entry{from.addEntryBlock()};
    builder_.setInsertionPointToStart(entry);
    NoteArrayArguments(entry, loc);
    mlir::Value i{entry->getArgument(3)}, n{entry->getArgument(4)};
    auto ite{merk::IteOp::create(builder_, loc, mlir::TypeRange{})};
    builder_.setInsertionPointToStart(&ite.getCondition().emplaceBlock());
    merk::GuardOp::create(builder_, loc,
        mlir::arith::CmpIOp::create(
            builder_, loc, mlir::arith::CmpIPredicate::slt, i, n));
    merk::YieldOp::create(builder_, loc, mlir::ValueRange{});
    builder_.setInsertionPointToStart(&ite.getThenRegion().emplaceBlock());
    merk::GuardOp::create(builder_, loc,
        merk::UnifyWordOp::create(builder_, loc, builder_.getI1Type(),
            entry->getArgument(0),
            ArrayElement(entry->getArgument(1), i, entry->getArgument(0), loc),
            ArrayElement(
                entry->getArgument(2), i, entry->getArgument(0), loc)));
    mlir::Value next{mlir::arith::AddIOp::create(
        builder_, loc, i, IntegerConstant(i64, 1, loc))};
    merk::CallOp::create(builder_, loc, mlir::TypeRange{}, from.getSymName(),
        mlir::ValueRange{entry->getArgument(0), entry->getArgument(1),
            entry->getArgument(2), next, n});
    merk::YieldOp::create(builder_, loc, mlir::ValueRange{});
    builder_.setInsertionPointToStart(&ite.getElseRegion().emplaceBlock());
    merk::YieldOp::create(builder_, loc, mlir::ValueRange{});
    builder_.setInsertionPointAfter(ite);
    merk::SucceedOp::create(builder_, loc, mlir::ValueRange{});
  }
  builder_.setInsertionPointToEnd(module_.getBody());
  unifyArray_ = merk::ProcOp::create(builder_, loc, "unify.array",
      builder_.getFunctionType({info, array, array}, {}),
      merk::Determinism::Semidet);
  unifyArray_.setPrivate();
  mlir::Block *entry{unifyArray_.addEntryBlock()};
  builder_.setInsertionPointToStart(entry);
  NoteArrayArguments(entry, loc);
  mlir::Value size{merk::ArraySizeOp::create(
      builder_, loc, i64, entry->getArgument(1))};
  merk::GuardOp::create(builder_, loc,
      mlir::arith::CmpIOp::create(builder_, loc, mlir::arith::CmpIPredicate::eq,
          size,
          merk::ArraySizeOp::create(builder_, loc, i64, entry->getArgument(2))));
  merk::CallOp::create(builder_, loc, mlir::TypeRange{}, from.getSymName(),
      mlir::ValueRange{entry->getArgument(0), entry->getArgument(1),
          entry->getArgument(2), IntegerConstant(i64, 0, loc), size});
  merk::SucceedOp::create(builder_, loc, mlir::ValueRange{});
  return unifyArray_;
}

void MerkGenerator::Equal(mlir::Value x, mlir::Value y,
    const semantics::Type *semantic, mlir::Location loc) {
  mlir::Type type{x.getType()};
  if (IsOpenType(semantic)) {
    OpenEqual(x, y, semantic, loc);
  } else if (mlir::isa<merk::WordType>(type) ||
      semantic->kind == semantics::TypeKind::Variable) {
    mlir::Value info{typeInfos_.lookup(semantic->var)};
    if (!info) {
      NotYet(equalAt_, "a unification of a type variable without a type_info");
      return;
    }
    merk::GuardOp::create(builder_, loc,
        merk::UnifyWordOp::create(
            builder_, loc, builder_.getI1Type(), info, x, y));
  } else if (mlir::isa<merk::ClosureType>(type) || IsAbstract(type)) {
    // Through the descriptor of the higher-order type, whose unification
    // procedure reports the error, as mmc's does, or of the abstract type,
    // which its module defines.
    mlir::Value info{TypeInfoOfType(semantic, loc, equalAt_)};
    if (!info) {
      failed_ = true;
      return;
    }
    auto word{merk::WordType::get(&context_)};
    merk::GuardOp::create(builder_, loc,
        merk::UnifyWordOp::create(builder_, loc, builder_.getI1Type(), info,
            Coerce(x, word, loc), Coerce(y, word, loc)));
  } else if (mlir::isa<merk::StringType>(type)) {
    merk::GuardOp::create(builder_, loc,
        merk::StringEqualOp::create(
            builder_, loc, builder_.getI1Type(), x, y));
  } else if (mlir::isa<merk::ArrayType>(type)) {
    mlir::Value info{
        TypeInfoOfType(semantic->arguments[0], loc, equalAt_)};
    if (!info) {
      failed_ = true;
      return;
    }
    auto words{merk::ArrayType::get(&context_, merk::WordType::get(&context_))};
    merk::CallOp::create(builder_, loc, mlir::TypeRange{},
        UnifyArrayProc().getSymName(),
        mlir::ValueRange{info, Coerce(x, words, loc), Coerce(y, words, loc)});
  } else if (mlir::isa<mlir::IntegerType>(type)) {
    merk::GuardOp::create(builder_, loc,
        mlir::arith::CmpIOp::create(
            builder_, loc, mlir::arith::CmpIPredicate::eq, x, y));
  } else if (mlir::isa<mlir::FloatType>(type)) {
    // As mmc's ==, a NaN is equal to nothing.
    merk::GuardOp::create(builder_, loc,
        mlir::arith::CmpFOp::create(
            builder_, loc, mlir::arith::CmpFPredicate::OEQ, x, y));
  } else {
    llvm::SmallVector<mlir::Value> inputs;
    for (unsigned var : VariablesOf(*semantic)) {
      mlir::Value info{typeInfos_.lookup(var)};
      if (!info) {
        NotYet(equalAt_, "a unification of a type variable without a type_info");
        return;
      }
      inputs.push_back(info);
    }
    inputs.append({x, y});
    merk::CallOp::create(builder_, loc, mlir::TypeRange{},
        UnifyProc(mlir::cast<merk::TermType>(type)).getSymName(), inputs);
  }
}

std::optional<merk::TermType> MerkGenerator::ComparisonType() {
  const semantics::ModuleSymbols *builtin{resolution_.Module("builtin")};
  if (builtin) {
    for (const semantics::Symbol *symbol :
        builtin->Lookup("comparison_result")) {
      if (symbol->kind == semantics::SymbolKind::Type) {
        std::optional<Represented> represented{
            Represent(types_.types().Named(*symbol, {}))};
        return mlir::cast<merk::TermType>(represented->type);
      }
    }
  }
  return std::nullopt;
}

mlir::Value MerkGenerator::ComparisonValue(
    llvm::StringRef functor, mlir::Location loc) {
  return merk::ConstructOp::create(builder_, loc, *ComparisonType(),
      merk::FunctorAttr::get(&context_, functor, 0), mlir::ValueRange{});
}

mlir::Value MerkGenerator::IsComparison(
    mlir::Value result, llvm::StringRef functor, mlir::Location loc) {
  return merk::HasFunctorOp::create(builder_, loc, result,
      merk::FunctorAttr::get(&context_, functor, 0));
}

mlir::Value MerkGenerator::ThreeWay(
    mlir::Value less, mlir::Value greater, mlir::Location loc) {
  return ComparisonIf(less, "<", loc, [&] {
    return ComparisonIf(
        greater, ">", loc, [&] { return ComparisonValue("=", loc); });
  });
}

mlir::Value MerkGenerator::ComparisonIf(mlir::Value condition,
    llvm::StringRef functor, mlir::Location loc,
    llvm::function_ref<mlir::Value()> otherwise) {
  auto ite{merk::IteOp::create(
      builder_, loc, mlir::TypeRange{*ComparisonType()})};
  mlir::OpBuilder::InsertionGuard guard{builder_};
  builder_.setInsertionPointToStart(&ite.getCondition().emplaceBlock());
  merk::GuardOp::create(builder_, loc, condition);
  merk::YieldOp::create(builder_, loc, mlir::ValueRange{});
  builder_.setInsertionPointToStart(&ite.getThenRegion().emplaceBlock());
  merk::YieldOp::create(
      builder_, loc, mlir::ValueRange{ComparisonValue(functor, loc)});
  builder_.setInsertionPointToStart(&ite.getElseRegion().emplaceBlock());
  merk::YieldOp::create(builder_, loc, mlir::ValueRange{otherwise()});
  return ite.getResult(0);
}

mlir::Value MerkGenerator::Compare(mlir::Value x, mlir::Value y,
    const semantics::Type *semantic, bool generic, mlir::Location loc) {
  namespace arith = mlir::arith;
  mlir::Type type{x.getType()};
  mlir::Type result{*ComparisonType()};
  if (IsOpenType(semantic)) {
    return OpenCompare(loc);
  }
  if (mlir::isa<merk::WordType>(type) ||
      semantic->kind == semantics::TypeKind::Variable) {
    mlir::Value info{typeInfos_.lookup(semantic->var)};
    if (!info) {
      NotYet(equalAt_, "a comparison of a type variable without a type_info");
      return nullptr;
    }
    return merk::CompareWordOp::create(builder_, loc, result, info, x, y);
  }
  if (mlir::isa<merk::ClosureType>(type) || IsAbstract(type)) {
    // As a unification of closures, or of an abstract type's terms.
    mlir::Value info{TypeInfoOfType(semantic, loc, equalAt_)};
    if (!info) {
      failed_ = true;
      return nullptr;
    }
    auto word{merk::WordType::get(&context_)};
    return merk::CompareWordOp::create(builder_, loc, result, info,
        Coerce(x, word, loc), Coerce(y, word, loc));
  }
  if (mlir::isa<merk::StringType>(type)) {
    mlir::Value order{merk::StringCompareOp::create(
        builder_, loc, builder_.getI32Type(), x, y)};
    mlir::Value zero{IntegerConstant(builder_.getI32Type(), 0, loc)};
    return ThreeWay(arith::CmpIOp::create(
                        builder_, loc, arith::CmpIPredicate::slt, order, zero),
        arith::CmpIOp::create(
            builder_, loc, arith::CmpIPredicate::sgt, order, zero),
        loc);
  }
  if (mlir::isa<merk::ArrayType>(type)) {
    mlir::Value info{TypeInfoOfType(semantic->arguments[0], loc, equalAt_)};
    if (!info) {
      failed_ = true;
      return nullptr;
    }
    auto words{merk::ArrayType::get(&context_, merk::WordType::get(&context_))};
    return merk::CallOp::create(builder_, loc, mlir::TypeRange{result},
        CompareArrayProc().getSymName(),
        mlir::ValueRange{info, Coerce(x, words, loc), Coerce(y, words, loc)})
        .getResult(0);
  }
  if (mlir::isa<mlir::IntegerType>(type)) {
    // A character's code point has no sign either.
    llvm::StringRef name{semantic->constructor->name};
    bool isUnsigned{name.starts_with("uint") || name == "character"};
    return ThreeWay(
        arith::CmpIOp::create(builder_, loc,
            isUnsigned ? arith::CmpIPredicate::ult : arith::CmpIPredicate::slt,
            x, y),
        arith::CmpIOp::create(builder_, loc,
            isUnsigned ? arith::CmpIPredicate::ugt : arith::CmpIPredicate::sgt,
            x, y),
        loc);
  }
  if (mlir::isa<mlir::FloatType>(type)) {
    if (generic) {
      // A software_error, which a try catches; mmc's program aborts.
      merk::ThrowIfOp::create(builder_, loc,
          arith::CmpFOp::create(
              builder_, loc, arith::CmpFPredicate::UNO, x, y),
          "Software Error: incomparable floats in compare/3");
    }
    // As mmc's for a float it knows is one, a NaN is equal to everything.
    return ThreeWay(
        arith::CmpFOp::create(builder_, loc, arith::CmpFPredicate::OLT, x, y),
        arith::CmpFOp::create(builder_, loc, arith::CmpFPredicate::OGT, x, y),
        loc);
  }
  llvm::SmallVector<mlir::Value> inputs;
  for (unsigned var : VariablesOf(*semantic)) {
    mlir::Value info{typeInfos_.lookup(var)};
    if (!info) {
      NotYet(equalAt_, "a comparison of a type variable without a type_info");
      return nullptr;
    }
    inputs.push_back(info);
  }
  inputs.append({x, y});
  return merk::CallOp::create(builder_, loc, mlir::TypeRange{result},
      CompareProc(mlir::cast<merk::TermType>(type)).getSymName(), inputs)
      .getResult(0);
}

void MerkGenerator::SucceedIf(
    mlir::Value condition, mlir::Value value, mlir::Location loc) {
  auto ite{merk::IteOp::create(builder_, loc, mlir::TypeRange{})};
  mlir::OpBuilder::InsertionGuard guard{builder_};
  builder_.setInsertionPointToStart(&ite.getCondition().emplaceBlock());
  merk::GuardOp::create(builder_, loc, condition);
  merk::YieldOp::create(builder_, loc, mlir::ValueRange{});
  builder_.setInsertionPointToStart(&ite.getThenRegion().emplaceBlock());
  merk::SucceedOp::create(builder_, loc, mlir::ValueRange{value});
  builder_.setInsertionPointToStart(&ite.getElseRegion().emplaceBlock());
  merk::YieldOp::create(builder_, loc, mlir::ValueRange{});
}

mlir::Value MerkGenerator::ConstructorIndex(
    mlir::Value term, mlir::ArrayAttr cases, mlir::Location loc) {
  mlir::Type i64{builder_.getI64Type()};
  auto op{merk::SwitchOp::create(builder_, loc, mlir::TypeRange{i64}, term,
      cases, /*canFail=*/false)};
  mlir::OpBuilder::InsertionGuard guard{builder_};
  for (auto [i, region] : llvm::enumerate(op.getCaseRegions())) {
    builder_.setInsertionPointToStart(&region.emplaceBlock());
    merk::YieldOp::create(
        builder_, loc, mlir::ValueRange{IntegerConstant(i64, i, loc)});
  }
  return op.getResult(0);
}

merk::ProcOp MerkGenerator::CompareProc(merk::TermType term) {
  if (merk::ProcOp proc{compareProcs_.lookup(term.getName())}) {
    return proc;
  }
  mlir::Location loc{builder_.getUnknownLoc()};
  auto i64{builder_.getI64Type()};
  const semantics::Type *type{terms_.lookup(term.getName())};
  const std::vector<semantics::ConstructorTypes> &constructors{
      *modes_.ops().ConstructorsOf(type)};
  // Which arguments the type declares as type variables, whichever its
  // instance. A tuple's all are.
  std::vector<std::vector<bool>> declaredVariable;
  semantics::TypeContext &context{types_.types()};
  if (type->kind == semantics::TypeKind::Tuple) {
    declaredVariable.emplace_back(type->arguments.size(), true);
  } else {
    llvm::SmallVector<const semantics::Type *> variables;
    for (unsigned i{0}; i < type->arguments.size(); ++i) {
      variables.push_back(context.Variable(i));
    }
    for (const semantics::ConstructorTypes &constructor :
        *modes_.ops().ConstructorsOf(
            context.Named(*type->constructor, variables))) {
      std::vector<bool> &declared{declaredVariable.emplace_back()};
      for (const semantics::Type *argument : constructor.arguments) {
        declared.push_back(argument->kind == semantics::TypeKind::Variable);
      }
    }
  }
  mlir::Type result{*ComparisonType()};
  mlir::OpBuilder::InsertionGuard guard{builder_};
  builder_.setInsertionPointToEnd(module_.getBody());
  llvm::SmallVector<unsigned> variables{VariablesOf(*type)};
  llvm::SmallVector<mlir::Type> inputs(
      variables.size(), merk::TypeInfoType::get(&context_));
  inputs.append({term, term});
  auto proc{merk::ProcOp::create(builder_, loc,
      ("compare." + term.getName()).str(),
      builder_.getFunctionType(inputs, result), merk::Determinism::Det)};
  proc.setPrivate();
  // Recorded first, for the calls of it on its terms' arguments.
  compareProcs_[term.getName()] = proc;
  mlir::Block *entry{proc.addEntryBlock()};
  builder_.setInsertionPointToStart(entry);
  auto saved{std::move(typeInfos_)};
  typeInfos_.clear();
  for (auto [i, var] : llvm::enumerate(variables)) {
    typeInfos_[var] = entry->getArgument(i);
  }
  mlir::Value x{entry->getArgument(variables.size())};
  mlir::Value y{entry->getArgument(variables.size() + 1)};
  NoteRoot(x, type);
  NoteRoot(y, type);
  // A user-defined comparison: its predicate's. A type with only a
  // user-defined equality has no order: comparing its values throws
  // software_error, as in mmc.
  if (const semantics::UserEquality *user{UserEqualityOf(type)}) {
    mlir::Value order;
    if (user->comparison) {
      CallUserPredicate(*user->comparison, true, type, x, y, loc, order);
    } else if (mlir::Value info{TypeInfoOfType(type, loc, equalAt_)}) {
      merk::ThrowIfOp::create(builder_, loc,
          mlir::arith::ConstantIntOp::create(
              builder_, loc, builder_.getI1Type(), 1),
          "Software Error: call to compare/3 for non-canonical type `%T'",
          info);
    } else {
      failed_ = true;
    }
    merk::SucceedOp::create(builder_, loc,
        mlir::ValueRange{order ? order : ComparisonValue("=", loc)});
    typeInfos_ = std::move(saved);
    return proc;
  }
  llvm::SmallVector<mlir::Attribute> cases;
  for (const semantics::ConstructorTypes &constructor : constructors) {
    cases.push_back(builder_.getArrayAttr({merk::FunctorAttr::get(&context_,
        constructor.functor.name, constructor.functor.arity)}));
  }
  // Constructors in the order declared, by their position in it, which a
  // switch gives whatever their layout.
  if (constructors.size() > 1) {
    mlir::ArrayAttr casesAttr{builder_.getArrayAttr(cases)};
    mlir::Value xIndex{ConstructorIndex(x, casesAttr, loc)};
    mlir::Value yIndex{ConstructorIndex(y, casesAttr, loc)};
    SucceedIf(mlir::arith::CmpIOp::create(builder_, loc,
                  mlir::arith::CmpIPredicate::slt, xIndex, yIndex),
        ComparisonValue("<", loc), loc);
    SucceedIf(mlir::arith::CmpIOp::create(builder_, loc,
                  mlir::arith::CmpIPredicate::sgt, xIndex, yIndex),
        ComparisonValue(">", loc), loc);
  }
  bool anyArguments{llvm::any_of(
      constructors, [](const semantics::ConstructorTypes &constructor) {
        return !constructor.arguments.empty();
      })};
  if (anyArguments) {
    auto op{merk::SwitchOp::create(builder_, loc, mlir::TypeRange{}, x,
        builder_.getArrayAttr(cases), /*canFail=*/false)};
    for (auto [c, constructor, region, attribute] :
        llvm::enumerate(constructors, op.getCaseRegions(), cases)) {
      builder_.setInsertionPointToStart(&region.emplaceBlock());
      bool ended{false};
      if (!constructor.arguments.empty()) {
        auto functor{mlir::cast<merk::FunctorAttr>(
            mlir::cast<mlir::ArrayAttr>(attribute)[0])};
        HiddenFields hidden;
        llvm::SmallVector<mlir::Type> fields{
            *CellFields(type, constructor, hidden)};
        auto xs{merk::DeconstructOp::create(builder_, loc, fields, x, functor)};
        auto ys{merk::DeconstructOp::create(builder_, loc, fields, y, functor)};
        std::size_t first{hidden.types.size()};
        llvm::SmallVector<bool> compared(hidden.places.size(), false);
        unsigned base{ExistentialBase(*type)};
        // Each existentially quantified variable's type_infos, taken before
        // any comparison: a later argument is live across the earlier ones'
        // comparisons, and a collection there needs its type_info.
        llvm::SmallVector<std::pair<mlir::Value, mlir::Value>> hiddenInfos;
        for (std::pair<unsigned, unsigned> place : hidden.places) {
          hiddenInfos.emplace_back(HiddenTypeInfo(xs.getFields(), place, loc),
              HiddenTypeInfo(ys.getFields(), place, loc));
        }
        // The first argument is the most significant, and the last one's
        // comparison, as a list's tail, is in tail position.
        for (auto [i, a, b, argument] : llvm::enumerate(
                 xs.getFields().drop_front(first),
                 ys.getFields().drop_front(first),
                 ArgumentsOf(type, constructor))) {
          // As mmc's, an argument of an existentially typed constructor is
          // compared by its type first: by the types of the existentially
          // quantified variables it has, that no argument before it has.
          for (unsigned var : VariablesOf(*argument)) {
            if (var < base || compared[var - base]) {
              continue;
            }
            compared[var - base] = true;
            auto [p, q]{hiddenInfos[var - base]};
            mlir::Value types{merk::CompareTypeInfoOp::create(
                builder_, loc, builder_.getI32Type(), p, q)};
            mlir::Value zero{IntegerConstant(builder_.getI32Type(), 0, loc)};
            SucceedIf(mlir::arith::CmpIOp::create(builder_, loc,
                          mlir::arith::CmpIPredicate::slt, types, zero),
                ComparisonValue("<", loc), loc);
            SucceedIf(mlir::arith::CmpIOp::create(builder_, loc,
                          mlir::arith::CmpIPredicate::sgt, types, zero),
                ComparisonValue(">", loc), loc);
            typeInfos_[var] = p;
          }
          NoteRoot(a, argument);
          NoteRoot(b, argument);
          mlir::Value order{Compare(a, b, argument,
              declaredVariable[c][i], loc)};
          if (!order) {
            break;
          }
          if (i + 1 == constructor.arguments.size()) {
            merk::SucceedOp::create(builder_, loc, mlir::ValueRange{order});
            ended = true;
          } else {
            auto ite{merk::IteOp::create(builder_, loc, mlir::TypeRange{})};
            mlir::OpBuilder::InsertionGuard inner{builder_};
            builder_.setInsertionPointToStart(&ite.getCondition().emplaceBlock());
            merk::GuardOp::create(
                builder_, loc, IsComparison(order, "=", loc));
            merk::YieldOp::create(builder_, loc, mlir::ValueRange{});
            builder_.setInsertionPointToStart(&ite.getThenRegion().emplaceBlock());
            merk::YieldOp::create(builder_, loc, mlir::ValueRange{});
            builder_.setInsertionPointToStart(&ite.getElseRegion().emplaceBlock());
            merk::SucceedOp::create(builder_, loc, mlir::ValueRange{order});
          }
        }
      }
      if (!ended) {
        merk::YieldOp::create(builder_, loc, mlir::ValueRange{});
      }
    }
    builder_.setInsertionPointAfter(op);
  }
  merk::SucceedOp::create(
      builder_, loc, mlir::ValueRange{ComparisonValue("=", loc)});
  typeInfos_ = std::move(saved);
  return proc;
}

// As mmc's, the array with fewer elements is the lesser, whatever they are.
merk::ProcOp MerkGenerator::CompareArrayProc() {
  if (compareArray_) {
    return compareArray_;
  }
  mlir::Location loc{builder_.getUnknownLoc()};
  auto info{merk::TypeInfoType::get(&context_)};
  auto word{merk::WordType::get(&context_)};
  auto array{merk::ArrayType::get(&context_, word)};
  auto i64{builder_.getI64Type()};
  mlir::Type result{*ComparisonType()};
  mlir::OpBuilder::InsertionGuard guard{builder_};
  builder_.setInsertionPointToEnd(module_.getBody());
  // From an index to the size, with the rest's as a tail call.
  auto from{merk::ProcOp::create(builder_, loc, "compare.array.from",
      builder_.getFunctionType({info, array, array, i64, i64}, result),
      merk::Determinism::Det)};
  from.setPrivate();
  {
    mlir::Block *entry{from.addEntryBlock()};
    builder_.setInsertionPointToStart(entry);
    NoteArrayArguments(entry, loc);
    mlir::Value i{entry->getArgument(3)}, n{entry->getArgument(4)};
    auto ite{merk::IteOp::create(builder_, loc, mlir::TypeRange{})};
    builder_.setInsertionPointToStart(&ite.getCondition().emplaceBlock());
    merk::GuardOp::create(builder_, loc,
        mlir::arith::CmpIOp::create(
            builder_, loc, mlir::arith::CmpIPredicate::slt, i, n));
    merk::YieldOp::create(builder_, loc, mlir::ValueRange{});
    builder_.setInsertionPointToStart(&ite.getThenRegion().emplaceBlock());
    mlir::Value order{merk::CompareWordOp::create(builder_, loc, result,
        entry->getArgument(0),
        ArrayElement(entry->getArgument(1), i, entry->getArgument(0), loc),
        ArrayElement(entry->getArgument(2), i, entry->getArgument(0), loc))};
    auto differs{merk::IteOp::create(builder_, loc, mlir::TypeRange{})};
    {
      mlir::OpBuilder::InsertionGuard inner{builder_};
      builder_.setInsertionPointToStart(&differs.getCondition().emplaceBlock());
      merk::GuardOp::create(builder_, loc, IsComparison(order, "=", loc));
      merk::YieldOp::create(builder_, loc, mlir::ValueRange{});
      builder_.setInsertionPointToStart(&differs.getThenRegion().emplaceBlock());
      merk::YieldOp::create(builder_, loc, mlir::ValueRange{});
      builder_.setInsertionPointToStart(&differs.getElseRegion().emplaceBlock());
      merk::SucceedOp::create(builder_, loc, mlir::ValueRange{order});
    }
    mlir::Value next{mlir::arith::AddIOp::create(
        builder_, loc, i, IntegerConstant(i64, 1, loc))};
    mlir::Value rest{merk::CallOp::create(builder_, loc, mlir::TypeRange{result},
        from.getSymName(),
        mlir::ValueRange{entry->getArgument(0), entry->getArgument(1),
            entry->getArgument(2), next, n})
                         .getResult(0)};
    merk::SucceedOp::create(builder_, loc, mlir::ValueRange{rest});
    builder_.setInsertionPointToStart(&ite.getElseRegion().emplaceBlock());
    merk::YieldOp::create(builder_, loc, mlir::ValueRange{});
    builder_.setInsertionPointAfter(ite);
    merk::SucceedOp::create(
        builder_, loc, mlir::ValueRange{ComparisonValue("=", loc)});
  }
  builder_.setInsertionPointToEnd(module_.getBody());
  compareArray_ = merk::ProcOp::create(builder_, loc, "compare.array",
      builder_.getFunctionType({info, array, array}, result),
      merk::Determinism::Det);
  compareArray_.setPrivate();
  mlir::Block *entry{compareArray_.addEntryBlock()};
  builder_.setInsertionPointToStart(entry);
  NoteArrayArguments(entry, loc);
  mlir::Value xSize{merk::ArraySizeOp::create(
      builder_, loc, i64, entry->getArgument(1))};
  mlir::Value ySize{merk::ArraySizeOp::create(
      builder_, loc, i64, entry->getArgument(2))};
  mlir::Value sizes{ThreeWay(
      mlir::arith::CmpIOp::create(
          builder_, loc, mlir::arith::CmpIPredicate::slt, xSize, ySize),
      mlir::arith::CmpIOp::create(
          builder_, loc, mlir::arith::CmpIPredicate::sgt, xSize, ySize),
      loc)};
  auto differs{merk::IteOp::create(builder_, loc, mlir::TypeRange{})};
  {
    mlir::OpBuilder::InsertionGuard inner{builder_};
    builder_.setInsertionPointToStart(&differs.getCondition().emplaceBlock());
    merk::GuardOp::create(builder_, loc, IsComparison(sizes, "=", loc));
    merk::YieldOp::create(builder_, loc, mlir::ValueRange{});
    builder_.setInsertionPointToStart(&differs.getThenRegion().emplaceBlock());
    merk::YieldOp::create(builder_, loc, mlir::ValueRange{});
    builder_.setInsertionPointToStart(&differs.getElseRegion().emplaceBlock());
    merk::SucceedOp::create(builder_, loc, mlir::ValueRange{sizes});
  }
  mlir::Value order{merk::CallOp::create(builder_, loc, mlir::TypeRange{result},
      from.getSymName(),
      mlir::ValueRange{entry->getArgument(0), entry->getArgument(1),
          entry->getArgument(2), IntegerConstant(i64, 0, loc), xSize})
                        .getResult(0)};
  merk::SucceedOp::create(builder_, loc, mlir::ValueRange{order});
  return compareArray_;
}

bool MerkGenerator::CompareBuiltin(const Body &goal, const Procedure &callee,
    llvm::ArrayRef<BodyVar> arguments, const ProcedureBody &body) {
  const semantics::Symbol &symbol{*callee.symbol};
  llvm::StringRef name{symbol.name};
  bool function{symbol.kind == semantics::SymbolKind::Function};
  BodyVar first, second;
  std::optional<BodyVar> output;
  if (function && arguments.size() == 3 &&
      (name == "ordering" || name == "array_compare")) {
    first = arguments[0];
    second = arguments[1];
    output = arguments[2];
  } else if (!function && name == "compare" && arguments.size() == 3) {
    output = arguments[0];
    first = arguments[1];
    second = arguments[2];
  } else if (!function && arguments.size() == 2 &&
      llvm::is_contained({"@<", "@=<", "@>", "@>=", "unify"}, name)) {
    first = arguments[0];
    second = arguments[1];
  } else {
    return false;
  }
  parser::CharBlock at{Where(goal.tokens)};
  mlir::Location loc{Location(goal.tokens)};
  const semantics::Type *type{body.vars[first].type};
  std::optional<Represented> represented{Represent(type)};
  if (!represented) {
    NotYet(at, "values of type " + semantics::TypeToString(*type));
    return true;
  }
  if (name == "unify") {
    if (!represented->dummy) {
      Test(goal, ValueOf(first), ValueOf(second), type);
    }
    return true;
  }
  equalAt_ = at;
  mlir::Value order;
  if (represented->dummy) {
    order = ComparisonValue("=", loc);
  } else {
    mlir::Value x{ValueOf(first)}, y{ValueOf(second)};
    llvm::StringSet<> seen;
    if (!x || !y || !Comparable(x.getType(), seen)) {
      NotYet(at, "this comparison");
      return true;
    }
    if (HasGivenType(*type)) {
      mlir::Value info{GivenTypeInfo(goal, type, loc)};
      if (!info) {
        return true;
      }
      auto word{merk::WordType::get(&context_)};
      order = merk::CompareWordOp::create(builder_, loc, *ComparisonType(), info,
          Coerce(x, word, loc), Coerce(y, word, loc));
    } else {
      order = Compare(x, y, type, /*generic=*/false, loc);
    }
    if (!order) {
      return true;
    }
  }
  if (output) {
    values_[*output] = order;
    return true;
  }
  mlir::Value holds;
  if (name == "@<" || name == "@>") {
    holds = IsComparison(order, name == "@<" ? "<" : ">", loc);
  } else {
    // Not the opposite one's.
    mlir::Value opposite{IsComparison(order, name == "@=<" ? ">" : "<", loc)};
    holds = mlir::arith::XOrIOp::create(builder_, loc, opposite,
        IntegerConstant(builder_.getI1Type(), 1, loc));
  }
  merk::GuardOp::create(builder_, loc, holds);
  return true;
}

} // namespace merkurius::codegen::mlirgen
