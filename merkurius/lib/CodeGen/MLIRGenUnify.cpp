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
        // A term with free or dummy arguments.
        NotYet(at, "this unification");
        return;
      }
      if (!hidden.types.empty()) {
        value = Coerce(value, (*cell)[arguments.size()], loc);
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

mlir::Value MerkGenerator::ArrayElement(
    mlir::Value array, mlir::Value index, mlir::Location loc) {
  return merk::ArrayGetOp::create(builder_, loc,
      mlir::cast<merk::ArrayType>(array.getType()).getElement(), array, index);
}

mlir::Value MerkGenerator::ArrayWithElement(mlir::Value array,
    mlir::Value index, mlir::Value element, mlir::Location loc) {
  return merk::ArraySetOp::create(
      builder_, loc, array.getType(), array, index, element)
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
            entry->getArgument(0), ArrayElement(entry->getArgument(1), i, loc),
            ArrayElement(entry->getArgument(2), i, loc)));
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
  if (mlir::isa<merk::WordType>(type) ||
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
      merk::ThrowIfOp::create(builder_, loc,
          arith::CmpFOp::create(
              builder_, loc, arith::CmpFPredicate::UNO, x, y),
          "incomparable floats in compare/3");
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
            std::pair<unsigned, unsigned> place{hidden.places[var - base]};
            mlir::Value p{HiddenTypeInfo(xs.getFields(), place, loc)};
            mlir::Value q{HiddenTypeInfo(ys.getFields(), place, loc)};
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
    mlir::Value i{entry->getArgument(3)}, n{entry->getArgument(4)};
    auto ite{merk::IteOp::create(builder_, loc, mlir::TypeRange{})};
    builder_.setInsertionPointToStart(&ite.getCondition().emplaceBlock());
    merk::GuardOp::create(builder_, loc,
        mlir::arith::CmpIOp::create(
            builder_, loc, mlir::arith::CmpIPredicate::slt, i, n));
    merk::YieldOp::create(builder_, loc, mlir::ValueRange{});
    builder_.setInsertionPointToStart(&ite.getThenRegion().emplaceBlock());
    mlir::Value order{merk::CompareWordOp::create(builder_, loc, result,
        entry->getArgument(0), ArrayElement(entry->getArgument(1), i, loc),
        ArrayElement(entry->getArgument(2), i, loc))};
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
