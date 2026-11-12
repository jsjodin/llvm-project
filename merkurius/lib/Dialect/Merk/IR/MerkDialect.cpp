//===-- lib/Dialect/Merk/IR/MerkDialect.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "merkurius/Dialect/Merk/IR/MerkDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace merkurius::merk;

#include "merkurius/Dialect/Merk/IR/MerkDialect.cpp.inc"
#include "merkurius/Dialect/Merk/IR/MerkEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "merkurius/Dialect/Merk/IR/MerkAttrs.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "merkurius/Dialect/Merk/IR/MerkTypes.cpp.inc"

void MerkDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "merkurius/Dialect/Merk/IR/MerkTypes.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "merkurius/Dialect/Merk/IR/MerkAttrs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "merkurius/Dialect/Merk/IR/MerkOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ProcOp, after the Toy tutorial's toy.func
//===----------------------------------------------------------------------===//

void ProcOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
    llvm::StringRef name, mlir::FunctionType type, Determinism determinism) {
  state.addAttribute(getSymNameAttrName(state.name),
      builder.getStringAttr(name));
  state.addAttribute(
      getFunctionTypeAttrName(state.name), mlir::TypeAttr::get(type));
  state.addAttribute(getDeterminismAttrName(state.name),
      DeterminismAttr::get(builder.getContext(), determinism));
  state.addRegion();
}

mlir::ParseResult ProcOp::parse(
    mlir::OpAsmParser &parser, mlir::OperationState &result) {
  return mlir::function_interface_impl::parseFunctionOp(
      parser, result,
      /*allowVariadic=*/false, getFunctionTypeAttrName(result.name),
      [](mlir::Builder &builder, llvm::ArrayRef<mlir::Type> argTypes,
          llvm::ArrayRef<mlir::Type> results,
          mlir::function_interface_impl::VariadicFlag,
          std::string &) { return builder.getFunctionType(argTypes, results); },
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void ProcOp::print(mlir::OpAsmPrinter &p) {
  mlir::function_interface_impl::printFunctionOp(p, *this,
      /*isVariadic=*/false, getFunctionTypeAttrName(), getArgAttrsAttrName(),
      getResAttrsAttrName());
}

//===----------------------------------------------------------------------===//
// SucceedOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult SucceedOp::verify() {
  ProcOp proc{(*this)->getParentOfType<ProcOp>()};
  if (!proc) {
    return emitOpError("is not in a procedure");
  }
  llvm::ArrayRef<mlir::Type> results{proc.getResultTypes()};
  if (getOutputs().size() != results.size()) {
    return emitOpError("has ")
        << getOutputs().size() << " outputs, but its procedure has "
        << results.size();
  }
  for (auto [output, type] : llvm::zip(getOutputs(), results)) {
    if (output.getType() != type) {
      return emitOpError("output of type ")
          << output.getType() << " does not match the procedure's " << type;
    }
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Arrays
//===----------------------------------------------------------------------===//

// That a value is of an array's element type.
static llvm::LogicalResult VerifyElement(
    mlir::Operation *op, mlir::Value array, mlir::Type type, llvm::StringRef what) {
  mlir::Type element{mlir::cast<ArrayType>(array.getType()).getElement()};
  // A flat record's value, which the array holds inline (lowering's
  // MakeRecords).
  auto record{mlir::dyn_cast<RecordType>(type)};
  auto term{mlir::dyn_cast<TermType>(element)};
  if (record && term && record.getName() == term.getName()) {
    return mlir::success();
  }
  if (type != element) {
    return op->emitOpError() << what << " of type " << type
                             << " is not of the array's element type " << element;
  }
  return mlir::success();
}

llvm::LogicalResult ArrayNewOp::verify() {
  if (!getInit()) {
    return mlir::success();
  }
  return VerifyElement(*this, getArray(), getInit().getType(), "init");
}

llvm::LogicalResult ArrayGetOp::verify() {
  return VerifyElement(*this, getArray(), getElement().getType(), "result");
}

llvm::LogicalResult ArraySetOp::verify() {
  return VerifyElement(*this, getArray(), getElement().getType(), "element");
}

llvm::LogicalResult ArrayFillOp::verify() {
  return VerifyElement(*this, getArray(), getElement().getType(), "element");
}

llvm::LogicalResult ArrayResizeOp::verify() {
  if (!getInit()) {
    return mlir::success();
  }
  return VerifyElement(*this, getArray(), getInit().getType(), "init");
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult ConstantOp::verify() {
  mlir::Type type{getType()};
  if (mlir::isa<StringType>(type) ? !mlir::isa<mlir::StringAttr>(getValue())
          : mlir::isa<mlir::IntegerType>(type)
          ? !mlir::isa<mlir::IntegerAttr>(getValue())
          : mlir::isa<mlir::FloatType>(type)
          ? !mlir::isa<mlir::FloatAttr>(getValue())
          : true) {
    return emitOpError("is not an integer, float or string constant");
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// SwitchOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult SwitchOp::verify() {
  if (getCases().size() != getCaseRegions().size()) {
    return emitOpError("has ")
        << getCases().size() << " cases but " << getCaseRegions().size()
        << " regions";
  }
  bool term{mlir::isa<TermType, RecordType>(getValue().getType())};
  for (mlir::Attribute c : getCases()) {
    auto constants{mlir::dyn_cast<mlir::ArrayAttr>(c)};
    if (!constants || constants.empty()) {
      return emitOpError("has a case that is not a list of constants");
    }
    for (mlir::Attribute constant : constants) {
      if (term) {
        if (!mlir::isa<FunctorAttr>(constant)) {
          return emitOpError("has a case constant that is not a functor");
        }
        continue;
      }
      auto integer{mlir::dyn_cast<mlir::IntegerAttr>(constant)};
      if (!integer || integer.getType() != getValue().getType()) {
        return emitOpError("has a case constant that is not of type ")
            << getValue().getType();
      }
    }
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ConstructOp and DeconstructOp
//===----------------------------------------------------------------------===//

// Whether a construction's or deconstruction's values of `types` are its
// functor's arguments, after any type_infos and typeclass_infos of an
// existentially typed constructor's hidden fields.
static bool AreFields(mlir::TypeRange types, FunctorAttr functor) {
  if (types.size() < functor.getArity()) {
    return false;
  }
  return llvm::all_of(types.drop_back(functor.getArity()), [](mlir::Type type) {
    return mlir::isa<TypeInfoType, TypeclassInfoType>(type);
  });
}

llvm::LogicalResult ConstructOp::verify() {
  if (!AreFields(getArguments().getTypes(), getFunctor())) {
    return emitOpError("has ")
        << getArguments().size() << " arguments for a functor of arity "
        << getFunctor().getArity();
  }
  return mlir::success();
}

llvm::LogicalResult DeconstructOp::verify() {
  if (!AreFields(getFields().getTypes(), getFunctor())) {
    return emitOpError("has ")
        << getFields().size() << " results for a functor of arity "
        << getFunctor().getArity();
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// CallOp and EntryOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult CallOp::verifySymbolUses(
    mlir::SymbolTableCollection &symbolTable) {
  auto proc{symbolTable.lookupNearestSymbolFrom<ProcOp>(*this, getCalleeAttr())};
  if (!proc) {
    return emitOpError("calls ") << getCalleeAttr() << ", which is no procedure";
  }
  mlir::FunctionType type{proc.getFunctionType()};
  if (getInputs().getTypes() != type.getInputs() ||
      getOutputs().getTypes() != type.getResults()) {
    return emitOpError("does not match the type of ") << getCalleeAttr();
  }
  return mlir::success();
}

// A nondet call's procedure is nondet or multi, takes its inputs, and gives
// its outputs to the continuation, which takes them as its arguments.
llvm::LogicalResult NondetCallOp::verifySymbolUses(
    mlir::SymbolTableCollection &symbolTable) {
  auto proc{symbolTable.lookupNearestSymbolFrom<ProcOp>(*this, getCalleeAttr())};
  if (!proc) {
    return emitOpError("calls ") << getCalleeAttr() << ", which is no procedure";
  }
  if (proc.getDeterminism() != Determinism::Nondet &&
      proc.getDeterminism() != Determinism::Multi) {
    return emitOpError("calls ") << getCalleeAttr()
                                 << ", which is not nondet or multi";
  }
  mlir::FunctionType type{proc.getFunctionType()};
  if (getInputs().getTypes() != type.getInputs() ||
      getContinuation().front().getArgumentTypes() != type.getResults()) {
    return emitOpError("does not match the type of ") << getCalleeAttr();
  }
  return mlir::success();
}

llvm::LogicalResult NondetCallOp::verify() {
  if (getContinuation().empty()) {
    return emitOpError("has no continuation");
  }
  return mlir::success();
}

// The continuation takes a word for each output.
llvm::LogicalResult NondetCallClosureOp::verify() {
  if (getContinuation().empty()) {
    return emitOpError("has no continuation");
  }
  if (!llvm::all_of(getContinuation().front().getArgumentTypes(),
          [](mlir::Type type) { return mlir::isa<WordType>(type); })) {
    return emitOpError("has a continuation that does not take words");
  }
  return mlir::success();
}

llvm::LogicalResult NondetCallMethodOp::verify() {
  if (getContinuation().empty()) {
    return emitOpError("has no continuation");
  }
  return mlir::success();
}

// A solution is of the innermost merk.each or merk.commit whose generator
// it is in, through the continuations of merk.each in it, and has the
// types of the continuation's arguments or of merk.commit's results.
llvm::LogicalResult SolutionOp::verify() {
  for (mlir::Region *region{(*this)->getParentRegion()}; region;
       region = region->getParentRegion()) {
    mlir::Operation *parent{region->getParentOp()};
    if (auto each{mlir::dyn_cast<EachOp>(parent)}) {
      if (region == &each.getContinuation()) {
        continue;
      }
      if (each.getContinuation().empty()) {
        return mlir::success();
      }
      if (getValues().getTypes() !=
          each.getContinuation().front().getArgumentTypes()) {
        return emitOpError("does not match the arguments of the continuation");
      }
      return mlir::success();
    }
    if (auto ite{mlir::dyn_cast<SoftIteOp>(parent)}) {
      if (region != &ite.getCondition()) {
        continue;
      }
      if (ite.getThenRegion().empty()) {
        return mlir::success();
      }
      if (getValues().getTypes() !=
          ite.getThenRegion().front().getArgumentTypes()) {
        return emitOpError("does not match the arguments of the then part");
      }
      return mlir::success();
    }
    if (auto commit{mlir::dyn_cast<CommitOp>(parent)}) {
      if (getValues().getTypes() != commit.getResultTypes()) {
        return emitOpError("does not match the results of merk.commit");
      }
      return mlir::success();
    }
    if (auto aggregate{mlir::dyn_cast<AggregateOp>(parent)}) {
      if (region != &aggregate.getGenerator()) {
        continue;
      }
      if (aggregate.getStep().empty()) {
        return mlir::success();
      }
      mlir::TypeRange arguments{
          aggregate.getStep().front().getArgumentTypes()};
      if (arguments.size() < aggregate.getNumResults() ||
          getValues().getTypes() !=
              arguments.drop_back(aggregate.getNumResults())) {
        return emitOpError("does not match the arguments of the step");
      }
      return mlir::success();
    }
    if (mlir::isa<ProcOp>(parent)) {
      break;
    }
  }
  return emitOpError(
      "is not in the generator of merk.each, merk.commit or merk.aggregate, "
      "or the condition of merk.soft_ite");
}

// The step takes a solution's values, then the accumulators, whose types
// are the operands' and the results', and yields whether to go on and the
// new accumulators.
llvm::LogicalResult AggregateOp::verify() {
  if (getInitial().getTypes() != getResultTypes()) {
    return emitOpError("results do not match its operands");
  }
  if (getStep().empty() || getStep().front().empty()) {
    return emitOpError("step does not end with merk.yield");
  }
  mlir::Block &step{getStep().front()};
  mlir::TypeRange arguments{step.getArgumentTypes()};
  if (arguments.size() < getNumResults() ||
      arguments.take_back(getNumResults()) != getResultTypes()) {
    return emitOpError("step's last arguments do not match its results");
  }
  auto yield{mlir::dyn_cast<YieldOp>(step.back())};
  if (!yield) {
    return emitOpError("step does not end with merk.yield");
  }
  mlir::TypeRange values{yield.getValues().getTypes()};
  if (values.empty() || !values.front().isInteger(1) ||
      values.drop_front() != getResultTypes()) {
    return emitOpError(
        "step does not yield an i1 and then values of its results' types");
  }
  return mlir::success();
}

llvm::LogicalResult ContRefOp::verifySymbolUses(
    mlir::SymbolTableCollection &symbolTable) {
  if (!symbolTable.lookupNearestSymbolFrom<ProcOp>(*this, getCalleeAttr())) {
    return emitOpError("names ") << getCalleeAttr() << ", which is no procedure";
  }
  return mlir::success();
}

llvm::LogicalResult TypeclassInfoOp::verifySymbolUses(
    mlir::SymbolTableCollection &symbolTable) {
  if (!symbolTable.lookupNearestSymbolFrom<InstanceOp>(*this, getInstanceAttr())) {
    return emitOpError("names ") << getInstanceAttr() << ", which is no instance";
  }
  return mlir::success();
}

llvm::LogicalResult ClosureOp::verifySymbolUses(
    mlir::SymbolTableCollection &symbolTable) {
  auto proc{symbolTable.lookupNearestSymbolFrom<ProcOp>(*this, getEntryAttr())};
  if (!proc) {
    return emitOpError("names ") << getEntryAttr() << ", which is no procedure";
  }
  mlir::FunctionType type{proc.getFunctionType()};
  mlir::Type word{WordType::get(getContext())};
  if (type.getNumInputs() == 0 ||
      !mlir::isa<ClosureType>(type.getInput(0)) ||
      llvm::count(type.getInputs(), word) + 1 != type.getNumInputs() ||
      llvm::count(type.getResults(), word) != type.getNumResults()) {
    return emitOpError("names ") << getEntryAttr()
                                 << ", which does not take a closure and words";
  }
  return mlir::success();
}

llvm::LogicalResult EntryOp::verifySymbolUses(
    mlir::SymbolTableCollection &symbolTable) {
  auto proc{symbolTable.lookupNearestSymbolFrom<ProcOp>(*this, getProcAttr())};
  if (!proc) {
    return emitOpError("names ") << getProcAttr() << ", which is no procedure";
  }
  if (!proc.getFunctionType().getInputs().empty() ||
      !proc.getFunctionType().getResults().empty()) {
    return emitOpError("names ")
        << getProcAttr() << ", which has inputs or outputs";
  }
  return mlir::success();
}

llvm::LogicalResult ConstructorOp::verifySymbolUses(
    mlir::SymbolTableCollection &symbolTable) {
  auto proc{symbolTable.lookupNearestSymbolFrom<ProcOp>(*this, getProcAttr())};
  if (!proc) {
    return emitOpError("names ") << getProcAttr() << ", which is no procedure";
  }
  if (!proc.getFunctionType().getInputs().empty() ||
      !proc.getFunctionType().getResults().empty()) {
    return emitOpError("names ")
        << getProcAttr() << ", which has inputs or outputs";
  }
  return mlir::success();
}

#define GET_OP_CLASSES
#include "merkurius/Dialect/Merk/IR/MerkOps.cpp.inc"
