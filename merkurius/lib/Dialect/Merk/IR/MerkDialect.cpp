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
  bool term{mlir::isa<TermType>(getValue().getType())};
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

#define GET_OP_CLASSES
#include "merkurius/Dialect/Merk/IR/MerkOps.cpp.inc"
