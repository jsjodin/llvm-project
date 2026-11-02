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
  auto buildFuncType{[](mlir::Builder &builder,
                         llvm::ArrayRef<mlir::Type> argTypes,
                         llvm::ArrayRef<mlir::Type> results,
                         mlir::function_interface_impl::VariadicFlag,
                         std::string &) {
    return builder.getFunctionType(argTypes, results);
  }};
  return mlir::function_interface_impl::parseFunctionOp(parser, result,
      /*allowVariadic=*/false, getFunctionTypeAttrName(result.name),
      buildFuncType, getArgAttrsAttrName(result.name),
      getResAttrsAttrName(result.name));
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
