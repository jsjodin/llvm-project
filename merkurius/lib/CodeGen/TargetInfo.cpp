//===-- lib/CodeGen/TargetInfo.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "merkurius/CodeGen/TargetInfo.h"
#include "merkurius/Dialect/Merk/IR/MerkDialect.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/TargetParser/Host.h"

namespace merkurius::codegen {

static constexpr llvm::StringLiteral kCPU{"merk.target_cpu"};
static constexpr llvm::StringLiteral kFeatures{"merk.target_features"};
static constexpr llvm::StringLiteral kOptimizationLevel{
    "merk.optimization_level"};
static constexpr llvm::StringLiteral kAddressSpaces{"merk.address_spaces"};

// The numbers LLVM's back ends give the address spaces.
static AddressSpaces AddressSpacesOf(const llvm::Triple &triple) {
  if (triple.isAMDGPU() || triple.isNVPTX()) {
    // Flat, global, local (LDS or shared), constant, private.
    return {0, 1, 3, 4, 5};
  }
  if (triple.isSPIROrSPIRV()) {
    // OpenCL's: private 0, global 1, constant 2, local 3, generic 4.
    return {4, 1, 3, 2, 0};
  }
  return {};
}

std::unique_ptr<TargetInfo> TargetInfo::Make(const TargetOptions &options,
    const std::string *dataLayout, std::string &error) {
  std::unique_ptr<TargetInfo> info{new TargetInfo};
  info->options_ = options;
  if (info->options_.triple.empty()) {
    info->options_.triple = llvm::sys::getDefaultTargetTriple();
  }
  info->triple_ = llvm::Triple{llvm::Triple::normalize(info->options_.triple)};
  const llvm::Target *target{
      llvm::TargetRegistry::lookupTarget(info->triple_, error)};
  if (!target) {
    return nullptr;
  }
  std::optional<llvm::CodeGenOptLevel> level{
      llvm::CodeGenOpt::getLevel(options.optimizationLevel)};
  if (!level) {
    error = "invalid optimization level " +
        std::to_string(options.optimizationLevel);
    return nullptr;
  }
  info->targetMachine_.reset(target->createTargetMachine(info->triple_,
      info->options_.cpu, info->options_.features, llvm::TargetOptions{},
      llvm::Reloc::PIC_, std::nullopt, *level));
  if (!info->targetMachine_) {
    error = "cannot make a target machine for " + info->triple_.str();
    return nullptr;
  }
  info->dataLayout_ = dataLayout
      ? llvm::DataLayout{*dataLayout}
      : info->targetMachine_->createDataLayout();
  info->addressSpaces_ = AddressSpacesOf(info->triple_);
  return info;
}

std::unique_ptr<TargetInfo> TargetInfo::Create(
    const TargetOptions &options, std::string &error) {
  return Make(options, nullptr, error);
}

std::unique_ptr<TargetInfo> TargetInfo::FromModule(
    mlir::ModuleOp module, std::string &error) {
  auto triple{module->getAttrOfType<mlir::StringAttr>(
      mlir::LLVM::LLVMDialect::getTargetTripleAttrName())};
  if (!triple) {
    error = "the module records no target triple";
    return nullptr;
  }
  TargetOptions options;
  options.triple = triple.getValue().str();
  if (auto cpu{module->getAttrOfType<mlir::StringAttr>(kCPU)}) {
    options.cpu = cpu.getValue().str();
  }
  if (auto features{module->getAttrOfType<mlir::StringAttr>(kFeatures)}) {
    options.features = features.getValue().str();
  }
  if (auto level{module->getAttrOfType<mlir::IntegerAttr>(kOptimizationLevel)}) {
    options.optimizationLevel = level.getInt();
  }
  std::optional<std::string> layout;
  if (auto attr{module->getAttrOfType<mlir::StringAttr>(
          mlir::LLVM::LLVMDialect::getDataLayoutAttrName())}) {
    layout = attr.getValue().str();
  }
  std::unique_ptr<TargetInfo> info{
      Make(options, layout ? &*layout : nullptr, error)};
  if (info) {
    if (auto spaces{
            module->getAttrOfType<merk::AddressSpacesAttr>(kAddressSpaces)}) {
      info->addressSpaces_ = {spaces.getGeneric(), spaces.getGlobal(),
          spaces.getShared(), spaces.getConstant(), spaces.getPrivate_()};
    }
  }
  return info;
}

void TargetInfo::Record(mlir::ModuleOp module) const {
  mlir::MLIRContext *context{module.getContext()};
  context->loadDialect<mlir::DLTIDialect, mlir::LLVM::LLVMDialect,
      merk::MerkDialect>();
  mlir::Builder builder{context};
  module->setAttr(mlir::LLVM::LLVMDialect::getTargetTripleAttrName(),
      builder.getStringAttr(triple_.str()));
  module->setAttr(mlir::LLVM::LLVMDialect::getDataLayoutAttrName(),
      builder.getStringAttr(dataLayout_.getStringRepresentation()));
  module->setAttr(mlir::DLTIDialect::kDataLayoutAttrName,
      mlir::translateDataLayout(dataLayout_, context));
  if (!options_.cpu.empty()) {
    module->setAttr(kCPU, builder.getStringAttr(options_.cpu));
  }
  if (!options_.features.empty()) {
    module->setAttr(kFeatures, builder.getStringAttr(options_.features));
  }
  module->setAttr(kOptimizationLevel,
      builder.getI32IntegerAttr(options_.optimizationLevel));
  module->setAttr(kAddressSpaces,
      merk::AddressSpacesAttr::get(context, addressSpaces_.generic,
          addressSpaces_.global, addressSpaces_.shared,
          addressSpaces_.constant, addressSpaces_.private_));
}

unsigned TargetInfo::TagBits() const {
  return llvm::Log2_32(PointerAlignBits(addressSpaces_.generic) / 8);
}

} // namespace merkurius::codegen
