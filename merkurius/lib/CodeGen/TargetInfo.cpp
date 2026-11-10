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
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/OpenMP/Utils/Utils.h"
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
// No -O was given (TargetOptions::optimizationLevelGiven).
static constexpr llvm::StringLiteral kNoOptimizationLevel{
    "merk.optimization_level_default"};
static constexpr llvm::StringLiteral kAddressSpaces{"merk.address_spaces"};
static constexpr llvm::StringLiteral kCodeObjectVersion{
    "merk.amdgpu_code_object_version"};
// A device compile of GPU processes' kernels (-fprocess-kernels).
static constexpr llvm::StringLiteral kProcessKernels{"merk.process_kernels"};

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
  // Code generation for an AMD GPU is at -O2 unless -O was given
  // (TargetOptions::optimizationLevelGiven); the passes before it follow -O.
  unsigned codeGenLevel{
      info->triple_.isAMDGPU() && !options.optimizationLevelGiven
          ? 2
          : options.optimizationLevel};
  std::optional<llvm::CodeGenOptLevel> level{
      llvm::CodeGenOpt::getLevel(codeGenLevel)};
  if (!level) {
    error = "invalid optimization level " +
        std::to_string(options.optimizationLevel);
    return nullptr;
  }
  // Constructors go in .init_array, as Clang's do by default, which is all
  // that today's dynamic loaders run, rather than in .ctors.
  llvm::TargetOptions targetOptions;
  targetOptions.UseInitArray = true;
  info->targetMachine_.reset(target->createTargetMachine(info->triple_,
      info->options_.cpu, info->options_.features, targetOptions,
      llvm::Reloc::PIC_, std::nullopt, *level));
  if (!info->targetMachine_) {
    error = "cannot make a target machine for " + info->triple_.str();
    return nullptr;
  }
  info->dataLayout_ = dataLayout
      ? llvm::DataLayout{*dataLayout}
      : info->targetMachine_->createDataLayout();
  // The collector's address space, on the host, is non-integral: the optimizer
  // must not turn its pointers into integers, which would hide a root from
  // the statepoint rewrite. A device has no collector.
  if (!dataLayout && !info->triple_.isAMDGPU() && !info->triple_.isNVPTX()) {
    llvm::Expected<llvm::DataLayout> layout{llvm::DataLayout::parse(
        info->dataLayout_.getStringRepresentation() + "-ni:1")};
    if (!layout) {
      error = llvm::toString(layout.takeError());
      return nullptr;
    }
    info->dataLayout_ = *layout;
  }
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
  options.optimizationLevelGiven = !module->hasAttr(kNoOptimizationLevel);
  if (auto version{
          module->getAttrOfType<mlir::IntegerAttr>(kCodeObjectVersion)}) {
    options.codeObjectVersion = version.getInt();
  }
  options.processKernels = module->hasAttr(kProcessKernels);
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

bool TargetInfo::ProcessKernels(mlir::ModuleOp module) {
  return module->hasAttr(kProcessKernels);
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
  if (!options_.optimizationLevelGiven) {
    module->setAttr(kNoOptimizationLevel, builder.getUnitAttr());
  }
  module->setAttr(kAddressSpaces,
      merk::AddressSpacesAttr::get(context, addressSpaces_.generic,
          addressSpaces_.global, addressSpaces_.shared,
          addressSpaces_.constant, addressSpaces_.private_));
  if (triple_.isAMDGPU()) {
    module->setAttr(kCodeObjectVersion,
        builder.getI32IntegerAttr(options_.codeObjectVersion));
  }
  if (options_.processKernels) {
    module->setAttr(kProcessKernels, builder.getUnitAttr());
  }
  // OpenMP's: the omp dialect's offload attributes, as Clang and Flang set
  // them, which its translation to LLVM IR reads.
  if (options_.openMP) {
    context->loadDialect<mlir::omp::OpenMPDialect>();
    llvm::SmallVector<llvm::Triple> targets;
    for (const std::string &target : options_.openMPTargets) {
      targets.emplace_back(target);
    }
    mlir::omp::setOffloadModuleInterfaceAttributes(module,
        mlir::omp::OffloadModuleOpts{/*targetDebugKind=*/0,
            /*assumeTeamsOversubscription=*/false,
            /*assumeThreadsOversubscription=*/false,
            /*assumeNoThreadState=*/false,
            /*assumeNoNestedParallelism=*/false,
            options_.openMPIsTargetDevice, triple_.isGPU(),
            /*forceUSM=*/false, options_.openMPVersion,
            options_.openMPHostIRFile, targets});
    mlir::omp::setOpenMPVersionAttribute(module, options_.openMPVersion);
  }
}

void TargetInfo::AddModuleFlags(llvm::Module &module) const {
  if (triple_.isAMDGPU()) {
    module.addModuleFlag(llvm::Module::Error, "amdhsa_code_object_version",
        options_.codeObjectVersion);
    module.setPICLevel(llvm::PICLevel::BigPIC);
  }
}

// From the back ends' return conventions: RetCC_X86_64_C and
// RetCC_X86Common (RAX, RDX and RCX; XMM0 and XMM1), RetCC_AArch64_AAPCS
// (X0 to X7; D0 to D7) and RetCC_AMDGPU_Func (VGPR0 to VGPR31, two to a
// 64-bit value). Elsewhere, two of each, as RISC-V's a0 and a1 and fa0 and
// fa1, the fewest of LLVM's targets.
TargetInfo::ResultRegisters TargetInfo::Results() const {
  if (triple_.getArch() == llvm::Triple::x86_64) {
    return {3, 2};
  }
  if (triple_.isAArch64()) {
    return {8, 8};
  }
  if (triple_.isAMDGPU()) {
    return {32 / (WordBits() / 32), 0, /*shared=*/true};
  }
  return {2, 2};
}

unsigned TargetInfo::TagBits() const {
  return llvm::Log2_32(PointerAlignBits(addressSpaces_.generic) / 8);
}

} // namespace merkurius::codegen
