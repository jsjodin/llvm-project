//===-- lib/CodeGen/Lowering.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "merkurius/CodeGen/Lowering.h"
#include "merkurius/Dialect/Merk/IR/MerkDialect.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Target/TargetMachine.h"

namespace merkurius::codegen {

namespace {

// A procedure's outputs as one LLVM value: none, the output, or a struct of
// them (design document, "Calling convention").
mlir::Type ResultType(mlir::MLIRContext *context, mlir::TypeRange outputs) {
  if (outputs.empty()) {
    return mlir::LLVM::LLVMVoidType::get(context);
  }
  if (outputs.size() == 1) {
    return outputs.front();
  }
  return mlir::LLVM::LLVMStructType::getLiteral(
      context, llvm::to_vector(outputs));
}

class Lowerer {
public:
  Lowerer(mlir::ModuleOp module) : module_{module}, rewriter_{module.getContext()} {}
  llvm::LogicalResult Lower();

private:
  void LowerProc(merk::ProcOp);
  void LowerSucceed(merk::SucceedOp);
  void LowerCall(merk::CallOp);
  void LowerEntry(merk::EntryOp);

  mlir::ModuleOp module_;
  mlir::IRRewriter rewriter_;
  llvm::StringMap<mlir::LLVM::LLVMFuncOp> funcs_;
};

void Lowerer::LowerProc(merk::ProcOp proc) {
  mlir::MLIRContext *context{proc.getContext()};
  mlir::FunctionType type{proc.getFunctionType()};
  auto funcType{mlir::LLVM::LLVMFunctionType::get(
      ResultType(context, type.getResults()), type.getInputs())};
  // A procedure defined here and not exported is local to the module; one
  // declared is defined in another.
  mlir::LLVM::Linkage linkage{!proc.isExternal() && proc.isPrivate()
          ? mlir::LLVM::Linkage::Internal
          : mlir::LLVM::Linkage::External};
  rewriter_.setInsertionPoint(proc);
  auto func{mlir::LLVM::LLVMFuncOp::create(rewriter_, proc.getLoc(),
      proc.getSymName(), funcType, linkage, /*dsoLocal=*/false,
      mlir::LLVM::CConv::Tail)};
  rewriter_.inlineRegionBefore(
      proc.getBody(), func.getBody(), func.getBody().end());
  funcs_[proc.getSymName()] = func;
  rewriter_.eraseOp(proc);
}

void Lowerer::LowerSucceed(merk::SucceedOp succeed) {
  rewriter_.setInsertionPoint(succeed);
  mlir::ValueRange outputs{succeed.getOutputs()};
  if (outputs.size() <= 1) {
    rewriter_.replaceOpWithNewOp<mlir::LLVM::ReturnOp>(succeed, outputs);
    return;
  }
  mlir::Location loc{succeed.getLoc()};
  mlir::Type type{ResultType(succeed.getContext(), outputs.getTypes())};
  mlir::Value result{mlir::LLVM::UndefOp::create(rewriter_, loc, type)};
  for (auto [i, output] : llvm::enumerate(outputs)) {
    result = mlir::LLVM::InsertValueOp::create(rewriter_, loc, result, output,
        llvm::ArrayRef<int64_t>{static_cast<int64_t>(i)});
  }
  rewriter_.replaceOpWithNewOp<mlir::LLVM::ReturnOp>(succeed, result);
}

void Lowerer::LowerCall(merk::CallOp call) {
  rewriter_.setInsertionPoint(call);
  mlir::LLVM::LLVMFuncOp func{funcs_.lookup(call.getCallee())};
  auto llvmCall{mlir::LLVM::CallOp::create(
      rewriter_, call.getLoc(), func, call.getInputs())};
  llvmCall.setCConv(mlir::LLVM::CConv::Tail);
  if (call.getNumResults() <= 1) {
    rewriter_.replaceOp(call, llvmCall.getResults());
    return;
  }
  llvm::SmallVector<mlir::Value> outputs;
  for (unsigned i = 0; i < call.getNumResults(); ++i) {
    outputs.push_back(mlir::LLVM::ExtractValueOp::create(rewriter_,
        call.getLoc(), llvmCall.getResult(),
        llvm::ArrayRef<int64_t>{static_cast<int64_t>(i)}));
  }
  rewriter_.replaceOp(call, outputs);
}

// merk_main, a C function that calls the main procedure.
void Lowerer::LowerEntry(merk::EntryOp entry) {
  mlir::MLIRContext *context{entry.getContext()};
  mlir::Location loc{entry.getLoc()};
  rewriter_.setInsertionPoint(entry);
  auto func{mlir::LLVM::LLVMFuncOp::create(rewriter_, loc, kEntrySymbol,
      mlir::LLVM::LLVMFunctionType::get(
          mlir::LLVM::LLVMVoidType::get(context), {}))};
  mlir::Block *block{func.addEntryBlock(rewriter_)};
  rewriter_.setInsertionPointToStart(block);
  auto call{mlir::LLVM::CallOp::create(
      rewriter_, loc, funcs_.lookup(entry.getProc()), mlir::ValueRange{})};
  call.setCConv(mlir::LLVM::CConv::Tail);
  mlir::LLVM::ReturnOp::create(rewriter_, loc, mlir::ValueRange{});
  rewriter_.eraseOp(entry);
}

llvm::LogicalResult Lowerer::Lower() {
  module_.getContext()->loadDialect<mlir::LLVM::LLVMDialect>();
  for (auto proc : llvm::make_early_inc_range(module_.getOps<merk::ProcOp>())) {
    LowerProc(proc);
  }
  llvm::SmallVector<mlir::Operation *> ops;
  module_.walk([&](mlir::Operation *op) {
    if (mlir::isa<merk::SucceedOp, merk::CallOp>(op)) {
      ops.push_back(op);
    }
  });
  for (mlir::Operation *op : ops) {
    if (auto succeed{mlir::dyn_cast<merk::SucceedOp>(op)}) {
      LowerSucceed(succeed);
    } else {
      LowerCall(mlir::cast<merk::CallOp>(op));
    }
  }
  for (auto entry : llvm::make_early_inc_range(module_.getOps<merk::EntryOp>())) {
    LowerEntry(entry);
  }
  // Every operation left must be the LLVM dialect's.
  mlir::WalkResult result{module_.walk([](mlir::Operation *op) {
    if (op->getDialect() &&
        op->getDialect()->getNamespace() == merk::MerkDialect::getDialectNamespace()) {
      op->emitOpError("is not lowered yet");
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  })};
  return llvm::failure(result.wasInterrupted());
}

} // namespace

llvm::LogicalResult LowerToLLVMDialect(
    mlir::ModuleOp module, const TargetInfo &) {
  return Lowerer{module}.Lower();
}

std::unique_ptr<llvm::Module> TranslateToLLVMIR(mlir::ModuleOp module,
    llvm::LLVMContext &context, const TargetInfo &target) {
  mlir::DialectRegistry registry;
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  module.getContext()->appendDialectRegistry(registry);
  std::unique_ptr<llvm::Module> llvmModule{mlir::translateModuleToLLVMIR(
      module, context, module.getName().value_or("module"))};
  if (llvmModule) {
    llvmModule->setTargetTriple(target.triple());
    llvmModule->setDataLayout(target.dataLayout());
  }
  return llvmModule;
}

void OptimizeLLVMIR(llvm::Module &module, const TargetInfo &target) {
  llvm::LoopAnalysisManager loops;
  llvm::FunctionAnalysisManager functions;
  llvm::CGSCCAnalysisManager sccs;
  llvm::ModuleAnalysisManager modules;
  llvm::PassBuilder builder{module.getContext(), &target.targetMachine()};
  builder.registerModuleAnalyses(modules);
  builder.registerCGSCCAnalyses(sccs);
  builder.registerFunctionAnalyses(functions);
  builder.registerLoopAnalyses(loops);
  builder.crossRegisterProxies(loops, functions, sccs, modules);
  llvm::OptimizationLevel level;
  switch (target.options().optimizationLevel) {
  case 0:
    level = llvm::OptimizationLevel::O0;
    break;
  case 1:
    level = llvm::OptimizationLevel::O1;
    break;
  case 2:
    level = llvm::OptimizationLevel::O2;
    break;
  default:
    level = llvm::OptimizationLevel::O3;
    break;
  }
  llvm::ModulePassManager passes{level == llvm::OptimizationLevel::O0
          ? builder.buildO0DefaultPipeline(level)
          : builder.buildPerModuleDefaultPipeline(level)};
  passes.run(module, modules);
}

bool EmitCode(llvm::Module &module, const TargetInfo &target,
    llvm::CodeGenFileType fileType, llvm::raw_pwrite_stream &os,
    std::string &error) {
  llvm::legacy::PassManager passes;
  if (target.targetMachine().addPassesToEmitFile(
          passes, os, nullptr, fileType)) {
    error = "the target cannot emit this kind of file";
    return false;
  }
  passes.run(module);
  return true;
}

} // namespace merkurius::codegen
