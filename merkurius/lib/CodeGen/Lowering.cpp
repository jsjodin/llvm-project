//===-- lib/CodeGen/Lowering.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "merkurius/CodeGen/Lowering.h"
#include "merkurius/CodeGen/MLIRGen.h"
#include "merkurius/Dialect/Merk/IR/MerkDialect.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Target/TargetMachine.h"

namespace merkurius::codegen {

namespace {

// A merk type's LLVM type: a string is a pointer to its bytes.
mlir::Type Convert(mlir::Type type) {
  if (mlir::isa<merk::StringType>(type)) {
    return mlir::LLVM::LLVMPointerType::get(type.getContext());
  }
  return type;
}

llvm::SmallVector<mlir::Type> Convert(mlir::TypeRange types) {
  return llvm::map_to_vector(types, [](mlir::Type t) { return Convert(t); });
}

// A procedure's outputs as one LLVM value: none, the output, or a struct of
// them (design document, "Calling convention").
mlir::Type ResultType(mlir::MLIRContext *context, mlir::TypeRange outputs) {
  if (outputs.empty()) {
    return mlir::LLVM::LLVMVoidType::get(context);
  }
  if (outputs.size() == 1) {
    return Convert(outputs.front());
  }
  return mlir::LLVM::LLVMStructType::getLiteral(context, Convert(outputs));
}

class Lowerer {
public:
  Lowerer(mlir::ModuleOp module) : module_{module}, rewriter_{module.getContext()} {}
  llvm::LogicalResult Lower();

private:
  void LowerProc(merk::ProcOp);
  void LowerSucceed(merk::SucceedOp);
  void LowerCall(merk::CallOp);
  void LowerConstant(merk::ConstantOp);
  void LowerForeignCall(merk::ForeignCallOp);
  void LowerEntry(merk::EntryOp);

  mlir::ModuleOp module_;
  mlir::IRRewriter rewriter_;
  llvm::StringMap<mlir::LLVM::LLVMFuncOp> funcs_;
  // The global of each string constant's bytes.
  llvm::StringMap<mlir::LLVM::GlobalOp> strings_;
};

void Lowerer::LowerProc(merk::ProcOp proc) {
  mlir::MLIRContext *context{proc.getContext()};
  mlir::FunctionType type{proc.getFunctionType()};
  auto funcType{mlir::LLVM::LLVMFunctionType::get(
      ResultType(context, type.getResults()), Convert(type.getInputs()))};
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
  for (mlir::Block &block : func.getBody()) {
    for (mlir::BlockArgument argument : block.getArguments()) {
      argument.setType(Convert(argument.getType()));
    }
  }
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

void Lowerer::LowerConstant(merk::ConstantOp constant) {
  mlir::Location loc{constant.getLoc()};
  auto string{mlir::dyn_cast<mlir::StringAttr>(constant.getValue())};
  if (!string) {
    rewriter_.setInsertionPoint(constant);
    rewriter_.replaceOpWithNewOp<mlir::LLVM::ConstantOp>(
        constant, constant.getType(), constant.getValue());
    return;
  }
  // A string is a constant global of its bytes and a NUL.
  mlir::LLVM::GlobalOp &global{strings_[string.getValue()]};
  if (!global) {
    mlir::OpBuilder::InsertionGuard guard{rewriter_};
    rewriter_.setInsertionPointToStart(module_.getBody());
    std::string bytes{string.getValue()};
    bytes.push_back('\0');
    auto type{mlir::LLVM::LLVMArrayType::get(
        rewriter_.getI8Type(), bytes.size())};
    global = mlir::LLVM::GlobalOp::create(rewriter_, loc, type,
        /*isConstant=*/true, mlir::LLVM::Linkage::Private,
        "merk.str." + std::to_string(strings_.size() - 1),
        rewriter_.getStringAttr(bytes));
    global.setUnnamedAddr(mlir::LLVM::UnnamedAddr::Global);
  }
  rewriter_.setInsertionPoint(constant);
  rewriter_.replaceOpWithNewOp<mlir::LLVM::AddressOfOp>(constant, global);
}

// A foreign_proc's C function returns its only output, or else stores its
// outputs through pointers after its inputs.
void Lowerer::LowerForeignCall(merk::ForeignCallOp call) {
  mlir::MLIRContext *context{call.getContext()};
  mlir::Location loc{call.getLoc()};
  llvm::SmallVector<mlir::Type> outputs{Convert(call.getOutputs().getTypes())};
  bool byPointer{outputs.size() > 1};
  auto pointer{mlir::LLVM::LLVMPointerType::get(context)};
  mlir::LLVM::LLVMFuncOp &func{funcs_[call.getCallee()]};
  if (!func) {
    llvm::SmallVector<mlir::Type> inputs{Convert(call.getInputs().getTypes())};
    if (byPointer) {
      inputs.append(outputs.size(), pointer);
    }
    mlir::Type result{outputs.size() == 1
            ? outputs.front()
            : mlir::LLVM::LLVMVoidType::get(context)};
    mlir::OpBuilder::InsertionGuard guard{rewriter_};
    rewriter_.setInsertionPointToEnd(module_.getBody());
    func = mlir::LLVM::LLVMFuncOp::create(rewriter_, loc, call.getCallee(),
        mlir::LLVM::LLVMFunctionType::get(result, inputs));
  }
  llvm::SmallVector<mlir::Value> arguments{call.getInputs()};
  llvm::SmallVector<mlir::Value> slots;
  if (byPointer) {
    mlir::OpBuilder::InsertionGuard guard{rewriter_};
    rewriter_.setInsertionPointToStart(
        &call->getParentOfType<mlir::LLVM::LLVMFuncOp>().getBody().front());
    mlir::Value one{mlir::LLVM::ConstantOp::create(
        rewriter_, loc, rewriter_.getI64Type(), rewriter_.getI64IntegerAttr(1))};
    for (mlir::Type type : outputs) {
      slots.push_back(
          mlir::LLVM::AllocaOp::create(rewriter_, loc, pointer, type, one));
    }
    arguments.append(slots);
  }
  rewriter_.setInsertionPoint(call);
  auto llvmCall{mlir::LLVM::CallOp::create(rewriter_, loc, func, arguments)};
  if (!byPointer) {
    rewriter_.replaceOp(call, llvmCall.getResults());
    return;
  }
  llvm::SmallVector<mlir::Value> values;
  for (auto [slot, type] : llvm::zip(slots, outputs)) {
    values.push_back(mlir::LLVM::LoadOp::create(rewriter_, loc, type, slot));
  }
  rewriter_.replaceOp(call, values);
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
    if (mlir::isa<merk::SucceedOp, merk::CallOp, merk::ConstantOp,
            merk::ForeignCallOp>(op)) {
      ops.push_back(op);
    }
  });
  for (mlir::Operation *op : ops) {
    if (auto succeed{mlir::dyn_cast<merk::SucceedOp>(op)}) {
      LowerSucceed(succeed);
    } else if (auto constant{mlir::dyn_cast<merk::ConstantOp>(op)}) {
      LowerConstant(constant);
    } else if (auto foreign{mlir::dyn_cast<merk::ForeignCallOp>(op)}) {
      LowerForeignCall(foreign);
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

bool LinkForeignCode(llvm::Module &module, mlir::ModuleOp mlirModule,
    const TargetInfo &target, llvm::StringRef clang, std::string &error) {
  auto code{mlirModule->getAttrOfType<mlir::StringAttr>(kForeignCodeAttrName)};
  if (!code) {
    return true;
  }
  if (clang.empty()) {
    error = "foreign code needs clang, which was not found";
    return false;
  }
  llvm::SmallString<128> source, bitcode;
  int fd;
  if (std::error_code ec{
          llvm::sys::fs::createTemporaryFile("merk-foreign", "c", fd, source)}) {
    error = "cannot write the foreign code: " + ec.message();
    return false;
  }
  llvm::FileRemover removeSource{source};
  {
    llvm::raw_fd_ostream os{fd, /*shouldClose=*/true};
    os << code.getValue();
  }
  if (std::error_code ec{
          llvm::sys::fs::createTemporaryFile("merk-foreign", "bc", bitcode)}) {
    error = "cannot write the foreign code: " + ec.message();
    return false;
  }
  llvm::FileRemover removeBitcode{bitcode};
  std::string triple{"--target=" + target.triple().str()};
  std::string optimization{
      "-O" + std::to_string(target.options().optimizationLevel)};
  llvm::SmallVector<llvm::StringRef> args{clang, triple, optimization, "-fPIC",
      "-c", "-emit-llvm", "-x", "c", source, "-o", bitcode};
  std::string cpu{"-mcpu=" + target.options().cpu};
  if (!target.options().cpu.empty()) {
    args.insert(args.begin() + 2, cpu);
  }
  std::string message;
  int status{llvm::sys::ExecuteAndWait(
      clang, args, std::nullopt, {}, 0, 0, &message)};
  if (status != 0) {
    error = "clang could not compile the module's foreign code" +
        (message.empty() ? std::string{} : ": " + message);
    return false;
  }
  llvm::SMDiagnostic diagnostic;
  std::unique_ptr<llvm::Module> foreign{
      llvm::parseIRFile(bitcode, diagnostic, module.getContext())};
  if (!foreign) {
    error = "cannot read the foreign code clang compiled: " +
        diagnostic.getMessage().str();
    return false;
  }
  foreign->setDataLayout(module.getDataLayout());
  foreign->setTargetTriple(module.getTargetTriple());
  if (llvm::Linker::linkModules(module, std::move(foreign))) {
    error = "cannot link the module's foreign code";
    return false;
  }
  // The functions around foreign_procs are called only from the module.
  for (llvm::Function &function : module) {
    if (!function.isDeclaration() &&
        function.getName().starts_with("merk_foreign_")) {
      function.setLinkage(llvm::GlobalValue::InternalLinkage);
    }
  }
  return true;
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
