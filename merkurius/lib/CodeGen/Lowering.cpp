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
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/OpenMP/Transforms/Passes.h"
#include "mlir/Dialect/OpenMP/Utils/Utils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Object/OffloadBinary.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

namespace merkurius::codegen {

namespace {

// The runtime's function that reports an uncaught exception and exits.
constexpr llvm::StringLiteral kThrowSymbol{"merk_uncaught_exception"};

// Whether a procedure of this determinism returns a success flag (design
// document, "Calling convention").
bool CanFail(merk::Determinism determinism) {
  return determinism == merk::Determinism::Semidet ||
      determinism == merk::Determinism::Failure ||
      determinism == merk::Determinism::CcNondet;
}

// A merk type's LLVM type: a string is a pointer to its bytes, and an array
// a pointer to its size and elements.
mlir::Type Convert(mlir::Type type) {
  if (mlir::isa<merk::StringType, merk::ArrayType>(type)) {
    return mlir::LLVM::LLVMPointerType::get(type.getContext());
  }
  return type;
}

llvm::SmallVector<mlir::Type> Convert(mlir::TypeRange types) {
  return llvm::map_to_vector(types, [](mlir::Type t) { return Convert(t); });
}

// A procedure's outputs as one LLVM value: none, the output, or a struct of
// them, after the success flag of one that can fail (design document,
// "Calling convention").
mlir::Type ResultType(
    mlir::MLIRContext *context, mlir::TypeRange outputs, bool canFail = false) {
  llvm::SmallVector<mlir::Type> types;
  if (canFail) {
    types.push_back(mlir::IntegerType::get(context, 1));
  }
  types.append(Convert(outputs));
  if (types.empty()) {
    return mlir::LLVM::LLVMVoidType::get(context);
  }
  if (types.size() == 1) {
    return types.front();
  }
  return mlir::LLVM::LLVMStructType::getLiteral(context, types);
}

class Lowerer {
public:
  Lowerer(mlir::ModuleOp module, const TargetInfo &target)
      : module_{module}, rewriter_{module.getContext()},
        cconv_{target.HasTailCallConvention() ? mlir::LLVM::CConv::Tail
                                              : mlir::LLVM::CConv::C} {
    auto offload{mlir::dyn_cast<mlir::omp::OffloadModuleInterface>(
        module.getOperation())};
    device_ = mlir::omp::isOpenMPModule(module) && offload &&
        offload.getIsTargetDevice();
  }
  llvm::LogicalResult Lower();

private:
  void LowerProc(merk::ProcOp);
  void LowerSucceed(merk::SucceedOp);
  void LowerConstant(merk::ConstantOp);
  void LowerForeignCall(merk::ForeignCallOp);
  void LowerEntry(merk::EntryOp);
  void LowerArray(mlir::Operation *);
  // An array element's word, from its value, and its value, from the word.
  mlir::Value ToWord(mlir::Value, mlir::Location);
  mlir::Value FromWord(mlir::Value word, mlir::Type, mlir::Location);
  // The address of an array's element.
  mlir::Value ElementAddress(mlir::Value array, mlir::Value index,
      mlir::Location);
  // A function of the runtime library, declared once.
  mlir::LLVM::LLVMFuncOp RuntimeFunc(llvm::StringRef name, mlir::Type result,
      llvm::ArrayRef<mlir::Type> arguments, bool noreturn = false);
  // Makes the if-then-elses, switches, negations, tests and calls that can
  // fail of a procedure branches between blocks.
  void Flatten(mlir::LLVM::LLVMFuncOp);
  // Flattens a block, whose goals that fail go to `fail`.
  void Flatten(mlir::Block *, mlir::Block *fail);
  // Flattens the regions of an OpenMP construct, whose goals are det.
  void FlattenRegions(mlir::Operation *);
  // OpenMP's passes after lowering, as Flang runs them: procedures called
  // from a target region become declare target, and a device compile keeps
  // only those and the target regions.
  llvm::LogicalResult FilterProcs();
  llvm::LogicalResult LowerOpenMP();
  mlir::Block *SplitAfter(mlir::Operation *);
  mlir::Block *Inline(mlir::Region &, mlir::Block *before);
  void Exit(mlir::Block *, mlir::Block *after);
  mlir::Block *After(mlir::Operation *);
  mlir::Block *Continue(mlir::Block *after);
  // Lowers a call; for a callee that can fail, returns its success flag.
  // A call in tail position returns, and sets `tail`.
  mlir::Value Call(merk::CallOp, bool *tail = nullptr);
  // A procedure's result: its success flag, if it has one, and outputs.
  mlir::Value Result(mlir::Location, mlir::Type, std::optional<bool> flag,
      mlir::ValueRange outputs);
  mlir::LLVM::GlobalOp StringGlobal(llvm::StringRef, mlir::Location);
  mlir::LLVM::LLVMFuncOp ThrowFunc();
  bool CanFail(llvm::StringRef proc) const { return canFail_.contains(proc); }

  mlir::ModuleOp module_;
  mlir::IRRewriter rewriter_;
  llvm::StringMap<mlir::LLVM::LLVMFuncOp> funcs_;
  // The global of each string constant's bytes.
  llvm::StringMap<mlir::LLVM::GlobalOp> strings_;
  // The procedures that can fail.
  llvm::StringSet<> canFail_;
  mlir::LLVM::LLVMFuncOp throw_;
  // Whether this is an OpenMP device compile.
  bool device_{false};
  // The procedures' calling convention, the target's (design document,
  // "Calling convention").
  mlir::LLVM::CConv cconv_;
};

void Lowerer::LowerProc(merk::ProcOp proc) {
  mlir::MLIRContext *context{proc.getContext()};
  mlir::FunctionType type{proc.getFunctionType()};
  bool canFail{codegen::CanFail(proc.getDeterminism())};
  if (canFail) {
    canFail_.insert(proc.getSymName());
  }
  auto funcType{mlir::LLVM::LLVMFunctionType::get(
      ResultType(context, type.getResults(), canFail),
      Convert(type.getInputs()))};
  // A procedure defined here and not exported is local to the module; one
  // declared is defined in another.
  mlir::LLVM::Linkage linkage{!proc.isExternal() && proc.isPrivate()
          ? mlir::LLVM::Linkage::Internal
          : mlir::LLVM::Linkage::External};
  rewriter_.setInsertionPoint(proc);
  auto func{mlir::LLVM::LLVMFuncOp::create(rewriter_, proc.getLoc(),
      proc.getSymName(), funcType, linkage, /*dsoLocal=*/false,
      cconv_)};
  // What omp-mark-declare-target found, for the passes after lowering.
  if (mlir::Attribute declare{proc->getDiscardableAttr("omp.declare_target")}) {
    func->setDiscardableAttr("omp.declare_target", declare);
  }
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
  auto func{succeed->getParentOfType<mlir::LLVM::LLVMFuncOp>()};
  bool canFail{CanFail(func.getSymName())};
  mlir::ValueRange outputs{succeed.getOutputs()};
  if (!canFail && outputs.empty()) {
    rewriter_.replaceOpWithNewOp<mlir::LLVM::ReturnOp>(succeed, outputs);
    return;
  }
  mlir::Value result{Result(succeed.getLoc(),
      func.getFunctionType().getReturnType(),
      canFail ? std::optional<bool>{true} : std::nullopt, outputs)};
  rewriter_.replaceOpWithNewOp<mlir::LLVM::ReturnOp>(succeed, result);
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
  mlir::LLVM::GlobalOp global{StringGlobal(string.getValue(), loc)};
  rewriter_.setInsertionPoint(constant);
  rewriter_.replaceOpWithNewOp<mlir::LLVM::AddressOfOp>(constant, global);
}

// A foreign_proc's C function returns its only output, or else stores its
// outputs through pointers after its inputs.
void Lowerer::LowerForeignCall(merk::ForeignCallOp call) {
  mlir::MLIRContext *context{call.getContext()};
  mlir::Location loc{call.getLoc()};
  llvm::SmallVector<mlir::Type> outputs{Convert(call.getOutputs().getTypes())};
  // The success flag of one that can fail is C's int, which it returns.
  bool canFail{call.getCanFail()};
  if (canFail) {
    outputs.erase(outputs.begin());
  }
  bool byPointer{canFail ? !outputs.empty() : outputs.size() > 1};
  mlir::Type flagType{rewriter_.getI32Type()};
  auto pointer{mlir::LLVM::LLVMPointerType::get(context)};
  mlir::LLVM::LLVMFuncOp &func{funcs_[call.getCallee()]};
  if (!func) {
    llvm::SmallVector<mlir::Type> inputs{Convert(call.getInputs().getTypes())};
    if (byPointer) {
      inputs.append(outputs.size(), pointer);
    }
    mlir::Type result{canFail   ? flagType
            : outputs.size() == 1 ? outputs.front()
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
  llvm::SmallVector<mlir::Value> values;
  if (canFail) {
    values.push_back(mlir::LLVM::ICmpOp::create(rewriter_, loc,
        mlir::LLVM::ICmpPredicate::ne, llvmCall.getResult(),
        mlir::LLVM::ConstantOp::create(rewriter_, loc, flagType,
            rewriter_.getI32IntegerAttr(0))));
  } else if (!byPointer) {
    rewriter_.replaceOp(call, llvmCall.getResults());
    return;
  }
  for (auto [slot, type] : llvm::zip(slots, outputs)) {
    values.push_back(mlir::LLVM::LoadOp::create(rewriter_, loc, type, slot));
  }
  rewriter_.replaceOp(call, values);
}

// merk_main, a C function that calls the main procedure.
mlir::LLVM::GlobalOp Lowerer::StringGlobal(llvm::StringRef value, mlir::Location loc) {
  // A string is a constant global of its bytes and a NUL.
  mlir::LLVM::GlobalOp &global{strings_[value]};
  if (!global) {
    mlir::OpBuilder::InsertionGuard guard{rewriter_};
    rewriter_.setInsertionPointToStart(module_.getBody());
    std::string bytes{value};
    bytes.push_back('\0');
    auto type{mlir::LLVM::LLVMArrayType::get(
        rewriter_.getI8Type(), bytes.size())};
    global = mlir::LLVM::GlobalOp::create(rewriter_, loc, type,
        /*isConstant=*/true, mlir::LLVM::Linkage::Private,
        "merk.str." + std::to_string(strings_.size() - 1),
        rewriter_.getStringAttr(bytes));
    global.setUnnamedAddr(mlir::LLVM::UnnamedAddr::Global);
  }
  return global;
}

mlir::Value Lowerer::Result(mlir::Location loc, mlir::Type type,
    std::optional<bool> flag, mlir::ValueRange outputs) {
  if (!flag && outputs.size() == 1) {
    return outputs.front();
  }
  if (flag && outputs.empty()) {
    return mlir::LLVM::ConstantOp::create(rewriter_, loc,
        rewriter_.getI1Type(), rewriter_.getBoolAttr(*flag));
  }
  mlir::Value result{mlir::LLVM::UndefOp::create(rewriter_, loc, type)};
  int64_t i{0};
  if (flag) {
    mlir::Value value{mlir::LLVM::ConstantOp::create(rewriter_, loc,
        rewriter_.getI1Type(), rewriter_.getBoolAttr(*flag))};
    result = mlir::LLVM::InsertValueOp::create(
        rewriter_, loc, result, value, llvm::ArrayRef<int64_t>{i++});
  }
  for (mlir::Value output : outputs) {
    result = mlir::LLVM::InsertValueOp::create(
        rewriter_, loc, result, output, llvm::ArrayRef<int64_t>{i++});
  }
  return result;
}

mlir::Value Lowerer::Call(merk::CallOp call, bool *tail) {
  mlir::Location loc{call.getLoc()};
  rewriter_.setInsertionPoint(call);
  mlir::LLVM::LLVMFuncOp func{funcs_.lookup(call.getCallee())};
  auto llvmCall{
      mlir::LLVM::CallOp::create(rewriter_, loc, func, call.getInputs())};
  llvmCall.setCConv(cconv_);
  mlir::Value result{
      llvmCall.getNumResults() ? llvmCall.getResult() : mlir::Value{}};
  // A call in tail position, whose solution is the procedure's, is a
  // tail call (design document, "Calling convention").
  auto succeed{mlir::dyn_cast_or_null<merk::SucceedOp>(call->getNextNode())};
  auto caller{call->getParentOfType<mlir::LLVM::LLVMFuncOp>()};
  if (succeed && llvm::equal(succeed.getOutputs(), call.getResults()) &&
      CanFail(caller.getSymName()) == CanFail(call.getCallee())) {
    llvmCall.setTailCallKind(mlir::LLVM::TailCallKind::MustTail);
    rewriter_.setInsertionPoint(succeed);
    rewriter_.replaceOpWithNewOp<mlir::LLVM::ReturnOp>(
        succeed, result ? mlir::ValueRange{result} : mlir::ValueRange{});
    // The call's results have no uses but the return's.
    rewriter_.eraseOp(call);
    if (tail) {
      *tail = true;
    }
    return nullptr;
  }
  bool canFail{CanFail(call.getCallee())};
  std::size_t outputs{call.getNumResults()};
  llvm::SmallVector<mlir::Value> values;
  if (!canFail && outputs <= 1) {
    values.append(llvmCall->result_begin(), llvmCall->result_end());
  } else {
    for (unsigned i = 0; i < outputs; ++i) {
      values.push_back(mlir::LLVM::ExtractValueOp::create(rewriter_, loc,
          result, llvm::ArrayRef<int64_t>{static_cast<int64_t>(i + canFail)}));
    }
  }
  rewriter_.replaceOp(call, values);
  if (!canFail) {
    return nullptr;
  }
  return outputs ? mlir::Value{mlir::LLVM::ExtractValueOp::create(
                       rewriter_, loc, result, llvm::ArrayRef<int64_t>{0})}
                 : result;
}

// Splits the block after `op`, and returns the block of the operations
// after it.
mlir::Block *Lowerer::SplitAfter(mlir::Operation *op) {
  return rewriter_.splitBlock(op->getBlock(), std::next(op->getIterator()));
}

// Inlines a region of a structured operation's one block before `before`.
mlir::Block *Lowerer::Inline(mlir::Region &region, mlir::Block *before) {
  mlir::Block *block{&region.front()};
  rewriter_.inlineRegionBefore(region, *before->getParent(), before->getIterator());
  for (mlir::BlockArgument argument : block->getArguments()) {
    argument.setType(Convert(argument.getType()));
  }
  return block;
}

// Ends a branch, whose block ends with a yield, by going to `after` with
// the yielded values, or by succeeding there if all `after` does is
// succeed, so that a call before is in tail position.
void Lowerer::Exit(mlir::Block *block, mlir::Block *after) {
  auto yield{mlir::dyn_cast<merk::YieldOp>(block->back())};
  if (!yield) {
    return; // it fails, or succeeds
  }
  rewriter_.setInsertionPoint(yield);
  if (auto succeed{mlir::dyn_cast<merk::SucceedOp>(after->front())}) {
    mlir::IRMapping mapping;
    mapping.map(after->getArguments(), yield.getValues());
    rewriter_.clone(*succeed, mapping);
  } else {
    mlir::LLVM::BrOp::create(rewriter_, yield.getLoc(), yield.getValues(), after);
  }
  rewriter_.eraseOp(yield);
}

// The block after a structured operation, whose results become its
// arguments.
mlir::Block *Lowerer::After(mlir::Operation *op) {
  mlir::Block *after{SplitAfter(op)};
  for (mlir::Value result : op->getResults()) {
    result.replaceAllUsesWith(
        after->addArgument(Convert(result.getType()), op->getLoc()));
  }
  return after;
}

// Erases the block after a structured operation if every branch has
// succeeded or failed on its own, and returns the block to go on with.
mlir::Block *Lowerer::Continue(mlir::Block *after) {
  if (after->hasNoPredecessors()) {
    rewriter_.eraseBlock(after);
    return nullptr;
  }
  return after;
}

void Lowerer::Flatten(mlir::Block *block, mlir::Block *fail) {
  for (mlir::Block *current{block}; current;) {
    mlir::Block *next{nullptr};
    for (mlir::Operation &op : llvm::make_early_inc_range(*current)) {
      mlir::Location loc{op.getLoc()};
      if (auto guard{mlir::dyn_cast<merk::GuardOp>(op)}) {
        next = SplitAfter(guard);
        rewriter_.setInsertionPoint(guard);
        rewriter_.replaceOpWithNewOp<mlir::LLVM::CondBrOp>(
            guard, guard.getCondition(), next, fail);
        break;
      }
      if (mlir::isa<merk::FailOp>(op)) {
        rewriter_.setInsertionPoint(&op);
        rewriter_.replaceOpWithNewOp<mlir::LLVM::BrOp>(&op, fail);
        break;
      }
      if (auto throwIf{mlir::dyn_cast<merk::ThrowIfOp>(op)}) {
        next = SplitAfter(throwIf);
        mlir::Block *raise{rewriter_.createBlock(next)};
        if (device_) {
          // A device has no runtime to report it yet: the kernel traps.
          mlir::LLVM::Trap::create(rewriter_, loc);
        } else {
          mlir::Value exception{mlir::LLVM::AddressOfOp::create(rewriter_,
              loc, StringGlobal(throwIf.getException(), loc))};
          mlir::LLVM::CallOp::create(
              rewriter_, loc, ThrowFunc(), mlir::ValueRange{exception});
        }
        mlir::LLVM::UnreachableOp::create(rewriter_, loc);
        rewriter_.setInsertionPoint(throwIf);
        rewriter_.replaceOpWithNewOp<mlir::LLVM::CondBrOp>(
            throwIf, throwIf.getCondition(), raise, next);
        break;
      }
      if (auto check{mlir::dyn_cast<merk::CheckIndexOp>(op)}) {
        // As an unsigned comparison, a negative index is out of range too.
        next = SplitAfter(check);
        rewriter_.setInsertionPoint(check);
        // The array's size is its first word; the array may still be
        // merk's, from an operation not lowered yet.
        mlir::Value array{check->getOperand(0)}, index{check->getOperand(1)};
        mlir::Value size{mlir::LLVM::LoadOp::create(
            rewriter_, loc, rewriter_.getI64Type(), array)};
        mlir::Value in{mlir::LLVM::ICmpOp::create(
            rewriter_, loc, mlir::LLVM::ICmpPredicate::ult, index, size)};
        mlir::Block *raise{rewriter_.createBlock(next)};
        if (device_) {
          mlir::LLVM::Trap::create(rewriter_, loc);
        } else {
          auto ptr{mlir::LLVM::LLVMPointerType::get(rewriter_.getContext())};
          auto i64{rewriter_.getI64Type()};
          mlir::Value what{mlir::LLVM::AddressOfOp::create(
              rewriter_, loc, StringGlobal(check.getWhat(), loc))};
          mlir::Value max{mlir::LLVM::SubOp::create(rewriter_, loc, size,
              mlir::LLVM::ConstantOp::create(rewriter_, loc, i64, 1))};
          mlir::LLVM::CallOp::create(rewriter_, loc,
              RuntimeFunc("merk_index_out_of_bounds",
                  mlir::LLVM::LLVMVoidType::get(rewriter_.getContext()),
                  {ptr, i64, i64}, /*noreturn=*/true),
              mlir::ValueRange{what, index, max});
        }
        mlir::LLVM::UnreachableOp::create(rewriter_, loc);
        rewriter_.setInsertionPoint(check);
        rewriter_.replaceOpWithNewOp<mlir::LLVM::CondBrOp>(
            check, in, next, raise);
        break;
      }
      if (auto call{mlir::dyn_cast<merk::CallOp>(op)}) {
        bool tail{false};
        if (mlir::Value succeeded{Call(call, &tail)}) {
          // The callee can fail: go on if it succeeded.
          next = SplitAfter(succeeded.getDefiningOp());
          rewriter_.setInsertionPointToEnd(current);
          mlir::LLVM::CondBrOp::create(rewriter_, loc, succeeded, next, fail);
          break;
        }
        if (tail) {
          break; // it returns
        }
        continue;
      }
      if (mlir::isa<mlir::omp::OpenMPDialect>(op.getDialect())) {
        FlattenRegions(&op);
        continue;
      }
      if (auto ite{mlir::dyn_cast<merk::IteOp>(op)}) {
        mlir::Block *after{After(ite)};
        mlir::Block *condition{Inline(ite.getCondition(), after)};
        mlir::Block *then{Inline(ite.getThenRegion(), after)};
        mlir::Block *otherwise{Inline(ite.getElseRegion(), after)};
        rewriter_.setInsertionPoint(ite);
        mlir::LLVM::BrOp::create(rewriter_, loc, condition);
        rewriter_.eraseOp(ite);
        if (auto yield{mlir::dyn_cast<merk::YieldOp>(condition->back())}) {
          rewriter_.setInsertionPoint(yield);
          rewriter_.replaceOpWithNewOp<mlir::LLVM::BrOp>(
              yield, yield.getValues(), then);
        }
        Exit(then, after);
        Exit(otherwise, after);
        Flatten(condition, otherwise);
        Flatten(then, fail);
        Flatten(otherwise, fail);
        next = Continue(after);
        break;
      }
      if (auto switchOp{mlir::dyn_cast<merk::SwitchOp>(op)}) {
        mlir::Block *after{After(switchOp)};
        llvm::SmallVector<mlir::Block *> blocks;
        for (mlir::Region &region : switchOp.getCaseRegions()) {
          blocks.push_back(Inline(region, after));
        }
        mlir::Block *otherwise{fail};
        if (!switchOp.getCanFail()) {
          otherwise = rewriter_.createBlock(after);
          mlir::LLVM::UnreachableOp::create(rewriter_, loc);
        }
        llvm::SmallVector<llvm::APInt> values;
        llvm::SmallVector<mlir::Block *> destinations;
        for (auto [c, block] : llvm::zip(switchOp.getCases(), blocks)) {
          for (mlir::Attribute constant : mlir::cast<mlir::ArrayAttr>(c)) {
            values.push_back(mlir::cast<mlir::IntegerAttr>(constant).getValue());
            destinations.push_back(block);
          }
        }
        rewriter_.setInsertionPoint(switchOp);
        llvm::SmallVector<mlir::ValueRange> operands(values.size());
        mlir::LLVM::SwitchOp::create(rewriter_, loc, switchOp.getValue(),
            otherwise, mlir::ValueRange{}, values, destinations, operands);
        rewriter_.eraseOp(switchOp);
        for (mlir::Block *block : blocks) {
          Exit(block, after);
        }
        for (mlir::Block *block : blocks) {
          Flatten(block, fail);
        }
        next = Continue(after);
        break;
      }
      if (auto notOp{mlir::dyn_cast<merk::NotOp>(op)}) {
        mlir::Block *after{After(notOp)};
        mlir::Block *goal{Inline(notOp.getGoal(), after)};
        rewriter_.setInsertionPoint(notOp);
        rewriter_.replaceOpWithNewOp<mlir::LLVM::BrOp>(notOp, goal);
        // If the goal succeeds, the negation fails; if it fails, the
        // negation succeeds.
        if (auto yield{mlir::dyn_cast<merk::YieldOp>(goal->back())}) {
          rewriter_.setInsertionPoint(yield);
          rewriter_.replaceOpWithNewOp<mlir::LLVM::BrOp>(yield, fail);
        }
        Flatten(goal, after);
        next = Continue(after);
        break;
      }
    }
    current = next;
  }
}

void Lowerer::FlattenRegions(mlir::Operation *op) {
  for (mlir::Region &region : op->getRegions()) {
    if (region.empty()) {
      continue;
    }
    mlir::OpBuilder::InsertionGuard guard{rewriter_};
    mlir::Block *fail{rewriter_.createBlock(&region, region.end())};
    mlir::LLVM::UnreachableOp::create(rewriter_, op->getLoc());
    Flatten(&region.front(), fail);
    if (fail->hasNoPredecessors()) {
      rewriter_.eraseBlock(fail);
    }
  }
}

void Lowerer::Flatten(mlir::LLVM::LLVMFuncOp func) {
  if (func.isExternal()) {
    return;
  }
  // Where a goal that fails goes at the top: a semidet procedure returns
  // false; a det one cannot fail.
  mlir::Location loc{func.getLoc()};
  mlir::Block *fail{rewriter_.createBlock(&func.getBody(), func.getBody().end())};
  if (CanFail(func.getSymName())) {
    mlir::Type type{func.getFunctionType().getReturnType()};
    llvm::SmallVector<mlir::Value> outputs;
    if (auto structType{mlir::dyn_cast<mlir::LLVM::LLVMStructType>(type)}) {
      for (mlir::Type output : structType.getBody().drop_front()) {
        outputs.push_back(mlir::LLVM::UndefOp::create(rewriter_, loc, output));
      }
    }
    mlir::LLVM::ReturnOp::create(
        rewriter_, loc, Result(loc, type, false, outputs));
  } else {
    mlir::LLVM::UnreachableOp::create(rewriter_, loc);
  }
  Flatten(&func.getBody().front(), fail);
  if (fail->hasNoPredecessors()) {
    rewriter_.eraseBlock(fail);
  }
}

mlir::LLVM::LLVMFuncOp Lowerer::RuntimeFunc(llvm::StringRef name,
    mlir::Type result, llvm::ArrayRef<mlir::Type> arguments, bool noreturn) {
  if (auto func{module_.lookupSymbol<mlir::LLVM::LLVMFuncOp>(name)}) {
    return func;
  }
  mlir::OpBuilder::InsertionGuard guard{rewriter_};
  rewriter_.setInsertionPointToEnd(module_.getBody());
  auto func{mlir::LLVM::LLVMFuncOp::create(rewriter_, module_.getLoc(), name,
      mlir::LLVM::LLVMFunctionType::get(result, arguments))};
  func.setNoUnwind(true);
  if (noreturn) {
    func->setAttr("passthrough",
        rewriter_.getArrayAttr({rewriter_.getStringAttr("noreturn")}));
  }
  return func;
}

// An element's word: an integer extended, a float's bits, a pointer's
// address.
mlir::Value Lowerer::ToWord(mlir::Value value, mlir::Location loc) {
  auto i64{rewriter_.getI64Type()};
  // Its type may still be merk's, from an operation not lowered yet.
  mlir::Type type{Convert(value.getType())};
  if (type == i64) {
    return value;
  }
  if (mlir::isa<mlir::IntegerType>(type)) {
    return mlir::LLVM::ZExtOp::create(rewriter_, loc, i64, value);
  }
  if (mlir::isa<mlir::LLVM::LLVMPointerType>(type)) {
    return mlir::LLVM::PtrToIntOp::create(rewriter_, loc, i64, value);
  }
  return mlir::LLVM::BitcastOp::create(rewriter_, loc, i64, value);
}

mlir::Value Lowerer::FromWord(
    mlir::Value word, mlir::Type type, mlir::Location loc) {
  if (type == word.getType()) {
    return word;
  }
  if (mlir::isa<mlir::IntegerType>(type)) {
    return mlir::LLVM::TruncOp::create(rewriter_, loc, type, word);
  }
  if (mlir::isa<mlir::LLVM::LLVMPointerType>(type)) {
    return mlir::LLVM::IntToPtrOp::create(rewriter_, loc, type, word);
  }
  return mlir::LLVM::BitcastOp::create(rewriter_, loc, type, word);
}

// Element I is the word after I + 1 words, the first being the size.
mlir::Value Lowerer::ElementAddress(
    mlir::Value array, mlir::Value index, mlir::Location loc) {
  auto i64{rewriter_.getI64Type()};
  mlir::Value offset{mlir::LLVM::AddOp::create(rewriter_, loc, index,
      mlir::LLVM::ConstantOp::create(rewriter_, loc, i64, 1))};
  return mlir::LLVM::GEPOp::create(rewriter_, loc,
      mlir::LLVM::LLVMPointerType::get(rewriter_.getContext()), i64, array,
      mlir::ValueRange{offset});
}

void Lowerer::LowerArray(mlir::Operation *op) {
  mlir::Location loc{op->getLoc()};
  rewriter_.setInsertionPoint(op);
  auto ptr{mlir::LLVM::LLVMPointerType::get(rewriter_.getContext())};
  auto i64{rewriter_.getI64Type()};
  auto call{[&](llvm::StringRef name, mlir::Type result,
                mlir::ValueRange arguments) {
    llvm::SmallVector<mlir::Type> types{llvm::map_to_vector(
        arguments, [](mlir::Value v) { return Convert(v.getType()); })};
    return mlir::LLVM::CallOp::create(rewriter_, loc,
        RuntimeFunc(name, result, types), arguments)
        .getResult();
  }};
  // An absent initial element: any word.
  auto init{[&](unsigned operand) -> mlir::Value {
    return op->getNumOperands() > operand
        ? ToWord(op->getOperand(operand), loc)
        : mlir::LLVM::ConstantOp::create(rewriter_, loc, i64, 0);
  }};
  // The operands, not the accessors: an array the procedure was passed is
  // already a pointer.
  auto operand{[&](unsigned i) { return op->getOperand(i); }};
  mlir::Value result;
  if (mlir::isa<merk::ArrayNewOp>(op)) {
    result = call("merk_array_new", ptr, {operand(0), init(1)});
  } else if (mlir::isa<merk::ArraySizeOp>(op)) {
    result = mlir::LLVM::LoadOp::create(rewriter_, loc, i64, operand(0));
  } else if (mlir::isa<merk::ArrayGetOp>(op)) {
    mlir::Value word{mlir::LLVM::LoadOp::create(rewriter_, loc, i64,
        ElementAddress(operand(0), operand(1), loc))};
    result = FromWord(word, Convert(op->getResult(0).getType()), loc);
  } else if (mlir::isa<merk::ArraySetOp>(op)) {
    mlir::LLVM::StoreOp::create(rewriter_, loc, ToWord(operand(2), loc),
        ElementAddress(operand(0), operand(1), loc));
    result = operand(0);
  } else if (mlir::isa<merk::ArrayFillOp>(op)) {
    result = call("merk_array_fill", ptr,
        {operand(0), operand(1), operand(2), ToWord(operand(3), loc)});
  } else if (mlir::isa<merk::ArrayCopyOp>(op)) {
    result = call("merk_array_copy", ptr, {operand(0)});
  } else {
    assert(mlir::isa<merk::ArrayResizeOp>(op));
    result = call("merk_array_resize", ptr, {operand(0), operand(1), init(2)});
  }
  rewriter_.replaceOp(op, result);
}

mlir::LLVM::LLVMFuncOp Lowerer::ThrowFunc() {
  if (!throw_) {
    mlir::OpBuilder::InsertionGuard guard{rewriter_};
    rewriter_.setInsertionPointToEnd(module_.getBody());
    mlir::MLIRContext *context{module_.getContext()};
    throw_ = mlir::LLVM::LLVMFuncOp::create(rewriter_, module_.getLoc(),
        kThrowSymbol,
        mlir::LLVM::LLVMFunctionType::get(mlir::LLVM::LLVMVoidType::get(context),
            {mlir::LLVM::LLVMPointerType::get(context)}));
    throw_.setNoUnwind(true);
    throw_->setAttr("passthrough",
        rewriter_.getArrayAttr({rewriter_.getStringAttr("noreturn")}));
  }
  return throw_;
}

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
  call.setCConv(cconv_);
  mlir::LLVM::ReturnOp::create(rewriter_, loc, mlir::ValueRange{});
  rewriter_.eraseOp(entry);
}

llvm::LogicalResult Lowerer::Lower() {
  module_.getContext()->loadDialect<mlir::LLVM::LLVMDialect>();
  if (mlir::omp::isOpenMPModule(module_) && mlir::failed(FilterProcs())) {
    return llvm::failure();
  }
  for (auto proc : llvm::make_early_inc_range(module_.getOps<merk::ProcOp>())) {
    LowerProc(proc);
  }
  for (auto func : llvm::to_vector(module_.getOps<mlir::LLVM::LLVMFuncOp>())) {
    Flatten(func);
  }
  llvm::SmallVector<mlir::Operation *> ops;
  module_.walk([&](mlir::Operation *op) {
    if (mlir::isa<merk::SucceedOp, merk::CallOp, merk::ConstantOp,
            merk::ForeignCallOp, merk::ArrayNewOp, merk::ArraySizeOp,
            merk::ArrayGetOp, merk::ArraySetOp, merk::ArrayFillOp,
            merk::ArrayCopyOp, merk::ArrayResizeOp>(op)) {
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
    } else if (!mlir::isa<merk::CallOp>(op)) {
      LowerArray(op);
    } else {
      Call(mlir::cast<merk::CallOp>(op));
    }
  }
  for (auto entry : llvm::make_early_inc_range(module_.getOps<merk::EntryOp>())) {
    LowerEntry(entry);
  }
  // Arithmetic, which is the arith dialect's.
  mlir::LLVMTypeConverter converter{module_.getContext()};
  mlir::RewritePatternSet patterns{module_.getContext()};
  mlir::arith::populateArithToLLVMConversionPatterns(converter, patterns);
  mlir::ConversionTarget target{*module_.getContext()};
  target.addIllegalDialect<mlir::arith::ArithDialect>();
  target.markUnknownOpDynamicallyLegal([](mlir::Operation *) { return true; });
  if (mlir::failed(mlir::applyPartialConversion(
          module_.getOperation(), target, std::move(patterns)))) {
    return llvm::failure();
  }
  if (mlir::omp::isOpenMPModule(module_) && mlir::failed(LowerOpenMP())) {
    return llvm::failure();
  }
  // Every operation left must be the LLVM dialect's or OpenMP's.
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

// The omp dialect's passes on the merk module, before it is lowered, so
// that a device compile lowers only what the device runs, as ClangIR emits
// only that: omp-mark-declare-target marks the procedures the target
// regions call, through merk.call, and omp-function-filter removes the
// others on the device, except those that hold target regions.
llvm::LogicalResult Lowerer::FilterProcs() {
  mlir::MLIRContext *context{module_.getContext()};
  merk::ProcOp::attachInterface<
      mlir::omp::DeclareTargetDefaultModel<merk::ProcOp>>(*context);
  mlir::PassManager mark{context};
  mark.addPass(mlir::omp::createMarkDeclareTargetPass());
  if (mlir::failed(mark.run(module_))) {
    return llvm::failure();
  }
  // A target region in a procedure that a target region calls would run on
  // the device, as reverse offloading would: the directive belongs to the
  // caller.
  for (auto proc : module_.getOps<merk::ProcOp>()) {
    mlir::omp::DeclareTargetAttr declare{
        mlir::cast<mlir::omp::DeclareTargetInterface>(proc.getOperation())
            .getDeclareTarget()};
    if (!declare ||
        declare.getDeviceType() == mlir::omp::DeclareTargetDeviceType::host) {
      continue;
    }
    mlir::WalkResult nested{proc.walk([&](mlir::omp::TargetOp target) {
      mlir::emitError(target.getLoc(),
          "a target region cannot be in " + proc.getSymName() +
              ", which a target region calls");
      return mlir::WalkResult::interrupt();
    })};
    if (nested.wasInterrupted()) {
      return llvm::failure();
    }
  }
  mlir::PassManager filter{context};
  filter.addPass(mlir::omp::createFunctionFilteringPass());
  return filter.run(module_);
}

// The omp dialect's passes after lowering, as ClangIR runs them after its
// lowering to the LLVM dialect: omp-mark-declare-target again, for what
// lowering calls, omp-host-op-filter, which leaves only the target regions
// of the host procedures that hold them on the device, and
// omp-stack-to-shared.
llvm::LogicalResult Lowerer::LowerOpenMP() {
  mlir::PassManager pm{module_.getContext()};
  pm.addPass(mlir::omp::createMarkDeclareTargetPass());
  pm.addPass(mlir::omp::createHostOpFilteringPass());
  pm.nest<mlir::LLVM::LLVMFuncOp>().addPass(
      mlir::omp::createStackToSharedPass());
  if (mlir::failed(pm.run(module_))) {
    return llvm::failure();
  }
  if (!device_) {
    return llvm::success();
  }
  // Only this module's procedures are compiled for the device so far: not
  // other modules', such as the library's, nor foreign code.
  for (auto func : module_.getOps<mlir::LLVM::LLVMFuncOp>()) {
    if (!func.isExternal()) {
      continue;
    }
    if (auto uses{mlir::SymbolTable::getSymbolUses(func, module_)};
        uses && !uses->empty()) {
      mlir::emitError(uses->begin()->getUser()->getLoc(),
          "cannot call " + func.getSymName() +
              " on an OpenMP device yet: it is not compiled for the device");
      return llvm::failure();
    }
  }
  return llvm::success();
}

} // namespace

llvm::LogicalResult LowerToLLVMDialect(
    mlir::ModuleOp module, const TargetInfo &target) {
  return Lowerer{module, target}.Lower();
}

std::unique_ptr<llvm::Module> TranslateToLLVMIR(mlir::ModuleOp module,
    llvm::LLVMContext &context, const TargetInfo &target) {
  mlir::DialectRegistry registry;
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  mlir::registerOpenMPDialectTranslation(registry);
  module.getContext()->appendDialectRegistry(registry);
  std::unique_ptr<llvm::Module> llvmModule{mlir::translateModuleToLLVMIR(
      module, context, module.getName().value_or("module"))};
  if (llvmModule) {
    llvmModule->setTargetTriple(target.triple());
    llvmModule->setDataLayout(target.dataLayout());
    target.AddModuleFlags(*llvmModule);
  }
  return llvmModule;
}

bool LinkForeignCode(llvm::Module &module, mlir::ModuleOp mlirModule,
    const TargetInfo &target, llvm::StringRef clang, std::string &error) {
  auto code{mlirModule->getAttrOfType<mlir::StringAttr>(kForeignCodeAttrName)};
  auto offload{mlir::dyn_cast<mlir::omp::OffloadModuleInterface>(
      mlirModule.getOperation())};
  // A device compile calls none of it (LowerOpenMP).
  if (!code || (offload && offload.getIsTargetDevice())) {
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

bool EmbedOffloadObjects(llvm::Module &module,
    const std::vector<std::string> &files, std::string &error) {
  for (const std::string &file : files) {
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> object{
        llvm::MemoryBuffer::getFileOrSTDIN(file)};
    if (!object) {
      error = "could not open '" + file + "' for embedding";
      return false;
    }
    llvm::embedBufferInModule(module, **object, ".llvm.offloading",
        llvm::Align(llvm::object::OffloadBinary::getAlignment()));
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
