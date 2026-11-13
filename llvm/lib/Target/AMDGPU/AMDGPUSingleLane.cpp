//===-- AMDGPUSingleLane.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// A function with the "amdgpu-single-lane" attribute runs with exactly one
/// lane of its wave active, so each of its values is the same in every
/// active lane, though the uniformity analysis cannot know it: a load
/// through a flat pointer, a call's result and an argument are divergent at
/// their sources. This pass reads each such source's value from the first
/// active lane, which is the one lane, so that everything computed from it
/// is uniform: it is kept in scalar registers, which spill to lanes of a
/// vector register rather than to scratch, and its branches need no exec
/// masking.
///
/// Code that the function inlines runs with the one lane too, so the pass
/// runs after inlining, on what the function holds then. A function with the
/// attribute inlined into one without it simply loses the reads.
///
/// The pass runs at every optimization level, optnone functions included: a
/// frontend may need the branches uniform, as a musttail call cannot be
/// reached by a divergent branch, which the CFG structurizer cannot handle.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"

#define DEBUG_TYPE "amdgpu-single-lane"

using namespace llvm;

namespace {

/// Whether readfirstlane takes values of type \p Ty.
bool isLaneOpType(Type *Ty) {
  if (auto *VTy = dyn_cast<FixedVectorType>(Ty))
    return !VTy->getElementType()->isIntegerTy(1) &&
           isLaneOpType(VTy->getElementType());
  if (auto *ITy = dyn_cast<IntegerType>(Ty))
    return ITy->getBitWidth() <= 64 || ITy->getBitWidth() % 32 == 0;
  return Ty->isPointerTy() || Ty->isFloatingPointTy();
}

/// Whether \p V's only uses return it, as a tail call's result, which is
/// left as it is so that the call stays a tail call.
bool onlyReturned(const Value *V) {
  return all_of(V->users(), [](const User *U) { return isa<ReturnInst>(U); });
}

/// Replaces the uses of \p V by its value in the first active lane, read at
/// \p Where. An aggregate's parts are read where they are extracted.
bool readFirstLane(Value *V, BasicBlock::iterator Where) {
  if (V->use_empty() || onlyReturned(V))
    return false;
  Type *Ty = V->getType();
  if (Ty->isAggregateType()) {
    bool Changed = false;
    SmallVector<ExtractValueInst *, 4> Parts;
    for (User *U : V->users())
      if (auto *Part = dyn_cast<ExtractValueInst>(U))
        Parts.push_back(Part);
    for (ExtractValueInst *Part : Parts)
      Changed |= readFirstLane(Part, std::next(Part->getIterator()));
    return Changed;
  }
  if (!isLaneOpType(Ty))
    return false;
  IRBuilder<> Builder(Where->getParent(), Where);
  Value *First =
      Builder.CreateIntrinsic(Intrinsic::amdgcn_readfirstlane, {Ty}, {V});
  V->replaceUsesWithIf(First, [First](Use &U) { return U.getUser() != First; });
  return true;
}

bool runSingleLane(Function &F, const TargetTransformInfo &TTI) {
  if (!F.hasFnAttribute("amdgpu-single-lane") || F.isDeclaration() ||
      AMDGPU::isEntryFunctionCC(F.getCallingConv()))
    return false;
  SmallVector<Instruction *> Sources;
  for (Instruction &I : instructions(F))
    if (!I.getType()->isVoidTy() && !isa<InvokeInst>(I) &&
        TTI.getValueUniformity(&I) == ValueUniformity::NeverUniform)
      Sources.push_back(&I);
  bool Changed = false;
  BasicBlock::iterator Entry = F.getEntryBlock().getFirstInsertionPt();
  for (Argument &A : F.args())
    if (TTI.getValueUniformity(&A) == ValueUniformity::NeverUniform)
      Changed |= readFirstLane(&A, Entry);
  for (Instruction *I : Sources) {
    BasicBlock::iterator Where = isa<PHINode>(I)
                                     ? I->getParent()->getFirstInsertionPt()
                                     : std::next(I->getIterator());
    Changed |= readFirstLane(I, Where);
  }
  return Changed;
}

class AMDGPUSingleLaneLegacy : public FunctionPass {
public:
  static char ID;
  AMDGPUSingleLaneLegacy() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override {
    return runSingleLane(
        F, getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F));
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<TargetTransformInfoWrapperPass>();
  }

  StringRef getPassName() const override { return "AMDGPU Single Lane"; }
};

} // namespace

PreservedAnalyses AMDGPUSingleLanePass::run(Function &F,
                                            FunctionAnalysisManager &AM) {
  if (!runSingleLane(F, AM.getResult<TargetIRAnalysis>(F)))
    return PreservedAnalyses::all();
  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}

char AMDGPUSingleLaneLegacy::ID = 0;
char &llvm::AMDGPUSingleLaneLegacyPassID = AMDGPUSingleLaneLegacy::ID;

INITIALIZE_PASS_BEGIN(AMDGPUSingleLaneLegacy, DEBUG_TYPE, "AMDGPU Single Lane",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_END(AMDGPUSingleLaneLegacy, DEBUG_TYPE, "AMDGPU Single Lane",
                    false, false)

FunctionPass *llvm::createAMDGPUSingleLaneLegacyPass() {
  return new AMDGPUSingleLaneLegacy();
}
