//===-- lib/CodeGen/RecordPhis.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RecordPhis.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Transforms/Utils/Local.h"

namespace merkurius::codegen {

namespace {

// Whether `type` is a struct or array whose fields are numbers or vectors of
// them, or such aggregates: a flat record, which holds no pointer that the
// collector must see.
bool IsFlatAggregate(llvm::Type *type) {
  if (!type->isAggregateType()) {
    return false;
  }
  for (unsigned i{0}, n{type->getNumContainedTypes()}; i != n; ++i) {
    llvm::Type *field{type->getContainedType(i)};
    if (field->isAggregateType()
            ? !IsFlatAggregate(field)
            : !field->isIntOrIntVectorTy() && !field->isFPOrFPVectorTy()) {
      return false;
    }
  }
  return true;
}

unsigned FieldCount(llvm::Type *type) {
  if (auto *structure{llvm::dyn_cast<llvm::StructType>(type)}) {
    return structure->getNumElements();
  }
  return llvm::cast<llvm::ArrayType>(type)->getNumElements();
}

llvm::Type *FieldType(llvm::Type *type, unsigned index) {
  if (auto *structure{llvm::dyn_cast<llvm::StructType>(type)}) {
    return structure->getElementType(index);
  }
  return llvm::cast<llvm::ArrayType>(type)->getElementType();
}

// Field `index` of `record`, read before `before`: the value inserted there
// if `record` is built by insertvalues or is a constant, else an
// extractvalue.
llvm::Value *Field(
    llvm::Value *record, unsigned index, llvm::Instruction *before) {
  if (llvm::Value * found{llvm::FindInsertedValue(record, {index})}) {
    return found;
  }
  return llvm::ExtractValueInst::Create(record, {index},
      record->getName() + ".f" + llvm::Twine(index), before->getIterator());
}

// Replaces `record`'s uses by `fields`: an extractvalue of field i by field
// i, or what it reads from it; any other use by the record built again from
// the fields before `before`, where every use can see it.
void ReplaceRecord(llvm::Instruction *record,
    llvm::ArrayRef<llvm::Value *> fields, llvm::Instruction *before) {
  llvm::Value *rebuilt{nullptr};
  for (llvm::Use &use : llvm::make_early_inc_range(record->uses())) {
    auto *extract{llvm::dyn_cast<llvm::ExtractValueInst>(use.getUser())};
    if (extract && extract->getAggregateOperand() == record) {
      llvm::Value *field{fields[extract->getIndices().front()]};
      if (extract->getNumIndices() > 1) {
        field = llvm::ExtractValueInst::Create(field,
            extract->getIndices().drop_front(), extract->getName(),
            extract->getIterator());
      }
      extract->replaceAllUsesWith(field);
      extract->eraseFromParent();
      continue;
    }
    if (!rebuilt) {
      llvm::IRBuilder<> builder{before};
      rebuilt = llvm::PoisonValue::get(record->getType());
      for (auto [index, field] : llvm::enumerate(fields)) {
        rebuilt = builder.CreateInsertValue(
            rebuilt, field, {static_cast<unsigned>(index)});
      }
    }
    use.set(rebuilt);
  }
}

// Splits `phi` into a phi of each field. A field that is a record itself
// becomes a phi that `work` splits in turn.
void SplitPhi(
    llvm::PHINode *phi, llvm::SmallVectorImpl<llvm::Instruction *> &work) {
  llvm::Type *type{phi->getType()};
  llvm::SmallVector<llvm::Value *> fields;
  for (unsigned index{0}, count{FieldCount(type)}; index != count; ++index) {
    auto *field{llvm::PHINode::Create(FieldType(type, index),
        phi->getNumIncomingValues(), phi->getName() + ".f" + llvm::Twine(index),
        phi->getIterator())};
    for (unsigned in{0}, n{phi->getNumIncomingValues()}; in != n; ++in) {
      llvm::BasicBlock *from{phi->getIncomingBlock(in)};
      field->addIncoming(
          Field(phi->getIncomingValue(in), index, from->getTerminator()), from);
    }
    if (field->getType()->isAggregateType()) {
      work.push_back(field);
    }
    fields.push_back(field);
  }
  // Every use goes, a phi's own around a loop too.
  ReplaceRecord(phi, fields, &*phi->getParent()->getFirstNonPHIIt());
  phi->eraseFromParent();
}

// Splits `select` into a select of each field.
void SplitSelect(llvm::SelectInst *select,
    llvm::SmallVectorImpl<llvm::Instruction *> &work) {
  llvm::Type *type{select->getType()};
  llvm::SmallVector<llvm::Value *> fields;
  for (unsigned index{0}, count{FieldCount(type)}; index != count; ++index) {
    llvm::Value *field{llvm::SelectInst::Create(select->getCondition(),
        Field(select->getTrueValue(), index, select),
        Field(select->getFalseValue(), index, select),
        select->getName() + ".f" + llvm::Twine(index), select->getIterator(),
        select)};
    if (auto *made{llvm::dyn_cast<llvm::SelectInst>(field)};
        made && made->getType()->isAggregateType()) {
      work.push_back(made);
    }
    fields.push_back(field);
  }
  ReplaceRecord(select, fields, select);
  select->eraseFromParent();
}

// Whether `instruction` is a phi of a value that its block's terminator
// makes, as an invoke's result, which no field can be read from before the
// terminator.
bool ReadsTerminator(llvm::Instruction &instruction) {
  auto *phi{llvm::dyn_cast<llvm::PHINode>(&instruction)};
  return phi && llvm::any_of(phi->incoming_values(), [](llvm::Value *value) {
    auto *made{llvm::dyn_cast<llvm::Instruction>(value)};
    return made && made->isTerminator();
  });
}

} // namespace

llvm::PreservedAnalyses ScalarizeRecordPhisPass::run(
    llvm::Function &function, llvm::FunctionAnalysisManager &) {
  llvm::SmallVector<llvm::Instruction *> work;
  for (llvm::BasicBlock &block : function) {
    for (llvm::Instruction &instruction : block) {
      if ((llvm::isa<llvm::PHINode>(instruction) ||
              llvm::isa<llvm::SelectInst>(instruction)) &&
          IsFlatAggregate(instruction.getType()) &&
          !ReadsTerminator(instruction)) {
        work.push_back(&instruction);
      }
    }
  }
  if (work.empty()) {
    return llvm::PreservedAnalyses::all();
  }
  while (!work.empty()) {
    llvm::Instruction *next{work.pop_back_val()};
    if (auto *phi{llvm::dyn_cast<llvm::PHINode>(next)}) {
      SplitPhi(phi, work);
    } else {
      SplitSelect(llvm::cast<llvm::SelectInst>(next), work);
    }
  }
  // The insertvalues that only the split phis read, and the records built
  // again that nothing reads.
  llvm::SmallVector<llvm::WeakTrackingVH> dead;
  for (llvm::Instruction &instruction : llvm::instructions(function)) {
    if (llvm::isa<llvm::InsertValueInst, llvm::ExtractValueInst>(instruction)) {
      dead.push_back(&instruction);
    }
  }
  llvm::RecursivelyDeleteTriviallyDeadInstructionsPermissive(dead);
  llvm::PreservedAnalyses preserved;
  preserved.preserveSet<llvm::CFGAnalyses>();
  return preserved;
}

} // namespace merkurius::codegen
