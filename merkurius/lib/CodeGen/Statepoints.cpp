//===-- lib/CodeGen/Statepoints.cpp - Code a collector can scan -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// After the optimizer, gives Mercury code statepoints and stack maps, with each
// root's type (docs/INPROGRESS.md, Statepoints and stack maps).
//
//===----------------------------------------------------------------------===//

#include "merkurius/CodeGen/Lowering.h"
#include "merkurius/CodeGen/TargetInfo.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/BuiltinGCs.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Statepoint.h"
#include "llvm/IR/ValueMap.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Transforms/Scalar/RewriteStatepointsForGC.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

namespace merkurius::codegen {

static constexpr unsigned kHeapAddressSpace{1};

// The statepoint rewrite does not take an aggregate holding a heap pointer, and
// a build without assertions would not follow the pointer. A call returns one
// when it has several outputs or can fail. No safepoint comes between the
// callee's return and the caller taking the fields apart, so such a pointer can
// cross as an integer: a function returns it as i64, and the caller gets it
// back with an inttoptr, which the rewrite takes as a base.

static bool IsHeapPointer(llvm::Type *type) {
  auto *pointer{llvm::dyn_cast<llvm::PointerType>(type)};
  return pointer && pointer->getAddressSpace() == kHeapAddressSpace;
}

static bool HoldsHeapPointer(llvm::Type *type) {
  auto *structure{llvm::dyn_cast<llvm::StructType>(type)};
  return structure && llvm::any_of(structure->elements(), IsHeapPointer);
}

static llvm::Type *WithoutHeapPointers(llvm::Type *type) {
  auto *structure{llvm::cast<llvm::StructType>(type)};
  llvm::SmallVector<llvm::Type *> fields;
  for (llvm::Type *field : structure->elements()) {
    fields.push_back(IsHeapPointer(field)
            ? llvm::Type::getInt64Ty(type->getContext())
            : field);
  }
  return llvm::StructType::get(type->getContext(), fields);
}

static llvm::FunctionType *WithoutHeapPointers(llvm::FunctionType *type) {
  return llvm::FunctionType::get(WithoutHeapPointers(type->getReturnType()),
      type->params(), type->isVarArg());
}

namespace {

// Replaces the aggregates of one function by their fields.
class Flattener {
public:
  explicit Flattener(llvm::Function &function) : function_{function} {}
  void Run();

private:
  using Fields = llvm::SmallVector<llvm::Value *, 4>;
  void Call(llvm::CallBase *);
  void Return(llvm::ReturnInst *);
  llvm::Value *Field(llvm::Value *aggregate, unsigned index);
  Fields All(llvm::Value *aggregate);
  bool Visit(llvm::Instruction &);

  llvm::Function &function_;
  llvm::DenseMap<llvm::Value *, Fields> fields_;
  llvm::SmallVector<llvm::Instruction *> dead_;
};

} // namespace

llvm::Value *Flattener::Field(llvm::Value *aggregate, unsigned index) {
  if (auto *constant{llvm::dyn_cast<llvm::Constant>(aggregate)}) {
    return constant->getAggregateElement(index);
  }
  auto it{fields_.find(aggregate)};
  if (it == fields_.end()) {
    llvm::report_fatal_error(llvm::Twine{"not yet implemented: an aggregate "
                                         "holding a heap pointer in "} +
        function_.getName());
  }
  return it->second[index];
}

Flattener::Fields Flattener::All(llvm::Value *aggregate) {
  Fields all;
  unsigned count{aggregate->getType()->getStructNumElements()};
  for (unsigned i = 0; i < count; ++i) {
    all.push_back(Field(aggregate, i));
  }
  return all;
}

void Flattener::Call(llvm::CallBase *call) {
  llvm::Type *old{call->getType()};
  call->mutateFunctionType(WithoutHeapPointers(call->getFunctionType()));
  auto *direct{llvm::dyn_cast<llvm::CallInst>(call)};
  if (direct && direct->isMustTailCall()) {
    return;
  }
  llvm::IRBuilder<> builder{call->getNextNode()};
  Fields &fields{fields_[call]};
  for (unsigned i = 0; i < old->getStructNumElements(); ++i) {
    llvm::Value *field{builder.CreateExtractValue(call, i)};
    if (IsHeapPointer(old->getStructElementType(i))) {
      field = builder.CreateIntToPtr(field, old->getStructElementType(i));
    }
    fields.push_back(field);
  }
}

void Flattener::Return(llvm::ReturnInst *ret) {
  auto *call{llvm::dyn_cast_or_null<llvm::CallInst>(ret->getReturnValue())};
  if (call && call->isMustTailCall()) {
    return;
  }
  llvm::IRBuilder<> builder{ret};
  llvm::Value *result{llvm::PoisonValue::get(function_.getReturnType())};
  Fields fields{All(ret->getReturnValue())};
  for (auto [i, field] : llvm::enumerate(fields)) {
    if (IsHeapPointer(field->getType())) {
      field = builder.CreatePtrToInt(field, builder.getInt64Ty());
    }
    result = builder.CreateInsertValue(result, field, i);
  }
  ret->setOperand(0, result);
}

bool Flattener::Visit(llvm::Instruction &instruction) {
  if (auto *call{llvm::dyn_cast<llvm::CallBase>(&instruction)}) {
    if (HoldsHeapPointer(call->getType())) {
      Call(call);
    }
    return true;
  }
  if (auto *ret{llvm::dyn_cast<llvm::ReturnInst>(&instruction)}) {
    if (ret->getReturnValue() &&
        function_.getReturnType() != ret->getReturnValue()->getType()) {
      Return(ret);
    }
    return true;
  }
  if (auto *extract{llvm::dyn_cast<llvm::ExtractValueInst>(&instruction)}) {
    if (fields_.contains(extract->getAggregateOperand()) ||
        HoldsHeapPointer(extract->getAggregateOperand()->getType())) {
      if (!llvm::is_contained(fields_[extract->getAggregateOperand()], extract)) {
        extract->replaceAllUsesWith(
            Field(extract->getAggregateOperand(), extract->getIndices()[0]));
        dead_.push_back(extract);
      }
    }
    return true;
  }
  if (!HoldsHeapPointer(instruction.getType())) {
    return true;
  }
  if (auto *insert{llvm::dyn_cast<llvm::InsertValueInst>(&instruction)}) {
    Fields fields{All(insert->getAggregateOperand())};
    fields[insert->getIndices()[0]] = insert->getInsertedValueOperand();
    fields_[insert] = fields;
  } else if (auto *select{llvm::dyn_cast<llvm::SelectInst>(&instruction)}) {
    llvm::IRBuilder<> builder{select};
    Fields thens{All(select->getTrueValue())};
    Fields elses{All(select->getFalseValue())};
    Fields fields;
    for (auto [one, other] : llvm::zip(thens, elses)) {
      fields.push_back(builder.CreateSelect(select->getCondition(), one, other));
    }
    fields_[select] = fields;
  } else if (!llvm::isa<llvm::PHINode>(instruction)) {
    return false;
  }
  dead_.push_back(&instruction);
  return true;
}

void Flattener::Run() {
  llvm::removeUnreachableBlocks(function_);
  // Phis first, as their operands can come after them.
  llvm::SmallVector<llvm::PHINode *> phis;
  for (llvm::Instruction &instruction : llvm::instructions(function_)) {
    auto *phi{llvm::dyn_cast<llvm::PHINode>(&instruction)};
    if (phi && HoldsHeapPointer(phi->getType())) {
      phis.push_back(phi);
    }
  }
  for (llvm::PHINode *phi : phis) {
    Fields fields;
    for (llvm::Type *type : phi->getType()->subtypes()) {
      fields.push_back(llvm::PHINode::Create(
          type, phi->getNumIncomingValues(), "", phi->getIterator()));
    }
    fields_[phi] = fields;
  }
  llvm::ReversePostOrderTraversal<llvm::Function *> order{&function_};
  for (llvm::BasicBlock *block : order) {
    for (llvm::Instruction &instruction :
        llvm::make_early_inc_range(*block)) {
      if (!Visit(instruction)) {
        std::string text;
        llvm::raw_string_ostream os{text};
        os << "not yet implemented: an aggregate holding a heap pointer:"
           << instruction;
        llvm::report_fatal_error(llvm::StringRef{text});
      }
    }
  }
  for (llvm::PHINode *phi : phis) {
    Fields fields{fields_[phi]};
    for (unsigned i = 0; i < fields.size(); ++i) {
      for (unsigned edge = 0; edge < phi->getNumIncomingValues(); ++edge) {
        llvm::cast<llvm::PHINode>(fields[i])->addIncoming(
            Field(phi->getIncomingValue(edge), i), phi->getIncomingBlock(edge));
      }
    }
  }
  for (llvm::Instruction *instruction : dead_) {
    instruction->dropAllReferences();
  }
  for (llvm::Instruction *instruction : dead_) {
    instruction->eraseFromParent();
  }
}

static void FlattenResults(llvm::Module &module) {
  llvm::SmallVector<llvm::Function *> changing;
  for (llvm::Function &function : module) {
    if (HoldsHeapPointer(function.getReturnType())) {
      changing.push_back(&function);
    }
  }
  for (llvm::Function *function : changing) {
    llvm::Function *replacement{llvm::Function::Create(
        WithoutHeapPointers(function->getFunctionType()),
        function->getLinkage(), function->getAddressSpace(), "", &module)};
    replacement->copyAttributesFrom(function);
    replacement->copyMetadata(function, 0);
    replacement->takeName(function);
    replacement->splice(replacement->begin(), function);
    for (auto [from, to] : llvm::zip(function->args(), replacement->args())) {
      from.replaceAllUsesWith(&to);
      to.takeName(&from);
    }
    function->replaceAllUsesWith(replacement);
    function->eraseFromParent();
  }
  for (llvm::Function &function : module) {
    if (!function.isDeclaration()) {
      Flattener{function}.Run();
    }
  }
}

// The value a pointer is made from by tagging or by a field's offset.
static const llvm::Value *CellOf(const llvm::Value *value) {
  for (;;) {
    if (auto *gep{llvm::dyn_cast<llvm::GEPOperator>(value)}) {
      value = gep->getPointerOperand();
    } else if (auto *cast{llvm::dyn_cast<llvm::BitCastOperator>(value)}) {
      value = cast->getOperand(0);
    } else {
      return value;
    }
  }
}

// In a function the collector scans: a call is a safepoint if it has a bundle
// and is not a tail call, which leaves no frame to scan. Any other is a leaf,
// which is why this is after the optimizer: an inlined musttail call is an
// ordinary call with its caller's frame under it.
static void MarkLeaves(llvm::Function &function) {
  for (llvm::Instruction &instruction :
      llvm::make_early_inc_range(llvm::instructions(function))) {
    auto *call{llvm::dyn_cast<llvm::CallBase>(&instruction)};
    if (!call) {
      continue;
    }
    auto deopt{call->getOperandBundle(llvm::LLVMContext::OB_deopt)};
    bool bundle{deopt.has_value()};
    auto *first{deopt && !deopt->Inputs.empty()
            ? llvm::dyn_cast<llvm::ConstantInt>(deopt->Inputs[0])
            : nullptr};
    bool inRegion{first && first->getSExtValue() == kInRegion};
    auto *direct{llvm::dyn_cast<llvm::CallInst>(call)};
    if (inRegion || (direct && direct->isMustTailCall())) {
      if (bundle) {
        llvm::CallBase *plain{llvm::CallBase::removeOperandBundle(
            call, llvm::LLVMContext::OB_deopt, call->getIterator())};
        call->replaceAllUsesWith(plain);
        call->eraseFromParent();
        call = plain;
      }
      call->addFnAttr(
          llvm::Attribute::get(call->getContext(), "gc-leaf-function"));
    } else if (!bundle) {
      if (call->getCallingConv() == llvm::CallingConv::Tail) {
        std::string text;
        llvm::raw_string_ostream os{text};
        os << "a call of Mercury code in " << function.getName()
           << " has no list of what is live across it:" << *call;
        llvm::report_fatal_error(llvm::StringRef{text});
      }
      call->addFnAttr(
          llvm::Attribute::get(call->getContext(), "gc-leaf-function"));
    }
  }
}

// Where a value's bits can be read just after it is defined.
static llvm::BasicBlock::iterator AfterDefinition(llvm::Value *value,
    llvm::Function &function) {
  if (auto *phi{llvm::dyn_cast<llvm::PHINode>(value)}) {
    return phi->getParent()->getFirstInsertionPt();
  }
  if (auto *instruction{llvm::dyn_cast<llvm::Instruction>(value)}) {
    return std::next(instruction->getIterator());
  }
  return function.getEntryBlock().getFirstInsertionPt();
}

// Calls `visit` with each entry of a bundle: its type's index, its value, and
// the type_infos after it, whose number is above the index.
template <typename Visit>
static void ForEachEntry(llvm::ArrayRef<llvm::Use> inputs, Visit visit) {
  for (std::size_t i = 0; i + 1 < inputs.size();) {
    auto *header{llvm::cast<llvm::ConstantInt>(inputs[i].get())};
    std::uint64_t bits{header->getZExtValue()};
    std::size_t count{static_cast<std::size_t>(bits >> 32)};
    visit(llvm::ConstantInt::get(header->getType(), bits & 0xffffffffu),
        inputs[i + 1].get(), inputs.slice(i + 2, count));
    i += 2 + count;
  }
}

namespace {
// Each value's type, as far as the bundles say, kept up to date through the
// rewrite's replacements.
using KnownTypes = llvm::ValueMap<const llvm::Value *, llvm::Constant *>;
} // namespace

// Records the bundles' types, for all the program points. The optimizer can
// have changed a later use from a tagged pointer to the untagged cell, or to
// a phi over them, which the bundle at the safepoint does not name. It is the
// same object. A phi's type is its operands'; they have the same layout, which
// is all the collector asks of the type.
static void RecordTypes(llvm::Function &function, KnownTypes &known) {
  llvm::SmallVector<const llvm::Value *> worklist;
  for (llvm::Instruction &instruction : llvm::instructions(function)) {
    auto *call{llvm::dyn_cast<llvm::CallBase>(&instruction)};
    auto deopt{call ? call->getOperandBundle(llvm::LLVMContext::OB_deopt)
                    : std::nullopt};
    if (!deopt) {
      continue;
    }
    ForEachEntry(deopt->Inputs,
        [&](llvm::Constant *type, llvm::Value *entry, llvm::ArrayRef<llvm::Use>) {
          const llvm::Value *value{CellOf(entry)};
          if (IsHeapPointer(value->getType()) &&
              !llvm::isa<llvm::Constant>(value) &&
              known.insert({value, type}).second) {
            worklist.push_back(value);
          }
        });
  }
  while (!worklist.empty()) {
    const llvm::Value *value{worklist.pop_back_val()};
    if (!llvm::isa<llvm::PHINode, llvm::SelectInst>(value)) {
      continue;
    }
    llvm::Constant *type{known.lookup(value)};
    for (const llvm::Use &use : llvm::cast<llvm::User>(value)->operands()) {
      const llvm::Value *operand{CellOf(use.get())};
      if (operand->getType()->isPointerTy() &&
          !llvm::isa<llvm::Constant>(operand) &&
          known.insert({operand, type}).second) {
        worklist.push_back(operand);
      }
    }
  }
}

// A clone of `chain`'s `last`, and of what it is computed from, before `place`.
static llvm::Value *CloneChain(
    llvm::Instruction *last, llvm::Instruction *place,
    const llvm::SetVector<llvm::Instruction *> &chain) {
  llvm::Instruction *clone{last->clone()};
  clone->insertBefore(place->getIterator());
  auto *parent{llvm::dyn_cast<llvm::Instruction>(last->getOperand(0))};
  if (parent && chain.contains(parent)) {
    clone->setOperand(0, CloneChain(parent, clone, chain));
  }
  return clone;
}

// A C pointer made from a heap pointer is not a root. The optimizer hoists one
// out of a loop, past safepoints, and its use then has the address from before
// a collection. Making it again at each use, before statepoints are made, has
// it use the relocated heap pointer, as GEPs on it do.
static void CastAtUses(llvm::Function &function) {
  llvm::SmallVector<llvm::AddrSpaceCastInst *> casts;
  for (llvm::Instruction &instruction : llvm::instructions(function)) {
    auto *cast{llvm::dyn_cast<llvm::AddrSpaceCastInst>(&instruction)};
    if (cast && IsHeapPointer(cast->getSrcTy())) {
      casts.push_back(cast);
    }
  }
  for (llvm::AddrSpaceCastInst *cast : casts) {
    llvm::SetVector<llvm::Instruction *> chain;
    chain.insert(cast);
    for (std::size_t i{0}; i < chain.size(); ++i) {
      for (llvm::User *user : chain[i]->users()) {
        auto *gep{llvm::dyn_cast<llvm::GetElementPtrInst>(user)};
        if (gep && gep->getPointerOperand() == chain[i]) {
          chain.insert(gep);
        }
      }
    }
    for (llvm::Instruction *link : chain) {
      for (llvm::Use &use : llvm::make_early_inc_range(link->uses())) {
        auto *user{llvm::cast<llvm::Instruction>(use.getUser())};
        if (chain.contains(user)) {
          continue;
        }
        if (llvm::isa<llvm::PHINode>(user)) {
          llvm::report_fatal_error(llvm::Twine{
              "not yet implemented: a C pointer from a heap pointer that "
              "goes through a phi, in "} + function.getName());
        }
        use.set(CloneChain(link, user, chain));
      }
    }
    for (llvm::Instruction *link : llvm::reverse(chain)) {
      link->eraseFromParent();
    }
  }
}

// A comparison with a constant needs only the value's bits, and a move does not
// change what it says: tags and being non-null stay, and static data does not
// move. The optimizer folds `(ptrtoint x) < 2` to a comparison of pointers, and
// sinks a ptrtoint towards its use, past safepoints. Either keeps an integer
// word `x`, which merk stopped tracking at its from_word, live as if it were a
// heap pointer, and no bundle lists it. Reading the bits where `x` is defined
// leaves it dead after that.
static void ReadBitsEarly(llvm::Function &function, KnownTypes &known) {
  llvm::SmallVector<llvm::ICmpInst *> compares;
  llvm::SmallVector<llvm::PtrToIntInst *> reads;
  for (llvm::Instruction &instruction : llvm::instructions(function)) {
    auto *compare{llvm::dyn_cast<llvm::ICmpInst>(&instruction)};
    if (compare && IsHeapPointer(compare->getOperand(0)->getType())) {
      compares.push_back(compare);
    }
    auto *read{llvm::dyn_cast<llvm::PtrToIntInst>(&instruction)};
    if (read && IsHeapPointer(read->getOperand(0)->getType()) &&
        !llvm::isa<llvm::Constant>(read->getOperand(0)) &&
        !known.count(CellOf(read->getOperand(0)))) {
      reads.push_back(read);
    }
  }
  for (llvm::PtrToIntInst *read : reads) {
    read->moveBefore(AfterDefinition(read->getOperand(0), function));
  }
  llvm::DenseMap<llvm::Value *, llvm::Value *> bits;
  llvm::Type *i64{llvm::Type::getInt64Ty(function.getContext())};
  for (llvm::ICmpInst *compare : compares) {
    llvm::Value *left{compare->getOperand(0)};
    llvm::Value *right{compare->getOperand(1)};
    auto *leftConstant{llvm::dyn_cast<llvm::Constant>(left)};
    auto *rightConstant{llvm::dyn_cast<llvm::Constant>(right)};
    bool leftKnown{known.count(CellOf(left)) != 0};
    bool rightKnown{known.count(CellOf(right)) != 0};
    // Two pointers the bundles list stay as they are: the rewrite relocates
    // both, and comparing them needs both current.
    if ((!leftConstant && !rightConstant && leftKnown && rightKnown) ||
        (leftConstant && rightConstant)) {
      continue;
    }
    for (unsigned side = 0; side < 2; ++side) {
      llvm::Value *value{compare->getOperand(side)};
      if (auto *constant{llvm::dyn_cast<llvm::Constant>(value)}) {
        compare->setOperand(side, llvm::ConstantExpr::getPtrToInt(constant, i64));
        continue;
      }
      llvm::Value *&read{bits[value]};
      if (!read) {
        read = new llvm::PtrToIntInst{
            value, i64, "", AfterDefinition(value, function)};
      }
      compare->setOperand(side, read);
    }
  }
}

// What the bundles said of a value the rewrite made, from what it was made of.
static llvm::Constant *FindType(const llvm::Value *root, KnownTypes &known) {
  llvm::SmallPtrSet<const llvm::Value *, 16> seen;
  llvm::SmallVector<const llvm::Value *> worklist{root};
  while (!worklist.empty()) {
    const llvm::Value *value{CellOf(worklist.pop_back_val())};
    if (!seen.insert(value).second) {
      continue;
    }
    if (llvm::Constant *type{known.lookup(value)}) {
      return type;
    }
    if (auto *relocate{llvm::dyn_cast<llvm::GCRelocateInst>(value)}) {
      worklist.push_back(relocate->getDerivedPtr());
    } else if (llvm::isa<llvm::PHINode, llvm::SelectInst>(value)) {
      for (const llvm::Use &use : llvm::cast<llvm::User>(value)->operands()) {
        if (use->getType()->isPointerTy()) {
          worklist.push_back(use.get());
        }
      }
    }
  }
  return nullptr;
}

namespace {
using TypeInfosOf =
    llvm::DenseMap<const llvm::Value *, llvm::SmallVector<llvm::Value *, 2>>;
using ValueSet = llvm::DenseSet<llvm::Value *>;
} // namespace

static bool IsFromStaticData(const llvm::Value *root);
static bool IsPolymorphicIndex(llvm::Module &module, llvm::ConstantInt *type);

// Whether a value is one the collector has to find, at a safepoint after it.
static bool IsLiveRoot(llvm::Value *value) {
  return IsHeapPointer(value->getType()) &&
      (llvm::isa<llvm::Instruction>(value) ||
          llvm::isa<llvm::Argument>(value)) &&
      !IsFromStaticData(value);
}

// Applies an instruction to what is live before it, from what is after it.
static void StepBack(llvm::Instruction &instruction, ValueSet &live) {
  live.erase(&instruction);
  if (llvm::isa<llvm::PHINode>(instruction)) {
    return;
  }
  for (llvm::Value *operand : instruction.operands()) {
    if (IsLiveRoot(operand)) {
      live.insert(operand);
    }
  }
}

// What is live at the end of `block`, given what is live into each successor.
static ValueSet LiveOut(llvm::BasicBlock &block,
    const llvm::DenseMap<llvm::BasicBlock *, ValueSet> &liveIn) {
  ValueSet live;
  for (llvm::BasicBlock *successor : llvm::successors(&block)) {
    auto it{liveIn.find(successor)};
    if (it != liveIn.end()) {
      live.insert(it->second.begin(), it->second.end());
    }
    for (llvm::PHINode &phi : successor->phis()) {
      live.erase(&phi);
    }
    for (llvm::PHINode &phi : successor->phis()) {
      llvm::Value *incoming{phi.getIncomingValueForBlock(&block)};
      if (IsLiveRoot(incoming)) {
        live.insert(incoming);
      }
    }
  }
  return live;
}

// Makes a phi or select of each of the type_infos of what `merge` chooses
// between, if each has them where the choice is made.
static bool MergeTypeInfos(llvm::Instruction *merge, TypeInfosOf &infos,
    llvm::DominatorTree &dominators, TypeInfosOf &made) {
  auto *phi{llvm::dyn_cast<llvm::PHINode>(merge)};
  auto *select{llvm::dyn_cast<llvm::SelectInst>(merge)};
  unsigned count{phi ? phi->getNumIncomingValues() : 2};
  llvm::SmallVector<const llvm::SmallVector<llvm::Value *, 2> *> incoming(count);
  std::size_t width{0};
  for (unsigned i{0}; i < count; ++i) {
    llvm::Value *value{phi ? phi->getIncomingValue(i) : select->getOperand(i + 1)};
    if (!IsLiveRoot(value)) {
      continue; // A constant takes on the others'.
    }
    auto it{infos.find(CellOf(value))};
    if (CellOf(value) == CellOf(merge) || it == infos.end() ||
        (width && width != it->second.size())) {
      return false;
    }
    width = it->second.size();
    incoming[i] = &it->second;
    llvm::Instruction *where{
        phi ? phi->getIncomingBlock(i)->getTerminator() : merge};
    for (llvm::Value *info : it->second) {
      auto *defined{llvm::dyn_cast<llvm::Instruction>(info)};
      if (defined && !dominators.dominates(defined, where)) {
        return false;
      }
    }
  }
  if (!width) {
    return false;
  }
  llvm::SmallVector<llvm::Value *, 2> merged;
  for (std::size_t w{0}; w < width; ++w) {
    // What a branch with a constant passes, which has no type to describe.
    llvm::Value *any{nullptr};
    for (const auto *from : incoming) {
      any = from ? (*from)[w] : any;
    }
    llvm::Type *type{any->getType()};
    if (phi) {
      auto *infoPhi{llvm::PHINode::Create(
          type, count, "type_info", phi->getParent()->begin())};
      for (unsigned i{0}; i < count; ++i) {
        infoPhi->addIncoming(
            incoming[i] ? (*incoming[i])[w] : any, phi->getIncomingBlock(i));
      }
      merged.push_back(infoPhi);
    } else {
      merged.push_back(llvm::SelectInst::Create(select->getCondition(),
          incoming[0] ? (*incoming[0])[w] : any,
          incoming[1] ? (*incoming[1])[w] : any, "type_info",
          select->getIterator()));
    }
  }
  infos[CellOf(merge)] = merged;
  made[CellOf(merge)] = std::move(merged);
  return true;
}

// The type_infos each value of the function has in some bundle, and a phi or
// select of values that have the same ones.
static TypeInfosOf GatherTypeInfos(
    llvm::Function &function, llvm::DominatorTree &dominators, TypeInfosOf &made) {
  TypeInfosOf infos{made};
  llvm::SmallVector<llvm::Instruction *> merges;
  for (llvm::Instruction &instruction : llvm::instructions(function)) {
    if (auto *call{llvm::dyn_cast<llvm::CallBase>(&instruction)}) {
      if (auto deopt{call->getOperandBundle(llvm::LLVMContext::OB_deopt)}) {
        ForEachEntry(deopt->Inputs,
            [&](llvm::Constant *, llvm::Value *value,
                llvm::ArrayRef<llvm::Use> typeInfos) {
              if (!typeInfos.empty()) {
                auto &found{infos[CellOf(value)]};
                found.clear();
                for (const llvm::Use &info : typeInfos) {
                  found.push_back(info.get());
                }
              }
            });
      }
    }
    if (IsHeapPointer(instruction.getType()) &&
        (llvm::isa<llvm::PHINode>(instruction) ||
            llvm::isa<llvm::SelectInst>(instruction))) {
      merges.push_back(&instruction);
    }
  }
  for (bool changed{true}; changed;) {
    changed = false;
    for (llvm::Instruction *merge : merges) {
      if (infos.count(CellOf(merge))) {
        continue;
      }
      const llvm::SmallVector<llvm::Value *, 2> *common{nullptr};
      bool same{true};
      unsigned first{llvm::isa<llvm::SelectInst>(merge) ? 1u : 0u};
      for (unsigned i{first}; i < merge->getNumOperands() && same; ++i) {
        llvm::Value *incoming{merge->getOperand(i)};
        if (!IsLiveRoot(incoming) || CellOf(incoming) == CellOf(merge)) {
          continue; // A constant has no type_infos to agree with.
        }
        auto it{infos.find(CellOf(incoming))};
        same = it != infos.end() && (!common || *common == it->second);
        if (same) {
          common = &it->second;
        }
      }
      if (same && common) {
        llvm::SmallVector<llvm::Value *, 2> copy{*common};
        infos[CellOf(merge)] = std::move(copy);
        changed = true;
      }
    }
    if (!changed) {
      // Those whose branches have different type_infos, as code that two
      // existentially typed constructors share: merge the type_infos too.
      for (llvm::Instruction *merge : merges) {
        if (!infos.count(CellOf(merge)) &&
            MergeTypeInfos(merge, infos, dominators, made)) {
          changed = true;
          break;
        }
      }
    }
  }
  return infos;
}

// Lists at a safepoint what is live there and its bundle does not have: the
// optimizer can keep a value alive for longer than the Mercury code did, and
// the rewrite would then make a root that nothing describes. Its type_infos
// are found here, while each is still one SSA value, and listing them makes
// the rewrite relocate them. That can make one live at an earlier safepoint.
static void ListLiveRoots(llvm::Function &function, KnownTypes &known) {
  TypeInfosOf made;
  for (bool changed{true}; changed;) {
    changed = false;
    llvm::DenseMap<llvm::BasicBlock *, ValueSet> liveIn;
    for (bool stable{false}; !stable;) {
      stable = true;
      for (llvm::BasicBlock *block : llvm::post_order(&function)) {
        ValueSet live{LiveOut(*block, liveIn)};
        for (llvm::Instruction &instruction : llvm::reverse(*block)) {
          StepBack(instruction, live);
        }
        ValueSet &in{liveIn[block]};
        if (in.size() != live.size()) {
          in = std::move(live);
          stable = false;
        }
      }
    }
    llvm::DominatorTree dominators{function};
    TypeInfosOf infos{GatherTypeInfos(function, dominators, made)};
    for (llvm::BasicBlock &block : function) {
      ValueSet live{LiveOut(block, liveIn)};
      for (llvm::Instruction &instruction :
          llvm::make_early_inc_range(llvm::reverse(block))) {
        auto *call{llvm::dyn_cast<llvm::CallBase>(&instruction)};
        auto deopt{call ? call->getOperandBundle(llvm::LLVMContext::OB_deopt)
                        : std::nullopt};
        if (deopt) {
          ValueSet after{live};
          after.erase(call);
          llvm::SmallPtrSet<const llvm::Value *, 16> listed;
          ForEachEntry(deopt->Inputs,
              [&](llvm::Constant *, llvm::Value *value,
                  llvm::ArrayRef<llvm::Use>) { listed.insert(CellOf(value)); });
          // In a fixed order, for the same code from the same module.
          llvm::SmallVector<llvm::Value *> missing;
          for (llvm::Value *value : after) {
            if (!listed.contains(CellOf(value))) {
              missing.push_back(value);
            }
          }
          std::vector<llvm::Value *> entries{
              deopt->Inputs.begin(), deopt->Inputs.end()};
          llvm::SmallVector<std::pair<std::string, llvm::Value *>> named;
          for (llvm::Value *value : missing) {
            std::string text;
            llvm::raw_string_ostream os{text};
            value->printAsOperand(os, false);
            named.push_back({std::move(text), value});
          }
          llvm::sort(named, [](const auto &a, const auto &b) {
            return a.first < b.first;
          });
          bool added{false};
          for (auto &[text, value] : named) {
            auto *type{llvm::cast_or_null<llvm::ConstantInt>(
                FindType(value, known))};
            if (!type) {
              continue;
            }
            llvm::SmallVector<llvm::Value *, 2> given;
            if (auto it{infos.find(CellOf(value))};
                it != infos.end() &&
                llvm::all_of(it->second, [&](llvm::Value *info) {
                  auto *defined{llvm::dyn_cast<llvm::Instruction>(info)};
                  return !defined || dominators.dominates(defined, call);
                })) {
              given = it->second;
            }
            if (given.empty() && IsPolymorphicIndex(*function.getParent(), type)) {
              std::string text;
              llvm::raw_string_ostream os{text};
              os << "not yet implemented: a garbage collector root whose "
                 << "type_infos are not known, in " << function.getName() << ":"
                 << *value;
              llvm::report_fatal_error(llvm::StringRef{text});
            }
            entries.push_back(llvm::ConstantInt::get(type->getType(),
                type->getZExtValue() | (std::uint64_t{given.size()} << 32)));
            entries.push_back(value);
            entries.insert(entries.end(), given.begin(), given.end());
            added = true;
          }
          if (added) {
            llvm::SmallVector<llvm::OperandBundleDef, 2> bundles;
            for (unsigned b = 0; b < call->getNumOperandBundles(); ++b) {
              llvm::OperandBundleUse use{call->getOperandBundleAt(b)};
              if (use.getTagID() != llvm::LLVMContext::OB_deopt) {
                bundles.emplace_back(use);
              }
            }
            bundles.emplace_back("deopt", std::move(entries));
            llvm::CallBase *replaced{llvm::CallBase::Create(
                call, bundles, call->getIterator())};
            replaced->takeName(call);
            call->replaceAllUsesWith(replaced);
            StepBack(*replaced, live);
            call->eraseFromParent();
            changed = true;
            continue;
          }
        }
        StepBack(instruction, live);
      }
    }
  }
}


// What static data holds is static or a number, and does not move.
static bool IsFromStaticData(const llvm::Value *root) {
  llvm::SmallPtrSet<const llvm::Value *, 16> seen;
  llvm::SmallVector<const llvm::Value *> worklist{root};
  while (!worklist.empty()) {
    const llvm::Value *value{CellOf(worklist.pop_back_val())};
    if (llvm::isa<llvm::Constant>(value) || !seen.insert(value).second) {
      continue;
    }
    if (auto *load{llvm::dyn_cast<llvm::LoadInst>(value)}) {
      if (!llvm::isa<llvm::Constant>(load->getPointerOperand())) {
        return false;
      }
    } else if (auto *relocate{llvm::dyn_cast<llvm::GCRelocateInst>(value)}) {
      worklist.push_back(relocate->getDerivedPtr());
    } else if (llvm::isa<llvm::PHINode>(value)) {
      for (const llvm::Use &use : llvm::cast<llvm::User>(value)->operands()) {
        worklist.push_back(use.get());
      }
    } else {
      return false;
    }
  }
  return true;
}

// Replaces a statepoint's bundle, a type's index before each value, by one
// index for each entry of its gc-live list, so that the stack map says each
// root's type, or -1 for a root the collector can ignore (docs/INPROGRESS.md,
// Statepoints and stack maps).
namespace {
struct StatepointRoots {
  llvm::GCStatepointInst *statepoint{nullptr};
  llvm::DenseMap<const llvm::Value *, llvm::Constant *> types;
  llvm::DenseMap<const llvm::Value *, llvm::SmallVector<llvm::Value *, 2>> infos;
  llvm::SmallMapVector<llvm::Value *, llvm::Value *, 4> slots;
  // The tag of a root that is the untagged cell of a term the bundle has.
  llvm::DenseMap<const llvm::Value *, std::uint64_t> tags;
  // What the bundle has that is dead after the call, so has no relocation.
  // It can still be the only pointer with a type to a cell that is live.
  llvm::SmallVector<llvm::Value *> dead;
};
} // namespace

// Type index of a root that a collection only updates, -1 being one it ignores.
static constexpr std::int64_t kForwardOnly{-2};

static std::int64_t RootKindOf(llvm::Module &module, llvm::Constant *type) {
  auto *table{llvm::cast<llvm::ConstantStruct>(
      module.getNamedGlobal(kRootTable)->getInitializer())};
  auto *index{llvm::cast<llvm::ConstantInt>(type)};
  return llvm::cast<llvm::ConstantInt>(
      table->getOperand(2 + 2 * (index->getZExtValue() & 0xffffffff)))
      ->getSExtValue();
}

static void FindRootTypes(llvm::GCStatepointInst *statepoint, KnownTypes &known,
    StatepointRoots &roots) {
  auto deopt{statepoint->getOperandBundle(llvm::LLVMContext::OB_deopt)};
  auto live{statepoint->getOperandBundle(llvm::LLVMContext::OB_gc_live)};
  llvm::Function &function{*statepoint->getFunction()};
  llvm::DenseMap<const llvm::Value *, llvm::Constant *> types;
  // An environment in the frame: the alloca, which the stack map gives as a
  // Direct location, and its layout's index.
  llvm::SmallMapVector<llvm::Value *, llvm::Value *, 4> slots;
  llvm::DenseMap<const llvm::Value *, llvm::SmallVector<llvm::Value *, 2>> infos;
  // The values the bundle has, and a cell's tag where it has the tagged term.
  llvm::SmallPtrSet<const llvm::Value *, 16> listed;
  llvm::DenseMap<const llvm::Value *, std::uint64_t> tags;
  if (deopt) {
    ForEachEntry(deopt->Inputs,
        [&](llvm::Constant *type, llvm::Value *value,
            llvm::ArrayRef<llvm::Use> typeInfos) {
          if (!IsHeapPointer(value->getType())) {
            if (!llvm::isa<llvm::AllocaInst>(value)) {
              std::string text;
              llvm::raw_string_ostream os{text};
              os << "not yet implemented: an environment that is not an alloca:"
                 << *value;
              llvm::report_fatal_error(llvm::StringRef{text});
            }
            slots.insert({value, type});
            return;
          }
          listed.insert(value);
          // A value listed at two types has the one that has type_infos, and
          // a hole's cell has its term's rather than the reference's.
          auto &found{infos[CellOf(value)]};
          if (typeInfos.empty() && !found.empty()) {
            return;
          }
          if (auto it{types.find(CellOf(value))};
              it != types.end() && typeInfos.empty() &&
              RootKindOf(*function.getParent(), type) == kRootRef) {
            return;
          }
          types[CellOf(value)] = type;
          if (auto *tagged{llvm::dyn_cast<llvm::GetElementPtrInst>(value)}) {
            llvm::APInt offset{64, 0};
            if (tagged->accumulateConstantOffset(
                    function.getDataLayout(), offset)) {
              tags[CellOf(value)] = offset.getZExtValue();
            }
          }
          found.clear();
          for (const llvm::Use &info : typeInfos) {
            found.push_back(info.get());
          }
        });
  }
  // The rewrite makes roots of the call's own arguments too. One the bundle
  // does not list is dead after the call, and the callee looks after it. It is
  // not put in the bundle, which inlining would then keep alive in the callee.
  llvm::SmallPtrSet<const llvm::Value *, 8> arguments;
  for (const llvm::Use &argument : statepoint->actual_args()) {
    arguments.insert(CellOf(argument.get()));
  }
  // A base that the rewrite made, as a phi of a derived pointer's bases, has
  // the type of the pointer it is the base of.
  for (const llvm::GCRelocateInst *relocate : statepoint->getGCRelocates()) {
    const llvm::Value *base{CellOf(relocate->getBasePtr())};
    const llvm::Value *derived{CellOf(relocate->getDerivedPtr())};
    if (arguments.contains(derived)) {
      arguments.insert(base);
    }
    llvm::Constant *type{nullptr};
    if (auto it{types.find(derived)}; it != types.end()) {
      type = it->second;
    } else if (auto at{types.find(base)}; at != types.end()) {
      type = at->second;
    } else {
      type = FindType(derived, known);
    }
    if (type) {
      types.try_emplace(base, type);
      types.try_emplace(derived, type);
    }
    // And its type_infos.
    if (auto it{infos.find(derived)}; it != infos.end()) {
      llvm::SmallVector<llvm::Value *, 2> copy{it->second};
      infos.try_emplace(base, std::move(copy));
    } else if (auto at{infos.find(base)}; at != infos.end()) {
      llvm::SmallVector<llvm::Value *, 2> copy{at->second};
      infos.try_emplace(derived, std::move(copy));
    }
  }
  llvm::Constant *dead{llvm::ConstantInt::getSigned(
      llvm::Type::getInt64Ty(function.getContext()), -1)};
  // Which root has which type, while the relocations still pair each base
  // with what is derived from it.
  roots.statepoint = statepoint;
  roots.slots = std::move(slots);
  if (live) {
    for (llvm::Value *root : live->Inputs) {
      auto it{types.find(CellOf(root))};
      llvm::Constant *type{it != types.end() ? it->second : nullptr};
      if (!type) {
        type = FindType(root, known);
      }
      if (!type && (IsFromStaticData(root) || arguments.contains(CellOf(root)))) {
        type = dead;
      }
      if (!type) {
        std::string text;
        llvm::raw_string_ostream os{text};
        os << "a root in " << function.getName()
           << " has no type at a safepoint:" << *root << "\n at:" << *statepoint;
        llvm::report_fatal_error(llvm::StringRef{text});
      }
      // Every value live before the rewrite is in the bundle (ListLiveRoots),
      // so one that is not is a base the rewrite keeps for a pointer into the
      // same cell. That pointer is live here too and says what the cell
      // holds; this only has to be updated when it moves. Its bits are the
      // untagged cell, which the type would misread as a constant.
      auto *known{llvm::cast<llvm::ConstantInt>(type)};
      if (!listed.contains(root) && !known->isNegative() && root == CellOf(root) &&
          infos.count(root) && tags.count(root) && tags.lookup(root) < 8) {
        // The cell of a term the bundle has, whose tag the rewrite left behind.
        roots.tags[root] = tags.lookup(root);
      } else if (!listed.contains(root) && !known->isNegative()) {
        type = llvm::ConstantInt::getSigned(
            llvm::Type::getInt64Ty(function.getContext()), kForwardOnly);
      }
      roots.types[root] = type;
      if (auto it{infos.find(CellOf(root))}; it != infos.end()) {
        roots.infos[root] = it->second;
      }
    }
  }
  if (deopt && live) {
    llvm::SmallPtrSet<const llvm::Value *, 16> relocated;
    for (llvm::Value *root : live->Inputs) {
      relocated.insert(root);
    }
    ForEachEntry(deopt->Inputs,
        [&](llvm::Constant *type, llvm::Value *value,
            llvm::ArrayRef<llvm::Use> typeInfos) {
          if (!IsHeapPointer(value->getType()) ||
              llvm::isa<llvm::Constant>(value) || IsFromStaticData(value) ||
              relocated.contains(value) || roots.types.count(value) ||
              RootKindOf(*function.getParent(), type) == kRootRef) {
            return;
          }
          roots.dead.push_back(value);
          roots.types[value] = type;
          auto &found{roots.infos[value]};
          for (const llvm::Use &info : typeInfos) {
            found.push_back(info.get());
          }
        });
  }
  // A base of another pointer, which is the tagged term, is the untagged cell.
  for (const llvm::GCRelocateInst *relocate : statepoint->getGCRelocates()) {
    if (relocate->getBasePtr() != relocate->getDerivedPtr()) {
      roots.types[relocate->getBasePtr()] = llvm::ConstantInt::getSigned(
          llvm::Type::getInt64Ty(function.getContext()), kForwardOnly);
    }
  }
}

// The type_infos of a relocation's root; none for a root that needs no type.
static llvm::ArrayRef<llvm::Value *> InfosOf(
    const StatepointRoots &roots, const llvm::GCRelocateInst *relocate) {
  llvm::Value *derived{relocate->getDerivedPtr()};
  auto it{roots.infos.find(derived)};
  auto *type{llvm::cast_or_null<llvm::ConstantInt>(roots.types.lookup(derived))};
  if (it == roots.infos.end() || !type || type->isNegative()) {
    return {};
  }
  return it->second;
}

// Whether a type index means a type that is walked by type_infos.
static bool IsPolymorphicIndex(llvm::Module &module, llvm::ConstantInt *type) {
  if (type->isNegative()) {
    return false;
  }
  auto *table{llvm::cast<llvm::ConstantStruct>(
      module.getNamedGlobal(kRootTable)->getInitializer())};
  auto *kind{llvm::cast<llvm::ConstantInt>(
      table->getOperand(2 + 2 * type->getZExtValue()))};
  return kind->getSExtValue() == kRootPolymorphicTerm ||
      kind->getSExtValue() == kRootWord;
}

// The stack map has a location pair for each relocation that is still used, in
// the order of getGCRelocates, and then for each heap pointer in the deopt
// state that is not one of those, in the order of the state: instruction
// selection adds those so that a collection can update them. The type indices
// follow the pairs. The deopt state is the number of pairs, a type for each, a
// layout and an address for each environment, and each root's type_infos.
static void EncodeRootTypes(StatepointRoots &roots) {
  llvm::GCStatepointInst *statepoint{roots.statepoint};
  llvm::Type *i64{llvm::Type::getInt64Ty(statepoint->getContext())};
  std::vector<const llvm::GCRelocateInst *> relocates{
      statepoint->getGCRelocates()};
  llvm::DenseSet<const llvm::Value *> relocated;
  for (const llvm::GCRelocateInst *relocate : relocates) {
    relocated.insert(relocate->getDerivedPtr());
  }
  std::size_t typeInfos{0};
  llvm::SmallSetVector<llvm::Value *, 4> added;
  for (const llvm::GCRelocateInst *relocate : relocates) {
    for (llvm::Value *info : InfosOf(roots, relocate)) {
      ++typeInfos;
      if (!relocated.contains(info)) {
        added.insert(info);
      }
    }
  }
  for (llvm::Value *value : roots.dead) {
    for (llvm::Value *info : roots.infos.lookup(value)) {
      ++typeInfos;
      if (!relocated.contains(info)) {
        added.insert(info);
      }
    }
  }
  // After the type_infos, in the deopt state, as the pairs are in its order.
  llvm::SmallVector<llvm::Value *> deadOperands;
  for (llvm::Value *value : roots.dead) {
    if (added.insert(value)) {
      deadOperands.push_back(value);
    }
  }
  std::size_t count{relocates.size() + added.size()};
  std::vector<llvm::Value *> encoded(
      1 + count, llvm::ConstantInt::getSigned(i64, -1));
  encoded[0] = llvm::ConstantInt::get(
      i64, count | (std::uint64_t{deadOperands.size()} << 32));
  for (auto [slot, layout] : roots.slots) {
    encoded.push_back(layout);
    encoded.push_back(slot);
  }
  encoded.resize(encoded.size() + typeInfos, llvm::ConstantInt::getSigned(i64, -1));
  encoded.insert(encoded.end(), deadOperands.begin(), deadOperands.end());
  llvm::SmallVector<llvm::OperandBundleDef, 2> bundles;
  for (unsigned b = 0; b < statepoint->getNumOperandBundles(); ++b) {
    llvm::OperandBundleUse use{statepoint->getOperandBundleAt(b)};
    if (use.getTagID() != llvm::LLVMContext::OB_deopt) {
      bundles.emplace_back(use);
    }
  }
  bundles.emplace_back("deopt", std::move(encoded));
  llvm::CallBase *replaced{llvm::CallBase::Create(
      statepoint, bundles, statepoint->getIterator())};
  statepoint->replaceAllUsesWith(replaced);
  statepoint->eraseFromParent();
  // Only now, as replacing the uses reverses their order.
  auto *final{llvm::cast<llvm::GCStatepointInst>(replaced)};
  auto deopt{final->getOperandBundle(llvm::LLVMContext::OB_deopt)};
  unsigned first{static_cast<unsigned>(deopt->Inputs.data() - final->op_begin())};
  unsigned place{first + 1};
  unsigned infoPlace{first + 1 + static_cast<unsigned>(count) +
      2 * static_cast<unsigned>(roots.slots.size())};
  for (const llvm::GCRelocateInst *relocate : final->getGCRelocates()) {
    auto *type{llvm::cast<llvm::ConstantInt>(
        roots.types.lookup(relocate->getDerivedPtr()))};
    llvm::ArrayRef<llvm::Value *> infos{InfosOf(roots, relocate)};
    if (infos.empty() && IsPolymorphicIndex(*final->getModule(), type)) {
      std::string text;
      llvm::raw_string_ostream os{text};
      os << "not yet implemented: a garbage collector root whose type_infos "
         << "are not known, in " << final->getFunction()->getName() << ":"
         << *relocate;
      llvm::report_fatal_error(llvm::StringRef{text});
    }
    for (llvm::Value *info : infos) {
      final->setOperand(infoPlace++, info);
    }
    std::uint64_t tag{roots.tags.lookup(relocate->getDerivedPtr())};
    final->setOperand(place++,
        infos.empty() && tag == 0
            ? type
            : llvm::ConstantInt::get(i64,
                  type->getZExtValue() | (std::uint64_t{infos.size()} << 32) |
                      (tag << 56)));
  }
  for (llvm::Value *value : added) {
    llvm::Constant *type{roots.types.lookup(value)};
    if (!type) {
      final->setOperand(place++, llvm::ConstantInt::getSigned(i64, -1));
      continue;
    }
    // Only what was counted above: a type_info added here has none.
    llvm::SmallVector<llvm::Value *, 2> infos;
    if (llvm::is_contained(roots.dead, value)) {
      infos = roots.infos.lookup(value);
    }
    for (llvm::Value *info : infos) {
      final->setOperand(infoPlace++, info);
    }
    final->setOperand(place++,
        infos.empty()
            ? type
            : llvm::ConstantInt::get(i64,
                  llvm::cast<llvm::ConstantInt>(type)->getZExtValue() |
                      (std::uint64_t{infos.size()} << 32)));
  }
}

// Mercury's calling convention, or Mercury code inlined into another's, as
// main's is into the C function merk_main. An OpenMP region's outlined function
// holds only calls marked as in the region.
static bool IsMercuryCode(llvm::Function &function) {
  if (function.getCallingConv() == llvm::CallingConv::Tail) {
    return true;
  }
  for (llvm::Instruction &instruction : llvm::instructions(function)) {
    auto *call{llvm::dyn_cast<llvm::CallBase>(&instruction)};
    auto deopt{call ? call->getOperandBundle(llvm::LLVMContext::OB_deopt)
                    : std::nullopt};
    if (!deopt) {
      continue;
    }
    auto *first{deopt->Inputs.empty()
            ? nullptr
            : llvm::dyn_cast<llvm::ConstantInt>(deopt->Inputs[0])};
    if (!first || first->getSExtValue() != kInRegion) {
      return true;
    }
  }
  return false;
}

// With no statepoint there is no stack map for the table to point to.
static void DropRootTable(llvm::Module &module) {
  llvm::GlobalVariable *table{module.getNamedGlobal(kRootTable)};
  if (!table) {
    return;
  }
  llvm::removeFromUsedLists(module, [&](llvm::Constant *constant) {
    return constant == table;
  });
  table->eraseFromParent();
  if (llvm::GlobalVariable *maps{module.getNamedGlobal("__LLVM_StackMaps")}) {
    maps->eraseFromParent();
  }
}

static void RemoveDeadCode(llvm::Function &function) {
  bool changed{true};
  while (changed) {
    changed = false;
    for (llvm::BasicBlock &block : function) {
      for (llvm::Instruction &instruction :
          llvm::make_early_inc_range(llvm::reverse(block))) {
        if (llvm::isInstructionTriviallyDead(&instruction)) {
          instruction.eraseFromParent();
          changed = true;
        }
      }
    }
  }
}

// Mercury code gets statepoints, here, after the optimizer, which handles them
// badly. Code an OpenMP region outlined is not rewritten: nothing collects
// during a region.
void RewriteForCollector(llvm::Module &module, const TargetInfo &target) {
  if (target.options().openMPIsTargetDevice) {
    return;
  }
  llvm::linkAllBuiltinGCs();
  FlattenResults(module);
  KnownTypes known;
  bool any{false};
  for (llvm::Function &function : module) {
    if (!function.isDeclaration() && IsMercuryCode(function)) {
      function.setGC("statepoint-example");
      function.addFnAttr("frame-pointer", "all");
      MarkLeaves(function);
      RecordTypes(function, known);
      CastAtUses(function);
      ReadBitsEarly(function, known);
      ListLiveRoots(function, known);
      any = true;
    }
  }
  if (!any) {
    DropRootTable(module);
    return;
  }
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
  llvm::ModulePassManager passes;
  passes.addPass(llvm::RewriteStatepointsForGC{});
  passes.run(module, modules);
  llvm::SmallVector<llvm::GCStatepointInst *> statepoints;
  for (llvm::Function &function : module) {
    for (llvm::Instruction &instruction : llvm::instructions(function)) {
      if (auto *statepoint{llvm::dyn_cast<llvm::GCStatepointInst>(&instruction)}) {
        statepoints.push_back(statepoint);
      }
    }
  }
  std::vector<StatepointRoots> all(statepoints.size());
  for (auto [statepoint, roots] : llvm::zip(statepoints, all)) {
    FindRootTypes(statepoint, known, roots);
  }
  // After the types, which pair a base with what is derived from it through
  // the relocations. Fast instruction selection skips a relocation only dead
  // code uses, and statepoint lowering expects to see every one.
  for (llvm::Function &function : module) {
    if (function.hasGC()) {
      RemoveDeadCode(function);
    }
  }
  for (StatepointRoots &roots : all) {
    EncodeRootTypes(roots);
  }
  if (statepoints.empty()) {
    DropRootTable(module);
  }
}

} // namespace merkurius::codegen
