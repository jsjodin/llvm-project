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
#include "llvm/Transforms/Scalar/RewriteStatepointsForGC.h"
#include "llvm/Transforms/Utils/Local.h"

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
    for (std::size_t i = 0; deopt && i + 1 < deopt->Inputs.size(); i += 2) {
      const llvm::Value *value{CellOf(deopt->Inputs[i + 1])};
      if (!llvm::isa<llvm::Constant>(value) &&
          known.insert({value, llvm::cast<llvm::Constant>(deopt->Inputs[i])})
              .second) {
        worklist.push_back(value);
      }
    }
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
static void EncodeRootTypes(
    llvm::GCStatepointInst *statepoint, KnownTypes &known) {
  auto deopt{statepoint->getOperandBundle(llvm::LLVMContext::OB_deopt)};
  auto live{statepoint->getOperandBundle(llvm::LLVMContext::OB_gc_live)};
  llvm::Function &function{*statepoint->getFunction()};
  llvm::DenseMap<const llvm::Value *, llvm::Constant *> types;
  for (std::size_t i = 0; deopt && i + 1 < deopt->Inputs.size(); i += 2) {
    types[CellOf(deopt->Inputs[i + 1])] =
        llvm::cast<llvm::Constant>(deopt->Inputs[i]);
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
  }
  llvm::Constant *dead{llvm::ConstantInt::getSigned(
      llvm::Type::getInt64Ty(function.getContext()), -1)};
  std::vector<llvm::Value *> encoded;
  if (live) {
    for (llvm::Value *root : live->Inputs) {
      auto it{types.find(CellOf(root))};
      if (it == types.end()) {
        if (llvm::Constant *type{FindType(root, known)}) {
          encoded.push_back(type);
          continue;
        }
      }
      if (it == types.end() &&
          (IsFromStaticData(root) || arguments.contains(CellOf(root)))) {
        encoded.push_back(dead);
        continue;
      }
      if (it == types.end()) {
        std::string text;
        llvm::raw_string_ostream os{text};
        os << "a root in " << function.getName()
           << " has no type at a safepoint:" << *root << "\n at:" << *statepoint;
        llvm::report_fatal_error(llvm::StringRef{text});
      }
      encoded.push_back(it->second);
    }
  }
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
    if (!function.isDeclaration() &&
        function.getCallingConv() == llvm::CallingConv::Tail) {
      function.setGC("statepoint-example");
      function.addFnAttr("frame-pointer", "all");
      MarkLeaves(function);
      RecordTypes(function, known);
      ReadBitsEarly(function, known);
      any = true;
    }
  }
  if (!any) {
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
  for (llvm::GCStatepointInst *statepoint : statepoints) {
    EncodeRootTypes(statepoint, known);
  }
  // After the types, which pair a base with what is derived from it through
  // the relocations. Fast instruction selection skips a relocation only dead
  // code uses, and statepoint lowering expects to see every one.
  for (llvm::Function &function : module) {
    if (function.hasGC()) {
      RemoveDeadCode(function);
    }
  }
}

} // namespace merkurius::codegen
