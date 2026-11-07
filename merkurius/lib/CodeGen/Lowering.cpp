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
#include "merkurius/Dialect/Mrt/IR/MrtDialect.h"
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
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include <functional>
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

// The layout of a type defined by constructors (design document, "Type
// representation"), chosen from its merk.type_def. Cells are headerless.
// A type of constants only is a word, each constant its number. A type of
// one constructor of one argument is the argument. Any other is a word
// that is a constant, a small integer shifted past the primary tag, or a
// pointer to a cell of a constructor's arguments, with a 3-bit primary tag
// for the constructor: tag 0 is the constants' if there are any, and when
// the constructors with arguments outnumber the tags left, the last tag is
// shared, and the cell's first word is a secondary tag.
struct TermLayout {
  enum class Kind { Enum, NoTag, Cells } kind{Kind::Cells};
  // Its LLVM type: a word for an Enum, the argument's for a NoTag, and a
  // pointer for Cells.
  mlir::Type type;
  struct Functor {
    std::string name;
    // The number of its cell's fields, of which the first `hidden` are an
    // existentially typed constructor's type_infos and typeclass_infos
    // (Merk_ConstructorAttr).
    unsigned arity;
    unsigned hidden{0};
    // A constant's word, or a cell's primary tag and secondary tag.
    std::int64_t value{0};
    unsigned tag{0};
    std::optional<std::int64_t> secondary;
    llvm::SmallVector<mlir::Type> fields;
  };
  std::vector<Functor> functors;
  bool constants{false};
  // The tag that constructors share, if they do.
  std::optional<unsigned> shared;

  const Functor *Find(merk::FunctorAttr functor) const {
    for (const Functor &f : functors) {
      if (f.name == functor.getName() &&
          f.arity - f.hidden == functor.getArity()) {
        return &f;
      }
    }
    return nullptr;
  }
};

constexpr unsigned kTagBits{3};
constexpr unsigned kTags{1u << kTagBits};

// A value known at compile time, which a static cell's field can hold
// (design document, "Type representation", ground terms): an integer or
// float constant, a string's bytes, a constant's word among cells, or a
// static cell, tagged. `key` names it, so that equal cells are one.
struct StaticValue {
  enum class Kind { Integer, Float, String, Word, Cell } kind;
  mlir::Type type;
  mlir::Attribute value;
  mlir::LLVM::GlobalOp cell;
  unsigned tag{0};
  std::string key;
};

// Whether a type constructor's type_infos have the number of their arguments
// in word 1: a tuple's and a higher-order type's.
static bool VariableArity(llvm::StringRef constructor) {
  return constructor.starts_with("{}") || constructor.starts_with("pred/") ||
      constructor.starts_with("func/");
}

class Lowerer {
public:
  Lowerer(mlir::ModuleOp module, const TargetInfo &target)
      : module_{module}, rewriter_{module.getContext()},
        cconv_{target.HasTailCallConvention() ? mlir::LLVM::CConv::Tail
                                              : mlir::LLVM::CConv::C},
        registers_{target.Results()} {
    auto offload{mlir::dyn_cast<mlir::omp::OffloadModuleInterface>(
        module.getOperation())};
    device_ = mlir::omp::isOpenMPModule(module) && offload &&
        offload.getIsTargetDevice();
  }
  llvm::LogicalResult Lower();

private:
  // A merk type's LLVM type: a string is a pointer to its bytes, an array a
  // pointer to its size and elements, and a term its layout's.
  mlir::Type Convert(mlir::Type type) const {
    if (mlir::isa<merk::StringType, merk::ArrayType, merk::TypeInfoType,
            merk::TypeclassInfoType, merk::ClosureType, merk::ContType,
            merk::EnvType, merk::RefType>(type)) {
      return mlir::LLVM::LLVMPointerType::get(type.getContext());
    }
    if (mlir::isa<merk::WordType>(type)) {
      return mlir::IntegerType::get(type.getContext(), 64);
    }
    if (auto term{mlir::dyn_cast<merk::TermType>(type)}) {
      auto it{layouts_.find(term.getName())};
      return it == layouts_.end()
          ? mlir::LLVM::LLVMPointerType::get(type.getContext())
          : it->second.type;
    }
    return type;
  }
  llvm::SmallVector<mlir::Type> Convert(mlir::TypeRange types) const {
    return llvm::map_to_vector(
        types, [this](mlir::Type t) { return Convert(t); });
  }
  // A procedure's outputs as one LLVM value: none, the output, or a struct
  // of them, after the success flag of one that can fail (design document,
  // "Calling convention").
  mlir::Type ResultType(mlir::MLIRContext *context, mlir::TypeRange outputs,
      bool canFail = false) const {
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
  // The layouts of the module's merk.type_defs, which are then erased.
  llvm::LogicalResult Layouts();
  const TermLayout *LayoutOf(mlir::Type type) const {
    auto term{mlir::dyn_cast<merk::TermType>(type)};
    if (!term) {
      return nullptr;
    }
    auto it{layouts_.find(term.getName())};
    return it == layouts_.end() ? nullptr : &it->second;
  }
  mlir::Type ResolveLayoutType(llvm::StringRef name, llvm::StringSet<> &seen);
  // Lowers the operations on terms to mrt's, while their operands still
  // have merk's types, which give their layouts: switches on terms are
  // recorded with theirs, for Flatten, which makes them mrt's too. A
  // construction of a ground term known at compile time is a static cell.
  llvm::LogicalResult LowerTerms();
  mlir::Value LoweredValue(mlir::Value, mlir::Location);
  mlir::Value WordsEqual(mlir::Value a, mlir::Value b, mlir::Location);
  // The static value that `value` is, if it is one.
  std::optional<StaticValue> StaticOf(mlir::Value value);
  // A static cell of a functor and its fields, one for each cell's contents,
  // whatever their types, since a cell is never updated.
  mlir::LLVM::GlobalOp StaticCell(const TermLayout::Functor &,
      llvm::ArrayRef<StaticValue> fields, mlir::Location);
  // A static value in a global's initializer, at the rewriter's insertion
  // point.
  mlir::Value Materialize(const StaticValue &, mlir::Location);
  // The address that many bytes from `address`.
  mlir::Value OffsetBytes(
      mlir::Value address, std::int64_t bytes, mlir::Location);
  // Lowers mrt's operations to the LLVM dialect's.
  void LowerMrt();
  void LowerAlloc(mrt::AllocOp);
  // The thread's heap pointer and limit (runtime/heap.h).
  mlir::LLVM::GlobalOp HeapGlobal();
  // A switch on a word: `to` by their values, or else to `otherwise`.
  void SwitchOn(mlir::Value value,
      llvm::ArrayRef<std::pair<std::int64_t, mlir::Block *>> to,
      mlir::Block *otherwise, mlir::Location);
  // The branches of a switch on a term to the blocks of its functors, in
  // the order of `functors`, or else `otherwise`, at the rewriter's
  // insertion point.
  void Dispatch(mlir::Value term, const TermLayout &,
      llvm::ArrayRef<std::pair<const TermLayout::Functor *, mlir::Block *>>,
      mlir::Block *otherwise, mlir::Block *before, mlir::Location);
  void LowerProc(merk::ProcOp);
  void LowerSucceed(merk::SucceedOp);
  void LowerConstant(merk::ConstantOp);
  void LowerToWord(merk::ToWordOp);
  void LowerFromWord(merk::FromWordOp);
  void LowerTypeInfo(merk::TypeInfoOp);
  // The method tables, as constant arrays of the procedures' addresses.
  void LowerInstances();
  llvm::LogicalResult LowerTypeclassInfo(merk::TypeclassInfoOp);
  // A method's call, through its typeclass_info's table.
  mlir::Value CallMethod(merk::CallMethodOp, bool *tail = nullptr);
  // The word of a typeclass_info's cell.
  void LowerTypeclassInfoWord(merk::TypeclassInfoWordOp);
  // A closure's cell, a captured value from it, and a call through it.
  void LowerClosure(merk::ClosureOp);
  void LowerClosureWord(merk::ClosureWordOp);
  mlir::Value CallClosure(merk::CallClosureOp, bool *tail = nullptr);
  // Outlined nondeterministic code's continuations and their environments
  // (design document, Lowering, "Nondeterministic code").
  mlir::Value CallCont(merk::CallContOp, bool *tail = nullptr);
  void LowerContRef(merk::ContRefOp);
  mlir::LLVM::LLVMStructType EnvType(mlir::ArrayAttr layout);
  void LowerEnvNew(merk::EnvNewOp);
  void LowerEnvGet(merk::EnvGetOp);
  void LowerEnvSet(merk::EnvSetOp);
  void LowerRefStore(merk::RefStoreOp);
  // A call of the code at `code`, with `first`, a pointer, and then `rest`.
  mlir::Value CallPointer(mlir::Operation *call, mlir::Value code,
      mlir::Value first, mlir::ValueRange rest, bool canFail, bool *tail);
  // The procedure's succeed that a call ends, if it is a tail call: only
  // conversions between values and words come between them, and each output
  // is the call's result at its place through conversions that give back
  // its bits, as their LLVM types are the same. `between` gets those
  // conversions, which go with the call.
  merk::SucceedOp TailSucceed(mlir::Operation *call,
      llvm::SmallVectorImpl<mlir::Operation *> &between);
  llvm::LogicalResult LowerTypeCtors();
  mlir::LLVM::GlobalOp RttiTable(mlir::Type, mlir::Location,
      llvm::function_ref<mlir::Value(mlir::Value)> make);
  mlir::LLVM::GlobalOp I16Table(
      llvm::ArrayRef<std::int64_t> values, mlir::Location);
  mlir::Value FieldTypeValue(mlir::Attribute type,
      llvm::DenseMap<mlir::Attribute, mlir::LLVM::GlobalOp> &cells);
  // The aggregate with the value at the position.
  mlir::Value InsertAt(mlir::Value aggregate,
      llvm::ArrayRef<std::int64_t> position, mlir::Value value, mlir::Location);
  // Sets the next field of an aggregate being built, and moves on `position`.
  void AppendField(mlir::Value &aggregate, std::int64_t &position,
      mlir::Value value, mlir::Location);
  void LowerTypeInfoWord(merk::TypeInfoWordOp);
  // Calls the entry at `offset` in the descriptor of `op`'s type_info, its
  // first operand, on its other two.
  mlir::Value CallDescriptorEntry(mlir::Operation *op, std::int32_t offset,
      mlir::Type result);
  llvm::LogicalResult LowerUnifyWord(merk::UnifyWordOp);
  llvm::LogicalResult LowerCompareWord(merk::CompareWordOp);
  void LowerStringCompare(merk::StringCompareOp);
  void LowerCompareTypeInfo(merk::CompareTypeInfoOp);
  void LowerStringEqual(merk::StringEqualOp);
  void LowerForeignCall(merk::ForeignCallOp);
  void LowerEntry(merk::EntryOp);
  mlir::Value CallRuntime(llvm::StringRef name, mlir::Type result,
      mlir::ValueRange arguments, mlir::Location);
  mlir::Value InitialElement(
      mlir::Operation *, unsigned operand, mlir::Location);
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
  // A call, of `func` or else the function `pointer` points to, of type
  // `funcType`, with the inputs. The results are as Call's.
  mlir::Value CallCode(mlir::Operation *call, mlir::LLVM::LLVMFuncOp func,
      mlir::Value pointer, mlir::LLVM::LLVMFunctionType funcType,
      llvm::SmallVector<mlir::Value> arguments, bool canFail, bool inMemory,
      bool *tail);
  // A procedure's result: its success flag, if it has one, and outputs.
  mlir::Value Result(mlir::Location, mlir::Type, std::optional<bool> flag,
      mlir::ValueRange outputs);
  mlir::LLVM::GlobalOp StringGlobal(llvm::StringRef, mlir::Location);
  mlir::LLVM::LLVMFuncOp ThrowFunc();
  bool CanFail(llvm::StringRef proc) const { return canFail_.contains(proc); }
  // Whether a procedure's outputs, after its success flag if it can fail,
  // need more registers than the target returns results in, so that it
  // returns them in memory, through a pointer before its inputs.
  bool InMemory(mlir::TypeRange outputs, bool canFail) const;
  bool InMemory(llvm::StringRef proc) const { return inMemory_.contains(proc); }
  // The struct of a procedure's outputs in memory, and the address of one.
  mlir::LLVM::LLVMStructType OutputsType(mlir::TypeRange outputs);
  mlir::Value OutputAddress(mlir::Value outputs, mlir::LLVM::LLVMStructType,
      unsigned i, mlir::Location);
  // An alloca at the start of the function, or of the OpenMP region that
  // holds `op`, which is outlined into a function of its own: each thread
  // needs its own, as Flang and ClangIR place allocas.
  mlir::Value Alloca(mlir::Operation *op, mlir::Type, mlir::Location);

  mlir::ModuleOp module_;
  mlir::IRRewriter rewriter_;
  llvm::StringMap<mlir::LLVM::LLVMFuncOp> funcs_;
  // The global of each string constant's bytes.
  llvm::StringMap<mlir::LLVM::GlobalOp> strings_;
  // The procedures that can fail, and those that return their outputs in
  // memory.
  llvm::StringSet<> canFail_;
  llvm::StringSet<> inMemory_;
  // Where a goal that fails goes, at the operation Flatten is at, and at
  // the top of its procedure: a call is a tail call only where they are
  // the same.
  mlir::Block *fail_{nullptr};
  mlir::Block *topFail_{nullptr};
  mlir::LLVM::LLVMFuncOp throw_;
  // Whether this is an OpenMP device compile.
  bool device_{false};
  // The procedures' calling convention, the target's (design document,
  // "Calling convention").
  mlir::LLVM::CConv cconv_;
  TargetInfo::ResultRegisters registers_;
  llvm::StringMap<TermLayout> layouts_;
  llvm::DenseMap<mlir::Operation *, const TermLayout *> termSwitches_;
  // A word that is no secondary tag, which a test of one reads instead of a
  // constant's cell, which it has none.
  mlir::LLVM::GlobalOp noSecondary_;
  // How many merk.rtti tables have been made, which numbers their names.
  unsigned rttiTables_{0};
  // The static values that LowerTerms has made, and the static cells by
  // their keys.
  llvm::DenseMap<mlir::Value, StaticValue> statics_;
  llvm::StringMap<mlir::LLVM::GlobalOp> staticCells_;
  // The type_infos known at compile time, for those made of them.
  struct StaticTypeInfo {
    mlir::LLVM::GlobalOp global;
    std::int64_t tag;
  };
  // The address of a static type_info, with its tag.
  mlir::Value TypeInfoAddress(const StaticTypeInfo &, mlir::Location);
  llvm::DenseMap<mlir::Value, StaticTypeInfo> staticTypeInfos_;
  llvm::StringMap<mlir::LLVM::GlobalOp> typeInfoCells_;
  mlir::LLVM::GlobalOp heap_;
};

llvm::LogicalResult Lowerer::Layouts() {
  mlir::MLIRContext *context{module_.getContext()};
  auto ptr{mlir::LLVM::LLVMPointerType::get(context)};
  auto word{mlir::IntegerType::get(context, 64)};
  for (auto def : llvm::make_early_inc_range(module_.getOps<merk::TypeDefOp>())) {
    TermLayout layout;
    unsigned constants{0}, cells{0};
    for (auto constructor :
        def.getConstructors().getAsRange<merk::ConstructorAttr>()) {
      (constructor.getFields().empty() ? constants : cells) += 1;
    }
    layout.constants = constants != 0;
    if (cells == 0) {
      layout.kind = TermLayout::Kind::Enum;
      layout.type = word;
    } else if (constants == 0 && cells == 1 &&
        def.getConstructors().size() == 1 &&
        mlir::cast<merk::ConstructorAttr>(def.getConstructors()[0])
                .getFields()
                .size() == 1) {
      layout.kind = TermLayout::Kind::NoTag;
    } else {
      layout.type = ptr;
    }
    // Tag 0 is the constants', if there are any; when the constructors
    // with arguments need more tags than are left, the last is shared.
    unsigned first{layout.constants ? 1u : 0u};
    unsigned available{kTags - first};
    unsigned constant{0}, cell{0};
    for (auto constructor :
        def.getConstructors().getAsRange<merk::ConstructorAttr>()) {
      TermLayout::Functor functor;
      functor.name = constructor.getName().str();
      functor.fields.assign(
          constructor.getFields().begin(), constructor.getFields().end());
      functor.arity = functor.fields.size();
      functor.hidden = constructor.getHidden();
      if (functor.arity == 0) {
        functor.value = layout.kind == TermLayout::Kind::Enum
            ? constant
            : static_cast<std::int64_t>(constant) << kTagBits;
        ++constant;
      } else if (cells <= available || cell + 1 < available) {
        functor.tag = first + cell++;
      } else {
        functor.tag = kTags - 1;
        functor.secondary = cell++ - (available - 1);
        layout.shared = functor.tag;
      }
      layout.functors.push_back(std::move(functor));
    }
    layouts_[def.getName()] = std::move(layout);
    rewriter_.eraseOp(def);
  }
  for (auto &entry : layouts_) {
    llvm::StringSet<> seen;
    ResolveLayoutType(entry.getKey(), seen);
  }
  return llvm::success();
}

// A type of one constructor of one argument is its argument's type, a chain of
// them the last's, and a cycle of them, which has no values but infinite ones, a
// pointer. `seen` has the types on the chain so far.
mlir::Type Lowerer::ResolveLayoutType(
    llvm::StringRef name, llvm::StringSet<> &seen) {
  TermLayout &layout{layouts_.find(name)->second};
  if (layout.type) {
    return layout.type;
  }
  if (!seen.insert(name).second) {
    return mlir::LLVM::LLVMPointerType::get(module_.getContext());
  }
  mlir::Type field{layout.functors.front().fields.front()};
  auto term{mlir::dyn_cast<merk::TermType>(field)};
  layout.type = term && layouts_.count(term.getName())
      ? ResolveLayoutType(term.getName(), seen)
      : Convert(field);
  return layout.type;
}

std::optional<StaticValue> Lowerer::StaticOf(mlir::Value value) {
  if (auto it{statics_.find(value)}; it != statics_.end()) {
    return it->second;
  }
  auto constant{value.getDefiningOp<merk::ConstantOp>()};
  if (!constant) {
    return std::nullopt;
  }
  StaticValue result;
  result.type = Convert(constant.getType());
  result.value = constant.getValue();
  if (auto integer{mlir::dyn_cast<mlir::IntegerAttr>(result.value)}) {
    result.kind = StaticValue::Kind::Integer;
    result.key = "i" + std::to_string(integer.getType().getIntOrFloatBitWidth()) +
        ":" + std::to_string(integer.getValue().getSExtValue());
  } else if (auto real{mlir::dyn_cast<mlir::FloatAttr>(result.value)}) {
    result.kind = StaticValue::Kind::Float;
    result.key = "f:" +
        std::to_string(real.getValue().bitcastToAPInt().getZExtValue());
  } else {
    llvm::StringRef string{mlir::cast<mlir::StringAttr>(result.value).getValue()};
    result.kind = StaticValue::Kind::String;
    result.key = "s" + std::to_string(string.size()) + ":" + string.str();
  }
  return result;
}

mlir::Value Lowerer::Materialize(const StaticValue &value, mlir::Location loc) {
  auto ptr{mlir::LLVM::LLVMPointerType::get(module_.getContext())};
  switch (value.kind) {
  case StaticValue::Kind::Integer:
  case StaticValue::Kind::Float:
    return mlir::LLVM::ConstantOp::create(rewriter_, loc, value.type,
        mlir::cast<mlir::TypedAttr>(value.value));
  case StaticValue::Kind::String:
    return mlir::LLVM::AddressOfOp::create(rewriter_, loc,
        StringGlobal(mlir::cast<mlir::StringAttr>(value.value).getValue(), loc));
  case StaticValue::Kind::Word: {
    std::int64_t word{mlir::cast<mlir::IntegerAttr>(value.value).getInt()};
    if (word == 0) {
      return mlir::LLVM::ZeroOp::create(rewriter_, loc, ptr);
    }
    mlir::Value integer{mlir::LLVM::ConstantOp::create(
        rewriter_, loc, rewriter_.getI64Type(), word)};
    return mlir::LLVM::IntToPtrOp::create(rewriter_, loc, ptr, integer);
  }
  case StaticValue::Kind::Cell: {
    mlir::Value address{
        mlir::LLVM::AddressOfOp::create(rewriter_, loc, value.cell)};
    if (value.tag == 0) {
      return address;
    }
    return mlir::LLVM::GEPOp::create(rewriter_, loc, ptr, rewriter_.getI8Type(),
        address,
        llvm::ArrayRef<mlir::LLVM::GEPArg>{static_cast<std::int32_t>(value.tag)});
  }
  }
  llvm_unreachable("a static value of no kind");
}

mlir::LLVM::GlobalOp Lowerer::StaticCell(const TermLayout::Functor &functor,
    llvm::ArrayRef<StaticValue> fields, mlir::Location loc) {
  std::string key{functor.secondary
          ? "(" + std::to_string(*functor.secondary) + ";"
          : std::string{"("}};
  for (const StaticValue &field : fields) {
    key += field.key + ",";
  }
  key += ")";
  mlir::LLVM::GlobalOp &cell{staticCells_[key]};
  if (cell) {
    return cell;
  }
  // A struct of the cell's words: a field narrower than a word is its
  // whole word, zero-extended, as a store of it writes it.
  mlir::MLIRContext *context{module_.getContext()};
  auto i64{rewriter_.getI64Type()};
  llvm::SmallVector<mlir::Type> words;
  llvm::SmallVector<StaticValue> values;
  if (functor.secondary) {
    StaticValue secondary;
    secondary.kind = StaticValue::Kind::Integer;
    secondary.type = i64;
    secondary.value = rewriter_.getI64IntegerAttr(*functor.secondary);
    values.push_back(secondary);
  }
  values.append(fields.begin(), fields.end());
  for (StaticValue &value : values) {
    auto integer{mlir::dyn_cast<mlir::IntegerType>(value.type)};
    if (integer && integer.getWidth() < 64) {
      value.value = rewriter_.getIntegerAttr(i64,
          mlir::cast<mlir::IntegerAttr>(value.value).getValue().zext(64));
      value.type = i64;
    }
    words.push_back(value.type);
  }
  auto cellType{mlir::LLVM::LLVMStructType::getLiteral(context, words)};
  mlir::OpBuilder::InsertionGuard guard{rewriter_};
  rewriter_.setInsertionPointToStart(module_.getBody());
  cell = mlir::LLVM::GlobalOp::create(rewriter_, loc, cellType,
      /*isConstant=*/true, mlir::LLVM::Linkage::Private,
      "merk.term." + std::to_string(staticCells_.size() - 1), mlir::Attribute{},
      /*alignment=*/8);
  cell.setUnnamedAddr(mlir::LLVM::UnnamedAddr::Global);
  rewriter_.createBlock(&cell.getInitializerRegion());
  mlir::Value init{mlir::LLVM::ZeroOp::create(rewriter_, loc, cellType)};
  for (auto [i, value] : llvm::enumerate(values)) {
    init = mlir::LLVM::InsertValueOp::create(rewriter_, loc, init,
        Materialize(value, loc),
        llvm::SmallVector<std::int64_t>{static_cast<std::int64_t>(i)});
  }
  mlir::LLVM::ReturnOp::create(rewriter_, loc, init);
  return cell;
}

// A value as its LLVM type, through a cast while it is still merk's, which goes
// once it is lowered.
mlir::Value Lowerer::LoweredValue(mlir::Value value, mlir::Location loc) {
  mlir::Type type{Convert(value.getType())};
  if (type == value.getType()) {
    return value;
  }
  return mlir::UnrealizedConversionCastOp::create(rewriter_, loc, type, value)
      .getResult(0);
}

mlir::Value Lowerer::WordsEqual(
    mlir::Value a, mlir::Value b, mlir::Location loc) {
  return mlir::LLVM::ICmpOp::create(
      rewriter_, loc, mlir::LLVM::ICmpPredicate::eq, a, b);
}

llvm::LogicalResult Lowerer::LowerTerms() {
  auto ptr{mlir::LLVM::LLVMPointerType::get(module_.getContext())};
  auto i64{rewriter_.getI64Type()};
  // Each with its term's type, which lowering the operation that makes the
  // term changes.
  llvm::SmallVector<std::pair<mlir::Operation *, mlir::Type>> ops;
  module_.walk([&](mlir::Operation *op) {
    if (mlir::isa<merk::ConstructOp, merk::DeconstructOp, merk::HasFunctorOp,
            merk::SwitchOp, merk::FieldRefOp>(op)) {
      mlir::Type type{mlir::isa<merk::ConstructOp>(op)
              ? op->getResult(0).getType()
              : op->getOperand(0).getType()};
      // Not a switch on an integer.
      if (mlir::isa<merk::TermType>(type)) {
        ops.push_back({op, type});
      }
    }
  });
  for (auto [op, type] : ops) {
    mlir::Location loc{op->getLoc()};
    rewriter_.setInsertionPoint(op);
    const TermLayout *layout{LayoutOf(type)};
    if (!layout) {
      return op->emitOpError("is on a type with no merk.type_def");
    }
    if (auto switchOp{mlir::dyn_cast<merk::SwitchOp>(op)}) {
      for (mlir::Attribute c : switchOp.getCases()) {
        for (mlir::Attribute functor : mlir::cast<mlir::ArrayAttr>(c)) {
          if (!layout->Find(mlir::cast<merk::FunctorAttr>(functor))) {
            return op->emitOpError("has a functor its type does not have");
          }
        }
      }
      termSwitches_[op] = layout;
      continue;
    }
    const TermLayout::Functor *functor{
        layout->Find(mlir::cast<merk::FunctorAttr>(op->getAttr("functor")))};
    if (!functor) {
      return op->emitOpError("has a functor its type does not have");
    }
    unsigned skip{functor->secondary ? 1u : 0u};
    if (auto ref{mlir::dyn_cast<merk::FieldRefOp>(op)}) {
      // The hole's address, inside the cell (design document, Lowering,
      // "Deterministic and semidet code", "Last call modulo constructor").
      // Temporary (chunk 3, docs/INPROGRESS.md): a pointer into a cell,
      // which a variant's statepoints cannot relocate without the cell; it
      // becomes the cell's pointer and the hole's offset.
      mlir::Value cell{LoweredValue(op->getOperand(0), loc)};
      if (functor->tag != 0) {
        cell = mrt::UntagOp::create(rewriter_, loc, cell, functor->tag);
      }
      rewriter_.replaceOp(op,
          OffsetBytes(cell,
              8 * static_cast<std::int64_t>(ref.getIndex() + skip), loc));
      continue;
    }
    if (auto construct{mlir::dyn_cast<merk::ConstructOp>(op)}) {
      mlir::Value result;
      std::optional<StaticValue> known;
      if (layout->kind == TermLayout::Kind::Enum) {
        result =
            mlir::LLVM::ConstantOp::create(rewriter_, loc, i64, functor->value);
        known = StaticValue{StaticValue::Kind::Integer, i64,
            rewriter_.getI64IntegerAttr(functor->value), {}, 0,
            "i64:" + std::to_string(functor->value)};
      } else if (layout->kind == TermLayout::Kind::NoTag) {
        mlir::Value argument{construct.getArguments().front()};
        result = LoweredValue(argument, loc);
        known = StaticOf(argument);
      } else if (functor->arity == 0) {
        mlir::Value word{
            mlir::LLVM::ConstantOp::create(rewriter_, loc, i64, functor->value)};
        result = mlir::LLVM::IntToPtrOp::create(rewriter_, loc, ptr, word);
        known = StaticValue{StaticValue::Kind::Word, ptr,
            rewriter_.getI64IntegerAttr(functor->value), {}, 0,
            "w:" + std::to_string(functor->value)};
      } else {
        // A ground term known at compile time is a static cell, which is
        // never allocated (decision D6).
        llvm::SmallVector<StaticValue> fields;
        for (auto [argument, field] :
            llvm::zip(construct.getArguments(), functor->fields)) {
          std::optional<StaticValue> value{StaticOf(argument)};
          if (!value || value->type != Convert(field)) {
            break;
          }
          fields.push_back(*value);
        }
        if (fields.size() == functor->arity) {
          mlir::LLVM::GlobalOp cell{StaticCell(*functor, fields, loc)};
          result = mlir::LLVM::AddressOfOp::create(rewriter_, loc, cell);
          known = StaticValue{StaticValue::Kind::Cell, ptr, {}, cell,
              functor->tag,
              "@" + cell.getSymName().str() + "+" +
                  std::to_string(functor->tag)};
        } else {
          result = mrt::AllocOp::create(rewriter_, loc, functor->arity + skip);
          if (functor->secondary) {
            mrt::FieldStoreOp::create(rewriter_, loc,
                mlir::LLVM::ConstantOp::create(
                    rewriter_, loc, i64, *functor->secondary),
                result, 0);
          }
          for (auto [i, argument] : llvm::enumerate(construct.getArguments())) {
            mrt::FieldStoreOp::create(
                rewriter_, loc, LoweredValue(argument, loc), result, i + skip);
          }
        }
        if (functor->tag != 0) {
          result = mrt::TagOp::create(rewriter_, loc, result, functor->tag);
        }
      }
      if (known) {
        statics_[result] = *known;
      }
      rewriter_.replaceOp(op, result);
      continue;
    }
    mlir::Value term{LoweredValue(op->getOperand(0), loc)};
    if (auto deconstruct{mlir::dyn_cast<merk::DeconstructOp>(op)}) {
      llvm::SmallVector<mlir::Value> fields;
      if (layout->kind == TermLayout::Kind::NoTag) {
        fields.push_back(term);
      } else {
        mlir::Value cell{term};
        if (functor->tag != 0) {
          cell = mrt::UntagOp::create(rewriter_, loc, term, functor->tag);
        }
        for (auto [i, field] : llvm::enumerate(functor->fields)) {
          fields.push_back(mrt::FieldLoadOp::create(
              rewriter_, loc, Convert(field), cell, i + skip));
        }
      }
      rewriter_.replaceOp(op, fields);
      continue;
    }
    mlir::Value has;
    if (layout->kind == TermLayout::Kind::Enum) {
      has = WordsEqual(term,
          mlir::LLVM::ConstantOp::create(rewriter_, loc, i64, functor->value),
          loc);
    } else if (layout->kind == TermLayout::Kind::NoTag) {
      has = mlir::LLVM::ConstantOp::create(
          rewriter_, loc, rewriter_.getI1Type(), 1);
    } else if (functor->arity == 0) {
      has = WordsEqual(
          mlir::LLVM::PtrToIntOp::create(rewriter_, loc, i64, term),
          mlir::LLVM::ConstantOp::create(rewriter_, loc, i64, functor->value),
          loc);
    } else {
      has = WordsEqual(mrt::PrimaryTagOp::create(rewriter_, loc, term),
          mlir::LLVM::ConstantOp::create(rewriter_, loc, i64, functor->tag),
          loc);
      if (functor->secondary) {
        // A constant has no cell to read a secondary tag from: the test
        // reads a word that is none instead.
        if (!noSecondary_) {
          mlir::OpBuilder::InsertionGuard guard{rewriter_};
          rewriter_.setInsertionPointToStart(module_.getBody());
          noSecondary_ = mlir::LLVM::GlobalOp::create(rewriter_, loc, i64,
              /*isConstant=*/true, mlir::LLVM::Linkage::Internal,
              "merk_no_secondary_tag", rewriter_.getI64IntegerAttr(-1));
        }
        mlir::Value address{mlir::LLVM::SelectOp::create(rewriter_, loc, has,
            mrt::UntagOp::create(rewriter_, loc, term, functor->tag),
            mlir::LLVM::AddressOfOp::create(rewriter_, loc, noSecondary_))};
        mlir::Value secondary{
            mrt::FieldLoadOp::create(rewriter_, loc, i64, address, 0)};
        has = mlir::LLVM::AndOp::create(rewriter_, loc, has,
            WordsEqual(secondary,
                mlir::LLVM::ConstantOp::create(
                    rewriter_, loc, i64, *functor->secondary),
                loc));
      }
    }
    rewriter_.replaceOp(op, has);
  }
  // The values are the operations', which lowering goes on to replace.
  statics_.clear();
  return llvm::success();
}

void Lowerer::SwitchOn(mlir::Value value,
    llvm::ArrayRef<std::pair<std::int64_t, mlir::Block *>> to,
    mlir::Block *otherwise, mlir::Location loc) {
  llvm::SmallVector<llvm::APInt> values;
  llvm::SmallVector<mlir::Block *> destinations;
  for (auto [v, block] : to) {
    values.push_back(llvm::APInt{64, static_cast<std::uint64_t>(v)});
    destinations.push_back(block);
  }
  llvm::SmallVector<mlir::ValueRange> operands(values.size());
  mlir::LLVM::SwitchOp::create(rewriter_, loc, value, otherwise,
      mlir::ValueRange{}, values, destinations, operands);
}

void Lowerer::Dispatch(mlir::Value term, const TermLayout &layout,
    llvm::ArrayRef<std::pair<const TermLayout::Functor *, mlir::Block *>> cases,
    mlir::Block *otherwise, mlir::Block *before, mlir::Location loc) {
  auto i64{rewriter_.getI64Type()};
  if (term.getType() != layout.type) {
    term = mlir::UnrealizedConversionCastOp::create(
        rewriter_, loc, layout.type, term)
               .getResult(0);
  }
  if (layout.kind == TermLayout::Kind::NoTag) {
    mlir::LLVM::BrOp::create(
        rewriter_, loc, cases.empty() ? otherwise : cases.front().second);
    return;
  }
  llvm::SmallVector<std::pair<std::int64_t, mlir::Block *>> constants,
      secondaries, primaries;
  for (auto [functor, block] : cases) {
    if (functor->arity == 0) {
      constants.push_back({functor->value, block});
    } else if (functor->secondary) {
      secondaries.push_back({*functor->secondary, block});
    } else {
      primaries.push_back({functor->tag, block});
    }
  }
  if (layout.kind == TermLayout::Kind::Enum) {
    SwitchOn(term, constants, otherwise, loc);
    return;
  }
  mlir::Block *here{rewriter_.getInsertionBlock()};
  mlir::Block::iterator point{rewriter_.getInsertionPoint()};
  mlir::Value tag{mrt::PrimaryTagOp::create(rewriter_, loc, term)};
  mlir::Value word{mlir::LLVM::PtrToIntOp::create(rewriter_, loc, i64, term)};
  point = rewriter_.getInsertionPoint();
  // The constants by their words, and the constructors that share a tag
  // by their secondary tags.
  unsigned typeConstants{static_cast<unsigned>(llvm::count_if(
      layout.functors, [](const auto &f) { return f.arity == 0; }))};
  if (typeConstants == 1 && !constants.empty()) {
    // Tag 0 is the type's only constant.
    primaries.push_back({0, constants.front().second});
  } else if (!constants.empty()) {
    mlir::Block *block{rewriter_.createBlock(before)};
    SwitchOn(word, constants, otherwise, loc);
    primaries.push_back({0, block});
  }
  if (!secondaries.empty()) {
    mlir::Block *block{rewriter_.createBlock(before)};
    mlir::Value secondary{mrt::FieldLoadOp::create(rewriter_, loc, i64,
        mrt::UntagOp::create(rewriter_, loc, term, *layout.shared), 0)};
    SwitchOn(secondary, secondaries, otherwise, loc);
    primaries.push_back({*layout.shared, block});
  }
  rewriter_.setInsertionPoint(here, point);
  SwitchOn(tag, primaries, otherwise, loc);
}

bool Lowerer::InMemory(mlir::TypeRange outputs, bool canFail) const {
  unsigned integers{canFail}, floats{0};
  for (mlir::Type type : Convert(outputs)) {
    if (mlir::isa<mlir::FloatType>(type) && !registers_.shared) {
      ++floats;
    } else {
      ++integers;
    }
  }
  return integers > registers_.integers || floats > registers_.floats;
}

mlir::LLVM::LLVMStructType Lowerer::OutputsType(mlir::TypeRange outputs) {
  return mlir::LLVM::LLVMStructType::getLiteral(
      module_.getContext(), Convert(outputs));
}

mlir::Value Lowerer::OutputAddress(mlir::Value outputs,
    mlir::LLVM::LLVMStructType type, unsigned i, mlir::Location loc) {
  return mlir::LLVM::GEPOp::create(rewriter_, loc,
      mlir::LLVM::LLVMPointerType::get(module_.getContext()), type, outputs,
      llvm::ArrayRef<mlir::LLVM::GEPArg>{0, static_cast<int32_t>(i)});
}

mlir::Value Lowerer::Alloca(
    mlir::Operation *op, mlir::Type type, mlir::Location loc) {
  mlir::OpBuilder::InsertionGuard guard{rewriter_};
  mlir::Block *allocas{
      &op->getParentOfType<mlir::LLVM::LLVMFuncOp>().getBody().front()};
  if (auto region{op->getParentOfType<mlir::omp::OutlineableOpenMPOpInterface>()}) {
    allocas = region.getAllocaBlock();
  }
  rewriter_.setInsertionPointToStart(allocas);
  mlir::Value one{mlir::LLVM::ConstantOp::create(
      rewriter_, loc, rewriter_.getI64Type(), rewriter_.getI64IntegerAttr(1))};
  return mlir::LLVM::AllocaOp::create(rewriter_, loc,
      mlir::LLVM::LLVMPointerType::get(module_.getContext()), type, one);
}

void Lowerer::LowerProc(merk::ProcOp proc) {
  mlir::MLIRContext *context{proc.getContext()};
  mlir::FunctionType type{proc.getFunctionType()};
  bool canFail{codegen::CanFail(proc.getDeterminism())};
  if (canFail) {
    canFail_.insert(proc.getSymName());
  }
  bool inMemory{InMemory(type.getResults(), canFail)};
  llvm::SmallVector<mlir::Type> inputs{Convert(type.getInputs())};
  auto pointer{mlir::LLVM::LLVMPointerType::get(context)};
  if (inMemory) {
    inMemory_.insert(proc.getSymName());
    inputs.insert(inputs.begin(), pointer);
  }
  auto funcType{mlir::LLVM::LLVMFunctionType::get(
      inMemory ? ResultType(context, {}, canFail)
               : ResultType(context, type.getResults(), canFail),
      inputs)};
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
  if (inMemory && !func.isExternal()) {
    func.getBody().front().insertArgument(0u, pointer, proc.getLoc());
  }
  funcs_[proc.getSymName()] = func;
  rewriter_.eraseOp(proc);
}

void Lowerer::LowerSucceed(merk::SucceedOp succeed) {
  rewriter_.setInsertionPoint(succeed);
  auto func{succeed->getParentOfType<mlir::LLVM::LLVMFuncOp>()};
  bool canFail{CanFail(func.getSymName())};
  mlir::ValueRange outputs{succeed.getOutputs()};
  if (InMemory(func.getSymName())) {
    mlir::Location loc{succeed.getLoc()};
    auto type{OutputsType(outputs.getTypes())};
    for (auto [i, output] : llvm::enumerate(outputs)) {
      mlir::LLVM::StoreOp::create(rewriter_, loc, output,
          OutputAddress(func.getArgument(0), type, i, loc));
    }
    outputs = {};
  }
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

void Lowerer::LowerToWord(merk::ToWordOp op) {
  mlir::Location loc{op.getLoc()};
  rewriter_.setInsertionPoint(op);
  mlir::Value value{op.getValue()};
  mlir::Type type{Convert(value.getType())};
  if (type != value.getType()) {
    value = mlir::UnrealizedConversionCastOp::create(
        rewriter_, loc, type, value)
                .getResult(0);
  }
  rewriter_.replaceOp(op, ToWord(value, loc));
}

void Lowerer::LowerFromWord(merk::FromWordOp op) {
  mlir::Location loc{op.getLoc()};
  rewriter_.setInsertionPoint(op);
  // Its operand may already be the LLVM type, which getWord cannot return.
  mlir::Type type{Convert(op.getType())};
  mlir::Value value{FromWord(op->getOperand(0), type, loc)};
  if (type != op.getType()) {
    value = mlir::UnrealizedConversionCastOp::create(
        rewriter_, loc, op.getType(), value)
                .getResult(0);
  }
  rewriter_.replaceOp(op, value);
}

mlir::Value Lowerer::InsertAt(mlir::Value aggregate,
    llvm::ArrayRef<std::int64_t> position, mlir::Value value,
    mlir::Location loc) {
  return mlir::LLVM::InsertValueOp::create(
      rewriter_, loc, aggregate, value, position);
}

void Lowerer::AppendField(mlir::Value &aggregate, std::int64_t &position,
    mlir::Value value, mlir::Location loc) {
  aggregate = InsertAt(aggregate, {position++}, value, loc);
}

mlir::Value Lowerer::TypeInfoAddress(
    const StaticTypeInfo &info, mlir::Location loc) {
  mlir::Value base{
      mlir::LLVM::AddressOfOp::create(rewriter_, loc, info.global)};
  if (info.tag == 0) {
    return base;
  }
  return mlir::LLVM::GEPOp::create(rewriter_, loc,
      mlir::LLVM::LLVMPointerType::get(module_.getContext()),
      rewriter_.getI8Type(), base,
      llvm::ArrayRef<mlir::LLVM::GEPArg>{static_cast<std::int32_t>(info.tag)});
}

void Lowerer::LowerTypeInfo(merk::TypeInfoOp op) {
  mlir::Location loc{op.getLoc()};
  auto ptr{mlir::LLVM::LLVMPointerType::get(op.getContext())};
  auto i64{rewriter_.getI64Type()};
  llvm::StringRef constructor{op.getConstructor()};
  std::size_t arguments{op.getArguments().size()};
  std::string symbol{"merk.type_ctor." + constructor.str()};
  auto descriptor{module_.lookupSymbol<mlir::LLVM::GlobalOp>(symbol)};
  assert(descriptor && "LowerTypeCtors checks that each has its descriptor");
  rewriter_.setInsertionPoint(op);
  if (arguments == 0) {
    StaticTypeInfo info{descriptor, 0};
    mlir::Value value{TypeInfoAddress(info, loc)};
    // The value that replaces it is what its users look up.
    staticTypeInfos_[value] = info;
    rewriter_.replaceOp(op, value);
    return;
  }
  // A tuple's constructor has no one arity: the cell says how many.
  bool variableArity{VariableArity(constructor)};
  bool isStatic{true};
  std::string key{symbol};
  llvm::SmallVector<StaticTypeInfo> known;
  for (mlir::Value argument : op.getArguments()) {
    auto it{staticTypeInfos_.find(argument)};
    if (it == staticTypeInfos_.end()) {
      isStatic = false;
      break;
    }
    known.push_back(it->second);
    key += "," + it->second.global.getSymName().str();
  }
  if (isStatic) {
    mlir::LLVM::GlobalOp &cell{typeInfoCells_[key]};
    if (!cell) {
      llvm::SmallVector<mlir::Type> words{ptr};
      if (variableArity) {
        words.push_back(i64);
      }
      words.append(arguments, ptr);
      auto cellType{
          mlir::LLVM::LLVMStructType::getLiteral(op.getContext(), words)};
      mlir::OpBuilder::InsertionGuard guard{rewriter_};
      rewriter_.setInsertionPointToStart(module_.getBody());
      cell = mlir::LLVM::GlobalOp::create(rewriter_, loc, cellType,
          /*isConstant=*/true, mlir::LLVM::Linkage::Private,
          "merk.type_info." + std::to_string(typeInfoCells_.size() - 1),
          mlir::Attribute{}, /*alignment=*/8);
      rewriter_.createBlock(&cell.getInitializerRegion());
      mlir::Value init{mlir::LLVM::ZeroOp::create(rewriter_, loc, cellType)};
      std::int64_t position{0};
      AppendField(init, position,
          TypeInfoAddress(StaticTypeInfo{descriptor, 0}, loc), loc);
      if (variableArity) {
        AppendField(init, position,
            mlir::LLVM::ConstantOp::create(
                rewriter_, loc, i64, static_cast<std::int64_t>(arguments)),
            loc);
      }
      for (const StaticTypeInfo &argument : known) {
        AppendField(init, position, TypeInfoAddress(argument, loc), loc);
      }
      mlir::LLVM::ReturnOp::create(rewriter_, loc, init);
    }
    StaticTypeInfo info{cell, 1};
    mlir::Value value{TypeInfoAddress(info, loc)};
    staticTypeInfos_[value] = info;
    rewriter_.replaceOp(op, value);
    return;
  }
  std::int64_t word{0};
  mlir::Value new_{mrt::AllocOp::create(rewriter_, loc, ptr,
      static_cast<std::int64_t>(arguments + (variableArity ? 2 : 1)))};
  mrt::FieldStoreOp::create(rewriter_, loc,
      TypeInfoAddress(StaticTypeInfo{descriptor, 0}, loc), new_, word++);
  if (variableArity) {
    mrt::FieldStoreOp::create(rewriter_, loc,
        mlir::LLVM::ConstantOp::create(
            rewriter_, loc, i64, static_cast<std::int64_t>(arguments)),
        new_, word++);
  }
  for (mlir::Value argument : op.getArguments()) {
    mrt::FieldStoreOp::create(rewriter_, loc, argument, new_, word++);
  }
  rewriter_.replaceOpWithNewOp<mrt::TagOp>(op, ptr, new_, 1);
}

void Lowerer::LowerInstances() {
  auto ptr{mlir::LLVM::LLVMPointerType::get(module_.getContext())};
  for (auto op :
      llvm::make_early_inc_range(module_.getOps<merk::InstanceOp>())) {
    mlir::Location loc{op.getLoc()};
    rewriter_.setInsertionPoint(op);
    std::optional<mlir::ArrayAttr> methods{op.getMethods()};
    // On a device nothing dispatches, since a method call is an error there,
    // and the procedures are not compiled for it.
    if (!methods || device_) {
      mlir::LLVM::GlobalOp::create(rewriter_, loc,
          mlir::LLVM::LLVMArrayType::get(ptr, 0), /*isConstant=*/true,
          mlir::LLVM::Linkage::External, op.getSymName(), mlir::Attribute{},
          /*alignment=*/8);
      rewriter_.eraseOp(op);
      continue;
    }
    auto type{mlir::LLVM::LLVMArrayType::get(ptr, methods->size())};
    auto global{mlir::LLVM::GlobalOp::create(rewriter_, loc, type,
        /*isConstant=*/true,
        op.getVisibility() == mlir::SymbolTable::Visibility::Private
            ? mlir::LLVM::Linkage::Private
            : mlir::LLVM::Linkage::External,
        op.getSymName(), mlir::Attribute{}, /*alignment=*/8)};
    rewriter_.createBlock(&global.getInitializerRegion());
    mlir::Value init{mlir::LLVM::UndefOp::create(rewriter_, loc, type)};
    for (auto [i, method] : llvm::enumerate(*methods)) {
      llvm::StringRef name{mlir::cast<mlir::FlatSymbolRefAttr>(method).getValue()};
      init = mlir::LLVM::InsertValueOp::create(rewriter_, loc, init,
          mlir::LLVM::AddressOfOp::create(
              rewriter_, loc, funcs_.lookup(name)),
          llvm::SmallVector<std::int64_t>{static_cast<std::int64_t>(i)});
    }
    mlir::LLVM::ReturnOp::create(rewriter_, loc, init);
    rewriter_.setInsertionPoint(global);
    rewriter_.eraseOp(op);
  }
}

llvm::LogicalResult Lowerer::LowerTypeclassInfo(merk::TypeclassInfoOp op) {
  mlir::Location loc{op.getLoc()};
  auto ptr{mlir::LLVM::LLVMPointerType::get(op.getContext())};
  auto table{module_.lookupSymbol<mlir::LLVM::GlobalOp>(op.getInstance())};
  if (!table) {
    return op.emitOpError("names no method table");
  }
  rewriter_.setInsertionPoint(op);
  bool isStatic{true};
  std::string key{op.getInstance().str()};
  llvm::SmallVector<StaticTypeInfo> known;
  for (mlir::Value argument : op.getArguments()) {
    auto it{staticTypeInfos_.find(argument)};
    if (it == staticTypeInfos_.end()) {
      isStatic = false;
      break;
    }
    known.push_back(it->second);
    key += "," + it->second.global.getSymName().str() + "+" +
        std::to_string(it->second.tag);
  }
  std::size_t arguments{op.getArguments().size()};
  if (isStatic) {
    mlir::LLVM::GlobalOp &cell{typeInfoCells_[key]};
    if (!cell) {
      llvm::SmallVector<mlir::Type> words(arguments + 1, ptr);
      auto cellType{
          mlir::LLVM::LLVMStructType::getLiteral(op.getContext(), words)};
      mlir::OpBuilder::InsertionGuard guard{rewriter_};
      rewriter_.setInsertionPointToStart(module_.getBody());
      cell = mlir::LLVM::GlobalOp::create(rewriter_, loc, cellType,
          /*isConstant=*/true, mlir::LLVM::Linkage::Private,
          "merk.typeclass_info." + std::to_string(typeInfoCells_.size() - 1),
          mlir::Attribute{}, /*alignment=*/8);
      rewriter_.createBlock(&cell.getInitializerRegion());
      mlir::Value init{mlir::LLVM::ZeroOp::create(rewriter_, loc, cellType)};
      std::int64_t position{0};
      init = mlir::LLVM::InsertValueOp::create(rewriter_, loc, init,
          mlir::LLVM::AddressOfOp::create(rewriter_, loc, table),
          llvm::SmallVector<std::int64_t>{position++});
      for (const StaticTypeInfo &argument : known) {
        init = mlir::LLVM::InsertValueOp::create(rewriter_, loc, init,
            TypeInfoAddress(argument, loc),
            llvm::SmallVector<std::int64_t>{position++});
      }
      mlir::LLVM::ReturnOp::create(rewriter_, loc, init);
    }
    StaticTypeInfo info{cell, 0};
    mlir::Value value{TypeInfoAddress(info, loc)};
    staticTypeInfos_[value] = info;
    rewriter_.replaceOp(op, value);
    return llvm::success();
  }
  std::int64_t word{0};
  mlir::Value new_{mrt::AllocOp::create(
      rewriter_, loc, ptr, static_cast<std::int64_t>(arguments + 1))};
  mrt::FieldStoreOp::create(rewriter_, loc,
      mlir::LLVM::AddressOfOp::create(rewriter_, loc, table), new_, word++);
  for (mlir::Value argument : op.getArguments()) {
    mrt::FieldStoreOp::create(rewriter_, loc, argument, new_, word++);
  }
  rewriter_.replaceOp(op, new_);
  return llvm::success();
}

void Lowerer::LowerTypeclassInfoWord(merk::TypeclassInfoWordOp op) {
  mlir::Location loc{op.getLoc()};
  rewriter_.setInsertionPoint(op);
  auto ptr{mlir::LLVM::LLVMPointerType::get(op.getContext())};
  mlir::Value address{mlir::LLVM::GEPOp::create(rewriter_, loc, ptr,
      rewriter_.getI8Type(), op->getOperand(0),
      llvm::ArrayRef<mlir::LLVM::GEPArg>{
          static_cast<std::int32_t>(8 * op.getWord())})};
  rewriter_.replaceOpWithNewOp<mlir::LLVM::LoadOp>(op, ptr, address, 8);
}

// A closure's cell: its entry's code, the number of values it captured, and
// those values (design document, Lowering, "Closures").
void Lowerer::LowerClosure(merk::ClosureOp op) {
  mlir::Location loc{op.getLoc()};
  rewriter_.setInsertionPoint(op);
  auto ptr{mlir::LLVM::LLVMPointerType::get(op.getContext())};
  std::int64_t captured{static_cast<std::int64_t>(op.getCaptured().size())};
  mlir::Value cell{mrt::AllocOp::create(rewriter_, loc, ptr, captured + 2)};
  std::int64_t word{0};
  mrt::FieldStoreOp::create(rewriter_, loc,
      mlir::LLVM::AddressOfOp::create(
          rewriter_, loc, funcs_.lookup(op.getEntry())),
      cell, word++);
  mrt::FieldStoreOp::create(rewriter_, loc,
      mlir::LLVM::ConstantOp::create(
          rewriter_, loc, rewriter_.getI64Type(), captured),
      cell, word++);
  for (mlir::Value value : op->getOperands()) {
    mlir::Type type{Convert(value.getType())};
    if (type != value.getType()) {
      value = mlir::UnrealizedConversionCastOp::create(rewriter_, loc, type, value)
                  .getResult(0);
    }
    mrt::FieldStoreOp::create(rewriter_, loc, value, cell, word++);
  }
  rewriter_.replaceOp(op, cell);
}

void Lowerer::LowerClosureWord(merk::ClosureWordOp op) {
  mlir::Location loc{op.getLoc()};
  rewriter_.setInsertionPoint(op);
  auto ptr{mlir::LLVM::LLVMPointerType::get(op.getContext())};
  mlir::Value address{mlir::LLVM::GEPOp::create(rewriter_, loc, ptr,
      rewriter_.getI8Type(), op->getOperand(0),
      llvm::ArrayRef<mlir::LLVM::GEPArg>{
          static_cast<std::int32_t>(8 * (op.getIndex() + 2))})};
  rewriter_.replaceOpWithNewOp<mlir::LLVM::LoadOp>(
      op, rewriter_.getI64Type(), address, 8);
}

// The descriptors' kinds, as runtime/write.cpp reads them: a term's is its
// layout's.
static std::optional<unsigned> DescriptorKind(llvm::StringRef kind) {
  static constexpr llvm::StringLiteral kinds[]{"int", "uint", "int8", "int16",
      "int32", "int64", "uint8", "uint16", "uint32", "uint64", "float", "char",
      "string", "array", "dummy", "void", "enum", "notag", "cells", "tuple",
      "closure"};
  for (auto [i, name] : llvm::enumerate(kinds)) {
    if (kind == name) {
      return i;
    }
  }
  return std::nullopt;
}

// Each is { i8 kind, i8 flags, i16 arity, i32 functor count, ptr module,
// ptr name, ptr unify, ptr compare, ptr functors, [8 x i16] byTag,
// ptr constants, ptr shared }, and a functor { ptr name, i32 arity, i8 tag,
// i8 hidden, i32 secondary, i64 constant, ptr fields } (design document, "Type
// descriptors", and runtime/write.cpp, which reads them).
// A private constant global, whose initializer is made by `make`, which is given
// the zero value and returns the value.
mlir::LLVM::GlobalOp Lowerer::RttiTable(mlir::Type type, mlir::Location loc,
    llvm::function_ref<mlir::Value(mlir::Value)> make) {
  mlir::OpBuilder::InsertionGuard guard{rewriter_};
  rewriter_.setInsertionPointToStart(module_.getBody());
  auto global{mlir::LLVM::GlobalOp::create(rewriter_, loc, type,
      /*isConstant=*/true, mlir::LLVM::Linkage::Private,
      "merk.rtti." + std::to_string(rttiTables_++), mlir::Attribute{},
      /*alignment=*/8)};
  rewriter_.createBlock(&global.getInitializerRegion());
  mlir::Value zero{mlir::LLVM::ZeroOp::create(rewriter_, loc, type)};
  mlir::LLVM::ReturnOp::create(rewriter_, loc, make(zero));
  return global;
}

mlir::LLVM::GlobalOp Lowerer::I16Table(
    llvm::ArrayRef<std::int64_t> values, mlir::Location loc) {
  auto i16{rewriter_.getI16Type()};
  return RttiTable(mlir::LLVM::LLVMArrayType::get(i16, values.size()), loc,
      [&](mlir::Value init) {
        for (auto [i, value] : llvm::enumerate(values)) {
          init = mlir::LLVM::InsertValueOp::create(rewriter_, loc, init,
              mlir::LLVM::ConstantOp::create(rewriter_, loc, i16,
                  rewriter_.getI16IntegerAttr(
                      static_cast<std::int16_t>(value))),
              llvm::SmallVector<std::int64_t>{static_cast<std::int64_t>(i)});
        }
        return init;
      });
}

// A field's type, in an initializer: a type parameter's number, tagged, or a
// descriptor, or a static cell of one and its arguments' types, tagged, or
// where an existentially typed constructor's cell has the type_info of an
// existentially quantified type variable: its hidden field, and the word of
// it if that is a typeclass_info, tagged 3 (runtime/write.cpp). `cells` has
// the cells made so far.
mlir::Value Lowerer::FieldTypeValue(mlir::Attribute type,
    llvm::DenseMap<mlir::Attribute, mlir::LLVM::GlobalOp> &cells) {
  auto ptr{mlir::LLVM::LLVMPointerType::get(module_.getContext())};
  auto i64{rewriter_.getI64Type()};
  mlir::Location loc{rewriter_.getUnknownLoc()};
  if (auto number{mlir::dyn_cast<mlir::IntegerAttr>(type)}) {
    mlir::Value word{mlir::LLVM::ConstantOp::create(
        rewriter_, loc, i64, (number.getInt() << kTagBits) | 2)};
    return mlir::LLVM::IntToPtrOp::create(rewriter_, loc, ptr, word);
  }
  if (auto hidden{mlir::dyn_cast<mlir::DenseI64ArrayAttr>(type)}) {
    std::int64_t place{hidden[0] | (hidden[1] << 16)};
    mlir::Value word{mlir::LLVM::ConstantOp::create(
        rewriter_, loc, i64, (place << kTagBits) | 3)};
    return mlir::LLVM::IntToPtrOp::create(rewriter_, loc, ptr, word);
  }
  auto parts{mlir::cast<mlir::ArrayAttr>(type)};
  std::string constructor{
      mlir::cast<mlir::StringAttr>(parts[0]).getValue().str()};
  std::string symbol{"merk.type_ctor." + constructor};
  if (parts.size() == 1) {
    return mlir::LLVM::AddressOfOp::create(rewriter_, loc, ptr, symbol);
  }
  // One cell for each type, however many fields have it.
  mlir::LLVM::GlobalOp cell{cells.lookup(type)};
  if (!cell) {
    std::size_t arguments{parts.size() - 1};
    bool variableArity{VariableArity(constructor)};
    llvm::SmallVector<mlir::Type> words{ptr};
    if (variableArity) {
      words.push_back(i64);
    }
    words.append(arguments, ptr);
    auto cellType{
        mlir::LLVM::LLVMStructType::getLiteral(module_.getContext(), words)};
    cell = RttiTable(cellType, loc, [&](mlir::Value init) {
      std::int64_t position{0};
      AppendField(init, position,
          mlir::LLVM::AddressOfOp::create(rewriter_, loc, ptr, symbol), loc);
      if (variableArity) {
        AppendField(init, position,
            mlir::LLVM::ConstantOp::create(
                rewriter_, loc, i64, static_cast<std::int64_t>(arguments)),
            loc);
      }
      for (mlir::Attribute argument : parts.getValue().drop_front()) {
        AppendField(init, position, FieldTypeValue(argument, cells), loc);
      }
      return init;
    });
    cells[type] = cell;
  }
  mlir::Value base{mlir::LLVM::AddressOfOp::create(rewriter_, loc, cell)};
  return mlir::LLVM::GEPOp::create(rewriter_, loc, ptr, rewriter_.getI8Type(),
      base, llvm::ArrayRef<mlir::LLVM::GEPArg>{1});
}

llvm::LogicalResult Lowerer::LowerTypeCtors() {
  mlir::MLIRContext *context{module_.getContext()};
  auto ptr{mlir::LLVM::LLVMPointerType::get(context)};
  auto i8{rewriter_.getI8Type()};
  auto i16{rewriter_.getI16Type()};
  auto i32{rewriter_.getI32Type()};
  auto type{mlir::LLVM::LLVMStructType::getLiteral(context,
      {i8, i8, i16, i32, ptr, ptr, ptr, ptr, ptr,
          mlir::LLVM::LLVMArrayType::get(i16, kTags), ptr, ptr})};
  auto i64{rewriter_.getI64Type()};
  auto functorType{mlir::LLVM::LLVMStructType::getLiteral(
      context, {ptr, i32, i8, i8, i32, i64, ptr})};
  llvm::DenseMap<mlir::Attribute, mlir::LLVM::GlobalOp> fieldCells;
  for (auto op :
      llvm::make_early_inc_range(module_.getOps<merk::TypeCtorOp>())) {
    mlir::Location loc{op.getLoc()};
    std::string symbol{"merk.type_ctor." + op.getConstructor().str()};
    llvm::StringRef linkage{op.getLinkage()};
    mlir::OpBuilder::InsertionGuard guard{rewriter_};
    rewriter_.setInsertionPointToStart(module_.getBody());
    if (linkage == "declaration") {
      mlir::LLVM::GlobalOp::create(rewriter_, loc, type, /*isConstant=*/true,
          mlir::LLVM::Linkage::External, symbol, mlir::Attribute{},
          /*alignment=*/8);
      rewriter_.eraseOp(op);
      continue;
    }
    llvm::StringRef kindName{op.getKind()};
    if (kindName == "term") {
      const TermLayout *layout{
          op.getTerm() ? LayoutOf(*op.getTerm()) : nullptr};
      if (!layout) {
        return op.emitOpError("has no term with a merk.type_def");
      }
      kindName = layout->kind == TermLayout::Kind::Enum ? "enum"
          : layout->kind == TermLayout::Kind::NoTag     ? "notag"
                                                        : "cells";
    }
    std::optional<unsigned> kind{DescriptorKind(kindName)};
    if (!kind) {
      return op.emitOpError("has an unknown kind");
    }
    mlir::LLVM::GlobalOp moduleName{StringGlobal(op.getModule(), loc)};
    mlir::LLVM::GlobalOp name{StringGlobal(op.getName(), loc)};
    rewriter_.setInsertionPointToStart(module_.getBody());
    auto global{mlir::LLVM::GlobalOp::create(rewriter_, loc, type,
        /*isConstant=*/true,
        linkage == "external"       ? mlir::LLVM::Linkage::External
            : linkage == "internal" ? mlir::LLVM::Linkage::Internal
                                    : mlir::LLVM::Linkage::LinkonceODR,
        symbol, mlir::Attribute{}, /*alignment=*/8)};
    rewriter_.createBlock(&global.getInitializerRegion());
    mlir::Value init{mlir::LLVM::ZeroOp::create(rewriter_, loc, type)};
    std::int64_t position{0};
    AppendField(init, position,
        mlir::LLVM::ConstantOp::create(rewriter_, loc, i8, *kind), loc);
    AppendField(init, position,
        mlir::LLVM::ConstantOp::create(rewriter_, loc, i8, 0), loc);
    AppendField(init, position,
        mlir::LLVM::ConstantOp::create(rewriter_, loc, i16, op.getArity()),
        loc);
    llvm::ArrayRef<mlir::Attribute> functors;
    if (op.getFunctors()) {
      functors = op.getFunctors()->getValue();
    }
    AppendField(init, position,
        mlir::LLVM::ConstantOp::create(
            rewriter_, loc, i32, static_cast<std::int32_t>(functors.size())),
        loc);
    AppendField(init, position,
        mlir::LLVM::AddressOfOp::create(rewriter_, loc, moduleName), loc);
    AppendField(init, position,
        mlir::LLVM::AddressOfOp::create(rewriter_, loc, name), loc);
    if (op.getUnify()) {
      AppendField(init, position,
          mlir::LLVM::AddressOfOp::create(rewriter_, loc, ptr, *op.getUnify()),
          loc);
    }
    if (op.getCompare()) {
      AppendField(init, position,
          mlir::LLVM::AddressOfOp::create(
              rewriter_, loc, ptr, *op.getCompare()),
          loc);
    }
    position = 8;
    const TermLayout *layout{op.getTerm() ? LayoutOf(*op.getTerm()) : nullptr};
    // By the word, or a functor's tag, its constant's number or its secondary
    // tag, to its position in the order declared (runtime/write.cpp).
    constexpr std::int64_t kConstantTag{0xffff}, kSharedTag{0xfffe};
    llvm::SmallVector<std::int64_t, kTags> byTag(kTags, kConstantTag);
    llvm::SmallVector<std::int64_t> constants, shared;
    mlir::LLVM::GlobalOp functorTable;
    if (!functors.empty()) {
      if (layout && layout->functors.size() != functors.size()) {
        return op.emitOpError("has other functors than its term");
      }
      llvm::SmallVector<mlir::LLVM::GlobalOp> names, fields;
      for (mlir::Attribute attribute : functors) {
        auto parts{mlir::cast<mlir::ArrayAttr>(attribute)};
        names.push_back(StringGlobal(
            mlir::cast<mlir::StringAttr>(parts[0]).getValue(), loc));
        llvm::ArrayRef<mlir::Attribute> types{parts.getValue().drop_front()};
        fields.push_back(types.empty()
                ? mlir::LLVM::GlobalOp{}
                : RttiTable(mlir::LLVM::LLVMArrayType::get(ptr, types.size()),
                      loc, [&](mlir::Value init) {
                        for (auto [i, type] : llvm::enumerate(types)) {
                          init = mlir::LLVM::InsertValueOp::create(rewriter_,
                              loc, init, FieldTypeValue(type, fieldCells),
                              llvm::SmallVector<std::int64_t>{
                                  static_cast<std::int64_t>(i)});
                        }
                        return init;
                      }));
      }
      functorTable = RttiTable(
          mlir::LLVM::LLVMArrayType::get(functorType, functors.size()), loc,
          [&](mlir::Value init) {
            for (auto [i, attribute] : llvm::enumerate(functors)) {
              const TermLayout::Functor *functor{
                  layout ? &layout->functors[i] : nullptr};
              std::int64_t index{static_cast<std::int64_t>(i)};
              std::size_t arity{
                  mlir::cast<mlir::ArrayAttr>(attribute).size() - 1};
              init = InsertAt(init, {index, 0},
                  mlir::LLVM::AddressOfOp::create(rewriter_, loc, names[i]),
                  loc);
              init = InsertAt(init, {index, 1},
                  mlir::LLVM::ConstantOp::create(
                      rewriter_, loc, i32, static_cast<std::int32_t>(arity)),
                  loc);
              init = InsertAt(init, {index, 2},
                  mlir::LLVM::ConstantOp::create(rewriter_, loc, i8,
                      functor ? static_cast<std::int32_t>(functor->tag) : 0),
                  loc);
              init = InsertAt(init, {index, 3},
                  mlir::LLVM::ConstantOp::create(rewriter_, loc, i8,
                      functor ? static_cast<std::int32_t>(functor->hidden) : 0),
                  loc);
              init = InsertAt(init, {index, 4},
                  mlir::LLVM::ConstantOp::create(rewriter_, loc, i32,
                      functor && functor->secondary
                          ? static_cast<std::int32_t>(*functor->secondary)
                          : -1),
                  loc);
              init = InsertAt(init, {index, 5},
                  mlir::LLVM::ConstantOp::create(
                      rewriter_, loc, i64, functor ? functor->value : 0),
                  loc);
              if (fields[i]) {
                init = InsertAt(init, {index, 6},
                    mlir::LLVM::AddressOfOp::create(rewriter_, loc, fields[i]),
                    loc);
              }
            }
            return init;
          });
      if (layout && layout->kind == TermLayout::Kind::Cells) {
        for (auto [i, functor] : llvm::enumerate(layout->functors)) {
          if (functor.arity == 0) {
            constants.push_back(i);
          } else if (functor.secondary) {
            byTag[functor.tag] = kSharedTag;
            shared.resize(*functor.secondary + 1);
            shared[*functor.secondary] = i;
          } else {
            byTag[functor.tag] = i;
          }
        }
      }
    }
    if (functorTable) {
      AppendField(init, position,
          mlir::LLVM::AddressOfOp::create(rewriter_, loc, functorTable), loc);
    } else {
      ++position;
    }
    for (auto [i, value] : llvm::enumerate(byTag)) {
      init = mlir::LLVM::InsertValueOp::create(rewriter_, loc, init,
          mlir::LLVM::ConstantOp::create(rewriter_, loc, i16,
              rewriter_.getI16IntegerAttr(static_cast<std::int16_t>(value))),
          llvm::SmallVector<std::int64_t>{9, static_cast<std::int64_t>(i)});
    }
    position = 10;
    if (!constants.empty()) {
      AppendField(init, position,
          mlir::LLVM::AddressOfOp::create(
              rewriter_, loc, I16Table(constants, loc)),
          loc);
    } else {
      ++position;
    }
    if (!shared.empty()) {
      AppendField(init, position,
          mlir::LLVM::AddressOfOp::create(
              rewriter_, loc, I16Table(shared, loc)),
          loc);
    }
    mlir::LLVM::ReturnOp::create(rewriter_, loc, init);
    rewriter_.eraseOp(op);
  }
  llvm::SmallVector<merk::TypeInfoOp> infos;
  module_.walk([&](merk::TypeInfoOp op) { infos.push_back(op); });
  for (merk::TypeInfoOp op : infos) {
    if (!module_.lookupSymbol<mlir::LLVM::GlobalOp>(
            "merk.type_ctor." + op.getConstructor().str())) {
      return op.emitOpError("has no merk.type_ctor");
    }
  }
  return llvm::success();
}

void Lowerer::LowerTypeInfoWord(merk::TypeInfoWordOp op) {
  mlir::Location loc{op.getLoc()};
  rewriter_.setInsertionPoint(op);
  auto ptr{mlir::LLVM::LLVMPointerType::get(op.getContext())};
  mlir::Value address{mlir::LLVM::GEPOp::create(rewriter_, loc, ptr,
      rewriter_.getI8Type(), op->getOperand(0),
      llvm::ArrayRef<mlir::LLVM::GEPArg>{
          static_cast<std::int32_t>(8 * op.getWord() - 1)})};
  rewriter_.replaceOpWithNewOp<mlir::LLVM::LoadOp>(op, ptr, address, 8);
}

mlir::Value Lowerer::CallDescriptorEntry(
    mlir::Operation *op, std::int32_t offset, mlir::Type result) {
  mlir::Location loc{op->getLoc()};
  rewriter_.setInsertionPoint(op);
  auto ptr{mlir::LLVM::LLVMPointerType::get(op->getContext())};
  auto i64{rewriter_.getI64Type()};
  mlir::Value info{op->getOperand(0)};
  // The descriptor: the type_info with tag 0, and a cell's first word with
  // tag 1.
  mlir::Value word{mlir::LLVM::PtrToIntOp::create(rewriter_, loc, i64, info)};
  mlir::Value isDescriptor{mlir::LLVM::ICmpOp::create(rewriter_, loc,
      mlir::LLVM::ICmpPredicate::eq,
      mlir::LLVM::AndOp::create(rewriter_, loc, word,
          mlir::LLVM::ConstantOp::create(rewriter_, loc, i64, kTags - 1)),
      mlir::LLVM::ConstantOp::create(rewriter_, loc, i64, 0))};
  mlir::Block *before{rewriter_.getInsertionBlock()};
  mlir::Block *join{rewriter_.splitBlock(before, rewriter_.getInsertionPoint())};
  mlir::Value descriptor{join->addArgument(ptr, loc)};
  mlir::Block *cell{rewriter_.createBlock(join)};
  mlir::Value first{mlir::LLVM::LoadOp::create(rewriter_, loc, ptr,
      mlir::LLVM::GEPOp::create(rewriter_, loc, ptr, rewriter_.getI8Type(), info,
          llvm::ArrayRef<mlir::LLVM::GEPArg>{-1}),
      8)};
  mlir::LLVM::BrOp::create(rewriter_, loc, mlir::ValueRange{first}, join);
  rewriter_.setInsertionPointToEnd(before);
  mlir::LLVM::CondBrOp::create(rewriter_, loc, isDescriptor, join,
      mlir::ValueRange{info}, cell, mlir::ValueRange{});
  rewriter_.setInsertionPointToStart(join);
  mlir::Value entry{mlir::LLVM::LoadOp::create(rewriter_, loc, ptr,
      mlir::LLVM::GEPOp::create(rewriter_, loc, ptr, rewriter_.getI8Type(),
          descriptor, llvm::ArrayRef<mlir::LLVM::GEPArg>{offset}),
      8)};
  auto type{mlir::LLVM::LLVMFunctionType::get(result, {ptr, i64, i64})};
  auto call{mlir::LLVM::CallOp::create(rewriter_, loc, type,
      mlir::ValueRange{entry, info, op->getOperand(1), op->getOperand(2)})};
  call.setCConv(cconv_);
  return call.getResult();
}

llvm::LogicalResult Lowerer::LowerUnifyWord(merk::UnifyWordOp op) {
  if (device_) {
    return op.emitOpError(
        "compares values of a type variable, which cannot be done on an OpenMP device yet");
  }
  // Past the header's word, the module's and the name's.
  mlir::Value result{CallDescriptorEntry(op, 24, rewriter_.getI1Type())};
  rewriter_.replaceOp(op, result);
  return llvm::success();
}

llvm::LogicalResult Lowerer::LowerCompareWord(merk::CompareWordOp op) {
  if (device_) {
    return op.emitOpError(
        "compares values of a type variable, which cannot be done on an OpenMP device yet");
  }
  // Past the unify entry too.
  mlir::Value result{CallDescriptorEntry(op, 32, Convert(op.getType()))};
  rewriter_.replaceOp(op, result);
  return llvm::success();
}

void Lowerer::LowerStringCompare(merk::StringCompareOp op) {
  rewriter_.setInsertionPoint(op);
  auto ptr{mlir::LLVM::LLVMPointerType::get(op.getContext())};
  auto i32{rewriter_.getI32Type()};
  auto call{mlir::LLVM::CallOp::create(rewriter_, op.getLoc(),
      RuntimeFunc("strcmp", i32, {ptr, ptr}),
      mlir::ValueRange{op->getOperand(0), op->getOperand(1)})};
  rewriter_.replaceOp(op, call.getResult());
}

void Lowerer::LowerCompareTypeInfo(merk::CompareTypeInfoOp op) {
  rewriter_.setInsertionPoint(op);
  auto ptr{mlir::LLVM::LLVMPointerType::get(op.getContext())};
  auto i32{rewriter_.getI32Type()};
  auto call{mlir::LLVM::CallOp::create(rewriter_, op.getLoc(),
      RuntimeFunc("merk_compare_type_infos", i32, {ptr, ptr}),
      mlir::ValueRange{op->getOperand(0), op->getOperand(1)})};
  rewriter_.replaceOp(op, call.getResult());
}

void Lowerer::LowerStringEqual(merk::StringEqualOp op) {
  mlir::Location loc{op.getLoc()};
  rewriter_.setInsertionPoint(op);
  auto ptr{mlir::LLVM::LLVMPointerType::get(op.getContext())};
  auto i32{rewriter_.getI32Type()};
  auto call{mlir::LLVM::CallOp::create(rewriter_, loc,
      RuntimeFunc("strcmp", i32, {ptr, ptr}),
      mlir::ValueRange{op->getOperand(0), op->getOperand(1)})};
  rewriter_.replaceOpWithNewOp<mlir::LLVM::ICmpOp>(op,
      mlir::LLVM::ICmpPredicate::eq, call.getResult(),
      mlir::LLVM::ConstantOp::create(rewriter_, loc, i32, 0));
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
    for (mlir::Type type : outputs) {
      slots.push_back(Alloca(call, type, loc));
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
  rewriter_.setInsertionPoint(call);
  llvm::SmallVector<mlir::Value> arguments{call.getInputs()};
  return CallCode(call, funcs_.lookup(call.getCallee()), nullptr, nullptr,
      arguments, CanFail(call.getCallee()), InMemory(call.getCallee()), tail);
}

mlir::Value Lowerer::CallMethod(merk::CallMethodOp call, bool *tail) {
  mlir::Location loc{call.getLoc()};
  rewriter_.setInsertionPoint(call);
  mlir::MLIRContext *context{call.getContext()};
  auto ptr{mlir::LLVM::LLVMPointerType::get(context)};
  // The table is the cell's first word.
  mlir::Value info{call->getOperand(0)};
  mlir::Value table{mlir::LLVM::LoadOp::create(rewriter_, loc, ptr, info, 8)};
  mlir::Value slot{mlir::LLVM::GEPOp::create(rewriter_, loc, ptr,
      rewriter_.getI8Type(), table,
      llvm::ArrayRef<mlir::LLVM::GEPArg>{
          static_cast<std::int32_t>(8 * call.getMethod())})};
  mlir::Value code{mlir::LLVM::LoadOp::create(rewriter_, loc, ptr, slot, 8)};
  return CallPointer(call, code, info, call.getInputs(),
      codegen::CanFail(call.getDeterminism()), tail);
}

mlir::Value Lowerer::CallClosure(merk::CallClosureOp call, bool *tail) {
  mlir::Location loc{call.getLoc()};
  rewriter_.setInsertionPoint(call);
  auto ptr{mlir::LLVM::LLVMPointerType::get(call.getContext())};
  // The entry's code is the cell's first word.
  mlir::Value closure{call->getOperand(0)};
  mlir::Value code{mlir::LLVM::LoadOp::create(rewriter_, loc, ptr, closure, 8)};
  return CallPointer(call, code, closure, call.getInputs(),
      codegen::CanFail(call.getDeterminism()), tail);
}

// A continuation takes its environment, then the solution's values, and
// returns whether to stop, as a semidet procedure's success. The operands
// are taken by place, as those that are the procedure's arguments are
// pointers already.
mlir::Value Lowerer::CallCont(merk::CallContOp call, bool *tail) {
  rewriter_.setInsertionPoint(call);
  return CallPointer(call, call->getOperand(0), call->getOperand(1),
      call->getOperands().drop_front(2), true, tail);
}

void Lowerer::LowerContRef(merk::ContRefOp op) {
  rewriter_.setInsertionPoint(op);
  mlir::LLVM::LLVMFuncOp func{funcs_.lookup(op.getCallee())};
  rewriter_.replaceOpWithNewOp<mlir::LLVM::AddressOfOp>(op, func);
}

// An environment is a struct of its layout in the frame's allocas.
mlir::LLVM::LLVMStructType Lowerer::EnvType(mlir::ArrayAttr layout) {
  llvm::SmallVector<mlir::Type> types;
  for (mlir::Attribute type : layout) {
    types.push_back(Convert(mlir::cast<mlir::TypeAttr>(type).getValue()));
  }
  return mlir::LLVM::LLVMStructType::getLiteral(module_.getContext(), types);
}

void Lowerer::LowerEnvNew(merk::EnvNewOp op) {
  mlir::Location loc{op.getLoc()};
  auto type{EnvType(op.getLayout())};
  mlir::Value env{Alloca(op, type, loc)};
  rewriter_.setInsertionPoint(op);
  for (auto [i, value] : llvm::enumerate(op.getValues())) {
    mlir::LLVM::StoreOp::create(
        rewriter_, loc, value, OutputAddress(env, type, i, loc));
  }
  rewriter_.replaceOp(op, env);
}

void Lowerer::LowerEnvGet(merk::EnvGetOp op) {
  mlir::Location loc{op.getLoc()};
  rewriter_.setInsertionPoint(op);
  auto type{EnvType(op.getLayout())};
  rewriter_.replaceOpWithNewOp<mlir::LLVM::LoadOp>(op,
      type.getBody()[op.getIndex()],
      OutputAddress(op->getOperand(0), type, op.getIndex(), loc));
}

void Lowerer::LowerEnvSet(merk::EnvSetOp op) {
  mlir::Location loc{op.getLoc()};
  rewriter_.setInsertionPoint(op);
  auto type{EnvType(op.getLayout())};
  rewriter_.replaceOpWithNewOp<mlir::LLVM::StoreOp>(op, op.getValue(),
      OutputAddress(op->getOperand(0), type, op.getIndex(), loc));
}

static bool IsNarrowInteger(mlir::Type type) {
  auto integer{mlir::dyn_cast<mlir::IntegerType>(type)};
  return integer && integer.getWidth() < 64;
}

// Fills a hole, as merk.construct would have stored the value there. The
// store is into a cell just made, which a collection may have moved since:
// the hole's address is a pointer derived from the cell's.
void Lowerer::LowerRefStore(merk::RefStoreOp op) {
  mlir::Location loc{op.getLoc()};
  rewriter_.setInsertionPoint(op);
  mlir::Value value{LoweredValue(op->getOperand(0), loc)};
  if (IsNarrowInteger(value.getType())) {
    value = mlir::LLVM::ZExtOp::create(
        rewriter_, loc, rewriter_.getI64Type(), value);
  }
  mlir::LLVM::StoreOp::create(
      rewriter_, loc, value, LoweredValue(op->getOperand(1), loc), 8);
  rewriter_.eraseOp(op);
}

mlir::Value Lowerer::CallPointer(mlir::Operation *call, mlir::Value code,
    mlir::Value first, mlir::ValueRange rest, bool canFail, bool *tail) {
  mlir::MLIRContext *context{call->getContext()};
  auto ptr{mlir::LLVM::LLVMPointerType::get(context)};
  bool inMemory{InMemory(call->getResultTypes(), canFail)};
  llvm::SmallVector<mlir::Value> arguments{first};
  arguments.append(rest.begin(), rest.end());
  llvm::SmallVector<mlir::Type> inputs;
  if (inMemory) {
    inputs.push_back(ptr);
  }
  inputs.push_back(ptr);
  llvm::SmallVector<mlir::Type> converted{Convert(rest.getTypes())};
  inputs.append(converted.begin(), converted.end());
  auto funcType{mlir::LLVM::LLVMFunctionType::get(
      inMemory ? ResultType(context, {}, canFail)
               : ResultType(context, call->getResultTypes(), canFail),
      inputs)};
  return CallCode(call, nullptr, code, funcType, arguments, canFail, inMemory, tail);
}

mlir::Value Lowerer::CallCode(mlir::Operation *call, mlir::LLVM::LLVMFuncOp func,
    mlir::Value pointer, mlir::LLVM::LLVMFunctionType funcType,
    llvm::SmallVector<mlir::Value> arguments, bool canFail, bool inMemory,
    bool *tail) {
  mlir::Location loc{call->getLoc()};
  // A call in tail position, whose solution is the procedure's, is a
  // tail call (design document, "Calling convention").
  llvm::SmallVector<mlir::Operation *> between;
  merk::SucceedOp succeed{TailSucceed(call, between)};
  auto caller{call->getParentOfType<mlir::LLVM::LLVMFuncOp>()};
  // Only where failing returns from the procedure: in a disjunction's
  // branch, it goes to the next branch. A call passing an environment in
  // the caller's frame needs the frame after the call.
  bool isTail{succeed && CanFail(caller.getSymName()) == canFail &&
      fail_ == topFail_ &&
      llvm::none_of(call->getOperands(), [](mlir::Value operand) {
        return operand.getDefiningOp<merk::EnvNewOp>() ||
            operand.getDefiningOp<mlir::LLVM::AllocaOp>();
      })};
  // A callee's outputs in memory are where the caller's go, for a tail
  // call, whose caller has the same outputs, and otherwise in a slot.
  mlir::Value outputs;
  auto outputsType{OutputsType(call->getResultTypes())};
  if (inMemory) {
    outputs = isTail ? caller.getArgument(0)
                     : Alloca(call, outputsType, loc);
    arguments.insert(arguments.begin(), outputs);
  }
  mlir::LLVM::CallOp llvmCall;
  if (func) {
    llvmCall = mlir::LLVM::CallOp::create(rewriter_, loc, func, arguments);
  } else {
    arguments.insert(arguments.begin(), pointer);
    llvmCall = mlir::LLVM::CallOp::create(rewriter_, loc, funcType, arguments);
  }
  llvmCall.setCConv(cconv_);
  mlir::Value result{
      llvmCall.getNumResults() ? llvmCall.getResult() : mlir::Value{}};
  if (isTail) {
    llvmCall.setTailCallKind(mlir::LLVM::TailCallKind::MustTail);
    rewriter_.setInsertionPoint(succeed);
    rewriter_.replaceOpWithNewOp<mlir::LLVM::ReturnOp>(
        succeed, result ? mlir::ValueRange{result} : mlir::ValueRange{});
    // The call's results have no uses but the conversions' and the return's.
    for (mlir::Operation *op : llvm::reverse(between)) {
      rewriter_.eraseOp(op);
    }
    rewriter_.eraseOp(call);
    if (tail) {
      *tail = true;
    }
    return nullptr;
  }
  llvm::SmallVector<mlir::Value> values;
  if (inMemory) {
    for (auto [i, type] : llvm::enumerate(outputsType.getBody())) {
      values.push_back(mlir::LLVM::LoadOp::create(rewriter_, loc, type,
          OutputAddress(outputs, outputsType, i, loc)));
    }
    rewriter_.replaceOp(call, values);
    return canFail ? result : nullptr;
  }
  std::size_t count{call->getNumResults()};
  if (!canFail && count <= 1) {
    values.append(llvmCall->result_begin(), llvmCall->result_end());
  } else {
    for (unsigned i = 0; i < count; ++i) {
      values.push_back(mlir::LLVM::ExtractValueOp::create(rewriter_, loc,
          result, llvm::ArrayRef<int64_t>{static_cast<int64_t>(i + canFail)}));
    }
  }
  rewriter_.replaceOp(call, values);
  if (!canFail) {
    return nullptr;
  }
  return count ? mlir::Value{mlir::LLVM::ExtractValueOp::create(
                       rewriter_, loc, result, llvm::ArrayRef<int64_t>{0})}
                 : result;
}

merk::SucceedOp Lowerer::TailSucceed(mlir::Operation *call,
    llvm::SmallVectorImpl<mlir::Operation *> &between) {
  mlir::Operation *op{call->getNextNode()};
  while (op && mlir::isa<merk::ToWordOp, merk::FromWordOp>(op)) {
    between.push_back(op);
    op = op->getNextNode();
  }
  auto succeed{mlir::dyn_cast_or_null<merk::SucceedOp>(op)};
  if (!succeed || succeed.getOutputs().size() != call->getNumResults()) {
    return nullptr;
  }
  for (auto [output, result] :
      llvm::zip(succeed.getOutputs(), call->getResults())) {
    // A word that came from a value of the same bits gives back the value.
    mlir::Value value{output};
    while (value != result) {
      mlir::Operation *from{value.getDefiningOp()};
      if (!from || !llvm::is_contained(between, from)) {
        return nullptr;
      }
      value = from->getOperand(0);
    }
    if (Convert(output.getType()) != Convert(result.getType())) {
      return nullptr;
    }
  }
  // Nothing else uses what goes with the call.
  llvm::SmallVector<mlir::Operation *> defining{between.begin(), between.end()};
  defining.push_back(call);
  for (mlir::Operation *def : defining) {
    for (mlir::Operation *user : def->getUsers()) {
      if (user != succeed && !llvm::is_contained(between, user)) {
        return nullptr;
      }
    }
  }
  return succeed;
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
// succeed, after converting values to words and back, so that a call before
// is in tail position.
void Lowerer::Exit(mlir::Block *block, mlir::Block *after) {
  auto yield{mlir::dyn_cast<merk::YieldOp>(block->back())};
  if (!yield) {
    return; // it fails, or succeeds
  }
  rewriter_.setInsertionPoint(yield);
  mlir::Operation *last{&after->front()};
  while (mlir::isa<merk::ToWordOp, merk::FromWordOp, mlir::arith::ConstantOp>(
      last)) {
    last = last->getNextNode();
  }
  if (mlir::isa<merk::SucceedOp>(last)) {
    mlir::IRMapping mapping;
    mapping.map(after->getArguments(), yield.getValues());
    for (mlir::Operation &op : *after) {
      rewriter_.clone(op, mapping);
      if (&op == last) {
        break;
      }
    }
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
      fail_ = fail;
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
      if (auto call{mlir::dyn_cast<merk::CallMethodOp>(op)}) {
        bool tail{false};
        if (mlir::Value succeeded{CallMethod(call, &tail)}) {
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
      if (auto call{mlir::dyn_cast<merk::CallContOp>(op)}) {
        bool tail{false};
        if (mlir::Value succeeded{CallCont(call, &tail)}) {
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
      if (auto disj{mlir::dyn_cast<merk::DisjOp>(op)}) {
        // Each branch in turn: one that fails goes to the next, and the
        // last to where the disjunction's failure goes.
        mlir::Block *anchor{SplitAfter(disj)};
        llvm::SmallVector<mlir::Block *> branches;
        for (mlir::Region &region : disj.getBranches()) {
          branches.push_back(Inline(region, anchor));
        }
        rewriter_.setInsertionPoint(disj);
        rewriter_.replaceOpWithNewOp<mlir::LLVM::BrOp>(disj, branches.front());
        rewriter_.eraseBlock(anchor);
        for (auto [i, branch] : llvm::enumerate(branches)) {
          Flatten(branch, i + 1 < branches.size() ? branches[i + 1] : fail);
        }
        break;
      }
      if (auto call{mlir::dyn_cast<merk::CallClosureOp>(op)}) {
        bool tail{false};
        if (mlir::Value succeeded{CallClosure(call, &tail)}) {
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
        if (const TermLayout *layout{termSwitches_.lookup(switchOp)}) {
          llvm::SmallVector<std::pair<const TermLayout::Functor *, mlir::Block *>>
              cases;
          for (auto [c, block] : llvm::zip(switchOp.getCases(), blocks)) {
            for (mlir::Attribute functor : mlir::cast<mlir::ArrayAttr>(c)) {
              cases.push_back(
                  {layout->Find(mlir::cast<merk::FunctorAttr>(functor)), block});
            }
          }
          rewriter_.setInsertionPoint(switchOp);
          Dispatch(switchOp.getValue(), *layout, cases, otherwise, after, loc);
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
  topFail_ = fail;
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

mlir::Value Lowerer::CallRuntime(llvm::StringRef name, mlir::Type result,
    mlir::ValueRange arguments, mlir::Location loc) {
  llvm::SmallVector<mlir::Type> types{llvm::map_to_vector(
      arguments, [this](mlir::Value v) { return Convert(v.getType()); })};
  return mlir::LLVM::CallOp::create(
      rewriter_, loc, RuntimeFunc(name, result, types), arguments)
      .getResult();
}

// An array operation's initial element as a word. An absent one: any word.
mlir::Value Lowerer::InitialElement(
    mlir::Operation *op, unsigned operand, mlir::Location loc) {
  return op->getNumOperands() > operand
      ? ToWord(op->getOperand(operand), loc)
      : mlir::LLVM::ConstantOp::create(
            rewriter_, loc, rewriter_.getI64Type(), 0);
}

void Lowerer::LowerArray(mlir::Operation *op) {
  mlir::Location loc{op->getLoc()};
  rewriter_.setInsertionPoint(op);
  auto ptr{mlir::LLVM::LLVMPointerType::get(rewriter_.getContext())};
  auto i64{rewriter_.getI64Type()};
  // The operands, not the accessors: an array the procedure was passed is
  // already a pointer.
  mlir::Value result;
  if (mlir::isa<merk::ArrayNewOp>(op)) {
    result = CallRuntime("merk_array_new", ptr,
        {op->getOperand(0), InitialElement(op, 1, loc)}, loc);
  } else if (mlir::isa<merk::ArraySizeOp>(op)) {
    result = mlir::LLVM::LoadOp::create(rewriter_, loc, i64, op->getOperand(0));
  } else if (mlir::isa<merk::ArrayGetOp>(op)) {
    mlir::Value word{mlir::LLVM::LoadOp::create(rewriter_, loc, i64,
        ElementAddress(op->getOperand(0), op->getOperand(1), loc))};
    result = FromWord(word, Convert(op->getResult(0).getType()), loc);
  } else if (mlir::isa<merk::ArraySetOp>(op)) {
    mlir::LLVM::StoreOp::create(rewriter_, loc, ToWord(op->getOperand(2), loc),
        ElementAddress(op->getOperand(0), op->getOperand(1), loc));
    result = op->getOperand(0);
  } else if (mlir::isa<merk::ArrayFillOp>(op)) {
    result = CallRuntime("merk_array_fill", ptr,
        {op->getOperand(0), op->getOperand(1), op->getOperand(2),
            ToWord(op->getOperand(3), loc)},
        loc);
  } else if (mlir::isa<merk::ArrayCopyOp>(op)) {
    result = CallRuntime("merk_array_copy", ptr, {op->getOperand(0)}, loc);
  } else {
    assert(mlir::isa<merk::ArrayResizeOp>(op));
    result = CallRuntime("merk_array_resize", ptr,
        {op->getOperand(0), op->getOperand(1), InitialElement(op, 2, loc)},
        loc);
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

mlir::LLVM::GlobalOp Lowerer::HeapGlobal() {
  if (!heap_) {
    mlir::OpBuilder::InsertionGuard guard{rewriter_};
    rewriter_.setInsertionPointToStart(module_.getBody());
    auto ptr{mlir::LLVM::LLVMPointerType::get(module_.getContext())};
    heap_ = mlir::LLVM::GlobalOp::create(rewriter_, module_.getLoc(),
        mlir::LLVM::LLVMStructType::getLiteral(module_.getContext(), {ptr, ptr}),
        /*isConstant=*/false, mlir::LLVM::Linkage::External, "merk_heap",
        mlir::Attribute{}, /*alignment=*/8, /*addrSpace=*/0,
        /*dsoLocal=*/false, mlir::LLVM::ThreadLocalMode::GeneralDynamic);
  }
  return heap_;
}

// An allocation bumps the heap's pointer if the cell fits before its limit,
// and otherwise calls the runtime (runtime/heap.h).
void Lowerer::LowerAlloc(mrt::AllocOp alloc) {
  mlir::Location loc{alloc.getLoc()};
  auto ptr{mlir::LLVM::LLVMPointerType::get(module_.getContext())};
  auto i64{rewriter_.getI64Type()};
  std::int64_t bytes{8 * static_cast<std::int64_t>(alloc.getWords())};
  mlir::Block *before{alloc->getBlock()};
  mlir::Block *after{rewriter_.splitBlock(before, alloc->getIterator())};
  mlir::Value cell{after->addArgument(ptr, loc)};
  rewriter_.replaceAllUsesWith(alloc.getResult(), cell);
  rewriter_.eraseOp(alloc);
  mlir::Block *fast{rewriter_.createBlock(after)};
  mlir::Block *slow{rewriter_.createBlock(after)};
  rewriter_.setInsertionPointToEnd(before);
  auto heapType{HeapGlobal().getType()};
  mlir::Value heap{mlir::LLVM::ThreadlocalAddressOp::create(rewriter_, loc, ptr,
      mlir::LLVM::AddressOfOp::create(rewriter_, loc, HeapGlobal()))};
  mlir::Value next{mlir::LLVM::LoadOp::create(rewriter_, loc, ptr, heap, 8)};
  mlir::Value end{mlir::LLVM::GEPOp::create(rewriter_, loc, ptr,
      rewriter_.getI8Type(), next,
      llvm::ArrayRef<mlir::LLVM::GEPArg>{static_cast<std::int32_t>(bytes)})};
  mlir::Value limit{mlir::LLVM::LoadOp::create(rewriter_, loc, ptr,
      mlir::LLVM::GEPOp::create(rewriter_, loc, ptr, heapType, heap,
          llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 1}),
      8)};
  mlir::Value fits{mlir::LLVM::ICmpOp::create(
      rewriter_, loc, mlir::LLVM::ICmpPredicate::ule, end, limit)};
  mlir::LLVM::CondBrOp::create(rewriter_, loc, fits, fast, mlir::ValueRange{},
      slow, mlir::ValueRange{}, std::make_pair(2000u, 1u));
  rewriter_.setInsertionPointToStart(fast);
  mlir::LLVM::StoreOp::create(rewriter_, loc, end, heap, 8);
  mlir::LLVM::BrOp::create(rewriter_, loc, mlir::ValueRange{next}, after);
  rewriter_.setInsertionPointToStart(slow);
  mlir::Value slowCell{mlir::LLVM::CallOp::create(rewriter_, loc,
      RuntimeFunc("merk_alloc_slow", ptr, {i64}),
      mlir::ValueRange{mlir::LLVM::ConstantOp::create(rewriter_, loc, i64, bytes)})
          .getResult()};
  mlir::LLVM::BrOp::create(rewriter_, loc, mlir::ValueRange{slowCell}, after);
}

// A tag is added to a cell's address, and a field is at its word's offset,
// so that a field of a tagged pointer is one getelementptr from it.
mlir::Value Lowerer::OffsetBytes(
    mlir::Value address, std::int64_t bytes, mlir::Location loc) {
  if (bytes == 0) {
    return address;
  }
  return mlir::LLVM::GEPOp::create(rewriter_, loc,
      mlir::LLVM::LLVMPointerType::get(module_.getContext()),
      rewriter_.getI8Type(), address,
      llvm::ArrayRef<mlir::LLVM::GEPArg>{static_cast<std::int32_t>(bytes)});
}

void Lowerer::LowerMrt() {
  auto i64{rewriter_.getI64Type()};
  llvm::SmallVector<mlir::Operation *> ops;
  module_.walk([&](mlir::Operation *op) {
    if (mlir::isa_and_present<mrt::MrtDialect>(op->getDialect())) {
      ops.push_back(op);
    }
  });
  for (mlir::Operation *op : ops) {
    mlir::Location loc{op->getLoc()};
    rewriter_.setInsertionPoint(op);
    llvm::TypeSwitch<mlir::Operation *>(op)
        .Case([&](mrt::AllocOp alloc) { LowerAlloc(alloc); })
        .Case([&](mrt::TagOp tag) {
          rewriter_.replaceOp(tag,
              OffsetBytes(
                  tag.getCell(), static_cast<std::int64_t>(tag.getTag()), loc));
        })
        .Case([&](mrt::UntagOp untag) {
          rewriter_.replaceOp(untag,
              OffsetBytes(untag.getTerm(),
                  -static_cast<std::int64_t>(untag.getTag()), loc));
        })
        .Case([&](mrt::PrimaryTagOp tag) {
          mlir::Value word{mlir::LLVM::PtrToIntOp::create(
              rewriter_, loc, i64, tag.getTerm())};
          rewriter_.replaceOpWithNewOp<mlir::LLVM::AndOp>(tag, word,
              mlir::LLVM::ConstantOp::create(rewriter_, loc, i64, kTags - 1));
        })
        .Case([&](mrt::FieldLoadOp load) {
          // A narrow field is its whole word (Type representation, "Words").
          mlir::Type type{load.getValue().getType()};
          mlir::Value address{OffsetBytes(load.getCell(),
              8 * static_cast<std::int64_t>(load.getWord()), loc)};
          if (IsNarrowInteger(type)) {
            mlir::Value word{
                mlir::LLVM::LoadOp::create(rewriter_, loc, i64, address, 8)};
            rewriter_.replaceOpWithNewOp<mlir::LLVM::TruncOp>(load, type, word);
          } else {
            rewriter_.replaceOpWithNewOp<mlir::LLVM::LoadOp>(
                load, type, address, 8);
          }
        })
        .Case([&](mrt::FieldStoreOp store) {
          mlir::Value value{store.getValue()};
          if (IsNarrowInteger(value.getType())) {
            value = mlir::LLVM::ZExtOp::create(rewriter_, loc, i64, value);
          }
          mlir::LLVM::StoreOp::create(rewriter_, loc, value,
              OffsetBytes(store.getCell(),
                  8 * static_cast<std::int64_t>(store.getWord()), loc),
              8);
          rewriter_.eraseOp(store);
        });
  }
}

// Reports an operation that does `what`, which needs the host's runtime, if it
// would run on a device: in a target region or a procedure one calls. Whether
// it did.
static bool RejectOnDevice(mlir::Operation *op, llvm::StringRef what) {
  auto proc{op->getParentOfType<merk::ProcOp>()};
  if (!op->getParentOfType<mlir::omp::TargetOp>() &&
      !(proc && proc->getDiscardableAttr("omp.declare_target"))) {
    return false;
  }
  op->emitOpError(what) << ", which cannot be done on an OpenMP device yet";
  return true;
}

llvm::LogicalResult Lowerer::Lower() {
  module_.getContext()->loadDialect<mlir::LLVM::LLVMDialect, mrt::MrtDialect>();
  // On a device nothing calls a descriptor's unify entry, since a generic
  // unification is an error there, and the procedure is not compiled for it.
  if (device_) {
    for (auto op : module_.getOps<merk::TypeCtorOp>()) {
      op.removeUnifyAttr();
      op.removeCompareAttr();
      op.removeFunctorsAttr();
    }
    // Nor does anything dispatch on a type class, and a table's symbols would
    // keep the filter below from removing the host's procedures.
    for (auto op : module_.getOps<merk::InstanceOp>()) {
      op.removeMethodsAttr();
    }
  }
  if (mlir::omp::isOpenMPModule(module_) && mlir::failed(FilterProcs())) {
    return llvm::failure();
  }
  // What stays on a device of a procedure that only holds a target region
  // does not run there; a procedure a target region calls does.
  if (device_) {
    bool bad{false};
    module_.walk([&](merk::CallMethodOp op) {
      bad = RejectOnDevice(op, "dispatches on a type class") || bad;
    });
    module_.walk([&](merk::TypeclassInfoOp op) {
      bad = RejectOnDevice(op, "dispatches on a type class") || bad;
    });
    module_.walk([&](merk::CompareTypeInfoOp op) {
      bad = RejectOnDevice(op, "compares types") || bad;
    });
    if (bad) {
      return llvm::failure();
    }
  }
  if (mlir::failed(Layouts())) {
    return llvm::failure();
  }
  LastCallModuloConstructor(module_, [this](mlir::Type type,
                                         merk::FunctorAttr functor) {
    const TermLayout *layout{LayoutOf(type)};
    if (!layout || layout->kind != TermLayout::Kind::Cells) {
      return false;
    }
    const TermLayout::Functor *found{layout->Find(functor)};
    return found && found->arity > 0;
  });
  if (mlir::failed(LowerTerms())) {
    return llvm::failure();
  }
  for (auto proc : llvm::make_early_inc_range(module_.getOps<merk::ProcOp>())) {
    LowerProc(proc);
  }
  for (auto func : llvm::to_vector(module_.getOps<mlir::LLVM::LLVMFuncOp>())) {
    Flatten(func);
    // A branch that cannot be reached, such as the else part of a condition
    // that cannot fail, is left without predecessors, and the dialect
    // conversions below skip such blocks.
    (void)mlir::eraseUnreachableBlocks(rewriter_, func->getRegions());
  }
  fail_ = topFail_ = nullptr;
  if (mlir::failed(LowerTypeCtors())) {
    return llvm::failure();
  }
  LowerInstances();
  llvm::SmallVector<mlir::Operation *> ops;
  module_.walk([&](mlir::Operation *op) {
    if (mlir::isa<merk::SucceedOp, merk::CallOp, merk::ConstantOp,
            merk::ToWordOp, merk::FromWordOp, merk::TypeInfoOp,
            merk::TypeInfoWordOp, merk::TypeclassInfoOp,
            merk::TypeclassInfoWordOp, merk::CallMethodOp, merk::ClosureOp,
            merk::ClosureWordOp, merk::CallClosureOp, merk::UnifyWordOp,
            merk::CompareWordOp,
            merk::StringCompareOp, merk::StringEqualOp,
            merk::CompareTypeInfoOp, merk::ForeignCallOp, merk::ArrayNewOp, merk::ArraySizeOp,
            merk::ArrayGetOp, merk::ArraySetOp, merk::ArrayFillOp,
            merk::ArrayCopyOp, merk::ArrayResizeOp, merk::ContRefOp,
            merk::EnvNewOp, merk::EnvGetOp, merk::EnvSetOp, merk::HoleOp,
            merk::RefStoreOp>(op)) {
      ops.push_back(op);
    }
  });
  for (mlir::Operation *op : ops) {
    if (auto succeed{mlir::dyn_cast<merk::SucceedOp>(op)}) {
      LowerSucceed(succeed);
    } else if (auto constant{mlir::dyn_cast<merk::ConstantOp>(op)}) {
      LowerConstant(constant);
    } else if (auto toWord{mlir::dyn_cast<merk::ToWordOp>(op)}) {
      LowerToWord(toWord);
    } else if (auto fromWord{mlir::dyn_cast<merk::FromWordOp>(op)}) {
      LowerFromWord(fromWord);
    } else if (auto typeInfo{mlir::dyn_cast<merk::TypeInfoOp>(op)}) {
      LowerTypeInfo(typeInfo);
    } else if (auto infoWord{mlir::dyn_cast<merk::TypeInfoWordOp>(op)}) {
      LowerTypeInfoWord(infoWord);
    } else if (auto dictionary{mlir::dyn_cast<merk::TypeclassInfoOp>(op)}) {
      if (mlir::failed(LowerTypeclassInfo(dictionary))) {
        return llvm::failure();
      }
    } else if (auto word{mlir::dyn_cast<merk::TypeclassInfoWordOp>(op)}) {
      LowerTypeclassInfoWord(word);
    } else if (auto method{mlir::dyn_cast<merk::CallMethodOp>(op)}) {
      CallMethod(method);
    } else if (auto closure{mlir::dyn_cast<merk::ClosureOp>(op)}) {
      LowerClosure(closure);
    } else if (auto captured{mlir::dyn_cast<merk::ClosureWordOp>(op)}) {
      LowerClosureWord(captured);
    } else if (auto call{mlir::dyn_cast<merk::CallClosureOp>(op)}) {
      CallClosure(call);
    } else if (auto unify{mlir::dyn_cast<merk::UnifyWordOp>(op)}) {
      if (mlir::failed(LowerUnifyWord(unify))) {
        return llvm::failure();
      }
    } else if (auto compare{mlir::dyn_cast<merk::CompareWordOp>(op)}) {
      if (mlir::failed(LowerCompareWord(compare))) {
        return llvm::failure();
      }
    } else if (auto stringCompare{mlir::dyn_cast<merk::StringCompareOp>(op)}) {
      LowerStringCompare(stringCompare);
    } else if (auto stringEqual{mlir::dyn_cast<merk::StringEqualOp>(op)}) {
      LowerStringEqual(stringEqual);
    } else if (auto types{mlir::dyn_cast<merk::CompareTypeInfoOp>(op)}) {
      LowerCompareTypeInfo(types);
    } else if (auto foreign{mlir::dyn_cast<merk::ForeignCallOp>(op)}) {
      LowerForeignCall(foreign);
    } else if (auto ref{mlir::dyn_cast<merk::ContRefOp>(op)}) {
      LowerContRef(ref);
    } else if (auto envNew{mlir::dyn_cast<merk::EnvNewOp>(op)}) {
      LowerEnvNew(envNew);
    } else if (auto envGet{mlir::dyn_cast<merk::EnvGetOp>(op)}) {
      LowerEnvGet(envGet);
    } else if (auto envSet{mlir::dyn_cast<merk::EnvSetOp>(op)}) {
      LowerEnvSet(envSet);
    } else if (auto hole{mlir::dyn_cast<merk::HoleOp>(op)}) {
      // Null, so that the collector finds no pointer there until the hole
      // is filled.
      rewriter_.setInsertionPoint(hole);
      rewriter_.replaceOpWithNewOp<mlir::LLVM::ZeroOp>(
          hole, Convert(hole.getType()));
    } else if (auto store{mlir::dyn_cast<merk::RefStoreOp>(op)}) {
      LowerRefStore(store);
    } else if (!mlir::isa<merk::CallOp>(op)) {
      LowerArray(op);
    } else {
      Call(mlir::cast<merk::CallOp>(op));
    }
  }
  for (auto entry : llvm::make_early_inc_range(module_.getOps<merk::EntryOp>())) {
    LowerEntry(entry);
  }
  // An array is a pointer now: the casts between them that a target
  // region's map entries make go.
  module_.walk([&](mlir::UnrealizedConversionCastOp cast) {
    if (cast->getNumOperands() == 1 && cast->getNumResults() == 1 &&
        Convert(cast.getOperand(0).getType()) ==
            Convert(cast.getResult(0).getType())) {
      cast.getResult(0).replaceAllUsesWith(cast.getOperand(0));
      rewriter_.eraseOp(cast);
    }
  });
  LowerMrt();
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
    if (mlir::isa_and_present<merk::MerkDialect, mrt::MrtDialect,
            mlir::arith::ArithDialect>(op->getDialect())) {
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
  // A closure's entry runs on the host only, so filtering would remove it and
  // the closure that names it: a closure on the device is an error first.
  if (device_) {
    bool bad{false};
    module_.walk([&](merk::ClosureOp op) {
      bad = RejectOnDevice(op, "makes a closure") || bad;
    });
    module_.walk([&](merk::CallClosureOp op) {
      bad = RejectOnDevice(op, "calls a closure") || bad;
    });
    if (bad) {
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
          func.getSymName() == "merk_alloc_slow"
              ? std::string{"cannot allocate a term on an OpenMP device yet:"
                            " the device has no heap"}
              : "cannot call " + func.getSymName() +
                  " on an OpenMP device yet: it is not compiled for the device");
      return llvm::failure();
    }
  }
  // The heap's declaration, which only host code that the filter removed
  // used.
  if (heap_ && mlir::SymbolTable::symbolKnownUseEmpty(heap_, module_)) {
    rewriter_.eraseOp(heap_);
    heap_ = nullptr;
  }
  return llvm::success();
}

} // namespace

llvm::LogicalResult LowerToLLVMDialect(
    mlir::ModuleOp module, const TargetInfo &target) {
  OutlineNondeterminism(module);
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
