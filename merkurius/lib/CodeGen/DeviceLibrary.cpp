//===-- lib/CodeGen/DeviceLibrary.cpp - The library on a device -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Copies the procedures a device's target regions reach from the library's
// merk bytecode into the device's module (design document, Library code on
// the device). A module declares another module's procedures that it
// calls, without a body; each one that a region reaches is replaced by the
// library's, and what that refers to is copied in turn: the procedures it
// calls and closures it makes, the descriptors of the types whose type_infos
// it makes, and the constructors of its types. The rest of the device's
// compile then treats the copies as the module's own procedures.
//
//===----------------------------------------------------------------------===//

#include "merkurius/CodeGen/DeviceLibrary.h"
#include "merkurius/CodeGen/Lowering.h"
#include "merkurius/CodeGen/MLIRGen.h"
#include "merkurius/Dialect/Merk/IR/MerkDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/TargetParser/AMDGPUTargetParser.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include <memory>
#include <optional>

namespace merkurius::codegen {
namespace {

// On a descriptor copied from a library module while the copies are made:
// the module, whose descriptors its fields name.
constexpr llvm::StringLiteral kCopiedFromAttrName{"merk.copied_from"};

// A library module's bytecode, read once.
struct LibraryModule {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  std::unique_ptr<mlir::SymbolTable> symbols;
};

class Copier {
public:
  Copier(mlir::ModuleOp module, llvm::StringRef directory, bool array)
      : module_{module}, symbols_{module}, directory_{directory},
        array_{array} {}

  llvm::LogicalResult Copy();

private:
  // An operation whose references are still to be followed: in the module,
  // or copied from `source`.
  struct Pending {
    mlir::Operation *op;
    mlir::ModuleOp source;
  };

  bool Own(llvm::StringRef symbol);
  LibraryModule *Load(llvm::StringRef name);
  merk::ProcOp Definition(llvm::StringRef symbol, mlir::ModuleOp &source);
  void Follow(const Pending &pending);
  void FollowSymbol(mlir::FlatSymbolRefAttr ref, mlir::Operation *user,
      mlir::ModuleOp source);
  void Replace(merk::ProcOp declaration, mlir::Operation *user);
  void Reached(mlir::Operation *copy, mlir::Operation *user);
  void CopyTypeInfoCtors();
  void CopyTypeCtors();
  bool DeclareFieldCtors(merk::TypeCtorOp ctor, llvm::StringSet<> &have);
  void CollectTerms(mlir::Operation *op, mlir::ModuleOp source);
  void CollectTermsOf(mlir::Type type, mlir::ModuleOp source);
  void CopyTypeDefs();

  mlir::ModuleOp module_;
  mlir::SymbolTable symbols_;
  std::string directory_;
  bool array_;
  bool failed_{false};
  llvm::StringMap<std::unique_ptr<LibraryModule>> loaded_;
  llvm::SmallVector<Pending> pending_;
  llvm::DenseSet<mlir::Operation *> followed_;
  // What was copied, with the module it came from.
  llvm::SmallVector<Pending> copied_;
  // The constructors of the copies' types, with where to find them.
  llvm::SmallVector<std::pair<std::string, mlir::ModuleOp>> terms_;
  // The foreign code of the library modules whose foreign_procs were copied.
  llvm::MapVector<mlir::StringAttr, mlir::Attribute> foreignCode_;
};

// Whether `symbol` is a procedure of the module's own, which the library
// does not have. One without a body is not compiled, and is reported by the
// device's checks.
bool Copier::Own(llvm::StringRef symbol) {
  std::optional<llvm::StringRef> name{module_.getName()};
  return name && symbol.size() > name->size() && symbol.starts_with(*name) &&
      symbol[name->size()] == '.';
}

// The library module `name`'s bytecode, or null if the directory has none or
// it is left out.
LibraryModule *Copier::Load(llvm::StringRef name) {
  if (auto it{loaded_.find(name)}; it != loaded_.end()) {
    return it->second.get();
  }
  std::unique_ptr<LibraryModule> &entry{loaded_[name]};
  llvm::SmallString<256> path{directory_};
  llvm::sys::path::append(path, name + ".merkbc");
  if ((name == "array" && !array_) || !llvm::sys::fs::exists(path)) {
    return nullptr;
  }
  mlir::ParserConfig config{module_.getContext()};
  mlir::OwningOpRef<mlir::ModuleOp> library{
      mlir::parseSourceFile<mlir::ModuleOp>(path, config)};
  if (!library) {
    mlir::emitError(module_.getLoc())
        << "cannot read the library module " << name << " for the device from "
        << path;
    failed_ = true;
    return nullptr;
  }
  entry = std::make_unique<LibraryModule>();
  entry->symbols = std::make_unique<mlir::SymbolTable>(*library);
  entry->module = std::move(library);
  return entry.get();
}

// The library's definition of the procedure `symbol`, from the module its
// name begins with, the longest such name first; null if none has one.
merk::ProcOp Copier::Definition(
    llvm::StringRef symbol, mlir::ModuleOp &source) {
  for (std::size_t dot{symbol.rfind('.')}; dot != llvm::StringRef::npos;
      dot = dot == 0 ? llvm::StringRef::npos : symbol.rfind('.', dot - 1)) {
    LibraryModule *library{Load(symbol.take_front(dot))};
    if (!library) {
      continue;
    }
    auto proc{library->symbols->lookup<merk::ProcOp>(symbol)};
    if (proc && !proc.isExternal()) {
      source = *library->module;
      return proc;
    }
  }
  return nullptr;
}

// Records on a copy the procedure that reached it, from `user`'s.
void Copier::Reached(mlir::Operation *copy, mlir::Operation *user) {
  auto proc{user->getParentOfType<merk::ProcOp>()};
  if (auto self{mlir::dyn_cast<merk::ProcOp>(user)}) {
    proc = self;
  }
  if (proc) {
    copy->setAttr(kReachedFromAttrName,
        mlir::StringAttr::get(module_.getContext(), proc.getSymName()));
    copy->setAttr(kReachedAtAttrName, mlir::LocationAttr{user->getLoc()});
  }
}

// Replaces a declaration that `user` reached with the library's procedure,
// or reports it if the library has none.
void Copier::Replace(merk::ProcOp declaration, mlir::Operation *user) {
  mlir::ModuleOp source;
  merk::ProcOp definition{Definition(declaration.getSymName(), source)};
  if (!definition) {
    if (failed_) {
      return;
    }
    mlir::InFlightDiagnostic diagnostic{mlir::emitError(user->getLoc())
        << "cannot call " << declaration.getSymName() << " "
        << DevicePlace(module_) << " yet: it is not compiled for the device"};
    NoteReachedFrom(diagnostic, user->getParentOfType<merk::ProcOp>());
    failed_ = true;
    return;
  }
  mlir::Operation *copy{definition->clone()};
  mlir::Block::iterator at{std::next(mlir::Block::iterator{declaration})};
  symbols_.erase(declaration);
  symbols_.insert(copy, at);
  mlir::SymbolTable::setSymbolVisibility(
      copy, mlir::SymbolTable::Visibility::Private);
  Reached(copy, user);
  followed_.insert(copy);
  copied_.push_back({copy, source});
  pending_.push_back({copy, source});
}

// Follows a reference that `user` makes, in code of the module or copied from
// `source`.
void Copier::FollowSymbol(mlir::FlatSymbolRefAttr ref, mlir::Operation *user,
    mlir::ModuleOp source) {
  mlir::Operation *symbol{symbols_.lookup(ref.getValue())};
  if (auto proc{mlir::dyn_cast_or_null<merk::ProcOp>(symbol)}) {
    if (proc.isExternal() && !Own(proc.getSymName())) {
      Replace(proc, user);
    } else if (followed_.insert(proc).second) {
      // The module's own procedure, or one copied already.
      pending_.push_back({proc, nullptr});
    }
    return;
  }
  if (symbol || !source) {
    // Another kind of symbol of the module, or a C function of foreign code.
    return;
  }
  // Something copied code refers to that the module does not have: a
  // procedure of the library module's own, such as a closure's entry, or an
  // instance, whose methods a device does not call. A C function of foreign
  // code is no symbol of the library module either.
  mlir::Operation *definition{mlir::SymbolTable::lookupSymbolIn(
      source, ref.getAttr())};
  if (!definition) {
    return;
  }
  mlir::Operation *copy{definition->clone()};
  symbols_.insert(copy, module_.getBody()->end());
  mlir::SymbolTable::setSymbolVisibility(
      copy, mlir::SymbolTable::Visibility::Private);
  copied_.push_back({copy, source});
  if (auto proc{mlir::dyn_cast<merk::ProcOp>(copy)}) {
    Reached(copy, user);
    followed_.insert(copy);
    pending_.push_back({copy, source});
  } else if (auto instance{mlir::dyn_cast<merk::InstanceOp>(copy)}) {
    instance.removeMethodsAttr();
  }
}

void Copier::Follow(const Pending &pending) {
  // Null for the module's own code, so that only what copies refer to is
  // looked for in a library module.
  mlir::ModuleOp source{pending.source};
  pending.op->walk([&](mlir::Operation *inner) {
    for (mlir::NamedAttribute attribute : inner->getAttrs()) {
      if (auto ref{mlir::dyn_cast<mlir::FlatSymbolRefAttr>(
              attribute.getValue())}) {
        FollowSymbol(ref, inner, source);
      }
    }
  });
}

// The constructors named by `type`, a copy's, to be found in `source`.
void Copier::CollectTermsOf(mlir::Type type, mlir::ModuleOp source) {
  type.walk([&](merk::TermType term) {
    terms_.push_back({term.getName().str(), source});
  });
}

void Copier::CollectTerms(mlir::Operation *op, mlir::ModuleOp source) {
  op->walk([&](mlir::Operation *inner) {
    for (mlir::Type type : inner->getResultTypes()) {
      CollectTermsOf(type, source);
    }
    for (mlir::Region &region : inner->getRegions()) {
      for (mlir::Block &block : region) {
        for (mlir::BlockArgument argument : block.getArguments()) {
          CollectTermsOf(argument.getType(), source);
        }
      }
    }
    inner->getAttrDictionary().walk(
        [&](mlir::Type type) { CollectTermsOf(type, source); });
  });
}

// A copied descriptor, which each module of the device's image that needs it
// has, and by which nothing unifies or compares on a device. Its functors
// stay, by which a message is copied (runtime/copy.h).
static void ForDevice(merk::TypeCtorOp ctor) {
  ctor.setLinkage("linkonce");
  ctor.removeUnifyAttr();
  ctor.removeCompareAttr();
}

// The descriptor of each type whose type_info a copy makes that the module
// has none of, as the copy's module has it.
void Copier::CopyTypeInfoCtors() {
  llvm::StringSet<> have;
  for (auto op : module_.getOps<merk::TypeCtorOp>()) {
    have.insert(op.getConstructor());
  }
  for (const Pending &copy : copied_) {
    mlir::ModuleOp source{copy.source};
    copy.op->walk([&](merk::TypeInfoOp info) {
      if (!have.insert(info.getConstructor()).second) {
        return;
      }
      for (auto op : source.getOps<merk::TypeCtorOp>()) {
        if (op.getConstructor() == info.getConstructor()) {
          mlir::OpBuilder builder{
              module_.getBody(), module_.getBody()->begin()};
          auto ctor{mlir::cast<merk::TypeCtorOp>(builder.clone(*op))};
          if (ctor.getLinkage() != "declaration") {
            ForDevice(ctor);
            if (std::optional<llvm::StringRef> name{source.getName()}) {
              ctor->setAttr(kCopiedFromAttrName,
                  mlir::StringAttr::get(module_.getContext(), *name));
            }
          }
          break;
        }
      }
    });
  }
}

// The constructors that a field's type names, by FieldTypeAttr's encoding:
// an array of its constructor and its arguments, or a number for a type
// parameter or a hidden type_info, which name none.
static void FieldConstructors(
    mlir::Attribute type, llvm::SmallVectorImpl<std::string> &names) {
  auto parts{mlir::dyn_cast<mlir::ArrayAttr>(type)};
  if (!parts || parts.empty()) {
    return;
  }
  names.push_back(mlir::cast<mlir::StringAttr>(parts[0]).getValue().str());
  for (mlir::Attribute argument : parts.getValue().drop_front()) {
    FieldConstructors(argument, names);
  }
}

// The constructors that the fields of a descriptor's functors name.
static void FieldConstructors(
    merk::TypeCtorOp ctor, llvm::SmallVectorImpl<std::string> &names) {
  if (!ctor.getFunctors()) {
    return;
  }
  for (mlir::Attribute functor : ctor.getFunctors()->getValue()) {
    for (mlir::Attribute type :
        mlir::cast<mlir::ArrayAttr>(functor).getValue().drop_front()) {
      FieldConstructors(type, names);
    }
  }
}

// The descriptor of each type whose type_info the module's code makes and
// that it only declares, from the library module that defines it, as no
// library is linked into a device's image; and the descriptors that the
// fields of each descriptor it has name, which a copy from a library module
// may not have, as declarations, from the module it came from, which are
// then copied the same way.
void Copier::CopyTypeCtors() {
  bool changed{true};
  while (changed) {
    changed = false;
    llvm::StringSet<> have;
    for (auto op : module_.getOps<merk::TypeCtorOp>()) {
      have.insert(op.getConstructor());
    }
    for (auto ctor :
        llvm::make_early_inc_range(module_.getOps<merk::TypeCtorOp>())) {
      if (ctor.getLinkage() != "declaration") {
        changed = DeclareFieldCtors(ctor, have) || changed;
        continue;
      }
      LibraryModule *library{Load(ctor.getModule())};
      if (!library) {
        continue;
      }
      merk::TypeCtorOp definition;
      for (auto op : library->module->getOps<merk::TypeCtorOp>()) {
        if (op.getConstructor() == ctor.getConstructor() &&
            op.getLinkage() != "declaration") {
          definition = op;
          break;
        }
      }
      if (!definition) {
        continue;
      }
      mlir::OpBuilder builder{ctor};
      auto copy{mlir::cast<merk::TypeCtorOp>(builder.clone(*definition))};
      copy->setAttr(kCopiedFromAttrName,
          mlir::StringAttr::get(module_.getContext(), ctor.getModule()));
      ForDevice(copy);
      ctor.erase();
      CollectTerms(copy, *library->module);
      changed = true;
    }
  }
  for (auto ctor : module_.getOps<merk::TypeCtorOp>()) {
    ctor->removeAttr(kCopiedFromAttrName);
  }
}

// Adds the descriptors that the fields of `ctor`, a copy from a library
// module, name and the module does not have, as that library module has
// them, a declaration or a definition. Whether it added any.
bool Copier::DeclareFieldCtors(merk::TypeCtorOp ctor, llvm::StringSet<> &have) {
  auto from{ctor->getAttrOfType<mlir::StringAttr>(kCopiedFromAttrName)};
  if (!from) {
    return false;
  }
  llvm::SmallVector<std::string> names;
  FieldConstructors(ctor, names);
  LibraryModule *library{nullptr};
  bool added{false};
  for (const std::string &name : names) {
    if (have.contains(name)) {
      continue;
    }
    if (!library && !(library = Load(from.getValue()))) {
      return added;
    }
    for (auto op : library->module->getOps<merk::TypeCtorOp>()) {
      if (op.getConstructor() == name) {
        mlir::OpBuilder builder{module_.getBody(), module_.getBody()->begin()};
        auto copy{mlir::cast<merk::TypeCtorOp>(builder.clone(*op))};
        if (copy.getLinkage() != "declaration") {
          copy->setAttr(kCopiedFromAttrName, from);
          ForDevice(copy);
          CollectTerms(copy, *library->module);
        }
        have.insert(name);
        added = true;
        break;
      }
    }
  }
  return added;
}

// The constructors of the copies' types that the module does not have, and
// of the types those name, from the library module each came from.
void Copier::CopyTypeDefs() {
  llvm::StringSet<> have;
  for (auto def : module_.getOps<merk::TypeDefOp>()) {
    have.insert(def.getName());
  }
  while (!terms_.empty()) {
    auto [name, source]{terms_.pop_back_val()};
    if (!have.insert(name).second) {
      continue;
    }
    for (auto def : source.getOps<merk::TypeDefOp>()) {
      if (def.getName() != name) {
        continue;
      }
      mlir::OpBuilder builder{module_.getBody(), module_.getBody()->begin()};
      CollectTerms(builder.clone(*def), source);
      break;
    }
  }
}

llvm::LogicalResult Copier::Copy() {
  module_.walk([&](mlir::omp::TargetOp target) {
    pending_.push_back({target, nullptr});
  });
  for (auto proc : module_.getOps<merk::ProcOp>()) {
    if (proc->hasAttr(kProcessRootAttrName)) {
      pending_.push_back({proc, nullptr});
    }
  }
  while (!pending_.empty()) {
    Follow(pending_.pop_back_val());
  }
  if (failed_) {
    return llvm::failure();
  }
  CopyTypeInfoCtors();
  CopyTypeCtors();
  for (const Pending &copy : copied_) {
    CollectTerms(copy.op, copy.source);
  }
  CopyTypeDefs();
  for (const Pending &copy : copied_) {
    mlir::ModuleOp source{copy.source};
    bool foreign{false};
    copy.op->walk([&](merk::ForeignCallOp call) {
      foreign = foreign || call.getCallee().starts_with("merk_foreign_");
    });
    mlir::Attribute code{source->getAttr(kForeignCodeAttrName)};
    if (foreign && code) {
      foreignCode_.insert({mlir::StringAttr::get(module_.getContext(),
                               source.getName().value_or("")),
          code});
    }
  }
  if (!foreignCode_.empty()) {
    llvm::SmallVector<mlir::NamedAttribute> entries;
    for (auto [name, code] : foreignCode_) {
      entries.emplace_back(name, code);
    }
    module_->setAttr(kDeviceForeignCodeAttrName,
        mlir::DictionaryAttr::get(module_.getContext(), entries));
  }
  return llvm::failure(failed_);
}

} // namespace

llvm::LogicalResult CopyDeviceLibrary(
    mlir::ModuleOp module, llvm::StringRef directory, bool array) {
  if (directory.empty()) {
    return llvm::success();
  }
  module.getContext()
      ->loadDialect<merk::MerkDialect, mlir::arith::ArithDialect,
          mlir::DLTIDialect, mlir::LLVM::LLVMDialect,
          mlir::omp::OpenMPDialect>();
  return Copier{module, directory, array}.Copy();
}

std::string ForeignFunctionGuard(llvm::StringRef name) {
  return ("#if !defined(MERK_DEVICE) || defined(MERK_WANT_" + name + ")").str();
}

namespace {

// Compiles a device's foreign code: CompileDeviceForeignCode.
class ForeignCompiler {
public:
  ForeignCompiler(mlir::ModuleOp module, const TargetInfo &target,
      llvm::StringRef clang)
      : module_{module}, target_{target}, clang_{clang} {}

  llvm::LogicalResult Compile();

private:
  void CompileUnit(llvm::StringRef module, llvm::StringRef code);
  std::vector<std::string> DeviceOptions() const;
  void LoadAvailable();
  bool Available(const llvm::GlobalValue &value);
  void CheckNeeds(llvm::Function &function, merk::ForeignCallOp call);
  mlir::InFlightDiagnostic Report(merk::ForeignCallOp call);

  mlir::ModuleOp module_;
  const TargetInfo &target_;
  std::string clang_;
  bool failed_{false};
  // Each C function called, with a call of it.
  llvm::MapVector<llvm::StringRef, merk::ForeignCallOp> calls_;
  llvm::SmallVector<mlir::Attribute> bitcode_;
  // What libc and libm for the GPU define, once read; whether they were
  // found.
  std::optional<llvm::StringSet<>> available_;
  bool libraries_{false};
};

// An error at a call of a foreign_proc's C function, named for the procedure
// that wraps it, with the chain of calls that reached that one.
mlir::InFlightDiagnostic ForeignCompiler::Report(merk::ForeignCallOp call) {
  auto proc{call->getParentOfType<merk::ProcOp>()};
  mlir::InFlightDiagnostic diagnostic{mlir::emitError(call.getLoc())
      << "cannot call " << (proc ? proc.getSymName() : call.getCallee()) << " "
      << DevicePlace(module_) << " yet: "};
  return diagnostic;
}

// What libc and libm for the device's GPU define, from their bitcode in the
// build or installation beside clang, as clang links them into an OpenMP
// device's image. Without them, the device's link finds what is missing.
void ForeignCompiler::LoadAvailable() {
  available_.emplace();
  llvm::SmallString<256> dir{
      llvm::sys::path::parent_path(llvm::sys::path::parent_path(clang_))};
  llvm::sys::path::append(dir, "lib", "amdgcn-amd-amdhsa");
  if (!target_.triple().isAMDGPU()) {
    return;
  }
  for (llvm::StringRef library : {"libc.bc", "libm.bc"}) {
    llvm::SmallString<256> path{dir};
    llvm::sys::path::append(path, library);
    llvm::LLVMContext context;
    llvm::SMDiagnostic diagnostic;
    std::unique_ptr<llvm::Module> module{
        llvm::getLazyIRFileModule(path, diagnostic, context)};
    if (!module) {
      continue;
    }
    libraries_ = true;
    for (const llvm::GlobalValue &value : module->global_values()) {
      if (!value.isDeclaration()) {
        available_->insert(value.getName());
      }
    }
  }
}

// Whether a device's image has what `value` declares.
bool ForeignCompiler::Available(const llvm::GlobalValue &value) {
  llvm::StringRef name{value.getName()};
  if (auto *function{llvm::dyn_cast<llvm::Function>(&value)};
      function && function->isIntrinsic()) {
    return true;
  }
  if (name.starts_with("merk_device_") || name.starts_with("merk_array_")) {
    return true;
  }
  // OpenMP's device runtime is in an offload image, not a GPU process's,
  // whose start object defines only omp_get_device_num.
  if (name.starts_with("omp_") || name.starts_with("__kmpc_")) {
    return !TargetInfo::ProcessKernels(module_) || name == "omp_get_device_num";
  }
  if (!available_) {
    LoadAvailable();
  }
  // Without the libraries' bitcode, only the host's runtime is known not to
  // be there.
  return libraries_ ? available_->contains(name) : !name.starts_with("merk_");
}

// Reports what `function`, the C function `call` calls, needs that a device
// does not have, following the functions of its module that it calls.
void ForeignCompiler::CheckNeeds(
    llvm::Function &function, merk::ForeignCallOp call) {
  llvm::SmallPtrSet<const llvm::Value *, 32> seen;
  llvm::SmallVector<const llvm::Value *> pending{&function};
  llvm::SetVector<const llvm::GlobalValue *> missing;
  while (!pending.empty()) {
    const llvm::Value *value{pending.pop_back_val()};
    if (!seen.insert(value).second) {
      continue;
    }
    if (const auto *global{llvm::dyn_cast<llvm::GlobalValue>(value)}) {
      if (global->isDeclaration()) {
        if (!Available(*global)) {
          missing.insert(global);
        }
        continue;
      }
      if (const auto *defined{llvm::dyn_cast<llvm::Function>(global)}) {
        for (const llvm::BasicBlock &block : *defined) {
          for (const llvm::Instruction &instruction : block) {
            for (const llvm::Value *operand : instruction.operands()) {
              pending.push_back(operand);
            }
          }
        }
      } else if (const auto *variable{
                     llvm::dyn_cast<llvm::GlobalVariable>(global)};
          variable && variable->hasInitializer()) {
        pending.push_back(variable->getInitializer());
      }
    } else if (const auto *constant{llvm::dyn_cast<llvm::Constant>(value)}) {
      for (const llvm::Value *operand : constant->operands()) {
        pending.push_back(operand);
      }
    }
  }
  for (const llvm::GlobalValue *global : missing) {
    mlir::InFlightDiagnostic diagnostic{Report(call)
        << "its foreign code calls " << global->getName()
        << ", which is not on the device"};
    NoteReachedFrom(diagnostic, call->getParentOfType<merk::ProcOp>());
    failed_ = true;
  }
}

// The device's GPU, as its CPU names it or the triple's sub-architecture
// does, as the driver gives it (amdgpu10.30-amd-amdhsa for gfx1030); empty if
// neither does.
static std::string DeviceGPU(const TargetInfo &target) {
  if (!target.options().cpu.empty() ||
      !target.triple().isAMDGCN() ||
      target.triple().getSubArch() == llvm::Triple::NoSubArch) {
    return target.options().cpu;
  }
  return llvm::AMDGPU::getArchNameAMDGCN(
      llvm::AMDGPU::getGPUKindFromSubArch(target.triple().getSubArch()))
      .str();
}

// clang's options for C that OpenMP compiles for the device, as clang
// compiles a C file's declare target functions for an offloading target: with
// the host's headers, for the device's GPU. The
// device's image has libc and libm for the GPU, whose functions the code calls
// as the host's headers declare them: not clang's wrappers for the GPU, whose
// maths functions are ROCm's. A GPU process's image has no OpenMP device
// runtime, so its code is compiled in CUDA mode, where a local whose address
// is taken stays on the thread's stack: in the default mode clang moves it to
// memory from __kmpc_alloc_shared, for a parallel region that the function
// might start, which the image does not have. Foreign code starts none.
std::vector<std::string> ForeignCompiler::DeviceOptions() const {
  std::vector<std::string> options{"-fopenmp",
      "--offload-arch=" + DeviceGPU(target_), "--offload-device-only", "-nogpulib", "-nogpuinc",
      "-O" + std::to_string(target_.options().optimizationLevel),
      "-mcode-object-version=" +
          std::to_string(target_.options().codeObjectVersion / 100),
      "-DMERK_DEVICE"};
  if (TargetInfo::ProcessKernels(module_)) {
    options.push_back("-fopenmp-cuda-mode");
  }
  return options;
}

// Compiles one module's foreign code for the C functions of it that are
// called.
void ForeignCompiler::CompileUnit(
    llvm::StringRef module, llvm::StringRef code) {
  llvm::SmallVector<std::pair<llvm::StringRef, merk::ForeignCallOp>> wanted;
  for (auto [name, call] : calls_) {
    if (code.contains((" " + name + "(").str())) {
      wanted.push_back({name, call});
    }
  }
  if (wanted.empty()) {
    return;
  }
  std::vector<std::string> options{DeviceOptions()};
  for (auto [name, call] : wanted) {
    options.push_back(("-DMERK_WANT_" + name).str());
  }
  // The code is the host's C, which OpenMP compiles for the device as
  // declare target.
  std::string unit{("#pragma omp begin declare target\n" + code +
      "\n#pragma omp end declare target\n")
                       .str()};
  std::string bitcode, error;
  // clang has reported the code's errors.
  if (!CompileForeignCode(clang_, unit, options, bitcode, error)) {
    mlir::InFlightDiagnostic diagnostic{
        mlir::emitError(wanted.front().second.getLoc())
        << module << "'s foreign code does not compile for the device"
        << (error.empty() ? "" : ": " + error)};
    NoteReachedFrom(
        diagnostic, wanted.front().second->getParentOfType<merk::ProcOp>());
    failed_ = true;
    return;
  }
  llvm::LLVMContext context;
  llvm::SMDiagnostic diagnostic;
  std::unique_ptr<llvm::Module> compiled{llvm::parseIR(
      llvm::MemoryBufferRef{bitcode, "foreign code"}, diagnostic, context)};
  if (!compiled) {
    mlir::emitError(wanted.front().second.getLoc())
        << "cannot read the foreign code clang compiled: "
        << diagnostic.getMessage();
    failed_ = true;
    return;
  }
  for (auto [name, call] : wanted) {
    llvm::Function *function{compiled->getFunction(name)};
    if (!function || function->isDeclaration()) {
      mlir::emitError(call.getLoc())
          << "the foreign code compiled for the device has no " << name;
      failed_ = true;
      continue;
    }
    CheckNeeds(*function, call);
  }
  bitcode_.push_back(mlir::StringAttr::get(module_.getContext(), bitcode));
}

llvm::LogicalResult ForeignCompiler::Compile() {
  module_.walk([&](merk::ForeignCallOp call) {
    if (call.getCallee().starts_with("merk_foreign_")) {
      calls_.insert({call.getCallee(), call});
    }
  });
  auto library{
      module_->getAttrOfType<mlir::DictionaryAttr>(kDeviceForeignCodeAttrName)};
  module_->removeAttr(kDeviceForeignCodeAttrName);
  if (calls_.empty()) {
    return llvm::success();
  }
  // clang would otherwise look for one on the machine.
  if (DeviceGPU(target_).empty()) {
    mlir::emitError(calls_.front().second.getLoc())
        << "compiling foreign code for "
        << (TargetInfo::ProcessKernels(module_) ? "a GPU process"
                                                : "an OpenMP device")
        << " needs its GPU, as "
           "-target-cpu or the triple names it, such as "
           "amdgpu10.30-amd-amdhsa for gfx1030";
    return llvm::failure();
  }
  if (auto own{module_->getAttrOfType<mlir::StringAttr>(kForeignCodeAttrName)}) {
    CompileUnit(module_.getName().value_or("the module"), own.getValue());
  }
  for (mlir::NamedAttribute entry : library ? library.getValue()
                                            : llvm::ArrayRef<mlir::NamedAttribute>{}) {
    CompileUnit(entry.getName(),
        mlir::cast<mlir::StringAttr>(entry.getValue()).getValue());
  }
  if (failed_) {
    return llvm::failure();
  }
  module_->setAttr(kDeviceForeignBitcodeAttrName,
      mlir::ArrayAttr::get(module_.getContext(), bitcode_));
  return llvm::success();
}

} // namespace

llvm::LogicalResult CompileDeviceForeignCode(
    mlir::ModuleOp module, const TargetInfo &target, llvm::StringRef clang) {
  return ForeignCompiler{module, target, clang}.Compile();
}

llvm::StringRef DevicePlace(mlir::ModuleOp module) {
  return TargetInfo::ProcessKernels(module) ? "in a GPU process"
                                            : "on an OpenMP device";
}

void NoteReachedFrom(
    mlir::InFlightDiagnostic &diagnostic, mlir::Operation *proc) {
  auto module{proc ? proc->getParentOfType<mlir::ModuleOp>() : nullptr};
  llvm::DenseSet<mlir::Operation *> seen;
  while (proc && seen.insert(proc).second) {
    auto from{proc->getAttrOfType<mlir::StringAttr>(kReachedFromAttrName)};
    if (!from) {
      return;
    }
    auto at{proc->getAttrOfType<mlir::LocationAttr>(kReachedAtAttrName)};
    mlir::Operation *caller{mlir::SymbolTable::lookupSymbolIn(module, from)};
    diagnostic.attachNote(at ? mlir::Location{at} : proc->getLoc())
        << "called by " << from.getValue() << " here: "
        << mlir::cast<mlir::SymbolOpInterface>(proc).getName();
    proc = caller;
  }
}

} // namespace merkurius::codegen
