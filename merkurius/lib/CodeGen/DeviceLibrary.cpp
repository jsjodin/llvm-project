//===-- lib/CodeGen/DeviceLibrary.cpp - The library on a device -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Copies the procedures a device's target regions reach from the library's
// merk bytecode into the device's module (docs/INPROGRESS.md, Library code on
// the device, step 2). A module declares another module's procedures that it
// calls, without a body; each one that a region reaches is replaced by the
// library's, and what that refers to is copied in turn: the procedures it
// calls and closures it makes, the descriptors of the types whose type_infos
// it makes, and the constructors of its types. The rest of the device's
// compile then treats the copies as the module's own procedures.
//
//===----------------------------------------------------------------------===//

#include "merkurius/CodeGen/DeviceLibrary.h"
#include "merkurius/Dialect/Merk/IR/MerkDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include <memory>

namespace merkurius::codegen {
namespace {

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
        << "cannot call " << declaration.getSymName()
        << " on an OpenMP device yet: it is not compiled for the device"};
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
// has, and by which nothing unifies or compares on a device.
static void ForDevice(merk::TypeCtorOp ctor) {
  ctor.setLinkage("linkonce");
  ctor.removeUnifyAttr();
  ctor.removeCompareAttr();
  ctor.removeFunctorsAttr();
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
          }
          break;
        }
      }
    });
  }
}

// The descriptor of each type whose type_info the module's code makes and
// that it only declares, from the library module that defines it, as no
// library is linked into a device's image.
void Copier::CopyTypeCtors() {
  for (auto declaration : llvm::make_early_inc_range(
           module_.getOps<merk::TypeCtorOp>())) {
    if (declaration.getLinkage() != "declaration") {
      continue;
    }
    LibraryModule *library{Load(declaration.getModule())};
    if (!library) {
      continue;
    }
    merk::TypeCtorOp definition;
    for (auto op : library->module->getOps<merk::TypeCtorOp>()) {
      if (op.getConstructor() == declaration.getConstructor() &&
          op.getLinkage() != "declaration") {
        definition = op;
        break;
      }
    }
    if (!definition) {
      continue;
    }
    mlir::OpBuilder builder{declaration};
    auto copy{mlir::cast<merk::TypeCtorOp>(builder.clone(*definition))};
    ForDevice(copy);
    declaration.erase();
    CollectTerms(copy, *library->module);
  }
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
