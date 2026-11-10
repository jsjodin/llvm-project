//===-- lib/CodeGen/MLIRGenProcesses.cpp - GPU processes' kernels ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Process kinds that run on a GPU (docs/INPROGRESS.md, GPU processes without
// omp target, step 4b). The device's compile of them (-fprocess-kernels)
// makes a kernel for each, which calls the kind's predicate. The host's
// compile embeds the image that the kernels were linked into, one for each
// GPU, and registers it with the runtime at start-up.
//
//===----------------------------------------------------------------------===//

#include "MLIRGenImpl.h"
#include "llvm/Support/MemoryBuffer.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace merkurius::codegen {

using namespace parser::literals;
using semantics::Procedure;

namespace mlirgen {

namespace {
// A mailbox handle's cell, as the kernel takes it: its five words
// (FlatCell).
constexpr unsigned kMailboxWords{5};
} // namespace

void MerkGenerator::CollectGpuKinds() {
  for (const parser::Item &item : tree_.items) {
    if (const auto *decl{std::get_if<const parser::ProcessDecl *>(&item)};
        decl && (*decl)->onGpu) {
      gpuKinds_.push_back(*decl);
    }
  }
}

// The procedure of a kind's predicate: its arguments, a mailbox for each
// message type, and the io state.
const Procedure *MerkGenerator::KindProcedure(const parser::ProcessDecl &decl) {
  std::size_t arity{decl.arguments.size() + decl.receives.size() + 2};
  for (const Procedure &procedure : modes_.procedures()) {
    const semantics::Symbol &symbol{*procedure.symbol};
    if (symbol.module == moduleName_ &&
        symbol.kind == semantics::SymbolKind::Predicate &&
        symbol.name == decl.name && symbol.arity == arity) {
      return &procedure;
    }
  }
  return nullptr;
}

void MerkGenerator::GenerateProcessKernels() {
  if (!target_.options().processKernels) {
    return;
  }
  for (const parser::ProcessDecl *decl : gpuKinds_) {
    GenerateProcessKernel(*decl);
  }
}

// The kernel merk_process_<kind>, which takes the kind's arguments and its
// mailbox's handle by value, and calls the kind's predicate. In the first
// version, the arguments are integers, floats, characters and mailboxes,
// and the kind is not polymorphic (decision D3); the kernel is one team,
// whose initial thread runs the predicate, as the runtime launches it. A
// mailbox's handle is five words, which the kernel puts in a cell on its
// stack: the predicate returns before the kernel does, and a handle that it
// sends is copied.
void MerkGenerator::GenerateProcessKernel(const parser::ProcessDecl &decl) {
  parser::CharBlock at{Where(decl.tokens)};
  mlir::Location loc{Location(at)};
  const Procedure *procedure{KindProcedure(decl)};
  // A kind whose predicate code generation left out has been reported.
  merk::ProcOp kind{procedure ? procs_.lookup(procedure) : nullptr};
  if (!kind || kind.isExternal()) {
    return;
  }
  const semantics::Signature *signature{types_.SignatureOf(*procedure->symbol)};
  if (!signature) {
    return;
  }
  if (signature->typeVars != 0) {
    messages_.Say(at, "code generation for %s"_todo_en_US,
        "a process kind that runs on a GPU and is polymorphic");
    return;
  }
  // The kind's arguments and its mailbox; the io state has no value.
  auto i64{builder_.getI64Type()};
  llvm::SmallVector<mlir::Type> parameters;
  llvm::SmallVector<mlir::Type> values;
  bool ok{true};
  for (const semantics::Type *type :
      signature->arguments.take_front(decl.arguments.size() + 1)) {
    std::optional<Represented> represented{Represent(type)};
    mlir::Type merkType{represented ? represented->type : mlir::Type{}};
    auto term{mlir::dyn_cast_if_present<merk::TermType>(merkType)};
    if (represented && !represented->dummy &&
        mlir::isa<mlir::IntegerType, mlir::FloatType>(merkType)) {
      parameters.push_back(merkType);
    } else if (term && term.getName().starts_with("process.mailbox(")) {
      parameters.append(kMailboxWords, i64);
    } else {
      messages_.Say(at, "code generation for %s"_todo_en_US,
          "a process kind that runs on a GPU and takes a value of type " +
              semantics::TypeToString(*type) +
              ": its kernel takes integers, floats, characters and "
              "mailboxes");
      ok = false;
      continue;
    }
    values.push_back(merkType);
  }
  if (!ok) {
    return;
  }
  mlir::OpBuilder::InsertionGuard guard{builder_};
  builder_.setInsertionPointToEnd(module_.getBody());
  auto kernel{
      merk::ProcOp::create(builder_, loc, ("merk_process_" + decl.name).str(),
          builder_.getFunctionType(parameters, {}), merk::Determinism::Det)};
  kernel->setAttr(kKernelAttrName, builder_.getUnitAttr());
  mlir::Block *entry{kernel.addEntryBlock()};
  builder_.setInsertionPointToStart(entry);
  auto generic{mlir::LLVM::LLVMPointerType::get(
      &context_, target_.addressSpaces().generic)};
  auto stack{mlir::LLVM::LLVMPointerType::get(
      &context_, target_.addressSpaces().private_)};
  llvm::SmallVector<mlir::Value> arguments;
  unsigned next{0};
  for (mlir::Type type : values) {
    if (!mlir::isa<merk::TermType>(type)) {
      arguments.push_back(entry->getArgument(next++));
      continue;
    }
    auto cellType{mlir::LLVM::LLVMArrayType::get(i64, kMailboxWords)};
    mlir::Value cell{mlir::LLVM::AllocaOp::create(
        builder_, loc, stack, cellType, IntegerConstant(i64, 1, loc))};
    if (stack != generic) {
      cell = mlir::LLVM::AddrSpaceCastOp::create(builder_, loc, generic, cell);
    }
    for (unsigned word{0}; word < kMailboxWords; ++word) {
      mlir::Value address{mlir::LLVM::GEPOp::create(builder_, loc, generic, i64,
          cell,
          llvm::ArrayRef<mlir::LLVM::GEPArg>{static_cast<std::int32_t>(word)})};
      mlir::LLVM::StoreOp::create(
          builder_, loc, entry->getArgument(next++), address);
    }
    arguments.push_back(
        mlir::UnrealizedConversionCastOp::create(builder_, loc, type, cell)
            .getResult(0));
  }
  merk::CallOp::create(
      builder_, loc, mlir::TypeRange{}, kind.getSymName(), arguments);
  merk::SucceedOp::create(builder_, loc, mlir::ValueRange{});
}

// A global of the module with `bytes`, constant and internal.
static mlir::LLVM::GlobalOp BytesGlobal(mlir::OpBuilder &builder,
    mlir::Location loc, llvm::StringRef name, llvm::StringRef bytes,
    std::uint64_t alignment) {
  auto type{mlir::LLVM::LLVMArrayType::get(
      builder.getI8Type(), static_cast<unsigned>(bytes.size()))};
  auto global{mlir::LLVM::GlobalOp::create(builder, loc, type,
      /*isConstant=*/true, mlir::LLVM::Linkage::Internal, name,
      builder.getStringAttr(bytes))};
  global.setAlignment(alignment);
  return global;
}

// The host's compile of a module with kinds that run on a GPU embeds the
// image of their kernels for each GPU that -fembed-process-image= names,
// as gpu:file, and registers each at start-up, with the module's name and
// the GPU's, in a procedure that a constructor calls, as message types are
// registered. The runtime loads an image when a process is first spawned
// on a GPU of its kind (step 4c). False after reporting an image that
// cannot be read.
bool MerkGenerator::RegisterProcessImages() {
  if (gpuKinds_.empty() || target_.options().IsDevice() ||
      target_.options().processImages.empty()) {
    return true;
  }
  mlir::Location loc{builder_.getUnknownLoc()};
  struct Image {
    std::string gpu;
    std::unique_ptr<llvm::MemoryBuffer> bytes;
  };
  llvm::SmallVector<Image> images;
  for (llvm::StringRef option : target_.options().processImages) {
    auto [gpu, file]{option.split(':')};
    if (gpu.empty() || file.empty()) {
      mlir::emitError(loc) << "-fembed-process-image=" << option
                           << ": expected the GPU and the image's file, as "
                              "gfx1030:file";
      return false;
    }
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> bytes{
        llvm::MemoryBuffer::getFile(file, /*IsText=*/false,
            /*RequiresNullTerminator=*/false)};
    if (!bytes) {
      mlir::emitError(loc) << "cannot read the GPU processes' image " << file
                           << ": " << bytes.getError().message();
      return false;
    }
    images.push_back({gpu.str(), std::move(*bytes)});
  }
  mlir::OpBuilder::InsertionGuard guard{builder_};
  builder_.setInsertionPointToEnd(module_.getBody());
  std::string prefix{moduleName_.str() + ".merk_process_image"};
  // C strings, and the image aligned as an ELF file's headers must be.
  mlir::LLVM::GlobalOp name{BytesGlobal(builder_, loc, prefix + ".module",
      (moduleName_ + llvm::StringRef("\0", 1)).str(), 1)};
  llvm::SmallVector<std::pair<mlir::LLVM::GlobalOp, mlir::LLVM::GlobalOp>>
      globals;
  for (const Image &image : images) {
    std::string stem{prefix + "." + image.gpu};
    globals.push_back({BytesGlobal(builder_, loc, stem + ".gpu",
                           image.gpu + std::string(1, '\0'), 1),
        BytesGlobal(builder_, loc, stem, image.bytes->getBuffer(), 16)});
  }
  std::string procName{moduleName_.str() + ".merk_register_process_images"};
  auto proc{merk::ProcOp::create(builder_, loc, procName,
      builder_.getFunctionType({}, {}), merk::Determinism::Det)};
  proc.setPrivate();
  builder_.setInsertionPointToStart(proc.addEntryBlock());
  auto i64{builder_.getI64Type()};
  for (auto [image, globalPair] : llvm::zip_equal(images, globals)) {
    auto [gpu, bytes]{globalPair};
    merk::ForeignCallOp::create(builder_, loc, mlir::TypeRange{},
        "merk_register_process_image",
        mlir::ValueRange{mlir::LLVM::AddressOfOp::create(builder_, loc, name),
            mlir::LLVM::AddressOfOp::create(builder_, loc, gpu),
            mlir::LLVM::AddressOfOp::create(builder_, loc, bytes),
            IntegerConstant(i64,
                static_cast<std::int64_t>(image.bytes->getBufferSize()), loc)},
        /*can_fail=*/false);
  }
  merk::SucceedOp::create(builder_, loc, mlir::ValueRange{});
  builder_.setInsertionPointToEnd(module_.getBody());
  merk::ConstructorOp::create(builder_, loc, procName);
  return true;
}

} // namespace mlirgen
} // namespace merkurius::codegen
