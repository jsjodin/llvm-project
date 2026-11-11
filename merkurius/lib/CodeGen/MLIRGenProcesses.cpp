//===-- lib/CodeGen/MLIRGenProcesses.cpp - GPU processes' kernels ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Process kinds that run on a GPU (docs/INPROGRESS.md, GPU processes without
// omp target, step 4b, and GPU processes that run loops, one kernel for a
// GPU's processes). The device's compile of them (-fprocess-kernels) makes a
// start for each, which calls the kind's predicate, and
// merk_process_dispatch, which starts a kind by its number. The image's one
// kernel, merk_place, comes from its start object (process-start.cpp): each
// of its waves runs a process, which it starts by merk_process_dispatch. The
// host's compile embeds the image, one for each GPU, and registers it with
// the runtime at start-up, with the kinds' names in the order that numbers
// them.
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

namespace {
// The word of `words`, a generic pointer, at `index`.
mlir::Value LoadWord(mlir::OpBuilder &builder, mlir::Location loc,
    mlir::Value words, std::int32_t index) {
  auto i64{builder.getI64Type()};
  mlir::Value address{mlir::LLVM::GEPOp::create(builder, loc, words.getType(),
      i64, words, llvm::ArrayRef<mlir::LLVM::GEPArg>{index})};
  return mlir::LLVM::LoadOp::create(builder, loc, i64, address);
}
} // namespace

void MerkGenerator::GenerateProcessStarts() {
  if (!target_.options().processKernels) {
    return;
  }
  llvm::SmallVector<merk::ProcOp> starts;
  for (const parser::ProcessDecl *decl : gpuKinds_) {
    starts.push_back(GenerateProcessStart(*decl));
  }
  // merk_process_dispatch(kind, words): the start of the kind with the
  // number, by the kinds' order in the module, which the host's runtime
  // finds by their names (RegisterProcessImages). Every image has it, as its
  // kernel calls it (process-start.cpp), even one of a module without kinds
  // that run on a GPU, which the host's compile leaves out.
  mlir::Location loc{builder_.getUnknownLoc()};
  auto i64{builder_.getI64Type()};
  mlir::OpBuilder::InsertionGuard guard{builder_};
  builder_.setInsertionPointToEnd(module_.getBody());
  auto dispatch{merk::ProcOp::create(builder_, loc, "merk_process_dispatch",
      builder_.getFunctionType({i64, i64}, {}), merk::Determinism::Det)};
  dispatch->setAttr(kProcessRootAttrName, builder_.getUnitAttr());
  mlir::Block *entry{dispatch.addEntryBlock()};
  builder_.setInsertionPointToStart(entry);
  if (llvm::none_of(starts, [](merk::ProcOp start) { return bool(start); })) {
    merk::SucceedOp::create(builder_, loc, mlir::ValueRange{});
    return;
  }
  llvm::SmallVector<mlir::Attribute> cases;
  llvm::SmallVector<merk::ProcOp> called;
  for (auto [number, start] : llvm::enumerate(starts)) {
    if (start) {
      cases.push_back(
          builder_.getArrayAttr({builder_.getI64IntegerAttr(number)}));
      called.push_back(start);
    }
  }
  auto op{merk::SwitchOp::create(builder_, loc, mlir::TypeRange{},
      entry->getArgument(0), builder_.getArrayAttr(cases), /*canFail=*/false)};
  for (auto [region, start] : llvm::zip_equal(op.getCaseRegions(), called)) {
    builder_.setInsertionPointToStart(&region.emplaceBlock());
    merk::CallOp::create(builder_, loc, mlir::TypeRange{}, start.getSymName(),
        mlir::ValueRange{entry->getArgument(1)});
    merk::YieldOp::create(builder_, loc, mlir::ValueRange{});
  }
  builder_.setInsertionPointAfter(op);
  merk::SucceedOp::create(builder_, loc, mlir::ValueRange{});
}

// The start of a kind, merk_process_<kind>(words), which takes the address
// of the process's argument words, the kind's arguments and then its
// mailbox's handle, and calls the kind's predicate. In the first version,
// the arguments are integers, floats, characters and mailboxes, a word
// each but a mailbox, whose handle is five words, and the kind is not
// polymorphic (decision D3). The start puts a mailbox's handle in a cell on
// its stack: the predicate returns before the start does, and a handle that
// it sends is copied. Null for a kind it cannot start, which it reported.
merk::ProcOp MerkGenerator::GenerateProcessStart(
    const parser::ProcessDecl &decl) {
  parser::CharBlock at{Where(decl.tokens)};
  mlir::Location loc{Location(at)};
  const Procedure *procedure{KindProcedure(decl)};
  // A kind whose predicate code generation left out has been reported.
  merk::ProcOp kind{procedure ? procs_.lookup(procedure) : nullptr};
  if (!kind || kind.isExternal()) {
    return nullptr;
  }
  const semantics::Signature *signature{types_.SignatureOf(*procedure->symbol)};
  if (!signature) {
    return nullptr;
  }
  if (signature->typeVars != 0) {
    messages_.Say(at, "code generation for %s"_todo_en_US,
        "a process kind that runs on a GPU and is polymorphic");
    return nullptr;
  }
  // The kind's arguments and its mailbox; the io state has no value.
  llvm::SmallVector<mlir::Type> values;
  bool ok{true};
  for (const semantics::Type *type :
      signature->arguments.take_front(decl.arguments.size() + 1)) {
    std::optional<Represented> represented{Represent(type)};
    mlir::Type merkType{represented ? represented->type : mlir::Type{}};
    auto term{mlir::dyn_cast_if_present<merk::TermType>(merkType)};
    if (!(represented && !represented->dummy &&
            mlir::isa<mlir::IntegerType, mlir::FloatType>(merkType)) &&
        !(term && term.getName().starts_with("process.mailbox("))) {
      messages_.Say(at, "code generation for %s"_todo_en_US,
          "a process kind that runs on a GPU and takes a value of type " +
              semantics::TypeToString(*type) +
              ": its start takes integers, floats, characters and "
              "mailboxes");
      ok = false;
      continue;
    }
    values.push_back(merkType);
  }
  if (!ok) {
    return nullptr;
  }
  auto i64{builder_.getI64Type()};
  mlir::OpBuilder::InsertionGuard guard{builder_};
  builder_.setInsertionPointToEnd(module_.getBody());
  auto start{
      merk::ProcOp::create(builder_, loc, ("merk_process_" + decl.name).str(),
          builder_.getFunctionType({i64}, {}), merk::Determinism::Det)};
  start.setPrivate();
  mlir::Block *entry{start.addEntryBlock()};
  builder_.setInsertionPointToStart(entry);
  auto generic{mlir::LLVM::LLVMPointerType::get(
      &context_, target_.addressSpaces().generic)};
  auto stack{mlir::LLVM::LLVMPointerType::get(
      &context_, target_.addressSpaces().private_)};
  mlir::Value words{mlir::LLVM::IntToPtrOp::create(
      builder_, loc, generic, entry->getArgument(0))};
  std::int32_t next{0};
  llvm::SmallVector<mlir::Value> arguments;
  for (mlir::Type type : values) {
    if (auto integer{mlir::dyn_cast<mlir::IntegerType>(type)}) {
      mlir::Value value{LoadWord(builder_, loc, words, next++)};
      if (integer.getWidth() < 64) {
        value = mlir::arith::TruncIOp::create(builder_, loc, integer, value);
      }
      arguments.push_back(value);
      continue;
    }
    if (auto floating{mlir::dyn_cast<mlir::FloatType>(type)}) {
      // The float's bits, in the low bits of its word.
      mlir::Value value{LoadWord(builder_, loc, words, next++)};
      auto bits{builder_.getIntegerType(floating.getWidth())};
      if (floating.getWidth() < 64) {
        value = mlir::arith::TruncIOp::create(builder_, loc, bits, value);
      }
      arguments.push_back(
          mlir::arith::BitcastOp::create(builder_, loc, floating, value));
      continue;
    }
    auto cellType{mlir::LLVM::LLVMArrayType::get(i64, kMailboxWords)};
    mlir::Value cell{mlir::LLVM::AllocaOp::create(
        builder_, loc, stack, cellType, IntegerConstant(i64, 1, loc))};
    if (stack != generic) {
      cell = mlir::LLVM::AddrSpaceCastOp::create(builder_, loc, generic, cell);
    }
    for (std::int32_t i{0}; i < static_cast<std::int32_t>(kMailboxWords); ++i) {
      mlir::Value address{mlir::LLVM::GEPOp::create(builder_, loc, generic, i64,
          cell, llvm::ArrayRef<mlir::LLVM::GEPArg>{i})};
      mlir::LLVM::StoreOp::create(
          builder_, loc, LoadWord(builder_, loc, words, next++), address);
    }
    arguments.push_back(
        mlir::UnrealizedConversionCastOp::create(builder_, loc, type, cell)
            .getResult(0));
  }
  merk::CallOp::create(
      builder_, loc, mlir::TypeRange{}, kind.getSymName(), arguments);
  merk::SucceedOp::create(builder_, loc, mlir::ValueRange{});
  return start;
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
// as gpu:file, and registers each at start-up, with the module's name, the
// GPU's and its kinds' names, in their order, separated by spaces, in a
// procedure that a constructor calls, as message types are registered. The
// runtime loads an image when a process is first spawned on a GPU of its kind
// (step 4c). False after reporting an image that cannot be read.
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
  std::string kindNames;
  for (const parser::ProcessDecl *decl : gpuKinds_) {
    kindNames += (kindNames.empty() ? "" : " ") + decl->name.str();
  }
  mlir::LLVM::GlobalOp kinds{BytesGlobal(
      builder_, loc, prefix + ".kinds", kindNames + std::string(1, '\0'), 1)};
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
                static_cast<std::int64_t>(image.bytes->getBufferSize()), loc),
            mlir::LLVM::AddressOfOp::create(builder_, loc, kinds)},
        /*can_fail=*/false);
  }
  merk::SucceedOp::create(builder_, loc, mlir::ValueRange{});
  builder_.setInsertionPointToEnd(module_.getBody());
  merk::ConstructorOp::create(builder_, loc, procName);
  return true;
}

} // namespace mlirgen
} // namespace merkurius::codegen
