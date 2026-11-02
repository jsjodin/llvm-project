//===-- lib/CodeGen/MLIRGen.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "merkurius/CodeGen/MLIRGen.h"
#include "merkurius/Dialect/Merk/IR/MerkDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>

namespace merkurius::codegen {

using namespace parser::literals;
using semantics::Body;
using semantics::BodyKind;
using semantics::BodyVar;
using semantics::Procedure;
using semantics::ProcedureBody;
using semantics::UnifyKind;

static void Mangle(llvm::raw_ostream &os, llvm::StringRef name) {
  for (char c : name) {
    if (llvm::isAlnum(c) || c == '_') {
      os << c;
    } else {
      os << '$' << llvm::format_hex_no_prefix(
                       static_cast<unsigned char>(c), 2, /*Upper=*/false);
    }
  }
}

std::string MangledName(const Procedure &procedure) {
  const semantics::Symbol &symbol{*procedure.symbol};
  std::string result;
  llvm::raw_string_ostream os{result};
  llvm::SmallVector<llvm::StringRef> parts;
  symbol.module.split(parts, '.');
  for (llvm::StringRef part : parts) {
    Mangle(os, part);
    os << '.';
  }
  Mangle(os, symbol.name);
  os << '_' << (symbol.kind == semantics::SymbolKind::Function ? "f" : "")
     << symbol.arity << '_' << procedure.number;
  return result;
}

namespace {

// A Mercury type as code generation represents it: an MLIR type, or none
// for a type with no values to pass, such as io.state.
struct Represented {
  mlir::Type type;
  bool dummy{false};
};

class MerkGenerator {
public:
  MerkGenerator(mlir::MLIRContext &context, llvm::StringRef module,
      const parser::TokenSequence &tokens, const parser::AllSources &sources,
      semantics::ModeChecking &modes,
      const semantics::DeterminismChecking &determinism,
      const semantics::PolymorphismAnalysis &polymorphism,
      const TargetInfo &target, parser::Messages &messages)
      : context_{context}, builder_{&context}, moduleName_{module},
        tokens_{tokens}, sources_{sources}, modes_{modes},
        determinism_{determinism}, target_{target}, messages_{messages} {
    for (const semantics::ProcedurePolymorphism &procedure :
        polymorphism.procedures()) {
      polymorphism_[procedure.body] = &procedure;
    }
  }

  mlir::OwningOpRef<mlir::ModuleOp> Generate();

private:
  // The source of a goal, for messages, and its location, for operations.
  parser::CharBlock Where(semantics::TokenRange range) const {
    return parser::CharBlock{tokens_.tokens[range.first].spelling.begin(),
        tokens_.tokens[range.last].spelling.end()};
  }
  mlir::Location Location(parser::CharBlock at);
  mlir::Location Location(semantics::TokenRange range) {
    return Location(Where(range));
  }
  // Reports the first construct of a procedure that is not handled yet;
  // the procedure is then left out.
  void NotYet(parser::CharBlock at, llvm::StringRef what) {
    if (!failed_) {
      messages_.Say(at, "code generation for %s"_todo_en_US, what.str());
    }
    failed_ = true;
  }

  std::optional<Represented> Represent(const semantics::Type *);
  bool IsOutput(const semantics::ArgMode &mode) {
    return modes_.ops().IsFree(mode.initial) &&
        !modes_.ops().IsFree(mode.final);
  }
  // The procedure's merk.proc, declared if it is not defined yet; null
  // after reporting the reason if it cannot be.
  merk::ProcOp ProcFor(const Procedure &, llvm::ArrayRef<const semantics::Type *>,
      parser::CharBlock at);
  void GenerateBody(const ProcedureBody &);
  void Goal(const Body &, const ProcedureBody &);
  void Call(const Body &, const Procedure &, llvm::ArrayRef<BodyVar> arguments,
      const ProcedureBody &);
  // A variable's value; null for one of a type with no values.
  mlir::Value ValueOf(BodyVar var) { return values_.lookup(var); }

  mlir::MLIRContext &context_;
  mlir::OpBuilder builder_;
  llvm::StringRef moduleName_;
  const parser::TokenSequence &tokens_;
  const parser::AllSources &sources_;
  semantics::ModeChecking &modes_;
  const semantics::DeterminismChecking &determinism_;
  const TargetInfo &target_;
  parser::Messages &messages_;
  llvm::DenseMap<const ProcedureBody *, const semantics::ProcedurePolymorphism *>
      polymorphism_;
  mlir::ModuleOp module_;
  llvm::DenseMap<const Procedure *, merk::ProcOp> procs_;
  // For the body being generated: each variable's value, and whether a
  // construct was not handled.
  llvm::DenseMap<BodyVar, mlir::Value> values_;
  bool failed_{false};
};

mlir::Location MerkGenerator::Location(parser::CharBlock at) {
  if (auto range{sources_.GetProvenanceRange(at)}) {
    if (auto pos{sources_.GetSourcePosition(range->start())}) {
      return mlir::FileLineColLoc::get(&context_, *pos->path, pos->line,
          pos->column);
    }
  }
  return builder_.getUnknownLoc();
}

std::optional<Represented> MerkGenerator::Represent(
    const semantics::Type *type) {
  if (type->kind != semantics::TypeKind::Named) {
    return std::nullopt;
  }
  const semantics::Symbol &constructor{*type->constructor};
  llvm::StringRef name{constructor.name};
  if (constructor.module.empty()) {
    // The primitive types (reference manual, "Primitive types"); int, uint
    // and float are a word wide (design document, "Type representation").
    if (name == "int" || name == "uint") {
      return Represented{builder_.getIntegerType(target_.WordBits())};
    }
    for (unsigned bits : {8, 16, 32, 64}) {
      if (name == "int" + std::to_string(bits) ||
          name == "uint" + std::to_string(bits)) {
        return Represented{builder_.getIntegerType(bits)};
      }
    }
    if (name == "float") {
      return Represented{builder_.getF64Type()};
    }
    if (name == "character") {
      return Represented{builder_.getIntegerType(32)};
    }
    return std::nullopt;
  }
  // The state of the world has no values to pass (design document,
  // "Calling convention").
  if (constructor.module == "io" && name == "state") {
    return Represented{nullptr, /*dummy=*/true};
  }
  return std::nullopt;
}

static std::optional<merk::Determinism> MerkDeterminism(
    parser::Determinism determinism) {
  switch (determinism) {
  case parser::Determinism::Det:
    return merk::Determinism::Det;
  case parser::Determinism::CcMulti:
    return merk::Determinism::CcMulti;
  default:
    return std::nullopt;
  }
}

merk::ProcOp MerkGenerator::ProcFor(const Procedure &procedure,
    llvm::ArrayRef<const semantics::Type *> types, parser::CharBlock at) {
  if (merk::ProcOp proc{procs_.lookup(&procedure)}) {
    return proc;
  }
  std::optional<parser::Determinism> determinism{
      determinism_.DeterminismOf(procedure)};
  std::optional<merk::Determinism> merkDeterminism;
  if (determinism) {
    merkDeterminism = MerkDeterminism(*determinism);
  }
  if (!merkDeterminism) {
    NotYet(at,
        determinism
            ? std::string{semantics::DeterminismName(*determinism)} +
                " procedures"
            : std::string{"procedures of unknown determinism"});
    return nullptr;
  }
  llvm::SmallVector<mlir::Type> inputs, outputs;
  for (auto [mode, type] : llvm::zip(procedure.modes, types)) {
    std::optional<Represented> represented{Represent(type)};
    if (!represented) {
      NotYet(at, "values of type " + semantics::TypeToString(*type));
      return nullptr;
    }
    if (!represented->dummy) {
      (IsOutput(mode) ? outputs : inputs).push_back(represented->type);
    }
  }
  mlir::OpBuilder::InsertionGuard guard{builder_};
  builder_.setInsertionPointToEnd(module_.getBody());
  auto proc{merk::ProcOp::create(builder_, Location(procedure.symbol->source),
      MangledName(procedure), builder_.getFunctionType(inputs, outputs),
      *merkDeterminism)};
  proc.setPrivate();
  procs_[&procedure] = proc;
  return proc;
}

void MerkGenerator::Call(const Body &goal, const Procedure &callee,
    llvm::ArrayRef<BodyVar> arguments, const ProcedureBody &body) {
  if (callee.modes.size() != arguments.size()) {
    NotYet(Where(goal.tokens), "calls with hidden arguments");
    return;
  }
  llvm::SmallVector<const semantics::Type *> types;
  for (BodyVar argument : arguments) {
    types.push_back(body.vars[argument].type);
  }
  merk::ProcOp proc{ProcFor(callee, types, Where(goal.tokens))};
  if (!proc) {
    return;
  }
  llvm::SmallVector<mlir::Value> inputs;
  llvm::SmallVector<BodyVar> outputs;
  for (auto [argument, mode] : llvm::zip(arguments, callee.modes)) {
    if (Represent(body.vars[argument].type)->dummy) {
      continue;
    }
    if (IsOutput(mode)) {
      outputs.push_back(argument);
    } else {
      inputs.push_back(ValueOf(argument));
    }
  }
  auto call{merk::CallOp::create(builder_, Location(goal.tokens),
      proc.getFunctionType().getResults(), proc.getSymName(), inputs)};
  for (auto [var, result] : llvm::zip(outputs, call.getResults())) {
    values_[var] = result;
  }
}

void MerkGenerator::Goal(const Body &goal, const ProcedureBody &body) {
  if (failed_) {
    return;
  }
  switch (goal.kind) {
  case BodyKind::Conj:
    for (const Body *conjunct : goal.goals) {
      Goal(*conjunct, body);
    }
    return;
  case BodyKind::Scope:
    Goal(*goal.goals.front(), body);
    return;
  case BodyKind::Call:
    Call(goal, *goal.modes->callee, goal.arguments, body);
    return;
  case BodyKind::Unify:
    if (goal.isVar && goal.unify == UnifyKind::Assign) {
      // X = Y: the variable assigned is the other's value.
      if (values_.count(goal.var)) {
        values_[goal.left] = values_.lookup(goal.var);
      } else {
        values_[goal.var] = values_.lookup(goal.left);
      }
      return;
    }
    if (goal.unify == UnifyKind::Call && goal.modes && goal.modes->callee) {
      llvm::SmallVector<BodyVar> arguments(
          goal.arguments.begin(), goal.arguments.end());
      arguments.push_back(goal.left);
      Call(goal, *goal.modes->callee, arguments, body);
      return;
    }
    NotYet(Where(goal.tokens), "this unification");
    return;
  default:
    NotYet(Where(goal.tokens),
        "goals of kind " + std::string{semantics::EnumToString(goal.kind)});
    return;
  }
}

void MerkGenerator::GenerateBody(const ProcedureBody &body) {
  const Procedure &procedure{*body.procedure};
  parser::CharBlock at{procedure.symbol->source};
  failed_ = false;
  if (body.method) {
    NotYet(at, "instance methods");
    return;
  }
  if (const auto *polymorphism{polymorphism_.lookup(&body)};
      polymorphism && !polymorphism->hidden.empty()) {
    NotYet(at, "polymorphic procedures");
    return;
  }
  llvm::SmallVector<const semantics::Type *> types;
  for (BodyVar var : body.head) {
    types.push_back(body.vars[var].type);
  }
  merk::ProcOp proc{ProcFor(procedure, types, at)};
  if (!proc) {
    return;
  }
  if (procedure.symbol->exported) {
    proc.setPublic();
  }
  values_.clear();
  mlir::Block *entry{proc.addEntryBlock()};
  mlir::OpBuilder::InsertionGuard guard{builder_};
  builder_.setInsertionPointToStart(entry);
  llvm::SmallVector<BodyVar> outputs;
  unsigned input{0};
  for (auto [var, mode] : llvm::zip(body.head, procedure.modes)) {
    if (Represent(body.vars[var].type)->dummy) {
      continue;
    }
    if (IsOutput(mode)) {
      outputs.push_back(var);
    } else {
      values_[var] = entry->getArgument(input++);
    }
  }
  Goal(*body.body, body);
  if (failed_) {
    // Calls already made of it keep its declaration.
    proc.getBody().dropAllReferences();
    proc.getBody().getBlocks().clear();
    proc.setPrivate();
    return;
  }
  llvm::SmallVector<mlir::Value> results;
  for (BodyVar var : outputs) {
    results.push_back(ValueOf(var));
  }
  merk::SucceedOp::create(builder_, Location(at), results);
}

mlir::OwningOpRef<mlir::ModuleOp> MerkGenerator::Generate() {
  context_.loadDialect<merk::MerkDialect>();
  mlir::OwningOpRef<mlir::ModuleOp> owned{
      mlir::ModuleOp::create(builder_.getUnknownLoc(), moduleName_)};
  module_ = *owned;
  target_.Record(module_);
  for (const ProcedureBody &body : determinism_.bodies()) {
    if (body.procedure->symbol->module == moduleName_ || body.method) {
      GenerateBody(body);
    }
  }
  // The program's entry: the main module's main(di, uo).
  for (const ProcedureBody &body : determinism_.bodies()) {
    const semantics::Symbol &symbol{*body.procedure->symbol};
    if (!body.method && symbol.module == moduleName_ &&
        symbol.kind == semantics::SymbolKind::Predicate &&
        symbol.name == "main" && symbol.arity == 2) {
      if (merk::ProcOp proc{procs_.lookup(body.procedure)}) {
        builder_.setInsertionPointToEnd(module_.getBody());
        merk::EntryOp::create(
            builder_, Location(symbol.source), proc.getSymName());
      }
    }
  }
  return owned;
}

} // namespace

mlir::OwningOpRef<mlir::ModuleOp> GenerateMerk(mlir::MLIRContext &context,
    llvm::StringRef module, const parser::TokenSequence &tokens,
    const parser::AllSources &sources, semantics::ModeChecking &modes,
    const semantics::DeterminismChecking &determinism,
    const semantics::PolymorphismAnalysis &polymorphism,
    const TargetInfo &target, parser::Messages &messages) {
  return MerkGenerator{context, module, tokens, sources, modes, determinism,
      polymorphism, target, messages}
      .Generate();
}

} // namespace merkurius::codegen
