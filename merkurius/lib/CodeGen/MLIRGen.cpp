//===-- lib/CodeGen/MLIRGen.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "merkurius/CodeGen/MLIRGen.h"
#include "merkurius/Dialect/Merk/IR/MerkDialect.h"
#include "merkurius/Parser/parse-tree.h"
#include "merkurius/Parser/term-reader.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/ConvertUTF.h"
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

// The C function of a procedure's foreign_proc: merk_foreign_, then its
// symbol with `_` doubled, `.` as `_D` and `$` as `_S`.
static std::string ForeignName(const Procedure &procedure) {
  std::string result{"merk_foreign_"};
  for (char c : MangledName(procedure)) {
    result += c == '_' ? "__" : c == '.' ? "_D" : c == '$' ? "_S"
                                                            : std::string(1, c);
  }
  return result;
}

// The only code point a UTF-8 string has.
static std::optional<llvm::UTF32> OnlyCodePoint(llvm::StringRef text) {
  if (text.empty()) {
    return std::nullopt;
  }
  const auto *source{reinterpret_cast<const llvm::UTF8 *>(text.data())};
  llvm::UTF32 codePoint;
  if (llvm::convertUTF8Sequence(&source, source + text.size(), &codePoint,
          llvm::strictConversion) != llvm::conversionOK ||
      source != reinterpret_cast<const llvm::UTF8 *>(text.end())) {
    return std::nullopt;
  }
  return codePoint;
}

// The value of an integer literal (reference manual, "Integer literals"):
// decimal, 0b, 0o or 0x digits with `_` between them, or 0' and a
// character, and a suffix for the sized types.
static std::optional<llvm::APInt> IntegerValue(
    llvm::StringRef text, unsigned bits) {
  bool negative{text.consume_front("-")};
  llvm::APInt value;
  if (text.consume_front("0'")) {
    std::optional<llvm::UTF32> codePoint{
        OnlyCodePoint(parser::UnquoteSpelling(("'" + text + "'").str()))};
    if (!codePoint) {
      return std::nullopt;
    }
    value = llvm::APInt{bits, *codePoint};
  } else {
    unsigned radix{10};
    if (text.consume_front("0b")) {
      radix = 2;
    } else if (text.consume_front("0o")) {
      radix = 8;
    } else if (text.consume_front("0x")) {
      radix = 16;
    }
    // No digit is i or u.
    text = text.take_until([](char c) { return c == 'i' || c == 'u'; });
    std::string digits{text};
    llvm::erase(digits, '_');
    if (llvm::StringRef{digits}.getAsInteger(radix, value)) {
      return std::nullopt;
    }
    value = value.zextOrTrunc(bits);
  }
  if (negative) {
    value.negate();
  }
  return value;
}

// The C type of a Mercury type's values (reference manual, "C data passing
// conventions"), for the primitive types code generation passes.
static std::string CType(const semantics::Type &type) {
  llvm::StringRef name{type.constructor->name};
  if (name == "int") {
    return "MR_Integer";
  }
  if (name == "uint") {
    return "MR_Unsigned";
  }
  if (name == "float") {
    return "MR_Float";
  }
  if (name == "character") {
    return "MR_Char";
  }
  if (name == "string") {
    return "MR_String";
  }
  return (name + "_t").str(); // intN and uintN
}

// The start of the C file of a module's foreign code: the C types of the
// manual's "C data passing conventions".
static constexpr llvm::StringLiteral kForeignPrelude{R"(/* Foreign code, written by merkc. */
#include <stdint.h>
typedef intptr_t MR_Integer;
typedef uintptr_t MR_Unsigned;
typedef double MR_Float;
typedef int32_t MR_Char;
typedef char *MR_String;
typedef const char *MR_ConstString;
typedef uintptr_t MR_Word;
typedef int MR_bool;
#define MR_TRUE 1
#define MR_FALSE 0
)"};

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
      const parser::ParseTree &tree, const parser::TokenSequence &tokens,
      const parser::AllSources &sources, const semantics::TypeChecking &types,
      semantics::ModeChecking &modes,
      const semantics::DeterminismChecking &determinism,
      const semantics::PolymorphismAnalysis &polymorphism,
      const TargetInfo &target, parser::Messages &messages)
      : context_{context}, builder_{&context}, moduleName_{module},
        tree_{tree}, tokens_{tokens}, sources_{sources}, types_{types},
        modes_{modes},
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
  // A procedure's body from its foreign_proc: a call of the C function
  // around the foreign_proc's code, which is added to foreignCode_.
  void GenerateForeign(const Procedure &, const parser::ForeignProcPragma &);
  void Goal(const Body &, const ProcedureBody &);
  // X = c, for a literal or a character c; null after reporting a constant
  // not handled.
  mlir::Value Constant(const Body &, const ProcedureBody &);
  // A #line directive for the C compiler, for a range of the source.
  std::string LineDirective(parser::TokenRange);
  void Call(const Body &, const Procedure &, llvm::ArrayRef<BodyVar> arguments,
      const ProcedureBody &);
  // A call of a predicate or function of int, uint, the sized integer
  // modules or float that code generation has built in, as arith
  // operations; false if it is to be called instead.
  bool Builtin(const Body &, const Procedure &,
      llvm::ArrayRef<BodyVar> arguments, const ProcedureBody &);
  // X = Y, or X = c, with both bound: a test.
  void Test(const Body &, mlir::Value, mlir::Value);
  void IfThenElse(const Body &, const ProcedureBody &);
  void Switch(const Body &, const ProcedureBody &);
  void Not(const Body &, const ProcedureBody &);
  // A goal with an OpenMP directive (design document, "Target regions").
  void Omp(const Body &, const ProcedureBody &);
  // A stack slot of the procedure for a value of `type`, as a generic
  // pointer, which omp.map.info maps.
  mlir::Value Slot(mlir::Type type, mlir::Location);
  // The variables bound in a branch, after `before`, in order.
  using Values = llvm::DenseMap<BodyVar, mlir::Value>;
  static std::vector<BodyVar> BoundSince(const Values &now, const Values &before);
  // A branch of an if-then-else or a switch, as generated: its block, the
  // values of the variables after it, and whether it fails.
  struct Branch {
    mlir::Block *block;
    Values values;
    bool fails;
  };
  // Ends the branches that succeed with the values of the variables they
  // all bind, and remakes `op` with those as its results, which are then
  // the variables' values.
  void Join(mlir::Operation *op, llvm::ArrayRef<Branch>, const Values &before,
      mlir::Location);
  // Whether the block being generated ends, after a goal that cannot
  // succeed.
  bool Terminated() {
    mlir::Block *block{builder_.getInsertionBlock()};
    return !block->empty() &&
        block->back().hasTrait<mlir::OpTrait::IsTerminator>();
  }
  mlir::Value IntegerConstant(mlir::Type type, std::int64_t value, mlir::Location loc) {
    return mlir::arith::ConstantOp::create(
        builder_, loc, builder_.getIntegerAttr(type, value));
  }
  // A variable's value; null for one of a type with no values.
  mlir::Value ValueOf(BodyVar var) { return values_.lookup(var); }

  mlir::MLIRContext &context_;
  mlir::OpBuilder builder_;
  llvm::StringRef moduleName_;
  const parser::ParseTree &tree_;
  const parser::TokenSequence &tokens_;
  const parser::AllSources &sources_;
  const semantics::TypeChecking &types_;
  semantics::ModeChecking &modes_;
  const semantics::DeterminismChecking &determinism_;
  const TargetInfo &target_;
  parser::Messages &messages_;
  llvm::DenseMap<const ProcedureBody *, const semantics::ProcedurePolymorphism *>
      polymorphism_;
  mlir::ModuleOp module_;
  llvm::DenseMap<const Procedure *, merk::ProcOp> procs_;
  // The C functions of the module's foreign_procs.
  std::string foreignCode_;
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
    if (name == "string") {
      return Represented{merk::StringType::get(&context_)};
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
  case parser::Determinism::Semidet:
    return merk::Determinism::Semidet;
  case parser::Determinism::Failure:
    return merk::Determinism::Failure;
  case parser::Determinism::Erroneous:
    return merk::Determinism::Erroneous;
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
  if (Builtin(goal, callee, arguments, body)) {
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
  if (failed_ || Terminated()) {
    return;
  }
  switch (goal.kind) {
  case BodyKind::Disj:
    if (goal.goals.empty()) {
      merk::FailOp::create(builder_, Location(goal.tokens));
      return;
    }
    NotYet(Where(goal.tokens), "disjunctions");
    return;
  case BodyKind::IfThenElse:
    IfThenElse(goal, body);
    return;
  case BodyKind::Switch:
    Switch(goal, body);
    return;
  case BodyKind::Not:
    Not(goal, body);
    return;
  case BodyKind::Conj:
    for (const Body *conjunct : goal.goals) {
      Goal(*conjunct, body);
    }
    return;
  case BodyKind::Scope:
    if (goal.directive) {
      Omp(goal, body);
      return;
    }
    Goal(*goal.goals.front(), body);
    return;
  case BodyKind::Call:
    Call(goal, *goal.modes->callee, goal.arguments, body);
    return;
  case BodyKind::Unify:
    if (goal.switchTest) {
      return; // the switch's case has tested it
    }
    if (goal.isVar && goal.unify == UnifyKind::Test) {
      // X = X succeeds, even for a NaN, as with mmc.
      if (goal.left != goal.var) {
        Test(goal, ValueOf(goal.left), ValueOf(goal.var));
      }
      return;
    }
    if (!goal.isVar && goal.arguments.empty() && goal.source &&
        (goal.unify == UnifyKind::Deconstruct ||
            goal.unify == UnifyKind::Test)) {
      if (mlir::Value constant{Constant(goal, body)}) {
        Test(goal, ValueOf(goal.left), constant);
      }
      return;
    }
    if (goal.isVar && goal.unify == UnifyKind::Assign) {
      // X = Y: the variable assigned is the other's value.
      if (values_.count(goal.var)) {
        values_[goal.left] = values_.lookup(goal.var);
      } else {
        values_[goal.var] = values_.lookup(goal.left);
      }
      return;
    }
    if (goal.unify == UnifyKind::Construct && !goal.isVar &&
        goal.arguments.empty() && goal.source) {
      if (mlir::Value value{Constant(goal, body)}) {
        values_[goal.left] = value;
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

mlir::Value MerkGenerator::Constant(const Body &goal, const ProcedureBody &body) {
  const semantics::Type *type{body.vars[goal.left].type};
  std::optional<Represented> represented{Represent(type)};
  if (!represented || represented->dummy) {
    NotYet(Where(goal.tokens),
        "values of type " + semantics::TypeToString(*type));
    return nullptr;
  }
  mlir::Type mlirType{represented->type};
  const semantics::Functor &functor{goal.source->functor};
  mlir::TypedAttr value;
  if (functor.kind == semantics::FunctorKind::Literal) {
    llvm::StringRef text{functor.literal->text};
    switch (functor.literal->kind) {
    case parser::LiteralKind::Integer:
      if (auto integer{mlir::dyn_cast<mlir::IntegerType>(mlirType)}) {
        if (std::optional<llvm::APInt> v{
                IntegerValue(text, integer.getWidth())}) {
          value = builder_.getIntegerAttr(integer, *v);
        }
      }
      break;
    case parser::LiteralKind::Float:
      if (auto f{mlir::dyn_cast<mlir::FloatType>(mlirType)}) {
        std::string digits{text};
        llvm::erase(digits, '_');
        llvm::APFloat v{f.getFloatSemantics()};
        if (auto status{v.convertFromString(
                digits, llvm::APFloat::rmNearestTiesToEven)}) {
          value = builder_.getFloatAttr(f, v);
        } else {
          llvm::consumeError(status.takeError());
        }
      }
      break;
    case parser::LiteralKind::String:
      if (mlir::isa<merk::StringType>(mlirType)) {
        value = mlir::StringAttr::get(parser::UnquoteSpelling(text), mlirType);
      }
      break;
    }
  } else if (functor.kind == semantics::FunctorKind::Name &&
      mlirType.isInteger(32) && type->constructor->name == "character") {
    // A character constant is its code point.
    if (std::optional<llvm::UTF32> codePoint{
            OnlyCodePoint(functor.name->name)}) {
      value = builder_.getI32IntegerAttr(*codePoint);
    }
  }
  if (!value) {
    NotYet(Where(goal.tokens), "this unification");
    return nullptr;
  }
  return merk::ConstantOp::create(
      builder_, Location(goal.tokens), value.getType(), value);
}

std::string MerkGenerator::LineDirective(parser::TokenRange range) {
  parser::CharBlock at{Where(range)};
  if (auto provenance{sources_.GetProvenanceRange(at)}) {
    if (auto pos{sources_.GetSourcePosition(provenance->start())}) {
      std::string path{*pos->path};
      std::string escaped;
      for (char c : path) {
        if (c == '"' || c == '\\') {
          escaped += '\\';
        }
        escaped += c;
      }
      return "#line " + std::to_string(pos->line) + " \"" + escaped + "\"\n";
    }
  }
  return "";
}

void MerkGenerator::GenerateForeign(
    const Procedure &procedure, const parser::ForeignProcPragma &pragma) {
  parser::CharBlock at{procedure.symbol->source};
  failed_ = false;
  const semantics::Signature *signature{
      types_.SignatureOf(*procedure.symbol)};
  if (!signature) {
    return;
  }
  merk::ProcOp proc{ProcFor(procedure, signature->arguments, at)};
  if (!proc) {
    return;
  }
  if (procedure.symbol->exported) {
    proc.setPublic();
  }
  // The foreign_proc's variables: its arguments', then a function's
  // result.
  llvm::SmallVector<const parser::ForeignProcArg *> arguments(
      pragma.arguments.begin(), pragma.arguments.end());
  if (pragma.result) {
    arguments.push_back(pragma.result);
  }
  // The C function takes the inputs, and returns the only output or else
  // stores the outputs through pointers after the inputs; that of a
  // procedure that can fail returns SUCCESS_INDICATOR, and stores every
  // output. A variable of a type with no values, such as io.state, is an
  // MR_Word set to 0.
  merk::Determinism determinism{proc.getDeterminism()};
  bool canFail{determinism == merk::Determinism::Semidet ||
      determinism == merk::Determinism::Failure ||
      determinism == merk::Determinism::CcNondet};
  std::string name{ForeignName(procedure)};
  std::string parameters, locals, stores;
  if (canFail) {
    locals += "  MR_bool SUCCESS_INDICATOR = MR_TRUE;\n";
  }
  llvm::SmallVector<std::pair<std::string, std::string>> outputs;
  for (auto [argument, type, mode] :
      llvm::zip(arguments, signature->arguments, procedure.modes)) {
    std::string var{argument->var->name};
    if (Represent(type)->dummy) {
      locals += "  MR_Word " + var + " = 0;\n  (void)" + var + ";\n";
    } else if (IsOutput(mode)) {
      outputs.emplace_back(CType(*type), var);
      locals += "  " + CType(*type) + " " + var + ";\n";
    } else {
      parameters += (parameters.empty() ? "" : ", ") + CType(*type) + " " + var;
    }
  }
  std::string result{"void"};
  if (outputs.size() == 1 && !canFail) {
    result = outputs.front().first;
    stores = "  return " + outputs.front().second + ";\n";
  } else {
    for (auto &[ctype, var] : outputs) {
      parameters += (parameters.empty() ? "" : ", ") + ctype + " *merk_out_" + var;
      stores += "  *merk_out_" + var + " = " + var + ";\n";
    }
    if (canFail) {
      result = "MR_bool";
      stores += "  return SUCCESS_INDICATOR;\n";
    }
  }
  foreignCode_ += "\n" + result + " " + name + "(" +
      (parameters.empty() ? "void" : parameters) + ") {\n" + locals + "  {\n" +
      LineDirective(pragma.code->tokens) + pragma.code->text.str() +
      "\n  }\n" + stores + "}\n";

  mlir::Block *entry{proc.addEntryBlock()};
  mlir::OpBuilder::InsertionGuard guard{builder_};
  builder_.setInsertionPointToStart(entry);
  mlir::Location loc{Location(Where(pragma.tokens))};
  llvm::SmallVector<mlir::Type> results;
  if (canFail) {
    results.push_back(builder_.getI1Type());
  }
  results.append(proc.getFunctionType().getResults().begin(),
      proc.getFunctionType().getResults().end());
  auto call{merk::ForeignCallOp::create(builder_, loc, results, name,
      entry->getArguments(), canFail)};
  mlir::ValueRange solution{call.getResults()};
  if (canFail) {
    merk::GuardOp::create(builder_, loc, solution.front());
    solution = solution.drop_front();
  }
  merk::SucceedOp::create(builder_, loc, solution);
}

void MerkGenerator::Test(const Body &goal, mlir::Value x, mlir::Value y) {
  mlir::Location loc{Location(goal.tokens)};
  mlir::Value equal;
  if (x && y && mlir::isa<mlir::IntegerType>(x.getType())) {
    equal = mlir::arith::CmpIOp::create(
        builder_, loc, mlir::arith::CmpIPredicate::eq, x, y);
  } else if (x && y && mlir::isa<mlir::FloatType>(x.getType())) {
    equal = mlir::arith::CmpFOp::create(
        builder_, loc, mlir::arith::CmpFPredicate::OEQ, x, y);
  } else {
    NotYet(Where(goal.tokens), "this unification");
    return;
  }
  merk::GuardOp::create(builder_, loc, equal);
}

std::vector<BodyVar> MerkGenerator::BoundSince(
    const Values &now, const Values &before) {
  std::vector<BodyVar> bound;
  for (const auto &[var, value] : now) {
    if (!before.count(var)) {
      bound.push_back(var);
    }
  }
  llvm::sort(bound);
  return bound;
}

void MerkGenerator::Join(mlir::Operation *op, llvm::ArrayRef<Branch> branches,
    const Values &before, mlir::Location loc) {
  // The variables that every branch that succeeds binds; those of a branch
  // only are its own.
  std::optional<std::vector<BodyVar>> results;
  for (const Branch &branch : branches) {
    if (branch.fails) {
      continue;
    }
    std::vector<BodyVar> bound{BoundSince(branch.values, before)};
    if (!results) {
      results = std::move(bound);
    } else {
      std::vector<BodyVar> both;
      std::set_intersection(results->begin(), results->end(), bound.begin(),
          bound.end(), std::back_inserter(both));
      results = std::move(both);
    }
  }
  if (!results) {
    results.emplace();
  }
  llvm::SmallVector<mlir::Type> types;
  for (const Branch &branch : branches) {
    if (branch.fails) {
      continue;
    }
    llvm::SmallVector<mlir::Value> values;
    for (BodyVar var : *results) {
      values.push_back(branch.values.lookup(var));
    }
    if (types.empty() && !values.empty()) {
      for (mlir::Value value : values) {
        types.push_back(value.getType());
      }
    }
    mlir::OpBuilder::InsertionGuard guard{builder_};
    builder_.setInsertionPointToEnd(branch.block);
    merk::YieldOp::create(builder_, loc, values);
  }
  // The operation again, with the results.
  mlir::OperationState state{op->getLoc(), op->getName()};
  state.addOperands(op->getOperands());
  state.addAttributes(op->getAttrs());
  state.addTypes(types);
  for (mlir::Region &region : op->getRegions()) {
    state.addRegion()->takeBody(region);
  }
  builder_.setInsertionPoint(op);
  mlir::Operation *joined{builder_.create(state)};
  op->erase();
  builder_.setInsertionPointAfter(joined);
  values_ = before;
  for (auto [var, result] : llvm::zip_equal(*results, joined->getResults())) {
    values_[var] = result;
  }
  if (llvm::all_of(branches, [](const Branch &b) { return b.fails; })) {
    // No branch succeeds, so neither does the goal.
    merk::FailOp::create(builder_, loc);
  }
}

void MerkGenerator::IfThenElse(const Body &goal, const ProcedureBody &body) {
  mlir::Location loc{Location(goal.tokens)};
  Values before{values_};
  auto ite{merk::IteOp::create(builder_, loc, mlir::TypeRange{})};
  llvm::SmallVector<Branch, 2> branches;
  {
    mlir::OpBuilder::InsertionGuard guard{builder_};
    // The condition yields what it binds, to the then part.
    builder_.setInsertionPointToStart(&ite.getCondition().emplaceBlock());
    Goal(*goal.goals[0], body);
    std::vector<BodyVar> bound{BoundSince(values_, before)};
    llvm::SmallVector<mlir::Value> yielded;
    for (BodyVar var : bound) {
      yielded.push_back(values_.lookup(var));
    }
    merk::YieldOp conditionYield;
    if (!Terminated()) {
      conditionYield = merk::YieldOp::create(builder_, loc, yielded);
    }
    mlir::Block &then{ite.getThenRegion().emplaceBlock()};
    values_ = before;
    for (auto [var, value] : llvm::zip(bound, yielded)) {
      values_[var] = then.addArgument(value.getType(), loc);
    }
    builder_.setInsertionPointToStart(&then);
    Goal(*goal.goals[1], body);
    branches.push_back({&then, values_, Terminated()});
    mlir::Block &otherwise{ite.getElseRegion().emplaceBlock()};
    values_ = before;
    builder_.setInsertionPointToStart(&otherwise);
    Goal(*goal.goals[2], body);
    branches.push_back({&otherwise, values_, Terminated()});
    if (failed_) {
      values_ = before;
      return;
    }
    Join(ite, branches, before, loc);
    // The condition passes the then part only the values it uses: those
    // the condition binds for itself are local to it.
    for (unsigned i = then.getNumArguments(); i-- > 0;) {
      if (then.getArgument(i).use_empty()) {
        then.eraseArgument(i);
        if (conditionYield) {
          conditionYield.getValuesMutable().erase(i);
        }
      }
    }
  }
}

// The constant of a switch's case: an integer literal, or a character's
// code point.
static std::optional<llvm::APInt> CaseConstant(const semantics::InstFunctor &functor,
    const semantics::Type &type, unsigned bits) {
  if (functor.arity != 0) {
    return std::nullopt;
  }
  if (functor.literal) {
    return IntegerValue(functor.name, bits);
  }
  if (type.kind == semantics::TypeKind::Named &&
      type.constructor->name == "character") {
    if (std::optional<llvm::UTF32> codePoint{OnlyCodePoint(functor.name)}) {
      return llvm::APInt{bits, *codePoint};
    }
  }
  return std::nullopt;
}

void MerkGenerator::Switch(const Body &goal, const ProcedureBody &body) {
  mlir::Location loc{Location(goal.tokens)};
  const semantics::Type *type{body.vars[goal.switchVar].type};
  mlir::Value value{ValueOf(goal.switchVar)};
  auto integer{value ? mlir::dyn_cast<mlir::IntegerType>(value.getType())
                     : mlir::IntegerType{}};
  if (!integer) {
    NotYet(Where(goal.tokens),
        "switches on values of type " + semantics::TypeToString(*type));
    return;
  }
  llvm::SmallVector<mlir::Attribute> cases;
  for (const semantics::SwitchCase &c : goal.cases) {
    llvm::SmallVector<mlir::Attribute> constants;
    std::vector<semantics::InstFunctor> functors{c.functor};
    functors.insert(functors.end(), c.others.begin(), c.others.end());
    for (const semantics::InstFunctor &functor : functors) {
      std::optional<llvm::APInt> v{
          CaseConstant(functor, *type, integer.getWidth())};
      if (!v) {
        NotYet(Where(goal.tokens),
            "switches on values of type " + semantics::TypeToString(*type));
        return;
      }
      constants.push_back(builder_.getIntegerAttr(integer, *v));
    }
    cases.push_back(builder_.getArrayAttr(constants));
  }
  Values before{values_};
  auto op{merk::SwitchOp::create(builder_, loc, mlir::TypeRange{}, value,
      builder_.getArrayAttr(cases), goal.canFail)};
  llvm::SmallVector<Branch> branches;
  {
    mlir::OpBuilder::InsertionGuard guard{builder_};
    for (auto [c, region] : llvm::zip(goal.cases, op.getCaseRegions())) {
      mlir::Block &block{region.emplaceBlock()};
      values_ = before;
      builder_.setInsertionPointToStart(&block);
      Goal(*c.goal, body);
      branches.push_back({&block, values_, Terminated()});
    }
  }
  if (failed_) {
    values_ = before;
    return;
  }
  Join(op, branches, before, loc);
}

void MerkGenerator::Not(const Body &goal, const ProcedureBody &body) {
  mlir::Location loc{Location(goal.tokens)};
  Values before{values_};
  auto op{merk::NotOp::create(builder_, loc)};
  {
    mlir::OpBuilder::InsertionGuard guard{builder_};
    builder_.setInsertionPointToStart(&op.getGoal().emplaceBlock());
    Goal(*goal.goals.front(), body);
    if (!Terminated()) {
      merk::YieldOp::create(builder_, loc, mlir::ValueRange{});
    }
  }
  values_ = before;
}

// The library's message for an exception, as mmc writes the exception: a
// domain_error of a name of the operation, as int.'//' or int.(<<).
static std::string DomainError(llvm::StringRef module, llvm::StringRef what,
    llvm::StringRef problem) {
  return ("domain_error(\"" + module + "." + what + ": " + problem + "\")").str();
}

bool MerkGenerator::Builtin(const Body &goal, const Procedure &callee,
    llvm::ArrayRef<BodyVar> arguments, const ProcedureBody &body) {
  const semantics::Symbol &symbol{*callee.symbol};
  llvm::StringRef module{symbol.module};
  bool isFloat{module == "float"};
  bool isUnsigned{module.starts_with("uint")};
  bool isInteger{llvm::is_contained(
      {"int", "uint", "int8", "int16", "int32", "int64", "uint8", "uint16",
          "uint32", "uint64"},
      module)};
  if (!isFloat && !isInteger) {
    return false;
  }
  // What is not built in is called: float's functions are the compiled
  // library's; the integer modules have no implementation yet.
  auto other{[&] {
    if (isFloat) {
      return false;
    }
    NotYet(Where(goal.tokens),
        ("calls of " + module + "." + symbol.name + "/" +
            std::to_string(symbol.arity))
            .str());
    return true;
  }};
  bool function{symbol.kind == semantics::SymbolKind::Function};
  llvm::SmallVector<mlir::Value> in;
  for (auto [i, argument, mode] : llvm::enumerate(arguments, callee.modes)) {
    bool result{function && i + 1 == arguments.size()};
    if (IsOutput(mode) != result || (!result && !ValueOf(argument))) {
      return other();
    }
    if (!result) {
      in.push_back(ValueOf(argument));
    }
  }
  mlir::Location loc{Location(goal.tokens)};
  llvm::StringRef name{symbol.name};
  namespace arith = mlir::arith;
  if (!function && isInteger && in.size() == 1 &&
      (name == "even" || name == "odd")) {
    mlir::Value low{arith::AndIOp::create(builder_, loc, in[0],
        IntegerConstant(in[0].getType(), 1, loc))};
    merk::GuardOp::create(builder_, loc,
        arith::CmpIOp::create(builder_, loc, arith::CmpIPredicate::eq, low,
            IntegerConstant(in[0].getType(), name == "odd", loc)));
    return true;
  }
  if (!function) {
    // The comparisons, which are tests.
    if (in.size() != 2) {
      return other();
    }
    mlir::Value flag;
    if (isFloat) {
      std::optional<arith::CmpFPredicate> predicate{
          name == "<"    ? arith::CmpFPredicate::OLT
              : name == ">"  ? arith::CmpFPredicate::OGT
              : name == "=<" ? arith::CmpFPredicate::OLE
              : name == ">=" ? arith::CmpFPredicate::OGE
                             : std::optional<arith::CmpFPredicate>{}};
      if (!predicate) {
        return other();
      }
      flag = arith::CmpFOp::create(builder_, loc, *predicate, in[0], in[1]);
    } else {
      std::optional<arith::CmpIPredicate> predicate{name == "<"
              ? (isUnsigned ? arith::CmpIPredicate::ult : arith::CmpIPredicate::slt)
              : name == ">"
              ? (isUnsigned ? arith::CmpIPredicate::ugt : arith::CmpIPredicate::sgt)
              : name == "=<"
              ? (isUnsigned ? arith::CmpIPredicate::ule : arith::CmpIPredicate::sle)
              : name == ">="
              ? (isUnsigned ? arith::CmpIPredicate::uge : arith::CmpIPredicate::sge)
              : std::optional<arith::CmpIPredicate>{}};
      if (!predicate) {
        return other();
      }
      flag = arith::CmpIOp::create(builder_, loc, *predicate, in[0], in[1]);
    }
    merk::GuardOp::create(builder_, loc, flag);
    return true;
  }
  mlir::Value result;
  BodyVar output{arguments.back()};
  if (isFloat) {
    if (in.size() == 2 && name == "+") {
      result = arith::AddFOp::create(builder_, loc, in[0], in[1]);
    } else if (in.size() == 2 && name == "-") {
      result = arith::SubFOp::create(builder_, loc, in[0], in[1]);
    } else if (in.size() == 2 && name == "*") {
      result = arith::MulFOp::create(builder_, loc, in[0], in[1]);
    } else if (in.size() == 2 && name == "/") {
      mlir::Value zero{arith::ConstantOp::create(
          builder_, loc, builder_.getFloatAttr(in[1].getType(), 0.0))};
      mlir::Value isZero{arith::CmpFOp::create(
          builder_, loc, arith::CmpFPredicate::OEQ, in[1], zero)};
      merk::ThrowIfOp::create(builder_, loc, isZero,
          DomainError(module, "\\'/\\'", "division by zero"));
      result = arith::DivFOp::create(builder_, loc, in[0], in[1]);
    } else if (in.size() == 2 && name == "unchecked_quotient") {
      result = arith::DivFOp::create(builder_, loc, in[0], in[1]);
    } else if (in.size() == 1 && name == "-") {
      // As mmc's programs negate: 0.0 - X, so that -0.0 is 0.0.
      mlir::Value zero{arith::ConstantOp::create(
          builder_, loc, builder_.getFloatAttr(in[0].getType(), 0.0))};
      result = arith::SubFOp::create(builder_, loc, zero, in[0]);
    } else if (in.size() == 1 && name == "+") {
      result = in[0];
    } else if (in.size() == 1 && name == "float" &&
        mlir::isa<mlir::IntegerType>(in[0].getType())) {
      result = arith::SIToFPOp::create(
          builder_, loc, builder_.getF64Type(), in[0]);
    } else {
      return other();
    }
    values_[output] = result;
    return true;
  }
  std::optional<Represented> outputType{Represent(body.vars[output].type)};
  if (in.empty()) {
    // The constants: the least and greatest values, and the width.
    auto type{outputType
            ? mlir::dyn_cast_or_null<mlir::IntegerType>(outputType->type)
            : mlir::IntegerType{}};
    if (!type) {
      return other();
    }
    unsigned bits{type.getWidth()};
    llvm::StringRef bounded{name};
    llvm::APInt value;
    if (bounded.consume_front("max_") && bounded == module) {
      value = isUnsigned ? llvm::APInt::getMaxValue(bits)
                         : llvm::APInt::getSignedMaxValue(bits);
    } else if (bounded.consume_front("min_") && bounded == module) {
      value = isUnsigned ? llvm::APInt::getZero(bits)
                         : llvm::APInt::getSignedMinValue(bits);
    } else if (name == ("bits_per_" + module).str()) {
      value = llvm::APInt{bits, bits};
    } else {
      return other();
    }
    values_[output] = arith::ConstantOp::create(
        builder_, loc, builder_.getIntegerAttr(type, value));
    return true;
  }
  auto type{mlir::cast<mlir::IntegerType>(in[0].getType())};
  unsigned bits{type.getWidth()};
  if (in.size() == 1 &&
      (name.starts_with("cast_") || name == "to_int" ||
          name == "from_int" || name == "to_uint")) {
    // A conversion between integer types keeps the value's low bits, and
    // extends it by its sign if its type is signed (library reference,
    // "int8" and the others).
    auto to{outputType
            ? mlir::dyn_cast_or_null<mlir::IntegerType>(outputType->type)
            : mlir::IntegerType{}};
    if (!to) {
      return other();
    }
    llvm::StringRef from{module};
    if (name.starts_with("cast_from_")) {
      from = name.drop_front(llvm::StringRef{"cast_from_"}.size());
    } else if (name == "from_int") {
      from = "int";
    }
    if (to.getWidth() < bits) {
      result = arith::TruncIOp::create(builder_, loc, to, in[0]);
    } else if (to.getWidth() == bits) {
      result = in[0];
    } else if (from.starts_with("uint")) {
      result = arith::ExtUIOp::create(builder_, loc, to, in[0]);
    } else {
      result = arith::ExtSIOp::create(builder_, loc, to, in[0]);
    }
    values_[output] = result;
    return true;
  }
  auto constant{[&](std::int64_t v) { return IntegerConstant(type, v, loc); }};
  auto equal{[&](mlir::Value x, std::int64_t v) {
    return arith::CmpIOp::create(
        builder_, loc, arith::CmpIPredicate::eq, x, constant(v));
  }};
  auto select{[&](mlir::Value c, mlir::Value t, mlir::Value f) {
    return arith::SelectOp::create(builder_, loc, c, t, f);
  }};
  // A division checks for a zero divisor; one of the least value by -1,
  // whose quotient overflows, wraps, as mmc's constant folding does.
  auto divide{[&](llvm::StringRef what, bool floor, bool remainder) {
    mlir::Value a{in[0]}, b{in[1]};
    merk::ThrowIfOp::create(builder_, loc, equal(b, 0),
        DomainError(module, what, "division by zero"));
    if (isUnsigned) {
      return remainder ? mlir::Value{arith::RemUIOp::create(builder_, loc, a, b)}
                       : mlir::Value{arith::DivUIOp::create(builder_, loc, a, b)};
    }
    mlir::Value minusOne{equal(b, -1)};
    mlir::Value divisor{select(minusOne, constant(1), b)};
    mlir::Value q{arith::DivSIOp::create(builder_, loc, a, divisor)};
    mlir::Value r{arith::RemSIOp::create(builder_, loc, a, divisor)};
    q = select(minusOne, arith::SubIOp::create(builder_, loc, constant(0), a), q);
    r = select(minusOne, constant(0), r);
    if (floor) {
      // Toward minus infinity: one less where the remainder is not 0 and
      // its sign is not the divisor's.
      mlir::Value nonzero{arith::CmpIOp::create(
          builder_, loc, arith::CmpIPredicate::ne, r, constant(0))};
      mlir::Value signs{arith::CmpIOp::create(builder_, loc,
          arith::CmpIPredicate::slt, arith::XOrIOp::create(builder_, loc, r, b),
          constant(0))};
      mlir::Value adjust{arith::AndIOp::create(builder_, loc, nonzero, signs)};
      q = select(adjust, arith::SubIOp::create(builder_, loc, q, constant(1)), q);
      r = select(adjust, arith::AddIOp::create(builder_, loc, r, b), r);
    }
    return remainder ? r : q;
  }};
  // The shift amount is an int; outside 0 to the width it is an error,
  // unless the shift is unchecked.
  auto shift{[&](bool left, bool checked, llvm::StringRef what) {
    mlir::Value amount{in[1]};
    auto amountType{mlir::cast<mlir::IntegerType>(amount.getType())};
    if (checked) {
      mlir::Value out{arith::CmpIOp::create(builder_, loc,
          arith::CmpIPredicate::uge, amount,
          IntegerConstant(amountType, bits, loc))};
      merk::ThrowIfOp::create(builder_, loc, out,
          DomainError(module, what, "second operand is out of range"));
    }
    if (amountType.getWidth() > bits) {
      amount = arith::TruncIOp::create(builder_, loc, type, amount);
    } else if (amountType.getWidth() < bits) {
      amount = arith::ExtUIOp::create(builder_, loc, type, amount);
    }
    if (left) {
      return mlir::Value{arith::ShLIOp::create(builder_, loc, in[0], amount)};
    }
    return isUnsigned
        ? mlir::Value{arith::ShRUIOp::create(builder_, loc, in[0], amount)}
        : mlir::Value{arith::ShRSIOp::create(builder_, loc, in[0], amount)};
  }};
  if (in.size() == 2) {
    if (name == "+" || name == "plus") {
      result = arith::AddIOp::create(builder_, loc, in[0], in[1]);
    } else if (name == "-" || name == "minus") {
      result = arith::SubIOp::create(builder_, loc, in[0], in[1]);
    } else if (name == "*" || name == "times") {
      result = arith::MulIOp::create(builder_, loc, in[0], in[1]);
    } else if (name == "//" || name == "/") {
      result = divide("\\'//\\'", false, false);
    } else if (name == "rem") {
      result = divide("rem", false, true);
    } else if (name == "div") {
      result = divide("\\'//\\'", true, false);
    } else if (name == "mod") {
      result = divide("\\'//\\'", true, true);
    } else if (name == "unchecked_quotient") {
      result = isUnsigned
          ? mlir::Value{arith::DivUIOp::create(builder_, loc, in[0], in[1])}
          : mlir::Value{arith::DivSIOp::create(builder_, loc, in[0], in[1])};
    } else if (name == "unchecked_rem") {
      result = isUnsigned
          ? mlir::Value{arith::RemUIOp::create(builder_, loc, in[0], in[1])}
          : mlir::Value{arith::RemSIOp::create(builder_, loc, in[0], in[1])};
    } else if (name == "<<") {
      result = shift(true, true, "(<<)");
    } else if (name == ">>") {
      result = shift(false, true, "(>>)");
    } else if (name == "unchecked_left_shift") {
      result = shift(true, false, "");
    } else if (name == "unchecked_right_shift") {
      result = shift(false, false, "");
    } else if (name == "/\\") {
      result = arith::AndIOp::create(builder_, loc, in[0], in[1]);
    } else if (name == "\\/") {
      result = arith::OrIOp::create(builder_, loc, in[0], in[1]);
    } else if (name == "xor") {
      result = arith::XOrIOp::create(builder_, loc, in[0], in[1]);
    } else if (name == "min") {
      result = isUnsigned
          ? mlir::Value{arith::MinUIOp::create(builder_, loc, in[0], in[1])}
          : mlir::Value{arith::MinSIOp::create(builder_, loc, in[0], in[1])};
    } else if (name == "max") {
      result = isUnsigned
          ? mlir::Value{arith::MaxUIOp::create(builder_, loc, in[0], in[1])}
          : mlir::Value{arith::MaxSIOp::create(builder_, loc, in[0], in[1])};
    } else {
      return other();
    }
  } else if (in.size() == 1 && (name == "-" || name == "+")) {
    result = name == "+"
        ? in[0]
        : mlir::Value{arith::SubIOp::create(builder_, loc, constant(0), in[0])};
  } else if (in.size() == 1 && !isUnsigned &&
      (name == "unchecked_abs" || name == "nabs")) {
    // nabs is minus the absolute value, which cannot overflow.
    mlir::Value negative{arith::CmpIOp::create(
        builder_, loc, arith::CmpIPredicate::slt, in[0], constant(0))};
    mlir::Value negated{arith::SubIOp::create(builder_, loc, constant(0), in[0])};
    result = name == "nabs" ? select(negative, in[0], negated)
                            : select(negative, negated, in[0]);
  } else if (in.size() == 1 && name == "\\") {
    result = arith::XOrIOp::create(builder_, loc, in[0], constant(-1));
  } else if (in.size() == 1 && name == "abs" && !isUnsigned) {
    // mmc: "int.abs: abs(min_int) would overflow".
    mlir::Value least{arith::ConstantOp::create(builder_, loc,
        builder_.getIntegerAttr(type, llvm::APInt::getSignedMinValue(bits)))};
    mlir::Value overflows{arith::CmpIOp::create(
        builder_, loc, arith::CmpIPredicate::eq, in[0], least)};
    merk::ThrowIfOp::create(builder_, loc, overflows,
        ("Software Error: " + module + ".abs: abs(min_" + module +
            ") would overflow")
            .str());
    mlir::Value negative{arith::CmpIOp::create(
        builder_, loc, arith::CmpIPredicate::slt, in[0], constant(0))};
    result = select(negative,
        arith::SubIOp::create(builder_, loc, constant(0), in[0]), in[0]);
  } else {
    return other();
  }
  values_[output] = result;
  return true;
}

mlir::Value MerkGenerator::Slot(mlir::Type type, mlir::Location loc) {
  auto proc{builder_.getInsertionBlock()->getParent()
                ->getParentOfType<merk::ProcOp>()};
  mlir::OpBuilder::InsertionGuard guard{builder_};
  builder_.setInsertionPointToStart(&proc.getBody().front());
  unsigned addressSpace{target_.allocaAddressSpace()};
  mlir::Value one{mlir::LLVM::ConstantOp::create(
      builder_, loc, builder_.getI64Type(), builder_.getI64IntegerAttr(1))};
  mlir::Value slot{mlir::LLVM::AllocaOp::create(builder_, loc,
      mlir::LLVM::LLVMPointerType::get(&context_, addressSpace), type, one)};
  if (addressSpace != 0) {
    // On a GPU the stack is an address space of its own.
    slot = mlir::LLVM::AddrSpaceCastOp::create(
        builder_, loc, mlir::LLVM::LLVMPointerType::get(&context_), slot);
  }
  return slot;
}

// Only the scalars, integers and floats, can cross into a target region
// so far.
static bool Mappable(mlir::Type type) {
  return mlir::isa<mlir::IntegerType, mlir::FloatType>(type);
}

void MerkGenerator::Omp(const Body &goal, const ProcedureBody &body) {
  const parser::OmpDirective &directive{*goal.directive};
  parser::CharBlock at{Where(goal.tokens)};
  if (directive.id != llvm::omp::Directive::OMPD_target) {
    NotYet(at,
        "the " + std::string{parser::EnumToString(directive.id)} +
            " directive");
    return;
  }
  if (!directive.clauses.empty()) {
    NotYet(at, "clauses of the target directive");
    return;
  }
  if (builder_.getInsertionBlock()->getParent()
          ->getParentOfType<mlir::omp::TargetOp>()) {
    NotYet(at, "target regions in target regions");
    return;
  }
  mlir::Location loc{Location(goal.tokens)};
  Values before{values_};
  // The goal, in a target region whose map entries are added after it:
  // they are the values it uses from outside and those it binds.
  // Without host_eval its kernel is generic, not SPMD.
  mlir::omp::TargetExtOperands operands;
  operands.kernelType = mlir::omp::TargetExecModeAttr::get(
      &context_, mlir::omp::TargetExecMode::generic);
  auto targetOp{mlir::omp::TargetOp::create(builder_, loc, operands)};
  mlir::Region &region{targetOp.getRegion()};
  mlir::Block *block{&region.emplaceBlock()};
  {
    mlir::OpBuilder::InsertionGuard guard{builder_};
    builder_.setInsertionPointToStart(block);
    Goal(*goal.goals.front(), body);
  }
  if (failed_) {
    values_ = before;
    return;
  }
  builder_.setInsertionPoint(targetOp);
  llvm::SetVector<mlir::Value> inputs;
  mlir::getUsedValuesDefinedAbove(region, inputs);
  // What it binds that is used after it: those of its nonlocals it binds.
  llvm::SmallVector<BodyVar> outputs;
  for (BodyVar var : BoundSince(values_, before)) {
    if (values_.lookup(var) && llvm::is_contained(goal.nonlocals, var)) {
      outputs.push_back(var);
    }
  }
  // The variable whose value an input is, if it is one's.
  auto varOf{[&](mlir::Value input) -> std::optional<BodyVar> {
    for (const auto &[var, value] : before) {
      if (value == input) {
        return var;
      }
    }
    return std::nullopt;
  }};
  for (mlir::Value input : inputs) {
    if (!Mappable(input.getType())) {
      std::optional<BodyVar> var{varOf(input)};
      NotYet(at,
          var ? "values of type " +
                  semantics::TypeToString(*body.vars[*var].type) +
                  " in target regions"
              : std::string{"values that are not scalars in target regions"});
      values_ = before;
      return;
    }
  }
  for (BodyVar var : outputs) {
    if (!Mappable(values_.lookup(var).getType())) {
      NotYet(at,
          "values of type " + semantics::TypeToString(*body.vars[var].type) +
              " in target regions");
      values_ = before;
      return;
    }
  }
  // As Flang maps a scalar: an input is copied in from its slot, an output
  // is stored in its slot, from which it is read after the region.
  auto ptr{mlir::LLVM::LLVMPointerType::get(&context_)};
  auto map{[&](mlir::Value slot, mlir::Type type, mlir::omp::ClauseMapFlags flags,
               mlir::omp::VariableCaptureKind capture, llvm::StringRef name) {
    return mlir::omp::MapInfoOp::create(builder_, loc, ptr, slot,
        mlir::TypeAttr::get(type),
        builder_.getAttr<mlir::omp::ClauseMapFlagsAttr>(flags),
        builder_.getAttr<mlir::omp::VariableCaptureKindAttr>(capture),
        /*varPtrPtr=*/mlir::Value{}, /*varPtrPtrType=*/mlir::TypeAttr{},
        /*members=*/mlir::ValueRange{}, /*membersIndex=*/mlir::ArrayAttr{},
        /*bounds=*/mlir::ValueRange{}, /*mapperId=*/mlir::FlatSymbolRefAttr{},
        builder_.getStringAttr(name), builder_.getBoolAttr(false));
  }};
  llvm::SmallVector<mlir::Value> maps;
  for (mlir::Value input : inputs) {
    mlir::Value slot{Slot(input.getType(), loc)};
    mlir::LLVM::StoreOp::create(builder_, loc, input, slot);
    // Named after the variable whose value it is, if it is one's.
    std::optional<BodyVar> var{varOf(input)};
    llvm::StringRef name{var ? llvm::StringRef{body.vars[*var].name} : ""};
    maps.push_back(map(slot, input.getType(),
        mlir::omp::ClauseMapFlags::to, mlir::omp::VariableCaptureKind::ByCopy,
        name));
  }
  llvm::SmallVector<mlir::Value> outputSlots;
  llvm::SmallVector<mlir::Type> outputTypes;
  for (BodyVar var : outputs) {
    mlir::Type type{values_.lookup(var).getType()};
    outputTypes.push_back(type);
    outputSlots.push_back(Slot(type, loc));
    maps.push_back(map(outputSlots.back(), type,
        mlir::omp::ClauseMapFlags::from, mlir::omp::VariableCaptureKind::ByRef,
        body.vars[var].name));
  }
  targetOp.getMapVarsMutable().assign(maps);
  // The region reads its inputs from their slots and stores its outputs;
  // an output may be an input's value, as in Y = X.
  builder_.setInsertionPointToStart(block);
  llvm::DenseMap<mlir::Value, mlir::Value> loaded;
  for (mlir::Value input : inputs) {
    mlir::Value value{mlir::LLVM::LoadOp::create(
        builder_, loc, input.getType(), block->addArgument(ptr, loc))};
    mlir::replaceAllUsesInRegionWith(input, value, targetOp.getRegion());
    loaded[input] = value;
  }
  llvm::SmallVector<mlir::Value> outputArgs;
  for (size_t i = 0; i < outputs.size(); ++i) {
    outputArgs.push_back(block->addArgument(ptr, loc));
  }
  // Where the goal ended: the last block of the region.
  mlir::Block *end{&targetOp.getRegion().back()};
  builder_.setInsertionPointToEnd(end);
  if (!Terminated()) {
    for (auto [var, arg] : llvm::zip(outputs, outputArgs)) {
      mlir::Value value{values_.lookup(var)};
      mlir::LLVM::StoreOp::create(
          builder_, loc, loaded.lookup(value) ? loaded.lookup(value) : value, arg);
    }
    mlir::omp::TerminatorOp::create(builder_, loc);
  }
  builder_.setInsertionPointAfter(targetOp);
  values_ = before;
  for (auto [var, type, slot] : llvm::zip(outputs, outputTypes, outputSlots)) {
    values_[var] = mlir::LLVM::LoadOp::create(builder_, loc, type, slot);
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
  if (Terminated()) {
    return; // it cannot succeed
  }
  llvm::SmallVector<mlir::Value> results;
  for (BodyVar var : outputs) {
    results.push_back(ValueOf(var));
  }
  merk::SucceedOp::create(builder_, Location(at), results);
}

mlir::OwningOpRef<mlir::ModuleOp> MerkGenerator::Generate() {
  context_.loadDialect<merk::MerkDialect, mlir::arith::ArithDialect>();
  mlir::OwningOpRef<mlir::ModuleOp> owned{
      mlir::ModuleOp::create(builder_.getUnknownLoc(), moduleName_)};
  module_ = *owned;
  target_.Record(module_);
  // A procedure with a foreign_proc for C runs its C code instead of its
  // clauses.
  for (const Procedure &procedure : modes_.procedures()) {
    if (const parser::ForeignProcPragma *pragma{
            modes_.ForeignProcOf(procedure)}) {
      GenerateForeign(procedure, *pragma);
    }
  }
  for (const ProcedureBody &body : determinism_.bodies()) {
    if ((body.procedure->symbol->module == moduleName_ || body.method) &&
        (body.method || !modes_.ForeignProcOf(*body.procedure))) {
      GenerateBody(body);
    }
  }
  if (!foreignCode_.empty()) {
    // The module's own foreign_decl and foreign_code for C come first.
    std::string code{kForeignPrelude};
    for (const parser::Item &item : tree_.items) {
      const parser::ForeignCode *foreign{nullptr};
      if (const auto *decl{std::get_if<const parser::ForeignDeclPragma *>(&item)};
          decl && (*decl)->language == parser::ForeignLanguage::C) {
        foreign = (*decl)->code;
      } else if (const auto *c{
                     std::get_if<const parser::ForeignCodePragma *>(&item)};
          c && (*c)->language == parser::ForeignLanguage::C) {
        foreign = (*c)->code;
      }
      if (!foreign) {
        continue;
      }
      if (foreign->includeFile) {
        code += "#include \"" + foreign->text.str() + "\"\n";
      } else {
        code += LineDirective(foreign->tokens) + foreign->text.str() + "\n";
      }
    }
    code += foreignCode_;
    module_->setAttr(kForeignCodeAttrName, builder_.getStringAttr(code));
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
    llvm::StringRef module, const parser::ParseTree &tree,
    const parser::TokenSequence &tokens, const parser::AllSources &sources,
    const semantics::TypeChecking &types, semantics::ModeChecking &modes,
    const semantics::DeterminismChecking &determinism,
    const semantics::PolymorphismAnalysis &polymorphism,
    const TargetInfo &target, parser::Messages &messages) {
  return MerkGenerator{context, module, tree, tokens, sources, types, modes,
      determinism,
      polymorphism, target, messages}
      .Generate();
}

} // namespace merkurius::codegen
