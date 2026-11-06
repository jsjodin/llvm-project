//===-- lib/CodeGen/MLIRGen.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "merkurius/CodeGen/MLIRGen.h"
#include "Decomposer.h"
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
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"
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
// conventions"): a tuple is an MR_Tuple, a type without its own is an MR_Word,
// and an array is an MR_ArrayPtr, as mmc passes it.
static std::string CType(const semantics::Type &type) {
  if (type.kind == semantics::TypeKind::Variable) {
    return "MR_Word";
  }
  if (type.kind == semantics::TypeKind::Tuple) {
    return "MR_Tuple";
  }
  llvm::StringRef name{type.constructor->name};
  if (!type.constructor->module.empty()) {
    return type.constructor->module == "array" && name == "array"
        ? "MR_ArrayPtr"
        : "MR_Word";
  }
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
typedef void *MR_Tuple;
typedef void *MR_ArrayPtr;
typedef int MR_bool;
#define MR_TRUE 1
#define MR_FALSE 0
static inline MR_Float MR_word_to_float(MR_Word word) {
  MR_Float number;
  __builtin_memcpy(&number, &word, sizeof number);
  return number;
}
static inline MR_Word MR_float_to_word(MR_Float number) {
  MR_Word word;
  __builtin_memcpy(&word, &number, sizeof word);
  return word;
}
#define MR_word_to_int64(word) ((int64_t)(word))
#define MR_word_to_uint64(word) ((uint64_t)(word))
#define MR_int64_to_word(value) ((MR_Word)(value))
#define MR_uint64_to_word(value) ((MR_Word)(value))
)"};

namespace {

// A Mercury type as code generation represents it: an MLIR type, or none
// for a type with no values to pass, such as io.state.
struct Represented {
  mlir::Type type;
  bool dummy{false};
};

// A reduction clause's operator, as Flang's ReductionProcessor identifies
// the intrinsic ones (flang/lib/Lower/Support/ReductionProcessor.cpp, see
// THIRD_PARTY.md).
enum class ReductionOp { Add, Multiply, Max, Min, And, Or, Xor };

// A value a parallel loop carries that a reduction clause names.
struct LoopReduction {
  const semantics::TranslatedLoop::Threaded *value;
  ReductionOp op;
  mlir::Type type;
  unsigned output{0}; // of the body procedure's results
  mlir::Value slot{};
};

// What OmpLoop found out about a parallel loop, for generating it.
struct ParallelLoop {
  llvm::ArrayRef<const Body *> goals; // the bounds' goals, the trip test, the call
  const Body *trip;
  const Body *call; // of the loop procedure
  const semantics::TranslatedLoop *translated;
  const ProcedureBody &loopBody;
  const Body *bodyCall; // the loop procedure's, of the body procedure
  const Procedure &bodyProcedure;
  llvm::ArrayRef<LoopReduction> reductions;
  bool teams;
  bool distribute;
  bool offloaded;
  parser::CharBlock at;
};

class MerkGenerator {
public:
  MerkGenerator(mlir::MLIRContext &context, llvm::StringRef module,
      const parser::ParseTree &tree, const parser::TokenSequence &tokens,
      const parser::AllSources &sources,
      const semantics::NameResolution &resolution,
      semantics::TypeChecking &types,
      semantics::ModeChecking &modes,
      const semantics::DeterminismChecking &determinism,
      const semantics::PolymorphismAnalysis &polymorphism,
      const semantics::LoopTranslation &loops, const TargetInfo &target,
      parser::Messages &messages, bool stubUnimplemented)
      : context_{context}, builder_{&context}, moduleName_{module},
        tree_{tree}, tokens_{tokens}, sources_{sources},
        resolution_{resolution}, types_{types},
        modes_{modes}, determinism_{determinism}, loops_{loops},
        target_{target}, messages_{messages},
        stubUnimplemented_{stubUnimplemented} {
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
  // the procedure is then left out, or with -fstub-unimplemented, in a
  // procedure's body, made a stub that reports it at run time.
  void NotYet(parser::CharBlock at, llvm::StringRef what) {
    if (!failed_) {
      if (stubUnimplemented_ && inBody_) {
        notYet_ = "code generation for " + what.str();
      } else {
        messages_.Say(at, "code generation for %s"_todo_en_US, what.str());
      }
    }
    failed_ = true;
  }

  std::optional<Represented> Represent(const semantics::Type *);
  // A type defined by constructors, or a tuple type, as !merk.term, with
  // the module's merk.type_def of it, made the first time; none for a type
  // with user-defined equality or a supertype, or with arguments not
  // represented. Each instance of a type has its own, with one layout among
  // them.
  std::optional<Represented> RepresentTerm(const semantics::Type *);
  // An existentially typed constructor's cell's hidden fields' types
  // (Merk_ConstructorAttr), and where the type_info of each of its
  // existentially quantified type variables is, by their order: its hidden
  // field, and the word of it if that is a typeclass_info, or else 0.
  struct HiddenFields {
    llvm::SmallVector<mlir::Type> types;
    llvm::SmallVector<std::pair<unsigned, unsigned>> places;
  };
  // None for another constructor; null if a variable's type_info is only
  // inside a constraint's argument's, which is not handled yet.
  std::optional<HiddenFields> HiddenFieldsOf(const semantics::ConstructorTypes &);
  // A constructor's arguments' types in `type`, an existentially quantified
  // type variable numbered from ExistentialBase(type) by its order, so that it
  // is none of the type's.
  llvm::SmallVector<const semantics::Type *> ArgumentsOf(
      const semantics::Type *type, const semantics::ConstructorTypes &);
  // The first such number: kCellExistential, or past the type's variables, as
  // an argument's type may have its own constructor's.
  unsigned ExistentialBase(const semantics::Type &);
  // The types of a constructor's cell's fields in `type`, its hidden fields
  // first, which are `hidden`; none if one is not represented.
  std::optional<llvm::SmallVector<mlir::Type>> CellFields(const semantics::Type *,
      const semantics::ConstructorTypes &, HiddenFields &hidden);
  // The constructor of `type` with that name and arity.
  const semantics::ConstructorTypes *ConstructorOf(
      const semantics::Type *, llvm::StringRef name, unsigned arity);
  // An existentially quantified type variable's type_info at `place` among a
  // deconstruction's fields.
  mlir::Value HiddenTypeInfo(mlir::ValueRange fields,
      std::pair<unsigned, unsigned> place, mlir::Location);
  // Whether values of a type can be compared yet: integers, floats, strings,
  // words, arrays of those that can be, and terms whose constructors'
  // arguments can be, at any depth.
  bool Comparable(mlir::Type, llvm::StringSet<> &seen);
  // The semidet procedure that unifies two terms of a type that is
  // Comparable, made the first time: it switches on the first, and tests
  // that the second has the same constructor and equal arguments.
  merk::ProcOp UnifyProc(merk::TermType);
  merk::ProcOp UnifyArrayProc();
  mlir::Value ArrayElement(mlir::Value array, mlir::Value index, mlir::Location);
  // The array with that element set, which may be the same array.
  mlir::Value ArrayWithElement(
      mlir::Value array, mlir::Value index, mlir::Value element, mlir::Location);
  mlir::Value ArraySize(mlir::Value array, mlir::Location);
  mlir::Value InArrayBounds(
      mlir::Value array, mlir::Value index, mlir::Location);
  mlir::Value CompareInts(mlir::arith::CmpIPredicate, mlir::Value, mlir::Value,
      mlir::Location);
  // Binds each of `out` to the value at its place. Returns true: the goal is
  // generated, as the builtins' callers ask.
  bool GiveOutputs(
      llvm::ArrayRef<BodyVar> out, std::initializer_list<mlir::Value> values);
  // Reports a call of a library module not compiled yet. Returns true: the goal
  // has been dealt with.
  bool LibraryCallNotYet(const Body &, const semantics::Symbol &);
  // Whether a call of an integer or float module that is not built in has been
  // dealt with, which it has only if it was reported.
  bool NotBuiltIn(const Body &, const semantics::Symbol &);
  mlir::Value IntegerDivide(mlir::Value a, mlir::Value b, llvm::StringRef module,
      llvm::StringRef what, bool floor, bool remainder, mlir::Location);
  mlir::Value IntegerShift(mlir::Value value, mlir::Value amount,
      llvm::StringRef module, llvm::StringRef what, bool left, bool checked,
      mlir::Location);
  // Comparison in the standard ordering (reference manual, "The standard
  // ordering"), as unification: compare.T for a term type, made the first
  // time, one procedure for arrays, and the value of a comparison, null after
  // an error. `generic` is whether the type is one that a type variable
  // stands for in its context: there, as through a type_info, a NaN cannot be
  // compared.
  merk::ProcOp CompareProc(merk::TermType);
  merk::ProcOp CompareArrayProc();
  mlir::Value Compare(mlir::Value x, mlir::Value y,
      const semantics::Type *, bool generic, mlir::Location);
  std::optional<merk::TermType> ComparisonType();
  mlir::Value ComparisonValue(llvm::StringRef functor, mlir::Location);
  mlir::Value IsComparison(
      mlir::Value result, llvm::StringRef functor, mlir::Location);
  mlir::Value ThreeWay(mlir::Value less, mlir::Value greater, mlir::Location);
  // The comparison `functor` if `condition` holds, or else what `otherwise`
  // makes, in the else part.
  mlir::Value ComparisonIf(mlir::Value condition, llvm::StringRef functor,
      mlir::Location, llvm::function_ref<mlir::Value()> otherwise);
  // Succeeds at once with `value`, if `condition` holds.
  void SucceedIf(mlir::Value condition, mlir::Value value, mlir::Location);
  // A term's constructor's place among `cases`, the cases of a switch on it.
  mlir::Value ConstructorIndex(
      mlir::Value term, mlir::ArrayAttr cases, mlir::Location);
  // A call of builtin's compare, its operators, ordering or unify.
  bool CompareBuiltin(const Body &, const Procedure &,
      llvm::ArrayRef<BodyVar> arguments, const ProcedureBody &);
  // A call of io's write or print, or their _line forms, to the standard
  // output.
  // A type's type variables, in order of first appearance.
  llvm::SmallVector<unsigned> VariablesOf(const semantics::Type &);
  // Makes the merk.type_ctor of a type's constructor, with its unify entry,
  // the first time; false with an error if it cannot have one yet.
  bool EnsureDescriptor(const semantics::Type &, parser::CharBlock at);
  bool EnsureVoidDescriptor();
  void EmitDescriptor(llvm::StringRef constructor, llvm::StringRef module,
      llvm::StringRef name, llvm::StringRef kind, llvm::StringRef linkage,
      unsigned arity, mlir::Type term, merk::ProcOp unify, merk::ProcOp compare, mlir::ArrayAttr functors);
  // A descriptor's table of its constructors, for merk.type_ctor, and a field's
  // type in it; null after an error.
  mlir::ArrayAttr FunctorsAttr(const semantics::Type *canonical, parser::CharBlock);
  mlir::Attribute FieldTypeAttr(const semantics::Type &, parser::CharBlock,
      const HiddenFields *hidden = nullptr);
  // A higher-order type's entry, which reports a unification or comparison of
  // closures and stops the program, as mmc's does.
  merk::ProcOp HigherOrderEntry(bool compare, llvm::StringRef constructor);
  merk::ProcOp DescriptorEntry(bool compare, llvm::StringRef constructor,
      const semantics::Type *canonical, mlir::Type represented, bool dummy,
      parser::CharBlock at);
  // The test that two values of a Comparable type are equal: arith's
  // comparison of integers or floats, or a call of the terms' unify
  // procedure.
  void Equal(mlir::Value, mlir::Value, const semantics::Type *, mlir::Location);
  // X = f(Y1, ..., Yn) of a term X: a construction, or a deconstruction,
  // with a test of X's constructor first if it can fail.
  void TermUnify(const Body &, const ProcedureBody &);
  bool IsOutput(const semantics::ArgMode &mode) {
    return modes_.ops().IsFree(mode.initial) &&
        !modes_.ops().IsFree(mode.final);
  }
  // The procedure's merk.proc, declared if it is not defined yet; null
  // after reporting the reason if it cannot be.
  merk::ProcOp ProcFor(const Procedure &, llvm::ArrayRef<const semantics::Type *>,
      parser::CharBlock at);
  // A procedure's body, or its word copy's (WordProcFor).
  void GenerateBody(const ProcedureBody &, bool words = false);
  // With -fstub-unimplemented, the body of a procedure that code generation
  // cannot make: it reports what is not implemented, and ends the program.
  void StubBody(merk::ProcOp, parser::CharBlock at);
  void GenerateUnchecked(const Procedure &);
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
  // A call of a predicate or function of the library's array module, which
  // code generation has built in as merk's array operations; false if it
  // is to be called instead.
  bool ArrayBuiltin(const Body &, const Procedure &,
      llvm::ArrayRef<BodyVar> arguments, const ProcedureBody &);
  // X = Y, or X = c, with both bound: a test.
  void Test(const Body &, mlir::Value, mlir::Value, const semantics::Type *);
  // A call in its procedure's own modes, each output a fresh value.
  // The type_infos and then the typeclass_infos a procedure takes, or none
  // with an error if it has what code generation does not handle yet.
  std::optional<llvm::SmallVector<mlir::Type>> HiddenInputs(
      const Procedure &, parser::CharBlock);
  // A typeclass_info, as polymorphism analysis says it is made.
  mlir::Value DictionaryValue(
      const semantics::Dictionary &, mlir::Location, parser::CharBlock);
  // A type class's methods' procedures, in the order of its instances'
  // method tables.
  llvm::SmallVector<const Procedure *> MethodSlots(const semantics::Symbol &);
  // The type_infos and typeclass_infos an existentially typed procedure gives
  // its caller, before its other outputs.
  llvm::SmallVector<mlir::Type> HiddenOutputs(const Procedure &);
  // What goals give are kept as values of variables that are not the body's,
  // so that they leave an if-then-else or a switch as a variable's value does:
  // a branch's own stay its own, as the types they describe do.
  static constexpr BodyVar kFirstGivenVar{1u << 30};
  mlir::Value Given(const void *node);
  void Give(const void *node, mlir::Value);
  // Whether what a clause gives can be made at the end of the body.
  bool CanMake(const semantics::TypeInfo &);
  bool CanMake(const semantics::Dictionary &);
  template <typename Range> bool CanMakeAll(const Range &nodes) {
    return llvm::all_of(nodes, [&](const auto *node) { return CanMake(*node); });
  }
  // The type_info a goal passes for a type that has types goals gave, which
  // has no procedure of its own to test or compare it.
  mlir::Value GivenTypeInfo(const Body &, const semantics::Type *, mlir::Location);
  // Add a call's hidden input to `inputs`; after an error, failed_ is set.
  void PassTypeInfo(const semantics::TypeInfo &,
      llvm::SmallVectorImpl<mlir::Value> &inputs, mlir::Location,
      parser::CharBlock at);
  void PassDictionary(const semantics::Dictionary &,
      llvm::SmallVectorImpl<mlir::Value> &inputs, mlir::Location,
      parser::CharBlock at);
  // The symbol of an instance's method table, declaring it if it is new.
  std::string InstanceSymbol(const semantics::InstanceHead &);
  // A procedure with no body yet, its inputs `hidden` and then its modes' and
  // types'; none with an error for a type code generation cannot represent.
  merk::ProcOp NewProc(llvm::StringRef name, parser::CharBlock at,
      llvm::ArrayRef<mlir::Type> hidden, llvm::ArrayRef<semantics::ArgMode> modes,
      llvm::ArrayRef<const semantics::Type *> types, merk::Determinism,
      llvm::ArrayRef<mlir::Type> hiddenOutputs = {});
  // An instance method's procedure, which takes the instance's hidden
  // arguments and then the method's.
  merk::ProcOp MethodProc(const ProcedureBody &);
  // The method tables of the module's instances, and the procedures in them,
  // which take a typeclass_info and the method's arguments as the type class
  // declares them, and call the instance's method.
  void GenerateInstances();
  // The body of an instance's method in this mode, if this module has it.
  const ProcedureBody *MethodBody(
      const semantics::InstanceHead &, const Procedure &);
  void GenerateDispatch(const semantics::InstanceHead &, const Procedure &,
      const ProcedureBody &, llvm::StringRef name);
  // `freeIsVoid`: a type variable with no type_info is one that nothing
  // determines, which has the type void.
  mlir::Value TypeInfoOfType(const semantics::Type *, mlir::Location,
      parser::CharBlock, bool freeIsVoid = false);
  mlir::Value TypeInfoValue(
      const semantics::TypeInfo &, mlir::Location, parser::CharBlock);
  // A value at another instance of its type, such as list(int) at list(T):
  // one word whichever, so they meet at the word.
  mlir::Value Coerce(mlir::Value, mlir::Type, mlir::Location);
  void Called(const Body &, const Procedure &, llvm::ArrayRef<BodyVar>,
      const ProcedureBody &);
  // Closures (design document, Lowering, "Closures"). X = Lambda: the lambda
  // expression's procedure, which takes the values it captures and then its
  // inputs and gives a word for each output, and a closure of its entry.
  void LambdaClosure(const Body &, const ProcedureBody &);
  // X = p(Y1, ..., Yn): a closure of a procedure with its first n arguments.
  void NameClosure(const Body &, const ProcedureBody &);
  // X(Y1, ...), or Z = X(Y1, ...).
  void HigherOrderCall(const Body &, const ProcedureBody &);
  // For each of a closure's arguments, after those it captured: whether it is
  // an output, and whether the procedure its entry calls leaves it out, as
  // of a type with no values.
  struct ClosureArgument {
    bool output;
    bool dummy;
  };
  // The entry of the closures of `proc` that capture its first `captured`
  // inputs, made the first time: it takes the closure and a word for each
  // input, calls the procedure with the captured values and those, and gives
  // a word for each output.
  merk::ProcOp ClosureEntry(merk::ProcOp proc, unsigned captured,
      llvm::ArrayRef<ClosureArgument>, parser::CharBlock at);
  // The entry of a closure of a procedure code generation has built in, with
  // the closure goal's `captured` values: a body of its own, made the first
  // time; null if the procedure is not built in.
  merk::ProcOp BuiltinEntry(const Body &, const Procedure &,
      llvm::ArrayRef<const semantics::Type *> types, unsigned captured,
      bool &failed);
  // A word copy of a procedure: it gives every output as a word, one of a
  // type with no values as the word 0, so that the procedures that give words,
  // closures' entries, lambda expressions' procedures and other copies, call
  // it in tail position as a tail call whatever the outputs' types (design
  // document, Lowering, "Closures"). Declared the first time, and its body
  // made after the module's procedures'. The body of a procedure of this
  // module that can have one: not a foreign_proc, nor existentially typed.
  merk::ProcOp WordProcFor(const ProcedureBody &);
  const ProcedureBody *CopyableBody(const Procedure &);
  // In a body that gives words: the words of its outputs, `outputs`.
  llvm::SmallVector<mlir::Value> WordResults(
      llvm::ArrayRef<BodyVar> outputs, const ProcedureBody &, mlir::Location);
  mlir::Value ZeroWord(mlir::Location loc) {
    return merk::ToWordOp::create(builder_, loc, merk::WordType::get(&context_),
        IntegerConstant(builder_.getI64Type(), 0, loc));
  }
  // Sets an output's value from the word a call gave: in a body that gives
  // words, the word of one of a type with no values is kept, so that it can
  // be the body's own.
  void SetFromWord(BodyVar, mlir::Value word, const ProcedureBody &,
      mlir::Location);
  void IfThenElse(const Body &, const ProcedureBody &);
  void Disjunction(const Body &, const ProcedureBody &);
  void DisjunctionFrom(const Body &, std::size_t next, const ProcedureBody &);
  void Ite(semantics::TokenRange, const Body &condition, const Body *thenPart,
      llvm::function_ref<void()> otherwise, const ProcedureBody &);
  void Switch(const Body &, const ProcedureBody &);
  void Not(const Body &, const ProcedureBody &);
  // A goal with an OpenMP directive (design document, "Target regions").
  void Omp(const Body &, const ProcedureBody &);
  // A loop with a loop-associated directive, so far parallel do and target
  // teams distribute parallel do, as omp.parallel, omp.wsloop and
  // omp.loop_nest, with omp.target, omp.teams and omp.distribute for the
  // latter, whose body calls the loop's body procedure.
  void OmpLoop(const Body &, const ProcedureBody &);
  // The leaves of a loop's directive around its omp.loop_nest, in the current
  // block, which is in the target region if it is offloaded.
  void GenerateParallelLoop(const ParallelLoop &, const ProcedureBody &,
      mlir::ValueRange hostEval, mlir::Location);
  template <typename Operands>
  void AddReductions(Operands &, llvm::ArrayRef<LoopReduction>,
      mlir::ValueRange vars, mlir::Location);
  void LoadReductions(
      llvm::ArrayRef<LoopReduction>, const Body &call, mlir::Location);
  // A slot of the procedure that a target region updates, mapped tofrom
  // by reference, as Flang maps a reduction's variable: the region uses
  // its pointer, which becomes the region's argument.
  struct UpdatedSlot {
    mlir::Value slot;
    mlir::Type type;
    llvm::StringRef name;
  };
  static const UpdatedSlot *FindUpdatedSlot(
      llvm::ArrayRef<UpdatedSlot>, mlir::Value slot);
  mlir::Value MapSlot(mlir::Value slot, mlir::Type, mlir::omp::ClauseMapFlags,
      mlir::omp::VariableCaptureKind, llvm::StringRef name, mlir::Location);
  mlir::Value ArrayPointer(mlir::Value array, mlir::Location);
  // A target region around what `generate` builds, for the goal, which
  // maps the values it uses and those it binds, and `updated`. Its kernel
  // is `mode` (KernelOf); `hostEval` are the values computed on the host
  // for it, which `generate` is given as the region's arguments.
  void Target(const Body &, const ProcedureBody &,
      llvm::function_ref<void(mlir::ValueRange hostEval)> generate,
      bool combined, mlir::omp::TargetExecMode mode,
      mlir::ValueRange hostEval = {}, llvm::ArrayRef<UpdatedSlot> slots = {});
  // The module's omp.declare_reduction for the operator on `type`, an
  // integer or a float, by value, made the first time, as Flang's ReductionProcessor makes
  // and names them.
  mlir::omp::DeclareReductionOp DeclareReduction(
      ReductionOp, mlir::Type type, mlir::Location);
  // The directive's leaves with their clauses; none after reporting a
  // clause that none of them allows.
  mlir::Value MapArray(mlir::Value array, mlir::omp::ClauseMapFlags,
      llvm::StringRef name, mlir::Location);
  void TargetData(const Body &, const ProcedureBody &, const omp::Leaf &);
  std::optional<omp::ConstructQueue> Decompose(const parser::OmpDirective &,
      parser::CharBlock at, std::optional<llvm::StringRef> counter = {});
  // Reports the leaf's clauses that are not generated yet: all but those
  // `handled` takes, of the leaf's directive, the clause and whether the
  // decomposition added it, and shared clauses the decomposition adds,
  // since a variable is shared already. Whether there were none.
  bool LeafClauses(const omp::Leaf &, parser::CharBlock at,
      llvm::function_ref<bool(llvm::omp::Directive, llvm::omp::Clause,
          bool synthesized)>
          handled = nullptr);
  // An error that leaves the procedure out.
  void Error(parser::CharBlock at, parser::MessageFixedText text,
      llvm::StringRef argument = {}) {
    if (!failed_) {
      messages_.Say(at, text, argument.str());
    }
    failed_ = true;
  }
  // A stack slot of the procedure for a value of `type`, as a generic
  // pointer, which omp.map.info maps.
  mlir::Value Slot(mlir::Type type, mlir::Location);
  // The variables bound in a branch, after `before`, in order.
  using Values = llvm::DenseMap<BodyVar, mlir::Value>;
  static std::optional<BodyVar> VarWithValue(const Values &,
      llvm::ArrayRef<BodyVar> preferred, mlir::Value input);
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
  const semantics::NameResolution &resolution_;
  semantics::TypeChecking &types_;
  semantics::ModeChecking &modes_;
  const semantics::DeterminismChecking &determinism_;
  const semantics::LoopTranslation &loops_;
  const TargetInfo &target_;
  parser::Messages &messages_;
  llvm::DenseMap<const ProcedureBody *, const semantics::ProcedurePolymorphism *>
      polymorphism_;
  mlir::ModuleOp module_;
  llvm::DenseMap<const Procedure *, merk::ProcOp> procs_;
  // The types of the module's merk.type_defs, by name, null for one that
  // cannot be represented; and their unify procedures.
  llvm::StringMap<const semantics::Type *> terms_;
  llvm::StringMap<merk::ProcOp> unifyProcs_;
  llvm::StringMap<merk::ProcOp> compareProcs_;
  merk::ProcOp compareArray_;
  // The C functions of the module's foreign_procs.
  std::string foreignCode_;
  // For the body being generated: each variable's value, and whether a
  // construct was not handled.
  llvm::DenseMap<BodyVar, mlir::Value> values_;
  merk::ProcOp unifyArray_;
  llvm::StringSet<> descriptors_;
  // Where the unification being generated is, for its errors.
  parser::CharBlock equalAt_;
  // The procedure's hidden type_infos, by type variable.
  llvm::DenseMap<unsigned, mlir::Value> typeInfos_;
  // Its hidden typeclass_infos, by constraint.
  llvm::DenseMap<unsigned, mlir::Value> dictionaries_;
  llvm::DenseMap<const semantics::MethodModes *, merk::ProcOp> methodProcs_;
  // The variables of what goals give, in the procedure being generated.
  llvm::DenseMap<const void *, BodyVar> givenVars_;
  const semantics::ProcedurePolymorphism *current_{nullptr};
  bool failed_{false};
  // Temporary (chunk 1 step 6, docs/INPROGRESS.md): -fstub-unimplemented,
  // for the library's procedures in modes code generation cannot make yet,
  // such as nondet ones; it goes when code generation makes them all.
  bool stubUnimplemented_{false};
  // Whether a procedure's body is being generated, and the first construct
  // of it not handled yet, with -fstub-unimplemented.
  bool inBody_{false};
  std::string notYet_;
  // The procedure being generated, whose lambda expressions' procedures are
  // named after it, and their number in the module.
  std::string procName_;
  unsigned lambdas_{0};
  // The procedure made for each lambda expression.
  llvm::DenseMap<const Body *, merk::ProcOp> lambdaProcs_;
  // In a body that gives words, its outputs, in order, those of a type with
  // no values among them; their words are values_ of the variables from
  // kDummyWordVar on.
  std::optional<llvm::SmallVector<BodyVar>> wordOutputs_;
  static constexpr BodyVar kDummyWordVar{1u << 29};
  static bool IsDummyWord(BodyVar var) {
    return var >= kDummyWordVar && var < kFirstGivenVar;
  }
  // The bodies of the module's procedures, the word copies made of them, and
  // those whose bodies are still to be made.
  llvm::DenseMap<const Procedure *, const ProcedureBody *> bodies_;
  llvm::DenseMap<const Procedure *, merk::ProcOp> wordProcs_;
  std::vector<const ProcedureBody *> pendingCopies_;
  llvm::StringMap<merk::ProcOp> entries_;
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
  // A type that a goal gave is a type variable to the code that has it.
  if (type->kind == semantics::TypeKind::Variable ||
      type->kind == semantics::TypeKind::Existential) {
    return Represented{merk::WordType::get(&context_)};
  }
  if (type->kind == semantics::TypeKind::Tuple) {
    return RepresentTerm(type);
  }
  if (type->kind == semantics::TypeKind::HigherOrder) {
    return Represented{merk::ClosureType::get(&context_)};
  }
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
  // An array of elements a word wide at most, each in a word (design
  // document, "Type representation").
  if (constructor.module == "array" && name == "array" &&
      type->arguments.size() == 1) {
    std::optional<Represented> element{Represent(type->arguments[0])};
    if (!element || element->dummy ||
        (!mlir::isa<merk::StringType, merk::ArrayType, merk::TermType,
             merk::WordType>(
             element->type) &&
            element->type.getIntOrFloatBitWidth() > target_.WordBits())) {
      return std::nullopt;
    }
    return Represented{merk::ArrayType::get(&context_, element->type)};
  }
  return RepresentTerm(type);
}

std::optional<Represented> MerkGenerator::RepresentTerm(
    const semantics::Type *type) {
  std::string name{semantics::TypeToString(*type)};
  auto term{merk::TermType::get(&context_, name)};
  if (auto it{terms_.find(name)}; it != terms_.end()) {
    return it->second ? std::make_optional(Represented{term}) : std::nullopt;
  }
  const parser::DuTypeDefn *definition{nullptr};
  if (type->kind == semantics::TypeKind::Named) {
    if (const auto *du{std::get_if<const parser::DuTypeDefn *>(
            &type->constructor->declaration)}) {
      definition = *du;
    }
  }
  const std::vector<semantics::ConstructorTypes> *constructors{
      modes_.ops().ConstructorsOf(type)};
  // A type too large to print has no name of its own.
  if (!constructors || name.front() == '<' ||
      (definition && (definition->attributes || definition->supertype))) {
    terms_[name] = nullptr;
    return std::nullopt;
  }
  // Recorded before its constructors' arguments, which may be of it.
  terms_[name] = type;
  llvm::SmallVector<mlir::Attribute> attributes;
  for (const semantics::ConstructorTypes &constructor : *constructors) {
    HiddenFields hidden;
    std::optional<llvm::SmallVector<mlir::Type>> fields{
        CellFields(type, constructor, hidden)};
    if (!fields) {
      terms_[name] = nullptr;
      return std::nullopt;
    }
    attributes.push_back(merk::ConstructorAttr::get(&context_,
        constructor.functor.name, *fields,
        static_cast<unsigned>(hidden.types.size())));
  }
  // At the module's start, after those made before it.
  mlir::OpBuilder::InsertionGuard guard{builder_};
  builder_.setInsertionPointToStart(module_.getBody());
  for (mlir::Operation &op : *module_.getBody()) {
    if (!mlir::isa<merk::TypeDefOp>(op)) {
      break;
    }
    builder_.setInsertionPointAfter(&op);
  }
  merk::TypeDefOp::create(builder_, builder_.getUnknownLoc(), name,
      builder_.getArrayAttr(attributes));
  return Represented{term};
}

static std::optional<merk::Determinism> MerkDeterminism(
    parser::Determinism determinism) {
  switch (determinism) {
  case parser::Determinism::Det:
    return merk::Determinism::Det;
  case parser::Determinism::CcMulti:
    return merk::Determinism::CcMulti;
  case parser::Determinism::CcNondet:
    return merk::Determinism::CcNondet;
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

// Which type variables of `count` the constraints have: those have their
// type_infos in the typeclass_infos.
static llvm::SmallVector<bool> ConstrainedVariables(
    unsigned count, llvm::ArrayRef<semantics::ClassConstraint> constraints) {
  llvm::SmallVector<bool> constrained(count, false);
  for (const semantics::ClassConstraint &constraint : constraints) {
    for (const semantics::Type *type : constraint.arguments) {
      semantics::ForEachVariable(*type, [&](unsigned var) {
        if (var < count) {
          constrained[var] = true;
        }
      });
    }
  }
  return constrained;
}

// Whether a type has a type that a goal gave, which only that goal's type_info
// describes.
static bool HasGivenType(const semantics::Type &type) {
  bool given{false};
  semantics::ForEachExistential(type, [&](const semantics::Type &) { given = true; });
  return given;
}

std::optional<llvm::SmallVector<mlir::Type>> MerkGenerator::HiddenInputs(
    const Procedure &procedure, parser::CharBlock at) {
  llvm::SmallVector<mlir::Type> hidden;
  const semantics::Signature *signature{types_.SignatureOf(*procedure.symbol)};
  if (!signature) {
    return hidden;
  }
  // As polymorphism analysis: each of the signature's universally quantified
  // type variables, whether or not an argument has it, as a loop's predicate's
  // can have one only its clause's other goals use.
  llvm::SmallVector<bool> constrained{
      ConstrainedVariables(signature->typeVars, signature->constraints)};
  for (unsigned var{0}; var < signature->typeVars - signature->existentials;
       ++var) {
    if (!constrained[var]) {
      hidden.push_back(merk::TypeInfoType::get(&context_));
    }
  }
  hidden.append(signature->constraints.size(),
      merk::TypeclassInfoType::get(&context_));
  return hidden;
}

llvm::SmallVector<mlir::Type> MerkGenerator::HiddenOutputs(
    const Procedure &procedure) {
  llvm::SmallVector<mlir::Type> hidden;
  const semantics::Signature *signature{types_.SignatureOf(*procedure.symbol)};
  if (!signature) {
    return hidden;
  }
  llvm::SmallVector<bool> constrained{ConstrainedVariables(
      signature->typeVars, signature->existentialConstraints)};
  for (unsigned var{signature->typeVars - signature->existentials};
       var < signature->typeVars; ++var) {
    if (!constrained[var]) {
      hidden.push_back(merk::TypeInfoType::get(&context_));
    }
  }
  hidden.append(signature->existentialConstraints.size(),
      merk::TypeclassInfoType::get(&context_));
  return hidden;
}

std::optional<MerkGenerator::HiddenFields> MerkGenerator::HiddenFieldsOf(
    const semantics::ConstructorTypes &constructor) {
  HiddenFields hidden;
  const semantics::Signature *signature{constructor.symbol
          ? types_.SignatureOf(*constructor.symbol)
          : nullptr};
  if (!signature || signature->existentials == 0) {
    return hidden;
  }
  // As polymorphism analysis passes them to a construction: the type_infos
  // of the variables the constraints do not have, then the typeclass_infos.
  unsigned universal{signature->typeVars - signature->existentials};
  llvm::ArrayRef<semantics::ClassConstraint> constraints{
      signature->existentialConstraints};
  llvm::SmallVector<bool> constrained{
      ConstrainedVariables(signature->typeVars, constraints)};
  hidden.places.resize(signature->existentials);
  for (unsigned var{universal}; var < signature->typeVars; ++var) {
    if (!constrained[var]) {
      hidden.places[var - universal] = {
          static_cast<unsigned>(hidden.types.size()), 0};
      hidden.types.push_back(merk::TypeInfoType::get(&context_));
    }
  }
  unsigned first{static_cast<unsigned>(hidden.types.size())};
  hidden.types.append(
      constraints.size(), merk::TypeclassInfoType::get(&context_));
  // A constrained variable's type_info is in the typeclass_info of the first
  // constraint that has it as an argument, after the method table and the
  // superclasses' typeclass_infos.
  for (unsigned var{universal}; var < signature->typeVars; ++var) {
    if (!constrained[var]) {
      continue;
    }
    std::optional<std::pair<unsigned, unsigned>> place;
    for (auto [i, constraint] : llvm::enumerate(constraints)) {
      unsigned supers{static_cast<unsigned>(
          types_.SuperclassesOf(*constraint.typeclass).size())};
      for (auto [j, argument] : llvm::enumerate(constraint.arguments)) {
        if (argument->kind == semantics::TypeKind::Variable &&
            argument->var == var) {
          place = {first + static_cast<unsigned>(i),
              supers + static_cast<unsigned>(j) + 1};
          break;
        }
      }
      if (place) {
        break;
      }
    }
    if (!place) {
      return std::nullopt;
    }
    hidden.places[var - universal] = *place;
  }
  return hidden;
}

// The least number of an existentially typed constructor's existentially
// quantified type variables, in ArgumentsOf's types: past a procedure's, and
// below polymorphism analysis's kExistentialVar on.
static constexpr unsigned kCellExistential{1u << 29};

unsigned MerkGenerator::ExistentialBase(const semantics::Type &type) {
  unsigned base{kCellExistential};
  semantics::ForEachVariable(
      type, [&](unsigned var) { base = std::max(base, var + 1); });
  return base;
}

llvm::SmallVector<const semantics::Type *> MerkGenerator::ArgumentsOf(
    const semantics::Type *type, const semantics::ConstructorTypes &constructor) {
  const semantics::Signature *signature{constructor.symbol
          ? types_.SignatureOf(*constructor.symbol)
          : nullptr};
  if (!signature || signature->existentials == 0) {
    return {constructor.arguments.begin(), constructor.arguments.end()};
  }
  unsigned universal{signature->typeVars - signature->existentials};
  unsigned base{ExistentialBase(*type)};
  semantics::TypeContext &context{types_.types()};
  llvm::SmallVector<const semantics::Type *> arguments;
  for (const semantics::Type *argument : signature->arguments.drop_back()) {
    arguments.push_back(context.Substitute(argument, [&](unsigned var) {
      if (var >= universal) {
        return context.Variable(base + var - universal);
      }
      return var < type->arguments.size() ? type->arguments[var]
                                          : context.Variable(var);
    }));
  }
  return arguments;
}

std::optional<llvm::SmallVector<mlir::Type>> MerkGenerator::CellFields(
    const semantics::Type *type, const semantics::ConstructorTypes &constructor,
    HiddenFields &hidden) {
  std::optional<HiddenFields> found{HiddenFieldsOf(constructor)};
  if (!found) {
    return std::nullopt;
  }
  hidden = std::move(*found);
  llvm::SmallVector<mlir::Type> fields{hidden.types};
  for (const semantics::Type *argument : ArgumentsOf(type, constructor)) {
    std::optional<Represented> represented{Represent(argument)};
    if (!represented || represented->dummy) {
      return std::nullopt;
    }
    fields.push_back(represented->type);
  }
  return fields;
}

const semantics::ConstructorTypes *MerkGenerator::ConstructorOf(
    const semantics::Type *type, llvm::StringRef name, unsigned arity) {
  const std::vector<semantics::ConstructorTypes> *constructors{
      modes_.ops().ConstructorsOf(type)};
  if (!constructors) {
    return nullptr;
  }
  for (const semantics::ConstructorTypes &constructor : *constructors) {
    if (constructor.functor.name == name && constructor.functor.arity == arity) {
      return &constructor;
    }
  }
  return nullptr;
}

mlir::Value MerkGenerator::HiddenTypeInfo(mlir::ValueRange fields,
    std::pair<unsigned, unsigned> place, mlir::Location loc) {
  mlir::Value field{fields[place.first]};
  if (place.second == 0) {
    return field;
  }
  return merk::TypeclassInfoWordOp::create(builder_, loc,
      merk::TypeInfoType::get(&context_), field, place.second);
}

mlir::Value MerkGenerator::Given(const void *node) {
  auto it{givenVars_.find(node)};
  return it == givenVars_.end() ? mlir::Value{} : values_.lookup(it->second);
}

void MerkGenerator::Give(const void *node, mlir::Value value) {
  auto [it, added]{givenVars_.try_emplace(
      node, kFirstGivenVar + static_cast<BodyVar>(givenVars_.size()))};
  values_[it->second] = value;
}

bool MerkGenerator::CanMake(const semantics::TypeInfo &info) {
  switch (info.kind) {
  case semantics::TypeInfoKind::Output:
    return static_cast<bool>(Given(&info));
  case semantics::TypeInfoKind::Dictionary:
    return CanMake(*info.dictionary);
  default:
    return llvm::all_of(info.arguments,
        [&](const semantics::TypeInfo *argument) { return CanMake(*argument); });
  }
}

bool MerkGenerator::CanMake(const semantics::Dictionary &dictionary) {
  switch (dictionary.kind) {
  case semantics::DictionaryKind::Output:
    return static_cast<bool>(Given(&dictionary));
  case semantics::DictionaryKind::Superclass:
    return CanMake(*dictionary.from);
  default:
    return CanMakeAll(dictionary.typeInfos) && CanMakeAll(dictionary.types) &&
        CanMakeAll(dictionary.premises) && CanMakeAll(dictionary.supers);
  }
}

llvm::SmallVector<const Procedure *> MerkGenerator::MethodSlots(
    const semantics::Symbol &typeclass) {
  llvm::SmallVector<const Procedure *> slots;
  const semantics::ModuleSymbols *symbols{typeclass.module == moduleName_
          ? &resolution_.module()
          : resolution_.Module(typeclass.module)};
  if (!symbols) {
    return slots;
  }
  for (const semantics::Symbol *symbol : symbols->symbols()) {
    if (symbol->owner == &typeclass) {
      llvm::ArrayRef<const Procedure *> procedures{modes_.ProceduresOf(*symbol)};
      slots.append(procedures.begin(), procedures.end());
    }
  }
  return slots;
}

std::string MerkGenerator::InstanceSymbol(
    const semantics::InstanceHead &head) {
  std::string text;
  llvm::raw_string_ostream os{text};
  os << "merk.instance." << head.typeclass->module << '.'
     << head.typeclass->name << '(';
  llvm::ListSeparator separator{","};
  for (const semantics::Type *type : head.types) {
    os << separator
       << semantics::TypeToString(*type,
              [](unsigned var) { return "V" + std::to_string(var); });
  }
  os << ')';
  if (!module_.lookupSymbol<merk::InstanceOp>(text)) {
    mlir::OpBuilder::InsertionGuard guard{builder_};
    builder_.setInsertionPointToStart(module_.getBody());
    merk::InstanceOp::create(builder_, builder_.getUnknownLoc(),
        builder_.getStringAttr(text), /*sym_visibility=*/mlir::StringAttr{},
        /*methods=*/mlir::ArrayAttr{});
  }
  return text;
}

mlir::Value MerkGenerator::Coerce(
    mlir::Value value, mlir::Type type, mlir::Location loc) {
  if (value.getType() == type) {
    return value;
  }
  auto word{merk::WordType::get(&context_)};
  // A value back from its word is the word's, and a word back to the type of
  // the value it came from is the value, in the same region.
  if (mlir::Operation *from{value.getDefiningOp()};
      from && from->getParentRegion() == builder_.getInsertionBlock()->getParent()) {
    if (auto fromWord{mlir::dyn_cast<merk::FromWordOp>(from)}; fromWord && type == word) {
      return fromWord.getWord();
    }
    if (auto toWord{mlir::dyn_cast<merk::ToWordOp>(from)};
        toWord && toWord.getValue().getType() == type) {
      return toWord.getValue();
    }
  }
  if (value.getType() != word) {
    value = merk::ToWordOp::create(builder_, loc, word, value);
  }
  if (type != word) {
    value = merk::FromWordOp::create(builder_, loc, type, value);
  }
  return value;
}

// Temporary (chunk 1 step 6, docs/INPROGRESS.md): the library's modules not
// compiled yet, whose types' descriptors each module using them makes.
static constexpr llvm::StringLiteral kUncompiledLibrary[]{"array", "bool",
    "builtin", "char", "exception", "int", "int16", "int32", "int64", "int8",
    "list", "string", "uint", "uint16", "uint32", "uint64", "uint8", "univ"};

// A type's arguments in its type_info: a higher-order type's are its
// arguments' types and then a function's result's, as polymorphism analysis
// has them.
static llvm::SmallVector<const semantics::Type *> TypeParts(
    const semantics::Type &type) {
  llvm::SmallVector<const semantics::Type *> parts{
      type.arguments.begin(), type.arguments.end()};
  if (type.result) {
    parts.push_back(type.result);
  }
  return parts;
}

// Whether a type's type_info has the number of its arguments in word 1, as a
// tuple's and a higher-order type's, whose descriptors are for any number.
static bool HasVariableArity(const semantics::Type &type) {
  return type.kind == semantics::TypeKind::Tuple ||
      type.kind == semantics::TypeKind::HigherOrder;
}

// A type constructor's name in merk.type_info, with its arity.
static std::optional<std::string> TypeConstructorName(
    const semantics::Type &type) {
  std::string arity{"/" + std::to_string(TypeParts(type).size())};
  if (type.kind == semantics::TypeKind::Tuple) {
    return "{}" + arity;
  }
  if (type.kind == semantics::TypeKind::HigherOrder) {
    return (type.predOrFunc == parser::PredOrFunc::Function ? "func" : "pred") +
        arity;
  }
  if (type.kind != semantics::TypeKind::Named) {
    return std::nullopt;
  }
  const semantics::Symbol &symbol{*type.constructor};
  std::string name{symbol.name};
  if (!symbol.module.empty()) {
    name = std::string{symbol.module} + "." + name;
  }
  return name + arity;
}

void MerkGenerator::EmitDescriptor(llvm::StringRef constructor,
    llvm::StringRef module, llvm::StringRef name, llvm::StringRef kind,
    llvm::StringRef linkage, unsigned arity, mlir::Type term,
    merk::ProcOp unify, merk::ProcOp compare, mlir::ArrayAttr functors) {
  mlir::OpBuilder::InsertionGuard guard{builder_};
  builder_.setInsertionPointToEnd(module_.getBody());
  merk::TypeCtorOp::create(builder_, builder_.getUnknownLoc(), constructor,
      module, name, kind, linkage, arity,
      term ? mlir::TypeAttr::get(term) : mlir::TypeAttr{},
      unify ? mlir::FlatSymbolRefAttr::get(unify.getSymNameAttr())
            : mlir::FlatSymbolRefAttr{},
      compare ? mlir::FlatSymbolRefAttr::get(compare.getSymNameAttr())
              : mlir::FlatSymbolRefAttr{},
      functors);
}

bool MerkGenerator::EnsureVoidDescriptor() {
  if (descriptors_.insert("void/0").second) {
    // There are no values of it: only a value nothing determines has it, and
    // nothing can compare those, so its entry is never reached.
    EmitDescriptor("void/0", "", "void", "void", "linkonce", 0, {},
        DescriptorEntry(false, "void/0", nullptr, {}, /*dummy=*/true, {}),
        DescriptorEntry(true, "void/0", nullptr, {}, /*dummy=*/true, {}), {});
  }
  return true;
}

merk::ProcOp MerkGenerator::HigherOrderEntry(
    bool compare, llvm::StringRef constructor) {
  mlir::Location loc{builder_.getUnknownLoc()};
  auto info{merk::TypeInfoType::get(&context_)};
  auto word{merk::WordType::get(&context_)};
  mlir::OpBuilder::InsertionGuard guard{builder_};
  builder_.setInsertionPointToEnd(module_.getBody());
  auto proc{merk::ProcOp::create(builder_, loc,
      ((compare ? "compare_entry." : "unify_entry.") + constructor).str(),
      compare ? builder_.getFunctionType(
                    {info, word, word}, mlir::Type{*ComparisonType()})
              : builder_.getFunctionType({info, word, word}, {}),
      compare ? merk::Determinism::Det : merk::Determinism::Semidet)};
  proc.setPrivate();
  mlir::Block *entry{proc.addEntryBlock()};
  builder_.setInsertionPointToStart(entry);
  // It does not return.
  merk::ForeignCallOp::create(builder_, loc, mlir::TypeRange{},
      "merk_higher_order_called",
      mlir::ValueRange{IntegerConstant(builder_.getI32Type(), compare, loc)},
      /*can_fail=*/false);
  if (compare) {
    merk::SucceedOp::create(builder_, loc, ComparisonValue("=", loc));
  } else {
    merk::SucceedOp::create(builder_, loc, mlir::ValueRange{});
  }
  return proc;
}

// The entry a generic unification or comparison calls: the type's arguments'
// type_infos out of its own, and its values out of their words.
merk::ProcOp MerkGenerator::DescriptorEntry(bool compare,
    llvm::StringRef constructor,
    const semantics::Type *canonical, mlir::Type represented, bool dummy,
    parser::CharBlock at) {
  mlir::Location loc{builder_.getUnknownLoc()};
  auto info{merk::TypeInfoType::get(&context_)};
  auto word{merk::WordType::get(&context_)};
  mlir::OpBuilder::InsertionGuard guard{builder_};
  builder_.setInsertionPointToEnd(module_.getBody());
  auto proc{merk::ProcOp::create(builder_, loc,
      ((compare ? "compare_entry." : "unify_entry.") + constructor).str(),
      compare ? builder_.getFunctionType(
                    {info, word, word}, mlir::Type{*ComparisonType()})
              : builder_.getFunctionType({info, word, word}, {}),
      compare ? merk::Determinism::Det : merk::Determinism::Semidet)};
  proc.setPrivate();
  mlir::Block *entry{proc.addEntryBlock()};
  builder_.setInsertionPointToStart(entry);
  mlir::Value result;
  if (!dummy) {
    auto saved{std::move(typeInfos_)};
    parser::CharBlock savedAt{equalAt_};
    typeInfos_.clear();
    equalAt_ = at;
    // A tuple's cell has the number of its arguments in word 1.
    unsigned first{canonical->kind == semantics::TypeKind::Tuple ? 2u : 1u};
    for (auto [i, argument] : llvm::enumerate(canonical->arguments)) {
      typeInfos_[argument->var] = merk::TypeInfoWordOp::create(builder_, loc,
          info, entry->getArgument(0), first + i);
    }
    mlir::Value x{Coerce(entry->getArgument(1), represented, loc)};
    mlir::Value y{Coerce(entry->getArgument(2), represented, loc)};
    if (compare) {
      result = Compare(x, y, canonical, /*generic=*/true, loc);
    } else {
      Equal(x, y, canonical, loc);
    }
    typeInfos_ = std::move(saved);
    equalAt_ = savedAt;
  } else if (compare) {
    result = ComparisonValue("=", loc);
  }
  merk::SucceedOp::create(builder_, loc,
      result ? mlir::ValueRange{result} : mlir::ValueRange{});
  return proc;
}

bool MerkGenerator::EnsureDescriptor(
    const semantics::Type &type, parser::CharBlock at) {
  std::optional<std::string> constructor{TypeConstructorName(type)};
  if (!constructor) {
    return false;
  }
  if (descriptors_.contains(*constructor)) {
    return true;
  }
  if (type.kind == semantics::TypeKind::HigherOrder) {
    // As mmc, a program that unifies or compares closures stops.
    descriptors_.insert(*constructor);
    EmitDescriptor(*constructor, "",
        type.predOrFunc == parser::PredOrFunc::Function ? "func" : "pred",
        "closure", "linkonce", TypeParts(type).size(), {},
        HigherOrderEntry(false, *constructor),
        HigherOrderEntry(true, *constructor), {});
    return true;
  }
  // Its most general instance, which its one unify entry works on.
  semantics::TypeContext &context{types_.types()};
  llvm::SmallVector<const semantics::Type *> variables;
  for (unsigned i{0}; i < type.arguments.size(); ++i) {
    variables.push_back(context.Variable(i));
  }
  const semantics::Type *canonical{type.kind == semantics::TypeKind::Tuple
          ? context.Tuple(variables)
          : context.Named(*type.constructor, variables)};
  std::string module, name{"{}"};
  std::string kind{"term"}, linkage{"linkonce"};
  if (type.kind == semantics::TypeKind::Tuple) {
    kind = "tuple";
  } else {
    module = type.constructor->module;
    name = type.constructor->name;
    if (module.empty()) {
      kind = name == "character" ? "char" : name;
    } else if (module == "array" && name == "array") {
      kind = "array";
    } else if (module == moduleName_) {
      linkage = type.constructor->exported ? "external" : "internal";
      // The state of the world, an abstract type that io does not define, is
      // a dummy whose linkonce descriptor each module using it makes, as
      // below.
    } else if (!llvm::is_contained(kUncompiledLibrary, llvm::StringRef{module}) &&
        !(module == "io" && name == "state")) {
      // The module that defines it has it; this one may not even see its
      // constructors.
      descriptors_.insert(*constructor);
      EmitDescriptor(*constructor, module, name, kind, "declaration",
          type.arguments.size(), {}, {}, {}, {});
      return true;
    }
  }
  std::optional<Represented> represented{Represent(canonical)};
  if (!represented) {
    NotYet(at, "type_infos of type " + semantics::TypeToString(type));
    return false;
  }
  if (represented->dummy) {
    kind = "dummy";
    linkage = "linkonce";
  }
  // Before its entry, whose unification of a recursive type needs it.
  descriptors_.insert(*constructor);
  llvm::StringSet<> seen;
  if (!represented->dummy && !Comparable(represented->type, seen)) {
    NotYet(at, "type_infos of type " + semantics::TypeToString(type));
    return false;
  }
  // A primitive type, or an array, has no constructors, nor has the state
  // of the world, an abstract dummy.
  mlir::ArrayAttr functors;
  if (represented->dummy && !modes_.ops().ConstructorsOf(canonical)) {
    functors = builder_.getArrayAttr({});
  } else if (represented->dummy ||
      mlir::isa_and_nonnull<merk::TermType>(represented->type)) {
    functors = FunctorsAttr(canonical, at);
    if (!functors) {
      failed_ = true;
      return false;
    }
  }
  merk::ProcOp unify{DescriptorEntry(false, *constructor, canonical,
      represented->type, represented->dummy, at)};
  merk::ProcOp compare{DescriptorEntry(true, *constructor, canonical,
      represented->type, represented->dummy, at)};
  EmitDescriptor(*constructor, module, name, kind, linkage,
      type.arguments.size(),
      mlir::isa_and_nonnull<merk::TermType>(represented->type)
          ? represented->type
          : mlir::Type{},
      unify, compare, functors);
  return true;
}

// A type parameter by its number, an existentially quantified type
// variable by the place of its type_info in `hidden`, and any other type as an
// array of its constructor and its arguments, in which their descriptors are
// ensured.
mlir::Attribute MerkGenerator::FieldTypeAttr(const semantics::Type &type,
    parser::CharBlock at, const HiddenFields *hidden) {
  if (type.kind == semantics::TypeKind::Variable) {
    // A canonical type's variables are its parameters, numbered from 0.
    if (hidden && type.var >= kCellExistential) {
      auto [field, word]{hidden->places[type.var - kCellExistential]};
      return builder_.getDenseI64ArrayAttr({field, word});
    }
    return builder_.getI64IntegerAttr(type.var);
  }
  std::optional<std::string> constructor{TypeConstructorName(type)};
  if (!constructor || !EnsureDescriptor(type, at)) {
    if (!constructor) {
      NotYet(at, "type_infos of type " + semantics::TypeToString(type));
    }
    return nullptr;
  }
  llvm::SmallVector<mlir::Attribute> parts{builder_.getStringAttr(*constructor)};
  for (const semantics::Type *argument : TypeParts(type)) {
    mlir::Attribute attribute{FieldTypeAttr(*argument, at, hidden)};
    if (!attribute) {
      return nullptr;
    }
    parts.push_back(attribute);
  }
  return builder_.getArrayAttr(parts);
}

mlir::ArrayAttr MerkGenerator::FunctorsAttr(
    const semantics::Type *canonical, parser::CharBlock at) {
  const std::vector<semantics::ConstructorTypes> *constructors{
      modes_.ops().ConstructorsOf(canonical)};
  if (!constructors) {
    return nullptr;
  }
  llvm::SmallVector<mlir::Attribute> functors;
  for (const semantics::ConstructorTypes &constructor : *constructors) {
    llvm::SmallVector<mlir::Attribute> parts{
        builder_.getStringAttr(constructor.functor.name)};
    std::optional<HiddenFields> hidden{HiddenFieldsOf(constructor)};
    if (!hidden) {
      NotYet(at, "type_infos of type " + semantics::TypeToString(*canonical));
      return nullptr;
    }
    for (const semantics::Type *argument :
        ArgumentsOf(canonical, constructor)) {
      mlir::Attribute attribute{FieldTypeAttr(*argument, at, &*hidden)};
      if (!attribute) {
        return nullptr;
      }
      parts.push_back(attribute);
    }
    functors.push_back(builder_.getArrayAttr(parts));
  }
  return builder_.getArrayAttr(functors);
}

mlir::Value MerkGenerator::TypeInfoOfType(const semantics::Type *type,
    mlir::Location loc, parser::CharBlock at, bool freeIsVoid) {
  if (type->kind == semantics::TypeKind::Variable) {
    mlir::Value info{typeInfos_.lookup(type->var)};
    if (!info && freeIsVoid) {
      // Nothing determines it, as the elements of [] that nothing else uses.
      EnsureVoidDescriptor();
      return merk::TypeInfoOp::create(builder_, loc,
          merk::TypeInfoType::get(&context_), "void/0", mlir::ValueRange{});
    }
    if (!info) {
      NotYet(at, "type_infos of type " + semantics::TypeToString(*type));
    }
    return info;
  }
  std::optional<std::string> constructor{TypeConstructorName(*type)};
  if (!constructor || !EnsureDescriptor(*type, at)) {
    if (!constructor) {
      NotYet(at, "type_infos of type " + semantics::TypeToString(*type));
    }
    return nullptr;
  }
  llvm::SmallVector<mlir::Value> arguments;
  for (const semantics::Type *argument : TypeParts(*type)) {
    mlir::Value info{TypeInfoOfType(argument, loc, at, freeIsVoid)};
    if (!info) {
      return nullptr;
    }
    arguments.push_back(info);
  }
  return merk::TypeInfoOp::create(builder_, loc,
      merk::TypeInfoType::get(&context_), *constructor, arguments);
}

mlir::Value MerkGenerator::TypeInfoValue(const semantics::TypeInfo &info,
    mlir::Location loc, parser::CharBlock at) {
  auto type{merk::TypeInfoType::get(&context_)};
  switch (info.kind) {
  case semantics::TypeInfoKind::Parameter:
    return typeInfos_.lookup(info.parameter);
  case semantics::TypeInfoKind::Constant:
    return TypeInfoOfType(info.type, loc, at, /*freeIsVoid=*/true);
  case semantics::TypeInfoKind::Void:
    EnsureVoidDescriptor();
    return merk::TypeInfoOp::create(
        builder_, loc, type, "void/0", mlir::ValueRange{});
  case semantics::TypeInfoKind::Cell: {
    std::optional<std::string> constructor{TypeConstructorName(*info.type)};
    if (!constructor) {
      NotYet(at, "type_infos of type " + semantics::TypeToString(*info.type));
      return nullptr;
    }
    if (!EnsureDescriptor(*info.type, at)) {
      return nullptr;
    }
    llvm::SmallVector<mlir::Value> arguments;
    for (const semantics::TypeInfo *argument : info.arguments) {
      mlir::Value value{TypeInfoValue(*argument, loc, at)};
      if (!value) {
        return nullptr;
      }
      arguments.push_back(value);
    }
    return merk::TypeInfoOp::create(builder_, loc, type, *constructor, arguments);
  }
  case semantics::TypeInfoKind::Dictionary: {
    mlir::Value dictionary{DictionaryValue(*info.dictionary, loc, at)};
    if (!dictionary) {
      return nullptr;
    }
    return merk::TypeclassInfoWordOp::create(builder_, loc, type, dictionary,
        info.index);
  }
  case semantics::TypeInfoKind::Argument: {
    mlir::Value from{TypeInfoValue(*info.arguments.front(), loc, at)};
    if (!from) {
      return nullptr;
    }
    const semantics::Type *whole{info.arguments.front()->type};
    // A tuple's cell has the number of its arguments in word 1.
    unsigned first{whole && HasVariableArity(*whole) ? 2u : 1u};
    return merk::TypeInfoWordOp::create(
        builder_, loc, type, from, first + info.index - 1);
  }
  case semantics::TypeInfoKind::Output: {
    mlir::Value value{Given(&info)};
    if (!value) {
      NotYet(at, "a type_info given where it is not available");
    }
    return value;
  }
  }
  llvm_unreachable("a type_info's kind");
}

mlir::Value MerkGenerator::DictionaryValue(const semantics::Dictionary &dictionary,
    mlir::Location loc, parser::CharBlock at) {
  auto type{merk::TypeclassInfoType::get(&context_)};
  if (builder_.getInsertionBlock()
          ->getParentOp()
          ->getParentOfType<mlir::omp::TargetOp>() ||
      mlir::isa<mlir::omp::TargetOp>(
          builder_.getInsertionBlock()->getParentOp())) {
    // Temporary (specialization, design document, "What polymorphism costs"):
    // a device has no run-time dispatch; specialization will remove these.
    NotYet(at, "type class constraints in target regions");
    return nullptr;
  }
  switch (dictionary.kind) {
  case semantics::DictionaryKind::Parameter:
    return dictionaries_.lookup(dictionary.parameter);
  case semantics::DictionaryKind::Superclass: {
    mlir::Value from{DictionaryValue(*dictionary.from, loc, at)};
    if (!from) {
      return nullptr;
    }
    return merk::TypeclassInfoWordOp::create(
        builder_, loc, type, from, dictionary.index);
  }
  case semantics::DictionaryKind::Constant:
  case semantics::DictionaryKind::Instance: {
    if (!dictionary.instance) {
      NotYet(at, "this typeclass_info");
      return nullptr;
    }
    llvm::SmallVector<mlir::Value> arguments;
    for (const semantics::Dictionary *super : dictionary.supers) {
      arguments.push_back(DictionaryValue(*super, loc, at));
    }
    for (const semantics::TypeInfo *info : dictionary.types) {
      arguments.push_back(TypeInfoValue(*info, loc, at));
    }
    for (const semantics::TypeInfo *info : dictionary.typeInfos) {
      arguments.push_back(TypeInfoValue(*info, loc, at));
    }
    for (const semantics::Dictionary *premise : dictionary.premises) {
      arguments.push_back(DictionaryValue(*premise, loc, at));
    }
    if (llvm::is_contained(arguments, mlir::Value{})) {
      return nullptr;
    }
    return merk::TypeclassInfoOp::create(builder_, loc, type,
        InstanceSymbol(*dictionary.instance), arguments);
  }
  case semantics::DictionaryKind::Output: {
    mlir::Value value{Given(&dictionary)};
    if (!value) {
      NotYet(at, "a typeclass_info given where it is not available");
    }
    return value;
  }
  }
  llvm_unreachable("a dictionary's kind");
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
  std::optional<llvm::SmallVector<mlir::Type>> hidden{
      HiddenInputs(procedure, at)};
  if (!hidden) {
    return nullptr;
  }
  merk::ProcOp proc{NewProc(MangledName(procedure), at, *hidden, procedure.modes,
      types, *merkDeterminism, HiddenOutputs(procedure))};
  if (proc) {
    procs_[&procedure] = proc;
  }
  return proc;
}

merk::ProcOp MerkGenerator::NewProc(llvm::StringRef name, parser::CharBlock at,
    llvm::ArrayRef<mlir::Type> hidden, llvm::ArrayRef<semantics::ArgMode> modes,
    llvm::ArrayRef<const semantics::Type *> types,
    merk::Determinism determinism, llvm::ArrayRef<mlir::Type> hiddenOutputs) {
  llvm::SmallVector<mlir::Type> inputs{hidden};
  llvm::SmallVector<mlir::Type> outputs{hiddenOutputs};
  for (auto [mode, type] : llvm::zip(modes, types)) {
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
  auto proc{merk::ProcOp::create(builder_, Location(at), name,
      builder_.getFunctionType(inputs, outputs), determinism)};
  proc.setPrivate();
  return proc;
}

merk::ProcOp MerkGenerator::MethodProc(const ProcedureBody &body) {
  if (merk::ProcOp proc{methodProcs_.lookup(body.method)}) {
    return proc;
  }
  const Procedure &procedure{*body.procedure};
  parser::CharBlock at{procedure.symbol->source};
  std::optional<parser::Determinism> determinism{
      determinism_.DeterminismOf(procedure)};
  std::optional<merk::Determinism> merkDeterminism;
  if (determinism) {
    merkDeterminism = MerkDeterminism(*determinism);
  }
  if (!merkDeterminism) {
    NotYet(at, "methods of this determinism");
    return nullptr;
  }
  llvm::SmallVector<mlir::Type> hidden;
  const auto *polymorphism{polymorphism_.lookup(&body)};
  if (!polymorphism) {
    NotYet(at, "this method's type_infos");
    return nullptr;
  }
  for (const semantics::HiddenArgument &argument : polymorphism->hidden) {
    hidden.push_back(argument.typeInfo
            ? mlir::Type{merk::TypeInfoType::get(&context_)}
            : mlir::Type{merk::TypeclassInfoType::get(&context_)});
  }
  std::string name{MangledName(procedure) + "$" +
      body.method->method->instanceText};
  merk::ProcOp proc{NewProc(name, at, hidden, procedure.modes,
      body.method->method->signature.arguments, *merkDeterminism)};
  if (proc) {
    methodProcs_[body.method] = proc;
  }
  return proc;
}

const ProcedureBody *MerkGenerator::MethodBody(
    const semantics::InstanceHead &head, const Procedure &procedure) {
  for (const ProcedureBody &body : determinism_.bodies()) {
    if (body.method && body.procedure == &procedure &&
        body.method->method->instance == head.item->decl) {
      return &body;
    }
  }
  return nullptr;
}

void MerkGenerator::GenerateInstances() {
  for (const semantics::InstanceHead &head : types_.instances()) {
    if (!head.item->local || head.item->decl->abstract) {
      continue;
    }
    std::string symbol{InstanceSymbol(head)};
    llvm::SmallVector<mlir::Attribute> table;
    for (const Procedure *procedure : MethodSlots(*head.typeclass)) {
      const ProcedureBody *found{MethodBody(head, *procedure)};
      parser::CharBlock at{procedure->symbol->source};
      if (!found) {
        NotYet(at, "instances without this method's body");
        continue;
      }
      std::string name{MangledName(*procedure) + "$" +
          found->method->method->instanceText + "$dispatch"};
      GenerateDispatch(head, *procedure, *found, name);
      table.push_back(mlir::FlatSymbolRefAttr::get(&context_, name));
    }
    // An instance declared in the interface is for other modules. An abstract
    // declaration has no head of its own, so any of the type class's counts.
    bool exported{head.item->exported};
    for (const semantics::InstanceItem &item : resolution_.instances()) {
      exported = exported || (item.local && item.exported && item.decl->abstract);
    }
    auto op{module_.lookupSymbol<merk::InstanceOp>(symbol)};
    op.setMethodsAttr(builder_.getArrayAttr(table));
    op.setVisibility(exported ? mlir::SymbolTable::Visibility::Public
                              : mlir::SymbolTable::Visibility::Private);
  }
}

void MerkGenerator::GenerateDispatch(const semantics::InstanceHead &head,
    const Procedure &procedure, const ProcedureBody &body,
    llvm::StringRef name) {
  parser::CharBlock at{procedure.symbol->source};
  mlir::Location loc{Location(at)};
  failed_ = false;
  // Temporary (chunk 1 step 4, docs/INPROGRESS.md): polymorphism analysis does
  // not analyse methods with existential types, nor their instances.
  if (!HiddenOutputs(procedure).empty()) {
    NotYet(at, "methods with existential types");
    return;
  }
  merk::ProcOp method{MethodProc(body)};
  const semantics::Signature *declared{types_.SignatureOf(*procedure.symbol)};
  if (!method || !declared) {
    return;
  }
  // As the type class declares it: the typeclass_info, then the method's own
  // hidden arguments, which follow the class's constraint.
  std::optional<llvm::SmallVector<mlir::Type>> hidden{
      HiddenInputs(procedure, at)};
  if (!hidden) {
    return;
  }
  llvm::SmallVector<mlir::Type> own;
  auto dictionaryType{merk::TypeclassInfoType::get(&context_)};
  own.push_back(dictionaryType);
  bool skipped{false};
  for (mlir::Type type : *hidden) {
    if (!skipped && type == dictionaryType) {
      skipped = true; // the class's own, which comes first
      continue;
    }
    own.push_back(type);
  }
  merk::ProcOp proc{NewProc(name, at, own, procedure.modes, declared->arguments,
      method.getDeterminism())};
  if (!proc) {
    return;
  }
  mlir::Block *entry{proc.addEntryBlock()};
  mlir::OpBuilder::InsertionGuard guard{builder_};
  builder_.setInsertionPointToStart(entry);
  mlir::Value dictionary{entry->getArgument(0)};
  // The instance's hidden arguments, from the typeclass_info's cell.
  unsigned word{1 +
      static_cast<unsigned>(types_.SuperclassesOf(*head.typeclass).size() +
          head.types.size())};
  unsigned instanceVars{0};
  for (const semantics::Type *type : head.types) {
    semantics::ForEachVariable(*type,
        [&](unsigned var) { instanceVars = std::max(instanceVars, var + 1); });
  }
  llvm::SmallVector<mlir::Value> inputs;
  for (bool constrained : ConstrainedVariables(instanceVars, head.constraints)) {
    if (!constrained) {
      inputs.push_back(merk::TypeclassInfoWordOp::create(builder_, loc,
          merk::TypeInfoType::get(&context_), dictionary, word++));
    }
  }
  for (std::size_t i{0}; i < head.constraints.size(); ++i) {
    inputs.push_back(merk::TypeclassInfoWordOp::create(
        builder_, loc, dictionaryType, dictionary, word++));
  }
  unsigned input{1};
  for (; input < own.size(); ++input) {
    inputs.push_back(entry->getArgument(input));
  }
  // The arguments: a type variable's word where the instance has a type.
  llvm::ArrayRef<const semantics::Type *> instance{
      body.method->method->signature.arguments};
  llvm::SmallVector<std::pair<bool, mlir::Type>> outputs;
  for (auto [mode, classType, instanceType] :
      llvm::zip(procedure.modes, declared->arguments, instance)) {
    Represented asDeclared{*Represent(classType)};
    Represented asInstance{*Represent(instanceType)};
    if (asDeclared.dummy) {
      continue;
    }
    if (IsOutput(mode)) {
      outputs.push_back({asInstance.dummy, asDeclared.type});
    } else {
      mlir::Value value{entry->getArgument(input++)};
      if (!asInstance.dummy) {
        inputs.push_back(Coerce(value, asInstance.type, loc));
      }
    }
  }
  auto call{merk::CallOp::create(builder_, loc,
      method.getFunctionType().getResults(), method.getSymName(), inputs)};
  llvm::SmallVector<mlir::Value> results;
  unsigned result{0};
  for (auto [dummy, type] : outputs) {
    if (dummy) {
      results.push_back(merk::ToWordOp::create(builder_, loc,
          merk::WordType::get(&context_),
          IntegerConstant(builder_.getI64Type(), 0, loc)));
    } else {
      results.push_back(Coerce(call.getResult(result++), type, loc));
    }
  }
  merk::SucceedOp::create(builder_, loc, results);
}

void MerkGenerator::Call(const Body &goal, const Procedure &callee,
    llvm::ArrayRef<BodyVar> arguments, const ProcedureBody &body) {
  if (callee.modes.size() != arguments.size()) {
    NotYet(Where(goal.tokens), "calls with hidden arguments");
    return;
  }
  if (goal.modes && goal.modes->implied) {
    // An implied mode: the call outputs to fresh values the arguments that
    // are already bound, and each is then tested against its own value.
    // The value stays for the call to read, as A = A mod B also passes A
    // as an input; the call reads its inputs before it sets its outputs.
    llvm::SmallVector<std::pair<BodyVar, mlir::Value>> bound;
    for (auto [argument, mode] : llvm::zip(arguments, callee.modes)) {
      if (IsOutput(mode) && ValueOf(argument) &&
          !Represent(body.vars[argument].type)->dummy &&
          !llvm::is_contained(llvm::make_first_range(bound), argument)) {
        bound.push_back({argument, values_.lookup(argument)});
      }
    }
    Called(goal, callee, arguments, body);
    for (auto [argument, value] : bound) {
      mlir::Value output{ValueOf(argument)};
      values_[argument] = value;
      if (failed_ || Terminated()) {
        continue;
      }
      Test(goal, value, output, body.vars[argument].type);
    }
    return;
  }
  Called(goal, callee, arguments, body);
}

void MerkGenerator::PassTypeInfo(const semantics::TypeInfo &info,
    llvm::SmallVectorImpl<mlir::Value> &inputs, mlir::Location loc,
    parser::CharBlock at) {
  mlir::Value value{TypeInfoValue(info, loc, at)};
  inputs.push_back(value);
  failed_ = failed_ || !value;
}

void MerkGenerator::PassDictionary(const semantics::Dictionary &dictionary,
    llvm::SmallVectorImpl<mlir::Value> &inputs, mlir::Location loc,
    parser::CharBlock at) {
  mlir::Value value{DictionaryValue(dictionary, loc, at)};
  inputs.push_back(value);
  failed_ = failed_ || !value;
}

void MerkGenerator::Called(const Body &goal, const Procedure &callee,
    llvm::ArrayRef<BodyVar> arguments, const ProcedureBody &body) {
  if (Builtin(goal, callee, arguments, body)) {
    return;
  }
  parser::CharBlock at{Where(goal.tokens)};
  mlir::Location loc{Location(goal.tokens)};
  std::optional<llvm::SmallVector<mlir::Type>> hidden{HiddenInputs(callee, at)};
  if (!hidden) {
    failed_ = true;
    return;
  }
  bool isMethod{callee.symbol->owner &&
      callee.symbol->owner->kind == semantics::SymbolKind::Typeclass};
  llvm::SmallVector<mlir::Type> hiddenOutputs{HiddenOutputs(callee)};
  if (isMethod && !hiddenOutputs.empty()) {
    NotYet(at, "methods with existential types");
    return;
  }
  bool polymorphic{!hidden->empty() || !hiddenOutputs.empty()};
  const semantics::PassedInfos *passed{nullptr};
  if (polymorphic) {
    const semantics::ProcedurePolymorphism *polymorphism{
        polymorphism_.lookup(&body)};
    if (polymorphism) {
      if (auto it{polymorphism->goals.find(&goal)};
          it != polymorphism->goals.end()) {
        passed = &it->second;
      }
    }
    if (!passed ||
        passed->typeInfos.size() + passed->dictionaries.size() !=
            hidden->size() + hiddenOutputs.size()) {
      NotYet(at, "this call's type_infos");
      return;
    }
  }
  // A method of an instance known here, whose body this module has, is called
  // directly: no typeclass_info is made, and the values keep their
  // representations, as a device needs (design document, "What polymorphism
  // costs").
  const semantics::Dictionary *known{nullptr};
  const ProcedureBody *direct{nullptr};
  if (isMethod) {
    const semantics::Dictionary *own{passed->dictionaries.front()};
    if ((own->kind == semantics::DictionaryKind::Constant ||
            own->kind == semantics::DictionaryKind::Instance) &&
        own->instance) {
      known = own;
      direct = MethodBody(*own->instance, callee);
    }
  }
  // A polymorphic callee takes its declared types, whatever the caller's are.
  llvm::SmallVector<const semantics::Type *> types;
  if (direct) {
    llvm::ArrayRef<const semantics::Type *> declared{
        direct->method->method->signature.arguments};
    types.append(declared.begin(), declared.end());
  } else if (polymorphic) {
    llvm::ArrayRef<const semantics::Type *> declared{
        types_.SignatureOf(*callee.symbol)->arguments};
    types.append(declared.begin(), declared.end());
  } else {
    for (BodyVar argument : arguments) {
      types.push_back(body.vars[argument].type);
    }
  }
  // In a body that gives words, a call whose outputs are the body's calls the
  // callee's word copy, which gives them as words too, so that it is a tail
  // call when it comes last.
  llvm::SmallVector<BodyVar> outputVars;
  for (auto [argument, mode] : llvm::zip(arguments, callee.modes)) {
    if (IsOutput(mode)) {
      outputVars.push_back(argument);
    }
  }
  const ProcedureBody *copyable{nullptr};
  if (wordOutputs_ && !isMethod && hiddenOutputs.empty() &&
      !(goal.modes && goal.modes->implied) &&
      llvm::equal(outputVars, *wordOutputs_)) {
    copyable = CopyableBody(callee);
  }
  merk::ProcOp proc;
  if (direct) {
    proc = MethodProc(*direct);
    if (!proc) {
      failed_ = true;
      return;
    }
  } else if (copyable) {
    proc = WordProcFor(*copyable);
    if (!proc) {
      return;
    }
  } else if (!isMethod) {
    proc = ProcFor(callee, types, at);
    if (!proc) {
      return;
    }
  }
  llvm::SmallVector<mlir::Value> inputs;
  mlir::Value dictionary;
  if (direct) {
    // The instance's hidden arguments come first.
    for (const semantics::TypeInfo *info : known->typeInfos) {
      PassTypeInfo(*info, inputs, loc, at);
    }
    for (const semantics::Dictionary *premise : known->premises) {
      PassDictionary(*premise, inputs, loc, at);
    }
  }
  // What this goal gives, in the order of the callee's first results.
  llvm::SmallVector<const void *> gives;
  if (passed) {
    for (const semantics::TypeInfo *info : passed->typeInfos) {
      if (info->kind == semantics::TypeInfoKind::Output && info->goal == &goal) {
        gives.push_back(info);
      } else {
        PassTypeInfo(*info, inputs, loc, at);
      }
    }
    for (auto [i, given] : llvm::enumerate(passed->dictionaries)) {
      if (given->kind == semantics::DictionaryKind::Output &&
          given->goal == &goal) {
        gives.push_back(given);
      } else if (!isMethod || i != 0) {
        PassDictionary(*given, inputs, loc, at);
      } else if (!direct) {
        // A method's type class's comes first, and chooses the procedure.
        dictionary = DictionaryValue(*given, loc, at);
        failed_ = failed_ || !dictionary;
      }
    }
  }
  if (failed_) {
    return;
  }
  // None for a value with no bits at a type variable's place, whose word
  // comes back unused.
  llvm::SmallVector<std::optional<BodyVar>> outputs;
  for (auto [argument, mode, type] : llvm::zip(arguments, callee.modes, types)) {
    std::optional<Represented> actual{Represent(body.vars[argument].type)};
    if (!actual) {
      NotYet(at, "values of type " +
              semantics::TypeToString(*body.vars[argument].type));
      return;
    }
    bool dummy{actual->dummy};
    if (Represent(type)->dummy) {
      continue;
    }
    if (IsOutput(mode)) {
      outputs.push_back(dummy ? std::nullopt : std::make_optional(argument));
    } else if (dummy) {
      inputs.push_back(merk::ToWordOp::create(builder_, loc,
          merk::WordType::get(&context_),
          IntegerConstant(builder_.getI64Type(), 0, loc)));
    } else {
      inputs.push_back(Coerce(ValueOf(argument), Represent(type)->type, loc));
    }
  }
  if (copyable) {
    llvm::ArrayRef<mlir::Type> types{proc.getFunctionType().getInputs()};
    if (types.size() != inputs.size()) {
      NotYet(at, "this call of a word copy");
      return;
    }
    for (auto [input, type] : llvm::zip(inputs, types)) {
      input = Coerce(input, type, loc);
    }
  }
  mlir::Operation *call;
  if (isMethod && !direct) {
    std::optional<parser::Determinism> determinism{
        determinism_.DeterminismOf(callee)};
    std::optional<merk::Determinism> merkDeterminism;
    if (determinism) {
      merkDeterminism = MerkDeterminism(*determinism);
    }
    if (!merkDeterminism) {
      NotYet(at, "methods of this determinism");
      return;
    }
    llvm::SmallVector<const Procedure *> slots{
        MethodSlots(*callee.symbol->owner)};
    llvm::SmallVector<mlir::Type> resultTypes;
    for (auto [mode, type] : llvm::zip(callee.modes, types)) {
      std::optional<Represented> represented{Represent(type)};
      if (represented && !represented->dummy && IsOutput(mode)) {
        resultTypes.push_back(represented->type);
      }
    }
    call = merk::CallMethodOp::create(builder_, loc, resultTypes, dictionary,
        llvm::find(slots, &callee) - slots.begin(), inputs, *merkDeterminism);
  } else {
    call = merk::CallOp::create(builder_, loc,
        proc.getFunctionType().getResults(), proc.getSymName(), inputs);
  }
  // A call that never succeeds ends the branch it is in, which then need not
  // bind what the other branches bind.
  if (std::optional<parser::Determinism> determinism{
          determinism_.DeterminismOf(callee)};
      determinism && (*determinism == parser::Determinism::Erroneous ||
                         *determinism == parser::Determinism::Failure)) {
    merk::FailOp::create(builder_, loc);
    return;
  }
  if (copyable) {
    for (auto [var, word] : llvm::zip_equal(outputVars, call->getResults())) {
      SetFromWord(var, word, body, loc);
    }
    return;
  }
  for (auto [node, result] : llvm::zip(gives, call->getResults())) {
    Give(node, result);
  }
  for (auto [var, result] :
      llvm::zip(outputs, call->getResults().drop_front(gives.size()))) {
    if (!var) {
      continue;
    }
    values_[*var] =
        Coerce(result, Represent(body.vars[*var].type)->type, loc);
  }
}

const ProcedureBody *MerkGenerator::CopyableBody(const Procedure &procedure) {
  auto it{bodies_.find(&procedure)};
  if (it == bodies_.end() || !HiddenOutputs(procedure).empty()) {
    return nullptr;
  }
  return it->second;
}

merk::ProcOp MerkGenerator::WordProcFor(const ProcedureBody &body) {
  const Procedure &procedure{*body.procedure};
  if (merk::ProcOp proc{wordProcs_.lookup(&procedure)}) {
    return proc;
  }
  parser::CharBlock at{procedure.symbol->source};
  std::optional<parser::Determinism> determinism{
      determinism_.DeterminismOf(procedure)};
  std::optional<merk::Determinism> merkDeterminism;
  if (determinism) {
    merkDeterminism = MerkDeterminism(*determinism);
  }
  if (!merkDeterminism) {
    NotYet(at, "word copies of procedures of this determinism");
    return nullptr;
  }
  std::optional<llvm::SmallVector<mlir::Type>> inputs{
      HiddenInputs(procedure, at)};
  if (!inputs) {
    return nullptr;
  }
  unsigned outputs{0};
  for (auto [var, mode] : llvm::zip(body.head, procedure.modes)) {
    const semantics::Type *type{body.vars[var].type};
    std::optional<Represented> represented{Represent(type)};
    if (!represented) {
      NotYet(at, "values of type " + semantics::TypeToString(*type));
      return nullptr;
    }
    if (IsOutput(mode)) {
      ++outputs;
    } else if (!represented->dummy) {
      inputs->push_back(represented->type);
    }
  }
  mlir::OpBuilder::InsertionGuard guard{builder_};
  builder_.setInsertionPointToEnd(module_.getBody());
  auto proc{merk::ProcOp::create(builder_, Location(at),
      MangledName(procedure) + "$words",
      builder_.getFunctionType(*inputs,
          llvm::SmallVector<mlir::Type>(outputs, merk::WordType::get(&context_))),
      *merkDeterminism)};
  proc.setPrivate();
  wordProcs_[&procedure] = proc;
  pendingCopies_.push_back(&body);
  return proc;
}

llvm::SmallVector<mlir::Value> MerkGenerator::WordResults(
    llvm::ArrayRef<BodyVar> outputs, const ProcedureBody &body,
    mlir::Location loc) {
  auto word{merk::WordType::get(&context_)};
  llvm::SmallVector<mlir::Value> results;
  for (BodyVar var : outputs) {
    if (Represent(body.vars[var].type)->dummy) {
      mlir::Value given{values_.lookup(kDummyWordVar + var)};
      results.push_back(given ? given : ZeroWord(loc));
    } else if (mlir::Value value{ValueOf(var)}) {
      results.push_back(Coerce(value, word, loc));
    } else {
      results.push_back(nullptr);
    }
  }
  return results;
}

void MerkGenerator::SetFromWord(BodyVar var, mlir::Value word,
    const ProcedureBody &body, mlir::Location loc) {
  Represented represented{*Represent(body.vars[var].type)};
  if (!represented.dummy) {
    values_[var] = Coerce(word, represented.type, loc);
  } else if (wordOutputs_) {
    values_[kDummyWordVar + var] = word;
  }
}

merk::ProcOp MerkGenerator::ClosureEntry(merk::ProcOp proc, unsigned captured,
    llvm::ArrayRef<ClosureArgument> arguments, parser::CharBlock at) {
  std::string name{proc.getSymName().str() + "$closure" + std::to_string(captured)};
  if (merk::ProcOp entry{entries_.lookup(name)}) {
    return entry;
  }
  mlir::Location loc{Location(at)};
  auto word{merk::WordType::get(&context_)};
  llvm::SmallVector<mlir::Type> inputs{merk::ClosureType::get(&context_)};
  unsigned outputs{0};
  for (const ClosureArgument &argument : arguments) {
    if (argument.output) {
      ++outputs;
    } else {
      inputs.push_back(word);
    }
  }
  llvm::ArrayRef<mlir::Type> types{proc.getFunctionType().getInputs()};
  unsigned passed{captured};
  for (const ClosureArgument &argument : arguments) {
    passed += !argument.output && !argument.dummy;
  }
  if (passed != types.size()) {
    NotYet(at, "this closure's arguments");
    return nullptr;
  }
  mlir::OpBuilder::InsertionGuard guard{builder_};
  builder_.setInsertionPointToEnd(module_.getBody());
  auto entry{merk::ProcOp::create(builder_, loc, name,
      builder_.getFunctionType(inputs, llvm::SmallVector<mlir::Type>(outputs, word)),
      proc.getDeterminism())};
  entry.setPrivate();
  entries_[name] = entry;
  mlir::Block *block{entry.addEntryBlock()};
  builder_.setInsertionPointToStart(block);
  mlir::Value closure{block->getArgument(0)};
  llvm::SmallVector<mlir::Value> values;
  for (unsigned i{0}; i < captured; ++i) {
    values.push_back(Coerce(
        merk::ClosureWordOp::create(builder_, loc, word, closure, i), types[i], loc));
  }
  unsigned input{1};
  for (const ClosureArgument &argument : arguments) {
    if (argument.output) {
      continue;
    }
    mlir::Value value{block->getArgument(input++)};
    if (!argument.dummy) {
      values.push_back(Coerce(value, types[values.size()], loc));
    }
  }
  auto call{merk::CallOp::create(builder_, loc,
      proc.getFunctionType().getResults(), proc.getSymName(), values)};
  llvm::SmallVector<mlir::Value> results;
  unsigned result{0};
  for (const ClosureArgument &argument : arguments) {
    if (argument.output) {
      results.push_back(argument.dummy
              ? ZeroWord(loc)
              : Coerce(call.getResult(result++), word, loc));
    }
  }
  merk::SucceedOp::create(builder_, loc, results);
  return entry;
}

merk::ProcOp MerkGenerator::BuiltinEntry(const Body &goal,
    const Procedure &callee, llvm::ArrayRef<const semantics::Type *> types,
    unsigned curried, bool &failed) {
  // Code generation builds in only the library's procedures.
  if (callee.symbol->module == moduleName_) {
    return nullptr;
  }
  std::string name{"merk.builtin." + MangledName(callee) + "$closure" +
      std::to_string(curried)};
  if (merk::ProcOp entry{entries_.lookup(name)}) {
    return entry;
  }
  std::optional<parser::Determinism> determinism{
      determinism_.DeterminismOf(callee)};
  std::optional<merk::Determinism> merkDeterminism;
  if (determinism) {
    merkDeterminism = MerkDeterminism(*determinism);
  }
  if (!merkDeterminism) {
    return nullptr;
  }
  parser::CharBlock at{Where(goal.tokens)};
  mlir::Location loc{Location(goal.tokens)};
  auto word{merk::WordType::get(&context_)};
  llvm::SmallVector<mlir::Type> inputs{merk::ClosureType::get(&context_)};
  unsigned outputs{0};
  for (auto [i, mode] : llvm::enumerate(callee.modes)) {
    if (IsOutput(mode)) {
      ++outputs;
    } else if (i >= curried) {
      inputs.push_back(word);
    }
  }
  // A call of it in a body of its own, whose variables are its arguments.
  ProcedureBody own;
  own.procedure = &callee;
  llvm::SmallVector<BodyVar> vars;
  for (const semantics::Type *type : types) {
    vars.push_back(static_cast<BodyVar>(own.vars.size()));
    own.vars.push_back({"", type});
  }
  Body call{BodyKind::Call, goal.tokens, nullptr};
  mlir::OpBuilder::InsertionGuard guard{builder_};
  builder_.setInsertionPointToEnd(module_.getBody());
  auto entry{merk::ProcOp::create(builder_, loc, name,
      builder_.getFunctionType(inputs, llvm::SmallVector<mlir::Type>(outputs, word)),
      *merkDeterminism)};
  entry.setPrivate();
  mlir::Block *block{entry.addEntryBlock()};
  builder_.setInsertionPointToStart(block);
  Values saved;
  saved.swap(values_);
  std::optional<llvm::SmallVector<BodyVar>> savedWords{std::move(wordOutputs_)};
  wordOutputs_.reset();
  mlir::Value closure{block->getArgument(0)};
  unsigned captured{0};
  unsigned input{1};
  bool represented{true};
  for (auto [i, mode, type] : llvm::enumerate(callee.modes, types)) {
    std::optional<Represented> value{Represent(type)};
    represented = represented && value;
    if (!represented || IsOutput(mode)) {
      continue;
    }
    mlir::Value word{i < curried
            ? (value->dummy ? mlir::Value{}
                            : mlir::Value{merk::ClosureWordOp::create(builder_,
                                  loc, merk::WordType::get(&context_), closure,
                                  captured++)})
            : block->getArgument(input++)};
    if (!value->dummy) {
      values_[vars[i]] = Coerce(word, value->type, loc);
    }
  }
  bool handled{represented && Builtin(call, callee, vars, own)};
  if (handled && !failed_ && !Terminated()) {
    llvm::SmallVector<mlir::Value> results;
    for (auto [i, mode, type] : llvm::enumerate(callee.modes, types)) {
      if (!IsOutput(mode)) {
        continue;
      }
      if (Represent(type)->dummy) {
        results.push_back(ZeroWord(loc));
      } else if (mlir::Value value{ValueOf(vars[i])}) {
        results.push_back(Coerce(value, word, loc));
      } else {
        NotYet(at, "this closure's outputs");
        break;
      }
    }
    if (!failed_) {
      merk::SucceedOp::create(builder_, loc, results);
    }
  }
  values_.swap(saved);
  wordOutputs_ = std::move(savedWords);
  if (!handled || failed_) {
    failed = failed_;
    entry.erase();
    return nullptr;
  }
  entries_[name] = entry;
  return entry;
}

// The variables a goal names, at any depth: the fields its kind has, since
// the others are 0, the first variable's number.
static void CollectVars(const Body &goal, llvm::DenseSet<BodyVar> &vars) {
  switch (goal.kind) {
  case BodyKind::Unify:
    vars.insert(goal.left);
    if (goal.isVar) {
      vars.insert(goal.var);
    }
    break;
  case BodyKind::HOCall:
    vars.insert(goal.closure);
    break;
  case BodyKind::Switch:
    vars.insert(goal.switchVar);
    break;
  default:
    break;
  }
  vars.insert(goal.arguments.begin(), goal.arguments.end());
  vars.insert(goal.lambdaHead.begin(), goal.lambdaHead.end());
  vars.insert(goal.nonlocals.begin(), goal.nonlocals.end());
  vars.insert(goal.promised.begin(), goal.promised.end());
  if (goal.switchOn) {
    vars.insert(*goal.switchOn);
  }
  for (const auto &[var, inst] : goal.entry) {
    vars.insert(var);
  }
  if (goal.lambda) {
    CollectVars(*goal.lambda, vars);
  }
  for (const Body *inner : goal.goals) {
    CollectVars(*inner, vars);
  }
  for (const semantics::SwitchCase &each : goal.cases) {
    CollectVars(*each.goal, vars);
  }
}

void MerkGenerator::LambdaClosure(const Body &goal, const ProcedureBody &body) {
  parser::CharBlock at{Where(goal.tokens)};
  mlir::Location loc{Location(goal.tokens)};
  llvm::ArrayRef<semantics::ArgMode> modes{goal.modes->lambda};
  std::optional<merk::Determinism> determinism{
      MerkDeterminism(goal.modes->determinism)};
  if (!determinism) {
    NotYet(at,
        std::string{semantics::DeterminismName(goal.modes->determinism)} +
            " lambda expressions");
    return;
  }
  if (modes.size() != goal.lambdaHead.size()) {
    NotYet(at, "this lambda expression's modes");
    return;
  }
  // What it captures, each in order: the body's type_infos, its
  // typeclass_infos, what goals gave, and the variables bound before it
  // that it uses.
  llvm::DenseSet<BodyVar> used;
  CollectVars(*goal.lambda, used);
  llvm::SmallVector<unsigned> infoKeys;
  for (const auto &[var, value] : typeInfos_) {
    if (value) {
      infoKeys.push_back(var);
    }
  }
  llvm::sort(infoKeys);
  llvm::SmallVector<unsigned> dictionaryKeys;
  for (const auto &[constraint, value] : dictionaries_) {
    if (value) {
      dictionaryKeys.push_back(constraint);
    }
  }
  llvm::sort(dictionaryKeys);
  llvm::SmallVector<BodyVar> captured;
  for (const auto &[var, value] : values_) {
    if (value && !IsDummyWord(var) &&
        (var >= kFirstGivenVar ||
            (used.contains(var) && !llvm::is_contained(goal.lambdaHead, var)))) {
      captured.push_back(var);
    }
  }
  llvm::sort(captured);
  llvm::SmallVector<mlir::Value> capturedValues;
  for (unsigned var : infoKeys) {
    capturedValues.push_back(typeInfos_.lookup(var));
  }
  for (unsigned constraint : dictionaryKeys) {
    capturedValues.push_back(dictionaries_.lookup(constraint));
  }
  for (BodyVar var : captured) {
    capturedValues.push_back(values_.lookup(var));
  }
  llvm::SmallVector<mlir::Type> inputs{
      llvm::map_range(capturedValues, [](mlir::Value v) { return v.getType(); })};
  llvm::SmallVector<BodyVar> outputs;
  llvm::SmallVector<ClosureArgument> arguments;
  for (auto [var, mode] : llvm::zip(goal.lambdaHead, modes)) {
    const semantics::Type *type{body.vars[var].type};
    std::optional<Represented> represented{Represent(type)};
    if (!represented) {
      NotYet(at, "values of type " + semantics::TypeToString(*type));
      return;
    }
    if (IsOutput(mode)) {
      outputs.push_back(var);
      arguments.push_back({true, false});
    } else {
      arguments.push_back({false, represented->dummy});
      if (!represented->dummy) {
        inputs.push_back(represented->type);
      }
    }
  }
  auto word{merk::WordType::get(&context_)};
  // A word copy's body has the same lambda expressions as the procedure's,
  // which capture values of the same types: one procedure serves both.
  merk::ProcOp proc{lambdaProcs_.lookup(&goal)};
  if (!proc ||
      proc.getFunctionType().getInputs() != llvm::ArrayRef<mlir::Type>{inputs}) {
    {
      mlir::OpBuilder::InsertionGuard guard{builder_};
      builder_.setInsertionPointToEnd(module_.getBody());
      proc = merk::ProcOp::create(builder_, loc,
          procName_ + "$lambda" + std::to_string(lambdas_++),
          builder_.getFunctionType(
              inputs, llvm::SmallVector<mlir::Type>(outputs.size(), word)),
          *determinism);
      proc.setPrivate();
    }
    // Its body, with the captured values and its inputs as the procedure's.
    Values savedValues;
    savedValues.swap(values_);
    llvm::DenseMap<unsigned, mlir::Value> savedInfos;
    savedInfos.swap(typeInfos_);
    llvm::DenseMap<unsigned, mlir::Value> savedDictionaries;
    savedDictionaries.swap(dictionaries_);
    std::optional<llvm::SmallVector<BodyVar>> savedWords{std::move(wordOutputs_)};
    std::string savedName{procName_};
    {
      mlir::OpBuilder::InsertionGuard guard{builder_};
      mlir::Block *entry{proc.addEntryBlock()};
      builder_.setInsertionPointToStart(entry);
      unsigned input{0};
      for (unsigned var : infoKeys) {
        typeInfos_[var] = entry->getArgument(input++);
      }
      for (unsigned constraint : dictionaryKeys) {
        dictionaries_[constraint] = entry->getArgument(input++);
      }
      for (BodyVar var : captured) {
        values_[var] = entry->getArgument(input++);
      }
      for (auto [var, argument] : llvm::zip(goal.lambdaHead, arguments)) {
        if (!argument.output && !argument.dummy) {
          values_[var] = entry->getArgument(input++);
        }
      }
      wordOutputs_ = outputs;
      procName_ = proc.getSymName().str();
      Goal(*goal.lambda, body);
      if (!failed_ && !Terminated()) {
        llvm::SmallVector<mlir::Value> results{WordResults(outputs, body, loc)};
        if (llvm::is_contained(results, mlir::Value{})) {
          NotYet(at, "this lambda expression's outputs");
        } else {
          merk::SucceedOp::create(builder_, loc, results);
        }
      }
    }
    values_.swap(savedValues);
    typeInfos_.swap(savedInfos);
    dictionaries_.swap(savedDictionaries);
    wordOutputs_ = std::move(savedWords);
    procName_ = savedName;
    if (failed_) {
      proc.getBody().dropAllReferences();
      proc.getBody().getBlocks().clear();
      return;
    }
    lambdaProcs_[&goal] = proc;
  }
  merk::ProcOp entry{ClosureEntry(
      proc, static_cast<unsigned>(capturedValues.size()), arguments, at)};
  if (!entry) {
    return;
  }
  llvm::SmallVector<mlir::Value> words;
  for (mlir::Value value : capturedValues) {
    words.push_back(Coerce(value, word, loc));
  }
  values_[goal.left] = merk::ClosureOp::create(builder_, loc,
      merk::ClosureType::get(&context_),
      mlir::FlatSymbolRefAttr::get(&context_, entry.getSymName()), words);
}

void MerkGenerator::NameClosure(const Body &goal, const ProcedureBody &body) {
  const Procedure &callee{*goal.modes->callee};
  parser::CharBlock at{Where(goal.tokens)};
  mlir::Location loc{Location(goal.tokens)};
  if (callee.symbol->owner &&
      callee.symbol->owner->kind == semantics::SymbolKind::Typeclass) {
    NotYet(at, "closures of methods");
    return;
  }
  // mmc reports such a closure as an error ("Where mmc and the manual
  // disagree").
  if (!HiddenOutputs(callee).empty()) {
    NotYet(at, "closures of existentially typed procedures");
    return;
  }
  std::optional<llvm::SmallVector<mlir::Type>> hidden{HiddenInputs(callee, at)};
  if (!hidden) {
    failed_ = true;
    return;
  }
  // A polymorphic callee takes its declared types; a closure's arguments
  // after the curried ones have its type's.
  const semantics::Type *closure{body.vars[goal.left].type};
  std::size_t curried{goal.arguments.size()};
  llvm::SmallVector<const semantics::Type *> types;
  if (!hidden->empty()) {
    llvm::ArrayRef<const semantics::Type *> declared{
        types_.SignatureOf(*callee.symbol)->arguments};
    types.append(declared.begin(), declared.end());
  } else {
    for (BodyVar argument : goal.arguments) {
      types.push_back(body.vars[argument].type);
    }
    types.append(closure->arguments.begin(), closure->arguments.end());
    if (closure->result) {
      types.push_back(closure->result);
    }
  }
  if (types.size() != callee.modes.size()) {
    NotYet(at, "this closure's arguments");
    return;
  }
  // What it captures: what the call in its body would pass, as polymorphism
  // analysis says, and the curried arguments.
  llvm::SmallVector<mlir::Value> captured;
  if (!hidden->empty()) {
    const semantics::PassedInfos *passed{nullptr};
    if (current_) {
      if (auto it{current_->goals.find(&goal)}; it != current_->goals.end()) {
        passed = &it->second;
      }
    }
    if (!passed ||
        passed->typeInfos.size() + passed->dictionaries.size() != hidden->size()) {
      NotYet(at, "this closure's type_infos");
      return;
    }
    for (const semantics::TypeInfo *info : passed->typeInfos) {
      PassTypeInfo(*info, captured, loc, at);
    }
    for (const semantics::Dictionary *dictionary : passed->dictionaries) {
      PassDictionary(*dictionary, captured, loc, at);
    }
    if (failed_) {
      return;
    }
  }
  auto word{merk::WordType::get(&context_)};
  for (auto [argument, type] : llvm::zip(goal.arguments, types)) {
    std::optional<Represented> declared{Represent(type)};
    std::optional<Represented> actual{Represent(body.vars[argument].type)};
    if (!declared || !actual) {
      NotYet(at, "values of type " + semantics::TypeToString(*type));
      return;
    }
    if (!declared->dummy) {
      captured.push_back(actual->dummy
              ? ZeroWord(loc)
              : Coerce(Coerce(ValueOf(argument), declared->type, loc), word, loc));
    }
  }
  for (mlir::Value &value : captured) {
    value = Coerce(value, word, loc);
  }
  bool failed{false};
  merk::ProcOp entry{BuiltinEntry(
      goal, callee, types, static_cast<unsigned>(curried), failed)};
  if (failed) {
    return;
  }
  if (!entry) {
    // A procedure of this module is called through its word copy.
    const ProcedureBody *copyable{CopyableBody(callee)};
    merk::ProcOp proc{copyable ? WordProcFor(*copyable) : ProcFor(callee, types, at)};
    if (!proc) {
      failed_ = true;
      return;
    }
    llvm::SmallVector<ClosureArgument> arguments;
    for (auto [mode, type] : llvm::drop_begin(
             llvm::zip(callee.modes, types), static_cast<unsigned>(curried))) {
      bool dummy{Represent(type)->dummy};
      bool output{IsOutput(mode)};
      arguments.push_back({output, dummy && !(output && copyable)});
    }
    entry = ClosureEntry(proc, static_cast<unsigned>(captured.size()), arguments, at);
    if (!entry) {
      return;
    }
  }
  values_[goal.left] = merk::ClosureOp::create(builder_, loc,
      merk::ClosureType::get(&context_),
      mlir::FlatSymbolRefAttr::get(&context_, entry.getSymName()), captured);
}

void MerkGenerator::HigherOrderCall(const Body &goal, const ProcedureBody &body) {
  parser::CharBlock at{Where(goal.tokens)};
  mlir::Location loc{Location(goal.tokens)};
  mlir::Value closure{ValueOf(goal.closure)};
  llvm::ArrayRef<semantics::ArgMode> modes{goal.modes->lambda};
  if (!closure || modes.size() != goal.arguments.size()) {
    NotYet(at, "this higher-order call");
    return;
  }
  std::optional<merk::Determinism> determinism{
      MerkDeterminism(goal.modes->determinism)};
  if (!determinism) {
    NotYet(at,
        std::string{semantics::DeterminismName(goal.modes->determinism)} +
            " closures");
    return;
  }
  // Every argument is a word, one of a type with no values the word 0.
  auto word{merk::WordType::get(&context_)};
  llvm::SmallVector<mlir::Value> inputs;
  llvm::SmallVector<BodyVar> outputs;
  llvm::SmallVector<std::pair<BodyVar, mlir::Value>> bound;
  for (auto [argument, mode] : llvm::zip(goal.arguments, modes)) {
    const semantics::Type *type{body.vars[argument].type};
    std::optional<Represented> represented{Represent(type)};
    if (!represented) {
      NotYet(at, "values of type " + semantics::TypeToString(*type));
      return;
    }
    if (IsOutput(mode)) {
      outputs.push_back(argument);
      // One bound already is tested against the call's.
      if (mlir::Value value{ValueOf(argument)}; value && !represented->dummy) {
        bound.push_back({argument, value});
      }
    } else {
      inputs.push_back(represented->dummy
              ? ZeroWord(loc)
              : Coerce(ValueOf(argument), word, loc));
    }
  }
  auto call{merk::CallClosureOp::create(builder_, loc,
      llvm::SmallVector<mlir::Type>(outputs.size(), word), closure, inputs,
      *determinism)};
  if (goal.modes->determinism == parser::Determinism::Erroneous ||
      goal.modes->determinism == parser::Determinism::Failure) {
    merk::FailOp::create(builder_, loc);
    return;
  }
  for (auto [var, result] : llvm::zip_equal(outputs, call.getResults())) {
    SetFromWord(var, result, body, loc);
  }
  for (auto [var, value] : bound) {
    mlir::Value output{ValueOf(var)};
    values_[var] = value;
    Test(goal, value, output, body.vars[var].type);
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
    Disjunction(goal, body);
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
  case BodyKind::HOCall:
    HigherOrderCall(goal, body);
    return;
  case BodyKind::Unify:
    if (goal.switchTest) {
      // The switch's case has tested it; a constructor with arguments is
      // taken apart.
      if (!goal.isVar && !goal.arguments.empty()) {
        TermUnify(goal, body);
      }
      return;
    }
    if (goal.lambda) {
      LambdaClosure(goal, body);
      return;
    }
    if (!goal.isVar && goal.unify == UnifyKind::Construct && goal.modes &&
        goal.modes->callee &&
        body.vars[goal.left].type->kind == semantics::TypeKind::HigherOrder) {
      NameClosure(goal, body);
      return;
    }
    if (goal.isVar && goal.unify == UnifyKind::Test) {
      // X = X succeeds, even for a NaN, as with mmc.
      if (goal.left != goal.var) {
        Test(goal, ValueOf(goal.left), ValueOf(goal.var),
            body.vars[goal.left].type);
      }
      return;
    }
    if (!goal.isVar && goal.unify != UnifyKind::Call &&
        goal.unify != UnifyKind::Assign &&
        mlir::isa_and_nonnull<merk::TermType>(
            Represent(body.vars[goal.left].type).value_or(Represented{})
                .type)) {
      TermUnify(goal, body);
      return;
    }
    if (!goal.isVar && goal.arguments.empty() && goal.source &&
        (goal.unify == UnifyKind::Deconstruct ||
            goal.unify == UnifyKind::Test)) {
      if (mlir::Value constant{Constant(goal, body)}) {
        Test(goal, ValueOf(goal.left), constant, body.vars[goal.left].type);
      }
      return;
    }
    if (goal.isVar && goal.unify == UnifyKind::Assign) {
      // X = Y: the variable assigned is the other's value; one of a type
      // with no values has none.
      if (values_.count(goal.var)) {
        values_[goal.left] = values_.lookup(goal.var);
      } else if (mlir::Value value{values_.lookup(goal.left)}) {
        values_[goal.var] = value;
      } else if (mlir::Value word{values_.lookup(kDummyWordVar + goal.var)}) {
        // The word of one of a type with no values, in a body that gives
        // words, goes with it.
        values_[kDummyWordVar + goal.left] = word;
      } else if (mlir::Value word{values_.lookup(kDummyWordVar + goal.left)}) {
        values_[kDummyWordVar + goal.var] = word;
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
  std::optional<llvm::SmallVector<mlir::Type>> hiddenInputs{
      HiddenInputs(procedure, at)};
  if (!hiddenInputs) {
    return;
  }
  if (!signature->constraints.empty()) {
    NotYet(at, "foreign_procs with type class constraints");
    return;
  }
  if (!HiddenOutputs(procedure).empty()) {
    NotYet(at, "foreign_procs with existential types");
    return;
  }
  std::optional<unsigned> hidden{static_cast<unsigned>(hiddenInputs->size())};
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
  // MR_Word set to 0. A type variable's value is an MR_Word, and its
  // type_info is one too, the parameter TypeInfo_for_T, first, as in mmc:
  // the manual does not describe them.
  merk::Determinism determinism{proc.getDeterminism()};
  bool canFail{determinism == merk::Determinism::Semidet ||
      determinism == merk::Determinism::Failure ||
      determinism == merk::Determinism::CcNondet};
  std::string name{ForeignName(procedure)};
  std::string parameters, locals, stores;
  if (canFail) {
    locals += "  MR_bool SUCCESS_INDICATOR = MR_TRUE;\n";
  }
  for (unsigned var{0}; var < *hidden; ++var) {
    parameters += (parameters.empty() ? "" : ", ");
    parameters += "MR_Word TypeInfo_for_" + signature->varNames[var].str();
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
  // A term is an MR_Word or MR_Tuple in C whatever its layout, so it crosses
  // as its word: a notag type's float would otherwise be in a float register.
  // So does a type_info.
  auto word{merk::WordType::get(&context_)};
  llvm::SmallVector<mlir::Value> inputs;
  for (mlir::Value argument : entry->getArguments()) {
    inputs.push_back(
        mlir::isa<merk::TermType, merk::TypeInfoType>(argument.getType())
            ? mlir::Value{merk::ToWordOp::create(builder_, loc, word, argument)}
            : argument);
  }
  llvm::SmallVector<mlir::Type> results;
  if (canFail) {
    results.push_back(builder_.getI1Type());
  }
  for (mlir::Type type : proc.getFunctionType().getResults()) {
    results.push_back(mlir::isa<merk::TermType>(type) ? word : type);
  }
  auto call{merk::ForeignCallOp::create(
      builder_, loc, results, name, inputs, canFail)};
  mlir::ValueRange solution{call.getResults()};
  if (canFail) {
    merk::GuardOp::create(builder_, loc, solution.front());
    solution = solution.drop_front();
  }
  llvm::SmallVector<mlir::Value> values;
  for (auto [value, type] :
      llvm::zip(solution, proc.getFunctionType().getResults())) {
    values.push_back(mlir::isa<merk::TermType>(type)
            ? mlir::Value{merk::FromWordOp::create(builder_, loc, type, value)}
            : value);
  }
  merk::SucceedOp::create(builder_, loc, values);
}

void MerkGenerator::TermUnify(const Body &goal, const ProcedureBody &body) {
  mlir::Location loc{Location(goal.tokens)};
  parser::CharBlock at{Where(goal.tokens)};
  const semantics::Type *type{body.vars[goal.left].type};
  auto term{mlir::cast<merk::TermType>(Represent(type)->type)};
  unsigned arity{static_cast<unsigned>(goal.arguments.size())};
  auto functor{merk::FunctorAttr::get(&context_, goal.functor.name, arity)};
  // An existentially typed constructor's cell has its hidden fields first,
  // and its arguments' types are its own, which the values' are coerced to.
  HiddenFields hidden;
  std::optional<llvm::SmallVector<mlir::Type>> cell;
  if (const semantics::ConstructorTypes *constructor{
          ConstructorOf(type, goal.functor.name, arity)}) {
    cell = CellFields(type, *constructor, hidden);
    if (!cell) {
      NotYet(at, "this unification");
      return;
    }
  }
  // What polymorphism analysis has a construction pass, and a deconstruction
  // give, by the cell's hidden fields.
  const semantics::PassedInfos *passed{nullptr};
  if (!hidden.types.empty() && current_) {
    if (auto it{current_->goals.find(&goal)}; it != current_->goals.end()) {
      passed = &it->second;
    }
  }
  if (passed &&
      passed->typeInfos.size() + passed->dictionaries.size() !=
          hidden.types.size()) {
    NotYet(at, "this unification's type_infos");
    return;
  }
  if (goal.unify == UnifyKind::Construct) {
    llvm::SmallVector<mlir::Value> arguments;
    if (!hidden.types.empty()) {
      if (!passed) {
        NotYet(at, "this unification's type_infos");
        return;
      }
      for (const semantics::TypeInfo *info : passed->typeInfos) {
        arguments.push_back(TypeInfoValue(*info, loc, at));
      }
      for (const semantics::Dictionary *dictionary : passed->dictionaries) {
        arguments.push_back(DictionaryValue(*dictionary, loc, at));
      }
      if (llvm::is_contained(arguments, mlir::Value{})) {
        failed_ = true;
        return;
      }
    }
    for (BodyVar argument : goal.arguments) {
      mlir::Value value{ValueOf(argument)};
      if (!value) {
        // A term with free or dummy arguments.
        NotYet(at, "this unification");
        return;
      }
      if (!hidden.types.empty()) {
        value = Coerce(value, (*cell)[arguments.size()], loc);
      }
      arguments.push_back(value);
    }
    values_[goal.left] =
        merk::ConstructOp::create(builder_, loc, term, functor, arguments);
    return;
  }
  mlir::Value value{ValueOf(goal.left)};
  if (!goal.switchTest &&
      (goal.canFail || goal.unify == UnifyKind::Test)) {
    merk::GuardOp::create(builder_, loc,
        merk::HasFunctorOp::create(builder_, loc, value, functor));
  }
  if (goal.arguments.empty()) {
    return;
  }
  // The arguments are new variables: mmc's form tests any that was bound
  // with a unification after it.
  llvm::SmallVector<mlir::Type> fields{hidden.types};
  for (BodyVar argument : goal.arguments) {
    std::optional<Represented> represented{
        Represent(body.vars[argument].type)};
    if (goal.unify == UnifyKind::Test || !represented || represented->dummy ||
        values_.count(argument)) {
      NotYet(at, "this unification");
      return;
    }
    fields.push_back(represented->type);
  }
  if (!hidden.types.empty()) {
    fields = *cell;
  }
  auto deconstruct{
      merk::DeconstructOp::create(builder_, loc, fields, value, functor)};
  mlir::ValueRange results{deconstruct.getFields()};
  // The type_infos and typeclass_infos it gives, which a deconstruction whose
  // arguments nothing uses gives none of.
  if (passed) {
    unsigned field{0};
    for (const semantics::TypeInfo *info : passed->typeInfos) {
      Give(info, results[field++]);
    }
    for (const semantics::Dictionary *dictionary : passed->dictionaries) {
      Give(dictionary, results[field++]);
    }
  }
  for (auto [argument, field] :
      llvm::zip(goal.arguments, results.drop_front(hidden.types.size()))) {
    values_[argument] = hidden.types.empty()
        ? field
        : Coerce(field, Represent(body.vars[argument].type)->type, loc);
  }
}

bool MerkGenerator::Comparable(mlir::Type type, llvm::StringSet<> &seen) {
  if (mlir::isa<mlir::IntegerType, mlir::FloatType, merk::StringType,
          merk::WordType, merk::ClosureType>(type)) {
    return true;
  }
  if (auto array{mlir::dyn_cast<merk::ArrayType>(type)}) {
    return Comparable(array.getElement(), seen);
  }
  auto term{mlir::dyn_cast<merk::TermType>(type)};
  if (!term) {
    return false;
  }
  if (!seen.insert(term.getName()).second) {
    return true;
  }
  const semantics::Type *semantic{terms_.lookup(term.getName())};
  for (const semantics::ConstructorTypes &constructor :
      *modes_.ops().ConstructorsOf(semantic)) {
    for (const semantics::Type *argument : constructor.arguments) {
      if (!Comparable(Represent(argument)->type, seen)) {
        return false;
      }
    }
  }
  return true;
}

merk::ProcOp MerkGenerator::UnifyProc(merk::TermType term) {
  if (merk::ProcOp proc{unifyProcs_.lookup(term.getName())}) {
    return proc;
  }
  mlir::Location loc{builder_.getUnknownLoc()};
  const semantics::Type *type{terms_.lookup(term.getName())};
  const std::vector<semantics::ConstructorTypes> &constructors{
      *modes_.ops().ConstructorsOf(type)};
  mlir::OpBuilder::InsertionGuard guard{builder_};
  builder_.setInsertionPointToEnd(module_.getBody());
  // It takes a type_info for each type variable of its type, first.
  llvm::SmallVector<unsigned> variables{VariablesOf(*type)};
  llvm::SmallVector<mlir::Type> inputs(
      variables.size(), merk::TypeInfoType::get(&context_));
  inputs.append({term, term});
  auto proc{merk::ProcOp::create(builder_, loc,
      ("unify." + term.getName()).str(), builder_.getFunctionType(inputs, {}),
      merk::Determinism::Semidet)};
  proc.setPrivate();
  // Recorded first, for the calls of it on its terms' arguments.
  unifyProcs_[term.getName()] = proc;
  mlir::Block *entry{proc.addEntryBlock()};
  builder_.setInsertionPointToStart(entry);
  auto saved{std::move(typeInfos_)};
  typeInfos_.clear();
  for (auto [i, var] : llvm::enumerate(variables)) {
    typeInfos_[var] = entry->getArgument(i);
  }
  mlir::Value x{entry->getArgument(variables.size())};
  mlir::Value y{entry->getArgument(variables.size() + 1)};
  llvm::SmallVector<mlir::Attribute> cases;
  for (const semantics::ConstructorTypes &constructor : constructors) {
    cases.push_back(builder_.getArrayAttr({merk::FunctorAttr::get(&context_,
        constructor.functor.name, constructor.functor.arity)}));
  }
  auto op{merk::SwitchOp::create(builder_, loc, mlir::TypeRange{}, x,
      builder_.getArrayAttr(cases), /*canFail=*/false)};
  for (auto [constructor, region, c] :
      llvm::zip(constructors, op.getCaseRegions(), cases)) {
    builder_.setInsertionPointToStart(&region.emplaceBlock());
    auto functor{mlir::cast<merk::FunctorAttr>(
        mlir::cast<mlir::ArrayAttr>(c)[0])};
    merk::GuardOp::create(builder_, loc,
        merk::HasFunctorOp::create(builder_, loc, y, functor));
    if (!constructor.arguments.empty()) {
      HiddenFields hidden;
      llvm::SmallVector<mlir::Type> fields{
          *CellFields(type, constructor, hidden)};
      auto xs{merk::DeconstructOp::create(builder_, loc, fields, x, functor)};
      auto ys{merk::DeconstructOp::create(builder_, loc, fields, y, functor)};
      // As mmc's, terms of an existentially typed constructor are equal only
      // if their types are the same.
      for (auto [k, place] : llvm::enumerate(hidden.places)) {
        mlir::Value a{HiddenTypeInfo(xs.getFields(), place, loc)};
        mlir::Value b{HiddenTypeInfo(ys.getFields(), place, loc)};
        merk::GuardOp::create(builder_, loc,
            mlir::arith::CmpIOp::create(builder_, loc,
                mlir::arith::CmpIPredicate::eq,
                merk::CompareTypeInfoOp::create(
                    builder_, loc, builder_.getI32Type(), a, b),
                IntegerConstant(builder_.getI32Type(), 0, loc)));
        typeInfos_[ExistentialBase(*type) + static_cast<unsigned>(k)] = a;
      }
      std::size_t first{hidden.types.size()};
      for (auto [a, b, argument] :
          llvm::zip(xs.getFields().drop_front(first),
              ys.getFields().drop_front(first),
              ArgumentsOf(type, constructor))) {
        Equal(a, b, argument, loc);
      }
    }
    merk::YieldOp::create(builder_, loc, mlir::ValueRange{});
  }
  builder_.setInsertionPointAfter(op);
  merk::SucceedOp::create(builder_, loc, mlir::ValueRange{});
  typeInfos_ = std::move(saved);
  return proc;
}

llvm::SmallVector<unsigned> MerkGenerator::VariablesOf(
    const semantics::Type &type) {
  llvm::SmallVector<unsigned> variables;
  semantics::ForEachVariable(type, [&](unsigned var) {
    if (!llvm::is_contained(variables, var)) {
      variables.push_back(var);
    }
  });
  return variables;
}

mlir::Value MerkGenerator::ArrayElement(
    mlir::Value array, mlir::Value index, mlir::Location loc) {
  return merk::ArrayGetOp::create(builder_, loc,
      mlir::cast<merk::ArrayType>(array.getType()).getElement(), array, index);
}

mlir::Value MerkGenerator::ArrayWithElement(mlir::Value array,
    mlir::Value index, mlir::Value element, mlir::Location loc) {
  return merk::ArraySetOp::create(
      builder_, loc, array.getType(), array, index, element)
      .getResult();
}

mlir::Value MerkGenerator::ArraySize(mlir::Value array, mlir::Location loc) {
  return merk::ArraySizeOp::create(builder_, loc, builder_.getI64Type(), array);
}

// As an unsigned comparison, a negative index is out of range too.
mlir::Value MerkGenerator::InArrayBounds(
    mlir::Value array, mlir::Value index, mlir::Location loc) {
  return CompareInts(
      mlir::arith::CmpIPredicate::ult, index, ArraySize(array, loc), loc);
}

mlir::Value MerkGenerator::CompareInts(mlir::arith::CmpIPredicate predicate,
    mlir::Value a, mlir::Value b, mlir::Location loc) {
  return mlir::arith::CmpIOp::create(builder_, loc, predicate, a, b);
}

bool MerkGenerator::GiveOutputs(
    llvm::ArrayRef<BodyVar> out, std::initializer_list<mlir::Value> values) {
  for (auto [var, value] : llvm::zip(out, values)) {
    values_[var] = value;
  }
  return true;
}

// Whether a call has that many inputs and outputs.
static bool HasShape(llvm::ArrayRef<mlir::Value> in, llvm::ArrayRef<BodyVar> out,
    unsigned inputs, unsigned outputs) {
  return in.size() == inputs && out.size() == outputs;
}

// An array's elements are words whatever its type, so one procedure serves.
merk::ProcOp MerkGenerator::UnifyArrayProc() {
  if (unifyArray_) {
    return unifyArray_;
  }
  mlir::Location loc{builder_.getUnknownLoc()};
  auto info{merk::TypeInfoType::get(&context_)};
  auto word{merk::WordType::get(&context_)};
  auto array{merk::ArrayType::get(&context_, word)};
  auto i64{builder_.getI64Type()};
  mlir::OpBuilder::InsertionGuard guard{builder_};
  builder_.setInsertionPointToEnd(module_.getBody());
  // From an index to the size: each element's test, then the rest's, as a
  // tail call.
  auto from{merk::ProcOp::create(builder_, loc, "unify.array.from",
      builder_.getFunctionType({info, array, array, i64, i64}, {}),
      merk::Determinism::Semidet)};
  from.setPrivate();
  {
    mlir::Block *entry{from.addEntryBlock()};
    builder_.setInsertionPointToStart(entry);
    mlir::Value i{entry->getArgument(3)}, n{entry->getArgument(4)};
    auto ite{merk::IteOp::create(builder_, loc, mlir::TypeRange{})};
    builder_.setInsertionPointToStart(&ite.getCondition().emplaceBlock());
    merk::GuardOp::create(builder_, loc,
        mlir::arith::CmpIOp::create(
            builder_, loc, mlir::arith::CmpIPredicate::slt, i, n));
    merk::YieldOp::create(builder_, loc, mlir::ValueRange{});
    builder_.setInsertionPointToStart(&ite.getThenRegion().emplaceBlock());
    merk::GuardOp::create(builder_, loc,
        merk::UnifyWordOp::create(builder_, loc, builder_.getI1Type(),
            entry->getArgument(0), ArrayElement(entry->getArgument(1), i, loc),
            ArrayElement(entry->getArgument(2), i, loc)));
    mlir::Value next{mlir::arith::AddIOp::create(
        builder_, loc, i, IntegerConstant(i64, 1, loc))};
    merk::CallOp::create(builder_, loc, mlir::TypeRange{}, from.getSymName(),
        mlir::ValueRange{entry->getArgument(0), entry->getArgument(1),
            entry->getArgument(2), next, n});
    merk::YieldOp::create(builder_, loc, mlir::ValueRange{});
    builder_.setInsertionPointToStart(&ite.getElseRegion().emplaceBlock());
    merk::YieldOp::create(builder_, loc, mlir::ValueRange{});
    builder_.setInsertionPointAfter(ite);
    merk::SucceedOp::create(builder_, loc, mlir::ValueRange{});
  }
  builder_.setInsertionPointToEnd(module_.getBody());
  unifyArray_ = merk::ProcOp::create(builder_, loc, "unify.array",
      builder_.getFunctionType({info, array, array}, {}),
      merk::Determinism::Semidet);
  unifyArray_.setPrivate();
  mlir::Block *entry{unifyArray_.addEntryBlock()};
  builder_.setInsertionPointToStart(entry);
  mlir::Value size{merk::ArraySizeOp::create(
      builder_, loc, i64, entry->getArgument(1))};
  merk::GuardOp::create(builder_, loc,
      mlir::arith::CmpIOp::create(builder_, loc, mlir::arith::CmpIPredicate::eq,
          size,
          merk::ArraySizeOp::create(builder_, loc, i64, entry->getArgument(2))));
  merk::CallOp::create(builder_, loc, mlir::TypeRange{}, from.getSymName(),
      mlir::ValueRange{entry->getArgument(0), entry->getArgument(1),
          entry->getArgument(2), IntegerConstant(i64, 0, loc), size});
  merk::SucceedOp::create(builder_, loc, mlir::ValueRange{});
  return unifyArray_;
}

void MerkGenerator::Equal(mlir::Value x, mlir::Value y,
    const semantics::Type *semantic, mlir::Location loc) {
  mlir::Type type{x.getType()};
  if (mlir::isa<merk::WordType>(type) ||
      semantic->kind == semantics::TypeKind::Variable) {
    mlir::Value info{typeInfos_.lookup(semantic->var)};
    if (!info) {
      NotYet(equalAt_, "a unification of a type variable without a type_info");
      return;
    }
    merk::GuardOp::create(builder_, loc,
        merk::UnifyWordOp::create(
            builder_, loc, builder_.getI1Type(), info, x, y));
  } else if (mlir::isa<merk::ClosureType>(type)) {
    // Through the descriptor of the higher-order type, whose unification
    // procedure reports the error, as mmc's does.
    mlir::Value info{TypeInfoOfType(semantic, loc, equalAt_)};
    if (!info) {
      failed_ = true;
      return;
    }
    auto word{merk::WordType::get(&context_)};
    merk::GuardOp::create(builder_, loc,
        merk::UnifyWordOp::create(builder_, loc, builder_.getI1Type(), info,
            Coerce(x, word, loc), Coerce(y, word, loc)));
  } else if (mlir::isa<merk::StringType>(type)) {
    merk::GuardOp::create(builder_, loc,
        merk::StringEqualOp::create(
            builder_, loc, builder_.getI1Type(), x, y));
  } else if (mlir::isa<merk::ArrayType>(type)) {
    mlir::Value info{
        TypeInfoOfType(semantic->arguments[0], loc, equalAt_)};
    if (!info) {
      failed_ = true;
      return;
    }
    auto words{merk::ArrayType::get(&context_, merk::WordType::get(&context_))};
    merk::CallOp::create(builder_, loc, mlir::TypeRange{},
        UnifyArrayProc().getSymName(),
        mlir::ValueRange{info, Coerce(x, words, loc), Coerce(y, words, loc)});
  } else if (mlir::isa<mlir::IntegerType>(type)) {
    merk::GuardOp::create(builder_, loc,
        mlir::arith::CmpIOp::create(
            builder_, loc, mlir::arith::CmpIPredicate::eq, x, y));
  } else if (mlir::isa<mlir::FloatType>(type)) {
    // As mmc's ==, a NaN is equal to nothing.
    merk::GuardOp::create(builder_, loc,
        mlir::arith::CmpFOp::create(
            builder_, loc, mlir::arith::CmpFPredicate::OEQ, x, y));
  } else {
    llvm::SmallVector<mlir::Value> inputs;
    for (unsigned var : VariablesOf(*semantic)) {
      mlir::Value info{typeInfos_.lookup(var)};
      if (!info) {
        NotYet(equalAt_, "a unification of a type variable without a type_info");
        return;
      }
      inputs.push_back(info);
    }
    inputs.append({x, y});
    merk::CallOp::create(builder_, loc, mlir::TypeRange{},
        UnifyProc(mlir::cast<merk::TermType>(type)).getSymName(), inputs);
  }
}

std::optional<merk::TermType> MerkGenerator::ComparisonType() {
  const semantics::ModuleSymbols *builtin{resolution_.Module("builtin")};
  if (builtin) {
    for (const semantics::Symbol *symbol :
        builtin->Lookup("comparison_result")) {
      if (symbol->kind == semantics::SymbolKind::Type) {
        std::optional<Represented> represented{
            Represent(types_.types().Named(*symbol, {}))};
        return mlir::cast<merk::TermType>(represented->type);
      }
    }
  }
  return std::nullopt;
}

mlir::Value MerkGenerator::ComparisonValue(
    llvm::StringRef functor, mlir::Location loc) {
  return merk::ConstructOp::create(builder_, loc, *ComparisonType(),
      merk::FunctorAttr::get(&context_, functor, 0), mlir::ValueRange{});
}

mlir::Value MerkGenerator::IsComparison(
    mlir::Value result, llvm::StringRef functor, mlir::Location loc) {
  return merk::HasFunctorOp::create(builder_, loc, result,
      merk::FunctorAttr::get(&context_, functor, 0));
}

mlir::Value MerkGenerator::ThreeWay(
    mlir::Value less, mlir::Value greater, mlir::Location loc) {
  return ComparisonIf(less, "<", loc, [&] {
    return ComparisonIf(
        greater, ">", loc, [&] { return ComparisonValue("=", loc); });
  });
}

mlir::Value MerkGenerator::ComparisonIf(mlir::Value condition,
    llvm::StringRef functor, mlir::Location loc,
    llvm::function_ref<mlir::Value()> otherwise) {
  auto ite{merk::IteOp::create(
      builder_, loc, mlir::TypeRange{*ComparisonType()})};
  mlir::OpBuilder::InsertionGuard guard{builder_};
  builder_.setInsertionPointToStart(&ite.getCondition().emplaceBlock());
  merk::GuardOp::create(builder_, loc, condition);
  merk::YieldOp::create(builder_, loc, mlir::ValueRange{});
  builder_.setInsertionPointToStart(&ite.getThenRegion().emplaceBlock());
  merk::YieldOp::create(
      builder_, loc, mlir::ValueRange{ComparisonValue(functor, loc)});
  builder_.setInsertionPointToStart(&ite.getElseRegion().emplaceBlock());
  merk::YieldOp::create(builder_, loc, mlir::ValueRange{otherwise()});
  return ite.getResult(0);
}

mlir::Value MerkGenerator::Compare(mlir::Value x, mlir::Value y,
    const semantics::Type *semantic, bool generic, mlir::Location loc) {
  namespace arith = mlir::arith;
  mlir::Type type{x.getType()};
  mlir::Type result{*ComparisonType()};
  if (mlir::isa<merk::WordType>(type) ||
      semantic->kind == semantics::TypeKind::Variable) {
    mlir::Value info{typeInfos_.lookup(semantic->var)};
    if (!info) {
      NotYet(equalAt_, "a comparison of a type variable without a type_info");
      return nullptr;
    }
    return merk::CompareWordOp::create(builder_, loc, result, info, x, y);
  }
  if (mlir::isa<merk::ClosureType>(type)) {
    // As a unification of closures.
    mlir::Value info{TypeInfoOfType(semantic, loc, equalAt_)};
    if (!info) {
      failed_ = true;
      return nullptr;
    }
    auto word{merk::WordType::get(&context_)};
    return merk::CompareWordOp::create(builder_, loc, result, info,
        Coerce(x, word, loc), Coerce(y, word, loc));
  }
  if (mlir::isa<merk::StringType>(type)) {
    mlir::Value order{merk::StringCompareOp::create(
        builder_, loc, builder_.getI32Type(), x, y)};
    mlir::Value zero{IntegerConstant(builder_.getI32Type(), 0, loc)};
    return ThreeWay(arith::CmpIOp::create(
                        builder_, loc, arith::CmpIPredicate::slt, order, zero),
        arith::CmpIOp::create(
            builder_, loc, arith::CmpIPredicate::sgt, order, zero),
        loc);
  }
  if (mlir::isa<merk::ArrayType>(type)) {
    mlir::Value info{TypeInfoOfType(semantic->arguments[0], loc, equalAt_)};
    if (!info) {
      failed_ = true;
      return nullptr;
    }
    auto words{merk::ArrayType::get(&context_, merk::WordType::get(&context_))};
    return merk::CallOp::create(builder_, loc, mlir::TypeRange{result},
        CompareArrayProc().getSymName(),
        mlir::ValueRange{info, Coerce(x, words, loc), Coerce(y, words, loc)})
        .getResult(0);
  }
  if (mlir::isa<mlir::IntegerType>(type)) {
    // A character's code point has no sign either.
    llvm::StringRef name{semantic->constructor->name};
    bool isUnsigned{name.starts_with("uint") || name == "character"};
    return ThreeWay(
        arith::CmpIOp::create(builder_, loc,
            isUnsigned ? arith::CmpIPredicate::ult : arith::CmpIPredicate::slt,
            x, y),
        arith::CmpIOp::create(builder_, loc,
            isUnsigned ? arith::CmpIPredicate::ugt : arith::CmpIPredicate::sgt,
            x, y),
        loc);
  }
  if (mlir::isa<mlir::FloatType>(type)) {
    if (generic) {
      merk::ThrowIfOp::create(builder_, loc,
          arith::CmpFOp::create(
              builder_, loc, arith::CmpFPredicate::UNO, x, y),
          "incomparable floats in compare/3");
    }
    // As mmc's for a float it knows is one, a NaN is equal to everything.
    return ThreeWay(
        arith::CmpFOp::create(builder_, loc, arith::CmpFPredicate::OLT, x, y),
        arith::CmpFOp::create(builder_, loc, arith::CmpFPredicate::OGT, x, y),
        loc);
  }
  llvm::SmallVector<mlir::Value> inputs;
  for (unsigned var : VariablesOf(*semantic)) {
    mlir::Value info{typeInfos_.lookup(var)};
    if (!info) {
      NotYet(equalAt_, "a comparison of a type variable without a type_info");
      return nullptr;
    }
    inputs.push_back(info);
  }
  inputs.append({x, y});
  return merk::CallOp::create(builder_, loc, mlir::TypeRange{result},
      CompareProc(mlir::cast<merk::TermType>(type)).getSymName(), inputs)
      .getResult(0);
}

void MerkGenerator::SucceedIf(
    mlir::Value condition, mlir::Value value, mlir::Location loc) {
  auto ite{merk::IteOp::create(builder_, loc, mlir::TypeRange{})};
  mlir::OpBuilder::InsertionGuard guard{builder_};
  builder_.setInsertionPointToStart(&ite.getCondition().emplaceBlock());
  merk::GuardOp::create(builder_, loc, condition);
  merk::YieldOp::create(builder_, loc, mlir::ValueRange{});
  builder_.setInsertionPointToStart(&ite.getThenRegion().emplaceBlock());
  merk::SucceedOp::create(builder_, loc, mlir::ValueRange{value});
  builder_.setInsertionPointToStart(&ite.getElseRegion().emplaceBlock());
  merk::YieldOp::create(builder_, loc, mlir::ValueRange{});
}

mlir::Value MerkGenerator::ConstructorIndex(
    mlir::Value term, mlir::ArrayAttr cases, mlir::Location loc) {
  mlir::Type i64{builder_.getI64Type()};
  auto op{merk::SwitchOp::create(builder_, loc, mlir::TypeRange{i64}, term,
      cases, /*canFail=*/false)};
  mlir::OpBuilder::InsertionGuard guard{builder_};
  for (auto [i, region] : llvm::enumerate(op.getCaseRegions())) {
    builder_.setInsertionPointToStart(&region.emplaceBlock());
    merk::YieldOp::create(
        builder_, loc, mlir::ValueRange{IntegerConstant(i64, i, loc)});
  }
  return op.getResult(0);
}

merk::ProcOp MerkGenerator::CompareProc(merk::TermType term) {
  if (merk::ProcOp proc{compareProcs_.lookup(term.getName())}) {
    return proc;
  }
  mlir::Location loc{builder_.getUnknownLoc()};
  auto i64{builder_.getI64Type()};
  const semantics::Type *type{terms_.lookup(term.getName())};
  const std::vector<semantics::ConstructorTypes> &constructors{
      *modes_.ops().ConstructorsOf(type)};
  // Which arguments the type declares as type variables, whichever its
  // instance. A tuple's all are.
  std::vector<std::vector<bool>> declaredVariable;
  semantics::TypeContext &context{types_.types()};
  if (type->kind == semantics::TypeKind::Tuple) {
    declaredVariable.emplace_back(type->arguments.size(), true);
  } else {
    llvm::SmallVector<const semantics::Type *> variables;
    for (unsigned i{0}; i < type->arguments.size(); ++i) {
      variables.push_back(context.Variable(i));
    }
    for (const semantics::ConstructorTypes &constructor :
        *modes_.ops().ConstructorsOf(
            context.Named(*type->constructor, variables))) {
      std::vector<bool> &declared{declaredVariable.emplace_back()};
      for (const semantics::Type *argument : constructor.arguments) {
        declared.push_back(argument->kind == semantics::TypeKind::Variable);
      }
    }
  }
  mlir::Type result{*ComparisonType()};
  mlir::OpBuilder::InsertionGuard guard{builder_};
  builder_.setInsertionPointToEnd(module_.getBody());
  llvm::SmallVector<unsigned> variables{VariablesOf(*type)};
  llvm::SmallVector<mlir::Type> inputs(
      variables.size(), merk::TypeInfoType::get(&context_));
  inputs.append({term, term});
  auto proc{merk::ProcOp::create(builder_, loc,
      ("compare." + term.getName()).str(),
      builder_.getFunctionType(inputs, result), merk::Determinism::Det)};
  proc.setPrivate();
  // Recorded first, for the calls of it on its terms' arguments.
  compareProcs_[term.getName()] = proc;
  mlir::Block *entry{proc.addEntryBlock()};
  builder_.setInsertionPointToStart(entry);
  auto saved{std::move(typeInfos_)};
  typeInfos_.clear();
  for (auto [i, var] : llvm::enumerate(variables)) {
    typeInfos_[var] = entry->getArgument(i);
  }
  mlir::Value x{entry->getArgument(variables.size())};
  mlir::Value y{entry->getArgument(variables.size() + 1)};
  llvm::SmallVector<mlir::Attribute> cases;
  for (const semantics::ConstructorTypes &constructor : constructors) {
    cases.push_back(builder_.getArrayAttr({merk::FunctorAttr::get(&context_,
        constructor.functor.name, constructor.functor.arity)}));
  }
  // Constructors in the order declared, by their position in it, which a
  // switch gives whatever their layout.
  if (constructors.size() > 1) {
    mlir::ArrayAttr casesAttr{builder_.getArrayAttr(cases)};
    mlir::Value xIndex{ConstructorIndex(x, casesAttr, loc)};
    mlir::Value yIndex{ConstructorIndex(y, casesAttr, loc)};
    SucceedIf(mlir::arith::CmpIOp::create(builder_, loc,
                  mlir::arith::CmpIPredicate::slt, xIndex, yIndex),
        ComparisonValue("<", loc), loc);
    SucceedIf(mlir::arith::CmpIOp::create(builder_, loc,
                  mlir::arith::CmpIPredicate::sgt, xIndex, yIndex),
        ComparisonValue(">", loc), loc);
  }
  bool anyArguments{llvm::any_of(
      constructors, [](const semantics::ConstructorTypes &constructor) {
        return !constructor.arguments.empty();
      })};
  if (anyArguments) {
    auto op{merk::SwitchOp::create(builder_, loc, mlir::TypeRange{}, x,
        builder_.getArrayAttr(cases), /*canFail=*/false)};
    for (auto [c, constructor, region, attribute] :
        llvm::enumerate(constructors, op.getCaseRegions(), cases)) {
      builder_.setInsertionPointToStart(&region.emplaceBlock());
      bool ended{false};
      if (!constructor.arguments.empty()) {
        auto functor{mlir::cast<merk::FunctorAttr>(
            mlir::cast<mlir::ArrayAttr>(attribute)[0])};
        HiddenFields hidden;
        llvm::SmallVector<mlir::Type> fields{
            *CellFields(type, constructor, hidden)};
        auto xs{merk::DeconstructOp::create(builder_, loc, fields, x, functor)};
        auto ys{merk::DeconstructOp::create(builder_, loc, fields, y, functor)};
        std::size_t first{hidden.types.size()};
        llvm::SmallVector<bool> compared(hidden.places.size(), false);
        unsigned base{ExistentialBase(*type)};
        // The first argument is the most significant, and the last one's
        // comparison, as a list's tail, is in tail position.
        for (auto [i, a, b, argument] : llvm::enumerate(
                 xs.getFields().drop_front(first),
                 ys.getFields().drop_front(first),
                 ArgumentsOf(type, constructor))) {
          // As mmc's, an argument of an existentially typed constructor is
          // compared by its type first: by the types of the existentially
          // quantified variables it has, that no argument before it has.
          for (unsigned var : VariablesOf(*argument)) {
            if (var < base || compared[var - base]) {
              continue;
            }
            compared[var - base] = true;
            std::pair<unsigned, unsigned> place{hidden.places[var - base]};
            mlir::Value p{HiddenTypeInfo(xs.getFields(), place, loc)};
            mlir::Value q{HiddenTypeInfo(ys.getFields(), place, loc)};
            mlir::Value types{merk::CompareTypeInfoOp::create(
                builder_, loc, builder_.getI32Type(), p, q)};
            mlir::Value zero{IntegerConstant(builder_.getI32Type(), 0, loc)};
            SucceedIf(mlir::arith::CmpIOp::create(builder_, loc,
                          mlir::arith::CmpIPredicate::slt, types, zero),
                ComparisonValue("<", loc), loc);
            SucceedIf(mlir::arith::CmpIOp::create(builder_, loc,
                          mlir::arith::CmpIPredicate::sgt, types, zero),
                ComparisonValue(">", loc), loc);
            typeInfos_[var] = p;
          }
          mlir::Value order{Compare(a, b, argument,
              declaredVariable[c][i], loc)};
          if (!order) {
            break;
          }
          if (i + 1 == constructor.arguments.size()) {
            merk::SucceedOp::create(builder_, loc, mlir::ValueRange{order});
            ended = true;
          } else {
            auto ite{merk::IteOp::create(builder_, loc, mlir::TypeRange{})};
            mlir::OpBuilder::InsertionGuard inner{builder_};
            builder_.setInsertionPointToStart(&ite.getCondition().emplaceBlock());
            merk::GuardOp::create(
                builder_, loc, IsComparison(order, "=", loc));
            merk::YieldOp::create(builder_, loc, mlir::ValueRange{});
            builder_.setInsertionPointToStart(&ite.getThenRegion().emplaceBlock());
            merk::YieldOp::create(builder_, loc, mlir::ValueRange{});
            builder_.setInsertionPointToStart(&ite.getElseRegion().emplaceBlock());
            merk::SucceedOp::create(builder_, loc, mlir::ValueRange{order});
          }
        }
      }
      if (!ended) {
        merk::YieldOp::create(builder_, loc, mlir::ValueRange{});
      }
    }
    builder_.setInsertionPointAfter(op);
  }
  merk::SucceedOp::create(
      builder_, loc, mlir::ValueRange{ComparisonValue("=", loc)});
  typeInfos_ = std::move(saved);
  return proc;
}

// As mmc's, the array with fewer elements is the lesser, whatever they are.
merk::ProcOp MerkGenerator::CompareArrayProc() {
  if (compareArray_) {
    return compareArray_;
  }
  mlir::Location loc{builder_.getUnknownLoc()};
  auto info{merk::TypeInfoType::get(&context_)};
  auto word{merk::WordType::get(&context_)};
  auto array{merk::ArrayType::get(&context_, word)};
  auto i64{builder_.getI64Type()};
  mlir::Type result{*ComparisonType()};
  mlir::OpBuilder::InsertionGuard guard{builder_};
  builder_.setInsertionPointToEnd(module_.getBody());
  // From an index to the size, with the rest's as a tail call.
  auto from{merk::ProcOp::create(builder_, loc, "compare.array.from",
      builder_.getFunctionType({info, array, array, i64, i64}, result),
      merk::Determinism::Det)};
  from.setPrivate();
  {
    mlir::Block *entry{from.addEntryBlock()};
    builder_.setInsertionPointToStart(entry);
    mlir::Value i{entry->getArgument(3)}, n{entry->getArgument(4)};
    auto ite{merk::IteOp::create(builder_, loc, mlir::TypeRange{})};
    builder_.setInsertionPointToStart(&ite.getCondition().emplaceBlock());
    merk::GuardOp::create(builder_, loc,
        mlir::arith::CmpIOp::create(
            builder_, loc, mlir::arith::CmpIPredicate::slt, i, n));
    merk::YieldOp::create(builder_, loc, mlir::ValueRange{});
    builder_.setInsertionPointToStart(&ite.getThenRegion().emplaceBlock());
    mlir::Value order{merk::CompareWordOp::create(builder_, loc, result,
        entry->getArgument(0), ArrayElement(entry->getArgument(1), i, loc),
        ArrayElement(entry->getArgument(2), i, loc))};
    auto differs{merk::IteOp::create(builder_, loc, mlir::TypeRange{})};
    {
      mlir::OpBuilder::InsertionGuard inner{builder_};
      builder_.setInsertionPointToStart(&differs.getCondition().emplaceBlock());
      merk::GuardOp::create(builder_, loc, IsComparison(order, "=", loc));
      merk::YieldOp::create(builder_, loc, mlir::ValueRange{});
      builder_.setInsertionPointToStart(&differs.getThenRegion().emplaceBlock());
      merk::YieldOp::create(builder_, loc, mlir::ValueRange{});
      builder_.setInsertionPointToStart(&differs.getElseRegion().emplaceBlock());
      merk::SucceedOp::create(builder_, loc, mlir::ValueRange{order});
    }
    mlir::Value next{mlir::arith::AddIOp::create(
        builder_, loc, i, IntegerConstant(i64, 1, loc))};
    mlir::Value rest{merk::CallOp::create(builder_, loc, mlir::TypeRange{result},
        from.getSymName(),
        mlir::ValueRange{entry->getArgument(0), entry->getArgument(1),
            entry->getArgument(2), next, n})
                         .getResult(0)};
    merk::SucceedOp::create(builder_, loc, mlir::ValueRange{rest});
    builder_.setInsertionPointToStart(&ite.getElseRegion().emplaceBlock());
    merk::YieldOp::create(builder_, loc, mlir::ValueRange{});
    builder_.setInsertionPointAfter(ite);
    merk::SucceedOp::create(
        builder_, loc, mlir::ValueRange{ComparisonValue("=", loc)});
  }
  builder_.setInsertionPointToEnd(module_.getBody());
  compareArray_ = merk::ProcOp::create(builder_, loc, "compare.array",
      builder_.getFunctionType({info, array, array}, result),
      merk::Determinism::Det);
  compareArray_.setPrivate();
  mlir::Block *entry{compareArray_.addEntryBlock()};
  builder_.setInsertionPointToStart(entry);
  mlir::Value xSize{merk::ArraySizeOp::create(
      builder_, loc, i64, entry->getArgument(1))};
  mlir::Value ySize{merk::ArraySizeOp::create(
      builder_, loc, i64, entry->getArgument(2))};
  mlir::Value sizes{ThreeWay(
      mlir::arith::CmpIOp::create(
          builder_, loc, mlir::arith::CmpIPredicate::slt, xSize, ySize),
      mlir::arith::CmpIOp::create(
          builder_, loc, mlir::arith::CmpIPredicate::sgt, xSize, ySize),
      loc)};
  auto differs{merk::IteOp::create(builder_, loc, mlir::TypeRange{})};
  {
    mlir::OpBuilder::InsertionGuard inner{builder_};
    builder_.setInsertionPointToStart(&differs.getCondition().emplaceBlock());
    merk::GuardOp::create(builder_, loc, IsComparison(sizes, "=", loc));
    merk::YieldOp::create(builder_, loc, mlir::ValueRange{});
    builder_.setInsertionPointToStart(&differs.getThenRegion().emplaceBlock());
    merk::YieldOp::create(builder_, loc, mlir::ValueRange{});
    builder_.setInsertionPointToStart(&differs.getElseRegion().emplaceBlock());
    merk::SucceedOp::create(builder_, loc, mlir::ValueRange{sizes});
  }
  mlir::Value order{merk::CallOp::create(builder_, loc, mlir::TypeRange{result},
      from.getSymName(),
      mlir::ValueRange{entry->getArgument(0), entry->getArgument(1),
          entry->getArgument(2), IntegerConstant(i64, 0, loc), xSize})
                        .getResult(0)};
  merk::SucceedOp::create(builder_, loc, mlir::ValueRange{order});
  return compareArray_;
}

bool MerkGenerator::CompareBuiltin(const Body &goal, const Procedure &callee,
    llvm::ArrayRef<BodyVar> arguments, const ProcedureBody &body) {
  const semantics::Symbol &symbol{*callee.symbol};
  llvm::StringRef name{symbol.name};
  bool function{symbol.kind == semantics::SymbolKind::Function};
  BodyVar first, second;
  std::optional<BodyVar> output;
  if (function && arguments.size() == 3 &&
      (name == "ordering" || name == "array_compare")) {
    first = arguments[0];
    second = arguments[1];
    output = arguments[2];
  } else if (!function && name == "compare" && arguments.size() == 3) {
    output = arguments[0];
    first = arguments[1];
    second = arguments[2];
  } else if (!function && arguments.size() == 2 &&
      llvm::is_contained({"@<", "@=<", "@>", "@>=", "unify"}, name)) {
    first = arguments[0];
    second = arguments[1];
  } else {
    return false;
  }
  parser::CharBlock at{Where(goal.tokens)};
  mlir::Location loc{Location(goal.tokens)};
  const semantics::Type *type{body.vars[first].type};
  std::optional<Represented> represented{Represent(type)};
  if (!represented) {
    NotYet(at, "values of type " + semantics::TypeToString(*type));
    return true;
  }
  if (name == "unify") {
    if (!represented->dummy) {
      Test(goal, ValueOf(first), ValueOf(second), type);
    }
    return true;
  }
  equalAt_ = at;
  mlir::Value order;
  if (represented->dummy) {
    order = ComparisonValue("=", loc);
  } else {
    mlir::Value x{ValueOf(first)}, y{ValueOf(second)};
    llvm::StringSet<> seen;
    if (!x || !y || !Comparable(x.getType(), seen)) {
      NotYet(at, "this comparison");
      return true;
    }
    if (HasGivenType(*type)) {
      mlir::Value info{GivenTypeInfo(goal, type, loc)};
      if (!info) {
        return true;
      }
      auto word{merk::WordType::get(&context_)};
      order = merk::CompareWordOp::create(builder_, loc, *ComparisonType(), info,
          Coerce(x, word, loc), Coerce(y, word, loc));
    } else {
      order = Compare(x, y, type, /*generic=*/false, loc);
    }
    if (!order) {
      return true;
    }
  }
  if (output) {
    values_[*output] = order;
    return true;
  }
  mlir::Value holds;
  if (name == "@<" || name == "@>") {
    holds = IsComparison(order, name == "@<" ? "<" : ">", loc);
  } else {
    // Not the opposite one's.
    mlir::Value opposite{IsComparison(order, name == "@=<" ? ">" : "<", loc)};
    holds = mlir::arith::XOrIOp::create(builder_, loc, opposite,
        IntegerConstant(builder_.getI1Type(), 1, loc));
  }
  merk::GuardOp::create(builder_, loc, holds);
  return true;
}

void MerkGenerator::Test(const Body &goal, mlir::Value x, mlir::Value y,
    const semantics::Type *type) {
  llvm::StringSet<> seen;
  if (!x || !y || !Comparable(x.getType(), seen)) {
    NotYet(Where(goal.tokens), "this unification");
    return;
  }
  equalAt_ = Where(goal.tokens);
  mlir::Location loc{Location(goal.tokens)};
  if (HasGivenType(*type)) {
    mlir::Value info{GivenTypeInfo(goal, type, loc)};
    if (!info) {
      return;
    }
    auto word{merk::WordType::get(&context_)};
    merk::GuardOp::create(builder_, loc,
        merk::UnifyWordOp::create(builder_, loc, builder_.getI1Type(), info,
            Coerce(x, word, loc), Coerce(y, word, loc)));
    return;
  }
  Equal(x, y, type, loc);
}

mlir::Value MerkGenerator::GivenTypeInfo(
    const Body &goal, const semantics::Type *type, mlir::Location loc) {
  parser::CharBlock at{Where(goal.tokens)};
  if (current_) {
    if (auto it{current_->goals.find(&goal)};
        it != current_->goals.end() && !it->second.typeInfos.empty()) {
      return TypeInfoValue(*it->second.typeInfos.front(), loc, at);
    }
  }
  NotYet(at, "a type_info for type " + semantics::TypeToString(*type));
  return nullptr;
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
  // The word of a variable of a type with no values that one branch's call
  // gave, in a body that gives words, leaves with the word 0 of the others,
  // so that the call stays a tail call.
  std::vector<BodyVar> words;
  for (const Branch &branch : branches) {
    if (branch.fails) {
      continue;
    }
    for (BodyVar var : BoundSince(branch.values, before)) {
      if (IsDummyWord(var) && !llvm::is_contained(*results, var)) {
        words.push_back(var);
      }
    }
  }
  results->insert(results->end(), words.begin(), words.end());
  llvm::sort(*results);
  results->erase(std::unique(results->begin(), results->end()), results->end());
  llvm::SmallVector<mlir::Type> types;
  for (const Branch &branch : branches) {
    if (branch.fails) {
      continue;
    }
    mlir::OpBuilder::InsertionGuard guard{builder_};
    builder_.setInsertionPointToEnd(branch.block);
    llvm::SmallVector<mlir::Value> values;
    for (BodyVar var : *results) {
      mlir::Value value{branch.values.lookup(var)};
      values.push_back(value || !IsDummyWord(var) ? value : ZeroWord(loc));
    }
    if (types.empty() && !values.empty()) {
      for (mlir::Value value : values) {
        types.push_back(value.getType());
      }
    }
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
  Ite(goal.tokens, *goal.goals[0], goal.goals[1],
      [&] { Goal(*goal.goals[2], body); }, body);
}

// A disjunction with at most one solution as the goals around it see it,
// or in a context where only its first solution is wanted: the branches in
// turn, as if-then-elses, each tried when the one before it fails. One
// that may have to give its next solution, when a goal after it fails, is
// nondeterministic.
void MerkGenerator::Disjunction(const Body &goal, const ProcedureBody &body) {
  auto determinism{body.goals.find(&goal)};
  if (determinism == body.goals.end() ||
      llvm::is_contained({parser::Determinism::Multi,
                             parser::Determinism::Nondet},
          determinism->second)) {
    NotYet(Where(goal.tokens), "nondeterministic disjunctions");
    return;
  }
  DisjunctionFrom(goal, 0, body);
}

void MerkGenerator::DisjunctionFrom(
    const Body &goal, std::size_t next, const ProcedureBody &body) {
  if (next + 1 == goal.goals.size()) {
    Goal(*goal.goals[next], body);
    return;
  }
  Ite(goal.tokens, *goal.goals[next], nullptr,
      [&] { DisjunctionFrom(goal, next + 1, body); }, body);
}

// if condition then `then` (or true) else otherwise.
void MerkGenerator::Ite(semantics::TokenRange tokens, const Body &condition,
    const Body *thenPart, llvm::function_ref<void()> otherwise,
    const ProcedureBody &body) {
  mlir::Location loc{Location(tokens)};
  Values before{values_};
  auto ite{merk::IteOp::create(builder_, loc, mlir::TypeRange{})};
  llvm::SmallVector<Branch, 2> branches;
  {
    mlir::OpBuilder::InsertionGuard guard{builder_};
    // The condition yields what it binds, to the then part.
    builder_.setInsertionPointToStart(&ite.getCondition().emplaceBlock());
    Goal(condition, body);
    std::vector<BodyVar> bound{BoundSince(values_, before)};
    llvm::SmallVector<mlir::Value> yielded;
    for (BodyVar var : bound) {
      yielded.push_back(values_.lookup(var));
    }
    merk::YieldOp conditionYield;
    bool conditionFails{Terminated()};
    if (!conditionFails) {
      conditionYield = merk::YieldOp::create(builder_, loc, yielded);
    }
    mlir::Block &then{ite.getThenRegion().emplaceBlock()};
    values_ = before;
    for (auto [var, value] : llvm::zip(bound, yielded)) {
      values_[var] = then.addArgument(value.getType(), loc);
    }
    builder_.setInsertionPointToStart(&then);
    if (conditionFails) {
      // A condition that cannot succeed, such as a unification mode
      // analysis replaced with fail: the then part is never reached, so
      // only the else part binds the variables after the if-then-else.
      merk::FailOp::create(builder_, loc);
    } else if (thenPart) {
      Goal(*thenPart, body);
    }
    branches.push_back({&then, values_, Terminated()});
    mlir::Block &otherwiseBlock{ite.getElseRegion().emplaceBlock()};
    values_ = before;
    builder_.setInsertionPointToStart(&otherwiseBlock);
    otherwise();
    branches.push_back({&otherwiseBlock, values_, Terminated()});
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
  bool term{value && mlir::isa<merk::TermType>(value.getType())};
  if (!integer && !term) {
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
      if (term) {
        constants.push_back(
            merk::FunctorAttr::get(&context_, functor.name, functor.arity));
        continue;
      }
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
// The library's software_error exception for a predicate, as mmc writes it.
static std::string SoftwareError(llvm::StringRef pred, unsigned arity,
    llvm::StringRef problem) {
  return ("Software Error: predicate `array." + pred + "'/" +
      llvm::Twine(arity) + ": Unexpected: " + problem)
      .str();
}

bool MerkGenerator::ArrayBuiltin(const Body &goal, const Procedure &callee,
    llvm::ArrayRef<BodyVar> arguments, const ProcedureBody &body) {
  const semantics::Symbol &symbol{*callee.symbol};
  llvm::StringRef name{symbol.name};
  bool function{symbol.kind == semantics::SymbolKind::Function};
  // The inputs and the outputs, each in order.
  llvm::SmallVector<mlir::Value> in;
  llvm::SmallVector<BodyVar> out;
  for (auto [argument, mode] : llvm::zip(arguments, callee.modes)) {
    if (IsOutput(mode)) {
      out.push_back(argument);
    } else if (mlir::Value value{ValueOf(argument)}) {
      in.push_back(value);
    } else {
      return false;
    }
  }
  mlir::Location loc{Location(goal.tokens)};
  namespace arith = mlir::arith;
  auto i64{builder_.getI64Type()};

  if (name == "make_empty_array" && HasShape(in, out, 0, 1)) {
    // Its element type is its output's.
    std::optional<Represented> array{Represent(body.vars[out[0]].type)};
    if (!array) {
      return LibraryCallNotYet(goal, symbol);
    }
    return GiveOutputs(out,
        {merk::ArrayNewOp::create(builder_, loc, array->type,
            IntegerConstant(i64, 0, loc), mlir::Value{})});
  }
  if (name == "init" && HasShape(in, out, 2, 1)) {
    merk::ThrowIfOp::create(builder_, loc,
        CompareInts(arith::CmpIPredicate::slt, in[0],
            IntegerConstant(i64, 0, loc), loc),
        SoftwareError("init", 3, "negative size"));
    return GiveOutputs(out,
        {merk::ArrayNewOp::create(builder_, loc,
            merk::ArrayType::get(&context_, in[1].getType()), in[0], in[1])});
  }
  if (name == "size" && HasShape(in, out, 1, 1)) {
    return GiveOutputs(out, {ArraySize(in[0], loc)});
  }
  if (name == "max" && HasShape(in, out, 1, 1)) {
    return GiveOutputs(out,
        {arith::SubIOp::create(
            builder_, loc, ArraySize(in[0], loc), IntegerConstant(i64, 1, loc))
                .getResult()});
  }
  if (name == "min" && HasShape(in, out, 1, 1)) {
    return GiveOutputs(out, {IntegerConstant(i64, 0, loc)});
  }
  if (name == "bounds" && HasShape(in, out, 1, 2)) {
    return GiveOutputs(out,
        {IntegerConstant(i64, 0, loc),
            arith::SubIOp::create(builder_, loc, ArraySize(in[0], loc),
                IntegerConstant(i64, 1, loc))});
  }
  if (name == "in_bounds" && HasShape(in, out, 2, 0)) {
    merk::GuardOp::create(builder_, loc, InArrayBounds(in[0], in[1], loc));
    return true;
  }
  if (name == "is_empty" && HasShape(in, out, 1, 0)) {
    merk::GuardOp::create(builder_, loc,
        CompareInts(arith::CmpIPredicate::eq, ArraySize(in[0], loc),
            IntegerConstant(i64, 0, loc), loc));
    return true;
  }
  // lookup(A, I) and elem(I, A).
  bool elem{name == "elem" || name == "unsafe_elem"};
  if ((name == "lookup" || name == "unsafe_lookup" || elem) &&
      HasShape(in, out, 2, 1)) {
    mlir::Value array{in[elem ? 1 : 0]}, index{in[elem ? 0 : 1]};
    if (!name.starts_with("unsafe")) {
      merk::CheckIndexOp::create(builder_, loc, array, index, "array.lookup");
    }
    return GiveOutputs(out, {ArrayElement(array, index, loc)});
  }
  if (name == "semidet_lookup" && HasShape(in, out, 2, 1)) {
    merk::GuardOp::create(builder_, loc, InArrayBounds(in[0], in[1], loc));
    return GiveOutputs(out, {ArrayElement(in[0], in[1], loc)});
  }
  // set(I, X, !A), set(A, I, X) and 'elem :='(I, A, X); slow_set's copy.
  bool slow{name == "slow_set" || name == "semidet_slow_set"};
  bool setElem{name == "elem :=" || name == "unsafe_elem :="};
  if ((name == "set" || name == "unsafe_set" || name == "semidet_set" || slow ||
          setElem) &&
      HasShape(in, out, 3, 1)) {
    mlir::Value array, index, element;
    if (setElem) {
      std::tie(index, array, element) = std::tuple{in[0], in[1], in[2]};
    } else if (function) {
      std::tie(array, index, element) = std::tuple{in[0], in[1], in[2]};
    } else {
      std::tie(index, element, array) = std::tuple{in[0], in[1], in[2]};
    }
    if (name.starts_with("semidet")) {
      merk::GuardOp::create(builder_, loc, InArrayBounds(array, index, loc));
    } else if (!name.starts_with("unsafe")) {
      merk::CheckIndexOp::create(builder_, loc, array, index, "array.set");
    }
    if (slow) {
      array = merk::ArrayCopyOp::create(builder_, loc, array.getType(), array);
    }
    return GiveOutputs(out, {ArrayWithElement(array, index, element, loc)});
  }
  if ((name == "swap" || name == "unsafe_swap") && HasShape(in, out, 3, 1)) {
    mlir::Value i{in[0]}, j{in[1]}, array{in[2]};
    if (name == "swap") {
      merk::CheckIndexOp::create(
          builder_, loc, array, i, "first argument of array.swap");
      merk::CheckIndexOp::create(
          builder_, loc, array, j, "second argument of array.swap");
    }
    mlir::Value x{ArrayElement(array, i, loc)}, y{ArrayElement(array, j, loc)};
    return GiveOutputs(
        out, {ArrayWithElement(ArrayWithElement(array, i, y, loc), j, x, loc)});
  }
  if (name == "fill" && HasShape(in, out, 2, 1)) {
    mlir::Value hi{arith::SubIOp::create(
        builder_, loc, ArraySize(in[1], loc), IntegerConstant(i64, 1, loc))};
    return GiveOutputs(out,
        {merk::ArrayFillOp::create(builder_, loc, in[1].getType(), in[1],
            IntegerConstant(i64, 0, loc), hi, in[0])});
  }
  if (name == "fill_range" && HasShape(in, out, 4, 1)) {
    mlir::Value element{in[0]}, lo{in[1]}, hi{in[2]}, array{in[3]};
    merk::ThrowIfOp::create(builder_, loc,
        CompareInts(arith::CmpIPredicate::sgt, lo, hi, loc),
        SoftwareError("fill_range", 5, "empty range"));
    merk::CheckIndexOp::create(
        builder_, loc, array, lo, "second argument of fill_range");
    merk::CheckIndexOp::create(
        builder_, loc, array, hi, "third argument of fill_range");
    return GiveOutputs(out,
        {merk::ArrayFillOp::create(
            builder_, loc, array.getType(), array, lo, hi, element)});
  }
  if (name == "copy" && HasShape(in, out, 1, 1)) {
    return GiveOutputs(out,
        {merk::ArrayCopyOp::create(builder_, loc, in[0].getType(), in[0])});
  }
  // resize(N, X, !A) and resize(A, N, X).
  if (name == "resize" && HasShape(in, out, 3, 1)) {
    mlir::Value array{in[function ? 0 : 2]}, n{in[function ? 1 : 0]},
        element{in[function ? 2 : 1]};
    merk::ThrowIfOp::create(builder_, loc,
        CompareInts(
            arith::CmpIPredicate::slt, n, IntegerConstant(i64, 0, loc), loc),
        SoftwareError("resize", 4, "cannot resize to a negative size"));
    return GiveOutputs(out,
        {merk::ArrayResizeOp::create(
            builder_, loc, array.getType(), array, n, element)});
  }
  // shrink(N, !A) and shrink(A, N).
  if (name == "shrink" && HasShape(in, out, 2, 1)) {
    mlir::Value array{in[function ? 0 : 1]}, n{in[function ? 1 : 0]};
    merk::ThrowIfOp::create(builder_, loc,
        CompareInts(arith::CmpIPredicate::sgt, n, ArraySize(array, loc), loc),
        SoftwareError("shrink", 3, "cannot shrink to a larger size"));
    merk::ThrowIfOp::create(builder_, loc,
        CompareInts(
            arith::CmpIPredicate::slt, n, IntegerConstant(i64, 0, loc), loc),
        SoftwareError("shrink", 3, "cannot shrink to a negative size"));
    return GiveOutputs(out,
        {merk::ArrayResizeOp::create(
            builder_, loc, array.getType(), array, n, mlir::Value{})});
  }
  return LibraryCallNotYet(goal, symbol);
}

static std::string DomainError(llvm::StringRef module, llvm::StringRef what,
    llvm::StringRef problem) {
  return ("domain_error(\"" + module + "." + what + ": " + problem + "\")").str();
}

// What is not built in is called: float's functions are the compiled library's;
// the integer modules have no implementation yet.
bool MerkGenerator::NotBuiltIn(
    const Body &goal, const semantics::Symbol &symbol) {
  return symbol.module != "float" && LibraryCallNotYet(goal, symbol);
}

bool MerkGenerator::LibraryCallNotYet(
    const Body &goal, const semantics::Symbol &symbol) {
  NotYet(Where(goal.tokens),
      ("calls of " + symbol.module + "." + symbol.name + "/" +
          std::to_string(symbol.arity))
          .str());
  return true;
}

// A division checks for a zero divisor; one of the least value by -1, whose
// quotient overflows, wraps, as mmc's constant folding does.
mlir::Value MerkGenerator::IntegerDivide(mlir::Value a, mlir::Value b,
    llvm::StringRef module, llvm::StringRef what, bool floor, bool remainder,
    mlir::Location loc) {
  namespace arith = mlir::arith;
  mlir::Type type{a.getType()};
  merk::ThrowIfOp::create(builder_, loc,
      CompareInts(
          arith::CmpIPredicate::eq, b, IntegerConstant(type, 0, loc), loc),
      DomainError(module, what, "division by zero"));
  if (module.starts_with("uint")) {
    return remainder ? mlir::Value{arith::RemUIOp::create(builder_, loc, a, b)}
                     : mlir::Value{arith::DivUIOp::create(builder_, loc, a, b)};
  }
  mlir::Value minusOne{CompareInts(
      arith::CmpIPredicate::eq, b, IntegerConstant(type, -1, loc), loc)};
  mlir::Value divisor{arith::SelectOp::create(
      builder_, loc, minusOne, IntegerConstant(type, 1, loc), b)};
  mlir::Value q{arith::DivSIOp::create(builder_, loc, a, divisor)};
  mlir::Value r{arith::RemSIOp::create(builder_, loc, a, divisor)};
  q = arith::SelectOp::create(builder_, loc, minusOne,
      arith::SubIOp::create(builder_, loc, IntegerConstant(type, 0, loc), a), q);
  r = arith::SelectOp::create(
      builder_, loc, minusOne, IntegerConstant(type, 0, loc), r);
  if (floor) {
    // Toward minus infinity: one less where the remainder is not 0 and its
    // sign is not the divisor's.
    mlir::Value nonzero{arith::CmpIOp::create(builder_, loc,
        arith::CmpIPredicate::ne, r, IntegerConstant(type, 0, loc))};
    mlir::Value signs{arith::CmpIOp::create(builder_, loc,
        arith::CmpIPredicate::slt, arith::XOrIOp::create(builder_, loc, r, b),
        IntegerConstant(type, 0, loc))};
    mlir::Value adjust{arith::AndIOp::create(builder_, loc, nonzero, signs)};
    q = arith::SelectOp::create(builder_, loc, adjust,
        arith::SubIOp::create(builder_, loc, q, IntegerConstant(type, 1, loc)),
        q);
    r = arith::SelectOp::create(
        builder_, loc, adjust, arith::AddIOp::create(builder_, loc, r, b), r);
  }
  return remainder ? r : q;
}

// The shift amount is an int; outside 0 to the width it is an error, unless the
// shift is unchecked.
mlir::Value MerkGenerator::IntegerShift(mlir::Value value, mlir::Value amount,
    llvm::StringRef module, llvm::StringRef what, bool left, bool checked,
    mlir::Location loc) {
  namespace arith = mlir::arith;
  auto type{mlir::cast<mlir::IntegerType>(value.getType())};
  unsigned bits{type.getWidth()};
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
    return arith::ShLIOp::create(builder_, loc, value, amount);
  }
  return module.starts_with("uint")
      ? mlir::Value{arith::ShRUIOp::create(builder_, loc, value, amount)}
      : mlir::Value{arith::ShRSIOp::create(builder_, loc, value, amount)};
}

bool MerkGenerator::Builtin(const Body &goal, const Procedure &callee,
    llvm::ArrayRef<BodyVar> arguments, const ProcedureBody &body) {
  const semantics::Symbol &symbol{*callee.symbol};
  llvm::StringRef module{symbol.module};
  if (module == "builtin" ||
      (module == "array" && symbol.name == "array_compare")) {
    return CompareBuiltin(goal, callee, arguments, body);
  }
  // Temporary (chunk 1 step 6, docs/INPROGRESS.md): the array predicates
  // are built in until the library's array module can be compiled with
  // type_infos.
  if (module == "array") {
    return ArrayBuiltin(goal, callee, arguments, body);
  }
  // Temporary (chunk 1 step 6, docs/INPROGRESS.md): modules of the library
  // that are not compiled yet and whose predicates are not built in, which
  // would otherwise be called and fail to link. Step 6 compiles them.
  if (llvm::is_contained(
          {"bool", "char", "exception", "list", "string", "univ"}, module)) {
    return LibraryCallNotYet(goal, symbol);
  }
  bool isFloat{module == "float"};
  bool isUnsigned{module.starts_with("uint")};
  bool isInteger{
      llvm::is_contained({"int", "uint", "int8", "int16", "int32", "int64",
                             "uint8", "uint16", "uint32", "uint64"},
          module)};
  if (!isFloat && !isInteger) {
    return false;
  }
  bool function{symbol.kind == semantics::SymbolKind::Function};
  if (function && isInteger && arguments.size() == 3 &&
      llvm::is_contained({"+", "-", "xor"}, symbol.name) &&
      !IsOutput(callee.modes[2]) &&
      IsOutput(callee.modes[0]) != IsOutput(callee.modes[1])) {
    // The reverse modes, X + Y = R with X or Y the output, which a
    // unification of two applications such as A + B = C + D can need:
    // the output is computed from R and the other argument, with
    // wrapping, as the forward modes.
    unsigned out{IsOutput(callee.modes[0]) ? 0u : 1u};
    mlir::Value known{ValueOf(arguments[1 - out])};
    mlir::Value r{ValueOf(arguments[2])};
    if (!known || !r) {
      return NotBuiltIn(goal, symbol);
    }
    mlir::Location loc{Location(goal.tokens)};
    namespace arith = mlir::arith;
    llvm::StringRef name{symbol.name};
    mlir::Value result;
    if (name == "+") {
      result = arith::SubIOp::create(builder_, loc, r, known);
    } else if (name == "xor") {
      result = arith::XOrIOp::create(builder_, loc, r, known);
    } else if (out == 0) {
      result = arith::AddIOp::create(builder_, loc, r, known); // X = R + Y
    } else {
      result = arith::SubIOp::create(builder_, loc, known, r); // Y = X - R
    }
    values_[arguments[out]] = result;
    return true;
  }
  llvm::SmallVector<mlir::Value> in;
  for (auto [i, argument, mode] : llvm::enumerate(arguments, callee.modes)) {
    bool result{function && i + 1 == arguments.size()};
    if (IsOutput(mode) != result || (!result && !ValueOf(argument))) {
      return NotBuiltIn(goal, symbol);
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
    mlir::Value low{arith::AndIOp::create(
        builder_, loc, in[0], IntegerConstant(in[0].getType(), 1, loc))};
    merk::GuardOp::create(builder_, loc,
        arith::CmpIOp::create(builder_, loc, arith::CmpIPredicate::eq, low,
            IntegerConstant(in[0].getType(), name == "odd", loc)));
    return true;
  }
  if (!function) {
    // The comparisons, which are tests.
    if (in.size() != 2) {
      return NotBuiltIn(goal, symbol);
    }
    mlir::Value flag;
    if (isFloat) {
      std::optional<arith::CmpFPredicate> predicate{name == "<"
              ? arith::CmpFPredicate::OLT
              : name == ">"  ? arith::CmpFPredicate::OGT
              : name == "=<" ? arith::CmpFPredicate::OLE
              : name == ">=" ? arith::CmpFPredicate::OGE
                             : std::optional<arith::CmpFPredicate>{}};
      if (!predicate) {
        return NotBuiltIn(goal, symbol);
      }
      flag = arith::CmpFOp::create(builder_, loc, *predicate, in[0], in[1]);
    } else {
      std::optional<arith::CmpIPredicate> predicate{name == "<"
              ? (isUnsigned ? arith::CmpIPredicate::ult
                            : arith::CmpIPredicate::slt)
              : name == ">"  ? (isUnsigned ? arith::CmpIPredicate::ugt
                                           : arith::CmpIPredicate::sgt)
              : name == "=<" ? (isUnsigned ? arith::CmpIPredicate::ule
                                           : arith::CmpIPredicate::sle)
              : name == ">=" ? (isUnsigned ? arith::CmpIPredicate::uge
                                           : arith::CmpIPredicate::sge)
                             : std::optional<arith::CmpIPredicate>{}};
      if (!predicate) {
        return NotBuiltIn(goal, symbol);
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
      result =
          arith::SIToFPOp::create(builder_, loc, builder_.getF64Type(), in[0]);
    } else {
      return NotBuiltIn(goal, symbol);
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
      return NotBuiltIn(goal, symbol);
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
      return NotBuiltIn(goal, symbol);
    }
    values_[output] = arith::ConstantOp::create(
        builder_, loc, builder_.getIntegerAttr(type, value));
    return true;
  }
  auto type{mlir::cast<mlir::IntegerType>(in[0].getType())};
  unsigned bits{type.getWidth()};
  if (in.size() == 1 &&
      (name.starts_with("cast_") || name == "to_int" || name == "from_int" ||
          name == "to_uint")) {
    // A conversion between integer types keeps the value's low bits, and
    // extends it by its sign if its type is signed (library reference,
    // "int8" and the others).
    auto to{outputType
            ? mlir::dyn_cast_or_null<mlir::IntegerType>(outputType->type)
            : mlir::IntegerType{}};
    if (!to) {
      return NotBuiltIn(goal, symbol);
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
  if (in.size() == 2) {
    if (name == "+" || name == "plus") {
      result = arith::AddIOp::create(builder_, loc, in[0], in[1]);
    } else if (name == "-" || name == "minus") {
      result = arith::SubIOp::create(builder_, loc, in[0], in[1]);
    } else if (name == "*" || name == "times") {
      result = arith::MulIOp::create(builder_, loc, in[0], in[1]);
    } else if (name == "//" || name == "/") {
      result = IntegerDivide(in[0], in[1], module, "\\'//\\'",
          /*floor=*/false, /*remainder=*/false, loc);
    } else if (name == "rem") {
      result = IntegerDivide(in[0], in[1], module, "rem",
          /*floor=*/false, /*remainder=*/true, loc);
    } else if (name == "div") {
      result = IntegerDivide(in[0], in[1], module, "\\'//\\'",
          /*floor=*/true, /*remainder=*/false, loc);
    } else if (name == "mod") {
      result = IntegerDivide(in[0], in[1], module, "\\'//\\'",
          /*floor=*/true, /*remainder=*/true, loc);
    } else if (name == "unchecked_quotient") {
      result = isUnsigned
          ? mlir::Value{arith::DivUIOp::create(builder_, loc, in[0], in[1])}
          : mlir::Value{arith::DivSIOp::create(builder_, loc, in[0], in[1])};
    } else if (name == "unchecked_rem") {
      result = isUnsigned
          ? mlir::Value{arith::RemUIOp::create(builder_, loc, in[0], in[1])}
          : mlir::Value{arith::RemSIOp::create(builder_, loc, in[0], in[1])};
    } else if (name == "<<") {
      result = IntegerShift(in[0], in[1], module, "(<<)", /*left=*/true,
          /*checked=*/true, loc);
    } else if (name == ">>") {
      result = IntegerShift(in[0], in[1], module, "(>>)", /*left=*/false,
          /*checked=*/true, loc);
    } else if (name == "unchecked_left_shift") {
      result = IntegerShift(in[0], in[1], module, "", /*left=*/true,
          /*checked=*/false, loc);
    } else if (name == "unchecked_right_shift") {
      result = IntegerShift(in[0], in[1], module, "", /*left=*/false,
          /*checked=*/false, loc);
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
      return NotBuiltIn(goal, symbol);
    }
  } else if (in.size() == 1 && (name == "-" || name == "+")) {
    result = name == "+" ? in[0]
                         : mlir::Value{arith::SubIOp::create(builder_, loc,
                               IntegerConstant(type, 0, loc), in[0])};
  } else if (in.size() == 1 && !isUnsigned &&
      (name == "unchecked_abs" || name == "nabs")) {
    // nabs is minus the absolute value, which cannot overflow.
    mlir::Value negative{arith::CmpIOp::create(builder_, loc,
        arith::CmpIPredicate::slt, in[0], IntegerConstant(type, 0, loc))};
    mlir::Value negated{arith::SubIOp::create(
        builder_, loc, IntegerConstant(type, 0, loc), in[0])};
    result = name == "nabs"
        ? arith::SelectOp::create(builder_, loc, negative, in[0], negated)
        : arith::SelectOp::create(builder_, loc, negative, negated, in[0]);
  } else if (in.size() == 1 && name == "\\") {
    result = arith::XOrIOp::create(
        builder_, loc, in[0], IntegerConstant(type, -1, loc));
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
    mlir::Value negative{arith::CmpIOp::create(builder_, loc,
        arith::CmpIPredicate::slt, in[0], IntegerConstant(type, 0, loc))};
    result = arith::SelectOp::create(builder_, loc, negative,
        arith::SubIOp::create(
            builder_, loc, IntegerConstant(type, 0, loc), in[0]),
        in[0]);
  } else {
    return NotBuiltIn(goal, symbol);
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

// Only the scalars, integers and floats, and arrays of them can cross into
// a target region so far.
static bool Scalar(mlir::Type type) {
  return mlir::isa<mlir::IntegerType, mlir::FloatType>(type);
}

static bool Mappable(mlir::Type type) {
  if (auto array{mlir::dyn_cast<merk::ArrayType>(type)}) {
    return Scalar(array.getElement());
  }
  return Scalar(type);
}

// The kernel of a target region with the leaves `queue`, target first, and
// whether the host computes its loop's bounds and step, as host_eval
// values, by Flang's patterns (KernelTypeVisitor and HostEvalVisitor in
// flang/lib/Lower/OpenMP/OpenMP.cpp): after target, teams distribute
// parallel do and parallel do are SPMD, with the bounds on the host;
// teams distribute is generic, with the bounds on the host too, from which
// the host computes the number of teams; anything else is generic. The
// simd and loop forms of Flang's patterns are not implemented yet, and
// neither is a target region whose goal is a single directive, which
// would continue the pattern as a combined directive does.
struct Kernel {
  mlir::omp::TargetExecMode mode{mlir::omp::TargetExecMode::generic};
  bool hostEvalBounds{false};
};
static bool IsDirectives(llvm::ArrayRef<llvm::omp::Directive> leaves,
    std::initializer_list<llvm::omp::Directive> pattern) {
  return llvm::equal(leaves, pattern);
}

static Kernel KernelOf(const omp::ConstructQueue &queue) {
  using llvm::omp::Directive;
  llvm::SmallVector<Directive> leaves;
  for (const omp::Leaf &leaf : queue) {
    leaves.push_back(leaf.id);
  }
  if (IsDirectives(leaves,
          {Directive::OMPD_target, Directive::OMPD_teams,
              Directive::OMPD_distribute, Directive::OMPD_parallel,
              Directive::OMPD_do}) ||
      IsDirectives(leaves,
          {Directive::OMPD_target, Directive::OMPD_parallel,
              Directive::OMPD_do})) {
    return {mlir::omp::TargetExecMode::spmd, true};
  }
  if (IsDirectives(leaves,
          {Directive::OMPD_target, Directive::OMPD_teams,
              Directive::OMPD_distribute})) {
    return {mlir::omp::TargetExecMode::generic, true};
  }
  return {};
}

// The goals of a conjunction, in order, flattened.
static void CollectConjuncts(const Body &goal, std::vector<const Body *> &goals) {
  if (goal.kind == BodyKind::Conj) {
    for (const Body *conjunct : goal.goals) {
      CollectConjuncts(*conjunct, goals);
    }
  } else {
    goals.push_back(&goal);
  }
}

// The last call in a goal, in the order of its text, of a predicate.
static const Body *LastCallOf(const Body &goal, const semantics::Symbol *callee) {
  const Body *found{nullptr};
  if (goal.kind == BodyKind::Call && goal.modes && goal.modes->callee &&
      goal.modes->callee->symbol == callee) {
    found = &goal;
  }
  for (const Body *inner : goal.goals) {
    if (const Body *last{LastCallOf(*inner, callee)}) {
      found = last;
    }
  }
  return found;
}

// A reduction is on the leaves the decomposition puts it on, do and teams, as
// Flang puts it, and the map of its variable that target implies is the
// reduction's.
static bool IsOmpLoopClauseHandled(
    llvm::omp::Directive leaf, llvm::omp::Clause id, bool synthesized) {
  if (synthesized) {
    return id == llvm::omp::OMPC_map && leaf == llvm::omp::OMPD_target;
  }
  return id == llvm::omp::OMPC_reduction &&
      (leaf == llvm::omp::OMPD_do || leaf == llvm::omp::OMPD_teams);
}

static bool IsNamedType(const semantics::Type &type, llvm::StringRef module,
    llvm::StringRef name) {
  return type.kind == semantics::TypeKind::Named &&
      type.constructor->module == module && type.constructor->name == name;
}

// The type of a value a loop carries, at the call of its loop procedure.
static const semantics::Type *ThreadedType(const ProcedureBody &body,
    const Body &call, const semantics::TranslatedLoop::Threaded &value) {
  return body.vars[call.arguments[value.in]].type;
}

static bool IsReduction(llvm::ArrayRef<LoopReduction> reductions,
    const semantics::TranslatedLoop::Threaded &value) {
  return llvm::any_of(
      reductions, [&](const LoopReduction &r) { return r.value == &value; });
}

static bool HasLeaf(const omp::ConstructQueue &queue, llvm::omp::Directive id) {
  return llvm::any_of(
      queue, [&](const omp::Leaf &leaf) { return leaf.id == id; });
}

// The operator of a reduction clause by Mercury's name: +, *, min and
// max, of ints and floats, and /\, \/ and xor, of ints.
static std::optional<ReductionOp> ReductionOpOf(llvm::StringRef name) {
  return llvm::StringSwitch<std::optional<ReductionOp>>(name)
      .Case("+", ReductionOp::Add)
      .Case("*", ReductionOp::Multiply)
      .Case("max", ReductionOp::Max)
      .Case("min", ReductionOp::Min)
      .Case("/\\", ReductionOp::And)
      .Case("\\/", ReductionOp::Or)
      .Case("xor", ReductionOp::Xor)
      .Default(std::nullopt);
}

// The array that an array a target region gives was, before the region
// updated it in place, if the region's own operations show it.
static mlir::Value UpdatedFrom(mlir::Value array) {
  while (mlir::Operation *op = array.getDefiningOp()) {
    if (!mlir::isa<merk::ArraySetOp, merk::ArrayFillOp>(op)) {
      break;
    }
    array = op->getOperand(0);
  }
  return array;
}

// The goal under the directive is the loop's translation: the goals of its
// bounds, the computation of its trip count, and the call of its loop
// predicate (design document, "From a nest to omp.loop_nest"). The bounds'
// goals are generated as they are; the trip count, and the loop predicate,
// are not needed: omp.loop_nest runs the iterations from Lo to Hi, by the
// step, inclusive, and each calls the body procedure with the iteration's
// counter, the threaded arrays as they were before the loop and the
// shared variables. By the directive's promise, an array that the body
// updates in place gives each iteration its own elements, so after the
// loop it is the array given (design document, "Loops", Under a parallel
// directive). The directive is decomposed into its leaves, each with its
// own clauses, as Flang decomposes it: parallel and do, or target, teams,
// distribute, parallel and do, where the loop is in a target region whose
// kernel is SPMD, given its bounds and step by the host as host_eval
// values.
void MerkGenerator::OmpLoop(const Body &goal, const ProcedureBody &body) {
  using namespace parser::literals;
  const parser::OmpDirective &directive{*goal.directive};
  parser::CharBlock at{Where(goal.tokens)};
  std::string name{parser::EnumToString(directive.id)};
  bool offloaded{directive.id ==
      llvm::omp::Directive::OMPD_target_teams_distribute_parallel_do};
  if (directive.id != llvm::omp::Directive::OMPD_parallel_do && !offloaded) {
    NotYet(at, "the " + name + " directive");
    return;
  }
  if (builder_.getInsertionBlock()
          ->getParent()
          ->getParentOfType<mlir::omp::TargetOp>()) {
    NotYet(at,
        offloaded ? std::string{"target regions in target regions"}
                  : name + " in target regions");
    return;
  }
  std::vector<const Body *> goals;
  CollectConjuncts(*goal.goals.front(), goals);
  const Body *call{nullptr};
  const semantics::TranslatedLoop *loop{nullptr};
  for (const Body *g : goals) {
    if (g->kind == BodyKind::Call && g->modes && g->modes->callee) {
      if (const auto *found{loops_.Of(*g->modes->callee->symbol)}) {
        call = g;
        loop = found;
      }
    }
  }
  if (!loop) {
    Error(at, "the %s directive must be before a loop"_err_en_US, name);
    return;
  }
  std::optional<omp::ConstructQueue> queue{Decompose(directive, at,
      loop->counter.empty() ? std::nullopt
                            : std::optional<llvm::StringRef>{loop->counter})};
  if (!queue) {
    return;
  }
  for (const omp::Leaf &leaf : *queue) {
    if (!LeafClauses(leaf, at, IsOmpLoopClauseHandled)) {
      return;
    }
  }
  // Its form: a for, and fromtos that do not end it; a loop has at most
  // one for.
  bool counted{false};
  for (const semantics::LoopIterator &iterator : loop->loop->iterators) {
    llvm::StringRef kind;
    switch (iterator.kind) {
    case parser::LoopIteratorKind::For:
      counted = true;
      break;
    case parser::LoopIteratorKind::Foreach:
      kind = "foreach";
      break;
    case parser::LoopIteratorKind::Count:
      kind = "count";
      break;
    case parser::LoopIteratorKind::Fromto:
      if (iterator.ends) {
        kind = "a fromto that ends the loop";
      }
      break;
    }
    if (!kind.empty()) {
      Error(Where(iterator.tokens),
          "a parallel do loop is counted by its for, so it cannot have %s"_err_en_US,
          kind);
      return;
    }
  }
  if (!counted) {
    Error(at, "a parallel do loop must have a for iterator"_err_en_US);
    return;
  }
  const Body *trip{nullptr};
  for (const Body *g : goals) {
    if (g->kind == BodyKind::IfThenElse &&
        g->goals.front()->kind == BodyKind::Call && g->goals.front()->call &&
        g->goals.front()->call->number == *loop->tripTest) {
      trip = g;
    }
  }
  const Procedure &loopProcedure{*call->modes->callee};
  const ProcedureBody *loopBody{determinism_.BodyOf(loopProcedure)};
  // The loop procedure's call of the body procedure.
  const Body *bodyCall{loopBody && loopBody->body
          ? LastCallOf(*loopBody->body, loop->bodyPredicate)
          : nullptr};
  if (!trip || !bodyCall) {
    NotYet(at, "parallel do loops whose translation mode analysis changed");
    return;
  }
  // The checks of the parallel loop annotations (design document): the
  // body is det and pure, and threads only arrays, until reductions.
  const Procedure &bodyProcedure{*bodyCall->modes->callee};
  std::optional<parser::Determinism> determinism{
      determinism_.DeterminismOf(bodyProcedure)};
  if (determinism != parser::Determinism::Det) {
    Error(at,
        "the body of a parallel do loop must be det, but it is %s"_err_en_US,
        determinism ? semantics::DeterminismName(*determinism) : "unknown");
    return;
  }
  const semantics::Signature *signature{
      types_.SignatureOf(*loop->bodyPredicate)};
  if (signature && signature->purity != parser::Purity::Pure) {
    Error(at, "the body of a parallel do loop must be pure"_err_en_US);
    return;
  }
  // The reductions: each value the loop carries that a reduction clause
  // names, by its name, or the one scalar it carries, for a clause that
  // names none (design document, "Loops", Reductions).
  llvm::SmallVector<LoopReduction> reductions;
  for (const parser::OmpClause &clause : directive.clauses) {
    const auto *reduction{
        std::get_if<const parser::OmpReductionClause *>(&clause)};
    if (!reduction) {
      continue;
    }
    if ((*reduction)->modifier) {
      NotYet(at, "reduction modifiers");
      return;
    }
    // An operator is any function the programmer promises is associative;
    // so far, those of int and float that Flang has as intrinsic ones.
    const parser::SymName &opName{*(*reduction)->op};
    std::optional<ReductionOp> op{ReductionOpOf(opName.name)};
    if (!op || opName.qualifiers.size() > 1) {
      NotYet(at,
          "reductions by operators other than +, *, min, max, /\\, \\/ and "
          "xor");
      return;
    }
    llvm::SmallVector<const semantics::TranslatedLoop::Threaded *> named;
    if ((*reduction)->variables.empty()) {
      for (const semantics::TranslatedLoop::Threaded &value : loop->threaded) {
        if (!IsNamedType(*ThreadedType(body, *call, value), "array", "array")) {
          named.push_back(&value);
        }
      }
      if (named.size() != 1) {
        Error(at,
            "a reduction clause without variables names the one scalar the loop carries from one iteration to the next, but it carries %s"_err_en_US,
            named.empty() ? "none" : "several");
        return;
      }
    }
    for (const parser::Variable *variable : (*reduction)->variables) {
      auto value{llvm::find_if(
          loop->threaded, [&](const semantics::TranslatedLoop::Threaded &v) {
            return v.name == variable->name;
          })};
      if (value == loop->threaded.end()) {
        Error(at,
            "%s is not a value the loop carries from one iteration to the next, a fromto's Last or a state variable"_err_en_US,
            variable->name);
        return;
      }
      named.push_back(&*value);
    }
    for (const semantics::TranslatedLoop::Threaded *value : named) {
      if (IsReduction(reductions, *value)) {
        Error(
            at, "%s is named in two reduction clauses"_err_en_US, value->name);
        return;
      }
      // The primitive int and float.
      bool real{IsNamedType(*ThreadedType(body, *call, *value), "", "float")};
      if (!real &&
          !IsNamedType(*ThreadedType(body, *call, *value), "", "int")) {
        NotYet(at,
            "reductions of type " +
                semantics::TypeToString(
                    *body.vars[call->arguments[value->in]].type));
        return;
      }
      if (real && *op != ReductionOp::Add && *op != ReductionOp::Multiply &&
          *op != ReductionOp::Max && *op != ReductionOp::Min) {
        Error(at,
            "the reduction operator %s is not defined on floats"_err_en_US,
            opName.name);
        return;
      }
      // A qualified operator is the type's own, int.max of an int.
      if (!opName.qualifiers.empty() &&
          opName.qualifiers.front() != (real ? "float" : "int")) {
        Error(at,
            real
                ? "the reduction operator %s is not defined on floats"_err_en_US
                : "the reduction operator %s is not defined on ints"_err_en_US,
            (opName.qualifiers.front() + "." + opName.name).str());
        return;
      }
      reductions.push_back({value, *op,
          Represent(body.vars[call->arguments[value->in]].type)->type});
    }
  }
  for (const semantics::TranslatedLoop::Threaded &value : loop->threaded) {
    if (IsNamedType(*ThreadedType(body, *call, value), "io", "state")) {
      Error(at,
          "a parallel do loop must not thread the io state, since its iterations run in any order"_err_en_US);
      return;
    }
    if (!IsReduction(reductions, value) &&
        !IsNamedType(*ThreadedType(body, *call, value), "array", "array")) {
      Error(value.fromto ? Where(value.fromto->tokens) : at,
          "a parallel do loop can carry %s from one iteration to the next only if it is named in a reduction clause or is an array updated in place"_err_en_US,
          value.fromto ? "a fromto's value" : "a state variable");
      return;
    }
  }
  // A reduction's next value is the body procedure's output that the loop
  // procedure passes on, to its own call, as the value's In.
  const Body *recursion{LastCallOf(*loopBody->body, loop->loopPredicate)};
  for (LoopReduction &reduction : reductions) {
    std::optional<unsigned> output;
    unsigned outputs{0};
    for (auto [argument, mode] :
        llvm::zip(bodyCall->arguments, bodyProcedure.modes)) {
      if (Represent(loopBody->vars[argument].type)->dummy || !IsOutput(mode)) {
        continue;
      }
      if (recursion && argument == recursion->arguments[reduction.value->in]) {
        output = outputs;
      }
      ++outputs;
    }
    if (!output) {
      NotYet(at, "reductions whose next value the loop's body does not give");
      return;
    }
    reduction.output = *output;
  }
  // The leaves around the loop's wrappers.
  bool teams{HasLeaf(*queue, llvm::omp::Directive::OMPD_teams)};
  bool distribute{HasLeaf(*queue, llvm::omp::Directive::OMPD_distribute)};
  // Each reduction's variable is a slot of the procedure, holding First, or
  // the state variable's value before the loop, which the partial results
  // are combined with; after the loop, Last, or the state variable's value
  // after it, is what the slot holds.
  mlir::Location loc{Location(goal.tokens)};
  Kernel kernel{offloaded ? KernelOf(*queue) : Kernel{}};
  bool evaluated{
      kernel.hostEvalBounds && !target_.options().openMPIsTargetDevice};
  // The loop's goals, before the region too, for First, which is one of
  // them, and for the host_eval values.
  if (!reductions.empty() || evaluated) {
    for (const Body *g : goals) {
      if (g != trip && g != call) {
        Goal(*g, body);
      }
    }
    if (failed_) {
      return;
    }
  }
  llvm::SmallVector<UpdatedSlot> updated;
  for (LoopReduction &reduction : reductions) {
    reduction.slot = Slot(reduction.type, loc);
    mlir::LLVM::StoreOp::create(builder_, loc,
        ValueOf(call->arguments[reduction.value->in]), reduction.slot);
    updated.push_back({reduction.slot, reduction.type, reduction.value->name});
  }
  ParallelLoop parallelLoop{goals, trip, call, loop, *loopBody, bodyCall,
      bodyProcedure, reductions, teams, distribute, offloaded, at};
  if (!offloaded) {
    GenerateParallelLoop(parallelLoop, body, {}, loc);
    if (!failed_) {
      LoadReductions(reductions, *call, loc);
    }
    return;
  }
  llvm::SmallVector<mlir::Value, 3> hostEval;
  if (evaluated) {
    BodyVar big{trip->goals.front()->arguments[0]};
    BodyVar small{trip->goals.front()->arguments[1]};
    hostEval = {ValueOf(loop->step > 0 ? small : big),
        ValueOf(loop->step > 0 ? big : small),
        IntegerConstant(builder_.getI64Type(), loop->step, loc)};
  }
  Target(
      goal, body,
      [&](mlir::ValueRange evaluated) {
        GenerateParallelLoop(parallelLoop, body, evaluated, loc);
      },
      /*combined=*/true, kernel.mode, hostEval, updated);
  if (!failed_) {
    LoadReductions(reductions, *call, loc);
  }
}

// Offloaded, the kernel is as Flang makes it (KernelOf): for target teams
// distribute parallel do, SPMD, and the host computes the bounds and the
// step before the target region and gives them to it as host_eval
// values, from which it computes the trip count and so the number of
// teams and threads, and the nest runs over them. The bounds'
// goals are in the region too, on both sides, so that the host and the
// device map the same values; the device's nest runs over those, as
// Flang's does, since the device has no host_eval.
void MerkGenerator::GenerateParallelLoop(const ParallelLoop &loop,
    const ProcedureBody &body, mlir::ValueRange hostEval, mlir::Location loc) {
  mlir::OpBuilder::InsertionGuard guard{builder_};
  // Teams and parallel end with terminators, added after what is in them,
  // since a goal is not generated after one.
  llvm::SmallVector<mlir::Block *, 2> terminated;
  // The leaves, each in the one before, as Flang marks them: a leaf
  // around the parallel, a combined one; with distribute, the parallel,
  // distribute and do are the parts of a composite one. The bounds'
  // goals are in the parallel, as Flang places them: on an AMD GPU, a
  // generic kernel traps when the parallel in its teams uses a value
  // computed outside the parallel.
  // A reduction is on teams and on the do, each with its own copy of the
  // variable, the teams' copy the do's original, as in Flang: by value,
  // by its slot's pointer, which teams has from the target region, mapped
  // tofrom.
  llvm::SmallVector<mlir::Value> originals;
  for (const LoopReduction &reduction : loop.reductions) {
    originals.push_back(reduction.slot);
  }
  llvm::SmallVector<mlir::Type> pointers(
      loop.reductions.size(), mlir::LLVM::LLVMPointerType::get(&context_));
  llvm::SmallVector<mlir::Location> locs(loop.reductions.size(), loc);
  if (loop.teams) {
    mlir::omp::TeamsOperands operands;
    AddReductions(operands, loop.reductions, originals, loc);
    auto op{mlir::omp::TeamsOp::create(builder_, loc, operands)};
    op.setCombined(true);
    terminated.push_back(
        builder_.createBlock(&op.getRegion(), {}, pointers, locs));
    originals.assign(op.getRegion().args_begin(), op.getRegion().args_end());
  }
  auto parallel{mlir::omp::ParallelOp::create(
      builder_, loc, mlir::omp::ParallelOperands{})};
  if (loop.distribute) {
    parallel.setComposite(true);
  } else {
    parallel.setCombined(true);
  }
  terminated.push_back(builder_.createBlock(&parallel.getRegion()));
  for (const Body *g : loop.goals) {
    if (g != loop.trip && g != loop.call) {
      Goal(*g, body);
    }
  }
  if (failed_) {
    return;
  }
  BodyVar big{loop.trip->goals.front()->arguments[0]};
  BodyVar small{loop.trip->goals.front()->arguments[1]};
  mlir::Value lower{ValueOf(loop.translated->step > 0 ? small : big)};
  mlir::Value upper{ValueOf(loop.translated->step > 0 ? big : small)};
  mlir::Type i64{builder_.getI64Type()};
  mlir::Value step{IntegerConstant(i64, loop.translated->step, loc)};
  if (!hostEval.empty()) {
    lower = hostEval[0];
    upper = hostEval[1];
    step = hostEval[2];
  }

  llvm::SmallVector<const semantics::Type *> types;
  for (BodyVar argument : loop.bodyCall->arguments) {
    types.push_back(loop.loopBody.vars[argument].type);
  }
  merk::ProcOp proc{ProcFor(loop.bodyProcedure, types, loop.at)};
  if (!proc) {
    return;
  }
  {
    mlir::OpBuilder::InsertionGuard guard{builder_};
    if (loop.distribute) {
      auto op{mlir::omp::DistributeOp::create(
          builder_, loc, mlir::omp::DistributeOperands{})};
      op.setComposite(true);
      builder_.createBlock(&op.getRegion());
    }
    mlir::omp::WsloopOperands operands;
    AddReductions(operands, loop.reductions, originals, loc);
    auto wsloop{mlir::omp::WsloopOp::create(builder_, loc, operands)};
    if (loop.distribute) {
      wsloop.setComposite(true);
    }
    mlir::Block *privates{
        builder_.createBlock(&wsloop.getRegion(), {}, pointers, locs)};
    mlir::omp::LoopNestOperands bounds;
    bounds.loopLowerBounds.push_back(lower);
    bounds.loopUpperBounds.push_back(upper);
    bounds.loopSteps.push_back(step);
    bounds.loopInclusive = builder_.getUnitAttr();
    bounds.collapseNumLoops = builder_.getI64IntegerAttr(1);
    auto nest{mlir::omp::LoopNestOp::create(builder_, loc, bounds)};
    mlir::Block *iteration{
        builder_.createBlock(&nest.getRegion(), {}, {i64}, {loc})};
    // The body procedure's inputs: its hidden arguments, which are the
    // procedure's own, for its type variables and constraints have the same
    // numbers, the counter, and the values the loop procedure was called
    // with for its other arguments.
    llvm::SmallVector<mlir::Value> inputs;
    if (const auto *hidden{
            polymorphism_.lookup(determinism_.BodyOf(loop.bodyProcedure))};
        hidden && !hidden->hidden.empty()) {
      // Temporary (specialization, design document, "What polymorphism
      // costs"): a target region would have to map them.
      if (loop.offloaded) {
        NotYet(loop.at, "target regions in polymorphic procedures");
        return;
      }
      for (const semantics::HiddenArgument &argument : hidden->hidden) {
        if (argument.output) {
          continue;
        }
        inputs.push_back(argument.typeInfo
                ? TypeInfoOfType(argument.typeInfo->type, loc, loop.at,
                      /*freeIsVoid=*/true)
                : dictionaries_.lookup(argument.dictionary->parameter));
      }
    }
    for (auto [argument, mode] :
        llvm::zip(loop.bodyCall->arguments, loop.bodyProcedure.modes)) {
      if (Represent(loop.loopBody.vars[argument].type)->dummy ||
          IsOutput(mode)) {
        continue;
      }
      auto position{llvm::find(loop.loopBody.head, argument)};
      if (position == loop.loopBody.head.end()) {
        NotYet(loop.at,
            "parallel do loops whose body takes a value the loop computes");
        return;
      }
      std::size_t k(position - loop.loopBody.head.begin());
      // A reduction's In is the do's copy of its variable.
      const auto *reduction{llvm::find_if(loop.reductions,
          [&](const LoopReduction &r) { return r.value->in == k; })};
      if (reduction != loop.reductions.end()) {
        inputs.push_back(
            mlir::LLVM::LoadOp::create(builder_, loc, reduction->type,
                privates->getArgument(reduction - loop.reductions.begin())));
        continue;
      }
      inputs.push_back(k == 0 ? iteration->getArgument(0)
                              : ValueOf(loop.call->arguments[k]));
    }
    auto bodyOp{merk::CallOp::create(builder_, loc,
        proc.getFunctionType().getResults(), proc.getSymName(), inputs)};
    for (auto [i, reduction] : llvm::enumerate(loop.reductions)) {
      mlir::LLVM::StoreOp::create(builder_, loc,
          bodyOp->getResult(reduction.output), privates->getArgument(i));
    }
    mlir::omp::YieldOp::create(builder_, loc);
  }
  for (mlir::Block *block : terminated) {
    builder_.setInsertionPointToEnd(block);
    mlir::omp::TerminatorOp::create(builder_, loc);
  }
  for (const semantics::TranslatedLoop::Threaded &value :
      loop.translated->threaded) {
    if (!IsReduction(loop.reductions, value)) {
      values_[loop.call->arguments[value.after]] =
          ValueOf(loop.call->arguments[value.in]);
    }
  }
}

template <typename Operands>
void MerkGenerator::AddReductions(Operands &operands,
    llvm::ArrayRef<LoopReduction> reductions, mlir::ValueRange vars,
    mlir::Location loc) {
  for (auto [reduction, var] : llvm::zip(reductions, vars)) {
    operands.reductionVars.push_back(var);
    operands.reductionByref.push_back(false);
    operands.reductionSyms.push_back(mlir::SymbolRefAttr::get(
        DeclareReduction(reduction.op, reduction.type, loc)));
  }
}

// After the loop, each reduction's Last is what its slot holds.
void MerkGenerator::LoadReductions(llvm::ArrayRef<LoopReduction> reductions,
    const Body &call, mlir::Location loc) {
  for (const LoopReduction &reduction : reductions) {
    values_[call.arguments[reduction.value->after]] =
        mlir::LLVM::LoadOp::create(
            builder_, loc, reduction.type, reduction.slot);
  }
}

// Its identity and combiner are Flang's (getReductionInitValue and
// createScalarCombiner), for by-value integers and floats.
mlir::omp::DeclareReductionOp MerkGenerator::DeclareReduction(
    ReductionOp op, mlir::Type type, mlir::Location loc) {
  bool real{mlir::isa<mlir::FloatType>(type)};
  llvm::StringRef prefix;
  switch (op) {
  case ReductionOp::Add:
    prefix = "add_reduction";
    break;
  case ReductionOp::Multiply:
    prefix = "multiply_reduction";
    break;
  case ReductionOp::Max:
    prefix = "max";
    break;
  case ReductionOp::Min:
    prefix = "min";
    break;
  case ReductionOp::And:
    prefix = "iand";
    break;
  case ReductionOp::Or:
    prefix = "ior";
    break;
  case ReductionOp::Xor:
    prefix = "ieor";
    break;
  }
  std::string name{(prefix + (real ? "_f" : "_i") +
      llvm::Twine{type.getIntOrFloatBitWidth()})
          .str()};
  if (auto found{module_.lookupSymbol<mlir::omp::DeclareReductionOp>(name)}) {
    return found;
  }
  // At the module's start, after those made before it.
  mlir::OpBuilder::InsertionGuard guard{builder_};
  builder_.setInsertionPointToStart(module_.getBody());
  for (mlir::Operation &op : *module_.getBody()) {
    if (!mlir::isa<mlir::omp::DeclareReductionOp>(op)) {
      break;
    }
    builder_.setInsertionPointAfter(&op);
  }
  auto decl{mlir::omp::DeclareReductionOp::create(builder_, loc, name,
      /*sym_visibility=*/nullptr, type, /*byref_element_type=*/nullptr)};
  builder_.createBlock(&decl.getInitializerRegion(), {}, {type}, {loc});
  mlir::Value identity;
  if (real) {
    const llvm::fltSemantics &semantics{
        mlir::cast<mlir::FloatType>(type).getFloatSemantics()};
    llvm::APFloat value{
        op == ReductionOp::Max ? llvm::APFloat::getLargest(semantics, true)
            : op == ReductionOp::Min
            ? llvm::APFloat::getLargest(semantics, false)
            : llvm::APFloat{semantics, op == ReductionOp::Multiply ? 1u : 0u}};
    identity = mlir::arith::ConstantOp::create(
        builder_, loc, type, builder_.getFloatAttr(type, value));
  } else {
    unsigned bits{type.getIntOrFloatBitWidth()};
    llvm::APInt value{op == ReductionOp::Max ? llvm::APInt::getSignedMinValue(bits)
            : op == ReductionOp::Min        ? llvm::APInt::getSignedMaxValue(bits)
            : op == ReductionOp::And        ? llvm::APInt::getAllOnes(bits)
            : op == ReductionOp::Multiply   ? llvm::APInt{bits, 1}
                                            : llvm::APInt{bits, 0}};
    identity = mlir::arith::ConstantOp::create(
        builder_, loc, type, builder_.getIntegerAttr(type, value));
  }
  mlir::omp::YieldOp::create(builder_, loc, identity);
  mlir::Block *combiner{builder_.createBlock(
      &decl.getReductionRegion(), {}, {type, type}, {loc, loc})};
  mlir::Value a{combiner->getArgument(0)};
  mlir::Value b{combiner->getArgument(1)};
  mlir::Value combined;
  switch (op) {
  case ReductionOp::Add:
    combined = real ? mlir::arith::AddFOp::create(builder_, loc, a, b).getResult()
                    : mlir::arith::AddIOp::create(builder_, loc, a, b).getResult();
    break;
  case ReductionOp::Multiply:
    combined = real ? mlir::arith::MulFOp::create(builder_, loc, a, b).getResult()
                    : mlir::arith::MulIOp::create(builder_, loc, a, b).getResult();
    break;
  case ReductionOp::Max:
    combined = real
        ? mlir::arith::MaxNumFOp::create(builder_, loc, a, b).getResult()
        : mlir::arith::MaxSIOp::create(builder_, loc, a, b).getResult();
    break;
  case ReductionOp::Min:
    combined = real
        ? mlir::arith::MinNumFOp::create(builder_, loc, a, b).getResult()
        : mlir::arith::MinSIOp::create(builder_, loc, a, b).getResult();
    break;
  case ReductionOp::And:
    combined = mlir::arith::AndIOp::create(builder_, loc, a, b);
    break;
  case ReductionOp::Or:
    combined = mlir::arith::OrIOp::create(builder_, loc, a, b);
    break;
  case ReductionOp::Xor:
    combined = mlir::arith::XOrIOp::create(builder_, loc, a, b);
    break;
  }
  mlir::omp::YieldOp::create(builder_, loc, combined);
  return decl;
}

std::optional<omp::ConstructQueue> MerkGenerator::Decompose(
    const parser::OmpDirective &directive, parser::CharBlock at,
    std::optional<llvm::StringRef> counter) {
  using namespace parser::literals;
  // Merkurius's own clauses, which the decomposition leaves out.
  for (const parser::OmpClause &clause : directive.clauses) {
    if (std::holds_alternative<const parser::OmpIndexClause *>(clause) ||
        std::holds_alternative<const parser::OmpUpperClause *>(clause)) {
      NotYet(at,
          std::holds_alternative<const parser::OmpIndexClause *>(clause)
              ? "the index clause"
              : "the upper clause");
      return std::nullopt;
    }
  }
  omp::Decomposition decomposition{omp::Decompose(
      target_.options().openMPVersion, directive, counter)};
  if (decomposition.queue.empty()) {
    std::string clause{decomposition.misplaced
            ? llvm::omp::getOpenMPClauseName(
                  omp::ClauseId(*decomposition.misplaced))
                  .str()
            : "a"};
    Error(at, "%s"_err_en_US,
        "the " + clause + " clause is not allowed on the " +
            std::string{parser::EnumToString(directive.id)} + " directive");
    return std::nullopt;
  }
  return std::move(decomposition.queue);
}

bool MerkGenerator::LeafClauses(const omp::Leaf &leaf, parser::CharBlock at,
    llvm::function_ref<bool(llvm::omp::Directive, llvm::omp::Clause, bool)>
        handled) {
  std::string directive{parser::EnumToString(leaf.id)};
  for (const parser::OmpClause *clause : leaf.clauses) {
    llvm::omp::Clause id{omp::ClauseId(*clause)};
    if (handled && handled(leaf.id, id, false)) {
      continue;
    }
    NotYet(at,
        "the " + llvm::omp::getOpenMPClauseName(id).str() + " clause of the " +
            directive + " directive");
    return false;
  }
  for (llvm::omp::Clause id : leaf.synthesized) {
    if (id != llvm::omp::OMPC_shared && !(handled && handled(leaf.id, id, true))) {
      NotYet(at,
          "the " + llvm::omp::getOpenMPClauseName(id).str() +
              " clause that the " + directive + " directive implies");
      return false;
    }
  }
  return true;
}

void MerkGenerator::Omp(const Body &goal, const ProcedureBody &body) {
  const parser::OmpDirective &directive{*goal.directive};
  parser::CharBlock at{Where(goal.tokens)};
  if (llvm::any_of(llvm::omp::getLeafConstructsOrSelf(directive.id),
          [](llvm::omp::Directive leaf) {
            return llvm::omp::getDirectiveAssociation(leaf) ==
                llvm::omp::Association::LoopNest;
          })) {
    OmpLoop(goal, body);
    return;
  }
  if (directive.id == llvm::omp::Directive::OMPD_target_data) {
    std::optional<omp::ConstructQueue> queue{Decompose(directive, at)};
    if (queue && LeafClauses(queue->front(), at,
            [](llvm::omp::Directive, llvm::omp::Clause id, bool) {
              return id == llvm::omp::OMPC_map;
            })) {
      TargetData(goal, body, queue->front());
    }
    return;
  }
  if (directive.id != llvm::omp::Directive::OMPD_target) {
    NotYet(at,
        "the " + std::string{parser::EnumToString(directive.id)} +
            " directive");
    return;
  }
  std::optional<omp::ConstructQueue> queue{Decompose(directive, at)};
  if (!queue || !LeafClauses(queue->front(), at)) {
    return;
  }
  if (builder_.getInsertionBlock()->getParent()
          ->getParentOfType<mlir::omp::TargetOp>()) {
    NotYet(at, "target regions in target regions");
    return;
  }
  Target(goal, body, [&](mlir::ValueRange) { Goal(*goal.goals.front(), body); },
      /*combined=*/false, KernelOf(*queue).mode);
}

// An array is mapped as ClangIR maps a pointer's section, p[0:n]: by its own
// pointer, with bounds over its size word and its elements, which only the
// host needs, to copy them.
mlir::Value MerkGenerator::MapArray(mlir::Value array,
    mlir::omp::ClauseMapFlags flags, llvm::StringRef name, mlir::Location loc) {
  auto ptr{mlir::LLVM::LLVMPointerType::get(&context_)};
  auto i64{builder_.getI64Type()};
  llvm::SmallVector<mlir::Value, 1> bounds;
  if (!target_.options().openMPIsTargetDevice) {
    // Words 0 to the size.
    mlir::Value upper{merk::ArraySizeOp::create(builder_, loc, i64, array)};
    mlir::Value extent{
        mlir::arith::AddIOp::create(
            builder_, loc, upper, IntegerConstant(i64, 1, loc))};
    bounds.push_back(mlir::omp::MapBoundsOp::create(builder_, loc,
        builder_.getType<mlir::omp::MapBoundsType>(),
        IntegerConstant(i64, 0, loc), upper, extent,
        /*stride=*/IntegerConstant(i64, 1, loc), /*strideInBytes=*/false,
        /*startIdx=*/IntegerConstant(i64, 0, loc)));
  }
  mlir::Value pointer{
      mlir::UnrealizedConversionCastOp::create(builder_, loc, ptr, array)
          .getResult(0)};
  return mlir::omp::MapInfoOp::create(builder_, loc, ptr, pointer,
      mlir::TypeAttr::get(i64),
      builder_.getAttr<mlir::omp::ClauseMapFlagsAttr>(flags),
      builder_.getAttr<mlir::omp::VariableCaptureKindAttr>(
          mlir::omp::VariableCaptureKind::ByRef),
      /*varPtrPtr=*/mlir::Value{}, /*varPtrPtrType=*/mlir::TypeAttr{},
      /*members=*/mlir::ValueRange{}, /*membersIndex=*/mlir::ArrayAttr{},
      bounds, /*mapperId=*/mlir::FlatSymbolRefAttr{},
      builder_.getStringAttr(name), builder_.getBoolAttr(false));
}

// Whether a variable named `text` is a version of the state variable !`name`,
// the variable of that name that has a value.
static bool IsStateVariableVersion(llvm::StringRef text, llvm::StringRef name) {
  unsigned version;
  return text.consume_front("STATE_VARIABLE_") && text.consume_front(name) &&
      text.consume_front("_") && !text.getAsInteger(10, version);
}

// A structured data region: the arrays it maps stay on the device for its
// goal, so that the map of a target region in it finds them there and copies
// nothing. As Flang lowers !$omp target data, omp.target_data holds the goal,
// and what the goal binds leaves the region through slots.
//
// A map names a variable, or a state variable !A, whose current version is
// mapped. The directive promises that the host does not use a mapped array's
// elements in the goal, which the compiler cannot see through calls. It does
// check at run time that !A leaves as the array that came in: at the end of
// the region, the device's elements are copied to that one.
void MerkGenerator::TargetData(const Body &goal, const ProcedureBody &body,
    const omp::Leaf &leaf) {
  using namespace parser::literals;
  parser::CharBlock at{Where(goal.tokens)};
  mlir::Location loc{Location(goal.tokens)};
  if (builder_.getInsertionBlock()->getParent()
          ->getParentOfType<mlir::omp::TargetOp>()) {
    NotYet(at, "target data in target regions");
    return;
  }
  // Failing would leave the region without unmapping its arrays.
  if (auto it{body.goals.find(goal.goals.front())};
      it != body.goals.end() && it->second != parser::Determinism::Det &&
      it->second != parser::Determinism::Erroneous) {
    Error(at,
        "the goal of a target data directive must be det, but it is %s"_err_en_US,
        semantics::DeterminismName(it->second));
    return;
  }
  struct Mapped {
    llvm::StringRef name;
    BodyVar var;
    bool state;
    bool copiedBack;
  };
  llvm::SmallVector<Mapped> mapped;
  llvm::SmallVector<mlir::Value> maps;
  Values before{values_};
  for (const parser::OmpClause *clause : leaf.clauses) {
    const auto *map{std::get_if<const parser::OmpMapClause *>(clause)};
    if (!map) {
      continue;
    }
    if (!(*map)->modifiers.empty()) {
      NotYet(at, "map modifiers on the target data directive");
      return;
    }
    mlir::omp::ClauseMapFlags flags{
        mlir::omp::ClauseMapFlags::to | mlir::omp::ClauseMapFlags::from};
    switch ((*map)->type.value_or(parser::OmpMapType::Tofrom)) {
    case parser::OmpMapType::Tofrom:
      break;
    case parser::OmpMapType::To:
      flags = mlir::omp::ClauseMapFlags::to;
      break;
    case parser::OmpMapType::From:
      flags = mlir::omp::ClauseMapFlags::from;
      break;
    case parser::OmpMapType::Alloc:
      flags = mlir::omp::ClauseMapFlags::storage;
      break;
    default:
      NotYet(at, "this map type on the target data directive");
      return;
    }
    for (const parser::OmpMapObject *object : (*map)->objects) {
      if (object->lower) {
        NotYet(at, "array sections on the target data directive");
        return;
      }
      llvm::StringRef name{object->variable->name};
      std::optional<BodyVar> found;
      bool state{false};
      for (BodyVar var : goal.nonlocals) {
        if (!before.lookup(var)) {
          continue;
        }
        if (body.vars[var].name == name) {
          found = var;
        } else if (IsStateVariableVersion(body.vars[var].name, name)) {
          found = var;
          state = true;
        }
      }
      if (!found) {
        Error(at,
            "%s is not a variable or state variable that the goal of the target data directive uses"_err_en_US,
            name);
        return;
      }
      if (llvm::any_of(mapped, [&](const Mapped &m) { return m.var == *found; })) {
        Error(at, "%s is mapped twice"_err_en_US, name);
        return;
      }
      mlir::Value value{before.lookup(*found)};
      if (!mlir::isa<merk::ArrayType>(value.getType())) {
        NotYet(at,
            "mapping values of type " +
                semantics::TypeToString(*body.vars[*found].type) +
                " with the target data directive");
        return;
      }
      maps.push_back(MapArray(value, flags, name, loc));
      mapped.push_back({name, *found, state,
          mlir::omp::bitEnumContainsAll(flags, mlir::omp::ClauseMapFlags::from)});
    }
  }
  if (maps.empty()) {
    Error(at, "a target data directive must have a map clause"_err_en_US);
    return;
  }
  mlir::omp::TargetDataOperands operands;
  operands.mapVars = maps;
  auto region{mlir::omp::TargetDataOp::create(builder_, loc, operands)};
  mlir::Block *block{new mlir::Block};
  region.getRegion().push_back(block);
  // What the goal binds and is used after it leaves in a word's slot, since
  // a value in the region does not dominate what follows it.
  llvm::SmallVector<std::tuple<BodyVar, mlir::Value, mlir::Type>> outputs;
  {
    mlir::OpBuilder::InsertionGuard guard{builder_};
    builder_.setInsertionPointToStart(block);
    Goal(*goal.goals.front(), body);
    if (failed_) {
      values_ = before;
      return;
    }
    for (BodyVar var : BoundSince(values_, before)) {
      mlir::Value value{values_.lookup(var)};
      if (value && llvm::is_contained(goal.nonlocals, var)) {
        mlir::Value slot{Slot(builder_.getI64Type(), loc)};
        mlir::LLVM::StoreOp::create(builder_, loc,
            Coerce(value, builder_.getI64Type(), loc), slot);
        outputs.emplace_back(var, slot, value.getType());
      }
    }
    mlir::omp::TerminatorOp::create(builder_, loc);
  }
  values_ = before;
  for (auto [var, slot, type] : outputs) {
    values_[var] = Coerce(mlir::LLVM::LoadOp::create(
                              builder_, loc, builder_.getI64Type(), slot),
        type, loc);
  }
  // A state variable's array must leave as the array that came in, and the
  // device's updates of it must come back.
  for (const Mapped &m : mapped) {
    if (!m.state) {
      continue;
    }
    for (auto &[var, slot, type] : outputs) {
      if (!IsStateVariableVersion(body.vars[var].name, m.name)) {
        continue;
      }
      if (!m.copiedBack) {
        Error(at,
            "the goal of the target data directive updates %s, whose map type does not copy it back; use tofrom or from"_err_en_US,
            m.name);
        return;
      }
      mlir::Value in{Coerce(before.lookup(m.var), builder_.getI64Type(), loc)};
      mlir::Value out{Coerce(values_.lookup(var), builder_.getI64Type(), loc)};
      merk::ThrowIfOp::create(builder_, loc,
          mlir::arith::CmpIOp::create(builder_, loc,
              mlir::arith::CmpIPredicate::ne, in, out),
          "Software Error: target data: the array " + m.name.str() +
              " was replaced in the region, not updated in place");
    }
  }
}

// The variable whose value `input` is, if it is one's: one of `preferred`, if
// several have the value, as an array updated in place does.
std::optional<BodyVar> MerkGenerator::VarWithValue(const Values &values,
    llvm::ArrayRef<BodyVar> preferred, mlir::Value input) {
  std::optional<BodyVar> found;
  for (const auto &[var, value] : values) {
    if (value != input) {
      continue;
    }
    if (llvm::is_contained(preferred, var)) {
      return var;
    }
    found = var;
  }
  return found;
}

const MerkGenerator::UpdatedSlot *MerkGenerator::FindUpdatedSlot(
    llvm::ArrayRef<UpdatedSlot> slots, mlir::Value slot) {
  const UpdatedSlot *found{llvm::find_if(
      slots, [&](const UpdatedSlot &u) { return u.slot == slot; })};
  return found == slots.end() ? nullptr : found;
}

// As Flang maps a scalar: the map entry of a slot, by its pointer.
mlir::Value MerkGenerator::MapSlot(mlir::Value slot, mlir::Type type,
    mlir::omp::ClauseMapFlags flags, mlir::omp::VariableCaptureKind capture,
    llvm::StringRef name, mlir::Location loc) {
  return mlir::omp::MapInfoOp::create(builder_, loc,
      mlir::LLVM::LLVMPointerType::get(&context_), slot,
      mlir::TypeAttr::get(type),
      builder_.getAttr<mlir::omp::ClauseMapFlagsAttr>(flags),
      builder_.getAttr<mlir::omp::VariableCaptureKindAttr>(capture),
      /*varPtrPtr=*/mlir::Value{}, /*varPtrPtrType=*/mlir::TypeAttr{},
      /*members=*/mlir::ValueRange{}, /*membersIndex=*/mlir::ArrayAttr{},
      /*bounds=*/mlir::ValueRange{}, /*mapperId=*/mlir::FlatSymbolRefAttr{},
      builder_.getStringAttr(name), builder_.getBoolAttr(false));
}

mlir::Value MerkGenerator::ArrayPointer(mlir::Value array, mlir::Location loc) {
  return mlir::UnrealizedConversionCastOp::create(
      builder_, loc, mlir::LLVM::LLVMPointerType::get(&context_), array)
      .getResult(0);
}

void MerkGenerator::Target(const Body &goal, const ProcedureBody &body,
    llvm::function_ref<void(mlir::ValueRange hostEval)> generate, bool combined,
    mlir::omp::TargetExecMode mode, mlir::ValueRange hostEval,
    llvm::ArrayRef<UpdatedSlot> slots) {
  parser::CharBlock at{Where(goal.tokens)};
  mlir::Location loc{Location(goal.tokens)};
  Values before{values_};
  // What `generate` builds, in a target region whose map entries are added
  // after it: they are the values it uses from outside and those it binds.
  // The host_eval arguments come before the map entries' arguments.
  mlir::omp::TargetExtOperands operands;
  operands.kernelType = mlir::omp::TargetExecModeAttr::get(&context_, mode);
  operands.hostEvalVars.append(hostEval.begin(), hostEval.end());
  auto targetOp{mlir::omp::TargetOp::create(builder_, loc, operands)};
  targetOp.setCombined(combined);
  mlir::Region &region{targetOp.getRegion()};
  mlir::Block *block{&region.emplaceBlock()};
  for (mlir::Value value : hostEval) {
    block->addArgument(value.getType(), loc);
  }
  {
    mlir::OpBuilder::InsertionGuard guard{builder_};
    builder_.setInsertionPointToStart(block);
    generate(block->getArguments());
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
  // The device has no runtime to allocate or fill an array.
  mlir::Operation *allocates{nullptr};
  region.walk([&](mlir::Operation *op) {
    if (mlir::isa<merk::ArrayNewOp, merk::ArrayCopyOp, merk::ArrayResizeOp,
            merk::ArrayFillOp>(op)) {
      allocates = op;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  if (allocates) {
    NotYet(at,
        mlir::isa<merk::ArrayFillOp>(allocates)
            ? "filling arrays in target regions"
            : "making arrays in target regions");
    values_ = before;
    return;
  }
  for (mlir::Value input : inputs) {
    if (!FindUpdatedSlot(slots, input) && !Mappable(input.getType())) {
      std::optional<BodyVar> var{VarWithValue(before, goal.nonlocals, input)};
      NotYet(at,
          var ? "values of type " +
                  semantics::TypeToString(*body.vars[*var].type) +
                  " in target regions"
              : std::string{"values that are not scalars in target regions"});
      values_ = before;
      return;
    }
  }
  // An array the region gives is one it was given, updated in place, since
  // the device cannot allocate one: it is mapped back, and after the
  // region is the given array. When the region's operations do not show
  // which, as when it calls a procedure that updates it, the region finds
  // which given array of its type it is, by its pointer, and gives its
  // number, and its pointer too: when the region runs on the host, a
  // procedure it calls may have allocated it.
  llvm::DenseMap<BodyVar, mlir::Value> updated;
  llvm::DenseMap<BodyVar, llvm::SmallVector<mlir::Value>> candidates;
  llvm::DenseMap<BodyVar, mlir::Type> arrayTypes;
  for (BodyVar var : outputs) {
    mlir::Value value{values_.lookup(var)};
    if (!Mappable(value.getType())) {
      NotYet(at,
          "values of type " + semantics::TypeToString(*body.vars[var].type) +
              " in target regions");
      values_ = before;
      return;
    }
    if (mlir::isa<merk::ArrayType>(value.getType())) {
      mlir::Value from{UpdatedFrom(value)};
      if (inputs.contains(from)) {
        updated[var] = from;
        continue;
      }
      arrayTypes[var] = value.getType();
      llvm::SmallVector<mlir::Value> &same{candidates[var]};
      for (mlir::Value input : inputs) {
        if (input.getType() == value.getType()) {
          same.push_back(input);
        }
      }
    }
  }
  llvm::erase_if(outputs, [&](BodyVar var) { return updated.contains(var); });
  // What is mapped back: what an output is, or may be.
  llvm::SmallPtrSet<mlir::Value, 4> mappedBack;
  for (const auto &[var, from] : updated) {
    mappedBack.insert(from);
  }
  for (const auto &[var, same] : candidates) {
    mappedBack.insert(same.begin(), same.end());
  }
  // As Flang maps a scalar: an input is copied in from its slot, an output
  // is stored in its slot, from which it is read after the region.
  auto ptr{mlir::LLVM::LLVMPointerType::get(&context_)};
  llvm::SmallVector<mlir::Value> maps;
  for (mlir::Value input : inputs) {
    if (const UpdatedSlot * u{FindUpdatedSlot(slots, input)}) {
      maps.push_back(MapSlot(input, u->type,
          mlir::omp::ClauseMapFlags::to | mlir::omp::ClauseMapFlags::from,
          mlir::omp::VariableCaptureKind::ByRef, u->name, loc));
      continue;
    }
    std::optional<BodyVar> var{VarWithValue(before, goal.nonlocals, input)};
    llvm::StringRef name{var ? llvm::StringRef{body.vars[*var].name} : ""};
    if (mlir::isa<merk::ArrayType>(input.getType())) {
      maps.push_back(MapArray(input,
          mappedBack.contains(input)
              ? mlir::omp::ClauseMapFlags::to | mlir::omp::ClauseMapFlags::from
              : mlir::omp::ClauseMapFlags::to,
          name, loc));
      continue;
    }
    mlir::Value slot{Slot(input.getType(), loc)};
    mlir::LLVM::StoreOp::create(builder_, loc, input, slot);
    // Named after the variable whose value it is, if it is one's.
    maps.push_back(MapSlot(slot, input.getType(), mlir::omp::ClauseMapFlags::to,
        mlir::omp::VariableCaptureKind::ByCopy, name, loc));
  }
  llvm::SmallVector<mlir::Value> outputSlots;
  llvm::SmallVector<mlir::Type> outputTypes;
  for (BodyVar var : outputs) {
    // An array whose given array the region finds gives its number.
    mlir::Type type{candidates.contains(var) ? builder_.getI64Type()
                                             : values_.lookup(var).getType()};
    outputTypes.push_back(type);
    outputSlots.push_back(Slot(type, loc));
    maps.push_back(
        MapSlot(outputSlots.back(), type, mlir::omp::ClauseMapFlags::from,
            mlir::omp::VariableCaptureKind::ByRef, body.vars[var].name, loc));
  }
  llvm::DenseMap<BodyVar, mlir::Value> pointerSlots;
  for (BodyVar var : outputs) {
    if (candidates.contains(var)) {
      pointerSlots[var] = Slot(ptr, loc);
      maps.push_back(
          MapSlot(pointerSlots[var], ptr, mlir::omp::ClauseMapFlags::from,
              mlir::omp::VariableCaptureKind::ByRef, body.vars[var].name, loc));
    }
  }
  targetOp.getMapVarsMutable().assign(maps);
  // The region reads its inputs from their slots and stores its outputs;
  // an output may be an input's value, as in Y = X. A region that is
  // teams around a parallel region reads them in the parallel, as Flang
  // does, for the reason OmpLoop computes its bounds there.
  mlir::Block *reads{block};
  if (auto teams{mlir::dyn_cast<mlir::omp::TeamsOp>(block->front())};
      teams && outputs.empty()) {
    reads = &teams.getRegion().front();
    if (auto parallel{mlir::dyn_cast<mlir::omp::ParallelOp>(reads->front())}) {
      reads = &parallel.getRegion().front();
    }
  }
  builder_.setInsertionPointToStart(reads);
  llvm::DenseMap<mlir::Value, mlir::Value> loaded;
  for (mlir::Value input : inputs) {
    // An array's argument is its pointer on the device, and so is an
    // updated slot's.
    mlir::Value argument{block->addArgument(ptr, loc)};
    mlir::Value value{FindUpdatedSlot(slots, input) ? argument
            : mlir::isa<merk::ArrayType>(input.getType())
            ? mlir::UnrealizedConversionCastOp::create(
                  builder_, loc, input.getType(), argument)
                  .getResult(0)
            : mlir::Value{mlir::LLVM::LoadOp::create(
                  builder_, loc, input.getType(), argument)}};
    mlir::replaceAllUsesInRegionWith(input, value, targetOp.getRegion());
    loaded[input] = value;
  }
  llvm::SmallVector<mlir::Value> outputArgs;
  for (size_t i = 0; i < outputs.size(); ++i) {
    outputArgs.push_back(block->addArgument(ptr, loc));
  }
  llvm::DenseMap<BodyVar, mlir::Value> pointerArgs;
  for (BodyVar var : outputs) {
    if (candidates.contains(var)) {
      pointerArgs[var] = block->addArgument(ptr, loc);
    }
  }
  // Where the goal ended: the last block of the region.
  mlir::Block *end{&targetOp.getRegion().back()};
  builder_.setInsertionPointToEnd(end);
  if (!Terminated()) {
    for (auto [var, arg] : llvm::zip(outputs, outputArgs)) {
      mlir::Value value{values_.lookup(var)};
      if (auto same{candidates.find(var)}; same != candidates.end()) {
        // The number of the given array whose pointer it is, or -1.
        auto i64{builder_.getI64Type()};
        mlir::LLVM::StoreOp::create(
            builder_, loc, ArrayPointer(value, loc), pointerArgs.lookup(var));
        mlir::Value number{IntegerConstant(i64, -1, loc)};
        for (auto [i, input] : llvm::enumerate(same->second)) {
          mlir::Value is{mlir::LLVM::ICmpOp::create(builder_, loc,
              mlir::LLVM::ICmpPredicate::eq, ArrayPointer(value, loc),
              ArrayPointer(loaded.lookup(input), loc))};
          number = mlir::arith::SelectOp::create(
              builder_, loc, is, IntegerConstant(i64, i, loc), number);
        }
        value = number;
      }
      mlir::LLVM::StoreOp::create(builder_, loc,
          loaded.lookup(value) ? loaded.lookup(value) : value, arg);
    }
    mlir::omp::TerminatorOp::create(builder_, loc);
  }
  builder_.setInsertionPointAfter(targetOp);
  values_ = before;
  for (auto [var, type, slot] : llvm::zip(outputs, outputTypes, outputSlots)) {
    values_[var] = mlir::LLVM::LoadOp::create(builder_, loc, type, slot);
    if (auto same{candidates.find(var)}; same != candidates.end()) {
      // Chosen among the pointers, which the arrays become.
      mlir::Value number{values_[var]};
      mlir::Value chosen{mlir::LLVM::LoadOp::create(
          builder_, loc, ptr, pointerSlots.lookup(var))};
      for (auto [i, input] : llvm::enumerate(same->second)) {
        mlir::Value is{mlir::arith::CmpIOp::create(builder_, loc,
            mlir::arith::CmpIPredicate::eq, number,
            IntegerConstant(type, i, loc))};
        chosen = mlir::LLVM::SelectOp::create(
            builder_, loc, is, ArrayPointer(input, loc), chosen);
      }
      values_[var] = mlir::UnrealizedConversionCastOp::create(
          builder_, loc, arrayTypes.lookup(var), chosen)
                         .getResult(0);
    }
  }
  for (auto [var, from] : updated) {
    values_[var] = from;
  }
}

void MerkGenerator::StubBody(merk::ProcOp proc, parser::CharBlock at) {
  mlir::Location loc{Location(at)};
  mlir::OpBuilder::InsertionGuard guard{builder_};
  builder_.setInsertionPointToStart(proc.addEntryBlock());
  auto string{merk::StringType::get(&context_)};
  mlir::Value what{merk::ConstantOp::create(builder_, loc, string,
      mlir::StringAttr::get(proc.getSymName() + ": " + notYet_, string))};
  merk::ForeignCallOp::create(builder_, loc, mlir::TypeRange{},
      "merk_not_implemented", mlir::ValueRange{what}, /*can_fail=*/false);
  merk::FailOp::create(builder_, loc);
}

// What mode analysis does not check yet in an unchecked procedure's clauses.
static llvm::StringRef UncheckedWhat(semantics::Unchecked reason) {
  switch (reason) {
  case semantics::Unchecked::ModeSpecificClauses:
    return "clauses for one mode";
  case semantics::Unchecked::ParallelConjunction:
    return "parallel conjunctions";
  case semantics::Unchecked::FieldClosure:
    return "closures of functions or fields";
  case semantics::Unchecked::Scope:
    return "trace goals";
  case semantics::Unchecked::Try:
    return "try goals";
  case semantics::Unchecked::Atomic:
    return "atomic goals";
  case semantics::Unchecked::Loop:
    return "loops";
  case semantics::Unchecked::InstParameters:
    return "lambda expressions with inst parameters";
  case semantics::Unchecked::InferredModes:
    return "calls of predicates whose modes are inferred";
  case semantics::Unchecked::TypeErrors:
  case semantics::Unchecked::None:
    break;
  }
  return "goals";
}

// A procedure whose clauses mode analysis did not check has no body to
// generate: it is reported, or with -fstub-unimplemented is a stub, rather
// than left for its callers to fail to link.
void MerkGenerator::GenerateUnchecked(const Procedure &procedure) {
  parser::CharBlock at{procedure.symbol->source};
  std::string what{("procedures with " +
      UncheckedWhat(modes_.UncheckedReason(procedure)) +
      ", which mode analysis does not check yet")
                       .str()};
  if (!stubUnimplemented_) {
    failed_ = false;
    NotYet(at, what);
    return;
  }
  const semantics::Signature *signature{types_.SignatureOf(*procedure.symbol)};
  if (!signature) {
    return;
  }
  llvm::SmallVector<const semantics::Type *> types{
      signature->arguments.begin(), signature->arguments.end()};
  // As in a body: a procedure without a signature code generation can make
  // is left out.
  failed_ = false;
  inBody_ = true;
  llvm::scope_exit leave{[&] { inBody_ = false; }};
  merk::ProcOp proc{ProcFor(procedure, types, at)};
  if (!proc || !proc.isExternal()) {
    return;
  }
  if (procedure.symbol->exported) {
    proc.setPublic();
  }
  notYet_ = "code generation for " + what;
  StubBody(proc, at);
}

void MerkGenerator::GenerateBody(const ProcedureBody &body, bool words) {
  const Procedure &procedure{*body.procedure};
  parser::CharBlock at{procedure.symbol->source};
  failed_ = false;
  inBody_ = true;
  notYet_.clear();
  llvm::scope_exit leave{[&] { inBody_ = false; }};
  const auto *polymorphism{polymorphism_.lookup(&body)};
  if (polymorphism &&
      polymorphism->unchecked != semantics::PolymorphismUnchecked::None) {
    NotYet(at, "this procedure's type_infos");
    return;
  }
  // A clause of an existentially typed procedure has the type it chose where
  // the declaration, and so its callers, have a type variable.
  std::size_t hiddenOutputs{
      body.method ? std::size_t{0} : HiddenOutputs(procedure).size()};
  llvm::SmallVector<const semantics::Type *> types;
  if (hiddenOutputs != 0) {
    llvm::ArrayRef<const semantics::Type *> declared{
        types_.SignatureOf(*procedure.symbol)->arguments};
    types.append(declared.begin(), declared.end());
  } else {
    for (BodyVar var : body.head) {
      types.push_back(body.vars[var].type);
    }
  }
  merk::ProcOp proc{body.method ? MethodProc(body)
          : words              ? WordProcFor(body)
                               : ProcFor(procedure, types, at)};
  if (!proc) {
    return;
  }
  if (procedure.symbol->exported && !body.method && !words) {
    proc.setPublic();
  }
  values_.clear();
  typeInfos_.clear();
  dictionaries_.clear();
  givenVars_.clear();
  current_ = polymorphism;
  procName_ = proc.getSymName().str();
  wordOutputs_.reset();
  mlir::Block *entry{proc.addEntryBlock()};
  mlir::OpBuilder::InsertionGuard guard{builder_};
  builder_.setInsertionPointToStart(entry);
  llvm::SmallVector<BodyVar> outputs;
  unsigned input{0};
  if (polymorphism) {
    for (const semantics::HiddenArgument &argument : polymorphism->hidden) {
      if (argument.output) {
        continue;
      }
      if (argument.typeInfo) {
        typeInfos_[argument.typeInfo->parameter] = entry->getArgument(input++);
      } else {
        dictionaries_[argument.dictionary->parameter] =
            entry->getArgument(input++);
      }
    }
    // A constrained type variable's is in a typeclass_info.
    for (auto [var, source] : llvm::enumerate(polymorphism->sources)) {
      if (source && !typeInfos_.lookup(var)) {
        mlir::Value value{TypeInfoValue(*source, Location(at), at)};
        if (!value) {
          failed_ = true;
          break;
        }
        typeInfos_[var] = value;
      }
    }
  }
  llvm::SmallVector<BodyVar> wordOutputs;
  for (auto [var, mode, type] : llvm::zip(body.head, procedure.modes, types)) {
    if (words && IsOutput(mode)) {
      wordOutputs.push_back(var);
    }
    if (Represent(type)->dummy) {
      continue;
    }
    if (IsOutput(mode)) {
      outputs.push_back(var);
    } else {
      values_[var] = entry->getArgument(input++);
    }
  }
  if (words) {
    wordOutputs_ = wordOutputs;
  }
  Goal(*body.body, body);
  if (failed_) {
    // Calls already made of it keep its declaration.
    proc.getBody().dropAllReferences();
    proc.getBody().getBlocks().clear();
    if (!notYet_.empty()) {
      StubBody(proc, at);
      return;
    }
    proc.setPrivate();
    return;
  }
  if (Terminated()) {
    return; // it cannot succeed
  }
  llvm::SmallVector<mlir::Value> results;
  if (words) {
    results = WordResults(wordOutputs, body, Location(at));
    if (llvm::is_contained(results, mlir::Value{})) {
      failed_ = true;
      return;
    }
    merk::SucceedOp::create(builder_, Location(at), results);
    return;
  }
  if (hiddenOutputs != 0) {
    // The clauses give the same types, but a clause that cannot succeed here
    // may have had its goals in a branch left behind.
    const semantics::PassedInfos *given{nullptr};
    for (const semantics::PassedInfos &clause : polymorphism->results) {
      if (llvm::all_of(clause.typeInfos,
              [&](const semantics::TypeInfo *i) { return CanMake(*i); }) &&
          llvm::all_of(clause.dictionaries,
              [&](const semantics::Dictionary *d) { return CanMake(*d); })) {
        given = &clause;
        break;
      }
    }
    if (!given) {
      NotYet(at, "this procedure's existential types");
      proc.getBody().dropAllReferences();
      proc.getBody().getBlocks().clear();
      proc.setPrivate();
      return;
    }
    for (const semantics::TypeInfo *info : given->typeInfos) {
      results.push_back(TypeInfoValue(*info, Location(at), at));
    }
    for (const semantics::Dictionary *dictionary : given->dictionaries) {
      results.push_back(DictionaryValue(*dictionary, Location(at), at));
    }
  }
  for (BodyVar var : outputs) {
    mlir::Type declared{
        proc.getFunctionType().getResult(static_cast<unsigned>(results.size()))};
    // A clause's type with no values, at a type variable: a word that nothing
    // reads.
    if (Represent(body.vars[var].type)->dummy) {
      results.push_back(merk::ToWordOp::create(builder_, Location(at), declared,
          IntegerConstant(builder_.getI64Type(), 0, Location(at))));
    } else {
      results.push_back(Coerce(ValueOf(var), declared, Location(at)));
    }
  }
  if (llvm::is_contained(results, mlir::Value{})) {
    failed_ = true;
    return;
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
    if (!body.method && body.procedure->symbol->module == moduleName_ &&
        !modes_.ForeignProcOf(*body.procedure)) {
      bodies_[body.procedure] = &body;
    }
  }
  for (const ProcedureBody &body : determinism_.bodies()) {
    if ((body.procedure->symbol->module == moduleName_ || body.method) &&
        (body.method || !modes_.ForeignProcOf(*body.procedure))) {
      GenerateBody(body);
    }
  }
  for (const Procedure &procedure : modes_.procedures()) {
    if (procedure.symbol->module == moduleName_ &&
        !modes_.ForeignProcOf(procedure) &&
        modes_.UncheckedReason(procedure) != semantics::Unchecked::None) {
      GenerateUnchecked(procedure);
    }
  }
  // The word copies that bodies called, and those that copies call; one of a
  // procedure whose body failed has no body either.
  for (std::size_t i{0}; i < pendingCopies_.size(); ++i) {
    const ProcedureBody &body{*pendingCopies_[i]};
    merk::ProcOp typed{procs_.lookup(body.procedure)};
    if (typed && !typed.isExternal()) {
      GenerateBody(body, /*words=*/true);
    }
  }
  GenerateInstances();
  // An exported type's descriptor is here, for the modules that import it,
  // which may not see its constructors. One that cannot be made yet is left
  // to those that use it, which fail to link.
  for (const semantics::Symbol *symbol : resolution_.module().symbols()) {
    if (symbol->kind != semantics::SymbolKind::Type || !symbol->exported ||
        !std::holds_alternative<const parser::DuTypeDefn *>(
            symbol->declaration)) {
      continue;
    }
    semantics::TypeContext &context{types_.types()};
    llvm::SmallVector<const semantics::Type *> variables;
    for (unsigned i{0}; i < symbol->arity; ++i) {
      variables.push_back(context.Variable(i));
    }
    const semantics::Type *canonical{context.Named(*symbol, variables)};
    llvm::StringSet<> seen;
    std::optional<Represented> represented{Represent(canonical)};
    if (represented && (represented->dummy ||
                           Comparable(represented->type, seen))) {
      EnsureDescriptor(*canonical, symbol->source);
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
    const semantics::NameResolution &resolution, semantics::TypeChecking &types,
    semantics::ModeChecking &modes,
    const semantics::DeterminismChecking &determinism,
    const semantics::PolymorphismAnalysis &polymorphism,
    const semantics::LoopTranslation &loops, const TargetInfo &target,
    parser::Messages &messages, bool stubUnimplemented) {
  return MerkGenerator{context, module, tree, tokens, sources, resolution, types, modes,
      determinism, polymorphism, loops, target, messages, stubUnimplemented}
      .Generate();
}

} // namespace merkurius::codegen
