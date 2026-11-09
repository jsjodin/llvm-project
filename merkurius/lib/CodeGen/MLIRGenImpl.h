//===-- lib/CodeGen/MLIRGenImpl.h - MLIR generation's class ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The class that generates a module's merk operations (MLIRGen.h), shared by
// the files that define its members: MLIRGen.cpp, which generates bodies and
// the module, and MLIRGenTypeInfo.cpp, MLIRGenCalls.cpp, MLIRGenUnify.cpp,
// MLIRGenControl.cpp, MLIRGenBuiltins.cpp and MLIRGenOpenMP.cpp.
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_LIB_CODEGEN_MLIRGENIMPL_H_
#define MERKURIUS_LIB_CODEGEN_MLIRGENIMPL_H_

#include "merkurius/CodeGen/MLIRGen.h"
#include "Decomposer.h"
#include "merkurius/CodeGen/Lowering.h"
#include "merkurius/Dialect/Merk/IR/MerkDialect.h"
#include "merkurius/Parser/parse-tree.h"
#include "merkurius/Parser/term-reader.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/SetVector.h"
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

namespace mlirgen {

// The only code point a UTF-8 string has.
std::optional<llvm::UTF32> OnlyCodePoint(llvm::StringRef text);

// The value of an integer literal (reference manual, "Integer literals"):
// decimal, 0b, 0o or 0x digits with `_` between them, or 0' and a
// character, and a suffix for the sized types.
std::optional<llvm::APInt> IntegerValue(
    llvm::StringRef text, unsigned bits);

std::optional<merk::Determinism> MerkDeterminism(
    parser::Determinism determinism);

// That of a procedure, lambda expression or closure, which may have more
// than one solution.
std::optional<merk::Determinism> ClosureDeterminism(
    parser::Determinism determinism);

// Which type variables of `count` the constraints have: those have their
// type_infos in the typeclass_infos.
llvm::SmallVector<bool> ConstrainedVariables(
    unsigned count, llvm::ArrayRef<semantics::ClassConstraint> constraints);

// Whether a type has a type that a goal gave, which only that goal's type_info
// describes.
bool HasGivenType(const semantics::Type &type);

// Whether a call has that many inputs and outputs.
bool HasShape(llvm::ArrayRef<mlir::Value> in, llvm::ArrayRef<BodyVar> out,
    unsigned inputs, unsigned outputs);

// The least number of an existentially typed constructor's existentially
// quantified type variables, in ArgumentsOf's types: past a procedure's, and
// below polymorphism analysis's kExistentialVar on.
inline constexpr unsigned kCellExistential{1u << 29};

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

// In typeInfos_, a type that a goal gave, of TypeKind::Existential, by its
// number from here.
inline constexpr unsigned kGivenTypeKey{semantics::kExistentialVar};

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
      parser::Messages &messages, const GenerationOptions &options)
      : context_{context}, builder_{&context}, moduleName_{module},
        tree_{tree}, tokens_{tokens}, sources_{sources},
        resolution_{resolution}, types_{types},
        modes_{modes}, determinism_{determinism}, loops_{loops},
        target_{target}, messages_{messages},
        stubUnimplemented_{options.stubUnimplemented},
        mmcUniqueness_{options.mmcUniqueness}, traceFlags_{options.traceFlags} {
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
  // How many of an instance's type variables its constraints do not have,
  // each with a type_info in its typeclass_info.
  unsigned UnconstrainedInstanceVariables(const semantics::InstanceHead &);
  // Where a constrained variable's type_info is: the typeclass_info of the
  // first constraint that has it as an argument, `first` being the first
  // typeclass_info's place, and the word of it, after the method table and
  // the superclasses' typeclass_infos.
  std::optional<std::pair<unsigned, unsigned>> ConstrainedPlace(unsigned var,
      llvm::ArrayRef<semantics::ClassConstraint>, unsigned first);
  // Where a closure's captures have the type_info of each type variable:
  // the capture, and the word of it if that is a typeclass_info, or else 0.
  using CapturePlaces =
      llvm::DenseMap<unsigned, std::pair<unsigned, unsigned>>;
  // What a closure's layout says of a capture of this type, as
  // merk.closure's `captures`; null if code generation has failed.
  void NoteArrayArguments(mlir::Block *entry, mlir::Location loc);
  mlir::Attribute UndescribedCapture(const semantics::Type &, parser::CharBlock);
  mlir::Attribute CaptureTypeAttr(const semantics::Type &,
      const CapturePlaces &, parser::CharBlock);
  // Whether a descriptor can be made for the type without an error.
  bool CanDescribe(const semantics::Type &);
  // Null on a device, which never copies a closure.
  mlir::ArrayAttr CapturesAttr(llvm::ArrayRef<mlir::Attribute>);
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
  // words, arrays of those that can be, terms whose constructors' arguments
  // can be, at any depth, and terms of abstract imported types.
  bool Comparable(mlir::Type, llvm::StringSet<> &seen);
  // Whether a type is a term of another module's abstract type, whose
  // constructors this module cannot see, so that its values are unified and
  // compared through its descriptor, which that module defines.
  bool IsAbstract(mlir::Type);
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
  // An argument free before and after, as of mode free >> free, carries
  // nothing either way, so it is not passed, as one of a type with no values
  // is not, and as mmc passes it.
  bool IsUnused(const semantics::ArgMode &mode) {
    return modes_.ops().IsFree(mode.initial) &&
        modes_.ops().IsFree(mode.final);
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
  void GenerateAggregate(const Procedure &);
  // A procedure's body from its foreign_proc: a call of the C function
  // around the foreign_proc's code, which is added to foreignCode_.
  void GenerateForeign(const Procedure &, const parser::ForeignProcPragma &);
  void Goal(const Body &, const ProcedureBody &);
  void GoalBody(const Body &, const ProcedureBody &);
  // Says which type_infos describe the polymorphic values bound so far.
  void AnnotateRoots(const ProcedureBody &);
  // Says which type_infos describe `value` of `type`, if it needs them and
  // this code has them. Returns whether nothing more is needed.
  bool NoteRoot(mlir::Value value, const semantics::Type *type);
  // Makes the descriptors a root's type needs for the runtime to name it.
  void EnsureRootDescriptors(const semantics::Type *);
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
  bool CompiledArrayCall(const Body &, const semantics::Symbol &);
  bool ArrayBuiltin(const Body &, const Procedure &,
      llvm::ArrayRef<BodyVar> arguments, const ProcedureBody &);
  // A call of a field's function or its update function, which no module
  // defines: the unifications that the manual defines it by.
  void FieldAccess(const Body &, const Procedure &,
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
  // What a goal gives says the type of a type variable, for typeInfos_.
  void GiveTypeInfo(const semantics::TypeInfo &, mlir::Value, mlir::Location);
  void GiveTypeInfo(const semantics::Type &, mlir::Value, mlir::Location);
  void GiveDictionary(const semantics::Dictionary &, mlir::Value, mlir::Location);
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
  // In a body that gives words, the conversions of the words that
  // merk.each's continuations and nondet calls' give, to the outputs' types,
  // where nothing uses them.
  static void EraseUnwordedOutputs(mlir::Region &);
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
  // Nondeterministic code (design document, Lowering, "Nondeterministic
  // code"). Whether a procedure or a goal, as the goals around it see it,
  // can have more than one solution.
  bool Nondeterministic(const Procedure &);
  static bool Nondeterministic(const Body &, const ProcedureBody &);
  // Whether a goal with at most one solution as the goals around it see it
  // has parts with more than one, of which it then takes the first.
  bool NeedsCommit(const Body &, const ProcedureBody &);
  void Commit(const Body &, const ProcedureBody &);
  // A disjunction with more than one solution, as merk.disj in merk.each.
  void NondetDisjunction(const Body &, const ProcedureBody &);
  // An if-then-else whose condition has more than one solution, as
  // merk.soft_ite in merk.each.
  void SoftIte(const Body &, const ProcedureBody &);
  // Starts merk.each, for a goal with branches of which one or more has
  // more than one solution, and generates the goal in its generator.
  merk::EachOp BeginEach(mlir::Location);
  // `condition` generates the condition; `otherwiseGoal` is the goal that
  // `otherwise` generates, if it is one.
  void Ite(semantics::TokenRange, llvm::function_ref<void()> condition,
      const Body *thenPart, llvm::function_ref<void()> otherwise,
      const ProcedureBody &, const Body *otherwiseGoal = nullptr);
  void Switch(const Body &, const ProcedureBody &);
  // A switch on a string or a float, whose cases are tested in turn: from
  // the case `next` on, each case's constants as `constants` has them.
  void TestCases(const Body &, std::size_t next, mlir::Value,
      llvm::ArrayRef<llvm::SmallVector<mlir::TypedAttr>> constants,
      const ProcedureBody &);
  // Whether the value is one of the constants.
  mlir::Value IsOneOf(
      mlir::Value, llvm::ArrayRef<mlir::TypedAttr> constants, mlir::Location);
  void Not(const Body &, const ProcedureBody &);
  // A goal with an OpenMP directive (design document, "Target regions").
  void Omp(const Body &, const ProcedureBody &);
  // A trace goal, and its compile_time and run_time conditions.
  void Trace(const Body &, const ProcedureBody &);
  bool CompileTimeHolds(const parser::TraceCondition &);
  mlir::Value RunTimeHolds(const parser::TraceCondition &, mlir::Location);
  // A loop with a loop-associated directive, so far parallel do and target
  // teams distribute parallel do, as omp.parallel, omp.wsloop and
  // omp.loop_nest, with omp.target, omp.teams and omp.distribute for the
  // latter, whose body calls the loop's body procedure.
  void OmpLoop(const Body &, const ProcedureBody &);
  // The leaves of a loop's directive around its omp.loop_nest, in the current
  // block, which is in the target region if it is offloaded.
  void GenerateParallelLoop(const ParallelLoop &, const ProcedureBody &,
      mlir::Location);
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
  // for it, which become the bounds of the host's loop nest.
  void Target(const Body &, const ProcedureBody &,
      llvm::function_ref<void()> generate, bool combined,
      mlir::omp::TargetExecMode mode,
      mlir::ValueRange hostEval = {}, llvm::ArrayRef<UpdatedSlot> slots = {},
      mlir::Value device = {});
  // The value of the leaf's device clause, a null one if it has none, or
  // nothing after an error.
  std::optional<mlir::Value> DeviceOf(const omp::Leaf &, const ProcedureBody &,
      parser::CharBlock at, mlir::Location);
  // The module's omp.declare_reduction for the operator on `type`, an
  // integer or a float, by value, made the first time, as Flang's ReductionProcessor makes
  // and names them.
  mlir::omp::DeclareReductionOp DeclareReduction(
      ReductionOp, mlir::Type type, mlir::Location);
  // The directive's leaves with their clauses; none after reporting a
  // clause that none of them allows.
  mlir::Value MapArray(mlir::Value array, mlir::omp::ClauseMapFlags,
      llvm::StringRef name, mlir::Location);
  // The number of words of the cell of a term that can cross into a target
  // region, by copying its cell: a term of a type of one constructor, of
  // scalars only, two or more, whose pointer has no tag, or a mailbox
  // handle. None for any other type.
  std::optional<unsigned> FlatCell(mlir::Type);
  // Whether a value of the type can cross into a target region.
  bool Mappable(mlir::Type);
  bool WordMessage(mlir::Type);
  // Whether a message of the type is an array of one-word elements.
  bool ArrayMessage(mlir::Type);
  // The message types that device code may send, by their keys.
  llvm::DenseMap<std::uint64_t, const semantics::Type *> messageTypes_;
  std::optional<std::uint64_t> MessageKey(
      const semantics::Type &, mlir::Type represented);
  void RegisterMessageTypes();
  // The target region that code is generated in now, if any.
  mlir::omp::TargetOp EnclosingTarget();
  // process.send(Mailbox, Message, !IO) in a target region: a call of the
  // runtime's merk_device_send, with the message as a word and its
  // type_info made before the region, so that the host's reaches the
  // device as one of the region's values.
  bool SendInTarget(const Body &, llvm::ArrayRef<BodyVar> arguments,
      const ProcedureBody &, bool inRegion);
  bool WriteOnDevice(const Body &, const semantics::Symbol &,
      llvm::ArrayRef<BodyVar> arguments);
  bool ReceiveInTarget(const Body &, llvm::ArrayRef<BodyVar> arguments,
      const ProcedureBody &, bool inRegion);
  mlir::Value TimeoutMilliseconds(mlir::Value timeout, mlir::Location);
  // The map entry of a term's cell of `words` words, by its pointer.
  mlir::Value MapCell(mlir::Value term, unsigned words, llvm::StringRef name,
      mlir::Location);
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
  // the variables' values. In `each`'s generator, they are the solutions of
  // the generator instead, and the continuation's arguments the values.
  void Join(mlir::Operation *op, llvm::ArrayRef<Branch>, const Values &before,
      mlir::Location, merk::EachOp each = nullptr);
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
  // The module's types that its interface declares abstractly, whose
  // merk.type_defs say so.
  llvm::DenseSet<const semantics::Symbol *> abstractTypes_;
  llvm::StringMap<merk::ProcOp> unifyProcs_;
  llvm::StringMap<merk::ProcOp> compareProcs_;
  merk::ProcOp compareArray_;
  // The C functions of the module's foreign_procs.
  std::string foreignCode_;
  // For the body being generated: each variable's value, and whether a
  // construct was not handled.
  llvm::DenseMap<BodyVar, mlir::Value> values_;
  llvm::DenseSet<mlir::Value> annotated_;
  // The type_infos of types goals gave, by type variable or kGivenTypeKey on.
  // The latest of each; another branch's fails NoteRoot's region check.
  llvm::DenseMap<unsigned, mlir::Value> givenInfos_;
  // The types of the roots seen so far, for their descriptors.
  llvm::SetVector<const semantics::Type *> rootTypes_;
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
  // For each type variable a goal gave, the type_info or typeclass_info
  // that has its type_info, and the word of that, or 0.
  llvm::DenseMap<unsigned, std::pair<const void *, unsigned>> givenTypes_;
  void NoteGivenType(const semantics::TypeInfo &);
  void NoteGivenTypes(const semantics::Dictionary &);
  const semantics::ProcedurePolymorphism *current_{nullptr};
  bool failed_{false};
  // Temporary (chunk 1 step 6, docs/INPROGRESS.md): -fstub-unimplemented,
  // for the library's procedures in modes code generation cannot make yet,
  // such as nondet ones; it goes when code generation makes them all.
  bool stubUnimplemented_{false};
  bool mmcUniqueness_{false};
  // --trace-flag=.
  std::vector<std::string> traceFlags_;
  // Whether a procedure's body is being generated, and the first construct
  // of it not handled yet, with -fstub-unimplemented.
  bool inBody_{false};
  // The goal whose first solution merk.commit is being generated for.
  const Body *committing_{nullptr};
  // The body of the committed choice lambda expression being generated,
  // whose first solution is the lambda expression's.
  const Body *committedLambda_{nullptr};
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

} // namespace mlirgen
} // namespace merkurius::codegen

#endif // MERKURIUS_LIB_CODEGEN_MLIRGENIMPL_H_
