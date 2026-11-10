//===-- lib/CodeGen/MLIRGenBuiltins.cpp - Built-in calls ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The library procedures that code generation builds in: arrays'
// operations, and the arithmetic and comparisons of the primitive types.
//
//===----------------------------------------------------------------------===//

#include "MLIRGenImpl.h"

namespace merkurius::codegen::mlirgen {

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
  // array(L) and from_list(L, A).
  if ((name == "array" || name == "from_list") && HasShape(in, out, 1, 1)) {
    std::optional<Represented> array{Represent(body.vars[out[0]].type)};
    if (!array || !mlir::isa<merk::TermType>(in[0].getType())) {
      return LibraryCallNotYet(goal, symbol);
    }
    return GiveOutputs(out,
        {merk::ArrayFromListOp::create(builder_, loc, array->type, in[0])});
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
  return CompiledArrayCall(goal, symbol);
}

// Temporary (chunk 1 step 6, docs/INPROGRESS.md): array.m's procedures
// that the library compiles, and array.m's own calls when it is the module,
// are called; the others are not yet implemented. The library has the
// sound stub's procedures, which a module checked with -fmmc-uniqueness
// numbers differently, so it cannot call them.
bool MerkGenerator::CompiledArrayCall(
    const Body &goal, const semantics::Symbol &symbol) {
  if (moduleName_ == "array") {
    return false;
  }
  llvm::StringRef name{symbol.name};
  bool compiled{llvm::is_contained({"generate", "generate_foldl",
                                       "generate_foldl2", "foldl", "foldl2",
                                       "foldl3", "foldl4", "foldl5"},
      name)};
  if (compiled && !mmcUniqueness_) {
    return false;
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

// "Field access functions are implemented directly as unifications"
// (reference manual, "Field selection"): the term is deconstructed by the
// field's constructor, which fails if it has another, and for an update
// constructed again, with the new value in the field's place and the term's
// other fields as they were.
void MerkGenerator::FieldAccess(const Body &goal, const Procedure &callee,
    llvm::ArrayRef<BodyVar> arguments, const ProcedureBody &body) {
  const semantics::Symbol &symbol{*callee.symbol};
  parser::CharBlock at{Where(goal.tokens)};
  mlir::Location loc{Location(goal.tokens)};
  bool update{symbol.kind == semantics::SymbolKind::FieldUpdate};
  const auto *constructor{
      std::get<const parser::Constructor *>(symbol.declaration)};
  llvm::StringRef label{
      update ? llvm::StringRef{symbol.name}.drop_back(3) : symbol.name};
  unsigned arity{static_cast<unsigned>(constructor->arguments.size())};
  unsigned index{0};
  while (!constructor->arguments[index]->field ||
      constructor->arguments[index]->field->name != label) {
    ++index;
  }
  const semantics::Type *type{body.vars[arguments[0]].type};
  mlir::Value term{ValueOf(arguments[0])};
  mlir::Value value{update ? ValueOf(arguments[1]) : mlir::Value{}};
  BodyVar result{arguments.back()};
  if (!term || (update && !value) || ValueOf(result)) {
    NotYet(at, "this mode of a field access");
    return;
  }
  llvm::StringRef name{constructor->name->name};
  const semantics::ConstructorTypes *types{ConstructorOf(type, name, arity)};
  HiddenFields hidden;
  std::optional<llvm::SmallVector<mlir::Type>> fields;
  if (types) {
    fields = CellFields(type, *types, hidden);
  }
  if (!fields || !hidden.types.empty()) {
    NotYet(at, "this field access");
    return;
  }
  auto functor{merk::FunctorAttr::get(&context_, name, arity)};
  if (modes_.ops().ConstructorsOf(type)->size() > 1) {
    merk::GuardOp::create(builder_, loc,
        merk::HasFunctorOp::create(builder_, loc, term, functor));
  }
  mlir::ValueRange cell{
      merk::DeconstructOp::create(builder_, loc, *fields, term, functor)
          .getFields()};
  for (auto [field, argument] : llvm::zip(cell, ArgumentsOf(type, *types))) {
    NoteRoot(field, argument);
  }
  if (!update) {
    values_[result] = cell[index];
    return;
  }
  llvm::SmallVector<mlir::Value> updated{cell};
  updated[index] = value;
  values_[result] = merk::ConstructOp::create(builder_, loc,
      mlir::cast<merk::TermType>(Represent(body.vars[result].type)->type),
      functor, updated);
}

bool MerkGenerator::Builtin(const Body &goal, const Procedure &callee,
    llvm::ArrayRef<BodyVar> arguments, const ProcedureBody &body) {
  const semantics::Symbol &symbol{*callee.symbol};
  if (symbol.kind == semantics::SymbolKind::Field ||
      symbol.kind == semantics::SymbolKind::FieldUpdate) {
    FieldAccess(goal, callee, arguments, body);
    return true;
  }
  llvm::StringRef module{symbol.module};
  if (module == "process" && symbol.name == "send" && symbol.arity == 4) {
    if (EnclosingTarget()) {
      return SendInTarget(goal, arguments, body, /*inRegion=*/true);
    }
    if (target_.options().openMPIsTargetDevice) {
      if (SendInTarget(goal, arguments, body, /*inRegion=*/false)) {
        return true;
      }
    } else if (mlir::Value message{ValueOf(arguments[1])};
        message && target_.options().openMP) {
      // The host's compile registers what the device's may send.
      MessageKey(*body.vars[arguments[1]].type, message.getType());
    }
  }
  // On the host, a team's mailbox is the library's, wherever its region
  // runs.
  if (module == "process" && symbol.name == "team_mailbox" &&
      symbol.arity == 4 && target_.options().openMPIsTargetDevice &&
      TeamMailboxOnDevice(goal, arguments, body, EnclosingTarget() != nullptr)) {
    return true;
  }
  if (module == "process" && symbol.name == "receive" && symbol.arity == 6) {
    if (EnclosingTarget()) {
      return ReceiveInTarget(goal, arguments, body, /*inRegion=*/true);
    }
    if (target_.options().openMPIsTargetDevice &&
        ReceiveInTarget(goal, arguments, body, /*inRegion=*/false)) {
      return true;
    }
  }
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
  if (llvm::is_contained({"bool", "exception", "univ"}, module)) {
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

} // namespace merkurius::codegen::mlirgen
