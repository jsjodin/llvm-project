//===-- MerkOps.td - The merk dialect's operations ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The merk dialect's operations (design document, "MLIR dialects"). Only
// those that code generation makes so far are defined.
//
//===----------------------------------------------------------------------===//

#ifndef MERKURIUS_DIALECT_MERK_IR_MERKOPS_TD
#define MERKURIUS_DIALECT_MERK_IR_MERKOPS_TD

include "merkurius/Dialect/Merk/IR/MerkDialect.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// After the Toy tutorial's toy.func (mlir/examples/toy), as func.func is.
def Merk_ProcOp : Merk_Op<"proc", [
    SymbolName, SymbolVisibility, FunctionOpInterface, IsolatedFromAbove
  ]> {
  let summary = "a procedure: one mode of a predicate or function";
  let description = [{
    A procedure, with its determinism. Its arguments are its inputs and its
    results its outputs; arguments of types that have no values, such as
    `io.state`, are left out. One without a body is defined in another
    module.

    ```mlir
    merk.proc @hello.main_2_0() attributes {determinism = #merk.determinism<det>} {
      merk.succeed
    }
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<StrAttr>:$sym_visibility,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs,
    Merk_DeterminismAttr:$determinism
  );
  let regions = (region AnyRegion:$body);

  let builders = [OpBuilder<(ins
    "llvm::StringRef":$name, "mlir::FunctionType":$type,
    "Determinism":$determinism)
  >];
  let extraClassDeclaration = [{
    llvm::ArrayRef<mlir::Type> getArgumentTypes() {
      return getFunctionType().getInputs();
    }
    llvm::ArrayRef<mlir::Type> getResultTypes() {
      return getFunctionType().getResults();
    }
    mlir::Region *getCallableRegion() {
      return isExternal() ? nullptr : &getBody();
    }
  }];
  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;
}

def Merk_SucceedOp : Merk_Op<"succeed", [
    Pure, ParentOneOf<["ProcOp", "IteOp", "SwitchOp"]>, ReturnLike, Terminator
  ]> {
  let summary = "produce the procedure's solution";
  let description = [{
    Ends a procedure's body, or a branch of an if-then-else or switch in
    tail position, with the procedure's outputs, in the order of its
    results.
  }];
  let arguments = (ins Variadic<AnyType>:$outputs);
  let assemblyFormat = "attr-dict ($outputs^ `:` type($outputs))?";
  let hasVerifier = 1;
}

def Merk_CallOp : Merk_Op<"call", [
    CallOpInterface, DeclareOpInterfaceMethods<SymbolUserOpInterface>
  ]> {
  let summary = "a first-order call";
  let description = [{
    Calls a procedure with its inputs; its results are the procedure's
    outputs. It is a call to MLIR's passes, such as the omp dialect's,
    which find the procedures a target region calls.

    ```mlir
    %y = merk.call @m.double_2_0(%x) : (i64) -> i64
    ```
  }];
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$inputs,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs);
  let results = (outs Variadic<AnyType>:$outputs);
  // CallOpInterface's argument and result attributes, which code generation
  // gives no call yet.
  let assemblyFormat = [{
    $callee `(` $inputs `)` (`arg_attrs` $arg_attrs^)? (`res_attrs` $res_attrs^)?
    attr-dict `:` functional-type($inputs, $outputs)
  }];
  let builders = [OpBuilder<(ins "mlir::TypeRange":$outputs,
      "llvm::StringRef":$callee, "mlir::ValueRange":$inputs), [{
    build($_builder, $_state, outputs,
        mlir::FlatSymbolRefAttr::get($_builder.getContext(), callee), inputs,
        /*arg_attrs=*/nullptr, /*res_attrs=*/nullptr);
  }]>];
  let extraClassDeclaration = [{
    // CallOpInterface's.
    mlir::CallInterfaceCallable getCallableForCallee() {
      return getCalleeAttr();
    }
    void setCalleeFromCallable(mlir::CallInterfaceCallable callee) {
      setCalleeAttr(mlir::cast<mlir::FlatSymbolRefAttr>(
          mlir::cast<mlir::SymbolRefAttr>(callee)));
    }
    mlir::Operation::operand_range getArgOperands() { return getInputs(); }
    mlir::MutableOperandRange getArgOperandsMutable() {
      return getInputsMutable();
    }
  }];
}

def Merk_ConstantOp : Merk_Op<"constant", [
    Pure, AllTypesMatch<["value", "result"]>
  ]> {
  let summary = "a constant of a primitive type";
  let description = [{
    An integer, float, character or string constant: a literal, or a
    character's code point.

    ```mlir
    %n = merk.constant 42 : i64
    %s = merk.constant "Hello\0A" : !merk.string
    ```
  }];
  let arguments = (ins TypedAttrInterface:$value);
  let results = (outs AnyType:$result);
  let assemblyFormat = "attr-dict $value";
  let hasVerifier = 1;
}

def Merk_ForeignCallOp : Merk_Op<"foreign_call"> {
  let summary = "a call of a C function";
  let description = [{
    Calls a function of the module's foreign code (the module's
    `merk.foreign_code` attribute), with C's calling convention: the
    function of a procedure's `pragma foreign_proc`.

    With `can_fail`, the function is that of a procedure that can fail:
    its first result is the foreign_proc's SUCCESS_INDICATOR, which the C
    function returns, and it stores the outputs through pointers.

    ```mlir
    %y = merk.foreign_call @merk_foreign_m__double_2_0(%x) : (i64) -> i64
    %ok = merk.foreign_call @merk_foreign_m__positive_1_0(%x) can_fail : (i64) -> i1
    ```
  }];
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$inputs,
      UnitAttr:$can_fail);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = [{
    $callee `(` $inputs `)` (`can_fail` $can_fail^)? attr-dict `:`
    functional-type($inputs, $outputs)
  }];
}

def Merk_YieldOp : Merk_Op<"yield", [
    Pure, ParentOneOf<["IteOp", "SwitchOp", "NotOp"]>, ReturnLike, Terminator
  ]> {
  let summary = "end a branch with the values it binds";
  let description = [{
    Ends a region of an if-then-else, switch or negation that succeeds:
    the condition of `merk.ite` with the values it binds, which the then
    part takes as its arguments, and the then and else parts and the cases
    of `merk.switch` with the operation's results.
  }];
  let arguments = (ins Variadic<AnyType>:$values);
  let assemblyFormat = "attr-dict ($values^ `:` type($values))?";
}

def Merk_FailOp : Merk_Op<"fail", [Terminator]> {
  let summary = "fail";
  let description = [{
    Fails: in the condition of `merk.ite`, the else part runs; in a
    negation, the negation succeeds; elsewhere the innermost of those
    fails, or the procedure, which is then semidet.
  }];
  let assemblyFormat = "attr-dict";
}

def Merk_GuardOp : Merk_Op<"guard"> {
  let summary = "continue if a test succeeded, else fail";
  let description = [{
    A test, such as a comparison or a unification of two bound values:
    execution continues if the flag is true, and fails as `merk.fail` does
    if it is false.

    ```mlir
    %lt = arith.cmpi slt, %x, %y : i64
    merk.guard %lt
    ```
  }];
  let arguments = (ins I1:$condition);
  let assemblyFormat = "$condition attr-dict";
}

def Merk_ThrowIfOp : Merk_Op<"throw_if"> {
  let summary = "throw an exception if a check failed";
  let description = [{
    Throws an exception if the flag is true, as the library's checks do,
    such as that of a division by zero. Until exceptions are implemented,
    an exception is not caught: the program writes "Uncaught Mercury
    exception:" and the exception, as mmc writes it, to the standard error,
    and exits with status 1.

    ```mlir
    merk.throw_if %zero, "domain_error(\"int.\\'//\\': division by zero\")"
    ```
  }];
  let arguments = (ins I1:$condition, StrAttr:$exception);
  let assemblyFormat = "$condition `,` $exception attr-dict";
}

def Merk_IteOp : Merk_Op<"ite", [RecursiveMemoryEffects]> {
  let summary = "if-then-else";
  let description = [{
    Runs the condition; if it succeeds, the then part, with the values the
    condition binds as its arguments; if it fails, the else part. Its
    results are those the parts that succeed yield: the variables both
    bind.

    ```mlir
    %y = merk.ite -> i64 {
      %lt = arith.cmpi slt, %x, %zero : i64
      merk.guard %lt
      merk.yield
    } then {
      %n = arith.subi %zero, %x : i64
      merk.yield %n : i64
    } else {
      merk.yield %x : i64
    }
    ```
  }];
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$condition,
      SizedRegion<1>:$thenRegion, SizedRegion<1>:$elseRegion);
  let assemblyFormat = [{
    (`->` type($results)^)? $condition `then` $thenRegion `else` $elseRegion
    attr-dict
  }];
  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins "mlir::TypeRange":$results), [{
    $_state.addTypes(results);
    for (int i = 0; i < 3; ++i) {
      $_state.addRegion();
    }
  }]>];
}

def Merk_SwitchOp : Merk_Op<"switch", [RecursiveMemoryEffects]> {
  let summary = "branch on a primitive value";
  let description = [{
    Runs the case whose constants include the value, one region for each
    case, in the order of `cases`. If no case has it, the switch fails if
    it has `can_fail`, and cannot happen otherwise.

    ```mlir
    %r = merk.switch %n : i64 -> i64 cases [[0], [1, 2]] can_fail {
      merk.yield %a : i64
    }, {
      merk.yield %b : i64
    }
    ```
  }];
  let arguments = (ins AnySignlessInteger:$value, ArrayAttr:$cases,
      UnitAttr:$can_fail);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region VariadicRegion<SizedRegion<1>>:$caseRegions);
  let assemblyFormat = [{
    $value `:` type($value) (`->` type($results)^)? `cases` $cases
    (`can_fail` $can_fail^)? $caseRegions attr-dict
  }];
  let hasVerifier = 1;
  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins "mlir::TypeRange":$results,
      "mlir::Value":$value, "mlir::ArrayAttr":$cases, "bool":$canFail), [{
    $_state.addTypes(results);
    $_state.addOperands(value);
    $_state.addAttribute(getCasesAttrName($_state.name), cases);
    if (canFail) {
      $_state.addAttribute(getCanFailAttrName($_state.name),
          $_builder.getUnitAttr());
    }
    for (std::size_t i = 0; i < cases.size(); ++i) {
      $_state.addRegion();
    }
  }]>];
}

def Merk_NotOp : Merk_Op<"not", [RecursiveMemoryEffects, NoRegionArguments]> {
  let summary = "negation";
  let description = [{
    Succeeds, binding nothing, if its goal fails, and fails if it succeeds.
  }];
  let regions = (region SizedRegion<1>:$goal);
  let assemblyFormat = "$goal attr-dict";
}

def Merk_EntryOp : Merk_Op<"entry", [
    DeclareOpInterfaceMethods<SymbolUserOpInterface>
  ]> {
  let summary = "the program's main procedure";
  let description = [{
    Names the procedure the program starts with, the main module's
    `main(di, uo)`, which the runtime calls.

    ```mlir
    merk.entry @hello.main_2_0
    ```
  }];
  let arguments = (ins FlatSymbolRefAttr:$proc);
  let assemblyFormat = "$proc attr-dict";
}

#endif // MERKURIUS_DIALECT_MERK_IR_MERKOPS_TD
