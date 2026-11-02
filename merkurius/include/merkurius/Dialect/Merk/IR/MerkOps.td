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
include "mlir/IR/SymbolInterfaces.td"
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
    merk.proc @hello.main_2_0() attributes {determinism = #merk<determinism det>} {
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
    Pure, HasParent<"ProcOp">, ReturnLike, Terminator
  ]> {
  let summary = "produce the procedure's solution";
  let description = [{
    Ends a procedure's body with its outputs, in the order of its results.
  }];
  let arguments = (ins Variadic<AnyType>:$outputs);
  let assemblyFormat = "attr-dict ($outputs^ `:` type($outputs))?";
  let hasVerifier = 1;
}

def Merk_CallOp : Merk_Op<"call", [
    DeclareOpInterfaceMethods<SymbolUserOpInterface>
  ]> {
  let summary = "a first-order call";
  let description = [{
    Calls a procedure with its inputs; its results are the procedure's
    outputs.

    ```mlir
    %y = merk.call @m.double_2_0(%x) : (i64) -> i64
    ```
  }];
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = [{
    $callee `(` $inputs `)` attr-dict `:` functional-type($inputs, $outputs)
  }];
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
