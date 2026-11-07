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
    Pure, ParentOneOf<["ProcOp", "IteOp", "SwitchOp", "DisjOp", "NondetCallOp",
        "NondetCallClosureOp", "NondetCallMethodOp", "EachOp", "SoftIteOp"]>,
    ReturnLike, Terminator
  ]> {
  let summary = "produce the procedure's solution";
  let description = [{
    Ends a procedure's body, or a branch of an if-then-else or switch in
    tail position, with the procedure's outputs, in the order of its
    results. In a nondet or multi procedure it gives one solution, and the
    procedure goes on to its next when the code that called it asks for
    one; nondeterministic code is in a disjunction's branch, a nondet
    call's continuation or the continuation of `merk.each`.
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

def Merk_ToWordOp : Merk_Op<"to_word", [Pure]> {
  let summary = "a value as the word a type variable holds";
  let description = [{
    The value's bits: a narrower integer zero-extended, a float's bits, a
    pointer's address.

    ```mlir
    %w = merk.to_word %x : f64
    ```
  }];
  let arguments = (ins AnyType:$value);
  let results = (outs Merk_WordType:$word);
  let assemblyFormat = "$value attr-dict `:` type($value)";
}

def Merk_FromWordOp : Merk_Op<"from_word", [Pure]> {
  let summary = "the value that a type variable's word holds";
  let description = [{
    The inverse of `merk.to_word`, at a type the caller knows.

    ```mlir
    %x = merk.from_word %w : f64
    ```
  }];
  let arguments = (ins Merk_WordType:$word);
  let results = (outs AnyType:$value);
  let assemblyFormat = "$word attr-dict `:` type($value)";
}

def Merk_TypeInfoOp : Merk_Op<"type_info", [Pure]> {
  let summary = "the type_info of a type constructor applied to arguments";
  let description = [{
    The constructor is named with its arity. A static value if all the
    arguments are, and otherwise a new cell.

    ```mlir
    %i = merk.type_info "int/0"
    %l = merk.type_info "list.list/1"(%t)
    ```
  }];
  let arguments = (ins StrAttr:$constructor,
      Variadic<Merk_TypeInfoType>:$arguments);
  let results = (outs Merk_TypeInfoType:$info);
  let assemblyFormat = "$constructor (`(` $arguments^ `)`)? attr-dict";
}

def Merk_TypeCtorOp : Merk_Op<"type_ctor", [HasParent<"mlir::ModuleOp">]> {
  let summary = "a type constructor's descriptor";
  let description = [{
    The static record every `merk.type_info` of the constructor points to
    (design document, "Type descriptors"). `kind` names what its values are,
    `term` for a type of constructors, whose layout `term` gives. `linkage`
    is `external` in the module that defines the type, `declaration` in one
    that imports it, and `internal` or `linkonce`; `unify` and `compare` are
    the entries that take the type_info and two words. `functors` has each
    constructor in the order declared, as its name then its fields' types: a
    field's is a type parameter's number, or an array of a constructor's name
    and its arguments' types.

    ```mlir
    merk.type_ctor "list.list/1" module "list" name "list" kind "term" linkage "linkonce" arity 1 term !merk.term<"list.list(T0)"> unify @unify_entry.list.list/1 compare @compare_entry.list.list/1
    ```
  }];
  let arguments = (ins StrAttr:$constructor, StrAttr:$module, StrAttr:$name,
      StrAttr:$kind, StrAttr:$linkage, I64Attr:$arity,
      OptionalAttr<TypeAttr>:$term, OptionalAttr<FlatSymbolRefAttr>:$unify,
      OptionalAttr<FlatSymbolRefAttr>:$compare, OptionalAttr<ArrayAttr>:$functors);
  let assemblyFormat = [{
    $constructor `module` $module `name` $name `kind` $kind `linkage` $linkage
    `arity` $arity (`term` $term^)? (`unify` $unify^)?
    (`compare` $compare^)? (`functors` $functors^)? attr-dict
  }];
}

def Merk_TypeInfoWordOp : Merk_Op<"type_info_word", [Pure]> {
  let summary = "a type_info from a word of a type_info's cell";
  let description = [{
    Word 0 of the cell is the descriptor, so an argument's is from word 1,
    or word 2 for a tuple's, whose word 1 is the number of arguments.

    ```mlir
    %t = merk.type_info_word %list, 1
    ```
  }];
  let arguments = (ins Merk_TypeInfoType:$info, I64Attr:$word);
  let results = (outs Merk_TypeInfoType:$result);
  let assemblyFormat = "$info `,` $word attr-dict";
}

def Merk_InstanceOp : Merk_Op<"instance", [
    Symbol, SymbolName, SymbolVisibility, HasParent<"mlir::ModuleOp">
  ]> {
  let summary = "a type class instance's method table";
  let description = [{
    The procedures of its methods, each taking the `typeclass_info` first and
    then the method's arguments as its declaration in the type class has them.
    One without `methods` is defined in another module.

    ```mlir
    merk.instance @"merk.instance.m.show(int)" methods [@m.show_2_0$int]
    ```
  }];
  let arguments = (ins SymbolNameAttr:$sym_name,
      OptionalAttr<StrAttr>:$sym_visibility,
      OptionalAttr<SymbolRefArrayAttr>:$methods);
  let assemblyFormat = [{
    $sym_name (`visibility` $sym_visibility^)? (`methods` $methods^)? attr-dict
  }];
}

def Merk_TypeclassInfoOp : Merk_Op<"typeclass_info", [
    Pure, DeclareOpInterfaceMethods<SymbolUserOpInterface>
  ]> {
  let summary = "the typeclass_info of an instance";
  let description = [{
    A static value if all the arguments are, and otherwise a new cell. The
    arguments are the type class's superclasses' `typeclass_info`s, its
    types' `type_info`s, and those of the instance's type variables and
    constraints (Merk_TypeclassInfoType).

    ```mlir
    %d = merk.typeclass_info @"merk.instance.m.show(int)"(%int : !merk.type_info)
    ```
  }];
  let arguments = (ins FlatSymbolRefAttr:$instance,
      Variadic<AnyTypeOf<[Merk_TypeInfoType, Merk_TypeclassInfoType]>>:$arguments);
  let results = (outs Merk_TypeclassInfoType:$info);
  let assemblyFormat = [{
    $instance (`(` $arguments^ `:` type($arguments) `)`)? attr-dict
  }];
}

def Merk_TypeclassInfoWordOp : Merk_Op<"typeclass_info_word", [Pure]> {
  let summary = "a type_info or typeclass_info from a word of a typeclass_info";
  let description = [{
    Word 0 is the method table, so a superclass's is from word 1.

    ```mlir
    %t = merk.typeclass_info_word %d, 1 : !merk.type_info
    ```
  }];
  let arguments = (ins Merk_TypeclassInfoType:$info, I64Attr:$word);
  let results = (outs AnyTypeOf<[Merk_TypeInfoType, Merk_TypeclassInfoType]>:$result);
  let assemblyFormat = "$info `,` $word attr-dict `:` type($result)";
}

def Merk_CallMethodOp : Merk_Op<"call_method"> {
  let summary = "a call of a type class's method through a typeclass_info";
  let description = [{
    Calls the procedure in the given place of the instance's method table,
    with the `typeclass_info` and then the inputs. A specialization pass sees
    the instance when the `typeclass_info` comes from a `merk.typeclass_info`.
    A nondet or multi method is called with `merk.nondet_call_method`, which
    outlining rewrites as a semidet call of this operation whose last inputs
    are a continuation and its environment.

    ```mlir
    %y = merk.call_method %d[0](%x) #merk.determinism<det> : (!merk.word) -> !merk.word
    ```
  }];
  let arguments = (ins Merk_TypeclassInfoType:$info, I64Attr:$method,
      Variadic<AnyType>:$inputs, Merk_DeterminismAttr:$determinism);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = [{
    $info `[` $method `]` `(` $inputs `)` $determinism attr-dict `:` functional-type($inputs, $outputs)
  }];
}

def Merk_ClosureOp : Merk_Op<"closure", [
    Pure, DeclareOpInterfaceMethods<SymbolUserOpInterface>
  ]> {
  let summary = "a closure of an entry and the values it captures";
  let description = [{
    A new cell of the entry's code, the number of captured values, and the
    captured values. The entry is a procedure taking the closure and then a
    word for each of the closure's inputs, and giving a word for each of its
    outputs; it reads the captured values with `merk.closure_word`.

    ```mlir
    %p = merk.closure @"m.add_3_0$closure1"(%one)
    ```
  }];
  let arguments = (ins FlatSymbolRefAttr:$entry,
      Variadic<Merk_WordType>:$captured);
  let results = (outs Merk_ClosureType:$closure);
  let assemblyFormat = "$entry (`(` $captured^ `)`)? attr-dict";
}

def Merk_ClosureWordOp : Merk_Op<"closure_word", [Pure]> {
  let summary = "a value a closure captured";
  let description = [{
    The captured value at that place, from 0.

    ```mlir
    %w = merk.closure_word %p, 0
    ```
  }];
  let arguments = (ins Merk_ClosureType:$closure, I64Attr:$index);
  let results = (outs Merk_WordType:$word);
  let assemblyFormat = "$closure `,` $index attr-dict";
}

def Merk_CallClosureOp : Merk_Op<"call_closure"> {
  let summary = "a higher-order call";
  let description = [{
    Calls the closure's entry with the closure and the inputs, a word for each
    of the closure's inputs, including one of a type with no values; its
    results are a word for each output. The determinism is the closure's inst's.
    A nondet or multi closure is called with `merk.nondet_call_closure`,
    which outlining rewrites as a semidet call of this operation whose last
    inputs are a continuation and its environment.

    ```mlir
    %y = merk.call_closure %p(%x) #merk.determinism<det> : (!merk.word) -> !merk.word
    ```
  }];
  let arguments = (ins Merk_ClosureType:$closure,
      Variadic<AnyTypeOf<[Merk_WordType, Merk_ContType, Merk_EnvType]>>:$inputs,
      Merk_DeterminismAttr:$determinism);
  let results = (outs Variadic<Merk_WordType>:$outputs);
  let assemblyFormat = [{
    $closure `(` $inputs `)` $determinism attr-dict `:` functional-type($inputs, $outputs)
  }];
}

def Merk_UnifyWordOp : Merk_Op<"unify_word"> {
  let summary = "whether two words of a type are equal, by its type_info";
  let description = [{
    Calls the unify entry of the type_info's descriptor.

    ```mlir
    %same = merk.unify_word %t, %x, %y
    ```
  }];
  let arguments = (ins Merk_TypeInfoType:$info, Merk_WordType:$x,
      Merk_WordType:$y);
  let results = (outs I1:$result);
  let assemblyFormat = "$info `,` $x `,` $y attr-dict";
}

def Merk_CompareWordOp : Merk_Op<"compare_word"> {
  let summary = "how two words of a type are ordered, by its type_info";
  let description = [{
    Calls the compare entry of the type_info's descriptor, and gives its
    `builtin.comparison_result`.

    ```mlir
    %r = merk.compare_word %t, %x, %y : !merk.term<"builtin.comparison_result">
    ```
  }];
  let arguments = (ins Merk_TypeInfoType:$info, Merk_WordType:$x,
      Merk_WordType:$y);
  let results = (outs Merk_TermType:$result);
  let assemblyFormat = "$info `,` $x `,` $y attr-dict `:` type($result)";
}

def Merk_CompareTypeInfoOp : Merk_Op<"compare_type_info"> {
  let summary = "how two types are ordered, by their type_infos";
  let description = [{
    Negative if the first is less, zero if they are the same type, and
    positive if it is greater, in mmc's order: by the type constructors'
    modules, names and arities, a primitive type's module being `builtin`
    and a tuple's name `tuple`, and then by their arguments in order. An
    existentially typed constructor's argument is compared by its type
    first.

    ```mlir
    %c = merk.compare_type_info %t, %u
    ```
  }];
  let arguments = (ins Merk_TypeInfoType:$x, Merk_TypeInfoType:$y);
  let results = (outs I32:$result);
  let assemblyFormat = "$x `,` $y attr-dict";
}

def Merk_StringCompareOp : Merk_Op<"string_compare"> {
  let summary = "how two strings are ordered, as C's strcmp";
  let description = [{
    Negative if the first is less, zero if they are equal, and positive if
    it is greater.

    ```mlir
    %c = merk.string_compare %x, %y
    ```
  }];
  let arguments = (ins Merk_StringType:$x, Merk_StringType:$y);
  let results = (outs I32:$result);
  let assemblyFormat = "$x `,` $y attr-dict";
}

def Merk_StringEqualOp : Merk_Op<"string_equal"> {
  let summary = "whether two strings are equal";
  let description = [{
    ```mlir
    %same = merk.string_equal %x, %y
    ```
  }];
  let arguments = (ins Merk_StringType:$x, Merk_StringType:$y);
  let results = (outs I1:$result);
  let assemblyFormat = "$x `,` $y attr-dict";
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

//===----------------------------------------------------------------------===//
// Terms of types defined by constructors.
//===----------------------------------------------------------------------===//

def Merk_TypeDefOp : Merk_Op<"type_def", [HasParent<"mlir::ModuleOp">]> {
  let summary = "the constructors of a type";
  let description = [{
    Lists the constructors of the type that `!merk.term<name>` names, in
    the order the type's definition declares them, with their arguments'
    types, for lowering to choose the type's layout from. A module has one
    for each such type its procedures use, its own or imported.

    ```mlir
    merk.type_def "list.list(int)" [#merk.constructor<"[]">,
        #merk.constructor<"[|]"(i64, !merk.term<"list.list(int)">)>]
    ```
  }];
  let arguments = (ins StrAttr:$name,
      TypedArrayAttrBase<Merk_ConstructorAttr, "constructors">:$constructors);
  let assemblyFormat = "$name $constructors attr-dict";
}

def Merk_ConstructOp : Merk_Op<"construct", [Pure]> {
  let summary = "a term of a constructor and its arguments";
  let description = [{
    X = f(Y1, ..., Yn), with X free: the term of the constructor with the
    arguments, which are the constructor's arguments' types. An
    existentially typed constructor's hidden fields' values come first
    (Merk_ConstructorAttr).

    ```mlir
    %l = merk.construct #merk.functor<"[|]", 2>(%h, %t : i64, !merk.term<"list.list(int)">) -> !merk.term<"list.list(int)">
    %b = merk.construct #merk.functor<"box", 1>(%d, %x : !merk.typeclass_info, !merk.word) -> !merk.term<"m.box">
    ```
  }];
  let arguments = (ins Merk_FunctorAttr:$functor,
      Variadic<AnyType>:$arguments);
  let results = (outs Merk_TermType:$term);
  let assemblyFormat = [{
    qualified($functor) (`(` $arguments^ `:` type($arguments) `)`)? attr-dict
    `->` qualified(type($term))
  }];
  let hasVerifier = 1;
}

// Last call modulo constructor (design document, Lowering, "Deterministic
// and semidet code", "Last call modulo constructor"): a cell is made with a
// hole, whose address a call is given to fill, so that the call is a tail
// call.
def Merk_HoleOp : Merk_Op<"hole"> {
  let summary = "a field to be filled after its cell is made";
  let description = [{
    An argument of merk.construct that a later merk.ref_store fills,
    through the address merk.field_ref gives. It is null until then, so
    that the collector finds no pointer there. It has effects, so that two
    holes are never one value and two cells with holes never one cell.

    ```mlir
    %h = merk.hole : !merk.term<"list.list(int)">
    %c = merk.construct #merk.functor<"[|]", 2>(%x, %h : i64, !merk.term<"list.list(int)">) -> !merk.term<"list.list(int)">
    ```
  }];
  let results = (outs AnyType:$value);
  let assemblyFormat = "attr-dict `:` type($value)";
}

def Merk_FieldRefOp : Merk_Op<"field_ref", [Pure]> {
  let summary = "the address of a field of a cell";
  let description = [{
    The address of argument `index` of a term made by merk.construct of the
    functor, whose argument there is merk.hole. The index counts the
    construction's arguments, an existentially typed constructor's hidden
    fields too.

    ```mlir
    %r = merk.field_ref %c, #merk.functor<"[|]", 2>[1] : !merk.term<"list.list(int)">
    ```
  }];
  let arguments = (ins Merk_TermType:$term, Merk_FunctorAttr:$functor,
      ConfinedAttr<I64Attr, [IntNonNegative]>:$index);
  let results = (outs Merk_RefType:$ref);
  let assemblyFormat = [{
    $term `,` qualified($functor) `[` $index `]` attr-dict `:` qualified(type($term))
  }];
  let builders = [OpBuilder<(ins "::mlir::Value":$term,
      "::merkurius::merk::FunctorAttr":$functor, "std::int64_t":$index), [{
    build($_builder, $_state,
        ::merkurius::merk::RefType::get($_builder.getContext()), term, functor,
        $_builder.getI64IntegerAttr(index));
  }]>];
}

def Merk_RefStoreOp : Merk_Op<"ref_store"> {
  let summary = "fill a hole";
  let description = [{
    Stores the value in the hole whose address the reference is, as
    merk.construct would have stored it there. The only store into a cell
    after it is made.

    ```mlir
    merk.ref_store %t, %r : !merk.term<"list.list(int)">
    ```
  }];
  let arguments = (ins AnyType:$value, Merk_RefType:$ref);
  let assemblyFormat = "$value `,` $ref attr-dict `:` type($value)";
}

def Merk_DeconstructOp : Merk_Op<"deconstruct", [NoMemoryEffect]> {
  let summary = "the arguments of a term whose constructor is known";
  let description = [{
    X = f(Y1, ..., Yn), with X bound and the Yi free, where X is known to
    be of f: in a switch's case for f, or after merk.has_functor. Its
    results are the arguments, after an existentially typed constructor's
    hidden fields. It is not speculatable, since another constructor's term
    has no such arguments.

    ```mlir
    %h, %t = merk.deconstruct #merk.functor<"[|]", 2> %l : !merk.term<"list.list(int)"> -> i64, !merk.term<"list.list(int)">
    ```
  }];
  let arguments = (ins Merk_TermType:$term, Merk_FunctorAttr:$functor);
  let results = (outs Variadic<AnyType>:$fields);
  let assemblyFormat = [{
    qualified($functor) $term attr-dict `:` qualified(type($term))
    (`->` type($fields)^)?
  }];
  let hasVerifier = 1;
}

def Merk_HasFunctorOp : Merk_Op<"has_functor", [NoMemoryEffect]> {
  let summary = "whether a term is of a constructor";
  let description = [{
    The test of X = f(...) with X bound, whose result `merk.guard` takes.

    ```mlir
    %is = merk.has_functor #merk.functor<"[]", 0> %l : !merk.term<"list.list(int)">
    ```
  }];
  let arguments = (ins Merk_TermType:$term, Merk_FunctorAttr:$functor);
  let results = (outs I1:$result);
  let assemblyFormat = [{
    qualified($functor) $term attr-dict `:` qualified(type($term))
  }];
  let builders = [OpBuilder<(ins "mlir::Value":$term, "FunctorAttr":$functor), [{
    build($_builder, $_state, $_builder.getI1Type(), term, functor);
  }]>];
}

//===----------------------------------------------------------------------===//
// Arrays, the library module array's predicates, which code generation
// builds in. Those that update an array do so in place and give the same
// array, since uniqueness checking has made sure its old version is dead;
// the others allocate one. None checks its indices: merk.check_index does.
//===----------------------------------------------------------------------===//

def Merk_ArrayNewOp : Merk_Op<"array_new"> {
  let summary = "allocate an array of a size, each element the same";
  let description = [{
    Without `init`, the elements are undefined: for an empty array.
  }];
  let arguments = (ins I64:$size, Optional<AnyType>:$init);
  let results = (outs Merk_ArrayType:$array);
  let assemblyFormat = "$size (`,` $init^ `:` type($init))? attr-dict `->` type($array)";
  let hasVerifier = 1;
}

def Merk_ArraySizeOp : Merk_Op<"array_size"> {
  let summary = "the number of an array's elements";
  let arguments = (ins Merk_ArrayType:$array);
  let results = (outs I64:$size);
  let assemblyFormat = "$array attr-dict `:` type($array)";
}

def Merk_ArrayGetOp : Merk_Op<"array_get"> {
  let summary = "an array's element";
  let arguments = (ins Merk_ArrayType:$array, I64:$index);
  let results = (outs AnyType:$element);
  let assemblyFormat = "$array `[` $index `]` attr-dict `:` type($array) `->` type($element)";
  let hasVerifier = 1;
}

def Merk_ArraySetOp : Merk_Op<"array_set", [
    AllTypesMatch<["array", "result"]>
  ]> {
  let summary = "set an array's element, in place";
  let arguments = (ins Merk_ArrayType:$array, I64:$index, AnyType:$element);
  let results = (outs Merk_ArrayType:$result);
  let assemblyFormat = [{
    $array `[` $index `]` `,` $element attr-dict `:` type($array) `,` type($element)
  }];
  let hasVerifier = 1;
}

def Merk_ArrayFillOp : Merk_Op<"array_fill", [
    AllTypesMatch<["array", "result"]>
  ]> {
  let summary = "set an array's elements from one index to another, in place";
  let description = [{
    Sets the elements from `lo` to `hi`, inclusive, to `element`; none if
    `lo` is greater than `hi`.
  }];
  let arguments = (ins Merk_ArrayType:$array, I64:$lo, I64:$hi, AnyType:$element);
  let results = (outs Merk_ArrayType:$result);
  let assemblyFormat = [{
    $array `[` $lo `,` $hi `]` `,` $element attr-dict `:` type($array) `,` type($element)
  }];
  let hasVerifier = 1;
}

def Merk_ArrayCopyOp : Merk_Op<"array_copy", [
    AllTypesMatch<["array", "result"]>
  ]> {
  let summary = "a new array with an array's elements";
  let arguments = (ins Merk_ArrayType:$array);
  let results = (outs Merk_ArrayType:$result);
  let assemblyFormat = "$array attr-dict `:` type($array)";
}

def Merk_ArrayResizeOp : Merk_Op<"array_resize", [
    AllTypesMatch<["array", "result"]>
  ]> {
  let summary = "a new array of a size, with an array's elements";
  let description = [{
    A new array of `size` elements: as many of the array's as fit, and then
    `init`; without `init`, `size` is no greater than the array's.
  }];
  let arguments = (ins Merk_ArrayType:$array, I64:$size, Optional<AnyType>:$init);
  let results = (outs Merk_ArrayType:$result);
  let assemblyFormat = [{
    $array `,` $size (`,` $init^ `:` type($init))? attr-dict `:` type($array)
  }];
  let hasVerifier = 1;
}

def Merk_CheckIndexOp : Merk_Op<"check_index"> {
  let summary = "throw index_out_of_bounds unless an index is an array's";
  let description = [{
    Throws the library's `index_out_of_bounds` exception unless `index` is
    from 0 to the array's size less one, with the message mmc's library
    gives: `what`, then ": index I not in range [0, Max]".

    ```mlir
    merk.check_index %a[%i], "array.lookup" : !merk.array<i64>
    ```
  }];
  let arguments = (ins Merk_ArrayType:$array, I64:$index, StrAttr:$what);
  let assemblyFormat = "$array `[` $index `]` `,` $what attr-dict `:` type($array)";
}

def Merk_YieldOp : Merk_Op<"yield", [
    Pure, ParentOneOf<["IteOp", "SwitchOp", "NotOp", "AggregateOp"]>,
    ReturnLike, Terminator
  ]> {
  let summary = "end a branch with the values it binds";
  let description = [{
    Ends a region of an if-then-else, switch or negation that succeeds:
    the condition of `merk.ite` with the values it binds, which the then
    part takes as its arguments, and the then and else parts and the cases
    of `merk.switch` with the operation's results. It ends the step of
    `merk.aggregate` with whether to go on, then the new accumulators.
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
  let summary = "branch on a primitive value or a term's constructor";
  let description = [{
    Runs the case whose constants include the value, one region for each
    case, in the order of `cases`: integer constants for an integer, and
    functors for a term. If no case has it, the switch fails if it has
    `can_fail`, and cannot happen otherwise. A case of one functor takes
    the term apart with merk.deconstruct.

    ```mlir
    %r = merk.switch %n : i64 -> i64 cases [[0], [1, 2]] can_fail {
      merk.yield %a : i64
    }, {
      merk.yield %b : i64
    }
    merk.switch %l : !merk.term<"list.list(int)"> cases
        [[#merk.functor<"[]", 0>], [#merk.functor<"[|]", 2>]] {
      ...
    }
    ```
  }];
  let arguments = (ins AnyTypeOf<[AnySignlessInteger, Merk_TermType]>:$value,
      ArrayAttr:$cases,
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

//===----------------------------------------------------------------------===//
// Nondeterministic code (design document, Lowering, "Nondeterministic code")
//===----------------------------------------------------------------------===//

def Merk_DisjOp : Merk_Op<"disj", [
    RecursiveMemoryEffects, NoRegionArguments, Terminator
  ]> {
  let summary = "a disjunction with more than one solution";
  let description = [{
    Runs its branches in turn: each gives its solutions, and when it fails,
    or the code that its solutions went to asks for another, the next
    branch runs; after the last, the disjunction fails. A branch ends as
    the code around the disjunction does, giving a solution with
    `merk.succeed` or `merk.solution`, or failing. A disjunction with at
    most one solution is an if-then-else.

    ```mlir
    merk.disj {
      merk.succeed %h : i64
    }, {
      merk.nondet_call @m.member_2_1(%t : !merk.term<"list.list(int)">) {
      ^bb0(%x: i64):
        merk.succeed %x : i64
      }
    }
    ```
  }];
  let regions = (region VariadicRegion<SizedRegion<1>>:$branches);
  let assemblyFormat = "$branches attr-dict";
  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins "unsigned":$count), [{
    for (unsigned i = 0; i < count; ++i) {
      $_state.addRegion();
    }
  }]>];
}

def Merk_NondetCallOp : Merk_Op<"nondet_call", [
    DeclareOpInterfaceMethods<SymbolUserOpInterface>, Terminator
  ]> {
  let summary = "a call of a nondet or multi procedure";
  let description = [{
    Calls a nondet or multi procedure, and runs its continuation once for
    each of its solutions, with the procedure's outputs as the
    continuation's arguments; after the last, it fails. The continuation is
    the rest of the code, which ends as the code around the call does.
  }];
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$inputs);
  let regions = (region SizedRegion<1>:$continuation);
  let assemblyFormat = [{
    $callee `(` ($inputs^ `:` type($inputs))? `)` $continuation attr-dict
  }];
  let hasVerifier = 1;
  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins "llvm::StringRef":$callee,
      "mlir::ValueRange":$inputs), [{
    $_state.addAttribute(getCalleeAttrName($_state.name),
        mlir::FlatSymbolRefAttr::get($_builder.getContext(), callee));
    $_state.addOperands(inputs);
    $_state.addRegion();
  }]>];
}

def Merk_NondetCallClosureOp : Merk_Op<"nondet_call_closure", [Terminator]> {
  let summary = "a higher-order call of a nondet or multi closure";
  let description = [{
    Calls the closure's entry with the closure and the inputs, a word for each
    of the closure's inputs, as `merk.call_closure` does, and runs its
    continuation once for each of its solutions, with a word for each output
    as the continuation's arguments; after the last, it fails.

    ```mlir
    merk.nondet_call_closure %p(%x : !merk.word) {
    ^bb0(%y: !merk.word):
      ...
    }
    ```
  }];
  let arguments = (ins Merk_ClosureType:$closure,
      Variadic<Merk_WordType>:$inputs);
  let regions = (region SizedRegion<1>:$continuation);
  let assemblyFormat = [{
    $closure `(` ($inputs^ `:` type($inputs))? `)` $continuation attr-dict
  }];
  let hasVerifier = 1;
  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins "mlir::Value":$closure,
      "mlir::ValueRange":$inputs), [{
    $_state.addOperands(closure);
    $_state.addOperands(inputs);
    $_state.addRegion();
  }]>];
}

def Merk_NondetCallMethodOp : Merk_Op<"nondet_call_method", [Terminator]> {
  let summary = "a call of a nondet or multi method through a typeclass_info";
  let description = [{
    Calls the procedure in the given place of the instance's method table,
    with the `typeclass_info` and then the inputs, as `merk.call_method`
    does, and runs its continuation once for each of its solutions, with the
    method's outputs as the continuation's arguments; after the last, it
    fails.

    ```mlir
    merk.nondet_call_method %d[0](%x : !merk.word) {
    ^bb0(%y: !merk.word):
      ...
    }
    ```
  }];
  let arguments = (ins Merk_TypeclassInfoType:$info, I64Attr:$method,
      Variadic<AnyType>:$inputs);
  let regions = (region SizedRegion<1>:$continuation);
  let assemblyFormat = [{
    $info `[` $method `]` `(` ($inputs^ `:` type($inputs))? `)` $continuation attr-dict
  }];
  let hasVerifier = 1;
  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins "mlir::Value":$info, "std::int64_t":$method,
      "mlir::ValueRange":$inputs), [{
    $_state.addOperands(info);
    $_state.addOperands(inputs);
    $_state.addAttribute(getMethodAttrName($_state.name),
        $_builder.getI64IntegerAttr(method));
    $_state.addRegion();
  }]>];
}

def Merk_EachOp : Merk_Op<"each", [RecursiveMemoryEffects, Terminator]> {
  let summary = "run the rest of the code for each solution of a goal";
  let description = [{
    Runs its generator, a goal with more than one solution, and its
    continuation for each solution that the generator gives with
    `merk.solution`, with its values as the continuation's arguments; after
    the last, it fails. It is a branching goal followed by more goals, which
    its branches share rather than each having a copy.

    ```mlir
    merk.each {
      merk.disj {
        %one = merk.constant 1 : i64
        merk.solution %one : i64
      }, {
        %two = merk.constant 2 : i64
        merk.solution %two : i64
      }
    } then {
    ^bb0(%x: i64):
      ...
    }
    ```
  }];
  let regions = (region SizedRegion<1>:$generator,
      SizedRegion<1>:$continuation);
  let assemblyFormat = "$generator `then` $continuation attr-dict";
  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins), [{
    $_state.addRegion();
    $_state.addRegion();
  }]>];
}

def Merk_SoftIteOp : Merk_Op<"soft_ite", [RecursiveMemoryEffects, Terminator]> {
  let summary = "if-then-else whose condition has more than one solution";
  let description = [{
    Runs its condition, a goal with more than one solution, and its then
    part for each solution that the condition gives with `merk.solution`,
    with its values as the then part's arguments; if the condition gives
    none, it runs its else part instead. The then part and the else part
    end as the code around the operation does. Once the condition has given
    a solution, the else part is not run, even if the then part fails: the
    if-then-else of a condition with more than one solution, a soft cut.

    ```mlir
    merk.soft_ite {
      merk.nondet_call @m.member_2_0(%l : !merk.term<"list.list(int)">) {
      ^bb0(%x: i64):
        merk.solution %x : i64
      }
    } then {
    ^bb0(%x: i64):
      ...
    } else {
      ...
    }
    ```
  }];
  let regions = (region SizedRegion<1>:$condition, SizedRegion<1>:$thenRegion,
      SizedRegion<1>:$elseRegion);
  let assemblyFormat =
      "$condition `then` $thenRegion `else` $elseRegion attr-dict";
  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins), [{
    $_state.addRegion();
    $_state.addRegion();
    $_state.addRegion();
  }]>];
}

def Merk_CommitOp : Merk_Op<"commit", [RecursiveMemoryEffects]> {
  let summary = "the first solution of a goal";
  let description = [{
    Runs its generator, a goal with more than one solution in a context
    where only its first is wanted, until it gives one with
    `merk.solution`; its results are that solution's values. It fails if
    the generator has none.
  }];
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$generator);
  let assemblyFormat = "(`->` type($results)^)? $generator attr-dict";
  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins "mlir::TypeRange":$results), [{
    $_state.addTypes(results);
    $_state.addRegion();
  }]>];
}

def Merk_AggregateOp : Merk_Op<"aggregate", [RecursiveMemoryEffects]> {
  let summary = "fold a step over the solutions of a goal";
  let description = [{
    Runs its generator, a goal with any number of solutions, which it gives
    with `merk.solution`, and its step once for each, in the order the
    generator gives them, as soon as it gives it. The step takes the
    solution's values and then the accumulators, and ends with `merk.yield`
    of whether to go on, an `i1`, and the new accumulators. The operands
    are the accumulators' first values; the results, their last, after the
    generator has no more solutions or a step said not to go on. It
    succeeds exactly once: the generator's remaining solutions are not
    tried after it, and backtracking into the generator does not undo a
    step, so the accumulators can be unique (`solutions.unsorted_aggregate`,
    `unsorted_aggregate2` and `do_while`).

    ```mlir
    %sum = merk.aggregate (%zero : i64) -> i64 {
      merk.nondet_call @m.g_1_0() {
      ^bb0(%x: i64):
        merk.solution %x : i64
      }
    } step {
    ^bb0(%x: i64, %acc: i64):
      %go = arith.constant true
      %s = arith.addi %acc, %x : i64
      merk.yield %go, %s : i1, i64
    }
    ```
  }];
  let arguments = (ins Variadic<AnyType>:$initial);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$generator, SizedRegion<1>:$step);
  let assemblyFormat = [{
    (`(` $initial^ `:` type($initial) `)` `->` type($results))?
    $generator `step` $step attr-dict
  }];
  let hasVerifier = 1;
  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins "mlir::ValueRange":$initial), [{
    $_state.addOperands(initial);
    $_state.addTypes(initial.getTypes());
    $_state.addRegion();
    $_state.addRegion();
  }]>];
}

def Merk_SolutionOp : Merk_Op<"solution", [Pure, Terminator]> {
  let summary = "give a solution of a goal whose solutions are wanted";
  let description = [{
    Gives a solution of the generator of the innermost `merk.each`,
    `merk.commit` or `merk.aggregate`, or the condition of
    `merk.soft_ite`, with its values.
  }];
  let arguments = (ins Variadic<AnyType>:$values);
  let assemblyFormat = "attr-dict ($values^ `:` type($values))?";
  let hasVerifier = 1;
}

// After outlining, which leaves none of the operations above but
// merk.disj: a nondet or multi procedure is a semidet one that takes a
// continuation and its environment after its inputs, whose success means
// that the code its solutions went to asked to stop.

def Merk_ContRefOp : Merk_Op<"cont_ref", [
    Pure, DeclareOpInterfaceMethods<SymbolUserOpInterface>
  ]> {
  let summary = "a continuation procedure";
  let arguments = (ins FlatSymbolRefAttr:$callee);
  let results = (outs Merk_ContType:$result);
  let assemblyFormat = "$callee attr-dict";
}

def Merk_EnvNewOp : Merk_Op<"env_new"> {
  let summary = "an environment in this frame";
  let description = [{
    An environment of the given values, then of slots of the given types
    that `merk.env_set` fills, in the frame of the procedure it is in. A
    call that passes it is not a tail call.
  }];
  let arguments = (ins Variadic<AnyType>:$values, TypeArrayAttr:$layout);
  let results = (outs Merk_EnvType:$result);
  let assemblyFormat = "`(` ($values^ `:` type($values))? `)` attr-dict `layout` $layout";
}

def Merk_EnvGetOp : Merk_Op<"env_get"> {
  let summary = "a value of an environment";
  let arguments = (ins Merk_EnvType:$env, I32Attr:$index,
      TypeArrayAttr:$layout);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$env `[` $index `]` attr-dict `:` type($result) `layout` $layout";
}

def Merk_EnvSetOp : Merk_Op<"env_set"> {
  let summary = "fill a slot of an environment";
  let arguments = (ins Merk_EnvType:$env, I32Attr:$index, AnyType:$value,
      TypeArrayAttr:$layout);
  let assemblyFormat = "$env `[` $index `]` `,` $value attr-dict `:` type($value) `layout` $layout";
}

def Merk_CallContOp : Merk_Op<"call_cont"> {
  let summary = "give a solution to a continuation";
  let description = [{
    Calls a continuation with its environment and a solution's values. It
    succeeds if the continuation asks to stop, and fails otherwise, as a
    call of a semidet procedure does.
  }];
  let arguments = (ins Merk_ContType:$cont, Merk_EnvType:$env,
      Variadic<AnyType>:$values);
  let assemblyFormat = "$cont `(` $env (`,` $values^)? `)` attr-dict (`:` type($values)^)?";
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
