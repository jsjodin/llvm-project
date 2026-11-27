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
    module. An input that the procedure clobbers, as of mode `di`, has the
    argument attribute `merk.clobbered`: its caller does not use it again,
    so a GPU process's procedure may free it (FreeAtLastUse).

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
    One without `methods` is defined in another module. `shape` says what
    follows the table in an instance's `typeclass_info`: how many superclass
    `typeclass_info`s, then how many `type_info`s, then how many constraints'
    `typeclass_info`s. It is the word before the first method, for the
    message copy and the collector.

    ```mlir
    merk.instance @"merk.instance.m.show(int)" methods [@m.show_2_0$int]
    ```
  }];
  let arguments = (ins SymbolNameAttr:$sym_name,
      OptionalAttr<StrAttr>:$sym_visibility,
      OptionalAttr<SymbolRefArrayAttr>:$methods,
      OptionalAttr<DenseI64ArrayAttr>:$shape);
  let assemblyFormat = [{
    $sym_name (`visibility` $sym_visibility^)? (`methods` $methods^)?
    (`shape` $shape^)? attr-dict
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
    A new cell of the entry's code, the address of a static layout, and the
    captured values. `captures` has what the layout says of each captured
    value: its type, as a field's in a descriptor, with a type variable as
    the place of its type_info among the captured values; or "type_info" or
    "typeclass_info". Without it, as on
    a device, which never copies a closure, the layout is null. The entry is a procedure taking the closure and then a
    word for each of the closure's inputs, and giving a word for each of its
    outputs; it reads the captured values with `merk.closure_word`.

    ```mlir
    %p = merk.closure @"m.add_3_0$closure1"(%one)
    ```
  }];
  let arguments = (ins FlatSymbolRefAttr:$entry,
      Variadic<Merk_WordType>:$captured, OptionalAttr<ArrayAttr>:$captures);
  let results = (outs Merk_ClosureType:$closure);
  let assemblyFormat =
      "$entry (`(` $captured^ `)`)? (`captures` $captures^)? attr-dict";
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

def Merk_ReceiveOp : Merk_Op<"receive", [
    AttrSizedOperandSegments, DeclareOpInterfaceMethods<SymbolUserOpInterface>
  ]> {
  let summary = "a receive: one message, which some case selects";
  let description = [{
    A receive goal (design document, "The receive goal"), as desugaring
    made it, or with `plain` a receive of one mailbox,
    `process.receive/5`: a case for each of its mailboxes, in the order
    first written, whose matcher, a semidet procedure, takes the case's
    captured values and a message, and gives what it selected, or fails.
    A receive goal's matcher gives the arm and the pattern's variables as
    a word, and the result is `process.receive_result(R)`:
    `receive_message` of what a matcher gave, or `receive_nothing` at the
    timeout. A plain receive's result is `process.received(R)`, `message`
    or `timeout`, and its mailbox closed is the library's error for one
    mailbox. `result_info` is R's type_info.

    A case's matcher is a procedure's symbol, or a unit when the selector
    was only a value, a closure: that case's one captured value is the
    closure, which the receive calls.

    `message_infos` are the type_infos of each case's messages, and
    `captured_counts` how many of `captured` each case's matcher takes,
    first case first. `captures`, one for each case, says what each
    captured value is, as `merk.closure`'s does, for lowering where it
    makes closures of the matchers; without it, as on a device, they have
    no layout.

    ```mlir
    %r = merk.receive %t with %ti matchers [@m.p$lambda0] counts [1]
        on(%box : !merk.term<"process.mailbox(m.msg)">) infos(%mi)
        captured(%ref : i64) captures [[["int/0"]]]
        : !merk.term<"process.timeout">
        -> !merk.term<"process.receive_result(T0)">
    ```
  }];
  let arguments = (ins Merk_TermType:$timeout, Merk_TypeInfoType:$result_info,
      Variadic<AnyType>:$mailboxes, Variadic<Merk_TypeInfoType>:$message_infos,
      Variadic<AnyType>:$captured, ArrayAttr:$matchers,
      DenseI64ArrayAttr:$captured_counts, OptionalAttr<ArrayAttr>:$captures,
      UnitAttr:$plain);
  let results = (outs Merk_TermType:$result);
  let assemblyFormat = [{
    $timeout `with` $result_info (`plain` $plain^)?
    `matchers` $matchers `counts` $captured_counts
    `on` `(` $mailboxes `:` type($mailboxes) `)`
    `infos` `(` $message_infos `)`
    (`captured` `(` $captured^ `:` type($captured) `)`)?
    (`captures` $captures^)?
    attr-dict `:` qualified(type($timeout)) `->` qualified(type($result))
  }];
  let hasVerifier = 1;
}

def Merk_SendOp : Merk_Op<"send"> {
  let summary = "a send: one message, to a mailbox";
  let description = [{
    `process.send/4`: the message, given the type_info of its type, is
    copied into the mailbox. On the host, lowering makes it the runtime's
    send, `merk_send`, with the handle's cell. With a `key` and a
    `ring_kind`, which code generation gives a send that a device runs
    (MLIRGenOpenMP.cpp, SendInTarget), it is the device runtime's send in
    any compile: of a word (ring kind 0), of an array of one-word elements
    by its pointer (1), or of a term copied with its type_info (2). Such a
    send names its message's type, `message_type`, as Mercury writes it;
    the device's message carries the name's key (`Key`), which the host's
    compile registers with the type's type_info. Any other on a device is
    the library's call, in code that the device does not run.

    ```mlir
    merk.send %m to %box with %ti : i64, !merk.term<"process.mailbox(int)">
    merk.send %m to %box as "int" ring_kind 0
        : i64, !merk.term<"process.mailbox(int)">
    ```
  }];
  let arguments = (ins AnyType:$message, AnyType:$mailbox,
      Optional<Merk_TypeInfoType>:$info, OptionalAttr<StrAttr>:$message_type,
      OptionalAttr<I64Attr>:$ring_kind);
  let assemblyFormat = [{
    $message `to` $mailbox (`with` $info^)? (`as` $message_type^)?
    (`ring_kind` $ring_kind^)? attr-dict `:` type($message) `,`
    type($mailbox)
  }];
  let extraClassDeclaration = [{
    // The key of a message type's name, by which a device's message names
    // its type to the host's runtime, and by which the host's compile
    // registers the type's type_info (MLIRGenOpenMP.cpp,
    // RegisterMessageTypes): a hash of the name.
    static std::uint64_t Key(llvm::StringRef messageType);
  }];
  let hasVerifier = 1;
}

def Merk_NewMailboxOp : Merk_Op<"new_mailbox"> {
  let summary = "a new mailbox";
  let description = [{
    `process.new_mailbox/3`: a mailbox of the current process for messages
    whose type's type_info is given. A mailbox is abstract outside the
    library, which alone builds one, so lowering makes it the library's
    call (Lowering.cpp, ExpandProcessOps).

    ```mlir
    %box = merk.new_mailbox with %ti : !merk.term<"process.mailbox(T0)">
    ```
  }];
  let arguments = (ins Merk_TypeInfoType:$info);
  let results = (outs AnyType:$mailbox);
  let assemblyFormat = [{
    `with` $info attr-dict `:` type($mailbox)
  }];
}

def Merk_CloseMailboxOp : Merk_Op<"close_mailbox"> {
  let summary = "a mailbox closed";
  let description = [{
    `process.close_mailbox/3`, given the type_info of the mailbox's
    messages: its queued messages are dropped, and later sends to it are
    too. On the host, lowering makes it the runtime's `merk_close`, with
    the handle's cell.

    ```mlir
    merk.close_mailbox %box with %ti : !merk.term<"process.mailbox(int)">
    ```
  }];
  let arguments = (ins AnyType:$mailbox, Merk_TypeInfoType:$info);
  let assemblyFormat = [{
    $mailbox `with` $info attr-dict `:` type($mailbox)
  }];
}

def Merk_StopOp : Merk_Op<"stop"> {
  let summary = "the process that owns a mailbox stopped";
  let description = [{
    `process.stop/3`, given the type_info of the mailbox's messages: the
    mailbox's owner is stopped. On the host, lowering makes it the
    runtime's `merk_stop`, with the handle's cell.

    ```mlir
    merk.stop %box with %ti : !merk.term<"process.mailbox(int)">
    ```
  }];
  let arguments = (ins AnyType:$mailbox, Merk_TypeInfoType:$info);
  let assemblyFormat = [{
    $mailbox `with` $info attr-dict `:` type($mailbox)
  }];
}

def Merk_MonitorOp : Merk_Op<"monitor"> {
  let summary = "a monitor of a mailbox";
  let description = [{
    `process.monitor/6`: when the target mailbox's owner ends, the watcher
    is sent the function's message of why, a `down_info`. The type_infos
    are the target's and the watcher's message types. The result, a
    `monitor_ref`, is abstract outside the library, which alone builds
    one, so lowering makes it the library's call.

    ```mlir
    %ref = merk.monitor %target, %watcher, %f with %ti, %wi
        : !merk.term<"process.mailbox(int)">,
          !merk.term<"process.mailbox(m.watch)">
        -> !merk.term<"process.monitor_ref">
    ```
  }];
  let arguments = (ins AnyType:$target, AnyType:$watcher,
      Merk_ClosureType:$function, Merk_TypeInfoType:$target_info,
      Merk_TypeInfoType:$watcher_info);
  let results = (outs AnyType:$ref);
  let assemblyFormat = [{
    $target `,` $watcher `,` $function `with` $target_info `,` $watcher_info
    attr-dict `:` type($target) `,` type($watcher) `->` type($ref)
  }];
}

def Merk_DemonitorOp : Merk_Op<"demonitor"> {
  let summary = "a monitor removed";
  let description = [{
    `process.demonitor/3`: the monitor sends nothing more. Its
    `monitor_ref` is abstract outside the library, so lowering makes it
    the library's call.

    ```mlir
    merk.demonitor %ref : !merk.term<"process.monitor_ref">
    ```
  }];
  let arguments = (ins AnyType:$ref);
  let assemblyFormat = [{
    $ref attr-dict `:` type($ref)
  }];
}

def Merk_SelectOp : Merk_Op<"select", [AttrSizedOperandSegments]> {
  let summary = "a receive goal, as lowering makes it";
  let description = [{
    Lowering's form of a `merk.receive`, made first in lowering and
    flattened into a loop (Lowering.cpp, ExpandReceives and LowerSelect):
    the word that a case's matcher gave for the message it selected, or
    failure once `milliseconds` pass (forever if negative). On a device,
    in a GPU process, it fails too once every mailbox is closed;
    `ring_kinds` are the mailboxes' ring kinds (device-ring.h) and
    `message_infos` their messages' type_infos as C pointers, for copied
    terms. On the host, with no ring kinds, every mailbox closed is the
    library's error, and `message_infos` are the type_infos, which make
    what a monitor left a message. `message_types` are the types the
    matchers take the messages in. A matcher that is a unit calls the
    case's closure, its one captured value, as `merk.receive`'s does; with
    `plain`, the receive is of one mailbox (`merk.receive`'s `plain`).
    Not made by code generation.

    ```mlir
    %w = merk.select %ms matchers [@m.p$lambda0] counts [1] kinds [0]
        on(%box : !merk.term<"process.mailbox(int)">) infos(%ti : !llvm.ptr)
        captured(%ref : i64) : !merk.word
    ```
  }];
  let arguments = (ins I64:$milliseconds, Variadic<AnyType>:$mailboxes,
      Variadic<AnyType>:$message_infos, Variadic<AnyType>:$captured,
      ArrayAttr:$matchers, DenseI64ArrayAttr:$captured_counts,
      DenseI64ArrayAttr:$ring_kinds, TypeArrayAttr:$message_types,
      UnitAttr:$plain);
  let results = (outs Merk_WordType:$word);
  let assemblyFormat = [{
    $milliseconds (`plain` $plain^)? `matchers` $matchers `counts` $captured_counts
    `kinds` $ring_kinds `types` $message_types
    `on` `(` $mailboxes `:` type($mailboxes) `)`
    `infos` `(` $message_infos `:` type($message_infos) `)`
    (`captured` `(` $captured^ `:` type($captured) `)`)?
    attr-dict `:` qualified(type($word))
  }];
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

    With `moves`, a GPU process's send of an array or a term
    (`merk_device_send_array`, `merk_device_send_term`) moves the message's
    arrays to the receiver rather than copying them: it is their last use,
    nothing frees them after it, and the sender's code must not use them
    again (docs/INPROGRESS.md, Messages between GPU processes on one GPU,
    step 2). FreeAtLastUse sets it, and lowering calls the send's `_moving`
    function, which gives each array that is a block of its own to the
    message, and copies any other.

    ```mlir
    %y = merk.foreign_call @merk_foreign_m__double_2_0(%x) : (i64) -> i64
    %ok = merk.foreign_call @merk_foreign_m__positive_1_0(%x) can_fail : (i64) -> i1
    merk.foreign_call @merk_device_send_array(%m, %k, %p) moves : (!merk.term<"process.mailbox(array.array(float))">, i64, !llvm.ptr) -> ()
    ```
  }];
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$inputs,
      UnitAttr:$can_fail, UnitAttr:$moves);
  let results = (outs Variadic<AnyType>:$outputs);
  let assemblyFormat = [{
    $callee `(` $inputs `)` (`can_fail` $can_fail^)? (`moves` $moves^)?
    attr-dict `:` functional-type($inputs, $outputs)
  }];
  let hasVerifier = 1;
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
    for each such type its procedures use, its own or imported, but for an
    abstract type of another module, whose constructors it cannot see: such
    a `!merk.term` has no layout, and its values are pointers.

    `abstract` marks a type of the module that its interface declares
    abstractly, which other modules pass as a pointer, so its layout must
    be one that passes as a pointer does: a pointer or a word.

    `open` marks the open layout of a type T, named `open.open(T)`, whose
    fields are all words (design document, Lowering, "Open layouts"):
    primary tag 7 is a variable's, so its constructors with arguments have
    tags 0 to 6, and an enumeration's constants are shifted past the tag.

    ```mlir
    merk.type_def "list.list(int)" [#merk.constructor<"[]">,
        #merk.constructor<"[|]"(i64, !merk.term<"list.list(int)">)>]
    merk.type_def "secret.box(T0)" [#merk.constructor<"box"(!merk.word, i64)>] abstract
    merk.type_def "open.open(list.list(int))" [#merk.constructor<"[]">,
        #merk.constructor<"[|]"(!merk.word, !merk.word)>] open
    ```
  }];
  let arguments = (ins StrAttr:$name,
      TypedArrayAttrBase<Merk_ConstructorAttr, "constructors">:$constructors,
      UnitAttr:$abstract, UnitAttr:$open);
  let assemblyFormat = [{
    $name $constructors (`abstract` $abstract^)? (`open` $open^)? attr-dict
  }];
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
  let results = (outs Merk_TermOrRecord:$term);
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
  let summary = "where a field of a cell is";
  let description = [{
    Argument `index` of a term made by merk.construct of the functor, whose
    argument there is merk.hole, as the term's pointer and the field's offset
    in bytes from it. Two values, not an address, so that the pointer is one
    the collector can relocate (design document, Lowering, "GC metadata and
    safepoints"). The index counts the construction's arguments, an
    existentially typed constructor's hidden fields too.

    ```mlir
    %r, %o = merk.field_ref %c, #merk.functor<"[|]", 2>[1] : !merk.term<"list.list(int)">
    ```
  }];
  let arguments = (ins Merk_TermType:$term, Merk_FunctorAttr:$functor,
      ConfinedAttr<I64Attr, [IntNonNegative]>:$index);
  let results = (outs Merk_RefType:$ref, I64:$offset);
  let assemblyFormat = [{
    $term `,` qualified($functor) `[` $index `]` attr-dict `:` qualified(type($term))
  }];
  let builders = [OpBuilder<(ins "::mlir::Value":$term,
      "::merkurius::merk::FunctorAttr":$functor, "std::int64_t":$index), [{
    build($_builder, $_state,
        ::merkurius::merk::RefType::get($_builder.getContext()),
        $_builder.getI64Type(), term, functor,
        $_builder.getI64IntegerAttr(index));
  }]>];
}

def Merk_RefStoreOp : Merk_Op<"ref_store"> {
  let summary = "fill a hole";
  let description = [{
    Stores the value in the hole at the offset from the reference, as
    merk.construct would have stored it there. The only store into a cell
    after it is made.

    ```mlir
    merk.ref_store %t, %r[%o] : !merk.term<"list.list(int)">
    ```
  }];
  let arguments = (ins AnyType:$value, Merk_RefType:$ref, I64:$offset);
  let assemblyFormat = "$value `,` $ref `[` $offset `]` attr-dict `:` type($value)";
}

def Merk_RootTypeOp : Merk_Op<"root_type", [NoMemoryEffect]> {
  let summary = "the type_infos that say a polymorphic value's type";
  let description = [{
    For a `!merk.word`, the type_info of its type variable; for a term whose
    type has type variables, theirs, by increasing number. Made by MLIRGen, and
    read and erased by the pass that makes `merk.live`.
  }];
  let arguments = (ins AnyType:$value, Variadic<Merk_TypeInfoType>:$infos);
  let assemblyFormat = "$value `:` type($value) `(` $infos `)` attr-dict";
}

def Merk_LiveOp : Merk_Op<"live"> {
  let summary = "what a collection at the next safepoint must find";
  let description = [{
    Before an operation at which a collection can happen: the values live
    across it, each with its merk type, which lowering loses. Made after
    outlining, for lowering, which turns it into the safepoint's deopt bundle
    (docs/INPROGRESS.md, Statepoints and stack maps).
  }];
  let arguments = (ins Variadic<AnyType>:$values, ArrayAttr:$types);
  let assemblyFormat = "`(` $values `)` $types attr-dict `:` type($values)";
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
  let arguments = (ins Merk_TermOrRecord:$term, Merk_FunctorAttr:$functor);
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
  let arguments = (ins Merk_TermOrRecord:$term, Merk_FunctorAttr:$functor);
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

def Merk_ArrayNewOp : Merk_Op<"array_new", [AttrSizedOperandSegments]> {
  let summary = "allocate an array of a size, each element the same";
  let description = [{
    Without `init`, the elements are undefined: for an empty array. The
    elements' width is their type's (runtime/array.h); for elements of a
    type variable, `!merk.word`, the type_info of what it stands for, from
    which the runtime finds it, and without one they are words.
  }];
  let arguments = (ins I64:$size, Optional<AnyType>:$init,
      Optional<Merk_TypeInfoType>:$element_type_info);
  let results = (outs Merk_ArrayType:$array);
  let assemblyFormat = [{
    $size (`,` $init^ `:` type($init))? (`width` `of` $element_type_info^)?
    attr-dict `->` type($array)
  }];
  let hasVerifier = 1;
}

def Merk_ArraySizeOp : Merk_Op<"array_size"> {
  let summary = "the number of an array's elements";
  let arguments = (ins Merk_ArrayType:$array);
  let results = (outs I64:$size);
  let assemblyFormat = "$array attr-dict `:` type($array)";
}

def Merk_ArrayWordsOp : Merk_Op<"array_words"> {
  let summary = "the words of an array, its header's and its elements'";
  let description = [{
    What a copy of the array copies: one word for its header, and its
    elements' words, which narrow elements share (runtime/array.h).
  }];
  let arguments = (ins Merk_ArrayType:$array);
  let results = (outs I64:$words);
  let assemblyFormat = "$array attr-dict `:` type($array)";
}

def Merk_ArrayGetOp : Merk_Op<"array_get"> {
  let summary = "an array's element";
  let description = [{
    For elements of a type variable, or of a type whose layout the module
    does not see, `element_type_info` is the type_info of their type, by
    which the runtime boxes a flat record that the array holds inline
    (runtime/record.h); without one, they are not records. The same holds
    for `merk.array_set`, `merk.array_fill` and `merk.array_resize`.
  }];
  let arguments = (ins Merk_ArrayType:$array, I64:$index,
      Optional<Merk_TypeInfoType>:$element_type_info);
  let results = (outs AnyType:$element);
  let assemblyFormat = [{
    $array `[` $index `]` (`of` $element_type_info^)? attr-dict `:`
    type($array) `->` type($element)
  }];
  let hasVerifier = 1;
}

def Merk_ArraySetOp : Merk_Op<"array_set", [
    AllTypesMatch<["array", "result"]>
  ]> {
  let summary = "set an array's element, in place";
  let arguments = (ins Merk_ArrayType:$array, I64:$index, AnyType:$element,
      Optional<Merk_TypeInfoType>:$element_type_info);
  let results = (outs Merk_ArrayType:$result);
  let assemblyFormat = [{
    $array `[` $index `]` `,` $element (`of` $element_type_info^)? attr-dict
    `:` type($array) `,` type($element)
  }];
  let hasVerifier = 1;
}

def Merk_ArrayElementRefOp : Merk_Op<"array_element_ref"> {
  let summary = "the address of an array's element, for an atomic operation";
  let description = [{
    The address of element `index` of an array of words, of ints or floats,
    which an OpenMP atomic operation updates in place (`%$omp atomic`,
    docs/INPROGRESS.md, Atomic updates of array elements). It is a generic
    pointer, or in the target's address space for atomic operations
    (TargetInfo::AtomicAddressSpace): on the host, the heap pointer
    cast out of the collector's address space, used before any safepoint.
  }];
  let arguments = (ins Merk_ArrayType:$array, I64:$index);
  let results = (outs AnyType:$address);
  let assemblyFormat = [{
    $array `[` $index `]` attr-dict `:` type($array) `->` type($address)
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
  let arguments = (ins Merk_ArrayType:$array, I64:$lo, I64:$hi, AnyType:$element,
      Optional<Merk_TypeInfoType>:$element_type_info);
  let results = (outs Merk_ArrayType:$result);
  let assemblyFormat = [{
    $array `[` $lo `,` $hi `]` `,` $element (`of` $element_type_info^)?
    attr-dict `:` type($array) `,` type($element)
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

def Merk_ArrayFromListOp : Merk_Op<"array_from_list"> {
  let summary = "a new array of a list's elements, in order";
  let description = [{
    `list` is a `list.list(T)` and the array's elements are `T`s, each the
    word the list's cell holds; with `reversed`, the last first, as
    `array.from_reverse_list` has them. The elements' width is as for
    `merk.array_new`.
  }];
  let arguments = (ins Merk_TermType:$list, UnitAttr:$reversed,
      Optional<Merk_TypeInfoType>:$element_type_info);
  let results = (outs Merk_ArrayType:$array);
  let assemblyFormat = [{
    (`reversed` $reversed^)? $list (`width` `of` $element_type_info^)?
    attr-dict `:` type($list) `->` type($array)
  }];
}

def Merk_ArrayResizeOp : Merk_Op<"array_resize", [
    AllTypesMatch<["array", "result"]>, AttrSizedOperandSegments
  ]> {
  let summary = "a new array of a size, with an array's elements";
  let description = [{
    A new array of `size` elements: as many of the array's as fit, and then
    `init`; without `init`, `size` is no greater than the array's. With the
    attribute `merk.clobbers`, the array is not used again, as
    `array.resize`'s `array_di` says, and a GPU process may free it after
    (FreeAtLastUse).
  }];
  let arguments = (ins Merk_ArrayType:$array, I64:$size, Optional<AnyType>:$init,
      Optional<Merk_TypeInfoType>:$element_type_info);
  let results = (outs Merk_ArrayType:$result);
  let assemblyFormat = [{
    $array `,` $size (`,` $init^ `:` type($init))? (`of` $element_type_info^)?
    attr-dict `:` type($array)
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

def Merk_FreeOp : Merk_Op<"free"> {
  let summary = "free a GPU process's cell, array or string at its last use";
  let description = [{
    Gives back the memory of a value that nothing uses after this: a term's
    cell, an array or a string, which a GPU process made or received
    (docs/INPROGRESS.md, Memory that a GPU process gives back). Freeing a
    value of a received message frees all of the message. A constant's word
    and a static cell are left alone, and on the host, which has a
    collector, and in a target region, whose arena goes as it ends, it does
    nothing.

    ```mlir
    merk.free %a : !merk.array<i64>
    ```
  }];
  let arguments = (ins AnyType:$value);
  let assemblyFormat = "$value attr-dict `:` type($value)";
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
    such as that of a division by zero: the value the text describes,
    which a try catches (the runtime's merk_library_error). Uncaught, the
    program writes "Uncaught Merkurius exception:" and the exception, as
    mmc writes it, to the standard error, and exits with status 1.

    ```mlir
    merk.throw_if %zero, "domain_error(\"int.\\'//\\': division by zero\")"
    ```

    With a type_info, the first `%T` of the exception's text is the name of
    its type, as mmc writes a type, which the runtime puts there.
  }];
  let arguments = (ins I1:$condition, StrAttr:$exception,
      Optional<Merk_TypeInfoType>:$type_info);
  let assemblyFormat =
      "$condition `,` $exception (`type` $type_info^)? attr-dict";
  let builders = [OpBuilder<(ins "mlir::Value":$condition,
      "llvm::StringRef":$exception), [{
    build($_builder, $_state, condition, exception, mlir::Value{});
  }]>];
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
  let arguments = (ins AnyTypeOf<[AnySignlessInteger, Merk_TermType,
                           Merk_RecordType]>:$value,
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
    `main(di, uo)`, which the runtime calls. With `mercury_messages`
    (-fmercury-messages), the runtime prints the messages that mmc's
    programs have too as those do.

    ```mlir
    merk.entry @hello.main_2_0
    merk.entry @hello.main_2_0 mercury_messages
    ```
  }];
  let arguments = (ins FlatSymbolRefAttr:$proc, UnitAttr:$mercury_messages);
  let assemblyFormat = "$proc (`mercury_messages` $mercury_messages^)? attr-dict";
}

def Merk_ConstructorOp : Merk_Op<"constructor", [
    DeclareOpInterfaceMethods<SymbolUserOpInterface>
  ]> {
  let summary = "a procedure that runs before the program starts";
  let description = [{
    Names a procedure of no inputs or outputs that runs at start-up, before
    the main procedure, as a C constructor: one that registers what the
    runtime needs of the module, such as the type_infos of the messages that
    target regions send.

    ```mlir
    merk.constructor @m.merk_register_message_types
    ```
  }];
  let arguments = (ins FlatSymbolRefAttr:$proc);
  let assemblyFormat = "$proc attr-dict";
}

def Merk_InitialiseOp : Merk_Op<"initialise", [
    DeclareOpInterfaceMethods<SymbolUserOpInterface>
  ]> {
  let summary = "a procedure that runs before or after main";
  let description = [{
    Names the procedure of an `initialise` directive, which the runtime
    calls before the program's main procedure, or with `final`, of a
    `finalise` directive, which it calls after (reference manual, "Module
    initialisation" and "Module finalisation"). It has no inputs or
    outputs: its I/O states have none. A module's run in the order of their
    ops.

    ```mlir
    merk.initialise @m.setup_2_0
    merk.initialise @m.teardown_2_0 final
    ```
  }];
  let arguments = (ins FlatSymbolRefAttr:$proc, UnitAttr:$final);
  let assemblyFormat = "$proc (`final` $final^)? attr-dict";
}

#endif // MERKURIUS_DIALECT_MERK_IR_MERKOPS_TD
