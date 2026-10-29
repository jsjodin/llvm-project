% Type, inst and mode definitions, as the converter reads them, from the
% reference manual's examples.
%
% RUN: merkc -fc1 -fdebug-dump-parse-tree-no-sema %s | FileCheck %s

:- module definitions.
:- interface.

:- type employee
        --->    employee(
                       name        :: string,
                       age         :: int,
                       department  :: string
                ).
% CHECK:      | | DuTypeDefn
% CHECK-NEXT: | | | name: SymName employee
% CHECK-NEXT: | | | parameters:
% CHECK-NEXT: | | | supertype: none
% CHECK-NEXT: | | | constructors:
% CHECK-NEXT: | | | | Constructor
% CHECK-NEXT: | | | | | existentialVars:
% CHECK-NEXT: | | | | | name: SymName employee
% CHECK-NEXT: | | | | | arguments:
% CHECK-NEXT: | | | | | | ConstructorArg
% CHECK-NEXT: | | | | | | | field: SymName name
% CHECK-NEXT: | | | | | | | type: NamedType
% CHECK-NEXT: | | | | | | | | name: SymName string
% CHECK-NEXT: | | | | | | | | arguments:
% CHECK-NEXT: | | | | | | ConstructorArg
% CHECK-NEXT: | | | | | | | field: SymName age
% CHECK:      | | | | | | ConstructorArg
% CHECK-NEXT: | | | | | | | field: SymName department
% CHECK:      | | | attributes: none

:- type set(T).
% CHECK:      | | AbstractTypeDecl
% CHECK-NEXT: | | | name: SymName set
% CHECK-NEXT: | | | parameters:
% CHECK-NEXT: | | | | T
% CHECK-NEXT: | | | attributes: none

:- type assoc_list(KeyType, ValueType)
        == list(pair(KeyType, ValueType)).
% CHECK:      | | EquivalenceTypeDefn
% CHECK-NEXT: | | | name: SymName assoc_list
% CHECK-NEXT: | | | parameters:
% CHECK-NEXT: | | | | KeyType
% CHECK-NEXT: | | | | ValueType
% CHECK-NEXT: | | | type: NamedType
% CHECK-NEXT: | | | | name: SymName list

:- implementation.

:- type set(T)
    --->    set(list(T))
            where equality is set_equals, comparison is set_compare.
% CHECK:      | | DuTypeDefn
% CHECK-NEXT: | | | name: SymName set
% CHECK:      | | | attributes: TypeAttributes
% CHECK-NEXT: | | | | equality: SymName set_equals
% CHECK-NEXT: | | | | comparison: SymName set_compare
% CHECK-NEXT: | | | | directArgs:
% CHECK-NEXT: | | | | abstractNoncanonical: false

:- type citrus_fruit =< fruit
    --->    lemon
    ;       orange.
% CHECK:      | | DuTypeDefn
% CHECK-NEXT: | | | name: SymName citrus_fruit
% CHECK-NEXT: | | | parameters:
% CHECK-NEXT: | | | supertype: NamedType
% CHECK-NEXT: | | | | name: SymName fruit

:- inst listskel(Inst) for list/1
    --->    []
    ;       [Inst | listskel(Inst)].
% CHECK:      | | InstDefn
% CHECK-NEXT: | | | name: SymName listskel
% CHECK-NEXT: | | | parameters:
% CHECK-NEXT: | | | | Inst
% CHECK-NEXT: | | | forType: NameAndArity list/1
% CHECK-NEXT: | | | inst: BoundInst

:- mode create_request == free >> request.
% CHECK:      | | ModeDefn
% CHECK-NEXT: | | | name: SymName create_request
% CHECK-NEXT: | | | parameters:
% CHECK-NEXT: | | | mode: MappingMode
% CHECK-NEXT: | | | | initial: BuiltinInst
% CHECK-NEXT: | | | | | kind: Free
% CHECK-NEXT: | | | | final: NamedInst
% CHECK-NEXT: | | | | | name: SymName request
