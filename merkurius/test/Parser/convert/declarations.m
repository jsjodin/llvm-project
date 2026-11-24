% Predicate, function and mode declarations, as the converter reads them.
%
% RUN: merkc -fc1 -fdebug-dump-parse-tree-no-sema %s | FileCheck %s

:- module declarations.
:- interface.
:- import_module io, list.

:- pred main(io::di, io::uo) is det.
% CHECK:      | | PredDecl
% CHECK-NEXT: | | | kind: Predicate
% CHECK-NEXT: | | | universalVars:
% CHECK-NEXT: | | | existentialVars:
% CHECK-NEXT: | | | purity: Pure
% CHECK-NEXT: | | | name: SymName main
% CHECK-NEXT: | | | arguments:
% CHECK-NEXT: | | | | TypeAndMode
% CHECK-NEXT: | | | | | type: NamedType
% CHECK-NEXT: | | | | | | name: SymName io
% CHECK-NEXT: | | | | | | arguments:
% CHECK-NEXT: | | | | | mode: NamedMode
% CHECK-NEXT: | | | | | | name: SymName di
% CHECK-NEXT: | | | | | | arguments:
% CHECK-NEXT: | | | | TypeAndMode
% CHECK-NEXT: | | | | | type: NamedType
% CHECK-NEXT: | | | | | | name: SymName io
% CHECK-NEXT: | | | | | | arguments:
% CHECK-NEXT: | | | | | mode: NamedMode
% CHECK-NEXT: | | | | | | name: SymName uo
% CHECK-NEXT: | | | | | | arguments:
% CHECK-NEXT: | | | result: none
% CHECK-NEXT: | | | withType: none
% CHECK-NEXT: | | | withInst: none
% CHECK-NEXT: | | | determinism: Det
% CHECK-NEXT: | | | existentialConstraints:
% CHECK-NEXT: | | | universalConstraints:

:- func length(list(T)) = int.
:- mode length(in) = out is det.
% CHECK:      | | PredDecl
% CHECK-NEXT: | | | kind: Function
% CHECK:      | | | name: SymName length
% CHECK:      | | | result: TypeAndMode
% CHECK-NEXT: | | | | type: NamedType
% CHECK-NEXT: | | | | | name: SymName int
% CHECK:      | | ModeDecl
% CHECK-NEXT: | | | kind: Function
% CHECK-NEXT: | | | name: SymName length
% CHECK-NEXT: | | | arguments:
% CHECK-NEXT: | | | | NamedMode
% CHECK-NEXT: | | | | | name: SymName in
% CHECK-NEXT: | | | | | arguments:
% CHECK-NEXT: | | | result: NamedMode
% CHECK-NEXT: | | | | name: SymName out
% CHECK-NEXT: | | | | arguments:
% CHECK-NEXT: | | | withInst: none
% CHECK-NEXT: | | | determinism: Det
% CHECK-NEXT: | | | constraints:

% The reference manual's example of both kinds of constraint
% ("Existential class constraints").
:- all [T1] some [T2] ((pred p(T1, T2) => (c1(T2), c2(T2))) <= c3(T1)).
% CHECK:      | | PredDecl
% CHECK-NEXT: | | | kind: Predicate
% CHECK-NEXT: | | | universalVars:
% CHECK-NEXT: | | | | T1
% CHECK-NEXT: | | | existentialVars:
% CHECK-NEXT: | | | | T2
% CHECK:      | | | existentialConstraints:
% CHECK-NEXT: | | | | ClassConstraint
% CHECK-NEXT: | | | | | name: SymName c1
% CHECK:      | | | | ClassConstraint
% CHECK-NEXT: | | | | | name: SymName c2
% CHECK:      | | | universalConstraints:
% CHECK-NEXT: | | | | ClassConstraint
% CHECK-NEXT: | | | | | name: SymName c3

% Quantifiers of each kind in any order, as mmc takes them, though the
% manual puts all before some ("Where mmc and the manual disagree").
:- some [T2] all [T3] some [T4] all [T1] pred q(T1, T2, T3, T4).
% CHECK:      | | PredDecl
% CHECK-NEXT: | | | kind: Predicate
% CHECK-NEXT: | | | universalVars:
% CHECK-NEXT: | | | | T3
% CHECK-NEXT: | | | | T1
% CHECK-NEXT: | | | existentialVars:
% CHECK-NEXT: | | | | T2
% CHECK-NEXT: | | | | T4

% A higher-order argument with its inst ("Builtin higher-order insts and
% modes").
:- pred foldl(pred(T, A, A), list(T), A, A).
:- mode foldl(pred(in, in, out) is det, in, in, out) is det.
% CHECK:      | | ModeDecl
% CHECK-NEXT: | | | kind: Predicate
% CHECK-NEXT: | | | name: SymName foldl
% CHECK-NEXT: | | | arguments:
% CHECK-NEXT: | | | | HigherOrderInst
% CHECK-NEXT: | | | | | kind: Predicate
% CHECK-NEXT: | | | | | any: false
% CHECK-NEXT: | | | | | arguments:
% CHECK-NEXT: | | | | | | NamedMode
% CHECK-NEXT: | | | | | | | name: SymName in

% A function declaration with a type has no result of its own: the type's
% is the function's (reference manual, "Predicate and function type
% declarations").
:- type adder == (func(int) = int).
:- func add(int) : adder.
% CHECK:      | | PredDecl
% CHECK-NEXT: | | | kind: Function
% CHECK-NEXT: | | | universalVars:
% CHECK-NEXT: | | | existentialVars:
% CHECK-NEXT: | | | purity: Pure
% CHECK-NEXT: | | | name: SymName add
% CHECK-NEXT: | | | arguments:
% CHECK-NEXT: | | | | TypeAndMode
% CHECK-NEXT: | | | | | type: NamedType
% CHECK-NEXT: | | | | | | name: SymName int
% CHECK-NEXT: | | | | | | arguments:
% CHECK-NEXT: | | | | | mode: none
% CHECK-NEXT: | | | result: none
% CHECK-NEXT: | | | withType: NamedType
% CHECK-NEXT: | | | | name: SymName adder
