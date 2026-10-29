% Type class and instance declarations, as the converter reads them, from
% the reference manual's examples.
%
% RUN: merkc -fc1 -fdebug-dump-parse-tree-no-sema %s | FileCheck %s

:- module typeclasses.
:- interface.
:- import_module io, list.

:- typeclass portrayable(T) where [
    pred portray(T::in, io.state::di, io.state::uo) is det
].
% CHECK:      | | TypeclassDecl
% CHECK-NEXT: | | | name: SymName portrayable
% CHECK-NEXT: | | | parameters:
% CHECK-NEXT: | | | | T
% CHECK-NEXT: | | | constraints:
% CHECK-NEXT: | | | abstract: false
% CHECK-NEXT: | | | methods:
% CHECK-NEXT: | | | | PredDecl
% CHECK-NEXT: | | | | | kind: Predicate
% CHECK:      | | | | | name: SymName portray
% CHECK:      | | | | | determinism: Det

:- typeclass euclidean(T) <= ring(T) where [
    func div(T::in, T::in) = (T::out) is det,
    func mod(T::in, T::in) = (T::out) is det
].
% CHECK:      | | TypeclassDecl
% CHECK-NEXT: | | | name: SymName euclidean
% CHECK-NEXT: | | | parameters:
% CHECK-NEXT: | | | | T
% CHECK-NEXT: | | | constraints:
% CHECK-NEXT: | | | | ClassConstraint
% CHECK-NEXT: | | | | | name: SymName ring
% CHECK:      | | | abstract: false
% CHECK-NEXT: | | | methods:
% CHECK-NEXT: | | | | PredDecl
% CHECK-NEXT: | | | | | kind: Function
% CHECK:      | | | | | name: SymName div
% CHECK:      | | | | PredDecl
% CHECK-NEXT: | | | | | kind: Function
% CHECK:      | | | | | name: SymName mod

:- typeclass foo(A, B) <= ((A -> B), (B -> A)).
% CHECK:      | | TypeclassDecl
% CHECK-NEXT: | | | name: SymName foo
% CHECK:      | | | constraints:
% CHECK-NEXT: | | | | FunctionalDependency
% CHECK-NEXT: | | | | | domain:
% CHECK-NEXT: | | | | | | A
% CHECK-NEXT: | | | | | range:
% CHECK-NEXT: | | | | | | B
% CHECK-NEXT: | | | | FunctionalDependency
% CHECK:      | | | abstract: true

:- instance portrayable(list(T)) <= portrayable(T).
% CHECK:      | | InstanceDecl
% CHECK-NEXT: | | | name: SymName portrayable
% CHECK-NEXT: | | | types:
% CHECK-NEXT: | | | | NamedType
% CHECK-NEXT: | | | | | name: SymName list
% CHECK:      | | | constraints:
% CHECK-NEXT: | | | | ClassConstraint
% CHECK:      | | | abstract: true
% CHECK-NEXT: | | | methods:

:- implementation.

:- instance portrayable(int) where [
    pred(portray/3) is io.write_int
].
% CHECK:      | | InstanceDecl
% CHECK-NEXT: | | | name: SymName portrayable
% CHECK:      | | | abstract: false
% CHECK-NEXT: | | | methods:
% CHECK-NEXT: | | | | MethodByName
% CHECK-NEXT: | | | | | kind: Predicate
% CHECK-NEXT: | | | | | method: NameAndArity portray/3
% CHECK-NEXT: | | | | | implementation: SymName io.write_int
