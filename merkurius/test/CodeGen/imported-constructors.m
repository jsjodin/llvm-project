% All of an imported type's constructors are known, even one that no clause
% uses, such as list's [|] here. The module has no main: it only has to
% compile.
%
% RUN: merkc -fc1 -emit-merk -triple x86_64-unknown-linux-gnu %s | FileCheck %s

:- module imported_constructors.
:- interface.
:- import_module list.
% CHECK: merk.type_def "list.list(int)" [#merk.constructor<"[]">, #merk.constructor<"[|]"(i64, !merk.term<"list.list(int)">)>]
:- pred is_empty(list(int)::in) is semidet.
:- implementation.
is_empty([]).
