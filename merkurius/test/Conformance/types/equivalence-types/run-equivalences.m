% MANUAL: Equivalence types
% An equivalence type abbreviates the type on its right, polymorphic or not,
% and the two are the same type in all respects: values pass between them,
% unify and compare.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:31
% OUTPUT-NEXT:2
% OUTPUT-NEXT:"two"
% OUTPUT-NEXT:60
% OUTPUT-NEXT:equal
% OUTPUT-NOT:{{.}}

:- module run_equivalences.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, pair, string.

:- type money == int.
:- type my_assoc_list(K, V) == list(pair(K, V)).
:- type transform == (func(int) = int).

main(!IO) :-
    Price = 10 : money,
    Total = Price * 3,
    io.write_line(Total + 1, !IO),
    AL = [1 - "one", 2 - "two"] : my_assoc_list(int, string),
    io.write_line(list.length(AL) : int, !IO),
    io.write_line(lookup(AL, 2), !IO),
    F = double : transform,
    io.write_line(F(Total), !IO),
    ( if Price = 10 then io.write_string("equal\n", !IO) else true ).

:- func double(int) = int.

double(X) = 2 * X.

:- func lookup(my_assoc_list(int, string), int) = string.

lookup([], _) = "none".
lookup([K - V | KVs], Key) = ( if K = Key then V else lookup(KVs, Key) ).
