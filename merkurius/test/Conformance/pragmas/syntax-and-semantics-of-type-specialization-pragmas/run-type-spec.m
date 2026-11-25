% MANUAL: Syntax and semantics of type specialization pragmas
% type_spec pragmas for all of a predicate's modes and for one, with one
% variable and two, and with a new variable on the right, which do not
% change what a program does.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:b70.5
% OUTPUT-NOT:{{.}}

:- module run_type_spec.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.

:- pred lookup(list({K, V})::in, K::in, V::out) is semidet.
:- pragma type_spec(lookup/3, K = int).
:- pragma type_spec(lookup(in, in, out), (K = string, V = int)).
:- pragma type_spec(lookup/3, K = list(U)).
lookup([{K0, V0} | T], K, V) :-
    ( if K = K0 then V = V0 else lookup(T, K, V) ).

main(!IO) :-
    ( if lookup([{1, "a"}, {2, "b"}], 2, V) then io.write_string(V, !IO) else true ),
    ( if lookup([{"x", 7}], "x", W) then io.write_int(W, !IO) else true ),
    ( if lookup([{[1], 0.5}], [1], F) then io.write_float(F, !IO) else true ),
    io.nl(!IO).
