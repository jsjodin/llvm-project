% MANUAL: When to use type specialization
% Type specialization of a predicate that calls type class methods and
% compares polymorphic values, for builtin and enumeration types.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:6
% OUTPUT-NEXT:3
% OUTPUT-NEXT:9
% OUTPUT-NEXT:green
% OUTPUT-NOT:{{.}}

:- module run_method_calls.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, string.

:- type colour ---> red ; green ; blue.
:- typeclass size(T) where [ func size(T) = int ].
:- instance size(int) where [ size(X) = X ].
:- instance size(colour) where [ size(C) = ( if C = red then 1 else 2 ) ].
:- func total(list(T)) = int <= size(T).
:- pragma type_spec(total/1, T = int).
:- pragma type_spec(total/1, T = colour).
total(L) = list.foldl(func(X, A) = A + size(X), L, 0).
:- pred max_of(list(T)::in, T::in, T::out) is det.
:- pragma type_spec(max_of/3, T = int).
:- pragma type_spec(max_of/3, T = colour).
max_of([], M, M).
max_of([X | Xs], M0, M) :-
    ( if compare((>), X, M0) then max_of(Xs, X, M) else max_of(Xs, M0, M) ).

main(!IO) :-
    io.write_int(total([1, 2, 3]), !IO), io.nl(!IO),
    io.write_int(total([red, blue]), !IO), io.nl(!IO),
    max_of([3, 9, 2], 0, MI), io.write_int(MI, !IO), io.nl(!IO),
    max_of([green, red], red, MC), io.write(MC, !IO), io.nl(!IO).
