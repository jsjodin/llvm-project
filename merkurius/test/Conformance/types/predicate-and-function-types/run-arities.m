% MANUAL: Predicate and function types
% The predicate types pred, pred(T), pred(T1, T2) and the function types
% (func) = T, func(T1) = T, func(T1, T2) = T: a closure of each is made,
% passed and called.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:pred
% OUTPUT-NEXT:{1, 2}
% OUTPUT-NEXT:{10, 8, 6}
% OUTPUT-NEXT:[2, 4, 6]
% OUTPUT-NEXT:20
% OUTPUT-NOT:{{.}}

:- module run_arities.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list.

main(!IO) :-
    P0 = ((pred) is semidet :- 1 < 2),
    P1 = (pred(X::out) is det :- X = 1),
    P2 = (pred(X::in, Y::out) is det :- Y = X + 1),
    F0 = ((func) = R :- R = 10),
    F1 = (func(X) = X * 2),
    F2 = (func(X, Y) = X - Y),
    ( if call(P0) then io.write_string("pred\n", !IO) else true ),
    call(P1, A),
    call(P2, A, B),
    io.write_line({A, B}, !IO),
    io.write_line({apply(F0), apply(F1, 4), apply(F2, 9, 3)}, !IO),
    io.write_line(list.map(F1, [1, 2, 3]), !IO),
    io.write_line(twice(F1, 5), !IO).

:- func twice(func(T) = T, T) = T.

twice(F, X) = F(F(X)).
