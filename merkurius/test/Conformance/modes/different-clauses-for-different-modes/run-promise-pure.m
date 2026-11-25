% MANUAL: Different clauses for different modes
% promise_pure also promises that clauses for different modes are
% equivalent, and that each clause is pure.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:8
% OUTPUT-NEXT:5
% OUTPUT-NEXT:odd
% OUTPUT-NOT:{{.}}

:- module run_promise_pure.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int.

:- pred twice(int, int).
:- mode twice(in, out) is det.
:- mode twice(out, in) is semidet.
:- pragma promise_pure(twice/2).
twice(X::in, Y::out) :- Y = X * 2.
twice(X::out, Y::in) :- Y mod 2 = 0, X = Y // 2.

main(!IO) :-
    twice(4, A),
    io.write_line(A, !IO),
    ( if twice(B, 10) then
        io.write_line(B, !IO)
    else
        io.write_string("odd\n", !IO)
    ),
    ( if twice(C, 7) then
        io.write_line(C, !IO)
    else
        io.write_string("odd\n", !IO)
    ).
