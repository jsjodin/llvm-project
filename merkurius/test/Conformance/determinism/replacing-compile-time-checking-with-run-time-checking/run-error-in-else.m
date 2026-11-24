% MANUAL: Replacing compile-time checking with run-time checking
% A goal that should never fail, checked at run time: error/1 is
% erroneous, so the if-then-else is det, and its else part need not bind
% the output.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:40
% OUTPUT-NOT:{{.}}

:- module run_error_in_else.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, require.

:- pred lookup(int::in, int::out) is semidet.
lookup(X, X * 10) :-
    X > 0.

:- pred q(int::in, int::out) is det.
q(A, B) :-
    ( if lookup(A, B0) then
        B = B0
    else
        error("lookup failed!")
    ).

main(!IO) :-
    q(4, B),
    io.write_line(B, !IO).
