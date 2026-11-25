% DESIGN: Loops / Syntax
% With -fdo-loops, do written as a call, with no layout before its
% parenthesis, stays a call, and as an operand do is written (do). Without
% the operator, which mmc does not have, the program means the same.
% MERKC-FLAGS: -fdo-loops
%
% RUN: merkc -O0 -fdo-loops %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:42
% OUTPUT-NEXT:[do, undo]
% OUTPUT-NOT:{{.}}
:- module run_do_as_call.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list.

:- type action ---> do(int) ; skip.
:- type step ---> (do) ; undo.

:- func twice(int) = int.
twice(X) = 2 * X.

main(!IO) :-
    A = do(twice(21)),
    ( if A = do(N) then io.write_int(N, !IO) else io.write_string("skip", !IO) ),
    io.nl(!IO),
    S = (do),
    io.write([S, undo], !IO), io.nl(!IO).
