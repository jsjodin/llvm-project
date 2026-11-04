% The else part of a condition that cannot fail is never reached, and is
% lowered with the rest: its arithmetic once stayed in the arith dialect,
% which the conversion to LLVM skips in a block with no predecessors.
%
% RUN: merkc %s -o %t && %t | FileCheck %s --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2 && %t2 | FileCheck %s --match-full-lines --strict-whitespace

:- module unreachable_branch.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int.

:- func f(int) = int.
f(X) = ( if X = X then X - 1 else X + X ).

:- pred g(int::in, int::out) is det.
g(X, Y) :-
    ( if X = X then
        Y = 7
    else
        Y = X * 3
    ).

main(!IO) :-
    io.write_int(f(3), !IO),
    io.nl(!IO),
    g(4, Y),
    io.write_int(Y, !IO),
    io.nl(!IO).
% mmc's output.
% CHECK:2
% CHECK-NEXT:7
