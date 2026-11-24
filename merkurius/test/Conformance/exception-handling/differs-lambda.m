% MANUAL: Exception handling
% DIFFERS: mmc-bug: mmc aborts on a try goal in a lambda expression
% ("Unexpected: try_goal" in simplify_goal_expr). merkc runs it.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:25 -1
% OUTPUT-NOT:{{.}}
% MMC-STATUS: rejected
% MMC:

:- module differs_lambda.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is cc_multi.
:- implementation.
:- import_module exception, int, list, string.

main(!IO) :-
    Divide = (pred(X::in, Y::out) is cc_multi :-
        ( try [] Z = 100 // X
        then Y = Z
        catch_any _ -> Y = -1
        )),
    Divide(4, A),
    Divide(0, B),
    io.format("%d %d\n", [i(A), i(B)], !IO).
