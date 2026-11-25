% MANUAL: Exception handling
% MANUAL: Creating higher-order terms
% Interaction: a try goal that catches a thrown string, in closures that
% make, capture and use the values.
% Written by utils/conformance/generate-interactions.py; change
% the generator rather than this file.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:1: ok 3 | ok 3 | caught even 2
% OUTPUT-NEXT:2: caught even 2 | caught even 2 | ok 9
% OUTPUT-NEXT:3: ok 9 | ok 9 | caught even 4
% OUTPUT-NEXT:4: caught even 4 | caught even 4 | ok 15
% OUTPUT-NOT:{{.}}
:- module closure_try_goals.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module exception, int, list, string.

:- type fdata == int.

:- func make(int) = fdata.
:- func use(fdata) = string.

make(N) = N.

use(N) = S :-
    promise_equivalent_solutions [S] (
        ( try []
            risky(N, R)
        then
            S = "ok " ++ string.int_to_string(R)
        catch Message ->
            S = "caught " ++ Message
        )
    ).

:- pred risky(int::in, int::out) is det.

risky(N, R) :-
    ( if N mod 2 = 0 then
        throw("even " ++ string.int_to_string(N))
    else
        R = N * 3
    ).

    % A closure that makes and uses a value, and one that captures a value
    % made outside it.
:- func run(int) = string.

run(N) = S :-
    Make = (func(K) = make(K)),
    V = apply(Make, N),
    Use = ((func) = use(V)),
    Both = list.map((func(K) = use(make(K))), [N, N + 1]),
    S = apply(Use) ++ " | " ++ string.join_list(" | ", Both).

main(!IO) :-
    list.foldl(show, 1 .. 4, !IO).

:- pred show(int::in, io::di, io::uo) is det.

show(N, !IO) :-
    io.format("%d: %s\n", [i(N), s(run(N))], !IO).
