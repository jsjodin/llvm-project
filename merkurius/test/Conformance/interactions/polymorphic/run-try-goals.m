% MANUAL: Exception handling
% MANUAL: Predicate and function type declarations
% Interaction: a try goal that catches a thrown string, in polymorphic
% procedures that carry the values.
% Written by utils/conformance/generate-interactions.py; change
% the generator rather than this file.
%
% RUN: merkc -O0 %s -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:1: ok 3 | ok 9
% OUTPUT-NEXT:2: caught even 2 | caught even 4
% OUTPUT-NEXT:3: ok 9 | ok 15
% OUTPUT-NEXT:4: caught even 4 | caught even 6
% OUTPUT-NOT:{{.}}
:- module polymorphic_try_goals.
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

    % Polymorphic code that carries the values, in lists, pairs and maybes,
    % compiled once for every type.
:- pred keep(T::in, T::out) is det.
:- pragma no_inline(keep/2).

keep(X, X).

:- type holder(T)
    --->    holder(T, int)
    ;       nothing.

:- func second(list(T), T) = T.
:- pragma no_inline(second/2).

second(Xs, Default) =
    ( if Xs = [_, X | _] then X else Default ).

:- func run(int) = string.

run(N) = S :-
    keep([make(N), make(N + 1)], Vs),
    keep(holder(make(N + 2), N), H),
    ( if H = holder(V, _) then
        Held = use(V)
    else
        Held = "nothing"
    ),
    S = use(second(list.reverse(Vs), make(0))) ++ " | " ++ Held.

main(!IO) :-
    list.foldl(show, 1 .. 4, !IO).

:- pred show(int::in, io::di, io::uo) is det.

show(N, !IO) :-
    io.format("%d: %s\n", [i(N), s(run(N))], !IO).
