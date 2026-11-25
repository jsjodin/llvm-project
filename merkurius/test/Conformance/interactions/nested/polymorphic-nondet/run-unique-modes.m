% MANUAL: Unique modes
% MANUAL: Predicate and function type declarations
% MANUAL: Determinism categories
% Interaction: an array updated in place in unique modes, in a nondet goal
% that makes the values, in polymorphic procedures that carry the values.
% Written by utils/conformance/generate-interactions.py; change
% the generator rather than this file.
% MODULES: Inputs/nested_polymorphic_nondet_unique_modes_lib.m
%
% RUN: merkc -O0 -I %S/Inputs %s %S/Inputs/nested_polymorphic_nondet_unique_modes_lib.m -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -I %S/Inputs %s %S/Inputs/nested_polymorphic_nondet_unique_modes_lib.m -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:1: 0 1 4 9 / 0 1 4 9 | 0 1 4 9 16 25 | 0 1 4 9 16 25 / 0 1 4 9 16 25 | 0 1 4 9 16 25 36 49
% OUTPUT-NEXT:2: 0 1 4 9 16 / 0 1 4 9 16 25 | 0 1 4 9 16 25 36 / 0 1 4 9 16 25 36 49
% OUTPUT-NEXT:3: 0 1 4 9 16 25 / 0 1 4 9 16 25 | 0 1 4 9 16 25 36 49 | 0 1 4 9 16 25 36 49 / 0 1 4 9 16 25 36 49 | 0 1 4 9 16 25 36 49 64 81
% OUTPUT-NEXT:4: 0 1 4 9 16 25 36 / 0 1 4 9 16 25 36 49 | 0 1 4 9 16 25 36 49 64 / 0 1 4 9 16 25 36 49 64 81
% OUTPUT-NOT:{{.}}
:- module nested_polymorphic_nondet_unique_modes.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, nested_polymorphic_nondet_unique_modes_lib, string.

    % The feature's values, with what the inner context makes of the same
    % number, which runs as the outer context uses them.
:- type nested
    --->    nested(int, fdata).

:- func make_nested(int) = nested.

make_nested(N) = nested(N, make(N)).

:- func use_nested(nested) = string.

use_nested(nested(K, V)) = use(V) ++ " / " ++ inner_run(K).

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
    keep([make_nested(N), make_nested(N + 1)], Vs),
    keep(holder(make_nested(N + 2), N), H),
    ( if H = holder(V, _) then
        Held = use_nested(V)
    else
        Held = "nothing"
    ),
    S = use_nested(second(list.reverse(Vs), make_nested(0))) ++ " | " ++ Held.

main(!IO) :-
    list.foldl(show, 1 .. 4, !IO).

:- pred show(int::in, io::di, io::uo) is det.

show(N, !IO) :-
    io.format("%d: %s\n", [i(N), s(run(N))], !IO).
