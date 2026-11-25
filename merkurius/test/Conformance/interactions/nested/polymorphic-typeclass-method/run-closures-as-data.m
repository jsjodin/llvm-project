% MANUAL: Calling higher-order terms
% MANUAL: Predicate and function type declarations
% MANUAL: Type classes
% Interaction: closures stored in a term and called, in instance methods
% that make and use the values, in polymorphic procedures that carry the
% values.
% Written by utils/conformance/generate-interactions.py; change
% the generator rather than this file.
% MODULES: Inputs/nested_polymorphic_typeclass_method_closures_as_data_lib.m
%
% RUN: merkc -O0 -I %S/Inputs %s %S/Inputs/nested_polymorphic_typeclass_method_closures_as_data_lib.m -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -I %S/Inputs %s %S/Inputs/nested_polymorphic_typeclass_method_closures_as_data_lib.m -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:1: 11 3/1 / 11 3/1 | offset 21 5/2 | 31 7/3 / 31 7/3 | offset 41 9/4
% OUTPUT-NEXT:2: 21 5/2 / 21 5/2 | offset 31 7/3 | 41 9/4 / 41 9/4 | offset 51 11/5
% OUTPUT-NEXT:3: 31 7/3 / 31 7/3 | offset 41 9/4 | 51 11/5 / 51 11/5 | offset 61 13/6
% OUTPUT-NEXT:4: 41 9/4 / 41 9/4 | offset 51 11/5 | 61 13/6 / 61 13/6 | offset 71 15/7
% OUTPUT-NOT:{{.}}
:- module nested_polymorphic_typeclass_method_closures_as_data.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, nested_polymorphic_typeclass_method_closures_as_data_lib, string.

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
