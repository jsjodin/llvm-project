% MANUAL: Discriminated unions
% MANUAL: Predicate and function type declarations
% MANUAL: Creating higher-order terms
% DESIGN: Sized floating point types
% Interaction: an enumeration, a notag and a record of numbers, in closures
% that make, capture and use the values, in polymorphic procedures that
% carry the values.
% Written by utils/conformance/generate-interactions.py; change
% the generator rather than this file.
% MODULES: Inputs/nested_polymorphic_closure_flat_values_lib.mt
%
% RUN: %instantiate %s float %t.float.d
% RUN: %instantiate --keep-name %S/Inputs/nested_polymorphic_closure_flat_values_lib.mt float %t.float.d
% RUN: merkc -O0 -I %t.float.d %t.float.d/nested_polymorphic_closure_flat_values_float.m %t.float.d/nested_polymorphic_closure_flat_values_lib.m -o %t.float.0
% RUN: %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -I %t.float.d %t.float.d/nested_polymorphic_closure_flat_values_float.m %t.float.d/nested_polymorphic_closure_flat_values_lib.m -o %t.float.2
% RUN: %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% OUTPUT-FLOAT:1: green 100 -1 1000 0.25 / green 100 -1 1000 0.25 | green 100 -1 1000 0.25 | blue 200 -2 2000 0.5 | red 300 -3 3000 0.75 / red 300 -3 3000 0.75 | red 300 -3 3000 0.75 | green 400 -4 4000 1.0
% OUTPUT-FLOAT-NEXT:2: blue 200 -2 2000 0.5 / blue 200 -2 2000 0.5 | blue 200 -2 2000 0.5 | red 300 -3 3000 0.75 | green 400 -4 4000 1.0 / green 400 -4 4000 1.0 | green 400 -4 4000 1.0 | blue 500 -5 5000 1.25
% OUTPUT-FLOAT-NEXT:3: red 300 -3 3000 0.75 / red 300 -3 3000 0.75 | red 300 -3 3000 0.75 | green 400 -4 4000 1.0 | blue 500 -5 5000 1.25 / blue 500 -5 5000 1.25 | blue 500 -5 5000 1.25 | red 600 -6 6000 1.5
% OUTPUT-FLOAT-NEXT:4: green 400 -4 4000 1.0 / green 400 -4 4000 1.0 | green 400 -4 4000 1.0 | blue 500 -5 5000 1.25 | red 600 -6 6000 1.5 / red 600 -6 6000 1.5 | red 600 -6 6000 1.5 | green 700 -7 7000 1.75
% OUTPUT-FLOAT-NOT:{{.}}
% RUN: %instantiate %s float32 %t.float32.d
% RUN: %instantiate --keep-name %S/Inputs/nested_polymorphic_closure_flat_values_lib.mt float32 %t.float32.d
% RUN: merkc -O0 -I %t.float32.d %t.float32.d/nested_polymorphic_closure_flat_values_float32.m %t.float32.d/nested_polymorphic_closure_flat_values_lib.m -o %t.float32.0
% RUN: %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: merkc -O2 -I %t.float32.d %t.float32.d/nested_polymorphic_closure_flat_values_float32.m %t.float32.d/nested_polymorphic_closure_flat_values_lib.m -o %t.float32.2
% RUN: %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% OUTPUT-FLOAT32:1: green 100 -1 1000 0.25 / green 100 -1 1000 0.25 | green 100 -1 1000 0.25 | blue 200 -2 2000 0.5 | red 300 -3 3000 0.75 / red 300 -3 3000 0.75 | red 300 -3 3000 0.75 | green 400 -4 4000 1.0
% OUTPUT-FLOAT32-NEXT:2: blue 200 -2 2000 0.5 / blue 200 -2 2000 0.5 | blue 200 -2 2000 0.5 | red 300 -3 3000 0.75 | green 400 -4 4000 1.0 / green 400 -4 4000 1.0 | green 400 -4 4000 1.0 | blue 500 -5 5000 1.25
% OUTPUT-FLOAT32-NEXT:3: red 300 -3 3000 0.75 / red 300 -3 3000 0.75 | red 300 -3 3000 0.75 | green 400 -4 4000 1.0 | blue 500 -5 5000 1.25 / blue 500 -5 5000 1.25 | blue 500 -5 5000 1.25 | red 600 -6 6000 1.5
% OUTPUT-FLOAT32-NEXT:4: green 400 -4 4000 1.0 / green 400 -4 4000 1.0 | green 400 -4 4000 1.0 | blue 500 -5 5000 1.25 | red 600 -6 6000 1.5 / red 600 -6 6000 1.5 | red 600 -6 6000 1.5 | green 700 -7 7000 1.75
% OUTPUT-FLOAT32-NOT:{{.}}
:- module nested_polymorphic_closure_flat_values.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, nested_polymorphic_closure_flat_values_lib, string.

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
