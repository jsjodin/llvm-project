% MANUAL: Type classes
% MANUAL: Predicate and function type declarations
% DESIGN: Loops / Under a parallel directive
% Interaction: a type class's methods called through a constraint, in a
% parallel loop's iterations, which make and use the values, in polymorphic
% procedures that carry the values.
% Written by utils/conformance/generate-interactions.py; change
% the generator rather than this file.
% MERKC-FLAGS: -fdo-loops -fopenmp
% TRANSLATION: Inputs/nested_polymorphic_parallel_do_type_classes_translation.m
% MODULES: Inputs/nested_polymorphic_parallel_do_type_classes_lib.m
% FILES: Inputs/nested_polymorphic_parallel_do_type_classes_lib_translation.m
%
% REQUIRES: libomp
% RUN: merkc -O0 -fdo-loops -fopenmp -I %S/Inputs %s %S/Inputs/nested_polymorphic_parallel_do_type_classes_lib.m %libomp-flags -o %t0
% RUN: %libomp-env OMP_NUM_THREADS=4 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: %libomp-env OMP_NUM_THREADS=4 MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops -fopenmp -I %S/Inputs %s %S/Inputs/nested_polymorphic_parallel_do_type_classes_lib.m %libomp-flags -o %t2
% RUN: %libomp-env OMP_NUM_THREADS=4 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: %libomp-env OMP_NUM_THREADS=4 MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O0 -fdo-loops -I %S/Inputs %s %S/Inputs/nested_polymorphic_parallel_do_type_classes_lib.m -o %t0s
% RUN: %t0s 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops -I %S/Inputs %s %S/Inputs/nested_polymorphic_parallel_do_type_classes_lib.m -o %t2s
% RUN: %t2s 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:1: rect=2, square=1 / rect=2, square=1 | square=4, square=1 | rect=12, square=1 | square=16, square=1 | rect=12, square=1 / rect=12, square=1 | square=16, square=1 | rect=30, square=1 | square=36, square=1
% OUTPUT-NEXT:2: square=4, square=1 / square=4, square=1 | rect=12, square=1 | square=16, square=1 | rect=30, square=1 | square=16, square=1 / square=16, square=1 | rect=30, square=1 | square=36, square=1 | rect=56, square=1
% OUTPUT-NEXT:3: rect=12, square=1 / rect=12, square=1 | square=16, square=1 | rect=30, square=1 | square=36, square=1 | rect=30, square=1 / rect=30, square=1 | square=36, square=1 | rect=56, square=1 | square=64, square=1
% OUTPUT-NEXT:4: square=16, square=1 / square=16, square=1 | rect=30, square=1 | square=36, square=1 | rect=56, square=1 | square=36, square=1 / square=36, square=1 | rect=56, square=1 | square=64, square=1 | rect=90, square=1
% OUTPUT-NOT:{{.}}
:- module nested_polymorphic_parallel_do_type_classes.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, nested_polymorphic_parallel_do_type_classes_lib, string.

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
