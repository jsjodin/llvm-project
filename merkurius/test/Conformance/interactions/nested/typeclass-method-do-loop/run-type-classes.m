% MANUAL: Type classes
% DESIGN: Loops
% Interaction: a type class's methods called through a constraint, in a do
% loop that makes, gathers and carries the values, in instance methods that
% make and use the values.
% Written by utils/conformance/generate-interactions.py; change
% the generator rather than this file.
% MERKC-FLAGS: -fdo-loops
% TRANSLATION: Inputs/nested_typeclass_method_do_loop_type_classes_translation.m
% MODULES: Inputs/nested_typeclass_method_do_loop_type_classes_lib.m
% FILES: Inputs/nested_typeclass_method_do_loop_type_classes_lib_translation.m
%
% RUN: merkc -O0 -fdo-loops -I %S/Inputs %s %S/Inputs/nested_typeclass_method_do_loop_type_classes_lib.m -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops -I %S/Inputs %s %S/Inputs/nested_typeclass_method_do_loop_type_classes_lib.m -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:1: rect=2, square=1 / rect=2, square=1 | square=4, square=1 | last rect=462, square=1 | offset square=4, square=1 / square=4, square=1 | rect=12, square=1 | last square=484, square=1
% OUTPUT-NEXT:2: square=4, square=1 / square=4, square=1 | rect=12, square=1 | last square=484, square=1 | offset rect=12, square=1 / rect=12, square=1 | square=16, square=1 | last rect=552, square=1
% OUTPUT-NEXT:3: rect=12, square=1 / rect=12, square=1 | square=16, square=1 | last rect=552, square=1 | offset square=16, square=1 / square=16, square=1 | rect=30, square=1 | last square=576, square=1
% OUTPUT-NEXT:4: square=16, square=1 / square=16, square=1 | rect=30, square=1 | last square=576, square=1 | offset rect=30, square=1 / rect=30, square=1 | square=36, square=1 | last rect=650, square=1
% OUTPUT-NOT:{{.}}
:- module nested_typeclass_method_do_loop_type_classes.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, nested_typeclass_method_do_loop_type_classes_lib, string.

    % The feature's values, with what the inner context makes of the same
    % number, which runs as the outer context uses them.
:- type nested
    --->    nested(int, fdata).

:- func make_nested(int) = nested.

make_nested(N) = nested(N, make(N)).

:- func use_nested(nested) = string.

use_nested(nested(K, V)) = use(V) ++ " / " ++ inner_run(K).

:- typeclass runner(T) where [
    func produce(T, int) = nested,
    func consume(T, nested) = string
].

:- type plain
    --->    plain.

:- type offset
    --->    offset(int).

:- instance runner(plain) where [
    produce(plain, N) = make_nested(N),
    consume(plain, V) = use_nested(V)
].

:- instance runner(offset) where [
    produce(offset(K), N) = make_nested(N + K),
    consume(offset(_), V) = "offset " ++ use_nested(V)
].

    % Through a constraint, so that the methods are called by the
    % dictionary.
:- func via(T, int) = string <= runner(T).
:- pragma no_inline(via/2).

via(R, N) = consume(R, produce(R, N)).

:- func run(int) = string.

run(N) = via(plain, N) ++ " | " ++ via(offset(1), N).

main(!IO) :-
    list.foldl(show, 1 .. 4, !IO).

:- pred show(int::in, io::di, io::uo) is det.

show(N, !IO) :-
    io.format("%d: %s\n", [i(N), s(run(N))], !IO).
