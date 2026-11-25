% MANUAL: Field access functions
% MANUAL: Type classes
% MANUAL: Predicate and function type declarations
% Interaction: field access and update functions, in polymorphic procedures
% that carry the values, in instance methods that make and use the values.
% Written by utils/conformance/generate-interactions.py; change
% the generator rather than this file.
% MODULES: Inputs/nested_typeclass_method_polymorphic_field_access_lib.m
%
% RUN: merkc -O0 -I %S/Inputs %s %S/Inputs/nested_typeclass_method_polymorphic_field_access_lib.m -o %t0
% RUN: %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t0 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -I %S/Inputs %s %S/Inputs/nested_typeclass_method_polymorphic_field_access_lib.m -o %t2
% RUN: %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t2 2>&1 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% OUTPUT:1: owner1 15 10 / owner1 15 10 | owner3 35 30 | offset owner2 25 20 / owner2 25 20 | owner4 45 40
% OUTPUT-NEXT:2: owner2 25 20 / owner2 25 20 | owner4 45 40 | offset owner3 35 30 / owner3 35 30 | owner5 55 50
% OUTPUT-NEXT:3: owner3 35 30 / owner3 35 30 | owner5 55 50 | offset owner4 45 40 / owner4 45 40 | owner6 65 60
% OUTPUT-NEXT:4: owner4 45 40 / owner4 45 40 | owner6 65 60 | offset owner5 55 50 / owner5 55 50 | owner7 75 70
% OUTPUT-NOT:{{.}}
:- module nested_typeclass_method_polymorphic_field_access.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, nested_typeclass_method_polymorphic_field_access_lib, string.

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
