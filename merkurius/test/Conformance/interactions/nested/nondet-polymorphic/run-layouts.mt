% MANUAL: Discriminated unions
% MANUAL: Determinism categories
% MANUAL: Predicate and function type declarations
% DESIGN: Sized floating point types
% Interaction: a type of more functors than a pointer has tags, in
% polymorphic procedures that carry the values, in a nondet goal that makes
% the values.
% Written by utils/conformance/generate-interactions.py; change
% the generator rather than this file.
% MODULES: Inputs/nested_nondet_polymorphic_layouts_lib.mt
%
% RUN: %instantiate %s float %t.float.d
% RUN: %instantiate --keep-name %S/Inputs/nested_nondet_polymorphic_layouts_lib.mt float %t.float.d
% RUN: merkc -O0 -I %t.float.d %t.float.d/nested_nondet_polymorphic_layouts_float.m %t.float.d/nested_nondet_polymorphic_layouts_lib.m -o %t.float.0
% RUN: %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -I %t.float.d %t.float.d/nested_nondet_polymorphic_layouts_float.m %t.float.d/nested_nondet_polymorphic_layouts_lib.m -o %t.float.2
% RUN: %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% OUTPUT-FLOAT:1: seven(three(0.5, one(1), 3), two(1, s1)) / seven(three(0.5, one(1), 3), two(1, s1)) | seven(three(1.5, one(3), 9), four(x, 9)) | seven(three(1.5, one(3), 9), four(x, 9)) / seven(three(1.5, one(3), 9), four(x, 9)) | seven(three(2.5, one(5), 15), two(5, s5))
% OUTPUT-FLOAT-NEXT:2: seven(three(1.5, one(3), 9), four(x, 9)) / seven(three(1.5, one(3), 9), four(x, 9)) | seven(three(2.5, one(5), 15), two(5, s5))
% OUTPUT-FLOAT-NEXT:3: seven(three(1.5, one(3), 9), four(x, 9)) / seven(three(1.5, one(3), 9), four(x, 9)) | seven(three(2.5, one(5), 15), two(5, s5)) | seven(three(2.5, one(5), 15), two(5, s5)) / seven(three(2.5, one(5), 15), two(5, s5)) | seven(three(3.5, one(7), 21), two(7, s7))
% OUTPUT-FLOAT-NEXT:4: seven(three(2.5, one(5), 15), two(5, s5)) / seven(three(2.5, one(5), 15), two(5, s5)) | seven(three(3.5, one(7), 21), two(7, s7))
% OUTPUT-FLOAT-NOT:{{.}}
% RUN: %instantiate %s float32 %t.float32.d
% RUN: %instantiate --keep-name %S/Inputs/nested_nondet_polymorphic_layouts_lib.mt float32 %t.float32.d
% RUN: merkc -O0 -I %t.float32.d %t.float32.d/nested_nondet_polymorphic_layouts_float32.m %t.float32.d/nested_nondet_polymorphic_layouts_lib.m -o %t.float32.0
% RUN: %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: merkc -O2 -I %t.float32.d %t.float32.d/nested_nondet_polymorphic_layouts_float32.m %t.float32.d/nested_nondet_polymorphic_layouts_lib.m -o %t.float32.2
% RUN: %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% OUTPUT-FLOAT32:1: seven(three(0.5, one(1), 3), two(1, s1)) / seven(three(0.5, one(1), 3), two(1, s1)) | seven(three(1.5, one(3), 9), four(x, 9)) | seven(three(1.5, one(3), 9), four(x, 9)) / seven(three(1.5, one(3), 9), four(x, 9)) | seven(three(2.5, one(5), 15), two(5, s5))
% OUTPUT-FLOAT32-NEXT:2: seven(three(1.5, one(3), 9), four(x, 9)) / seven(three(1.5, one(3), 9), four(x, 9)) | seven(three(2.5, one(5), 15), two(5, s5))
% OUTPUT-FLOAT32-NEXT:3: seven(three(1.5, one(3), 9), four(x, 9)) / seven(three(1.5, one(3), 9), four(x, 9)) | seven(three(2.5, one(5), 15), two(5, s5)) | seven(three(2.5, one(5), 15), two(5, s5)) / seven(three(2.5, one(5), 15), two(5, s5)) | seven(three(3.5, one(7), 21), two(7, s7))
% OUTPUT-FLOAT32-NEXT:4: seven(three(2.5, one(5), 15), two(5, s5)) / seven(three(2.5, one(5), 15), two(5, s5)) | seven(three(3.5, one(7), 21), two(7, s7))
% OUTPUT-FLOAT32-NOT:{{.}}
:- module nested_nondet_polymorphic_layouts.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, nested_nondet_polymorphic_layouts_lib, solutions, string.

    % The feature's values, with what the inner context makes of the same
    % number, which runs as the outer context uses them.
:- type nested
    --->    nested(int, fdata).

:- func make_nested(int) = nested.

make_nested(N) = nested(N, make(N)).

:- func use_nested(nested) = string.

use_nested(nested(K, V)) = use(V) ++ " / " ++ inner_run(K).

    % Values made in a nondet goal, each solution's in turn.
:- pred made(int::in, string::out) is nondet.

made(N, S) :-
    list.member(K, [N, N + 1, N + 2]),
    K mod 2 = 1,
    V = make_nested(K),
    S = use_nested(V).

:- func run(int) = string.

run(N) = S :-
    solutions(made(N), Ss),
    S = string.join_list(" | ", Ss).

main(!IO) :-
    list.foldl(show, 1 .. 4, !IO).

:- pred show(int::in, io::di, io::uo) is det.

show(N, !IO) :-
    io.format("%d: %s\n", [i(N), s(run(N))], !IO).
