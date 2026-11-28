% MANUAL: Discriminated unions
% MANUAL: Exception handling
% DESIGN: Loops
% DESIGN: Sized floating point types
% Interaction: a type of more functors than a pointer has tags, in a do loop
% that makes, gathers and carries the values, in values thrown as exceptions
% and caught.
% Written by utils/conformance/generate-interactions.py; change
% the generator rather than this file.
% MERKC-FLAGS: -fdo-loops
% TRANSLATION: Inputs/nested_try_do_loop_layouts_translation.mt
% MODULES: Inputs/nested_try_do_loop_layouts_lib.mt
% FILES: Inputs/nested_try_do_loop_layouts_lib_translation.mt
%
% RUN: %instantiate %s float %t.float.d
% RUN: %instantiate --keep-name %S/Inputs/nested_try_do_loop_layouts_lib.mt float %t.float.d
% RUN: merkc -O0 -fdo-loops -I %t.float.d %t.float.d/nested_try_do_loop_layouts_float.m %t.float.d/nested_try_do_loop_layouts_lib.m -o %t.float.0
% RUN: %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops -I %t.float.d %t.float.d/nested_try_do_loop_layouts_float.m %t.float.d/nested_try_do_loop_layouts_lib.m -o %t.float.2
% RUN: %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% OUTPUT-FLOAT:1: returned seven(three(0.5, one(1), 3), two(1, s1)) / seven(three(0.5, one(1), 3), two(1, s1)) | seven(three(1.0, one(2), 6), eight(c0)) | last seven(three(10.5, one(21), 63), four(x, 441))
% OUTPUT-FLOAT-NEXT:2: caught seven(three(1.0, one(2), 6), eight(c0)) / seven(three(1.0, one(2), 6), eight(c0)) | seven(three(1.5, one(3), 9), four(x, 9)) | last seven(three(11.0, one(22), 66), eight(c0))
% OUTPUT-FLOAT-NEXT:3: returned seven(three(1.5, one(3), 9), four(x, 9)) / seven(three(1.5, one(3), 9), four(x, 9)) | seven(three(2.0, one(4), 12), eight(c0)) | last seven(three(11.5, one(23), 69), two(23, s23))
% OUTPUT-FLOAT-NEXT:4: caught seven(three(2.0, one(4), 12), eight(c0)) / seven(three(2.0, one(4), 12), eight(c0)) | seven(three(2.5, one(5), 15), two(5, s5)) | last seven(three(12.0, one(24), 72), eight(c0))
% OUTPUT-FLOAT-NOT:{{.}}
% RUN: %instantiate %s float32 %t.float32.d
% RUN: %instantiate --keep-name %S/Inputs/nested_try_do_loop_layouts_lib.mt float32 %t.float32.d
% RUN: merkc -O0 -fdo-loops -I %t.float32.d %t.float32.d/nested_try_do_loop_layouts_float32.m %t.float32.d/nested_try_do_loop_layouts_lib.m -o %t.float32.0
% RUN: %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops -I %t.float32.d %t.float32.d/nested_try_do_loop_layouts_float32.m %t.float32.d/nested_try_do_loop_layouts_lib.m -o %t.float32.2
% RUN: %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% OUTPUT-FLOAT32:1: returned seven(three(0.5, one(1), 3), two(1, s1)) / seven(three(0.5, one(1), 3), two(1, s1)) | seven(three(1.0, one(2), 6), eight(c0)) | last seven(three(10.5, one(21), 63), four(x, 441))
% OUTPUT-FLOAT32-NEXT:2: caught seven(three(1.0, one(2), 6), eight(c0)) / seven(three(1.0, one(2), 6), eight(c0)) | seven(three(1.5, one(3), 9), four(x, 9)) | last seven(three(11.0, one(22), 66), eight(c0))
% OUTPUT-FLOAT32-NEXT:3: returned seven(three(1.5, one(3), 9), four(x, 9)) / seven(three(1.5, one(3), 9), four(x, 9)) | seven(three(2.0, one(4), 12), eight(c0)) | last seven(three(11.5, one(23), 69), two(23, s23))
% OUTPUT-FLOAT32-NEXT:4: caught seven(three(2.0, one(4), 12), eight(c0)) / seven(three(2.0, one(4), 12), eight(c0)) | seven(three(2.5, one(5), 15), two(5, s5)) | last seven(three(12.0, one(24), 72), eight(c0))
% OUTPUT-FLOAT32-NOT:{{.}}
:- module nested_try_do_loop_layouts.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module exception, int, list, nested_try_do_loop_layouts_lib, string, univ.

    % The feature's values, with what the inner context makes of the same
    % number, which runs as the outer context uses them.
:- type nested
    --->    nested(int, fdata).

:- func make_nested(int) = nested.

make_nested(N) = nested(N, make(N)).

:- func use_nested(nested) = string.

use_nested(nested(K, V)) = use(V) ++ " / " ++ inner_run(K).

    % A value thrown as an exception, caught and used.
:- pred thrower(int::in, string::out) is det.

thrower(N, S) :-
    V = make_nested(N),
    ( if N mod 2 = 0 then
        throw(V)
    else
        S = "returned " ++ use_nested(V)
    ).

:- func run(int) = string.

run(N) = S :-
    promise_equivalent_solutions [R] (
        try(thrower(N), R)
    ),
    (
        R = succeeded(S)
    ;
        R = exception(E),
        ( if univ_to_type(E, V) then
            S = "caught " ++ use_nested(V)
        else
            S = "caught something else"
        )
    ).

main(!IO) :-
    list.foldl(show, 1 .. 4, !IO).

:- pred show(int::in, io::di, io::uo) is det.

show(N, !IO) :-
    io.format("%d: %s\n", [i(N), s(run(N))], !IO).
