% MANUAL: Discriminated unions
% MANUAL: Type classes
% MANUAL: Creating higher-order terms
% DESIGN: Sized floating point types
% Interaction: a type of more functors than a pointer has tags, in closures
% that make, capture and use the values, in instance methods that make and
% use the values.
% Written by utils/conformance/generate-interactions.py; change
% the generator rather than this file.
% MODULES: Inputs/nested_typeclass_method_closure_layouts_lib.mt
%
% RUN: %instantiate %s float %t.float.d
% RUN: %instantiate --keep-name %S/Inputs/nested_typeclass_method_closure_layouts_lib.mt float %t.float.d
% RUN: merkc -O0 -I %t.float.d %t.float.d/nested_typeclass_method_closure_layouts_float.m %t.float.d/nested_typeclass_method_closure_layouts_lib.m -o %t.float.0
% RUN: %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -I %t.float.d %t.float.d/nested_typeclass_method_closure_layouts_float.m %t.float.d/nested_typeclass_method_closure_layouts_lib.m -o %t.float.2
% RUN: %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% OUTPUT-FLOAT:1: seven(three(0.5, one(1), 3), two(1, s1)) / seven(three(0.5, one(1), 3), two(1, s1)) | seven(three(0.5, one(1), 3), two(1, s1)) | seven(three(1.0, one(2), 6), eight(c0)) | offset seven(three(1.0, one(2), 6), eight(c0)) / seven(three(1.0, one(2), 6), eight(c0)) | seven(three(1.0, one(2), 6), eight(c0)) | seven(three(1.5, one(3), 9), four(x, 9))
% OUTPUT-FLOAT-NEXT:2: seven(three(1.0, one(2), 6), eight(c0)) / seven(three(1.0, one(2), 6), eight(c0)) | seven(three(1.0, one(2), 6), eight(c0)) | seven(three(1.5, one(3), 9), four(x, 9)) | offset seven(three(1.5, one(3), 9), four(x, 9)) / seven(three(1.5, one(3), 9), four(x, 9)) | seven(three(1.5, one(3), 9), four(x, 9)) | seven(three(2.0, one(4), 12), eight(c0))
% OUTPUT-FLOAT-NEXT:3: seven(three(1.5, one(3), 9), four(x, 9)) / seven(three(1.5, one(3), 9), four(x, 9)) | seven(three(1.5, one(3), 9), four(x, 9)) | seven(three(2.0, one(4), 12), eight(c0)) | offset seven(three(2.0, one(4), 12), eight(c0)) / seven(three(2.0, one(4), 12), eight(c0)) | seven(three(2.0, one(4), 12), eight(c0)) | seven(three(2.5, one(5), 15), two(5, s5))
% OUTPUT-FLOAT-NEXT:4: seven(three(2.0, one(4), 12), eight(c0)) / seven(three(2.0, one(4), 12), eight(c0)) | seven(three(2.0, one(4), 12), eight(c0)) | seven(three(2.5, one(5), 15), two(5, s5)) | offset seven(three(2.5, one(5), 15), two(5, s5)) / seven(three(2.5, one(5), 15), two(5, s5)) | seven(three(2.5, one(5), 15), two(5, s5)) | seven(three(3.0, one(6), 18), eight(c0))
% OUTPUT-FLOAT-NOT:{{.}}
% RUN: %instantiate %s float32 %t.float32.d
% RUN: %instantiate --keep-name %S/Inputs/nested_typeclass_method_closure_layouts_lib.mt float32 %t.float32.d
% RUN: merkc -O0 -I %t.float32.d %t.float32.d/nested_typeclass_method_closure_layouts_float32.m %t.float32.d/nested_typeclass_method_closure_layouts_lib.m -o %t.float32.0
% RUN: %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: merkc -O2 -I %t.float32.d %t.float32.d/nested_typeclass_method_closure_layouts_float32.m %t.float32.d/nested_typeclass_method_closure_layouts_lib.m -o %t.float32.2
% RUN: %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% OUTPUT-FLOAT32:1: seven(three(0.5, one(1), 3), two(1, s1)) / seven(three(0.5, one(1), 3), two(1, s1)) | seven(three(0.5, one(1), 3), two(1, s1)) | seven(three(1.0, one(2), 6), eight(c0)) | offset seven(three(1.0, one(2), 6), eight(c0)) / seven(three(1.0, one(2), 6), eight(c0)) | seven(three(1.0, one(2), 6), eight(c0)) | seven(three(1.5, one(3), 9), four(x, 9))
% OUTPUT-FLOAT32-NEXT:2: seven(three(1.0, one(2), 6), eight(c0)) / seven(three(1.0, one(2), 6), eight(c0)) | seven(three(1.0, one(2), 6), eight(c0)) | seven(three(1.5, one(3), 9), four(x, 9)) | offset seven(three(1.5, one(3), 9), four(x, 9)) / seven(three(1.5, one(3), 9), four(x, 9)) | seven(three(1.5, one(3), 9), four(x, 9)) | seven(three(2.0, one(4), 12), eight(c0))
% OUTPUT-FLOAT32-NEXT:3: seven(three(1.5, one(3), 9), four(x, 9)) / seven(three(1.5, one(3), 9), four(x, 9)) | seven(three(1.5, one(3), 9), four(x, 9)) | seven(three(2.0, one(4), 12), eight(c0)) | offset seven(three(2.0, one(4), 12), eight(c0)) / seven(three(2.0, one(4), 12), eight(c0)) | seven(three(2.0, one(4), 12), eight(c0)) | seven(three(2.5, one(5), 15), two(5, s5))
% OUTPUT-FLOAT32-NEXT:4: seven(three(2.0, one(4), 12), eight(c0)) / seven(three(2.0, one(4), 12), eight(c0)) | seven(three(2.0, one(4), 12), eight(c0)) | seven(three(2.5, one(5), 15), two(5, s5)) | offset seven(three(2.5, one(5), 15), two(5, s5)) / seven(three(2.5, one(5), 15), two(5, s5)) | seven(three(2.5, one(5), 15), two(5, s5)) | seven(three(3.0, one(6), 18), eight(c0))
% OUTPUT-FLOAT32-NOT:{{.}}
:- module nested_typeclass_method_closure_layouts.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, nested_typeclass_method_closure_layouts_lib, string.

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
