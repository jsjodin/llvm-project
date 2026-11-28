% MANUAL: Discriminated unions
% MANUAL: Creating higher-order terms
% MANUAL: Type classes
% DESIGN: Sized floating point types
% Interaction: a type of more functors than a pointer has tags, in instance
% methods that make and use the values, in closures that make, capture and
% use the values.
% Written by utils/conformance/generate-interactions.py; change
% the generator rather than this file.
% MODULES: Inputs/nested_closure_typeclass_method_layouts_lib.mt
%
% RUN: %instantiate %s float %t.float.d
% RUN: %instantiate --keep-name %S/Inputs/nested_closure_typeclass_method_layouts_lib.mt float %t.float.d
% RUN: merkc -O0 -I %t.float.d %t.float.d/nested_closure_typeclass_method_layouts_float.m %t.float.d/nested_closure_typeclass_method_layouts_lib.m -o %t.float.0
% RUN: %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -I %t.float.d %t.float.d/nested_closure_typeclass_method_layouts_float.m %t.float.d/nested_closure_typeclass_method_layouts_lib.m -o %t.float.2
% RUN: %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% OUTPUT-FLOAT:1: seven(three(0.5, one(1), 3), two(1, s1)) / seven(three(0.5, one(1), 3), two(1, s1)) | offset seven(three(1.0, one(2), 6), eight(c0)) | seven(three(0.5, one(1), 3), two(1, s1)) / seven(three(0.5, one(1), 3), two(1, s1)) | offset seven(three(1.0, one(2), 6), eight(c0)) | seven(three(1.0, one(2), 6), eight(c0)) / seven(three(1.0, one(2), 6), eight(c0)) | offset seven(three(1.5, one(3), 9), four(x, 9))
% OUTPUT-FLOAT-NEXT:2: seven(three(1.0, one(2), 6), eight(c0)) / seven(three(1.0, one(2), 6), eight(c0)) | offset seven(three(1.5, one(3), 9), four(x, 9)) | seven(three(1.0, one(2), 6), eight(c0)) / seven(three(1.0, one(2), 6), eight(c0)) | offset seven(three(1.5, one(3), 9), four(x, 9)) | seven(three(1.5, one(3), 9), four(x, 9)) / seven(three(1.5, one(3), 9), four(x, 9)) | offset seven(three(2.0, one(4), 12), eight(c0))
% OUTPUT-FLOAT-NEXT:3: seven(three(1.5, one(3), 9), four(x, 9)) / seven(three(1.5, one(3), 9), four(x, 9)) | offset seven(three(2.0, one(4), 12), eight(c0)) | seven(three(1.5, one(3), 9), four(x, 9)) / seven(three(1.5, one(3), 9), four(x, 9)) | offset seven(three(2.0, one(4), 12), eight(c0)) | seven(three(2.0, one(4), 12), eight(c0)) / seven(three(2.0, one(4), 12), eight(c0)) | offset seven(three(2.5, one(5), 15), two(5, s5))
% OUTPUT-FLOAT-NEXT:4: seven(three(2.0, one(4), 12), eight(c0)) / seven(three(2.0, one(4), 12), eight(c0)) | offset seven(three(2.5, one(5), 15), two(5, s5)) | seven(three(2.0, one(4), 12), eight(c0)) / seven(three(2.0, one(4), 12), eight(c0)) | offset seven(three(2.5, one(5), 15), two(5, s5)) | seven(three(2.5, one(5), 15), two(5, s5)) / seven(three(2.5, one(5), 15), two(5, s5)) | offset seven(three(3.0, one(6), 18), eight(c0))
% OUTPUT-FLOAT-NOT:{{.}}
% RUN: %instantiate %s float32 %t.float32.d
% RUN: %instantiate --keep-name %S/Inputs/nested_closure_typeclass_method_layouts_lib.mt float32 %t.float32.d
% RUN: merkc -O0 -I %t.float32.d %t.float32.d/nested_closure_typeclass_method_layouts_float32.m %t.float32.d/nested_closure_typeclass_method_layouts_lib.m -o %t.float32.0
% RUN: %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: merkc -O2 -I %t.float32.d %t.float32.d/nested_closure_typeclass_method_layouts_float32.m %t.float32.d/nested_closure_typeclass_method_layouts_lib.m -o %t.float32.2
% RUN: %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% OUTPUT-FLOAT32:1: seven(three(0.5, one(1), 3), two(1, s1)) / seven(three(0.5, one(1), 3), two(1, s1)) | offset seven(three(1.0, one(2), 6), eight(c0)) | seven(three(0.5, one(1), 3), two(1, s1)) / seven(three(0.5, one(1), 3), two(1, s1)) | offset seven(three(1.0, one(2), 6), eight(c0)) | seven(three(1.0, one(2), 6), eight(c0)) / seven(three(1.0, one(2), 6), eight(c0)) | offset seven(three(1.5, one(3), 9), four(x, 9))
% OUTPUT-FLOAT32-NEXT:2: seven(three(1.0, one(2), 6), eight(c0)) / seven(three(1.0, one(2), 6), eight(c0)) | offset seven(three(1.5, one(3), 9), four(x, 9)) | seven(three(1.0, one(2), 6), eight(c0)) / seven(three(1.0, one(2), 6), eight(c0)) | offset seven(three(1.5, one(3), 9), four(x, 9)) | seven(three(1.5, one(3), 9), four(x, 9)) / seven(three(1.5, one(3), 9), four(x, 9)) | offset seven(three(2.0, one(4), 12), eight(c0))
% OUTPUT-FLOAT32-NEXT:3: seven(three(1.5, one(3), 9), four(x, 9)) / seven(three(1.5, one(3), 9), four(x, 9)) | offset seven(three(2.0, one(4), 12), eight(c0)) | seven(three(1.5, one(3), 9), four(x, 9)) / seven(three(1.5, one(3), 9), four(x, 9)) | offset seven(three(2.0, one(4), 12), eight(c0)) | seven(three(2.0, one(4), 12), eight(c0)) / seven(three(2.0, one(4), 12), eight(c0)) | offset seven(three(2.5, one(5), 15), two(5, s5))
% OUTPUT-FLOAT32-NEXT:4: seven(three(2.0, one(4), 12), eight(c0)) / seven(three(2.0, one(4), 12), eight(c0)) | offset seven(three(2.5, one(5), 15), two(5, s5)) | seven(three(2.0, one(4), 12), eight(c0)) / seven(three(2.0, one(4), 12), eight(c0)) | offset seven(three(2.5, one(5), 15), two(5, s5)) | seven(three(2.5, one(5), 15), two(5, s5)) / seven(three(2.5, one(5), 15), two(5, s5)) | offset seven(three(3.0, one(6), 18), eight(c0))
% OUTPUT-FLOAT32-NOT:{{.}}
:- module nested_closure_typeclass_method_layouts.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module int, list, nested_closure_typeclass_method_layouts_lib, string.

    % The feature's values, with what the inner context makes of the same
    % number, which runs as the outer context uses them.
:- type nested
    --->    nested(int, fdata).

:- func make_nested(int) = nested.

make_nested(N) = nested(N, make(N)).

:- func use_nested(nested) = string.

use_nested(nested(K, V)) = use(V) ++ " / " ++ inner_run(K).

    % A closure that makes and uses a value, and one that captures a value
    % made outside it.
:- func run(int) = string.

run(N) = S :-
    Make = (func(K) = make_nested(K)),
    V = apply(Make, N),
    Use = ((func) = use_nested(V)),
    Both = list.map((func(K) = use_nested(make_nested(K))), [N, N + 1]),
    S = apply(Use) ++ " | " ++ string.join_list(" | ", Both).

main(!IO) :-
    list.foldl(show, 1 .. 4, !IO).

:- pred show(int::in, io::di, io::uo) is det.

show(N, !IO) :-
    io.format("%d: %s\n", [i(N), s(run(N))], !IO).
