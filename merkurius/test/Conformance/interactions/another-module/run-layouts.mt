% MANUAL: Discriminated unions
% MANUAL: Modules
% DESIGN: Sized floating point types
% Interaction: a type of more functors than a pointer has tags, in the
% feature in another module.
% Written by utils/conformance/generate-interactions.py; change
% the generator rather than this file.
% MODULES: Inputs/another_module_layouts_lib.mt
%
% RUN: %instantiate %s float %t.float.d
% RUN: %instantiate --keep-name %S/Inputs/another_module_layouts_lib.mt float %t.float.d
% RUN: merkc -O0 -I %t.float.d %t.float.d/another_module_layouts_float.m %t.float.d/another_module_layouts_lib.m -o %t.float.0
% RUN: %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -I %t.float.d %t.float.d/another_module_layouts_float.m %t.float.d/another_module_layouts_lib.m -o %t.float.2
% RUN: %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% OUTPUT-FLOAT:1: seven(three(0.5, one(1), 3), two(1, s1)) | seven(three(1.0, one(2), 6), eight(c0))
% OUTPUT-FLOAT-NEXT:2: seven(three(1.0, one(2), 6), eight(c0)) | seven(three(1.5, one(3), 9), four(x, 9))
% OUTPUT-FLOAT-NEXT:3: seven(three(1.5, one(3), 9), four(x, 9)) | seven(three(2.0, one(4), 12), eight(c0))
% OUTPUT-FLOAT-NEXT:4: seven(three(2.0, one(4), 12), eight(c0)) | seven(three(2.5, one(5), 15), two(5, s5))
% OUTPUT-FLOAT-NOT:{{.}}
% RUN: %instantiate %s float32 %t.float32.d
% RUN: %instantiate --keep-name %S/Inputs/another_module_layouts_lib.mt float32 %t.float32.d
% RUN: merkc -O0 -I %t.float32.d %t.float32.d/another_module_layouts_float32.m %t.float32.d/another_module_layouts_lib.m -o %t.float32.0
% RUN: %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: merkc -O2 -I %t.float32.d %t.float32.d/another_module_layouts_float32.m %t.float32.d/another_module_layouts_lib.m -o %t.float32.2
% RUN: %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: env MERK_GC_EVERY_ALLOC=1 %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% OUTPUT-FLOAT32:1: seven(three(0.5, one(1), 3), two(1, s1)) | seven(three(1.0, one(2), 6), eight(c0))
% OUTPUT-FLOAT32-NEXT:2: seven(three(1.0, one(2), 6), eight(c0)) | seven(three(1.5, one(3), 9), four(x, 9))
% OUTPUT-FLOAT32-NEXT:3: seven(three(1.5, one(3), 9), four(x, 9)) | seven(three(2.0, one(4), 12), eight(c0))
% OUTPUT-FLOAT32-NEXT:4: seven(three(2.0, one(4), 12), eight(c0)) | seven(three(2.5, one(5), 15), two(5, s5))
% OUTPUT-FLOAT32-NOT:{{.}}
:- module another_module_layouts.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module another_module_layouts_lib, int, list, string.

    % The type, make and use are another module's.
:- func run(int) = string.

run(N) = S :-
    Vs = list.map(make, [N, N + 1]),
    S = string.join_list(" | ", list.map(use, Vs)).

main(!IO) :-
    list.foldl(show, 1 .. 4, !IO).

:- pred show(int::in, io::di, io::uo) is det.

show(N, !IO) :-
    io.format("%d: %s\n", [i(N), s(run(N))], !IO).
