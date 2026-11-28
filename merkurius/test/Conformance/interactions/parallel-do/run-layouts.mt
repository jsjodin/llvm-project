% MANUAL: Discriminated unions
% DESIGN: Loops / Under a parallel directive
% DESIGN: Sized floating point types
% Interaction: a type of more functors than a pointer has tags, in a
% parallel loop's iterations, which make and use the values.
% Written by utils/conformance/generate-interactions.py; change
% the generator rather than this file.
% MERKC-FLAGS: -fdo-loops -fopenmp
% TRANSLATION: Inputs/parallel_do_layouts_translation.mt
%
% REQUIRES: libomp
% RUN: %instantiate %s float %t.float.d
% RUN: merkc -O0 -fdo-loops -fopenmp %t.float.d/parallel_do_layouts_float.m %libomp-flags -o %t.float.0
% RUN: %libomp-env OMP_NUM_THREADS=4 %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: %libomp-env OMP_NUM_THREADS=4 MERK_GC_EVERY_ALLOC=1 %t.float.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops -fopenmp %t.float.d/parallel_do_layouts_float.m %libomp-flags -o %t.float.2
% RUN: %libomp-env OMP_NUM_THREADS=4 %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: %libomp-env OMP_NUM_THREADS=4 MERK_GC_EVERY_ALLOC=1 %t.float.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: merkc -O0 -fdo-loops %t.float.d/parallel_do_layouts_float.m -o %t.float.0s
% RUN: %t.float.0s 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops %t.float.d/parallel_do_layouts_float.m -o %t.float.2s
% RUN: %t.float.2s 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT --match-full-lines --strict-whitespace
% OUTPUT-FLOAT:1: seven(three(0.5, one(1), 3), two(1, s1)) | seven(three(1.0, one(2), 6), eight(c0)) | seven(three(1.5, one(3), 9), four(x, 9)) | seven(three(2.0, one(4), 12), eight(c0))
% OUTPUT-FLOAT-NEXT:2: seven(three(1.0, one(2), 6), eight(c0)) | seven(three(1.5, one(3), 9), four(x, 9)) | seven(three(2.0, one(4), 12), eight(c0)) | seven(three(2.5, one(5), 15), two(5, s5))
% OUTPUT-FLOAT-NEXT:3: seven(three(1.5, one(3), 9), four(x, 9)) | seven(three(2.0, one(4), 12), eight(c0)) | seven(three(2.5, one(5), 15), two(5, s5)) | seven(three(3.0, one(6), 18), eight(c0))
% OUTPUT-FLOAT-NEXT:4: seven(three(2.0, one(4), 12), eight(c0)) | seven(three(2.5, one(5), 15), two(5, s5)) | seven(three(3.0, one(6), 18), eight(c0)) | seven(three(3.5, one(7), 21), two(7, s7))
% OUTPUT-FLOAT-NOT:{{.}}
% RUN: %instantiate %s float32 %t.float32.d
% RUN: merkc -O0 -fdo-loops -fopenmp %t.float32.d/parallel_do_layouts_float32.m %libomp-flags -o %t.float32.0
% RUN: %libomp-env OMP_NUM_THREADS=4 %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: %libomp-env OMP_NUM_THREADS=4 MERK_GC_EVERY_ALLOC=1 %t.float32.0 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops -fopenmp %t.float32.d/parallel_do_layouts_float32.m %libomp-flags -o %t.float32.2
% RUN: %libomp-env OMP_NUM_THREADS=4 %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: %libomp-env OMP_NUM_THREADS=4 MERK_GC_EVERY_ALLOC=1 %t.float32.2 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: merkc -O0 -fdo-loops %t.float32.d/parallel_do_layouts_float32.m -o %t.float32.0s
% RUN: %t.float32.0s 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% RUN: merkc -O2 -fdo-loops %t.float32.d/parallel_do_layouts_float32.m -o %t.float32.2s
% RUN: %t.float32.2s 2>&1 | FileCheck %s --check-prefix=OUTPUT-FLOAT32 --match-full-lines --strict-whitespace
% OUTPUT-FLOAT32:1: seven(three(0.5, one(1), 3), two(1, s1)) | seven(three(1.0, one(2), 6), eight(c0)) | seven(three(1.5, one(3), 9), four(x, 9)) | seven(three(2.0, one(4), 12), eight(c0))
% OUTPUT-FLOAT32-NEXT:2: seven(three(1.0, one(2), 6), eight(c0)) | seven(three(1.5, one(3), 9), four(x, 9)) | seven(three(2.0, one(4), 12), eight(c0)) | seven(three(2.5, one(5), 15), two(5, s5))
% OUTPUT-FLOAT32-NEXT:3: seven(three(1.5, one(3), 9), four(x, 9)) | seven(three(2.0, one(4), 12), eight(c0)) | seven(three(2.5, one(5), 15), two(5, s5)) | seven(three(3.0, one(6), 18), eight(c0))
% OUTPUT-FLOAT32-NEXT:4: seven(three(2.0, one(4), 12), eight(c0)) | seven(three(2.5, one(5), 15), two(5, s5)) | seven(three(3.0, one(6), 18), eight(c0)) | seven(three(3.5, one(7), 21), two(7, s7))
% OUTPUT-FLOAT32-NOT:{{.}}
:- module parallel_do_layouts.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.
:- implementation.
:- import_module @TM@, array, char, int, list, string.

    % More functors with arguments than a pointer has tags, constants, and
    % arguments of every kind of word.
:- type fdata
    --->    c0
    ;       c1
    ;       one(int)
    ;       two(int, string)
    ;       three(@T@, fdata, int)
    ;       four(char, int, int, int)
    ;       five(string)
    ;       six(int)
    ;       seven(fdata, fdata)
    ;       eight(fdata).

:- func make(int) = fdata.
:- func use(fdata) = string.

make(N) = seven(three(@FROM_INT@(N) / 2.0@S@, one(N), N * 3),
    ( if N mod 2 = 0 then
        eight(c0)
    else if N mod 3 = 0 then
        four('x', N, -N, N * N)
    else
        two(N, "s" ++ string.int_to_string(N))
    )).

use(c0) = "c0".
use(c1) = "c1".
use(one(I)) = "one(" ++ string.int_to_string(I) ++ ")".
use(two(I, S)) = "two(" ++ string.int_to_string(I) ++ ", " ++ S ++ ")".
use(three(F, D, I)) = "three(" ++ string.@T@_to_string(F) ++ ", " ++
    use(D) ++ ", " ++ string.int_to_string(I) ++ ")".
use(four(C, A, B, D)) = "four(" ++ string.char_to_string(C) ++ ", " ++
    string.int_to_string(A + B + D) ++ ")".
use(five(S)) = "five(" ++ S ++ ")".
use(six(I)) = "six(" ++ string.int_to_string(I) ++ ")".
use(seven(A, B)) = "seven(" ++ use(A) ++ ", " ++ use(B) ++ ")".
use(eight(D)) = "eight(" ++ use(D) ++ ")".

    % Each iteration of a parallel loop, on one of four threads, makes a
    % value and stores what use makes of it.
:- pred fill(int::in, array(string)::array_di, array(string)::array_uo)
    is det.

fill(N, !A) :-
    %$omp parallel do
    ( for(I, 0, 3) do
        V = make(N + I),
        !A ^ unsafe_elem(I) := use(V)
    ).

:- func run(int) = string.

run(N) = S :-
    array.init(4, "", A0),
    fill(N, A0, A),
    S = string.join_list(" | ", array.to_list(A)).

main(!IO) :-
    list.foldl(show, 1 .. 4, !IO).

:- pred show(int::in, io::di, io::uo) is det.

show(N, !IO) :-
    io.format("%d: %s\n", [i(N), s(run(N))], !IO).
