% array.unsafe_noinit, a Merkurius extension (library/array.m): an array
% whose elements are not set, of ints, floats, narrow ints, chars, an
% enumeration and records of numbers, each written in full before it is
% read, and an empty one. mmc has no unsafe_noinit, so the output is the
% program's own, which array.init in its place gives too.
%
% RUN: merkc -fc1 -emit-merk -triple x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=MERK
% RUN: merkc -fc1 -emit-llvm -triple x86_64-unknown-linux-gnu %s -o - | FileCheck %s --check-prefix=LLVM
% RUN: merkc %s -o %t && %t | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace
% RUN: merkc -O2 %s -o %t.O2 && %t.O2 | FileCheck %s --check-prefix=OUTPUT --match-full-lines --strict-whitespace

% MERK-LABEL: merk.proc @array_noinit.main_2_0()
% MERK: merk.throw_if %{{.*}}, "Software Error: predicate `array.unsafe_noinit'/2: Unexpected: negative size"
% MERK-NEXT: merk.array_new %{{.*}} -> <i64>
% MERK: merk.array_new %{{.*}} -> <f64>
% LLVM: call ptr addrspace(1) @merk_array_alloc(i64 10, i64 0, i64 0)
% LLVM: call ptr addrspace(1) @merk_array_alloc(i64 5, i64 0, i64 0)
% LLVM: call ptr addrspace(1) @merk_array_alloc(i64 7, i64 1, i64 0)
% LLVM: call ptr addrspace(1) @merk_array_alloc(i64 3, i64 3, i64 0)
% LLVM: call ptr addrspace(1) @merk_array_alloc(i64 4, i64 3, i64 0)
% LLVM: call ptr addrspace(1) @merk_array_alloc(i64 4, i64 4, i64 16)
% LLVM: call ptr addrspace(1) @merk_array_alloc(i64 0, i64 0, i64 0)
% LLVM: call ptr addrspace(1) @merk_array_alloc(i64 3, i64 0, i64 0)
% LLVM-NOT: merk_array_new

% OUTPUT:ints: 285
% OUTPUT-NEXT:floats: 7.5
% OUTPUT-NEXT:int8s: -21
% OUTPUT-NEXT:chars: abc
% OUTPUT-NEXT:colours: [red, green, blue, red]
% OUTPUT-NEXT:points: 6.0, 12.0
% OUTPUT-NEXT:empty: 0
% OUTPUT-NEXT:function: 3

:- module array_noinit.
:- interface.
:- import_module io.
:- pred main(io::di, io::uo) is det.

:- implementation.
:- import_module array, char, float, int, int8, list, string.

:- type colour
    --->    red
    ;       green
    ;       blue.

:- type point
    --->    point(float, float).

main(!IO) :-
    array.unsafe_noinit(10, Ints0),
    squares(0, Ints0, Ints),
    io.format("ints: %d\n", [i(array.foldl(add_int, Ints, 0))], !IO),

    array.unsafe_noinit(5, Floats0),
    halves(0, Floats0, Floats),
    io.format("floats: %.1f\n",
        [f(array.foldl(add_float, Floats, 0.0))], !IO),

    array.unsafe_noinit(7, Bytes0),
    negatives(0, Bytes0, Bytes),
    io.format("int8s: %d\n",
        [i(array.foldl(add_int8, Bytes, 0))], !IO),

    array.unsafe_noinit(3, Chars0),
    Chars1 = Chars0 ^ elem(0) := 'a',
    Chars2 = Chars1 ^ elem(1) := 'b',
    Chars = Chars2 ^ elem(2) := 'c',
    io.format("chars: %s\n",
        [s(string.from_char_list(array.to_list(Chars)))], !IO),

    array.unsafe_noinit(4, Colours0),
    Colours1 = Colours0 ^ elem(0) := red,
    Colours2 = Colours1 ^ elem(1) := green,
    Colours3 = Colours2 ^ elem(2) := blue,
    Colours = Colours3 ^ elem(3) := red,
    array.to_list(Colours, ColourList),
    io.write_string("colours: ", !IO),
    io.write(ColourList, !IO),
    io.nl(!IO),

    array.unsafe_noinit(4, Points0),
    points(0, Points0, Points),
    array.foldl(add_point, Points, point(0.0, 0.0)) = point(X, Y),
    io.format("points: %.1f, %.1f\n", [f(X), f(Y)], !IO),

    array.unsafe_noinit(0, Empty : array(int)),
    io.format("empty: %d\n", [i(array.size(Empty))], !IO),

    Made = array.unsafe_noinit(3) : array(int),
    io.format("function: %d\n", [i(array.size(Made))], !IO).

:- pred squares(int::in, array(int)::array_di, array(int)::array_uo) is det.
squares(I, !A) :-
    ( if I < array.size(!.A) then
        !A ^ elem(I) := I * I,
        squares(I + 1, !A)
    else
        true
    ).

:- pred halves(int::in, array(float)::array_di, array(float)::array_uo)
    is det.
halves(I, !A) :-
    ( if I < array.size(!.A) then
        !A ^ elem(I) := float(I) * 0.75,
        halves(I + 1, !A)
    else
        true
    ).

:- pred negatives(int::in, array(int8)::array_di, array(int8)::array_uo)
    is det.
negatives(I, !A) :-
    ( if I < array.size(!.A) then
        !A ^ elem(I) := int8.cast_from_int(-I),
        negatives(I + 1, !A)
    else
        true
    ).

:- pred points(int::in, array(point)::array_di, array(point)::array_uo)
    is det.
points(I, !A) :-
    ( if I < array.size(!.A) then
        !A ^ elem(I) := point(float(I), float(2 * I)),
        points(I + 1, !A)
    else
        true
    ).

:- func add_int(int, int) = int.
add_int(X, S) = S + X.

:- func add_float(float, float) = float.
add_float(X, S) = S + X.

:- func add_int8(int8, int) = int.
add_int8(B, S) = S + int8.to_int(B).

:- func add_point(point, point) = point.
add_point(point(X, Y), point(SX, SY)) = point(SX + X, SY + Y).
